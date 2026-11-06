"""riffchart-ui: local web app (FastAPI). Serves the browser UI from ui/static and a JSON API under /api.

    riffchart-ui [--port 8765] [--no-browser] [--config configs/ui.yaml]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import time
import subprocess
import threading
import uuid
import webbrowser
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..paths import PROJECT, project_path
from ..product import clonehero
from ..product.cli import AUDIO_EXT, display_name, find_audio
from ..formats.loader import load_song
from ..formats.textio import parse_ini
from ..formats.chart_writer import write_chart_text
from .chartjson import json_to_song, peaks, song_to_json
from .checker import load_rules, song_issues
from .jobs import JobQueue, Options
from .library import (ART_FILES, MAX_ART_BYTES, Library, ProtectedPathError, check_writable, clean_meta,
                      write_art, write_ini_fields)

STATIC = Path(__file__).parent / "static"


def load_config(path: str | Path = "configs/ui.yaml") -> dict:
    from ..paths import load_config as _load
    cfg = _load(path)
    for k in ("workspace", "uploads"):
        cfg[k] = project_path(cfg[k])
    cfg["export_dir"] = Path(cfg["export_dir"]).expanduser()
    cfg["protected"] = [Path(p).expanduser() for p in cfg.get("protected") or []]
    cfg["clone_hero_settings"] = Path(cfg["clone_hero_settings"]).expanduser() if cfg.get("clone_hero_settings") else None
    return cfg


def default_charter_factory(cfg: dict):
    def make():
        from ..product.core import RiffCharter
        return RiffCharter(cfg.get("product_config", "configs/product.yaml"))
    return make


def _safe_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "audio"


def _ini(folder: Path) -> dict:
    p = next((x for x in folder.iterdir() if x.name.lower() == "song.ini"), None)
    return parse_ini(p) if p else {}


def create_app(cfg: dict, make_charter=None, start_worker: bool = True) -> FastAPI:
    lib = Library(cfg["workspace"], cfg["export_dir"], cfg["protected"])
    check_writable(cfg["uploads"], cfg["protected"])
    queue = JobQueue(cfg["workspace"], cfg["stage_weights"], make_charter or default_charter_factory(cfg),
                     log_file=Path(cfg["log_file"]) if cfg.get("log_file") else None)
    rules = load_rules(cfg.get("eval_config", "configs/eval.yaml"))
    app = FastAPI(title="RiffChart", docs_url=None, redoc_url=None)
    app.state.queue, app.state.lib, app.state.cfg = queue, lib, cfg
    if start_worker:
        queue.start()

    def song_folder(song_id: str) -> Path:
        try:
            return lib.folder(song_id)
        except KeyError:
            raise HTTPException(404, f"no song {song_id!r}")

    # ---- status
    @app.get("/api/status")
    def status():
        settings = cfg["clone_hero_settings"]
        folders = clonehero.song_folders(settings) if settings else []
        exe = clonehero.find_exe()
        return {"model": queue.model_state, "workspace": str(lib.workspace), "export_dir": str(lib.export_dir),
                "clone_hero": {"exe": str(exe) if exe else None,
                               "export_registered": any(Path(p).resolve() == lib.export_dir.resolve()
                                                        for p in folders if p),
                               "running": clonehero.is_running()},
                "audio_ext": sorted(AUDIO_EXT)}

    # ---- jobs
    @app.get("/api/jobs")
    def jobs():
        return queue.list()

    @app.post("/api/jobs/upload")
    async def upload(files: list[UploadFile] = File(...), options: str = Form("{}")):
        import json
        try:
            opts = Options.parse(json.loads(options))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        added, skipped = [], []
        for f in files:
            fname = _safe_filename(f.filename or "audio")
            if Path(fname).suffix.lower() not in AUDIO_EXT:
                skipped.append(f"{fname}: unsupported format (use mp3, m4a, ogg, opus, webm, flac or wav)")
                continue
            # folder drops send "Album/Artist - Song/song.ogg": keep the parent folder name for naming
            parent = _safe_filename(Path((f.filename or "").replace("\\", "/")).parent.name) if "/" in (
                f.filename or "").replace("\\", "/") else ""
            d = cfg["uploads"] / uuid.uuid4().hex[:12] / (parent or Path(fname).stem)
            d.mkdir(parents=True)
            dest = d / fname
            with open(dest, "wb") as out:
                while chunk := await f.read(1 << 20):
                    out.write(chunk)
            j = queue.add(dest, display_name(dest), opts, upload=True)
            added.append(j.to_dict())
        return {"added": added, "skipped": skipped}

    @app.post("/api/jobs/paths")
    def add_paths(body: dict = Body(...)):
        try:
            opts = Options.parse(body.get("options"))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        paths = [p.strip().strip('"') for p in body.get("paths", []) if p and p.strip()]
        missing = [p for p in paths if not Path(p).exists()]
        files, unsupported = find_audio([p for p in paths if p not in missing])
        added = [queue.add(f, display_name(f), opts).to_dict() for f in files]
        return {"added": added, "skipped": [f"{p}: not found" for p in missing] +
                [f"{f.name}: unsupported format" for f in unsupported]}

    @app.delete("/api/jobs/{job_id}")
    def cancel(job_id: str):
        if not queue.cancel(job_id):
            raise HTTPException(409, "can't remove a song while it is being charted")
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: str):
        if not queue.retry(job_id):
            raise HTTPException(409, "can't retry this job")
        return {"ok": True}

    @app.post("/api/jobs/clear")
    def clear():
        return {"removed": queue.clear_finished()}

    # ---- songs
    @app.get("/api/songs")
    def songs():
        return lib.list()

    @app.get("/api/songs/{song_id}")
    def song(song_id: str):
        song_folder(song_id)
        return lib.info(song_id)

    def audio_files(folder: Path) -> list[Path]:
        # song first (the playback clock), then guitar, then any other stems
        fs = [f for f in folder.iterdir() if f.suffix.lower() in {".ogg", ".opus", ".mp3", ".wav"}
              and not f.name.lower().startswith("preview")]
        order = {"song": 0, "guitar": 1}
        return sorted(fs, key=lambda f: (order.get(f.stem.lower(), 2), f.name.lower()))

    @app.get("/api/songs/{song_id}/chart")
    def chart(song_id: str):
        folder = song_folder(song_id)
        song, ini, chart_path = load_song(folder)
        stems = audio_files(folder)
        audio_s = None
        for f in stems:
            try:
                audio_s = max(audio_s or 0.0, sf.info(str(f)).duration)
            except Exception:
                pass
        d = song_to_json(song, ini, audio_s)
        d.update(id=song_id, file=chart_path.name, stems=[f.name for f in stems],
                 title=ini.get("name", song_id), artist=ini.get("artist", ""), warnings=song.warnings[:20])
        return d

    @app.post("/api/songs/{song_id}/chart")
    def save_chart(song_id: str, body: dict = Body(...)):
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        try:
            song = json_to_song(body)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, f"bad chart data: {e}")
        if not any(len(t) for t in song.tracks.values()):
            raise HTTPException(400, "refusing to save a chart with no notes in any difficulty")
        ini = _ini(folder)
        text = write_chart_text(song, name=ini.get("name", ""), artist=ini.get("artist", ""),
                                charter=ini.get("charter", "RiffChart"), ini=ini)
        hist = folder / ".riffchart" / "history"
        hist.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("notes.chart", "notes.mid"):
            old = folder / name
            if old.exists():
                shutil.copy2(old, hist / f"{stamp}-{name}")
        for old in sorted(hist.glob("*-notes.*"))[:-int(cfg.get("history_keep", 30))]:
            old.unlink(missing_ok=True)
        tmp = folder / "notes.chart.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(folder / "notes.chart")
        if (folder / "notes.mid").exists():         # CH prefers notes.mid: move it aside (copy is in history)
            (folder / "notes.mid").replace(folder / "notes.mid.bak")
        return {"saved": stamp, "notes": {d: len(t) for d, t in song.tracks.items()}, "history": str(hist)}

    @app.post("/api/check")
    def check_chart(body: dict = Body(...)):
        """Playability issues for an unsaved chart from the editor (same rules as the evaluation)."""
        try:
            song = json_to_song(body)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, f"bad chart data: {e}")
        audio_s = body.get("audio_s")
        return song_issues(song, float(body.get("offset_s", 0) or 0), float(audio_s) if audio_s else None, rules)

    @app.get("/api/songs/{song_id}/peaks")
    def song_peaks(song_id: str):
        folder = song_folder(song_id)
        cache = cfg.get("cache", cfg["workspace"].parent / "cache") / song_id
        out = []
        for f in audio_files(folder):
            try:
                out.append(peaks(f, cache))
            except Exception as e:                       # undecodable stem: skip its waveform
                out.append({"name": f.stem, "error": str(e)})
        return out

    @app.get("/api/songs/{song_id}/file/{name}")
    def song_file(song_id: str, name: str):
        p = song_folder(song_id) / _safe_filename(name)
        if not p.is_file() or p.suffix.lower() not in {".ogg", ".opus", ".mp3", ".wav", ".png", ".jpg"}:
            raise HTTPException(404)
        return FileResponse(p)

    def folder_mix(folder: Path) -> tuple[np.ndarray, int]:
        """The song's stems summed back into one stereo mix (what the song was charted from)."""
        stems = audio_files(folder)
        if not stems:
            raise HTTPException(409, "this song has no audio to chart from")
        mix, sr = None, None
        for f in stems:
            x, r = sf.read(str(f), dtype="float32", always_2d=True)
            x = np.repeat(x[:, :1], 2, 1) if x.shape[1] == 1 else x[:, :2]
            if mix is None:
                mix, sr = x, r
            elif r == sr:
                n = max(len(mix), len(x))
                mix = np.pad(mix, ((0, n - len(mix)), (0, 0))) + np.pad(x, ((0, n - len(x)), (0, 0)))
        mix *= min(1.0, 0.99 / max(float(np.abs(mix).max()), 1e-6))
        return mix, sr

    # ---- re-chart a section (M5): analysis cached per song, then a take is a few seconds
    feats_mem: dict = {}

    def section_cache(song_id: str) -> Path:
        return Path(cfg.get("cache", cfg["workspace"].parent / "cache")) / song_id / "section_feats.npz"

    def cached_features(song_id: str, folder: Path) -> dict | None:
        from ..product.section import audio_key, load_features
        key = audio_key(audio_files(folder))
        hit = feats_mem.get(song_id)
        if hit and hit[0] == key:
            return hit[1]
        f = load_features(section_cache(song_id), key)
        if f is not None:
            feats_mem.clear()
            feats_mem[song_id] = (key, f)
        return f

    @app.get("/api/songs/{song_id}/section")
    def section_status(song_id: str):
        folder = song_folder(song_id)
        busy = any(j["status"] == "running" for j in queue.list())
        return {"analysed": cached_features(song_id, folder) is not None, "model": queue.model_state,
                "busy": busy}

    @app.post("/api/songs/{song_id}/section")
    def rechart_section(song_id: str, body: dict = Body(...)):
        """One new take of the chart between body.start and body.end (ticks) of body.chart (the editor's
        current, possibly unsaved chart). Returns the new notes for that range; nothing is written."""
        from ..product.core import Style
        from ..product.section import audio_key, chart_section, save_features, section_features
        folder = song_folder(song_id)
        try:
            opts = Options.parse(body.get("options"))
            song = json_to_song(body["chart"])
            start, end = int(body["start"]), int(body["end"])
            offset_s = float(body["chart"].get("offset_s", 0) or 0)
            seed = int(body.get("seed", 0))
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, f"bad request: {e}")
        if end <= start:
            raise HTTPException(400, "the section is empty")
        t0 = time.perf_counter()
        feats = cached_features(song_id, folder)
        analysed = False
        if feats is None:
            mix, sr = folder_mix(folder)
            key = audio_key(audio_files(folder))
            queue.log(f"section re-chart: analysing {song_id[:60]}")
            feats = queue.run_on_model(lambda ac: section_features(ac, mix, sr))
            save_features(section_cache(song_id), key, feats)
            feats_mem.clear()
            feats_mem[song_id] = (key, feats)
            analysed = True
        style = Style(opts.density, opts.chords, opts.follow)
        try:
            out = queue.run_on_model(lambda ac: chart_section(ac, feats, song, offset_s, start, end,
                                                              opts.difficulties, style, seed))
        except ValueError as e:
            raise HTTPException(400, str(e))
        out["seconds"] = round(time.perf_counter() - t0, 2)
        out["analysed"] = analysed
        queue.log(f"section re-chart {song_id[:50]}: ticks {out['start']}-{out['end']}, "
                  + ", ".join(f"{d} {len(v)}" for d, v in out["notes"].items()) + f" ({out['seconds']} s)")
        return out

    @app.post("/api/songs/{song_id}/rechart")
    def rechart(song_id: str, body: dict = Body(default={})):
        """Chart a song again from its own audio (stems summed back into the mix), keeping its names.
        The chart being replaced goes to the song's history first."""
        folder = song_folder(song_id)
        try:
            opts = Options.parse(body.get("options"))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        mix, sr = folder_mix(folder)
        d = cfg["uploads"] / uuid.uuid4().hex[:12]
        d.mkdir(parents=True)
        src = d / "mix.wav"
        sf.write(str(src), mix, sr, subtype="PCM_16")
        ini = _ini(folder)
        meta = {"title": ini.get("name", song_id), "artist": ini.get("artist", ""), "album": ini.get("album", ""),
                "genre": ini.get("genre", ""), "year": ini.get("year", "")}
        j = queue.add(src, meta["title"] or song_id, opts, upload=True, target=song_id, meta=meta)
        return j.to_dict()

    def update_export(song_id: str) -> str | None:
        """Copy the song to the Clone Hero folder again (or for the first time)."""
        try:
            return str(lib.export(song_id, overwrite=True))
        except ProtectedPathError as e:
            raise HTTPException(403, str(e))

    @app.post("/api/songs/{song_id}/meta")
    def set_meta(song_id: str, body: dict = Body(...)):
        """Song details (title, artist, album, genre, year, guitar difficulty) into song.ini. With
        "export": true the copy in the Clone Hero folder is updated too."""
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        try:
            fields = clean_meta(body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if fields:
            write_ini_fields(folder / "song.ini", fields)
        out = lib.info(song_id)
        if body.get("export"):
            out = dict(lib.info(song_id), exported_to=update_export(song_id))
            out.update(exported=True, export_stale=False)
        return out

    @app.post("/api/songs/{song_id}/art")
    async def set_art(song_id: str, file: UploadFile = File(...)):
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        data = await file.read(MAX_ART_BYTES + 1)
        try:
            write_art(folder, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return lib.info(song_id)

    @app.delete("/api/songs/{song_id}/art")
    def remove_art(song_id: str):
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        for n in ART_FILES:
            (folder / n).unlink(missing_ok=True)
        return lib.info(song_id)

    @app.delete("/api/songs/{song_id}")
    def remove_song(song_id: str):
        """Remove a song from the app's library: moved to data/ui/trash, not deleted. Exports are kept."""
        folder = song_folder(song_id)
        if any(j["target"] == song_id or j["song"] == song_id for j in queue.list() if j["status"] == "running"):
            raise HTTPException(409, "this song is being charted right now")
        trash = cfg["workspace"].parent / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        dest = trash / f"{song_id} ({time.strftime('%Y%m%d-%H%M%S')})"
        folder.rename(dest)
        return {"moved_to": str(dest)}

    @app.post("/api/songs/{song_id}/export")
    def export(song_id: str, body: dict = Body(default={})):
        song_folder(song_id)
        try:
            dest = lib.export(song_id, overwrite=bool(body.get("overwrite")))
        except FileExistsError:
            raise HTTPException(409, "already exported; export again to replace it")
        except ProtectedPathError as e:
            raise HTTPException(403, str(e))
        return {"folder": str(dest)}

    @app.post("/api/songs/{song_id}/test")
    def test_in_ch(song_id: str, body: dict = Body(default={})):
        folder = song_folder(song_id)
        diff = str(body.get("difficulty", "expert")).lower()
        if diff not in ("expert", "hard", "medium", "easy"):
            raise HTTPException(400, "difficulty must be expert, hard, medium or easy")
        try:
            cmd = clonehero.launch_preview(folder, diff)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return {"command": cmd}

    @app.post("/api/songs/{song_id}/reveal")
    def reveal(song_id: str):
        folder = song_folder(song_id)
        if os.name == "nt":
            subprocess.Popen(["explorer", str(folder)])
        return {"folder": str(folder)}

    @app.post("/api/clonehero/register")
    def register_export_folder():
        settings = cfg["clone_hero_settings"]
        if not settings or not settings.exists():
            raise HTTPException(404, "Clone Hero settings.ini not found (clone_hero_settings in configs/ui.yaml)")
        if clonehero.is_running():
            raise HTTPException(409, "close Clone Hero first: it overwrites settings.ini when it exits")
        lib.export_dir.mkdir(parents=True, exist_ok=True)
        changed = clonehero.add_song_folder(settings, lib.export_dir)
        return {"changed": changed}

    @app.exception_handler(ProtectedPathError)
    def protected(_, e):
        return JSONResponse({"detail": str(e)}, status_code=403)

    @app.middleware("http")
    async def no_stale_pages(request, call_next):
        # the page's files change with every update: make browsers revalidate instead of using a stale copy
        resp = await call_next(request)
        if not request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="riffchart-ui", description="RiffChart local web app")
    ap.add_argument("--config", default="configs/ui.yaml")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    port = args.port or int(cfg.get("port", 8765))
    host = cfg.get("host", "127.0.0.1")
    import uvicorn
    app = create_app(cfg)
    url = f"http://{host}:{port}/"
    print(f"RiffChart is running at {url}  (Ctrl+C to stop)", flush=True)
    if cfg.get("open_browser", True) and not args.no_browser:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    os.chdir(PROJECT)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
