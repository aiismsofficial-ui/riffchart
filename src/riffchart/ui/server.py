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
from fastapi.responses import FileResponse, JSONResponse, Response
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
from ..product import covers
from ..product import highways
from ..product.visuals import (VIDEO_FILES, default_highway, hw_mode, refresh_background, refresh_highway,
                               set_bg_mode, set_hw_mode, write_own_background, write_own_highway)
from .library import (MAX_ART_BYTES, Library, ProtectedPathError, check_writable, clean_meta, remove_art,
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
    # highway designs (and the library's default highway) live next to the songs unless configured
    highways.DESIGNS_DIR = (project_path(cfg["highways"]) if cfg.get("highways")
                            else Path(cfg["workspace"]).parent / "highways")
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
        try:                                       # keep the chart's own [Song] Offset: the page sends ticks
            song.chart_offset_s = load_song(folder)[0].chart_offset_s      # relative to it
        except Exception:
            pass
        text = write_chart_text(song, name=ini.get("name", ""), artist=ini.get("artist", ""),
                                charter=ini.get("charter", "RiffChart"), ini=ini)
        hist = folder / ".riffchart" / "history"
        hist.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if list(hist.glob(f"{stamp}-*notes.*")):          # two saves in one second: keep both
            time.sleep(1.0)
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

    @app.post("/api/songs/{song_id}/preview")
    def set_preview(song_id: str, body: dict = Body(...)):
        """Where the song-list preview starts (song.ini preview_start_time, ms). Clone Hero and YARG play
        about 30 s from there."""
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        try:
            start = float(body["start_s"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "start_s must be a number of seconds")
        length = float(_ini(folder).get("song_length", 0) or 0) / 1000.0
        if not (0 <= start and (length <= 0 or start < length)):
            raise HTTPException(400, "the preview must start inside the song")
        write_ini_fields(folder / "song.ini", {"preview_start_time": str(int(round(start * 1000)))})
        return {"preview_s": round(start, 3), "export_stale": lib.export_stale(song_id)}

    # ---- chart history: the version each save / re-chart replaced, in .riffchart/history
    hist_counts: dict[tuple, dict] = {}

    def history_files(folder: Path) -> list[Path]:
        h = folder / ".riffchart" / "history"
        return sorted(h.glob("*-notes.*"), reverse=True) if h.is_dir() else []

    def history_song(folder: Path, f: Path):
        from ..formats.chart_parser import parse_chart
        from ..formats.midi_parser import parse_midi
        ini = _ini(folder)
        return (parse_midi(f, ini) if f.suffix.lower() in (".mid", ".midi") else parse_chart(f, ini)), ini

    @app.get("/api/songs/{song_id}/history")
    def history(song_id: str):
        """Newest first: name, made (when that version was written), replaced (when a save or re-chart
        replaced it), reason (save / rechart), notes per difficulty."""
        folder = song_folder(song_id)
        out = []
        for f in history_files(folder):
            m = re.match(r"(\d{8}-\d{6})-(?:(rechart)-)?notes\.", f.name)
            if not m:
                continue
            st = f.stat()
            key = (str(f), st.st_size, int(st.st_mtime))
            if key not in hist_counts:
                try:
                    song, _ = history_song(folder, f)
                    hist_counts[key] = {d: len(t) for d, t in song.tracks.items()}
                except Exception as e:                       # unreadable: still listed
                    hist_counts[key] = {"error": str(e)[:200]}
            replaced = time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M%S"))
            out.append({"name": f.name, "made": st.st_mtime, "replaced": replaced,
                        "reason": m.group(2) or "save", "notes": hist_counts[key]})
        return out

    @app.get("/api/songs/{song_id}/history/{name}")
    def history_version(song_id: str, name: str):
        """One old version as editor JSON (to look at it and, by saving, restore it)."""
        folder = song_folder(song_id)
        f = next((x for x in history_files(folder) if x.name == name), None)
        if f is None:
            raise HTTPException(404, f"no history version {name!r}")
        try:
            song, ini = history_song(folder, f)
        except Exception as e:
            raise HTTPException(422, f"that version can't be read: {e}")
        return song_to_json(song, ini)

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
            if {"name", "artist"} & set(fields):
                refresh_background(folder)             # the background shows the title and artist
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
    def delete_art(song_id: str):
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        remove_art(folder)
        return lib.info(song_id)

    @app.post("/api/songs/{song_id}/art/url")
    def set_art_from_url(song_id: str, body: dict = Body(...)):
        """A cover the user picked from the online search. Only cover hosts are fetched."""
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        try:
            data = covers.fetch(str(body.get("url", "")), MAX_ART_BYTES)
            write_art(folder, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except OSError as e:
            raise HTTPException(502, f"couldn't download that cover: {e}")
        return lib.info(song_id)

    @app.get("/api/covers/search")
    def cover_search(artist: str = "", title: str = "", album: str = ""):
        """Cover candidates from the iTunes Search API and MusicBrainz / Cover Art Archive. Only the artist,
        title and album are sent, and only when the user asks (the details panel's Find online)."""
        if not (artist.strip() or title.strip()):
            raise HTTPException(400, "need an artist or a title to search for")
        return covers.search(artist.strip(), title.strip(), album.strip())

    @app.post("/api/songs/{song_id}/background")
    def set_background(song_id: str, body: dict = Body(...)):
        """mode: auto (made from the cover) | none. ("own" is set by uploading an image.)"""
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        mode = body.get("mode")
        if mode not in ("auto", "none"):
            raise HTTPException(400, "background mode must be auto or none")
        set_bg_mode(folder, mode)
        refresh_background(folder)
        return lib.info(song_id)

    @app.post("/api/songs/{song_id}/background/file")
    async def set_background_file(song_id: str, file: UploadFile = File(...)):
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        try:
            write_own_background(folder, await file.read((20 << 20) + 1))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return lib.info(song_id)

    @app.post("/api/songs/{song_id}/highway")
    def set_highway(song_id: str, body: dict = Body(...)):
        """mode: riffchart | cover (colours and a faint copy of the cover) | design (+ "design": preset id) |
        none. ("own" by uploading.)"""
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        mode = body.get("mode")
        if mode not in ("riffchart", "cover", "design", "none"):
            raise HTTPException(400, "highway mode must be riffchart, cover, design or none")
        if mode == "design":
            _design_or_404(str(body.get("design") or ""))
        set_hw_mode(folder, mode, body.get("design") if mode == "design" else None)
        refresh_highway(folder)
        return lib.info(song_id)

    # ---- highway designer
    def _design_or_404(i: str):
        try:
            return highways.get_design(i)
        except (KeyError, ValueError):
            raise HTTPException(404, f"no highway design {i!r}")

    def ch_highways_dir() -> Path | None:
        st = cfg.get("clone_hero_settings")
        return Path(st).parent / "Custom" / "Highways" if st and Path(st).parent.exists() else None

    @app.get("/api/highways")
    def list_highways():
        return {"designs": highways.list_designs(), "default": default_highway(), "blank": highways.DEFAULT,
                "clone_hero": bool(ch_highways_dir())}

    async def _form_design(design: str, pattern, pattern_from: str):
        import json as _json
        try:
            d = highways.clean_design(_json.loads(design))
        except ValueError as e:
            raise HTTPException(400, str(e))
        pat = None
        if pattern is not None:
            try:
                pat = highways.clean_pattern(await pattern.read(highways.MAX_PATTERN + 1))
            except ValueError as e:
                raise HTTPException(400, str(e))
        elif pattern_from:
            pat = _design_or_404(pattern_from)[1]
        return d, pat

    @app.post("/api/highways/render")
    async def render_highway(design: str = Form(...), pattern: UploadFile | None = File(None),
                             pattern_from: str = Form("")):
        """Live preview of an unsaved design (pattern: a new image, or pattern_from: a saved design's)."""
        d, pat = await _form_design(design, pattern, pattern_from)
        return Response(highways.render(d, pat), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/highways")
    async def save_highway(design: str = Form(...), pattern: UploadFile | None = File(None),
                           pattern_from: str = Form(""), remove_pattern: bool = Form(False)):
        """Create (no id) or update a preset. Songs already using it are redrawn, and their Clone Hero copies
        updated if those were up to date."""
        d, pat = await _form_design(design, pattern, pattern_from)
        if remove_pattern:
            pat = None
        elif pattern is None and pattern_from and pattern_from == d.get("id"):
            pat = None                                   # unchanged: keep the saved one
        saved = highways.save_design(d, pat, keep_pattern=not remove_pattern)
        using = [f.name for f in highways.songs_using(lib.workspace, saved["id"])]
        res = apply_highway(using, "design", saved["id"], export=True)
        return {"design": saved, **res}

    @app.delete("/api/highways/{design_id}")
    def delete_highway(design_id: str):
        _design_or_404(design_id)
        highways.delete_design(design_id)
        return {"ok": True, "default": default_highway()}

    @app.get("/api/highways/{design_id}/image")
    def highway_image(design_id: str):
        _design_or_404(design_id)
        return FileResponse(highways.preview_path(design_id), media_type="image/png",
                            headers={"Cache-Control": "no-cache"})

    def apply_highway(song_ids, mode: str, design: str | None, export: bool, skip_own: bool = False) -> dict:
        updated, exported, skipped = [], [], []
        for sid in song_ids:
            try:
                folder = lib.folder(sid)
            except KeyError:
                continue
            if skip_own and hw_mode(folder) == "own":
                skipped.append(sid)
                continue
            check_writable(folder, lib.protected)
            current = (lib.export_dir / sid / "notes.chart").exists() and not lib.export_stale(sid)
            set_hw_mode(folder, mode, design)
            refresh_highway(folder)
            updated.append(sid)
            if export and current:
                lib.export(sid, overwrite=True)
                exported.append(sid)
        return {"updated": updated, "exported": exported, "skipped_own": skipped}

    @app.post("/api/highways/apply")
    def apply_highways(body: dict = Body(...)):
        """{"highway": riffchart | cover | none | design:<id>, "songs": "all" | [ids], "export": bool,
        "default": bool}. all: every song except those with their own image; export: also update Clone Hero
        copies that were up to date; default: new songs get it too."""
        v = str(body.get("highway", ""))
        mode, design = ("design", v[7:]) if v.startswith("design:") else (v, None)
        if mode not in ("riffchart", "cover", "none", "design"):
            raise HTTPException(400, "highway must be riffchart, cover, none or design:<id>")
        if design is not None:
            _design_or_404(design)
        songs = body.get("songs") or []
        if songs == "all":
            ids = [p.name for p in sorted(lib.workspace.iterdir()) if p.is_dir()]
        elif isinstance(songs, list):
            ids = [str(x) for x in songs]
        else:
            raise HTTPException(400, 'songs must be "all" or a list of song ids')
        res = apply_highway(ids, mode, design, bool(body.get("export")), skip_own=songs == "all")
        if body.get("default"):
            highways.set_library_default(v)
        return dict(res, default=default_highway())

    @app.post("/api/highways/default")
    def set_default_highway(body: dict = Body(...)):
        try:
            highways.set_library_default(str(body.get("highway", "")))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))
        return {"default": default_highway()}

    @app.post("/api/highways/{design_id}/install")
    def install_highway(design_id: str):
        """Copy the design into Clone Hero's Custom/Highways folder (players pick it in Clone Hero for every
        song)."""
        _design_or_404(design_id)
        dest = ch_highways_dir()
        if dest is None:
            raise HTTPException(404, "Clone Hero's folder wasn't found (clone_hero_settings in configs/ui.yaml)")
        check_writable(dest, lib.protected)
        return {"installed": str(highways.install_in_clone_hero(design_id, dest))}

    @app.post("/api/songs/{song_id}/highway/file")
    async def set_highway_file(song_id: str, file: UploadFile = File(...)):
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        try:
            write_own_highway(folder, await file.read((10 << 20) + 1))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return lib.info(song_id)

    @app.delete("/api/songs/{song_id}/video")
    def delete_video(song_id: str):
        folder = song_folder(song_id)
        check_writable(folder, lib.protected)
        for n in VIDEO_FILES:
            (folder / n).unlink(missing_ok=True)
        write_ini_fields(folder / "song.ini", {"video_start_time": ""})
        return lib.info(song_id)

    def tidy_changes() -> list[dict]:
        """Songs whose names the download tidier would change: [{id, old, new}] (title / artist / album)."""
        from ..product.meta import SongMeta, clean
        out = []
        for s in lib.list():
            if s.get("error"):
                continue
            old = {"title": s["title"], "artist": s["artist"], "album": s.get("album", "")}
            m = clean(SongMeta(old["title"], old["artist"], old["album"]), loose=True)
            new = {"title": m.title, "artist": m.artist, "album": m.album}
            if new != old:
                out.append({"id": s["id"], "old": old, "new": new, "exported": s.get("exported", False)})
        return out

    @app.get("/api/tidy")
    def tidy_preview():
        return tidy_changes()

    @app.post("/api/tidy")
    def tidy_apply(body: dict = Body(...)):
        """Apply the tidied names to the chosen songs; "export": true also updates their Clone Hero copies."""
        want = set(body.get("ids") or [])
        done, exported, failed = [], [], []
        for ch in tidy_changes():
            if ch["id"] not in want:
                continue
            try:
                folder = song_folder(ch["id"])
                check_writable(folder, lib.protected)
                write_ini_fields(folder / "song.ini", clean_meta(ch["new"]))
                refresh_background(folder)
                done.append(ch["id"])
                if body.get("export") and ch["exported"]:
                    lib.export(ch["id"], overwrite=True)
                    exported.append(ch["id"])
            except (HTTPException, ValueError, OSError) as e:
                failed.append({"id": ch["id"], "error": getattr(e, "detail", None) or str(e)})
        return {"renamed": done, "exported": exported, "failed": failed}

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
