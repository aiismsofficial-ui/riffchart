"""Songs in the app's workspace: listing, export to the Clone Hero folder, path safety."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..formats.loader import load_song
from ..formats.song import DIFFICULTIES
from ..product.highways import song_design
from ..product.visuals import (ART_FILES, BG_FILES, HW_FILES, VIDEO_FILES, bg_mode, find_art, find_first, hw_mode,
                               image_ext, refresh_visuals)

AUDIO_FILES = {".ogg", ".opus", ".mp3", ".wav"}
# song.ini keys the details panel edits, in file order: (key, field name in the API)
META_KEYS = (("name", "title"), ("artist", "artist"), ("album", "album"), ("genre", "genre"), ("year", "year"),
             ("diff_guitar", "diff_guitar"))
MAX_ART_BYTES = 10 << 20


def write_art(folder: Path, data: bytes) -> str:
    """Save a cover as album.png / album.jpg (by content), removing the other one. A background that follows
    the cover is made again."""
    ext = image_ext(data)
    if ext is None:
        raise ValueError("the cover must be a PNG or JPEG image")
    if len(data) > MAX_ART_BYTES:
        raise ValueError("the cover is too big (10 MB max)")
    name = f"album.{ext}"
    tmp = folder / f"{name}.tmp"
    tmp.write_bytes(data)
    tmp.replace(folder / name)
    for other in ART_FILES:
        if other != name:
            (folder / other).unlink(missing_ok=True)
    refresh_visuals(folder)
    return name


def remove_art(folder: Path) -> None:
    for n in ART_FILES:
        (folder / n).unlink(missing_ok=True)
    refresh_visuals(folder)


def _one_line(v) -> str:
    return " ".join(str(v if v is not None else "").split())          # no newlines into song.ini


def clean_meta(body: dict) -> dict[str, str]:
    """Validated song.ini values from the details form (only the fields it sent)."""
    out = {}
    for key, field in META_KEYS:
        if field not in body:
            continue
        v = _one_line(body[field])[:200]
        if field == "title" and not v:
            raise ValueError("the title can't be empty")
        if field == "year" and v and not (v.isdigit() and len(v) <= 4):
            raise ValueError("the year should be a number like 1999")
        if field == "diff_guitar" and v:
            if not (v.lstrip("-").isdigit() and -1 <= int(v) <= 6):
                raise ValueError("the guitar difficulty should be 0 to 6")
            v = str(int(v))
        out[key] = v
    return out


def write_ini_fields(path: Path, fields: dict[str, str]) -> None:
    """Set (or, when empty, remove) keys in song.ini, keeping every other line as it is."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else ["[song]"]
    if not lines or not lines[0].strip().startswith("["):
        lines.insert(0, "[song]")
    out, seen = [], set()
    for line in lines:
        k = line.split("=", 1)[0].strip().lower() if "=" in line else ""
        if k in fields:
            if k in seen or not fields[k]:
                continue                                   # duplicate or cleared
            line = f"{k} = {fields[k]}"
            seen.add(k)
        out.append(line)
    at = 1
    for key, _ in META_KEYS:                               # new keys go near the top, in the usual order
        if key in fields and fields[key] and key not in seen:
            out.insert(at, f"{key} = {fields[key]}")
            at += 1
        elif key in seen:
            at = next((i + 1 for i, x in enumerate(out) if x.split("=", 1)[0].strip().lower() == key), at)
    known = {k for k, _ in META_KEYS}
    out += [f"{k} = {v}" for k, v in fields.items() if v and k not in seen and k not in known]
    tmp = path.with_suffix(".ini.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.replace(path)


class ProtectedPathError(PermissionError):
    pass


def is_inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def check_writable(path: Path, protected: list[Path]) -> None:
    """Refuse any write inside a protected folder (the user's chart libraries are read-only)."""
    for root in protected:
        if is_inside(path, root) or is_inside(root, path):
            raise ProtectedPathError(f"{path} is inside the read-only folder {root}")


class Library:
    def __init__(self, workspace: Path, export_dir: Path, protected: list[Path]):
        self.workspace = Path(workspace)
        self.export_dir = Path(export_dir)
        self.protected = [Path(p) for p in protected]
        check_writable(self.workspace, self.protected)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[float, dict]] = {}

    def folder(self, song_id: str) -> Path:
        """Workspace folder for a song id (its folder name), rejecting anything that escapes the workspace."""
        p = self.workspace / song_id
        if not song_id or song_id in (".", "..") or "/" in song_id or "\\" in song_id or not is_inside(p, self.workspace):
            raise KeyError(song_id)
        if not (p / "notes.chart").exists() and not (p / "notes.mid").exists():
            raise KeyError(song_id)
        return p

    def info(self, song_id: str) -> dict:
        p = self.folder(song_id)
        chart = p / "notes.chart" if (p / "notes.chart").exists() else p / "notes.mid"
        mtime = chart.stat().st_mtime
        ini_p = p / "song.ini"
        key = (mtime, ini_p.stat().st_mtime if ini_p.exists() else 0.0)
        hit = self._cache.get(song_id)
        if hit and hit[0] == key:
            d = dict(hit[1])
        else:
            song, ini, _ = load_song(p)
            d = {"id": song_id, "title": ini.get("name", song_id), "artist": ini.get("artist", ""),
                 "album": ini.get("album", ""), "genre": ini.get("genre", ""), "year": ini.get("year", ""),
                 "charter": ini.get("charter", ""), "diff_guitar": ini.get("diff_guitar", ""),
                 "length_s": round(float(ini.get("song_length", 0) or 0) / 1000, 1),
                 "notes": {dd: len(song.tracks[dd]) for dd in DIFFICULTIES if dd in song.tracks},
                 "charted": mtime}
            self._cache[song_id] = (key, d)
            d = dict(d)
        d["stems"] = sorted(f.name for f in p.iterdir() if f.suffix.lower() in AUDIO_FILES)
        art = find_art(p)
        d["art"] = f"{art.name}?v={int(art.stat().st_mtime)}" if art else ""
        bg = find_first(p, BG_FILES)
        d["background"] = {"file": f"{bg.name}?v={int(bg.stat().st_mtime)}" if bg else "", "mode": bg_mode(p)}
        vid = find_first(p, VIDEO_FILES)
        d["video"] = {"file": vid.name, "mb": round(vid.stat().st_size / 2**20, 1)} if vid else None
        hw = find_first(p, HW_FILES)
        d["highway"] = {"file": f"{hw.name}?v={int(hw.stat().st_mtime)}" if hw else "", "mode": hw_mode(p)}
        if d["highway"]["mode"] == "design":
            sd = song_design(p)
            d["highway"]["design"] = {"id": sd[0].get("id", ""), "name": sd[0]["name"]} if sd else None
        d["exported"] = (self.export_dir / song_id / "notes.chart").exists()
        d["export_stale"] = d["exported"] and self.export_stale(song_id)
        return d

    def export_stale(self, song_id: str) -> bool:
        """True if the chart, song.ini, cover, background or video changed since the copy in the export folder
        was made (export keeps file times, so a newer file here means an edit)."""
        src, dest = self.workspace / song_id, self.export_dir / song_id
        for name in ("notes.chart", "notes.mid", "song.ini") + ART_FILES + BG_FILES + VIDEO_FILES + HW_FILES:
            a, b = src / name, dest / name
            if a.exists() != b.exists():
                return True
            if a.exists() and a.stat().st_mtime > b.stat().st_mtime + 1:
                return True
        return False

    def list(self) -> list[dict]:
        out = []
        for p in self.workspace.iterdir():
            if p.is_dir() and ((p / "notes.chart").exists() or (p / "notes.mid").exists()):
                try:
                    out.append(self.info(p.name))
                except Exception as e:                    # an unreadable chart still shows up
                    out.append({"id": p.name, "title": p.name, "artist": "", "error": str(e), "notes": {},
                                "charted": p.stat().st_mtime, "stems": [], "exported": False})
        return sorted(out, key=lambda d: -d["charted"])

    def export(self, song_id: str, overwrite: bool = False) -> Path:
        """Copy a song folder into the export (Clone Hero) folder. Existing exports are replaced only with
        overwrite; the replaced copy is kept as <name>.old until the new copy is complete."""
        src = self.folder(song_id)
        dest = self.export_dir / song_id
        check_writable(dest, self.protected)
        if dest.exists() and not overwrite:
            raise FileExistsError(f"{dest} already exists")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.export_dir / f".{song_id}.part-{int(time.time())}"
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("*.bak", ".riffchart", ".autochart"))
        if dest.exists():
            old = self.export_dir / f".{song_id}.old"
            shutil.rmtree(old, ignore_errors=True)
            dest.rename(old)
            tmp.rename(dest)
            shutil.rmtree(old, ignore_errors=True)
        else:
            tmp.rename(dest)
        return dest
