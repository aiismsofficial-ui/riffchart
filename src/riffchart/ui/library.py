"""Songs in the app's workspace: listing, export to the Clone Hero folder, path safety."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..formats.loader import load_song
from ..formats.song import DIFFICULTIES

AUDIO_FILES = {".ogg", ".opus", ".mp3", ".wav"}


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
        hit = self._cache.get(song_id)
        if hit and hit[0] == mtime:
            d = dict(hit[1])
        else:
            song, ini, _ = load_song(p)
            d = {"id": song_id, "title": ini.get("name", song_id), "artist": ini.get("artist", ""),
                 "charter": ini.get("charter", ""), "diff_guitar": ini.get("diff_guitar", ""),
                 "length_s": round(float(ini.get("song_length", 0) or 0) / 1000, 1),
                 "notes": {dd: len(song.tracks[dd]) for dd in DIFFICULTIES if dd in song.tracks},
                 "charted": mtime}
            self._cache[song_id] = (mtime, d)
            d = dict(d)
        d["stems"] = sorted(f.name for f in p.iterdir() if f.suffix.lower() in AUDIO_FILES)
        exp = self.export_dir / song_id / "notes.chart"
        d["exported"] = exp.exists()
        d["export_stale"] = d["exported"] and exp.stat().st_mtime < mtime - 1
        return d

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
