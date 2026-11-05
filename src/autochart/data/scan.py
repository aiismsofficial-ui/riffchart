"""Find song folders in the (read-only) chart library."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..formats.loader import CHART_FILES

AUDIO_EXTS = (".ogg", ".opus", ".mp3", ".wav", ".flac")
# Stems that together form the full mix. preview/crowd are excluded on purpose.
MIX_STEMS = ("song", "guitar", "rhythm", "bass", "drums", "drums_1", "drums_2", "drums_3", "drums_4",
             "keys", "vocals", "vocals_1", "vocals_2", "backing")
GUITAR_STEMS = ("guitar",)


@dataclass
class SongFolder:
    song_id: str
    path: Path
    chart: Path
    audio: dict[str, Path] = field(default_factory=dict)  # stem name -> file

    @property
    def has_guitar_stem(self) -> bool:
        return "guitar" in self.audio

    @property
    def mix_files(self) -> list[Path]:
        return [self.audio[s] for s in MIX_STEMS if s in self.audio]


def song_id_for(rel: str) -> str:
    return hashlib.sha1(rel.replace("\\", "/").lower().encode("utf-8")).hexdigest()[:12]


def scan_library(root: str | Path, tag: str = "") -> list[SongFolder]:
    """tag: prefix for the song-id hash of extra libraries ("" for the main library)."""
    root = Path(root)
    out = []
    for dirpath, _dirs, files in os.walk(root):
        lower = {f.lower(): f for f in files}
        chart = next((lower[n] for n in CHART_FILES if n in lower), None)
        if chart is None:
            continue
        audio = {}
        for lf, f in lower.items():
            stem, ext = os.path.splitext(lf)
            if ext in AUDIO_EXTS and stem in MIX_STEMS + ("preview", "crowd"):
                # prefer .ogg > .opus > others if duplicates exist
                if stem not in audio or AUDIO_EXTS.index(ext) < AUDIO_EXTS.index(audio[stem].suffix.lower()):
                    audio[stem] = Path(dirpath) / f
        d = Path(dirpath)
        rel = str(d.relative_to(root))
        out.append(SongFolder(song_id_for(f"{tag}/{rel}" if tag else rel), d, d / chart, audio))
    out.sort(key=lambda s: str(s.path).lower())
    return out


def scan_all(cfg: dict) -> list[tuple[str, SongFolder]]:
    """(library tag, song) for the main library ("main") and every extra library."""
    out = [("main", s) for s in scan_library(cfg["library_root"])]
    for lib in cfg.get("extra_libraries") or []:
        out += [(lib["tag"], s) for s in scan_library(lib["root"], lib["tag"])]
    return out
