"""What "Bring up to date" would add to a library song (no charting): its own instrument tracks, lyrics,
backgrounds at the sharpness setting. The work itself is RiffCharter.update_song (core.py); this module only
looks at the folder, so the library can show it without loading any model.

A song that was brought up to date is marked in .riffchart/update.json, so a song with no lyrics online or no
instrument loud enough for its own track isn't offered again until something new can be added (VERSION).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

VERSION = 2                       # raise when update_song learns something new for old songs (2: bass charts)
MARK = ".riffchart/update.json"
STEMS = ("bass", "drums", "vocals", "keys")
AUDIO = {".ogg", ".opus", ".mp3", ".wav"}


def marked(folder: Path) -> int:
    try:
        return int(json.loads((Path(folder) / MARK).read_text(encoding="utf-8")).get("version", 0))
    except Exception:
        return 0


def mark(folder: Path, result: dict) -> None:
    p = Path(folder) / MARK
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": VERSION, "when": time.time(), "result": result}, indent=1),
                 encoding="utf-8")


_cfg: dict = {}


def settings() -> tuple[bool, bool]:
    """(split instruments, lyrics online) from configs/product.yaml (+ .local), re-read when it changes."""
    from ..paths import load_config, project_path
    p = project_path("configs/product.yaml")
    local = p.with_name("product.local.yaml")
    key = tuple(f.stat().st_mtime if f.exists() else 0 for f in (p, local))
    if _cfg.get("key") != key:
        c = load_config("configs/product.yaml")
        _cfg.update(key=key, v=(bool(c.get("audio", {}).get("split_instruments", True)),
                                c.get("lyrics", "online") in ("online", "local")))
    return _cfg["v"]


def _ours(folder: Path) -> bool:
    """Charted by RiffChart (a human chart never gets an AI part added to it)."""
    from ..formats.textio import parse_ini
    ini = Path(folder) / "song.ini"
    from .bass import ours
    return ini.exists() and ours(parse_ini(ini))


def needs(folder: Path, split: bool | None = None, lyrics_online: bool | None = None) -> list[str]:
    """[] when nothing to do, else some of "instruments", "bass", "lyrics", "background"."""
    if split is None or lyrics_online is None:
        s, o = settings()
        split = s if split is None else split
        lyrics_online = o if lyrics_online is None else lyrics_online
    from . import lyrics
    from .visuals import bg_anim, bg_mode, find_first, BG_FILES, still_size
    folder = Path(folder)
    if marked(folder) >= VERSION:
        return []
    out = []
    audio = [f.stem.lower() for f in folder.iterdir() if f.suffix.lower() in AUDIO
             and not f.stem.lower().startswith("preview")]
    if split and "song" in audio and set(audio) <= {"song", "guitar"}:
        out.append("instruments")
    chart = folder / "notes.chart"
    from . import bass
    if chart.exists() and bass.enabled() and ("bass" in audio or "instruments" in out) and not bass.in_chart(folder) and _ours(folder):
        out.append("bass")
    if lyrics_online and chart.exists() and lyrics.load(folder) is None:
        out.append("lyrics")
    if bg_anim(folder):
        out.append("background")
    elif bg_mode(folder) == "auto":
        bg = find_first(folder, BG_FILES)
        try:
            from PIL import Image
            small = bg is not None and Image.open(bg).size[1] < still_size()[1]
        except Exception:
            small = False
        if small:
            out.append("background")
    return out
