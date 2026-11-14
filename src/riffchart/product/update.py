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

VERSION = 4                       # raise when update_song learns something new (2: bass, 3: keys, 4: webm video)
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


def settings() -> tuple[bool, bool, bool]:
    """(split instruments, lyrics online, music videos to VP8 webm) from configs/product.yaml (+ .local),
    re-read when it changes."""
    from ..paths import load_config, project_path
    p = project_path("configs/product.yaml")
    local = p.with_name("product.local.yaml")
    key = tuple(f.stat().st_mtime if f.exists() else 0 for f in (p, local))
    if _cfg.get("key") != key:
        c = load_config("configs/product.yaml")
        _cfg.update(key=key, v=(bool(c.get("audio", {}).get("split_instruments", True)),
                                c.get("lyrics", "online") in ("online", "local"),
                                bool(c.get("convert_video", True) and c.get("video_webm", True))))
    return _cfg["v"]


def _ours(folder: Path) -> bool:
    """Charted by RiffChart (a human chart never gets an AI part added to it)."""
    from ..formats.textio import parse_ini
    ini = Path(folder) / "song.ini"
    from .bass import ours
    return ini.exists() and ours(parse_ini(ini))


def chart_has_lyrics(folder: Path) -> bool:
    """notes.chart has lyric events of its own (typed in Moonscraper, or a hand-made chart)."""
    try:
        return '= E "lyric ' in (Path(folder) / "notes.chart").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def wants_lyrics(folder: Path, ini: dict) -> bool:
    """Lyrics may be added: a RiffChart .chart with neither lyrics.json nor lyric events of its own."""
    from . import lyrics
    from .bass import ours
    return ((Path(folder) / "notes.chart").exists() and ours(ini) and lyrics.load(folder) is None
            and not chart_has_lyrics(folder))


def needs(folder: Path, split: bool | None = None, lyrics_online: bool | None = None,
          webm: bool | None = None) -> list[str]:
    """[] when nothing to do, else some of "instruments", "bass", "keys", "lyrics", "video", "background"."""
    if split is None or lyrics_online is None or webm is None:
        s, o, w = settings()
        split = s if split is None else split
        lyrics_online = o if lyrics_online is None else lyrics_online
        webm = w if webm is None else webm
    from . import lyrics
    from .visuals import bg_anim, bg_mode, find_first, BG_FILES, still_size, video_info, wants_convert
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
    for part, spec in bass.PARTS.items():         # bass / keys charts for RiffChart songs that have the track
        has = spec["file"][:-4] in audio or ("instruments" in out and part == "bass")
        if chart.exists() and has and bass.enabled(part=part) and not bass.in_chart(folder, part) and _ours(folder):
            out.append(part)
    if lyrics_online and _ours(folder):
        from ..formats.textio import parse_ini
        if wants_lyrics(folder, parse_ini(folder / "song.ini")):
            out.append("lyrics")
    anim = bg_anim(folder)
    if webm and not anim and wants_convert(video_info(folder)):
        out.append("video")                       # a music video as .mp4 (or VP9 / AV1): to VP8 webm
    if anim:
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
