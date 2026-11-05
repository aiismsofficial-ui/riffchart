"""Song metadata from audio tags (mutagen), falling back to an "Artist - Title" file name."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SongMeta:
    title: str
    artist: str
    album: str = ""
    genre: str = ""
    year: str = ""


def _first(tags, key: str) -> str:
    try:
        v = tags.get(key)
    except Exception:
        return ""
    if not v:
        return ""
    return str(v[0] if isinstance(v, list) else v).strip()


GENERIC = {"song", "audio", "track", "mix", "full", "guitar", "preview", "music"}


def from_filename(path: Path) -> SongMeta:
    # CH-style folders hold "song.ogg": take "Artist - Title" from the folder name instead
    name = path.parent.name if path.stem.lower() in GENERIC else path.stem
    stem = re.sub(r"^\d{1,3}\s*[-._]\s+", "", name).strip()        # "01 - " / "01. " track numbers
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return SongMeta(title.strip() or path.stem, artist.strip() or "Unknown Artist")
    return SongMeta(stem or path.stem, "Unknown Artist")


# video-site leftovers in titles: "(Official Video)", "[Official Audio]", "(Lyric Video)", "(HD)", ...
_JUNK = re.compile(r"\s*[\(\[](?:[^\)\]]*\b(?:official|lyrics?|visuali[sz]er|audio|video|hd|hq|4k|remaster(?:ed)?"
                   r"(?: \d{4})?|explicit|clean|music video|mv)\b[^\)\]]*)[\)\]]", re.I)


def clean(meta: SongMeta) -> SongMeta:
    """Tidy names from downloads: drop "(Official Visualizer)"-style suffixes and a title that repeats
    the artist ("Artist - Title - Artist", "Title - Artist")."""
    artist = _JUNK.sub("", meta.artist).strip() or meta.artist
    title = _JUNK.sub("", meta.title).strip() or meta.title
    a = artist.casefold()
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) > 1:
        rest = [p for p in parts if p.casefold() != a]
        if rest and len(rest) < len(parts):
            title = " - ".join(rest)
    if artist.casefold() in ("", "unknown artist") and len(parts) == 2:
        artist, title = parts
    return SongMeta(title, artist, meta.album, meta.genre, meta.year)


def read_meta(path: str | Path) -> SongMeta:
    return clean(_read_meta(path))


def _read_meta(path: str | Path) -> SongMeta:
    path = Path(path)
    meta = from_filename(path)
    try:
        import mutagen
        f = mutagen.File(str(path), easy=True)
    except Exception:
        f = None
    if f is None or f.tags is None:
        return meta
    t = f.tags
    year = _first(t, "date")[:4]
    return SongMeta(_first(t, "title") or meta.title, _first(t, "artist") or meta.artist, _first(t, "album"),
                    _first(t, "genre"), year if year.isdigit() else "")
