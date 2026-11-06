"""Song metadata from audio tags (mutagen), falling back to an "Artist - Title" file name."""
from __future__ import annotations

import re
import unicodedata
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
                   r"(?: \d{4})?|explicit|clean|music video|mv|cc|subtitles|subbed)\b[^\)\]]*)[\)\]]", re.I)

# downloaders append the video id: "Song [dQw4w9WgXcQ]"
_YTID = re.compile(r"\s*\[[A-Za-z0-9_-]{11}\]$")

# converter sites: "song-128-ytshorts.savetube.me", "y2mate.com - Song", "Song (320 kbps)"
_TLD = r"(?:me|com|net|org|cc|io|to|app|co|ch|nu|is|tv|cx|ws|fm|pro|lol|site|online|biz|info)"
_SITE_END = re.compile(rf"[\s_-]*(?:[A-Za-z0-9]+\.)+{_TLD}$", re.I)
_SITE_START = re.compile(rf"^(?:[A-Za-z0-9]+\.)+{_TLD}\s*[-_]\s*", re.I)
_BITRATE = re.compile(r"[\s_-]*[\(\[]?\b(?:64|96|128|160|192|256|320)\s*(?:k|kbps)?[\)\]]?$", re.I)
_SLUG_JUNK = re.compile(r"(?:\s+(?:official|lyrics?|visuali[sz]er|lyric|music|video|audio|hd|4k|mv))+$", re.I)
# an uploader that is a record label, not the artist: "Roadrunner Records" + "Devildriver - Clouds ..."
_LABEL = re.compile(r"\b(?:records|recordings|entertainment|music group)$", re.I)
# "... Rock Cover by Our Last Night" outside brackets (in a file name)
_COVER_BY = re.compile(r"\s+(?:(\w+)\s+)?cover\s+by\s+([^()\[\]]+)$", re.I)


def _fold(s: str) -> str:
    """For comparing names: no accents, case or punctuation ("Måneskin" == "maneskin")."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if c.isalnum() and not unicodedata.combining(c)).casefold()


def _unquote(s: str) -> str:
    m = re.fullmatch(r"(['\"‘’“”]+)(.+?)(['\"‘’“”]+)", s)
    return m.group(2).strip() if m else s


def _tidy_text(s: str, slug: bool = True) -> str:
    """Site names, bitrates and file-name slugs ("front-porch-step-drown", "Choosin_Texas_-_Ella").
    slug=False for artists: "blink-182" is a band name, not a slug."""
    s = _YTID.sub("", _JUNK.sub("", s)).strip()
    for _ in range(2):
        s = _BITRATE.sub("", _SITE_END.sub("", _SITE_START.sub("", s))).strip()
    if not slug:
        pass
    elif " " not in s and "_" in s:
        s = re.sub(r"\s+", " ", s.replace("_", " ")).strip()
    elif " " not in s and s.count("-") >= 1 and s == s.lower() and re.search(r"[a-z]", s):
        s = _SLUG_JUNK.sub("", s.replace("-", " ")).strip()
        s = " ".join(w[:1].upper() + w[1:] for w in s.split())
    return re.sub(r"\s{2,}", " ", s).strip(" -_")


def _channel_part(p: str, artist_f: str) -> bool:
    """A title part that is the uploader's channel: "ManeskinVEVO", "Artist - Topic", the artist itself."""
    f = _fold(p)
    if not f:
        return True
    if f == artist_f or (artist_f and f in (artist_f + "vevo", artist_f + "official", artist_f + "tv")):
        return True
    return " " not in p.strip() and p.strip().endswith("VEVO")


# a title that starts with the artist: "SOiL Halo", "Muse: Uprising", "Muse | Uprising" -> the song. After
# only a space, a joining word means the artist's name is part of the title ("Queen of the Night").
_JOINERS = {"of", "the", "and", "in", "on", "is", "a", "an", "to", "for", "from", "with", "at", "by", "or", "&",
            "n", "vs", "x", "feat", "ft", "featuring", "s", "song", "theme", "blues", "rag", "boogie", "pt", "part"}
_SEP = re.compile(r"\s*(?:[:|~–—-]|//)\s*")


def strip_artist(title: str, artist: str, space_ok: bool = True) -> str:
    """The title without a leading copy of the artist's name (accents, case and punctuation ignored).
    space_ok=False: only after a separator ("Muse: Uprising"), not "SOiL Halo" (real titles do that: 4 of
    2355 human-named charts, e.g. "Ted Nugent Guitar Battle")."""
    a = _fold(artist)
    if not a or artist.casefold() == "unknown artist" or _fold(title) == a:
        return title
    acc, end = "", 0
    for m in re.finditer(r"[^\s:|~–—]+", title):               # shortest run of words that folds to the artist
        acc += _fold(m.group())
        end = m.end()
        if acc == a:
            break
        if not a.startswith(acc):
            return title
    else:
        return title
    rest = title[end:]
    sep = _SEP.match(rest)
    if sep and sep.end() > 0 and not rest[:1].isalnum():
        out = rest[sep.end():].strip()
    elif rest[:1] == " " and space_ok:
        out = rest.strip()
        first = re.sub(r"[^\w&]", "", out.split(" ", 1)[0]).casefold() if out else ""
        if not out or first in _JOINERS or out[:1] in "'’(":
            return title
    else:                                                       # "Kissin' Time" by Kiss: no word break
        return title
    return out if len(_fold(out)) >= 2 else title


def clean(meta: SongMeta, loose: bool = False) -> SongMeta:
    """Tidy names from downloads: "(Official Visualizer)"-style suffixes, converter-site names and bitrates,
    file-name slugs, "ArtistVEVO" / channel parts, a title that repeats the artist ("Artist - Title - Artist",
    "Title - Artist"), a title starting with the artist ("Muse: Uprising"; "SOiL Halo" only from a file-name slug,
    or with loose=True, for the Tidy names suggestions the user reviews) and a trailing part already named in
    the title ("Song (Cover by X) - X")."""
    artist = _tidy_text(meta.artist, slug=False) or meta.artist
    artist = re.sub(r"\s+-\s+Topic$", "", artist).strip() or artist
    title = _tidy_text(meta.title) or meta.title
    unknown = artist.casefold() in ("", "unknown artist")
    a = _fold(artist)
    parts = [_unquote(p.strip()) for p in re.split(r"\s+-\s+|\s+-$|^-\s+", title)]
    parts = [p for p in parts if p.strip(" -")]
    if len(parts) > 1:
        rest = [p for p in parts if not _channel_part(p, "" if unknown else a)]
        while len(rest) > 1 and len(_fold(rest[-1])) >= 4 and _fold(rest[-1]) in _fold(" ".join(rest[:-1])):
            rest = rest[:-1]                                # "... (Rock Cover by Our Last Night) - Our Last Night"
        if rest:
            parts = rest
    title = " - ".join(parts) or title
    cover = _COVER_BY.search(title)
    if unknown and cover:                                   # "Song - Artist Rock Cover by Band": the band plays it
        artist = cover.group(2).strip()
        title = title[:cover.start()].strip() + f" ({(cover.group(1) or '').strip().title() + ' ' if cover.group(1) else ''}Cover)"
        return SongMeta(title, artist, "", meta.genre, meta.year)
    if (unknown or _LABEL.search(artist)) and len(parts) == 2:
        artist, title = parts
    if not unknown:
        slug = " " not in meta.title.strip() and bool(re.search(r"[-_]", meta.title))
        title = strip_artist(title, artist, space_ok=loose or slug)
    album = _tidy_text(meta.album) if meta.album else ""
    if album and _fold(album) in (_fold(meta.title), _fold(title), _fold(f"{artist} - {title}")):
        album = ""                                          # downloaders often copy the video title here
    return SongMeta(title, artist, album, meta.genre, meta.year)


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


def cover_art(path: str | Path) -> bytes | None:
    """The cover image embedded in an audio file's tags (mp3 APIC, m4a covr, flac / ogg / opus pictures),
    PNG or JPEG only. Front cover first when the file says which is which."""
    try:
        import mutagen
        f = mutagen.File(str(path))
    except Exception:
        return None
    if f is None:
        return None
    pics: list[tuple[int, bytes]] = []                     # (picture type, data); 3 = front cover
    try:
        for p in getattr(f, "pictures", None) or []:       # flac
            pics.append((p.type, p.data))
        t = f.tags
        if t is not None:
            if hasattr(t, "getall"):                       # id3
                pics += [(a.type, a.data) for a in t.getall("APIC")]
            if "covr" in t:                                # mp4
                pics += [(3, bytes(c)) for c in t["covr"]]
            if "metadata_block_picture" in t:              # ogg vorbis / opus
                import base64
                from mutagen.flac import Picture
                for b in t["metadata_block_picture"]:
                    try:
                        pic = Picture(base64.b64decode(b))
                        pics.append((pic.type, pic.data))
                    except Exception:
                        pass
    except Exception:
        return None
    pics = [x for x in pics if x[1][:8] == b"\x89PNG\r\n\x1a\n" or x[1][:3] == b"\xff\xd8\xff"]
    if not pics:
        return None
    return sorted(pics, key=lambda x: x[0] != 3)[0][1]
