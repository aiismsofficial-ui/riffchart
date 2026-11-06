"""Online cover search (opt-in, only when the user presses Find online): the iTunes Search API and
MusicBrainz + Cover Art Archive. Only the artist, title and album text are sent. Downloads are limited to the
cover hosts these services use, over https, with a size cap."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

# MusicBrainz asks every client to identify itself (https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting)
USER_AGENT = "RiffChart/1.0 (open-source Clone Hero / YARG autocharter; +https://huggingface.co/riffchart)"
COVER_HOSTS = ("mzstatic.com", "coverartarchive.org", "archive.org")
TIMEOUT = 10
_last_mb = 0.0


def _host_ok(url: str) -> bool:
    u = urllib.parse.urlsplit(url)
    host = (u.hostname or "").lower()
    return u.scheme == "https" and any(host == h or host.endswith("." + h) for h in COVER_HOSTS)


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _host_ok(newurl):
            raise urllib.error.URLError(f"redirect to an unexpected host: {urllib.parse.urlsplit(newurl).hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_CheckedRedirect)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read(4 << 20).decode("utf-8"))


def fetch(url: str, max_bytes: int) -> bytes:
    """Download a picked cover. Only https cover hosts (redirects checked too); PNG / JPEG checked by the caller."""
    if not _host_ok(url):
        raise ValueError("that isn't a cover from the search")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _opener.open(req, timeout=TIMEOUT * 2) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("that cover is too big")
    return data


def itunes(artist: str, title: str, limit: int = 8) -> list[dict]:
    term = " ".join(x for x in (artist, title) if x)
    q = urllib.parse.urlencode({"term": term, "media": "music", "entity": "song", "limit": 25})
    out, seen = [], set()
    for r in _get_json(f"https://itunes.apple.com/search?{q}").get("results", []):
        art = r.get("artworkUrl100") or ""
        key = (r.get("collectionName") or "").casefold()
        if not art or key in seen:
            continue
        seen.add(key)
        out.append({"source": "Apple Music", "thumb": art.replace("100x100bb", "300x300bb"),
                    "full": art.replace("100x100bb", "1200x1200bb"),
                    "label": f"{r.get('collectionName', '')} · {r.get('artistName', '')}",
                    "year": (r.get("releaseDate") or "")[:4]})
        if len(out) >= limit:
            break
    return out


def musicbrainz(artist: str, title: str, album: str = "", limit: int = 6) -> list[dict]:
    global _last_mb
    parts = []
    if title:
        parts.append(f'recording:"{_mb_escape(title)}"')
    if artist:
        parts.append(f'artist:"{_mb_escape(artist)}"')
    if album:
        parts.append(f'release:"{_mb_escape(album)}"')
    wait = 1.1 - (time.monotonic() - _last_mb)              # MusicBrainz: at most one request a second
    if wait > 0:
        time.sleep(wait)
    _last_mb = time.monotonic()
    q = urllib.parse.urlencode({"query": " AND ".join(parts), "fmt": "json", "limit": 25})
    url = f"https://musicbrainz.org/ws/2/recording?{q}"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as e:
        if e.code != 503:                                   # 503 = busy / rate limited: one more try
            raise
        time.sleep(1.5)
        data = _get_json(url)
    _last_mb = time.monotonic()
    out, seen = [], set()
    for rec in data.get("recordings", []):
        for rel in rec.get("releases", []):
            rg = (rel.get("release-group") or {}).get("id")
            if not rg or rg in seen:
                continue
            seen.add(rg)
            base = f"https://coverartarchive.org/release-group/{rg}/front"
            out.append({"source": "MusicBrainz", "thumb": f"{base}-250", "full": f"{base}-1200",
                        "label": f"{rel.get('title', '')} · {(rec.get('artist-credit') or [{}])[0].get('name', '')}",
                        "year": (rel.get("date") or "")[:4]})
            if len(out) >= limit:
                return out
    return out


def _mb_escape(s: str) -> str:
    return "".join("\\" + c if c in '+-&|!(){}[]^"~*?:\\/' else c for c in s)


def search(artist: str, title: str, album: str = "") -> dict:
    """{"results": [...], "errors": [...]}: Apple first (usually the best match), then MusicBrainz."""
    results, errors = [], []
    for name, fn in (("Apple Music", lambda: itunes(artist, title)),
                     ("MusicBrainz", lambda: musicbrainz(artist, title, album))):
        try:
            results += fn()
        except (OSError, ValueError) as e:
            errors.append(f"{name}: {e}")
    return {"results": results, "errors": errors}
