"""Text decoding and song.ini parsing."""
from __future__ import annotations

import re
from pathlib import Path


def decode_text(data: bytes) -> str:
    """Decode chart/ini bytes: honour BOMs, then UTF-8, then cp1252/latin-1 fallback."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("cp1252")
        except UnicodeDecodeError:
            return data.decode("latin-1")


_INI_LINE = re.compile(r"^\s*([^=\[;#]+?)\s*=\s*(.*?)\s*$")


def parse_ini(path: str | Path) -> dict[str, str]:
    """Lowercased key -> raw string value. Section headers are ignored (song.ini has one section)."""
    out: dict[str, str] = {}
    try:
        text = decode_text(Path(path).read_bytes())
    except OSError:
        return out
    for line in text.splitlines():
        m = _INI_LINE.match(line)
        if m:
            out.setdefault(m.group(1).strip().lower(), m.group(2))
    return out


def ini_float(ini: dict[str, str], key: str, default: float | None = None) -> float | None:
    try:
        return float(ini[key].strip())
    except (KeyError, ValueError, AttributeError):
        return default


def song_delay_s(ini: dict[str, str]) -> float:
    """song.ini delay (ms), falling back to delay_seconds (YARG SongMetadata.cs)."""
    d = ini_float(ini, "delay", 0.0) or 0.0
    if d != 0:
        return d / 1000.0
    return ini_float(ini, "delay_seconds", 0.0) or 0.0
