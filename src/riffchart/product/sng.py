"""Single-file song packages (.sng) and Moonscraper.

.sng (github.com/mdsitton/SngFileFormat, version 1), read by Clone Hero v1.1+ and YARG: little-endian
  header      "SNGPKG", uint32 version, 16-byte xor mask
  metadata    uint64 length (bytes after it), uint64 count, then (int32 len, key, int32 len, value) pairs:
              the song.ini fields (song.ini itself is not packed)
  file index  uint64 length, uint64 count, then (uint8 name len, name, uint64 size, uint64 absolute offset)
  file data   uint64 length, then each file masked: byte[i] ^ mask[i % 16] ^ (i & 0xFF), i from the file's start
"""
from __future__ import annotations

import os
import re
import secrets
import struct
import subprocess
from pathlib import Path

import numpy as np

VERSION = 1
SKIP = {"song.ini"}                                  # its fields go in the metadata
KEEP_EXT = {".chart", ".mid", ".ogg", ".opus", ".mp3", ".wav", ".png", ".jpg", ".jpeg", ".webm", ".mp4",
            ".avi", ".ogv", ".mpeg", ".vp8"}
_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _mask(data: bytes, mask: bytes) -> bytes:
    a = np.frombuffer(data, np.uint8)
    i = np.arange(len(a), dtype=np.uint64)
    key = np.frombuffer(mask, np.uint8)[(i % 16).astype(np.int64)] ^ (i & 0xFF).astype(np.uint8)
    return (a ^ key).tobytes()


def song_files(folder: Path) -> list[Path]:
    """What goes in the package: chart, audio, images, video (not song.ini, backups, .riffchart)."""
    out = []
    for f in sorted(Path(folder).iterdir()):
        n = f.name.lower()
        if not f.is_file() or n in SKIP or f.suffix.lower() not in KEEP_EXT or n.startswith("."):
            continue
        if _BAD_NAME.search(f.name) or len(f.name.encode("utf-8")) > 255 or f.name.endswith((" ", ".")):
            continue
        out.append(f)
    return out


def write_sng(folder: Path, out: Path, ini: dict[str, str] | None = None) -> Path:
    """Pack a song folder into one .sng file. ini: the song.ini fields (read from the folder if None)."""
    from ..formats.textio import parse_ini
    folder, out = Path(folder), Path(out)
    if ini is None:
        ini = parse_ini(folder / "song.ini") if (folder / "song.ini").exists() else {}
    files = song_files(folder)
    if not any(f.name.lower() in ("notes.chart", "notes.mid") for f in files):
        raise ValueError("the song has no notes.chart or notes.mid")
    mask = secrets.token_bytes(16)
    meta = b"".join(struct.pack("<i", len(k.encode())) + k.encode() + struct.pack("<i", len(str(v).encode()))
                    + str(v).encode() for k, v in ini.items() if str(v) != "")
    meta_sec = struct.pack("<Q", len(meta) + 8) + struct.pack("<Q", sum(1 for v in ini.values() if str(v) != "")) + meta
    names = [f.name for f in files]
    sizes = [f.stat().st_size for f in files]
    index_len = 8 + sum(1 + len(n.encode()) + 16 for n in names)
    start = 6 + 4 + 16 + len(meta_sec) + 8 + index_len + 8            # first file's absolute offset
    index, off = b"", start
    for n, s in zip(names, sizes):
        index += struct.pack("<B", len(n.encode())) + n.encode() + struct.pack("<QQ", s, off)
        off += s
    tmp = out.with_suffix(out.suffix + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as w:
        w.write(b"SNGPKG" + struct.pack("<I", VERSION) + mask)
        w.write(meta_sec)
        w.write(struct.pack("<QQ", index_len, len(files)) + index)
        w.write(struct.pack("<Q", sum(sizes)))
        for f in files:
            w.write(_mask(f.read_bytes(), mask))
    tmp.replace(out)
    return out


def read_sng(path: Path) -> tuple[dict[str, str], dict[str, bytes]]:
    """(metadata, {file name: contents}) of a .sng file."""
    b = Path(path).read_bytes()
    if b[:6] != b"SNGPKG":
        raise ValueError("not a .sng file")
    mask, p = b[10:26], 26
    mlen, count = struct.unpack_from("<QQ", b, p)
    q, meta = p + 16, {}
    for _ in range(count):
        (kl,) = struct.unpack_from("<i", b, q); k = b[q + 4:q + 4 + kl].decode(); q += 4 + kl
        (vl,) = struct.unpack_from("<i", b, q); v = b[q + 4:q + 4 + vl].decode(); q += 4 + vl
        meta[k] = v
    p += 8 + mlen
    flen, fcount = struct.unpack_from("<QQ", b, p)
    q, files = p + 16, {}
    for _ in range(fcount):
        nl = b[q]; name = b[q + 1:q + 1 + nl].decode(); q += 1 + nl
        size, off = struct.unpack_from("<QQ", b, q); q += 16
        files[name] = _mask(b[off:off + size], mask)
    return meta, files


# ---------------------------------------------------------------- Moonscraper
def moonscraper_exe(path: str | Path | None) -> Path | None:
    """The exe from a path to it or to its folder."""
    if not path:
        return None
    p = Path(str(path).strip().strip('"')).expanduser()
    if p.is_file() and p.suffix.lower() == ".exe":
        return p
    if p.is_dir():
        for name in ("Moonscraper Chart Editor.exe", "MoonscraperChartEditor.exe"):
            if (p / name).is_file():
                return p / name
        hits = sorted(p.glob("Moonscraper*.exe"))
        if hits:
            return hits[0]
    return None


def find_moonscraper(configured: str | None = None) -> Path | None:
    """The configured path, MOONSCRAPER_EXE, then the usual places (one folder deep)."""
    for c in (configured, os.environ.get("MOONSCRAPER_EXE")):
        exe = moonscraper_exe(c)
        if exe:
            return exe
    home = Path.home()
    places = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
              Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
              home / "AppData/Local/Programs", home / "Desktop", home / "Downloads", home / "Documents",
              home / "Games", Path("C:/Games")]
    for base in places:
        if not base.is_dir():
            continue
        exe = moonscraper_exe(base)
        if exe:
            return exe
        try:
            for d in base.iterdir():
                if d.is_dir() and "moonscraper" in d.name.lower():
                    exe = moonscraper_exe(d)
                    if exe:
                        return exe
        except OSError:
            continue
    return None


def open_in_moonscraper(chart: Path, exe: Path | None) -> list[str]:
    """Start Moonscraper with the chart file as its argument."""
    if exe is None:
        raise FileNotFoundError("Moonscraper isn't set up: choose Moonscraper Chart Editor.exe in Settings")
    cmd = [str(exe), str(Path(chart).resolve())]
    subprocess.Popen(cmd, cwd=str(exe.parent))
    return cmd
