"""Highway designs: the user's own highways (colours, an optional pattern image, optional text), saved as named
presets and applied to songs (each song keeps its own copy, so deleting or editing a preset never breaks a
song until it is applied again). Rendering reuses visuals.make_highway, so designs tile like the built-in one.

Presets live in DESIGNS_DIR (data/ui/highways): <id>.json, <id>.pattern.png (if any) and library.json (the
highway new songs get). A song using a design has .riffchart/highway_design.json (+ highway_pattern.png)."""
from __future__ import annotations

import io
import json
import re
import shutil
from pathlib import Path

import numpy as np

from ..paths import project_path

DESIGNS_DIR = project_path("data/ui/highways")
BUILTIN = ("riffchart", "cover", "none")
MAX_PATTERN = 10 << 20
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

DEFAULT = {"name": "My highway", "top": "#16181e", "bottom": "#0e0f13", "rail": "#f5a524", "divider": "#30343e",
           "glow": 50, "rail_width": 14, "lanes": True, "pattern_opacity": 25, "pattern_fit": "mirror",
           "pattern_grey": False, "text": "", "text_color": "#ffffff", "text_opacity": 12, "text_size": 90}


# ---------------------------------------------------------------- validation
def _int(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def clean_design(d: dict) -> dict:
    """A complete, safe design from whatever the page sent (unknown keys dropped, numbers clamped)."""
    if not isinstance(d, dict):
        raise ValueError("a design must be an object")
    out = {}
    name = " ".join(str(d.get("name") or "").split())[:40]
    out["name"] = name or DEFAULT["name"]
    for k in ("top", "bottom", "rail", "divider", "text_color"):
        v = str(d.get(k) or DEFAULT[k])
        if not _HEX.match(v):
            raise ValueError(f"{k} must be a colour like #ff8800")
        out[k] = v.lower()
    out["glow"] = _int(d.get("glow"), 0, 100, DEFAULT["glow"])
    out["rail_width"] = _int(d.get("rail_width"), 0, 40, DEFAULT["rail_width"])
    out["lanes"] = bool(d.get("lanes", True))
    out["pattern_opacity"] = _int(d.get("pattern_opacity"), 0, 100, DEFAULT["pattern_opacity"])
    out["pattern_fit"] = d.get("pattern_fit") if d.get("pattern_fit") in ("mirror", "tile", "stretch") else "mirror"
    out["pattern_grey"] = bool(d.get("pattern_grey", False))
    out["text"] = " ".join(str(d.get("text") or "").split())[:24]
    out["text_opacity"] = _int(d.get("text_opacity"), 0, 100, DEFAULT["text_opacity"])
    out["text_size"] = _int(d.get("text_size"), 20, 200, DEFAULT["text_size"])
    if d.get("id"):
        out["id"] = check_id(str(d["id"]))
    return out


def check_id(i: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", i):
        raise ValueError(f"bad highway id {i!r}")
    return i


def _rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def clean_pattern(data: bytes) -> bytes:
    """Any PNG / JPEG / WebP / GIF frame, as an RGB PNG at most 1024 px on its long side."""
    from PIL import Image
    if len(data) > MAX_PATTERN:
        raise ValueError("the pattern image is too big (10 MB max)")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        raise ValueError("the pattern must be an image (PNG, JPEG, WebP or GIF)")
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    im = im.convert("RGB")
    im.thumbnail((1024, 1024))
    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    return out.getvalue()


# ---------------------------------------------------------------- rendering
def _texture(pattern: bytes, fit: str, grey: bool) -> np.ndarray:
    """HxWx3 float texture from the pattern that tiles top to bottom (mirror), repeats it (tile: seamless only
    if the image is), or stretches it over the tile (stretch)."""
    from PIL import Image
    from .visuals import HW_SIZE
    W, H = HW_SIZE
    im = Image.open(io.BytesIO(pattern)).convert("L" if grey else "RGB")
    tex = Image.new(im.mode, (W, H))
    if fit == "stretch":
        tex = im.resize((W, H), Image.LANCZOS)
    elif fit == "tile":
        h = max(16, round(im.height * W / im.width))
        part = im.resize((W, h), Image.LANCZOS)
        for y in range(0, H, h):
            tex.paste(part, (0, y))
    else:
        half = im.resize((W, H // 2), Image.LANCZOS)
        tex.paste(half, (0, 0))
        tex.paste(half.transpose(Image.FLIP_TOP_BOTTOM), (0, H // 2))
    a = np.asarray(tex.convert("RGB"), np.float32)
    return a


def _text_overlay(text: str, color: str, opacity: int, size: int):
    """The text twice per tile (so it scrolls past every half tile), centred, upright as the player reads it."""
    from PIL import Image, ImageDraw, ImageFont
    from .visuals import HW_SIZE
    W, H = HW_SIZE
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not text or opacity <= 0:
        return None
    try:
        font = ImageFont.load_default(size)
    except TypeError:                                    # very old Pillow
        font = ImageFont.load_default()
    d = ImageDraw.Draw(ov)
    while size > 20 and d.textlength(text, font=font) > W * 0.8:
        size -= 6
        font = ImageFont.load_default(size)
    fill = _rgb(color) + (round(255 * opacity / 100),)
    for cy in (H // 4, 3 * H // 4):
        d.text((W / 2, cy), text, font=font, fill=fill, anchor="mm")
    return ov


def render(design: dict, pattern: bytes | None = None) -> bytes:
    """PNG bytes of a design (512x1024, tiles vertically)."""
    from .visuals import _png, make_highway
    d = clean_design(design)
    rail = _rgb(d["rail"])
    glow = tuple(int(c * 0.9 * d["glow"] / 100) for c in rail) if d["glow"] and d["rail_width"] else None
    tex = _texture(pattern, d["pattern_fit"], d["pattern_grey"]) if pattern else None
    ov = _text_overlay(d["text"], d["text_color"], d["text_opacity"], d["text_size"])
    im = make_highway(_rgb(d["top"]), _rgb(d["bottom"]), rail, _rgb(d["divider"]), glow, tex,
                      d["pattern_opacity"] / 100, d["rail_width"], d["lanes"], ov)
    return _png(im)


# ---------------------------------------------------------------- presets
def _dir() -> Path:
    return Path(DESIGNS_DIR)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "highway"
    return s if s[0].isalnum() else "h" + s


def list_designs() -> list[dict]:
    out = []
    if _dir().exists():
        for p in sorted(_dir().glob("*.json")):
            if p.name == "library.json":
                continue
            try:
                d = clean_design(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
            d["id"] = p.stem
            d["has_pattern"] = (_dir() / f"{p.stem}.pattern.png").exists()
            d["mtime"] = int(p.stat().st_mtime)
            out.append(d)
    return sorted(out, key=lambda d: d["name"].lower())


def get_design(i: str) -> tuple[dict, bytes | None]:
    p = _dir() / f"{check_id(i)}.json"
    if not p.exists():
        raise KeyError(i)
    d = clean_design(json.loads(p.read_text(encoding="utf-8")))
    d["id"] = i
    pat = _dir() / f"{i}.pattern.png"
    return d, (pat.read_bytes() if pat.exists() else None)


def save_design(design: dict, pattern: bytes | None = None, keep_pattern: bool = True) -> dict:
    """Create (no id) or update a preset. pattern: new image bytes; keep_pattern=False removes the old one."""
    d = clean_design(design)
    _dir().mkdir(parents=True, exist_ok=True)
    i = d.pop("id", None)
    if i is None or not (_dir() / f"{i}.json").exists():
        base = i or _slug(d["name"])
        i, n = base, 2
        while (_dir() / f"{i}.json").exists():
            i, n = f"{base}-{n}", n + 1
    pat = _dir() / f"{i}.pattern.png"
    if pattern is not None:
        pat.write_bytes(clean_pattern(pattern))
    elif not keep_pattern:
        pat.unlink(missing_ok=True)
    (_dir() / f"{i}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    (_dir() / f"{i}.preview.png").write_bytes(render(d, pat.read_bytes() if pat.exists() else None))
    out, _ = get_design(i)
    out["has_pattern"] = pat.exists()
    return out


def delete_design(i: str) -> None:
    check_id(i)
    for suffix in (".json", ".pattern.png", ".preview.png"):
        (_dir() / f"{i}{suffix}").unlink(missing_ok=True)
    if library_default() == f"design:{i}":
        set_library_default("riffchart")


def preview_path(i: str) -> Path:
    """The saved preset's rendered image (made again if missing)."""
    p = _dir() / f"{check_id(i)}.preview.png"
    if not p.exists():
        d, pat = get_design(i)
        p.write_bytes(render(d, pat))
    return p


# ---------------------------------------------------------------- library default (the highway new songs get)
def library_default() -> str | None:
    """riffchart / cover / none / design:<id> chosen in the app, or None (then configs/product.yaml decides)."""
    try:
        v = json.loads((_dir() / "library.json").read_text(encoding="utf-8")).get("default")
    except (OSError, ValueError, AttributeError):
        return None
    if v in BUILTIN:
        return v
    if isinstance(v, str) and v.startswith("design:") and (_dir() / f"{v[7:]}.json").exists():
        return v
    return None


def set_library_default(v: str) -> None:
    if v not in BUILTIN:
        if not v.startswith("design:"):
            raise ValueError("default must be riffchart, cover, none or design:<id>")
        get_design(v[7:])                                # must exist
    _dir().mkdir(parents=True, exist_ok=True)
    (_dir() / "library.json").write_text(json.dumps({"default": v}), encoding="utf-8")


# ---------------------------------------------------------------- per song
def _song_files(folder: Path) -> tuple[Path, Path]:
    s = folder / ".riffchart"
    return s / "highway_design.json", s / "highway_pattern.png"


def song_design(folder: Path) -> tuple[dict, bytes | None] | None:
    dj, dp = _song_files(folder)
    try:
        d = clean_design(json.loads(dj.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    return d, (dp.read_bytes() if dp.exists() else None)


def copy_to_song(folder: Path, i: str) -> dict:
    """Give the song its own copy of preset i (used by refresh_highway in "design" mode)."""
    d, pat = get_design(i)
    dj, dp = _song_files(folder)
    dj.parent.mkdir(parents=True, exist_ok=True)
    if pat:
        dp.write_bytes(pat)
    else:
        dp.unlink(missing_ok=True)
    dj.write_text(json.dumps(d), encoding="utf-8")
    return d


def songs_using(workspace: Path, i: str) -> list[Path]:
    out = []
    for f in sorted(Path(workspace).iterdir()):
        dj, _ = _song_files(f)
        if dj.exists():
            try:
                if json.loads(dj.read_text(encoding="utf-8")).get("id") == i:
                    out.append(f)
            except (OSError, ValueError):
                pass
    return out


def install_in_clone_hero(i: str, highways_dir: Path) -> Path:
    """Copy the preset's image into Clone Hero's Custom/Highways folder, where players pick it for every song."""
    d, _ = get_design(i)
    src = preview_path(i)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", d["name"]).strip(" .") or i
    highways_dir.mkdir(parents=True, exist_ok=True)
    dest = highways_dir / f"RiffChart - {name}.png"
    shutil.copyfile(src, dest)
    return dest
