"""Song visuals in the Clone Hero / YARG folder: cover (album.png / .jpg), background image (background.jpg /
.png, shown behind the highway) and background video (video.mp4 / .webm).

The background can follow the cover ("auto": the cover blurred and darkened to 1920x1080), be the user's own
image ("own"), or be off ("none"). The choice is kept in .riffchart/visuals.json (never exported)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

ART_FILES = ("album.png", "album.jpg")
BG_FILES = ("background.jpg", "background.png")
VIDEO_FILES = ("video.mp4", "video.webm")
BG_SIZE = (1920, 1080)
MODES = ("auto", "own", "none")
# a cover embedded in an audio file shows up as a one-frame "video" stream: not a music video
_STILL_CODECS = {"mjpeg", "png", "bmp", "gif", "tiff", "webp"}


def image_ext(data: bytes) -> str | None:
    """'png' / 'jpg' from the file's first bytes, else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return None


def find_first(folder: Path, names) -> Path | None:
    return next((folder / n for n in names if (folder / n).is_file()), None)


def find_art(folder: Path) -> Path | None:
    return find_first(folder, ART_FILES)


# ---------------------------------------------------------------- background image
def background_from_cover(data: bytes, size: tuple[int, int] = BG_SIZE) -> bytes:
    """A 16:9 JPEG for behind the highway: the cover filling the frame, heavily blurred, darkened, with a soft
    vignette, so notes stay readable on any cover."""
    from PIL import Image, ImageEnhance, ImageFilter
    W, H = size
    im = Image.open(io.BytesIO(data)).convert("RGB")
    s = max(W / im.width, H / im.height)
    im = im.resize((max(W, round(im.width * s)), max(H, round(im.height * s))), Image.LANCZOS)
    left, top = (im.width - W) // 2, (im.height - H) // 2
    im = im.crop((left, top, left + W, top + H))
    # blur at 1/8 size (fast, and a much wider blur than GaussianBlur alone would manage)
    small = im.resize((W // 8, H // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5))
    im = small.resize((W, H), Image.BICUBIC)
    im = ImageEnhance.Color(im).enhance(1.15)
    im = ImageEnhance.Brightness(im).enhance(0.5)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    shade = np.clip(1.0 - 0.45 * np.clip(r - 0.35, 0, None) ** 1.5, 0.35, 1.0)
    px = (np.asarray(im, np.float32) * shade[..., None]).clip(0, 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(px).save(out, "JPEG", quality=88, optimize=True)
    return out.getvalue()


def _state_path(folder: Path) -> Path:
    return folder / ".riffchart" / "visuals.json"


def bg_mode(folder: Path) -> str:
    """auto / own / none. Without a saved choice: an existing background is the user's ("own"), else auto."""
    try:
        m = json.loads(_state_path(folder).read_text(encoding="utf-8")).get("background")
        if m in MODES:
            return m
    except (OSError, ValueError):
        pass
    return "own" if find_first(folder, BG_FILES) else "auto"


def set_bg_mode(folder: Path, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"background must be one of {', '.join(MODES)}")
    p = _state_path(folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    st["background"] = mode
    p.write_text(json.dumps(st), encoding="utf-8")


def _remove_bg(folder: Path) -> None:
    for n in BG_FILES:
        (folder / n).unlink(missing_ok=True)


def refresh_background(folder: Path) -> None:
    """Bring the background in line with the mode: auto follows the cover (none without a cover)."""
    mode = bg_mode(folder)
    if not _state_path(folder).exists():
        set_bg_mode(folder, mode)               # remember it, or the background made here would look like "own"
    if mode == "none":
        _remove_bg(folder)
    elif mode == "auto":
        art = find_art(folder)
        if art is None:
            _remove_bg(folder)
            return
        try:
            data = background_from_cover(art.read_bytes())
        except Exception:                       # unreadable cover: leave the background as it is
            return
        tmp = folder / "background.jpg.tmp"
        tmp.write_bytes(data)
        _remove_bg(folder)
        tmp.replace(folder / "background.jpg")


def write_own_background(folder: Path, data: bytes) -> str:
    ext = image_ext(data)
    if ext is None:
        raise ValueError("the background must be a PNG or JPEG image")
    if len(data) > (20 << 20):
        raise ValueError("the background is too big (20 MB max)")
    name = f"background.{ext}"
    tmp = folder / f"{name}.tmp"
    tmp.write_bytes(data)
    _remove_bg(folder)
    tmp.replace(folder / name)
    set_bg_mode(folder, "own")
    return name


# ---------------------------------------------------------------- background video
def save_video(src: str | Path, folder: Path) -> dict | None:
    """Copy the music video's picture (no sound, no re-encoding) from a video file into the song folder as
    video.mp4 (H.264 / HEVC / ...) or video.webm (VP8 / VP9 / AV1). Returns {file, codec, start_ms} or None
    when the source has no real video stream. start_ms: where in the video the song's audio starts."""
    import av
    src = Path(src)
    try:
        c = av.open(str(src))
    except Exception:
        return None
    with c:
        vids = [s for s in c.streams.video if s.codec_context.name not in _STILL_CODECS
                and not (s.disposition & av.stream.Disposition.attached_pic)]
        if not vids:
            return None
        v = vids[0]
        codec = v.codec_context.name
        ext = "webm" if codec in ("vp8", "vp9", "av1") else "mp4"
        a = c.streams.audio[0] if c.streams.audio else None
        a0 = float(a.start_time * a.time_base) if a is not None and a.start_time is not None else 0.0
        v0 = float(v.start_time * v.time_base) if v.start_time is not None else 0.0
        tmp = folder / f"video.{ext}.tmp"
        with av.open(str(tmp), "w", format="webm" if ext == "webm" else "mp4") as out:
            o = out.add_stream_from_template(v)
            for pkt in c.demux(v):
                if pkt.dts is None:
                    continue
                pkt.stream = o
                out.mux(pkt)
    for n in VIDEO_FILES:
        (folder / n).unlink(missing_ok=True)
    tmp.replace(folder / f"video.{ext}")
    return {"file": f"video.{ext}", "codec": codec, "start_ms": int(round((a0 - v0) * 1000))}
