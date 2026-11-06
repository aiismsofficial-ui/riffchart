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
def background_from_cover(data: bytes, title: str = "", artist: str = "",
                          size: tuple[int, int] = BG_SIZE) -> bytes:
    """A 16:9 JPEG for behind the highway. The highway covers the middle third, so: a blurred, darkened copy of
    the cover fills the frame, the sharp cover sits on the left and the title / artist on the right."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
    W, H = size
    cover = Image.open(io.BytesIO(data)).convert("RGB")
    s = max(W / cover.width, H / cover.height)
    im = cover.resize((max(W, round(cover.width * s)), max(H, round(cover.height * s))), Image.LANCZOS)
    left, top = (im.width - W) // 2, (im.height - H) // 2
    im = im.crop((left, top, left + W, top + H))
    # blur at 1/8 size (fast, and a much wider blur than GaussianBlur alone would manage)
    small = im.resize((W // 8, H // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
    im = small.resize((W, H), Image.BICUBIC)
    im = ImageEnhance.Color(im).enhance(1.2)
    im = ImageEnhance.Brightness(im).enhance(0.42)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    shade = np.clip(1.0 - 0.45 * np.clip(r - 0.35, 0, None) ** 1.5, 0.35, 1.0)
    im = Image.fromarray((np.asarray(im, np.float32) * shade[..., None]).clip(0, 255).astype(np.uint8))

    # the sharp cover, left of the highway, with a soft shadow
    side = round(H * 0.44)                                  # 475 px at 1080p
    cx, cy = round(W * 0.15), H // 2                       # centre of the left panel
    x0, y0 = cx - side // 2, cy - side // 2
    sharp = cover.resize((side, side), Image.LANCZOS) if cover.width == cover.height else \
        _square(cover).resize((side, side), Image.LANCZOS)
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).rectangle((x0 + 10, y0 + 16, x0 + side + 10, y0 + side + 16), fill=170)
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    im = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), im, shadow)
    im.paste(sharp, (x0, y0))
    ImageDraw.Draw(im).rectangle((x0 - 1, y0 - 1, x0 + side, y0 + side), outline=(255, 255, 255), width=1)

    # title / artist, right of the highway
    if title or artist:
        d = ImageDraw.Draw(im)
        tx, maxw = round(W * 0.72), round(W * 0.255)
        tfont = ImageFont.load_default(size=round(H * 0.052))
        afont = ImageFont.load_default(size=round(H * 0.034))
        lines = _wrap(d, title, tfont, maxw, 3)
        lh_t, lh_a = round(H * 0.064), round(H * 0.046)
        total = len(lines) * lh_t + (lh_a + 14 if artist else 0)
        ty = cy - total // 2
        for ln in lines:
            d.text((tx, ty), ln, font=tfont, fill=(255, 255, 255), stroke_width=1, stroke_fill=(255, 255, 255))
            ty += lh_t
        if artist:
            d.text((tx, ty + 14), _wrap(d, artist, afont, maxw, 1)[0], font=afont, fill=(205, 210, 220))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=90, optimize=True)
    return out.getvalue()


def _square(im):
    s = min(im.width, im.height)
    return im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width - s) // 2 + s, (im.height - s) // 2 + s))


def _wrap(draw, text: str, font, maxw: int, max_lines: int) -> list[str]:
    """Word-wrap to maxw pixels, at most max_lines (the last one ends in … if cut)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    lines = lines or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += "…"
    for i, ln in enumerate(lines):                         # one long word: cut it
        while draw.textlength(ln, font=font) > maxw and len(ln) > 2:
            ln = ln[:-2] + "…"
        lines[i] = ln
    return lines


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
        from ..formats.textio import parse_ini
        ini = parse_ini(folder / "song.ini") if (folder / "song.ini").exists() else {}
        try:
            data = background_from_cover(art.read_bytes(), ini.get("name", ""), ini.get("artist", ""))
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
