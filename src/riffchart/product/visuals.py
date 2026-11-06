"""Song visuals in the Clone Hero / YARG folder: cover (album.png / .jpg), background image (background.jpg /
.png, shown behind the highway), background video (video.mp4 / .webm) and highway image (highway.png).

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


def _state(folder: Path) -> dict:
    try:
        st = json.loads(_state_path(folder).read_text(encoding="utf-8"))
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def _set_state(folder: Path, key: str, value: str) -> None:
    p = _state_path(folder)
    p.parent.mkdir(parents=True, exist_ok=True)
    st = _state(folder)
    st[key] = value
    p.write_text(json.dumps(st), encoding="utf-8")


def bg_mode(folder: Path) -> str:
    """auto / own / none. Without a saved choice: an existing background is the user's ("own"), else auto."""
    m = _state(folder).get("background")
    if m in MODES:
        return m
    return "own" if find_first(folder, BG_FILES) else "auto"


def set_bg_mode(folder: Path, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"background must be one of {', '.join(MODES)}")
    _set_state(folder, "background", mode)


def _remove_bg(folder: Path) -> None:
    for n in BG_FILES:
        (folder / n).unlink(missing_ok=True)


def refresh_background(folder: Path) -> None:
    """Bring the background in line with the mode: auto follows the cover (none without a cover)."""
    mode = bg_mode(folder)
    if "background" not in _state(folder):
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


# ---------------------------------------------------------------- highway (Clone Hero: "Song Highways" setting)
HW_FILES = ("highway.png", "highway.jpg")
HW_SIZE = (512, 1024)                           # Clone Hero's recommended size; it tiles down the highway
HW_MODES = ("riffchart", "cover", "design", "own", "none")
RIFFCHART_COLORS = {"top": (22, 24, 30), "bottom": (14, 15, 19), "rail": (245, 165, 36), "divider": (48, 52, 62),
                    "glow": (120, 75, 10)}


def default_highway() -> str:
    """Library-wide default for songs without a choice: the one picked in the app's highway designer
    ("riffchart", "cover", "none" or "design:<id>"), else `highway:` in configs/product.yaml."""
    from .highways import library_default
    v = library_default()
    if v:
        return v
    try:
        from ..paths import load_config
        m = load_config("configs/product.yaml").get("highway", "riffchart")
    except Exception:
        m = "riffchart"
    return m if m in ("riffchart", "cover", "none") else "riffchart"


def make_highway(top, bottom, rail, divider, glow=None, texture=None, texture_alpha=0.22, rail_w=14,
                 lanes=True, overlay=None) -> "Image.Image":
    """A 512x1024 highway that tiles vertically: a dark surface (gradient that wraps), five lanes, coloured
    rails with a soft inner glow, optionally a faint texture (HxWx3, mixed in at texture_alpha) and an RGBA
    overlay (text / logo) under the lane lines. Notes and beat lines are drawn by the game."""
    from PIL import Image, ImageDraw, ImageFilter
    W, H = HW_SIZE
    t = (1 - np.cos(np.linspace(0, 2 * np.pi, H, endpoint=False))) / 2          # 0 -> 1 -> 0: seamless
    a, b = np.array(top, np.float32), np.array(bottom, np.float32)
    px = np.repeat((a[None] * (1 - t[:, None]) + b[None] * t[:, None])[:, None, :], W, axis=1)
    if texture is not None and texture_alpha > 0:
        px = px * (1 - texture_alpha) + np.asarray(texture, np.float32) * texture_alpha
    im = Image.fromarray(px.clip(0, 255).astype(np.uint8))
    if overlay is not None:
        im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    if glow:
        g = Image.new("RGB", (W, H), (0, 0, 0))
        gd = ImageDraw.Draw(g)
        gd.rectangle((0, 0, rail_w + 18, H), fill=tuple(glow))
        gd.rectangle((W - rail_w - 18, 0, W, H), fill=tuple(glow))
        # blur a 3-tile strip and keep the middle, so the glow is seamless top to bottom too
        g3 = Image.new("RGB", (W, 3 * H))
        for k in range(3):
            g3.paste(g, (0, k * H))
        g = g3.filter(ImageFilter.GaussianBlur(14)).crop((0, H, W, 2 * H))
        im = Image.fromarray(np.clip(np.asarray(im, np.int16) + np.asarray(g, np.int16) // 2, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    lane = (W - 2 * rail_w) / 5
    if lanes:
        for i in range(1, 5):
            xx = round(rail_w + i * lane)
            d.line((xx, 0, xx, H), fill=tuple(divider), width=2)
    if rail_w > 0:
        rail = tuple(rail)
        light = tuple(min(255, c + 60) for c in rail)
        d.rectangle((0, 0, rail_w - 1, H), fill=rail)
        d.rectangle((W - rail_w, 0, W, H), fill=rail)
        d.line((rail_w, 0, rail_w, H), fill=light, width=1)
        d.line((W - rail_w - 1, 0, W - rail_w - 1, H), fill=light, width=1)
    return im


def riffchart_highway() -> bytes:
    c = RIFFCHART_COLORS
    return _png(make_highway(c["top"], c["bottom"], c["rail"], c["divider"], c["glow"]))


def highway_from_cover(data: bytes) -> bytes:
    """Rails in the cover's most vivid colour, a surface tinted with its main colour, and a faint mirrored copy
    of the cover running down the highway."""
    import colorsys
    from PIL import Image, ImageFilter
    cover = Image.open(io.BytesIO(data)).convert("RGB")
    q = cover.resize((96, 96)).quantize(8, method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[:24], np.float32).reshape(-1, 3)
    counts = np.bincount(np.asarray(q).ravel(), minlength=len(pal))[:len(pal)]
    hsv = [colorsys.rgb_to_hsv(*(c / 255)) for c in pal]
    common = [i for i in range(len(pal)) if counts[i] > 0.04 * counts.sum()] or list(range(len(pal)))
    vivid = max(common, key=lambda i: hsv[i][1] * hsv[i][2])
    dom = max(range(len(pal)), key=lambda i: counts[i])
    h, s, v = hsv[vivid]
    rail = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, max(s, 0.55), max(v, 0.85)))
    hd, sd, _ = hsv[dom]
    top = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hd, min(sd, 0.5), 0.13))
    bottom = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hd, min(sd, 0.5), 0.07))
    W, H = HW_SIZE
    half = cover.convert("L").resize((W, H // 2)).filter(ImageFilter.GaussianBlur(2))
    tex = Image.new("L", (W, H))
    tex.paste(half, (0, 0))
    tex.paste(half.transpose(Image.FLIP_TOP_BOTTOM), (0, H // 2))             # mirrored so it tiles
    a = np.asarray(tex, np.float32)
    a = (a - a.min()) / (np.ptp(a) + 1e-6) * 60
    divider = tuple(min(255, int(c * 0.35) + 30) for c in rail)
    glow = tuple(int(c * 0.45) for c in rail)
    return _png(make_highway(top, bottom, rail, divider, glow, np.repeat(a[..., None], 3, axis=2)))


def _png(im) -> bytes:
    out = io.BytesIO()
    im.save(out, "PNG", optimize=True)
    return out.getvalue()


def hw_mode(folder: Path) -> str:
    """riffchart / cover / own / none. Without a saved choice: an existing highway is the user's, else the
    library default."""
    m = _state(folder).get("highway")
    if m in HW_MODES:
        return m
    return "own" if find_first(folder, HW_FILES) else default_highway().split(":")[0]


def set_hw_mode(folder: Path, mode: str, design: str | None = None) -> None:
    """design: the preset id to copy into the song (mode "design")."""
    if mode not in HW_MODES:
        raise ValueError(f"highway must be one of {', '.join(HW_MODES)}")
    if mode == "design" and design:
        from .highways import copy_to_song
        copy_to_song(folder, design)
    _set_state(folder, "highway", mode)


def _remove_hw(folder: Path) -> None:
    for n in HW_FILES:
        (folder / n).unlink(missing_ok=True)


def refresh_highway(folder: Path) -> None:
    """Make highway.png for the mode: the RiffChart highway, one from the cover (the RiffChart one when there's
    no cover), the user's own (left alone) or none."""
    from . import highways
    mode = hw_mode(folder)
    if "highway" not in _state(folder):
        set_hw_mode(folder, mode)
        dflt = default_highway()
        if mode == "design" and dflt.startswith("design:"):
            try:                                        # a new song gets the library's design
                highways.copy_to_song(folder, dflt[7:])
            except (KeyError, ValueError, OSError):
                pass
    if mode == "own":
        return
    if mode == "none":
        _remove_hw(folder)
        return
    data = None
    if mode == "design":
        sd = highways.song_design(folder)
        if sd:
            try:
                data = highways.render(*sd)
            except Exception:                           # broken pattern image: the RiffChart one
                data = None
    if data is None:
        art = find_art(folder) if mode == "cover" else None
        try:
            data = highway_from_cover(art.read_bytes()) if art else riffchart_highway()
        except Exception:                               # unreadable cover
            data = riffchart_highway()
    tmp = folder / "highway.png.tmp"
    tmp.write_bytes(data)
    _remove_hw(folder)
    tmp.replace(folder / "highway.png")


def write_own_highway(folder: Path, data: bytes) -> str:
    ext = image_ext(data)
    if ext is None:
        raise ValueError("the highway must be a PNG or JPEG image")
    if len(data) > (10 << 20):
        raise ValueError("the highway image is too big (10 MB max)")
    name = f"highway.{ext}"
    tmp = folder / f"{name}.tmp"
    tmp.write_bytes(data)
    _remove_hw(folder)
    tmp.replace(folder / name)
    set_hw_mode(folder, "own")
    return name


def refresh_visuals(folder: Path) -> None:
    """Background and highway after the cover or names changed."""
    refresh_background(folder)
    refresh_highway(folder)


# ---------------------------------------------------------------- background video
# what Clone Hero plays (its docs: mp4 H.264, webm VP8, ogv Theora, mpeg; only VP8 webm on every platform).
# VP9 / AV1 (most YouTube downloads) and HEVC are converted to VP8 webm.
CH_VIDEO_OK = {"h264", "vp8", "theora", "mpeg1video", "mpeg2video"}
_vinfo_cache: dict[tuple, dict] = {}
_DECODERS = {"libdav1d": "av1", "libaom-av1": "av1", "libvpx": "vp8", "libvpx-vp9": "vp9"}   # decoder -> format
CONVERTING: dict[str, dict] = {}          # song folder -> {progress, error} while / after converting


def video_info(folder: Path) -> dict | None:
    """{file, mb, codec, width, height, ok} of the song's video (ok: Clone Hero plays it), or None."""
    vid = find_first(folder, VIDEO_FILES)
    if vid is None:
        return None
    st = vid.stat()
    key = (str(vid), st.st_size, int(st.st_mtime))
    if key not in _vinfo_cache:
        info = {"codec": "", "width": 0, "height": 0}
        try:
            import av
            with av.open(str(vid)) as c:
                v = c.streams.video[0]
                name = v.codec_context.name
                info = {"codec": _DECODERS.get(name, name), "width": v.width, "height": v.height}
        except Exception:
            pass
        _vinfo_cache[key] = info
    d = dict(_vinfo_cache[key], file=vid.name, mb=round(st.st_size / 2**20, 1))
    d["ok"] = d["codec"] in CH_VIDEO_OK
    conv = CONVERTING.get(str(folder))
    if conv:
        d["converting"] = dict(conv)
    return d


def convert_video(folder: Path, progress=None, max_height: int = 720) -> dict:
    """Re-encode the song's video as VP8 webm (plays in Clone Hero on every platform and in YARG), at most
    max_height lines, same frame rate (max 60), starting at 0 (video_start_time still applies). About 10-40x
    real time on a desktop CPU. progress(fraction) is called now and then. Returns video_info."""
    import av
    src = find_first(folder, VIDEO_FILES)
    if src is None:
        raise FileNotFoundError("this song has no video")
    tmp = folder / "video.webm.tmp"
    with av.open(str(src)) as c:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        rate = v.average_rate or v.guessed_rate or 30
        if rate > 60:
            rate = 60
        dur = float(c.duration / 1e6) if c.duration else float(v.duration * v.time_base) if v.duration else 0.0
        w, h = v.width, v.height
        if h > max_height:
            w, h = round(w * max_height / h / 2) * 2, max_height
        w, h = w - w % 2, h - h % 2
        with av.open(str(tmp), "w", format="webm") as o:
            s = o.add_stream("libvpx", rate=rate)
            s.width, s.height, s.pix_fmt = w, h, "yuv420p"
            s.options = {"deadline": "good", "cpu-used": "5", "crf": "20", "b": "2M" if h > 480 else "1200k",
                         "auto-alt-ref": "0"}
            s.thread_count = 0
            t0, last, n = None, -1.0, 0
            for fr in c.decode(v):
                if fr.time is None:
                    continue
                t0 = fr.time if t0 is None else t0
                t = fr.time - t0
                if t < last + 0.5 / float(rate):              # frame rate capped: drop the extra frames
                    continue
                last = t
                out = fr.reformat(width=w, height=h, format="yuv420p")
                out.pts = round(t * float(rate))
                out.time_base = s.codec_context.time_base
                for p in s.encode(out):
                    o.mux(p)
                n += 1
                if progress and dur and n % 60 == 0:
                    progress(min(0.99, t / dur))
            for p in s.encode():
                o.mux(p)
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError("no picture could be read from the video")
    for f in VIDEO_FILES:
        (folder / f).unlink(missing_ok=True)
    tmp.replace(folder / "video.webm")
    if progress:
        progress(1.0)
    return video_info(folder)


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
        codec = _DECODERS.get(v.codec_context.name, v.codec_context.name)
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
