"""Animated backgrounds: the full-screen video behind the highway (where a music video plays).

A preset is an effect plus colours. It is drawn over a base picture: the song's own background (the one made
from its cover, with the title) or, for the generic version installed in Clone Hero, a plain gradient. The
loop is rendered once (LOOP_S seconds, 1280x720, 24 fps, VP8 .webm, the format Clone Hero plays everywhere) and
then repeated to the song's length by copying the encoded packets with shifted timestamps, which takes a
moment and needs no re-encoding. So it works whether or not Clone Hero loops a song's video.

Every effect is periodic in t (one loop), so the repeats join without a jump.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .highway_render import _hue_shift, _rgb

SIZE = (1280, 720)
FPS = 24
LOOP_S = 8

PRESETS = {
    "embers":    {"name": "Embers", "effect": "embers", "color": "#ff7a1a", "color2": "#ffd166"},
    "bokeh":     {"name": "Bokeh lights", "effect": "bokeh", "color": "#ffc46b", "color2": "#ff5fa2"},
    "aurora":    {"name": "Aurora", "effect": "aurora", "color": "#38ff9c", "color2": "#4d7cff"},
    "starfield": {"name": "Starfield", "effect": "starfield", "color": "#cfe0ff", "color2": "#8fb3ff"},
    "beams":     {"name": "Stage lights", "effect": "beams", "color": "#ffffff", "color2": "#5ab0ff"},
    "plasma":    {"name": "Plasma", "effect": "plasma", "color": "#ff3ec8", "color2": "#22e4ff"},
    "rain":      {"name": "Rain", "effect": "rain", "color": "#9ad7ff", "color2": "#ffffff"},
    "storm":     {"name": "Thunderstorm", "effect": "storm", "color": "#cfe6ff", "color2": "#7a8cff"},
    "synthwave": {"name": "Synthwave sunset", "effect": "synthwave", "color": "#ff3ec8", "color2": "#ffb347"},
    "smoke":     {"name": "Smoke", "effect": "smoke", "color": "#b8c0d0", "color2": "#6a5acd"},
    "equalizer": {"name": "Equalizer", "effect": "equalizer", "color": "#22e4ff", "color2": "#ff3ec8"},
}


def base_from_image(data: bytes | None, size=SIZE, dim: float = 0.8) -> np.ndarray:
    """The picture under the effect: the song's background (fitted, slightly dimmed so effects show), or a
    dark gradient."""
    from PIL import Image, ImageOps
    W, H = size
    if data:
        im = ImageOps.fit(Image.open(io.BytesIO(data)).convert("RGB"), (W, H), Image.LANCZOS)
        return np.asarray(im, np.float32) * dim
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top, bot = np.array([18, 16, 28], np.float32), np.array([6, 6, 10], np.float32)
    return np.broadcast_to(top * (1 - y) + bot * y, (H, W, 3)).copy()


class Animator:
    def __init__(self, preset: dict, base: np.ndarray, fps: int = FPS, seconds: float = LOOP_S,
                 strength: float = 0.8):
        self.p, self.base, self.fps = preset, base, fps
        self.H, self.W = base.shape[:2]
        self.n = int(round(seconds * fps))
        self.amp = strength
        self.c1, self.c2 = _rgb(preset["color"]) / 255, _rgb(preset["color2"]) / 255
        self.rng = np.random.default_rng(7)
        W, H, r = self.W, self.H, self.rng
        self.x = np.arange(W, dtype=np.float32)[None, :]
        self.y = np.arange(H, dtype=np.float32)[:, None]
        self.s = W / 1280
        e = preset["effect"]
        if e == "embers":
            m = 160
            self.px, self.py = r.uniform(0, W, m), r.uniform(0, H, m)
            self.pv = r.integers(1, 3, m)                       # screens per loop: whole numbers = seamless
            self.pw, self.pph = r.uniform(10, 40, m) * self.s, r.uniform(0, 2 * np.pi, m)
            self.pa, self.pr = r.uniform(0.3, 1, m), r.uniform(1.2, 3.2, m) * self.s
            self.pc = r.random(m)
        elif e == "bokeh":
            m = 38
            self.px, self.py = r.uniform(0, W, m), r.uniform(0, H, m)
            self.pr = r.uniform(18, 70, m) * self.s
            self.pdx, self.pdy = r.integers(-1, 2, m), r.integers(-1, 1, m)  # whole screens per loop (or still)
            self.pdy[self.pdy == 0] = -1
            self.pa, self.pph, self.pc = r.uniform(0.25, 0.8, m), r.uniform(0, 2 * np.pi, m), r.random(m)
        elif e == "starfield":
            m = 420
            self.sa = r.uniform(0, 2 * np.pi, m)
            self.sz0 = r.uniform(0, 1, m)
            self.sb = r.uniform(0.4, 1, m)
        elif e == "rain":
            m = 340
            self.px, self.py = r.uniform(0, W * 1.2, m), r.uniform(0, H, m)
            self.pv = r.integers(2, 5, m)
            self.pa = r.uniform(0.15, 0.6, m)
        elif e == "storm":
            self.flash = np.zeros(self.n, np.float32)
            for _ in range(3):
                i = r.integers(0, self.n)
                for j, v in enumerate((1.0, 0.35, 0.8, 0.3, 0.12, 0.05)):
                    self.flash[(i + j) % self.n] = max(self.flash[(i + j) % self.n], v)
            self.bolt_seed = int(r.integers(0, 1 << 30))
        elif e == "equalizer":
            self.bands = 48
            self.bf = r.integers(1, 5, (self.bands, 3))          # whole cycles per loop per partial
            self.bp = r.uniform(0, 2 * np.pi, (self.bands, 3))

    # ------------------------------------------------------------ helpers
    def _dots(self, buf, xs, ys, rs, amps):
        H, W = buf.shape
        for x, y, r, a in zip(xs, ys, rs, amps):
            k = int(np.ceil(r * 3))
            x0, x1, y0, y1 = int(max(0, x - k)), int(min(W, x + k + 1)), int(max(0, y - k)), int(min(H, y + k + 1))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            buf[y0:y1, x0:x1] += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * r * r))

    def _mix(self, fx_rgb: np.ndarray, mode: str = "add") -> np.ndarray:
        if mode == "screen":
            out = 255 - (255 - self.base) * (255 - np.clip(fx_rgb, 0, 255)) / 255
        else:
            out = self.base + fx_rgb
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------ frames
    def frame(self, i: int) -> np.ndarray:
        t = i / self.n
        th = 2 * np.pi * t
        e, W, H, A, s = self.p["effect"], self.W, self.H, self.amp, self.s
        c1, c2 = self.c1[None, None], self.c2[None, None]
        if e == "embers":
            buf1, buf2 = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
            ys = (self.py - self.pv * t * (H + 60)) % (H + 60) - 30
            xs = self.px + self.pw * np.sin(self.pph + th * self.pv)
            flick = 0.6 + 0.4 * np.sin(self.pph * 3 + th * 4)
            hot = self.pc < 0.35
            self._dots(buf1, xs[~hot], ys[~hot], self.pr[~hot], (self.pa * flick)[~hot])
            self._dots(buf2, xs[hot], ys[hot], self.pr[hot], (self.pa * flick)[hot])
            glow = np.exp(-((self.y - H) / (H * 0.35)) ** 2) * (0.25 + 0.05 * np.sin(th * 2))
            fx = (buf1[..., None] * c1 + buf2[..., None] * c2) * 255 * 2.4 + glow[..., None] * c1 * 140
            return self._mix(fx * A)
        if e == "bokeh":
            out = np.zeros((H, W, 3), np.float32)
            xs = (self.px + self.pdx * t * W) % W
            ys = (self.py + self.pdy * t * H) % H
            for x, y, r, a, ph, c in zip(xs, ys, self.pr, self.pa, self.pph, self.pc):
                k = int(r * 1.3) + 2
                x0, x1, y0, y1 = int(max(0, x - k)), int(min(W, x + k)), int(max(0, y - k)), int(min(H, y + k))
                if x1 <= x0 or y1 <= y0:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
                disc = np.clip((1.05 - d) * 8, 0, 1) * (0.55 + 0.45 * d ** 3)         # flat disc, brighter rim
                col = self.c1 if c < 0.6 else self.c2
                out[y0:y1, x0:x1] += disc[..., None] * col * 255 * a * (0.75 + 0.25 * np.sin(ph + th))
            return self._mix(out * A * 0.7, "screen")
        if e == "aurora":
            yy, xx = self.y / H, self.x / W * 2 * np.pi
            fx = np.zeros((H, W), np.float32)
            tint = np.zeros((H, W, 3), np.float32)
            for j, (f, ph, wd, col) in enumerate(((1, 0.0, 0.10, self.c1), (2, 1.7, 0.07, self.c2),
                                                  (3, 3.9, 0.05, self.c1))):
                cy = 0.35 + 0.12 * np.sin(f * xx + th * (1 if j % 2 == 0 else -1) + ph)
                band = np.exp(-((yy - cy) / wd) ** 2) * (0.6 + 0.4 * np.sin(xx * (j + 2) - th * (j + 1)))
                curtain = np.clip((yy - cy + 0.25) / 0.25, 0, 1) * np.exp(-np.clip(yy - cy, 0, None) / 0.12)
                v = band * 0.8 + curtain * 0.35 * band.max(axis=0, keepdims=True)
                fx += v
                tint += v[..., None] * col[None, None]
            return self._mix(tint * 255 * A * 0.75)
        if e == "starfield":
            buf = np.zeros((H, W), np.float32)
            z = (self.sz0 + t) % 1.0                                 # one pass toward the viewer per loop
            rr = (z ** 2.2) * np.hypot(W, H) * 0.6
            xs, ys = W / 2 + np.cos(self.sa) * rr, H / 2 + np.sin(self.sa) * rr
            fade = np.clip(z * 4, 0, 1) * np.clip((1 - z) * 6, 0, 1)
            for k in range(4):                                       # a short trail toward the centre
                q = 1 - k * 0.03
                self._dots(buf, W / 2 + (xs - W / 2) * q, H / 2 + (ys - H / 2) * q,
                           (0.8 + 2.4 * z) * s, self.sb * fade * (1 - k * 0.22))
            fx = buf[..., None] * (c1 * 0.7 + c2 * 0.3) * 255
            return self._mix(fx * A * 1.3)
        if e == "beams":
            out = np.zeros((H, W, 3), np.float32)
            for j in range(5):
                ox = W * (0.1 + 0.2 * j)
                ang = 0.45 * np.sin(th * (1 if j % 2 else -1) + j * 1.3)          # radians from straight down
                dx, dy = self.x - ox, self.y + 40 * s
                a = np.arctan2(dx, dy) - ang
                beam = np.exp(-(a / 0.07) ** 2) * np.clip(1 - dy / (H * 1.3), 0, 1)
                col = self.c1 if j % 2 == 0 else self.c2
                out += beam[..., None] * col * 255 * (0.6 + 0.4 * np.sin(th * 2 + j))
            haze = 0.08 * (1 + np.sin(th))
            return self._mix(out * A * 0.8 + haze * 255 * c2 * A, "screen")
        if e == "plasma":
            xx, yy = self.x / W * 2 * np.pi, self.y / H * 2 * np.pi
            v = (np.sin(2 * xx + th) + np.sin(3 * yy - th) + np.sin(2 * (xx + yy) + 2 * th)
                 + np.sin(np.hypot(xx - np.pi, yy - np.pi) * 2 - th))
            v = (v / 4 + 1) / 2
            fx = (v[..., None] * c1 + (1 - v)[..., None] * c2) * (v * (1 - v) * 4)[..., None] ** 1.5
            return self._mix(fx * 255 * A * 0.45)
        if e == "rain":
            buf = np.zeros((H, W), np.float32)
            for x0, y0, v, a in zip(self.px, self.py, self.pv, self.pa):
                y = (y0 + v * t * H) % H
                x = x0 - 0.25 * y
                n = int(30 * s * v)
                ys = (np.round(y) - np.arange(n)).astype(int)
                xs = np.round(x + 0.25 * np.arange(n)).astype(int)
                ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                buf[ys[ok], xs[ok]] += a * (1 - np.arange(n)[ok] / n)
            dark = 0.85
            fx = buf[..., None] * c1 * 255
            return np.clip(self.base * dark + fx * A * 1.5, 0, 255).astype(np.uint8)
        if e == "storm":
            from PIL import Image, ImageDraw, ImageFilter
            f = float(self.flash[i])
            img = self.base * (0.55 + 0.9 * f * A)
            if f > 0.5:
                r = np.random.default_rng(self.bolt_seed + i // 3)
                im = Image.new("L", (W, H))
                dr = ImageDraw.Draw(im)
                x, y = r.uniform(W * 0.1, W * 0.9), 0.0
                while y < H * r.uniform(0.5, 0.9):
                    nx, ny = x + r.normal(0, 25 * s), y + r.uniform(15, 40) * s
                    dr.line(((x, y), (nx, ny)), fill=255, width=max(1, round(3 * s)))
                    x, y = nx, ny
                core = np.asarray(im, np.float32) / 255
                halo = np.asarray(im.filter(ImageFilter.GaussianBlur(12 * s)), np.float32) / 255
                img = img + (core * 255 + halo * 500)[..., None] * (self.c1 * 0.7 + 0.3)[None, None] * f
            img = img + (np.exp(-((self.y) / (H * 0.5)) ** 2) * f * 90)[..., None] * c2
            return np.clip(img, 0, 255).astype(np.uint8)
        if e == "synthwave":
            img = self.base * 0.6
            hor = H * 0.58
            sun_r = H * 0.22
            d = np.hypot(self.x - W / 2, self.y - (hor - sun_r * 0.35))
            sun = (d < sun_r).astype(np.float32)
            stripes = np.where(self.y > hor - sun_r * 0.55,
                               (np.sin((self.y - hor) / (6 * s) - th * 2) > -0.2).astype(np.float32), 1.0)
            grad = np.clip((self.y - (hor - sun_r * 1.4)) / (sun_r * 1.4), 0, 1)
            suncol = c2 * (1 - grad[..., None]) + c1 * grad[..., None]
            img = img + (sun * stripes * (self.y < hor))[..., None] * suncol * 255 * A
            img = img + (np.exp(-np.clip(d - sun_r, 0, None) / (H * 0.12)) * 0.35 * (self.y < hor))[..., None] * c1 * 255 * A
            below = self.y >= hor
            depth = np.clip((self.y - hor) / (H - hor), 1e-3, 1)
            zz = 1 / depth
            hl = np.abs(((zz * 2 + t * 4) % 1.0) - 0.5) < 0.04 * zz ** 0.3 * 2
            vx = (self.x - W / 2) * depth
            vl = np.abs(((vx / (60 * s)) % 1.0) - 0.5) > 0.47
            grid = (hl | vl) & below
            img = np.where(below[..., None], img * 0.25, img)
            img = img + grid[..., None] * c1 * 255 * A * (0.4 + 0.6 * depth[..., None])
            return np.clip(img, 0, 255).astype(np.uint8)
        if e == "smoke":
            xx, yy = self.x / W * 2 * np.pi, self.y / H * 2 * np.pi
            v = np.zeros((H, W), np.float32)
            for f, a in ((1, 0.5), (2, 0.3), (4, 0.2)):
                v += a * np.sin(f * xx + np.sin(f * yy + th) * 1.5 + th) * np.sin(f * yy * 0.7 - th + f)
            v = np.clip((v + 0.6) / 1.4, 0, 1) ** 2 * np.clip((self.y / H - 0.2) / 0.8, 0, 1)
            fx = v[..., None] * (c1 * 0.7 + c2 * 0.3) * 255
            return self._mix(fx * A * 1.6, "screen")
        if e == "equalizer":
            img = self.base * 0.8
            bw = W / self.bands
            lv = (np.sin(self.bf * th + self.bp).mean(axis=1) * 0.5 + 0.5) ** 1.5
            lv = 0.15 + 0.85 * lv
            col_x = np.clip((self.x / bw).astype(int), 0, self.bands - 1)[0]
            height = lv[col_x] * H * 0.35
            gap = ((self.x % bw) > bw * 0.18)
            bars = (self.y > H - height[None, :]) & gap
            seg = ((H - self.y) % (10 * s)) > 3 * s
            frac = np.clip((H - self.y) / (H * 0.35), 0, 1)
            col = c1 * (1 - frac[..., None]) + c2 * frac[..., None]
            img = img + (bars & seg)[..., None] * col * 255 * A
            refl = (self.y > H - height[None, :] - 4) & gap
            return np.clip(img + refl[..., None] * 0, 0, 255).astype(np.uint8)
        return self.base.astype(np.uint8)


def encode_loop(anim: Animator, size=SIZE, crf: int = 16, bitrate: str = "4M") -> bytes:
    import av
    W, H = size
    buf = io.BytesIO()
    with av.open(buf, "w", format="webm") as c:
        s = c.add_stream("libvpx", rate=anim.fps)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"deadline": "good", "cpu-used": "5", "crf": str(crf), "b": bitrate, "auto-alt-ref": "0",
                     "kf-max-dist": str(anim.n)}
        for i in range(anim.n):
            for p in s.encode(av.VideoFrame.from_ndarray(anim.frame(i), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def repeat_to(loop: bytes, seconds: float, out: Path) -> None:
    """Copy the loop's packets end to end until `seconds` (no re-encoding). The loop starts on a keyframe."""
    import av
    src = av.open(io.BytesIO(loop))
    ins = src.streams.video[0]
    packets = [(p.pts, p.duration, p.is_keyframe, bytes(p)) for p in src.demux(ins) if p.size]
    tb, (W, H) = ins.time_base, (ins.codec_context.width, ins.codec_context.height)
    src.close()
    last_pts, last_dur = packets[-1][0], packets[-1][1] or packets[1][0] - packets[0][0]
    loop_pts = last_pts + last_dur
    with av.open(str(out), "w", format="webm") as dst:
        o = dst.add_stream("libvpx", rate=FPS)
        o.width, o.height, o.pix_fmt = W, H, "yuv420p"
        o.time_base = tb
        k = 0
        while k * loop_pts * tb < seconds:
            for pts, dur, key, data in packets:
                q = av.Packet(data)
                q.pts = q.dts = pts + k * loop_pts
                q.duration = dur
                q.time_base = tb
                q.is_keyframe = key
                q.stream = o
                dst.mux(q)
            k += 1


def render(preset_id: str, base_image: bytes | None, size=SIZE, seconds: float = LOOP_S, fps: int = FPS,
           strength: float = 0.8, crf: int = 28, bitrate: str = "1500k") -> bytes:
    p = PRESETS[preset_id]
    a = Animator(p, base_from_image(base_image, size), fps, seconds, strength)
    return encode_loop(a, size, crf, bitrate)


def preview(preset_id: str, base_image: bytes | None) -> bytes:
    return render(preset_id, base_image, (480, 270), seconds=LOOP_S, fps=20, crf=30, bitrate="700k")
