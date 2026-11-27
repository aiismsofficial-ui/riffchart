"""Highway designs as layers, still or animated.

A design is drawn in four layers, each a float32 HxWx3 array (512x1024 by default, tiling top to bottom):

  surface   the colours, pattern image, vignette, centre glow, fret-coloured lanes and text
  glow      the edges' soft light, added on top (pulses and flickers scale it)
  effect    the animation for frame t (added, or for "hue" a colour change), between surface and lines
  lines     lane lines and edges, drawn last with their own alpha so effects pass under them

Every effect is periodic in t (0..1 = one loop) and in y (one tile), so the video loops without a jump and
the highway tiles without a seam, whatever Clone Hero does with it (scroll it, stretch it, repeat it).
A still highway (highway.png) is the same layers with no effect, so a design looks the same both ways.

Videos are VP8 .webm (yuv420p), the one format Clone Hero's video highways accept on every platform.
"""
from __future__ import annotations

import colorsys
import io
from pathlib import Path

import numpy as np

FRET_RGB = [(34, 197, 94), (239, 68, 68), (234, 179, 8), (59, 130, 246), (249, 115, 22)]
EFFECTS = {
    "none": "None (a still image)",
    "pulse": "Pulse: the edges breathe",
    "hue": "Colour cycle: edges and glow change colour",
    "flow": "Plasma: slow flowing colour",
    "aurora": "Aurora: drifting light ribbons",
    "sparks": "Sparks rushing down the highway",
    "warp": "Warp speed: light streaks",
    "rain": "Rain: thin falling streaks",
    "lightning": "Lightning along the edges",
    "grid": "Synthwave grid rolling toward you",
    "fire": "Fire licking up the edges",
    "sheen": "Sheen: a band of light sweeping down",
    "chevrons": "Chevrons: arrows rushing toward you",
    "matrix": "Digital rain: falling blocks in the lanes",
    "ripple": "Ripples: rings spreading across the highway",
    "lava": "Lava: molten blobs drifting down",
    "lanechase": "Lane chase: the lanes light up in turn (fret colours)",
}


def _rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


# ---------------------------------------------------------------- static layers

def layers(d: dict, texture: np.ndarray | None, overlay, size=(512, 1024)) -> dict:
    """The design's still layers. d = highways.clean_design output; texture HxWx3 float or None; overlay RGBA
    PIL image (text) or None."""
    from PIL import Image, ImageDraw, ImageFilter
    W, H = size
    s = W / 512                                                  # previews are drawn smaller
    rail_w = round(d["rail_width"] * s)
    lane = (W - 2 * rail_w) / 5
    x = np.arange(W, dtype=np.float32)
    # surface: a gradient top -> middle -> top, so the tile wraps
    t = (1 - np.cos(np.linspace(0, 2 * np.pi, H, endpoint=False))) / 2
    a, b = _rgb(d["top"]), _rgb(d["bottom"])
    surf = np.repeat((a[None] * (1 - t[:, None]) + b[None] * t[:, None])[:, None, :], W, axis=1)
    if texture is not None and d["pattern_opacity"] > 0:
        tex = texture if texture.shape[:2] == (H, W) else np.asarray(
            Image.fromarray(texture.clip(0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32)
        surf = _blend(surf, tex, d.get("pattern_blend", "normal"), d["pattern_opacity"] / 100)
    if d.get("fret_tint", 0):                                    # each lane faintly in its fret colour
        k = d["fret_tint"] / 100 * 0.35
        lane_i = np.clip(((x - rail_w) / lane).astype(int), 0, 4)
        tint = np.array(FRET_RGB, np.float32)[lane_i]           # (W, 3)
        inside = ((x >= rail_w) & (x < W - rail_w))[:, None]
        surf = np.where(inside[None], surf * (1 - k) + tint[None] * k, surf)
    if d.get("center_glow", 0):
        g = np.exp(-((x - W / 2) / (W * 0.22)) ** 2)[None, :, None]
        surf = surf + g * (d["center_glow"] / 100) * 70 * (_rgb(d["rail"]) / 255 * 0.6 + 0.4)
    if d.get("vignette", 0):
        v = 1 - (d["vignette"] / 100) * 0.75 * (np.abs(x - W / 2) / (W / 2)) ** 2
        surf = surf * v[None, :, None]
    if overlay is not None:
        ov = overlay if overlay.size == (W, H) else overlay.resize((W, H), Image.BILINEAR)
        o = np.asarray(ov, np.float32)
        al = o[..., 3:4] / 255
        surf = surf * (1 - al) + o[..., :3] * al
    # glow: blurred bands inside the edges, made over a 3-tile strip so it is seamless
    glow = np.zeros((H, W, 3), np.float32)
    if d["glow"] and rail_w:
        col = tuple(int(c) for c in _rgb(d["rail"]) * 0.9 * d["glow"] / 100)
        gi = Image.new("RGB", (W, 3 * H))
        gd = ImageDraw.Draw(gi)
        reach = rail_w + round(18 * s) + (round(22 * s) if d.get("rail_style") == "neon" else 0)
        gd.rectangle((0, 0, reach, 3 * H), fill=col)
        gd.rectangle((W - reach, 0, W, 3 * H), fill=col)
        gi = gi.filter(ImageFilter.GaussianBlur(14 * s)).crop((0, H, W, 2 * H))
        glow = np.asarray(gi, np.float32) / 2
    # lines: RGBA
    li = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(li)
    if d["lanes"]:
        div = tuple(int(c) for c in _rgb(d["divider"]))
        lw = max(1, round(d.get("lane_width", 2) * s))
        style = d.get("lane_style", "solid")
        for i in range(1, 5):
            xx = round(rail_w + i * lane)
            if style == "dashed":
                step = max(8, round(64 * s))
                for y0 in range(0, H, step):
                    ld.line((xx, y0, xx, y0 + step * 0.55), fill=div + (255,), width=lw)
            else:
                ld.line((xx, 0, xx, H), fill=div + (255,), width=lw)
        if style == "glow":
            halo = li.filter(ImageFilter.GaussianBlur(5 * s))
            li = Image.alpha_composite(halo, li)
            ld = ImageDraw.Draw(li)
    if rail_w > 0:
        r = _rgb(d["rail"])
        style = d.get("rail_style", "solid")
        for x0, x1, inner in ((0, rail_w - 1, rail_w), (W - rail_w, W - 1, W - rail_w - 1)):
            if style == "chrome":                                # light -> rail colour -> dark across the edge
                for k in range(x0, x1 + 1):
                    f = (k - x0) / max(1, x1 - x0)
                    f = f if x0 == 0 else 1 - f
                    c = np.clip(r * (0.55 + 0.9 * np.sin(np.pi * (0.15 + 0.7 * f))) + 40 * (1 - f), 0, 255)
                    ld.line((k, 0, k, H), fill=tuple(int(v) for v in c) + (255,))
            elif style == "double":
                third = max(1, (x1 - x0 + 1) // 3)
                ld.rectangle((x0, 0, x0 + third - 1, H), fill=tuple(int(v) for v in r) + (255,))
                ld.rectangle((x1 - third + 1, 0, x1, H), fill=tuple(int(v) for v in r) + (255,))
            elif style == "neon":                                # a bright core with the colour around it
                ld.rectangle((x0, 0, x1, H), fill=tuple(int(v) for v in r * 0.85) + (255,))
                cx = (x0 + x1) // 2
                ld.rectangle((cx - max(1, rail_w // 6), 0, cx + max(1, rail_w // 6), H),
                             fill=tuple(int(v) for v in np.clip(r * 0.4 + 170, 0, 255)) + (255,))
            else:
                ld.rectangle((x0, 0, x1, H), fill=tuple(int(v) for v in r) + (255,))
            if style in ("solid", "neon"):
                light = tuple(int(v) for v in np.clip(r + 60, 0, 255))
                ld.line((inner, 0, inner, H), fill=light + (255,), width=1)
    lines = np.asarray(li, np.float32)
    return {"surface": surf, "glow": glow, "lines_rgb": lines[..., :3], "lines_a": lines[..., 3:4] / 255,
            "rail_w": rail_w, "size": (W, H)}


def _blend(base, tex, mode, a):
    if mode == "screen":
        m = 255 - (255 - base) * (255 - tex) / 255
    elif mode == "multiply":
        m = base * tex / 255
    elif mode == "overlay":
        m = np.where(base < 128, 2 * base * tex / 255, 255 - 2 * (255 - base) * (255 - tex) / 255)
    else:
        m = tex
    return base * (1 - a) + m * a


def compose(L: dict, effect: np.ndarray | None = None, glow_gain: float = 1.0, recolor=None) -> np.ndarray:
    """One RGB uint8 frame from the layers."""
    img = L["surface"] + L["glow"] * glow_gain
    if effect is not None:
        img = img + effect
    lr, la = L["lines_rgb"], L["lines_a"]
    if recolor is not None:
        lr, img = recolor(lr), img
    img = img * (1 - la) + lr * la
    return np.clip(img, 0, 255).astype(np.uint8)


def still(d: dict, texture, overlay, size=(512, 1024)) -> np.ndarray:
    return compose(layers(d, texture, overlay, size))


# ---------------------------------------------------------------- effects

def _hue_shift(rgb: np.ndarray, turn: float) -> np.ndarray:
    """Rotate hue by `turn` (0..1) keeping lightness: a fast YIQ rotation."""
    th = 2 * np.pi * turn
    c, s = np.cos(th), np.sin(th)
    m = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], np.float32)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)
    full = np.linalg.inv(m) @ rot @ m
    return np.clip(rgb @ full.T.astype(np.float32), 0, 255)


def _glow_sprite(r: float) -> np.ndarray:
    k = int(np.ceil(r * 3))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * r * r)).astype(np.float32)


def _add_streak(buf, x, y, length, width, amp):
    """A vertical streak whose head is at (x, y), its tail above it, wrapping top to bottom."""
    H, W = buf.shape
    x0, x1 = int(max(0, x - 3 * width)), int(min(W, x + 3 * width + 1))
    if x1 <= x0:
        return
    xs = np.arange(x0, x1)
    prof_x = np.exp(-((xs - x) / max(0.6, width)) ** 2)
    n = max(2, int(length))
    ys = (np.round(y) - np.arange(n)).astype(int) % H
    prof_y = (1 - np.arange(n) / n) ** 1.5
    buf[ys[:, None], xs[None, :]] += amp * prof_y[:, None] * prof_x[None, :]


class Animator:
    """Effect frames for one design. Random choices are made once (seeded), so frame i is always the same."""

    def __init__(self, d: dict, L: dict, fps: int):
        self.d, self.L, self.fps = d, L, fps
        self.W, self.H = L["size"]
        self.n = max(2, int(round(d["anim_seconds"] * fps)))
        self.k = int(d["anim_speed"])                           # whole cycles per loop: seamless
        self.amp = d["anim_strength"] / 100
        self.col = _rgb(d["anim_color"])
        self.rng = np.random.default_rng(1234)
        W, H = self.W, self.H
        self.x = np.arange(W, dtype=np.float32)[None, :]
        self.y = np.arange(H, dtype=np.float32)[:, None]
        rw = L["rail_w"]
        self.inner = (rw, W - rw)
        if d["anim"] in ("sparks", "warp", "rain"):
            count = {"sparks": 70, "warp": 46, "rain": 140}[d["anim"]]
            self.px = self.rng.uniform(rw + 4, W - rw - 4, count)
            self.py = self.rng.uniform(0, H, count)
            self.pv = self.rng.integers(1, 4, count) * self.k     # tiles per loop, whole numbers: seamless
            self.pa = self.rng.uniform(0.4, 1.0, count)
        if d["anim"] == "lightning":
            self.bolts = [self._bolt_plan() for _ in range(self.n)]
        r = self.rng
        if d["anim"] == "matrix":
            self.cell = H / 64                                    # 64 cells a tile: wraps exactly
            self.ncol = max(3, int((W - 2 * rw) / (16 * W / 512)))
            self.drops = [(c, r.uniform(0, H), int(r.integers(1, 3)) * self.k, r.uniform(0.18, 0.35))
                          for c in range(self.ncol) for _ in range(2)]
            self.cellph = r.uniform(0, 2 * np.pi, (64, self.ncol))
        if d["anim"] == "ripple":
            m = 9
            self.rx = r.uniform(rw + 20 * W / 512, W - rw - 20 * W / 512, m)
            self.ry = r.uniform(0, H, m)
            self.rp = r.uniform(0, 1, m)
            self.rr = r.uniform(0.35, 0.6, m) * (W - 2 * rw)
        if d["anim"] == "lava":
            m = 7
            self.bx = r.uniform(0.25, 0.75, m)
            self.bf = r.integers(1, 3, m)
            self.bph = r.uniform(0, 2 * np.pi, m)
            self.by = r.uniform(0, H, m)
            self.bv = r.integers(1, 3, m) * self.k
            self.br = r.uniform(0.09, 0.16, m) * (W - 2 * rw)

    def _bolt_plan(self):
        r = self.rng
        on = r.random() < 0.28
        return (on, r.integers(0, 2), r.uniform(0, self.H), r.uniform(0.25, 0.7) * self.H, r.integers(0, 1 << 30))

    def frame(self, i: int) -> np.ndarray:
        d, L = self.d, self.L
        t = i / self.n
        th = 2 * np.pi * t
        e = d["anim"]
        W, H, k, amp, col = self.W, self.H, self.k, self.amp, self.col
        if e == "none":
            return compose(L)
        if e == "pulse":
            g = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(k * th))
            edge = np.exp(-np.minimum(self.x, W - 1 - self.x) / (W * 0.12))
            fx = (edge * (0.5 + 0.5 * np.cos(k * th)) * amp * 60)[..., None] * (col / 255)[None, None]
            return compose(L, np.broadcast_to(fx, (H, W, 3)), glow_gain=0.4 + amp * 1.6 * g)
        if e == "hue":
            turn = (k * t) % 1.0
            rec = lambda rgb: _hue_shift(rgb, turn)
            Lh = dict(L, glow=_hue_shift(L["glow"], turn))
            return compose(Lh, None, 1.0 + amp * 0.5, rec)
        if e == "flow":
            yy = self.y * 2 * np.pi / H
            xx = self.x / W * 2 * np.pi
            v = (np.sin(2 * yy + 1.3 * xx + k * th) + np.sin(3 * yy - 2.1 * xx - k * th)
                 + np.sin(1 * yy + 3.7 * xx + 2 * k * th) + np.sin(4 * yy + 0.8 * xx - k * th + 1.0))
            v = (v / 4 + 1) / 2                                  # 0..1
            v = v ** 2.2
            return compose(L, v[..., None] * amp * 140 * (col / 255)[None, None])
        if e == "aurora":
            yy = self.y * 2 * np.pi / H
            fx = np.zeros((H, W), np.float32)
            for j, (f, ph, wd) in enumerate(((1, 0.0, 0.10), (2, 2.1, 0.07), (3, 4.2, 0.05))):
                cx = W / 2 + W * 0.28 * np.sin(f * yy + k * th * (1 if j % 2 == 0 else -1) + ph)
                fx += np.exp(-((self.x - cx) / (W * wd)) ** 2) * (0.6 + 0.4 * np.sin(yy * (j + 1) - k * th))
            c2 = _hue_shift(col[None], 0.18)[0]
            tint = (col / 255)[None, None] * (0.5 + 0.5 * np.sin(yy))[..., None] + \
                   (c2 / 255)[None, None] * (0.5 - 0.5 * np.sin(yy))[..., None]
            return compose(L, fx[..., None] * amp * 110 * tint)
        if e in ("sparks", "warp", "rain"):
            buf = np.zeros((H, W), np.float32)
            s = W / 512
            for x, y0, v, a in zip(self.px, self.py, self.pv, self.pa):
                y = (y0 + v * t * H) % H
                if e == "sparks":
                    _add_streak(buf, x, y, 46 * s * v / k, 2.4 * s, a * 1.0)
                    _add_streak(buf, x, y, 7 * s, 6 * s, a * 1.2)
                elif e == "warp":
                    _add_streak(buf, x, y, 190 * s * v / k, 2.0 * s, a * 1.0)
                else:
                    _add_streak(buf, x, y, 60 * s, 1.3 * s, a * 0.8)
            c = col if e != "warp" else col * 0.5 + 127
            buf = 1 - np.exp(-buf * 1.6)                         # soft clip: overlaps glow instead of burning out
            return compose(L, buf[..., None] * amp * 330 * (c / 255)[None, None])
        if e == "lightning":
            from PIL import Image, ImageDraw, ImageFilter
            on, side, y0, length, seed = self.bolts[i]
            prev = self.bolts[i - 1]
            gain = 0.6
            if not on and not prev[0]:
                return compose(L, None, glow_gain=gain)
            if not on:
                on, side, y0, length, seed = prev                # afterglow frame, dimmer
                fade = 0.45
            else:
                fade = 1.0
            r = np.random.default_rng(seed)
            im = Image.new("L", (W, H))
            dr = ImageDraw.Draw(im)
            rw = self.L["rail_w"]
            x = (rw + 2) if side == 0 else (W - rw - 3)
            pts, y = [], y0
            while y < y0 + length:
                pts.append((x + r.normal(0, 9 * W / 512), y % H))
                y += r.uniform(12, 30) * H / 1024
            for p, q in zip(pts, pts[1:]):
                if abs(p[1] - q[1]) < H / 2:
                    dr.line((p, q), fill=255, width=max(1, round(2 * W / 512)))
            if len(pts) > 4 and r.random() < 0.7:                # a branch across the lane
                bx, by = pts[len(pts) // 2]
                dirn = 1 if side == 0 else -1
                for _ in range(6):
                    nx, ny = bx + dirn * r.uniform(8, 20) * W / 512, by + r.uniform(-6, 18) * H / 1024
                    dr.line(((bx, by), (nx, ny)), fill=180, width=1)
                    bx, by = nx, ny
            core = np.asarray(im, np.float32) / 255
            halo = np.asarray(im.filter(ImageFilter.GaussianBlur(7 * W / 512)), np.float32) / 255
            fx = (core * 255 + halo * 600 * amp)[..., None] * (col / 255 * 0.6 + 0.4)[None, None]
            return compose(L, fx * fade, glow_gain=gain + 1.2 * amp * fade)
        if e == "grid":
            gap = H / 8
            off = (k * t * gap) % gap                             # one line spacing per cycle: seamless
            phase = ((self.y - off) % gap) / gap
            lines = np.exp(-(np.minimum(phase, 1 - phase) * gap / (2.2 * W / 512)) ** 2)
            vert = np.zeros((1, W), np.float32)
            lane = (W - 2 * self.L["rail_w"]) / 5
            for j in range(6):
                cx = self.L["rail_w"] + j * lane
                vert = np.maximum(vert, np.exp(-((self.x - cx) / (3 * W / 512)) ** 2))
            fx = np.maximum(lines * 0.9, vert * 0.5)
            return compose(L, fx[..., None] * amp * 200 * (col / 255)[None, None])
        if e == "fire":
            yy = self.y * 2 * np.pi / H
            dist = np.minimum(self.x - self.inner[0], self.inner[1] - self.x).clip(0) / W   # from the edges
            n = (np.sin(6 * yy + k * th * 2 + 9 * self.x / W) + np.sin(11 * yy + k * th * 3 - 5 * self.x / W)
                 + np.sin(17 * yy + k * th * 4 + 13 * self.x / W)) / 3
            reach = 0.12 + 0.22 * (n + 1) / 2 * (0.6 + 0.4 * np.sin(3 * yy - k * th * 2) ** 2)
            f = np.clip(1 - dist / reach, 0, 1) ** 1.4
            c = (col / 255)[None, None]
            core = (f ** 3)[..., None] * np.array([1.0, 0.95, 0.8], np.float32)[None, None]   # white-hot near the edge
            return compose(L, (f[..., None] * c * 260 + core * 200) * amp)
        if e == "sheen":
            yc = (k * t % 1.0) * H
            dy = np.minimum(np.abs(self.y - yc), H - np.abs(self.y - yc))
            band = np.exp(-(dy / (H * 0.06)) ** 2) * (0.75 + 0.25 * np.cos((self.x - W / 2) / W * np.pi))
            return compose(L, band[..., None] * amp * 120 * (col / 255 * 0.5 + 0.5)[None, None],
                           glow_gain=1 + 0.8 * amp)
        if e == "chevrons":
            s = W / 512
            gap = H / 6                                           # six arrows a tile
            off = (k * t * gap) % gap
            half = (W - 2 * self.L["rail_w"]) / 2
            xn = np.abs(self.x - W / 2) / half                    # 0 in the middle, 1 at the edges
            phase = ((self.y - off + xn * gap * 0.45) % gap) / gap   # a V pointing down: toward the player
            dist = np.minimum(phase, 1 - phase) * gap
            line = np.exp(-(dist / (5 * s)) ** 2) + 0.35 * np.exp(-(dist / (16 * s)) ** 2)
            line = line * np.clip(1.15 - xn, 0, 1)
            return compose(L, line[..., None] * amp * 170 * (col / 255)[None, None], glow_gain=1 + 0.4 * amp)
        if e == "matrix":
            rw = self.L["rail_w"]
            colw = (W - 2 * rw) / self.ncol
            cell = self.cell
            ci = (np.arange(H) // cell).astype(int) % 64
            yc = (ci + 0.5) * cell
            buf = np.zeros((H, W), np.float32)
            inside = (self.y[:, 0] % cell) < cell * 0.72
            for c in range(self.ncol):
                v = np.zeros(H, np.float32)
                for cc, y0, sp, tail in self.drops:
                    if cc != c:
                        continue
                    head = (y0 + sp * t * H) % H
                    behind = (head - yc) % H                       # how far above the head (wraps)
                    v = np.maximum(v, np.exp(-behind / (H * tail)) * (1 + 1.5 * (behind < cell)))
                flick = 0.7 + 0.3 * np.sin(self.cellph[ci, c] + k * th * 3)
                x0 = int(round(rw + c * colw + colw * 0.18))
                x1 = int(round(rw + c * colw + colw * 0.82))
                buf[:, x0:x1] = (v * flick * inside)[:, None]
            c2 = col * 0.4 + 153
            fx = np.minimum(buf, 1)[..., None] * (col / 255)[None, None] * 210 + \
                 np.clip(buf - 1, 0, 1.5)[..., None] * (c2 / 255)[None, None] * 120
            return compose(L, fx * amp)
        if e == "ripple":
            s = W / 512
            fx = np.zeros((H, W), np.float32)
            for x0, y0, p, R in zip(self.rx, self.ry, self.rp, self.rr):
                f = (k * t + p) % 1.0
                dy = np.abs(self.y - y0)
                dy = np.minimum(dy, H - dy)                         # wrap top / bottom
                dd = np.hypot(self.x - x0, dy)
                r = f * R
                fade = np.sin(np.pi * f) ** 1.5                     # 0 when a ring starts and ends: seamless
                fx += (np.exp(-((dd - r) / (4 * s)) ** 2) + 0.4 * np.exp(-((dd - r) / (14 * s)) ** 2)
                       + 0.5 * np.exp(-((dd - r * 0.8) / (2.5 * s)) ** 2)) * fade
            fx = 1 - np.exp(-fx * 1.4)
            return compose(L, fx[..., None] * amp * 330 * (col / 255 * 0.7 + 0.3)[None, None], glow_gain=1 + 0.3 * amp)
        if e == "lava":
            rw = self.L["rail_w"]
            field = np.zeros((H, W), np.float32)
            for bx, bf, ph, y0, v, R in zip(self.bx, self.bf, self.bph, self.by, self.bv, self.br):
                cx = rw + (W - 2 * rw) * (bx + 0.12 * np.sin(bf * th + ph))
                cy = (y0 + v * t * H) % H
                dy = np.abs(self.y - cy)
                dy = np.minimum(dy, H - dy)
                field += R * R / ((self.x - cx) ** 2 + (dy * 0.8) ** 2 + 1)
            blob = np.clip((field - 0.8) / 0.6, 0, 1)
            rim = np.clip(1 - np.abs(field - 1.0) / 0.35, 0, 1) * (field < 1.6)
            hot = np.array([1.0, 0.9, 0.55], np.float32)[None, None]
            fx = blob[..., None] * (col / 255)[None, None] * 190 + (blob ** 3)[..., None] * hot * 90 + \
                 rim[..., None] * (col / 255)[None, None] * 120
            return compose(L, fx * amp)
        if e == "lanechase":
            rw = self.L["rail_w"]
            lane = (W - 2 * rw) / 5
            pos = (k * t * 5) % 5
            fx = np.zeros((H, W, 3), np.float32)
            for j in range(5):
                dj = min(abs(pos - j), 5 - abs(pos - j))             # around the ring of lanes: seamless
                g = np.exp(-dj * dj / 0.6)
                cx = rw + (j + 0.5) * lane
                prof = np.exp(-((self.x - cx) / (lane * 0.42)) ** 2) + 0.5 * np.exp(-((self.x - cx) / (lane * 0.12)) ** 2)
                fx += (prof * g)[..., None] * np.array(FRET_RGB[j], np.float32)[None, None] / 255
            return compose(L, np.broadcast_to(fx * amp * 300, (H, W, 3)), glow_gain=1 + 0.5 * amp)
        return compose(L)


# ---------------------------------------------------------------- video

def encode_webm(frames, fps: int, size, out: Path | None = None, crf: int = 12, bitrate: str = "6M") -> bytes:
    """VP8 .webm from RGB uint8 frames. Returns the bytes (and writes `out` if given)."""
    import av
    W, H = size
    buf = io.BytesIO()
    with av.open(buf, "w", format="webm") as c:
        s = c.add_stream("libvpx", rate=fps)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"deadline": "good", "cpu-used": "4", "crf": str(crf), "b": bitrate, "auto-alt-ref": "1",
                     "lag-in-frames": "16"}
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    data = buf.getvalue()
    if out is not None:
        Path(out).write_bytes(data)
    return data


def video(d: dict, texture, overlay, size=(512, 1024), fps: int = 30, crf: int = 12, bitrate: str = "6M") -> bytes:
    L = layers(d, texture, overlay, size)
    a = Animator(d, L, fps)
    return encode_webm((a.frame(i) for i in range(a.n)), fps, size, crf=crf, bitrate=bitrate)
