"""What the player plays along with: the guitar, or other instruments where the guitar drops out.

The chart model listens to two inputs: the separated guitar stem and the full mix. Songs without much
guitar (pop, electronic, ballads), and guitar songs with quiet passages, leave the guitar input nearly
empty there, so Expert goes quiet while the lower difficulties (generated separately) kept charting the
band. Players liked that (Creep, 2026-11-04), so it is made deliberate:

  follow = guitar  the guitar stem only (the original behaviour)
  follow = auto    guitar, plus the melodic stems (piano + other + vocals) where the guitar is quiet
  follow = all     guitar + melodic stems everywhere (for songs with little or no guitar)

The blend is what the model hears (melodic part level-matched to the song's usual guitar level) and,
at its real level, what is written as guitar.ogg, so the instrument being played is the one Clone Hero
mutes on a miss. song.ogg is the mix minus that track, so the two still add up to the full mix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

MODES = ("guitar", "auto", "all")
MELODIC = ("piano", "other", "vocals")


@dataclass
class Playable:
    audio: np.ndarray        # (n, 2) the play-along track as written (real levels)
    model_in: np.ndarray     # (n, 2) what the chart model hears in place of the guitar stem
    weight: np.ndarray       # (frames,) 0 = guitar only, 1 = melodic part added, per frame
    frame_s: float
    filled_s: float          # seconds where other instruments were added


def _env(x: np.ndarray, hop: int) -> np.ndarray:
    m = x.mean(1) if x.ndim == 2 else x
    n = len(m) // hop
    return np.sqrt((m[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)


def playable_part(stems: dict[str, np.ndarray], sr: int, mode: str = "auto", quiet_ratio: float = 0.3,
                  frame_s: float = 0.25, smooth_s: float = 4.0, fade_s: float = 1.0) -> Playable:
    if mode not in MODES:
        raise ValueError(f"follow must be one of {', '.join(MODES)}")
    guitar = stems["guitar"]
    n = len(guitar)
    hop = max(1, int(sr * frame_s))
    nf = int(np.ceil(n / hop))
    zero = Playable(guitar, guitar, np.zeros(nf), frame_s, 0.0)
    mel_parts = [stems[k] for k in MELODIC if k in stems]
    if mode == "guitar" or not mel_parts:
        return zero
    melodic = np.sum(mel_parts, 0)[:n]
    eg = median_filter(_env(guitar, hop), size=max(1, int(smooth_s / frame_s)), mode="nearest")
    em = median_filter(_env(melodic, hop), size=max(1, int(smooth_s / frame_s)), mode="nearest")
    loud = eg[eg > np.percentile(eg, 20)] if len(eg) else eg
    ref_g = float(np.percentile(loud, 75)) if len(loud) else 0.0
    ref_m = float(np.percentile(em, 75)) if len(em) else 0.0
    if mode == "all":
        w = np.ones(len(eg))
    else:
        if ref_g < 1e-4:                                   # practically no guitar in the song at all
            w = np.ones(len(eg))
        else:
            w = ((eg < quiet_ratio * ref_g) & (em > quiet_ratio * max(ref_m, 1e-6))).astype(float)
    w = np.clip(uniform_filter1d(w, size=max(1, int(fade_s / frame_s)), mode="nearest"), 0, 1)
    w = np.concatenate([w, np.full(max(0, nf - len(w)), w[-1] if len(w) else 0.0)])[:nf]
    if not w.any():
        return zero
    ws = np.repeat(w, hop)[:n, None].astype(np.float32)
    filled = w > 0.5
    lvl = float(np.median(em[filled[: len(em)]])) if filled[: len(em)].any() else ref_m
    gain = float(np.clip((ref_g if ref_g > 1e-4 else lvl) / max(lvl, 1e-6), 0.5, 4.0))
    audio = (guitar + ws * melodic).astype(np.float32)
    model_in = (guitar + ws * gain * melodic).astype(np.float32)
    return Playable(audio, model_in, w, frame_s, float(filled.sum() * frame_s))
