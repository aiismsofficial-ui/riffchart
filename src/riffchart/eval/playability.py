"""Playability checker: rule violations per minute for one difficulty.

Rules (thresholds per difficulty live in configs/eval.yaml):
  open_chord      open note combined with frets (not a valid CH chord)
  stacked         two chord events closer than `stack_s` (duplicate / flam notes)
  chord_size      more simultaneous frets than the difficulty allows
  chord_span      chord spanning more frets than allowed (e.g. G+O on Hard)
  too_fast        inter-onset interval below the difficulty minimum
  fast_jump       pitch jump >= `jump_frets` with IOI below `jump_ioi_s`
  sustain_overlap sustain ends less than `sustain_gap_s` before the next onset (or past it)
  tiny_sustain    sustain shorter than `min_sustain_s`
  bad_hopo        HOPO on the same fret set as the previous note (nothing to hammer/pull)
  out_of_audio    notes before 0 s or after the audio end
Human charts are scored with the same checker so rates can be compared with a human baseline.
"""
from __future__ import annotations

import numpy as np

from ..formats.song import OPEN_BIT, NoteType, popcount
from .metrics import Events, pitch

RULES = ("open_chord", "stacked", "chord_size", "chord_span", "too_fast", "fast_jump",
         "sustain_overlap", "tiny_sustain", "bad_hopo", "out_of_audio")


def _span(mask: np.ndarray) -> np.ndarray:
    m = mask & 0x1F
    bits = (m[:, None] >> np.arange(5)) & 1
    has = bits.any(1)
    hi = np.where(has, 4 - np.argmax(bits[:, ::-1], 1), 0)
    lo = np.where(has, np.argmax(bits, 1), 0)
    return hi - lo


def find_issues(ev: Events, rules: dict, audio_s: float | None = None) -> list[tuple[str, int]]:
    """Every rule violation as (rule, note index). Pair rules point at the later note, except
    sustain_overlap, which points at the sustained note. `check` counts these."""
    out: list[tuple[str, int]] = []
    n = len(ev)
    if not n:
        return out

    def add(rule: str, hit: np.ndarray, shift: int = 0) -> None:
        out.extend((rule, int(i) + shift) for i in np.flatnonzero(hit))

    pc = popcount(ev.mask)
    add("open_chord", ((ev.mask & OPEN_BIT) > 0) & (pc > 1))
    add("chord_size", pc > rules["max_chord"])
    add("chord_span", (pc > 1) & (_span(ev.mask) > rules["max_span"]))
    if n > 1:
        ioi = np.diff(ev.t)
        add("stacked", ioi < rules["stack_s"], 1)
        add("too_fast", (ioi >= rules["stack_s"]) & (ioi < rules["min_ioi_s"]), 1)
        dp = np.abs(np.diff(pitch(ev.mask)))
        add("fast_jump", (dp >= rules["jump_frets"]) & (ioi < rules["jump_ioi_s"]), 1)
        end = ev.t[:-1] + ev.sus[:-1]
        add("sustain_overlap", (ev.sus[:-1] > 0) & (end > ev.t[1:] - rules["sustain_gap_s"]))
        add("bad_hopo", (ev.mask[1:] == ev.mask[:-1]) & (ev.ntype[1:] == NoteType.HOPO), 1)
    add("tiny_sustain", (ev.sus > 0) & (ev.sus < rules["min_sustain_s"]))
    hi = audio_s if audio_s is not None else np.inf
    add("out_of_audio", (ev.t < 0) | (ev.t > hi))
    return out


def check(ev: Events, rules: dict, audio_s: float | None = None) -> dict[str, float]:
    """rules: per-difficulty thresholds. Returns counts, per-minute rates and the total rate."""
    c = {r: 0 for r in RULES}
    for rule, _ in find_issues(ev, rules, audio_s):
        c[rule] += 1
    n = len(ev)
    minutes = (audio_s if audio_s else (ev.t.max() - ev.t.min() if n > 1 else 60.0)) / 60.0
    out = {f"pv_{k}": v / max(minutes, 1e-3) for k, v in c.items()}
    out["pv_total"] = sum(c.values()) / max(minutes, 1e-3)
    return out
