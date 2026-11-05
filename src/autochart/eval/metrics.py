"""Chart-vs-chart metrics. Everything is in audio seconds, chord-level events.

Matching uses maximum bipartite matching within a tolerance (mir_eval.util.match_events), so one
estimated chord can match at most one reference chord.

Metrics on matched pairs (at the 50 ms tolerance):
  fret_exact     mask identical (chords and opens included)
  single_fret    both single notes and same fret
  chord_agree    both chords or both singles
  chord_exact    among pairs where both are chords: identical mask
  contour        for consecutive reference notes i, i+1 both matched (to j < j'), sign of the pitch
                 step agrees (pitch = mean fret index, open = -1)
  sustain_acc    both sustained (> min) or both not
  sustain_len    median |log2(est / ref)| over pairs where both sustain
  hopo_f1 / tap_f1 / type_acc   note-type agreement
"""
from __future__ import annotations

from dataclasses import dataclass

import mir_eval
import numpy as np

from ..formats.song import OPEN_BIT, GuitarTrack, NoteType, Song, popcount

SUSTAIN_MIN_S = 0.05


@dataclass
class Events:
    """Chord events of one difficulty in audio seconds."""

    t: np.ndarray       # onset seconds
    mask: np.ndarray    # uint8
    sus: np.ndarray     # sustain seconds
    ntype: np.ndarray   # NoteType

    def __len__(self) -> int:
        return len(self.t)

    @staticmethod
    def from_track(song: Song, tr: GuitarTrack, offset_s: float = 0.0) -> "Events":
        on, sus = song.times(tr)
        # Charting programs write 1-tick (or very short) sustains on plain notes; anything below a
        # 1/16-beat is not a sustain (cf. YARG MidReader sustain cutoff comment).
        sus = np.where(tr.sustain < song.resolution / 16, 0.0, sus)
        return Events(on + offset_s, tr.mask.copy(), sus, tr.ntype.copy())

    @staticmethod
    def empty() -> "Events":
        z = np.zeros(0)
        return Events(z, z.astype(np.uint8), z, z.astype(np.uint8))


def pitch(mask: np.ndarray) -> np.ndarray:
    """Mean fret index (G=0..O=4); open = -1."""
    m = np.asarray(mask, np.uint8)
    bits = ((m[:, None] >> np.arange(5)) & 1).astype(float)
    n = bits.sum(1)
    p = (bits * np.arange(5)).sum(1) / np.maximum(n, 1)
    return np.where((m & OPEN_BIT) > 0, -1.0, p)


def match(ref: Events, est: Events, tol: float) -> np.ndarray:
    """(k, 2) array of (ref_idx, est_idx) pairs sorted by ref index."""
    if len(ref) == 0 or len(est) == 0:
        return np.zeros((0, 2), int)
    pairs = mir_eval.util.match_events(ref.t, est.t, tol)
    return np.array(sorted(pairs), int).reshape(-1, 2)


def _prf(n_match: int, n_ref: int, n_est: int) -> tuple[float, float, float]:
    p = n_match / n_est if n_est else 0.0
    r = n_match / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _f1(ref_pos: np.ndarray, est_pos: np.ndarray) -> float:
    """F1 of a binary attribute; NaN when neither side has any positive (nothing to agree on)."""
    if not ref_pos.any() and not est_pos.any():
        return float("nan")
    tp = int((ref_pos & est_pos).sum())
    _, _, f = _prf(tp, int(ref_pos.sum()), int(est_pos.sum()))
    return f


def evaluate(ref: Events, est: Events, duration_s: float | None = None) -> dict[str, float]:
    out: dict[str, float] = {"n_ref": len(ref), "n_est": len(est)}
    for tol in (0.025, 0.05):
        k = len(match(ref, est, tol))
        p, r, f = _prf(k, len(ref), len(est))
        ms = int(tol * 1000)
        out[f"P@{ms}"], out[f"R@{ms}"], out[f"F1@{ms}"] = p, r, f

    pairs = match(ref, est, 0.05)
    nan = float("nan")
    if len(pairs):
        ri, ei = pairs[:, 0], pairs[:, 1]
        rm, em = ref.mask[ri], est.mask[ei]
        rc, ec = popcount(rm) > 1, popcount(em) > 1
        out["fret_exact"] = float((rm == em).mean())
        single = ~rc & ~ec
        out["single_fret"] = float((rm[single] == em[single]).mean()) if single.any() else nan
        out["chord_agree"] = float((rc == ec).mean())
        both = rc & ec
        out["chord_exact"] = float((rm[both] == em[both]).mean()) if both.any() else nan
        # contour over consecutive matched reference notes
        consec = np.where(np.diff(ri) == 1)[0]
        consec = consec[ei[consec + 1] > ei[consec]]
        if len(consec):
            pr, pe = pitch(ref.mask), pitch(est.mask)
            dr = np.sign(pr[ri[consec + 1]] - pr[ri[consec]])
            de = np.sign(pe[ei[consec + 1]] - pe[ei[consec]])
            out["contour"] = float((dr == de).mean())
        else:
            out["contour"] = nan
        rs, es = ref.sus[ri] > SUSTAIN_MIN_S, est.sus[ei] > SUSTAIN_MIN_S
        out["sustain_acc"] = float((rs == es).mean())
        out["sustain_f1"] = _f1(rs, es)
        bs = rs & es
        out["sustain_len"] = float(np.median(np.abs(np.log2(est.sus[ei][bs] / ref.sus[ri][bs])))) if bs.any() else nan
        rt, et = ref.ntype[ri], est.ntype[ei]
        out["type_acc"] = float((rt == et).mean())
        out["hopo_f1"] = _f1(rt == NoteType.HOPO, et == NoteType.HOPO)
        out["tap_f1"] = _f1(rt == NoteType.TAP, et == NoteType.TAP)
    else:
        for k in ("fret_exact", "single_fret", "chord_agree", "chord_exact", "contour", "sustain_acc",
                  "sustain_f1", "sustain_len", "type_acc", "hopo_f1", "tap_f1"):
            out[k] = nan

    span = duration_s or max(float(ref.t.max() - ref.t.min()) if len(ref) else 0.0, 1.0)
    out["nps_ref"] = len(ref) / span
    out["nps_est"] = len(est) / span
    out["density_ratio"] = out["nps_est"] / out["nps_ref"] if out["nps_ref"] else nan
    out["chord_rate_ref"] = float((popcount(ref.mask) > 1).mean()) if len(ref) else nan
    out["chord_rate_est"] = float((popcount(est.mask) > 1).mean()) if len(est) else nan
    return out
