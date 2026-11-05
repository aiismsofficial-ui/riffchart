"""Beat-relative note grid: 48 positions per beat (covers 1/16 = 12, 1/32 = 6, 1/64 = 3 and all
triplet subdivisions 16 / 8 / 4; Phase 3 analysis: 99.99% of human onsets fit 1/24 of a beat).

MUSICAL_POSITIONS restricts snapping to positions a charter actually uses: multiples of 3
(1/16-beat, i.e. 64th-note grid) or 4 (1/12-beat, sextuplets). This is what the model will emit.
"""
from __future__ import annotations

import numpy as np

from ..formats.song import TempoMap

GRID = 48
MUSICAL_POSITIONS = np.array([k for k in range(GRID) if k % 3 == 0 or k % 4 == 0])
# Model vocabulary: 32nds (multiples of 6) and triplet subdivisions (multiples of 4), 16 positions.
# Human charts never use the 64th-only positions (k % 6 == 3: 0.0% of 200 train songs), but snapping
# human notes onto slightly offset tracker grids put targets there, which taught the model to place
# notes a 64th early (Phase 4 ablation A1: 21% of its false positives).
MODEL_POSITIONS = np.array([k for k in range(GRID) if k % 6 == 0 or k % 4 == 0])


def snap(tm: TempoMap, times_s: np.ndarray, positions: np.ndarray = MUSICAL_POSITIONS) -> np.ndarray:
    """Audio seconds -> grid units (beat * 48 + position) on the nearest allowed position."""
    beat = np.asarray(tm.sec_to_tick(times_s), float) / tm.resolution
    base = np.floor(beat)
    frac = (beat - base) * GRID
    cand = np.concatenate([positions, [GRID]])
    k = cand[np.abs(frac[:, None] - cand[None, :]).argmin(1)]
    return (base * GRID + k).astype(np.int64)


def grid_to_ticks(tm: TempoMap, units: np.ndarray) -> np.ndarray:
    return np.asarray(units) * (tm.resolution // GRID) if tm.resolution % GRID == 0 else \
        np.round(np.asarray(units) * tm.resolution / GRID).astype(np.int64)


def grid_to_sec(tm: TempoMap, units: np.ndarray) -> np.ndarray:
    return tm.tick_to_sec(np.asarray(units, float) * tm.resolution / GRID)


def phase_class(units: np.ndarray) -> np.ndarray:
    """Rhythmic role of a grid position: 0 beat, 1 8th, 2 16th, 3 triplet-family, 4 32nd/finer."""
    p = np.asarray(units) % GRID
    out = np.full(p.shape, 4)
    out[p % 6 == 0] = 4          # 32nds (overwritten below if coarser)
    out[p % 4 == 0] = 3          # 1/12: sextuplets / 8th triplets
    out[p % 12 == 0] = 2         # 16ths
    out[p % 24 == 0] = 1         # 8ths
    out[p == 0] = 0
    return out
