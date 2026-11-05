"""User density control (the app's "sparser / denser" slider), applied after decoding.

The decoder's note-start logit bias turned out to be a weak and noisy control on the v2 model (Expert
moved about +-10% even at bias +-3, lower difficulties varied more between seeds than between settings;
scripts/density_sweep.py, NOTES "density control"). So density is set here, deterministically:

  density < 0  keep the strongest chords: rank by the learned reducer's keep probability (post/reduce.py,
               the model for the next easier difficulty) and keep round((1 + 0.5 * density) * n) of them,
               so -1 keeps 50%. Downbeat chords rank high in that model, so the rhythm skeleton stays.
  density > 0  Hard / Medium / Easy take back Expert chords they dropped, best-ranked first (this
               difficulty's reducer model on the Expert chart), up to +60% at +1, with frets fitted to the
               difficulty (chord size, Easy without orange). Expert can't borrow from above; the caller
               adds a positive decoding bias for it instead.

The playability fixer runs afterwards, so added notes respect the difficulty's minimum gaps.
"""
from __future__ import annotations

import numpy as np

from ..formats.hopo import chart_hopo_threshold, natural_hopo
from ..formats.song import OPEN_BIT, GuitarTrack, NoteType, Song
from .playability import _fit_chord, rebuild
from .reduce import _finish, features

NEXT_EASIER = {"expert": "hard", "hard": "medium", "medium": "easy", "easy": "easy"}
MAX_CHORD = {"expert": 3, "hard": 3, "medium": 2, "easy": 2}
THIN_PER_UNIT = 0.5      # density -1 keeps 50% of chords
ADD_PER_UNIT = 0.6       # density +1 adds up to 60% more chords (lower difficulties)


def keep_scores(song: Song, tr: GuitarTrack, as_diff: str, models: dict) -> np.ndarray:
    """Reducer keep probability of each chord of `tr` for difficulty `as_diff`."""
    if len(tr) == 0:
        return np.zeros(0)
    return models["keep"].predict_proba(features(song, tr, as_diff))[:, 1]


def thin(song: Song, tr: GuitarTrack, density: float, models: dict) -> GuitarTrack:
    n = len(tr)
    keep_n = int(round((1 + THIN_PER_UNIT * density) * n))
    if density >= 0 or n == 0 or keep_n >= n:
        return tr
    score = keep_scores(song, tr, NEXT_EASIER[tr.difficulty], models)
    keep = np.zeros(n, bool)
    keep[np.argsort(-score, kind="stable")[:max(keep_n, 1)]] = True
    return _finish(song, tr, tr.difficulty, keep, tr.mask.astype(int))


def _fit(mask: int, diff: str) -> int:
    m = _fit_chord(int(mask), MAX_CHORD[diff], 4)
    if diff == "easy" and not m & OPEN_BIT and m & 16:
        m = (m >> 1) or 1                                # Easy has no orange
    return m


def thicken(song: Song, tr: GuitarTrack, ex: GuitarTrack, density: float, models: dict) -> GuitarTrack:
    """Add Expert chords missing from a lower difficulty, best-ranked first."""
    if density <= 0 or tr.difficulty == "expert" or len(ex) == 0:
        return tr
    have = set(int(t) for t in tr.tick)
    cand = np.array([i for i, t in enumerate(ex.tick) if int(t) not in have], int)
    k = min(len(cand), int(round(ADD_PER_UNIT * density * max(len(tr), 1))))
    if k <= 0:
        return tr
    score = keep_scores(song, ex, tr.difficulty, models)[cand]
    add = cand[np.argsort(-score, kind="stable")[:k]]
    tick = np.concatenate([tr.tick, ex.tick[add]])
    mask = np.concatenate([tr.mask, [_fit(m, tr.difficulty) for m in ex.mask[add]]]).astype(np.uint8)
    sus = np.concatenate([tr.sustain, ex.sustain[add]])
    is_tap = np.concatenate([tr.ntype == NoteType.TAP, ex.ntype[add] == NoteType.TAP])
    added = np.concatenate([np.zeros(len(tr), bool), np.ones(len(add), bool)])
    old_type = np.concatenate([tr.ntype, np.zeros(len(add), np.uint8)])
    o = np.argsort(tick, kind="stable")
    tick, mask, sus, is_tap, added, old_type = tick[o], mask[o], sus[o], is_tap[o], added[o], old_type[o]
    # added chords play as the game would place them by default (natural HOPO or strum)
    nat = natural_hopo(tick, mask, chart_hopo_threshold(song.resolution), midi_exception=False)
    ntype = np.where(added, np.where(is_tap, NoteType.TAP, np.where(nat, NoteType.HOPO, NoteType.STRUM)), old_type)
    return rebuild(tr.difficulty, tick, mask, sus, ntype.astype(np.uint8), song.resolution, tr.star_power, tr.solos)


def adjust(song: Song, density: float, models: dict) -> dict[str, GuitarTrack]:
    tracks = dict(song.tracks)
    if not density:
        return tracks
    ex = tracks.get("expert")
    for d, tr in list(tracks.items()):
        tracks[d] = thin(song, tr, density, models) if density < 0 else \
            thicken(song, tr, ex if ex is not None else GuitarTrack.empty("expert"), density, models)
    return tracks
