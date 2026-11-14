"""Bass charts without a bass model: the guitar chart model listens to the bass track (Demucs' bass stem, the
song's bass.ogg) instead of the guitar, with the full mix as its second input as always, and the result goes
through the same post-processing (playability fixes, Hard / Medium / Easy, star power).

Measured on held-out songs against human bass charts (scripts/eval_bass.py; NOTES.md "Bass from stems").
The chart is written as the .chart's [<Diff>DoubleBass] sections (PART BASS in a notes.mid); Clone Hero and
YARG offer it as the song's bass part. Songs whose bass is too quiet for its own track get no bass chart.
"""
from __future__ import annotations

import copy

import numpy as np

from ..formats.song import GuitarTrack, NoteType, Song, popcount

MIN_LEVEL = 0.02          # the bass track's RMS against the whole song's: below this it isn't charted (follow.py)
SECTION = "ExpertDoubleBass"


def enabled(config: str = "configs/product.yaml") -> bool:
    """Settings: chart the bass too (bass_chart in product.yaml / product.local.yaml)."""
    from ..paths import load_config
    return bool(load_config(config).get("bass_chart", True))


def loud_enough(bass: np.ndarray, mix: np.ndarray, min_level: float = MIN_LEVEL) -> bool:
    ref = float(np.sqrt(np.mean(np.asarray(mix, np.float32) ** 2))) or 1e-9
    return float(np.sqrt(np.mean(np.asarray(bass, np.float32) ** 2))) >= min_level * ref


def singles(tracks: dict[str, GuitarTrack]) -> dict[str, GuitarTrack]:
    """Every chord made one note (its lowest fret): bass parts are nearly all single notes."""
    out = copy.deepcopy(tracks)
    for tr in out.values():
        fix_same_fret_hopos(tr, single=True)
    return out


def fix_same_fret_hopos(tr: GuitarTrack, single: bool = False) -> int:
    """(single: chords to their lowest fret, then) a HOPO on the same fret as the note before can't be played
    as one (nothing to hammer on): make it a strum. Returns how many were changed."""
    m = tr.mask.astype(np.int64)
    if single:
        m = np.where(popcount(tr.mask) > 1, m & -m, m)
        tr.mask = m.astype(tr.mask.dtype)
    nt = np.asarray(tr.ntype).copy()
    bad = np.zeros(len(m), bool)
    bad[1:] = (m[1:] == m[:-1]) & (nt[1:] == NoteType.HOPO)
    nt[bad] = NoteType.STRUM
    tr.ntype = nt.astype(np.asarray(tr.ntype).dtype)
    return int(bad.sum())


def retime(tracks: dict[str, GuitarTrack], src: Song, dst: Song, shift_s: float = 0.0) -> dict[str, GuitarTrack]:
    """Tracks made against `src`'s tempo map moved onto `dst`'s (same seconds; shift_s is subtracted, e.g. the
    chart's audio offset), on a 1/48-beat grid."""
    if src.tempo is dst.tempo and not shift_s:
        return tracks
    step = max(1, dst.resolution // 48)

    def tk(ticks):
        t = dst.tempo.sec_to_tick(np.maximum(src.tempo.tick_to_sec(np.asarray(ticks)) - shift_s, 0.0))
        return (np.round(np.asarray(t, float) / step) * step).astype(np.int64)

    out = {}
    for d, tr in tracks.items():
        tr = copy.deepcopy(tr)
        on = tk(tr.tick)
        end = tk(np.asarray(tr.tick) + np.asarray(tr.sustain))
        tr.tick, tr.sustain = on, np.where(np.asarray(tr.sustain) > 0, np.maximum(end - on, 0), 0)
        tr.star_power = [(int(a), max(int(b) - int(a), 1)) for a, b in
                         zip(tk([p[0] for p in tr.star_power]), tk([p[0] + p[1] for p in tr.star_power]))]
        tr.solos = [(int(a), int(b)) for a, b in zip(tk([s[0] for s in tr.solos]), tk([s[1] for s in tr.solos]))]
        keep = np.concatenate([[True], np.diff(on) > 0]) if len(on) else np.zeros(0, bool)   # one chord a tick
        for f in ("tick", "mask", "sustain", "ntype", "forced", "disjoint"):
            setattr(tr, f, np.asarray(getattr(tr, f))[keep])
        out[d] = tr
    return out


OUR_CHARTERS = {"riffchart", "autochart"}      # autochart: the project's first name, in older songs' song.ini


def ours(ini: dict) -> bool:
    """Charted by RiffChart: only those get a bass part added (never a human chart)."""
    return str(ini.get("charter", "")).strip().lower() in OUR_CHARTERS


def in_chart(folder) -> bool:
    """The song's notes.chart already has a bass part (cheap: no parsing)."""
    from pathlib import Path
    p = Path(folder) / "notes.chart"
    try:
        return p.exists() and "[ExpertDoubleBass]" in p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
