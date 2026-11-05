"""Melody shape in fast passages: does the chart move where the human chart moves?

    python scripts/eval_fast_contour.py SYSTEM [SYSTEM ...] [--split val] [--diff expert] [--gap 0.15]

For consecutive single notes of the human chart closer than --gap seconds that both match a generated
note (+-50 ms): direction agreement (up / down / same, by fret position) and how often the generated
chart repeats a fret where the human chart moves (the "same note again and again" complaint).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riffchart.baselines.common import split_songs  # noqa: E402
from riffchart.eval.metrics import Events, match, pitch  # noqa: E402
from riffchart.formats.loader import load_song  # noqa: E402
from riffchart.formats.song import popcount  # noqa: E402
from riffchart.formats.textio import song_delay_s  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("systems", nargs="+")
ap.add_argument("--split", default="val")
ap.add_argument("--diff", default="expert")
ap.add_argument("--gap", type=float, default=0.15)
args = ap.parse_args()
rows = list(split_songs("data", args.split).itertuples())
common = [r for r in rows if all((Path("data/outputs") / s / r.song_id / "notes.chart").exists() for s in args.systems)]
print(f"{len(common)} songs with output from every system")
for v in args.systems:
    agree = n = moves = rep = 0
    for r in common:
        ref, _, _ = load_song(r.path)
        est, ini, _ = load_song(Path("data/outputs") / v / r.song_id)
        if args.diff not in ref.tracks or args.diff not in est.tracks:
            continue
        R = Events.from_track(ref, ref.tracks[args.diff], r.delay_s + r.align_correction_ms / 1000.0)
        E = Events.from_track(est, est.tracks[args.diff], song_delay_s(ini))
        mp = match(R, E, 0.05)
        m = np.full(len(R), -1)
        m[mp[:, 0]] = mp[:, 1]
        for i in range(len(R) - 1):
            if R.t[i + 1] - R.t[i] > args.gap or m[i] < 0 or m[i + 1] < 0 or m[i] == m[i + 1]:
                continue
            if popcount(R.mask[i:i + 2]).max() > 1:
                continue
            dr = np.sign(pitch(R.mask[i + 1:i + 2])[0] - pitch(R.mask[i:i + 1])[0])
            de = np.sign(pitch(E.mask[m[i + 1]:m[i + 1] + 1])[0] - pitch(E.mask[m[i]:m[i] + 1])[0])
            n += 1
            agree += dr == de
            if dr != 0:
                moves += 1
                rep += de == 0
    print(f"{v:<22} fast pairs {n:5d}  direction agreement {agree / max(n, 1):.3f}  "
          f"repeats where the guitar moves {rep / max(moves, 1):.1%}", flush=True)
