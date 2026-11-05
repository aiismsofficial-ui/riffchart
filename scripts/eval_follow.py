"""Validate the difficulty ladder and the follow modes through the real product pipeline.

    python scripts/eval_follow.py [--split val] [--songs 15]

For each song: load the mix, run separation + beat tracking once (RiffCharter.analyse), then chart it with
each variant and write data/outputs/fl_<variant>/<song_id>. Variants:
  base          follow=guitar, ladder off   (the product before this change)
  ladder        follow=guitar, ladder on
  ladder_auto   follow=auto,   ladder on
Then the harness scores each variant against the human charts, and the script prints per difficulty:
F1@50, density vs human, lower/Expert chord ratio, and "inversions" (16-beat windows where an easier
difficulty has more chords than the next harder one).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riffchart.audio.io import load_mix  # noqa: E402
from riffchart.baselines.common import split_songs, write_song_folder  # noqa: E402
from riffchart.product.core import OUT_SR, RiffCharter, Style  # noqa: E402

VARIANTS = {"base": ("guitar", False), "ladder": ("guitar", True), "ladder_auto": ("auto", True)}
ORDER = ("expert", "hard", "medium", "easy")


def inversions(song) -> dict[str, int]:
    win = song.resolution * 16
    out = {}
    for k in range(1, 4):
        lo, hi = song.tracks.get(ORDER[k]), song.tracks.get(ORDER[k - 1])
        if lo is None or hi is None or not len(lo):
            continue
        n = int(max(lo.tick.max(), hi.tick.max() if len(hi) else 0) // win) + 1
        a, b = np.bincount(lo.tick // win, minlength=n), np.bincount(hi.tick // win, minlength=n)
        out[ORDER[k]] = int((a > b).sum())
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--songs", type=int, default=15)
args = ap.parse_args()
ac = RiffCharter()
songs = split_songs("data", args.split)
songs = songs[(songs.audio_s > 90) & (songs.audio_s < 330)]
songs = songs.sort_values("real_hard", ascending=False, kind="stable").head(args.songs)   # human reductions first
stats = {v: {"inv": {d: 0 for d in ORDER[1:]}, "ratio": {d: [] for d in ORDER[1:]}, "filled": []} for v in VARIANTS}
t_all = time.perf_counter()
for i, r in enumerate(songs.itertuples(), 1):
    t = time.perf_counter()
    mix = load_mix([Path(r.path) / f for f in json.loads(r.audio_files)], OUT_SR, mono=False)
    an = ac.analyse(mix)
    for v, (follow, lad) in VARIANTS.items():
        ac.post_cfg.setdefault("ladder", {})["enabled"] = lad
        song, play, _ = ac.chart_analysis(an, style=Style(follow=follow))
        write_song_folder(Path("data/outputs") / f"fl_{v}" / r.song_id, song, str(r.name), str(r.artist), "riffchart",
                          float(r.audio_s))
        for d, n in inversions(song).items():
            stats[v]["inv"][d] += n
        ne = max(len(song.tracks["expert"]), 1)
        for d in ORDER[1:]:
            stats[v]["ratio"][d].append(len(song.tracks[d]) / ne)
        stats[v]["filled"].append(play.filled_s)
    el = time.perf_counter() - t_all
    print(f"[{i}/{len(songs)}] {r.name[:40]}  {time.perf_counter() - t:.0f}s  "
          f"(about {el / i * (len(songs) - i) / 60:.0f} min left)", flush=True)

py = sys.executable
print("\nscoring...", flush=True)
for v in VARIANTS:
    subprocess.run([py, "-m", "riffchart.eval.harness", f"fl_{v}", "--split", args.split], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"\n{'variant':<12} {'diff':<7} {'F1@50':>6} {'P@50':>6} {'dens':>6} {'vsE':>6} {'inv':>5}")
for v in VARIANTS:
    summ = json.loads((Path("data/eval") / f"fl_{v}" / f"summary_{args.split}.json").read_text())
    for d in ORDER:
        s = summ.get(d, {})
        ratio = f"{np.median(stats[v]['ratio'][d]):.2f}" if d != "expert" else "1.00"
        inv = stats[v]["inv"].get(d, "")
        print(f"{v:<12} {d:<7} {s.get('F1@50', float('nan')):6.3f} {s.get('P@50', float('nan')):6.3f} "
              f"{s.get('density_ratio', float('nan')):6.2f} {ratio:>6} {inv!s:>5}", flush=True)
    print(f"{v:<12} filled with other instruments: {np.sum(stats[v]['filled']):.0f} s over {len(songs)} songs", flush=True)
print(f"\nhuman lower/Expert chord ratio (median, train): hard 0.78, medium 0.56, easy 0.43", flush=True)
