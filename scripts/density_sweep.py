"""How strongly does the note-start logit bias change chart density on the product model?

    python scripts/density_sweep.py [--songs 3] [--biases -3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3] [--diffs expert,easy]

Charts the first N `val` songs at production decoding with each bias (raw logit units) and prints
notes per difficulty relative to bias 0, before and after post-processing.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riffchart.baselines.common import split_songs  # noqa: E402
from riffchart.formats.song import DIFFICULTIES  # noqa: E402
from riffchart.model.dataset import Song as SongData  # noqa: E402
from riffchart.model.generate import Charter, events_to_song  # noqa: E402
from riffchart.paths import project_path  # noqa: E402
from riffchart.post.pipeline import postprocess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--songs", type=int, default=3)
ap.add_argument("--biases", default="-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3")
ap.add_argument("--diffs", default="expert,easy")
args = ap.parse_args()
biases = [float(x) for x in args.biases.split(",")]
diffs = [d for d in DIFFICULTIES if d in args.diffs.split(",")]
prod = yaml.safe_load(open(project_path("configs/product.yaml"), encoding="utf-8"))
post_cfg = yaml.safe_load(open(project_path("configs/post.yaml"), encoding="utf-8"))
rules = yaml.safe_load(open(project_path("configs/eval.yaml"), encoding="utf-8"))["playability"]
gaps = {d: float(rules[d]["min_ioi_s"]) for d in DIFFICULTIES}
torch.backends.cuda.matmul.allow_tf32 = True
ch = Charter(str(project_path(prod["chart_ckpt"])))
s = split_songs("data", "val")
s = s[(s.audio_s > 120) & (s.audio_s < 300)].head(args.songs)
raw = {b: {d: [] for d in diffs} for b in biases}
post = {b: {d: [] for d in diffs} for b in biases}
total = len(s) * len(biases)
k = 0
for r in s.itertuples():
    sd = SongData(r.song_id)
    beats, downs = sd.grids["ft"]
    for b in biases:
        k += 1
        t = time.perf_counter()
        ev = ch.chart(sd.mel, beats, downs, diffs, temperature=prod["decoding"]["temperature"], seed=0,
                      audio_s=sd.audio_s, pos_bias=b, min_gap_s=gaps, source="extra")
        song = events_to_song(ev, beats, downs)
        for d in diffs:
            raw[b][d].append(len(song.tracks[d]))
        song = postprocess(song, post_cfg, audio_s=float(r.audio_s), mel=sd.mel, beats=beats, downbeats=downs,
                           lower=[d for d in diffs if d != "expert"])
        for d in diffs:
            post[b][d].append(len(song.tracks[d]))
        print(f"[{k}/{total}] {r.name[:30]} bias {b:+.1f}: " +
              " ".join(f"{d} {raw[b][d][-1]}->{post[b][d][-1]}" for d in diffs) + f" ({time.perf_counter() - t:.0f}s)",
              flush=True)
print("\nnotes relative to bias 0 (mean over songs), raw -> after post-processing")
for b in biases:
    cells = []
    for d in diffs:
        rr = np.mean(np.array(raw[b][d]) / np.maximum(raw[0.0][d], 1))
        pp = np.mean(np.array(post[b][d]) / np.maximum(post[0.0][d], 1))
        cells.append(f"{d} {rr:.2f} -> {pp:.2f}")
    print(f"bias {b:+.1f}: " + "  ".join(cells), flush=True)
