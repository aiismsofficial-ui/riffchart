"""Average Expert metrics over sampling seeds: snap_<model>_<split> (seed 0) and snap_<model>_<split>_s<k>_E.

    python scripts/compare_seeds.py v2s60k v3p2k5 v3p20k [--seeds 1,2]
Also runs the fast-passage check (scripts/eval_fast_contour.py) per seed and averages it.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--seeds", default="1,2")
ap.add_argument("--splits", default="val")
args = ap.parse_args()
KEYS = ("F1@50", "fret_exact", "contour", "hopo_f1")
for split in args.splits.split(","):
    print(f"\n## {split}, Expert: mean over seeds (min..max)")
    print(f"{'model':<8} {'seeds':>5} " + " ".join(f"{k:>18}" for k in KEYS) + f" {'fast agree':>18} {'fast repeats':>18}")
    for m in args.models:
        systems = [f"snap_{m}_{split}"] + [f"snap_{m}_{split}_s{s}_E" for s in args.seeds.split(",")]
        systems = [s for s in systems if (Path("data/eval") / s / f"summary_{split}.json").exists()]
        vals = {k: [] for k in KEYS}
        for s in systems:
            e = json.loads((Path("data/eval") / s / f"summary_{split}.json").read_text())["expert"]
            for k in KEYS:
                vals[k].append(e[k])
        out = subprocess.run([sys.executable, "scripts/eval_fast_contour.py", *systems, "--split", split],
                             capture_output=True, text=True).stdout
        ag = [float(x) for x in re.findall(r"direction agreement ([\d.]+)", out)]
        rp = [float(x) for x in re.findall(r"repeats where the guitar moves ([\d.]+)%", out)]
        fmt = lambda v: f"{np.mean(v):.3f} ({min(v):.3f}..{max(v):.3f})"   # noqa: E731
        fmtp = lambda v: f"{np.mean(v):.1f}% ({min(v):.1f}..{max(v):.1f})"  # noqa: E731
        print(f"{m:<8} {len(systems):>5} " + " ".join(f"{fmt(vals[k]):>18}" for k in KEYS) +
              f" {fmt(ag):>18} {fmtp(rp):>18}", flush=True)
