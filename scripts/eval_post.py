"""Score post-processing variants of one generated system on a split (Phase 5 selection on val).

    python scripts/eval_post.py <system> --split val --variant NAME "key=val key=val" [--variant ...]
Each variant runs riffchart.post.cli with those --set overrides, then the harness; the table
(mean over songs, per difficulty) is printed and appended to data/eval/post_variants.csv.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("system")
ap.add_argument("--split", default="val")
ap.add_argument("--variant", nargs=2, action="append", metavar=("NAME", "SETS"), default=[])
ap.add_argument("--include-raw", action="store_true")
args = ap.parse_args()
py = sys.executable
KEYS = ("F1@25", "F1@50", "P@50", "R@50", "fret_exact", "contour", "sustain_acc", "hopo_f1", "density_ratio", "pv_total")
rows = []
todo = ([(args.system, None)] if args.include_raw else []) + [(f"{args.system}__{n}", s) for n, s in args.variant]
for name, sets in todo:
    if sets is not None:
        shutil.rmtree(Path("data/outputs") / name, ignore_errors=True)
        cmd = [py, "-m", "riffchart.post.cli", args.system, name, "--split", args.split]
        for s in sets.split():
            cmd += ["--set", s]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([py, "-m", "riffchart.eval.harness", name, "--split", args.split], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summ = json.loads((Path("data/eval") / name / f"summary_{args.split}.json").read_text())
    for d in ("expert", "hard", "medium", "easy"):
        if d in summ:
            rows.append({"variant": name.split("__")[-1], "diff": d, **{k: summ[d].get(k) for k in KEYS}})
    print("done", name, flush=True)
df = pd.DataFrame(rows)
for d in ("expert", "hard", "medium", "easy"):
    print(f"\n{d}\n" + df[df["diff"] == d].drop(columns="diff").round(3).to_string(index=False))
out = Path("data/eval/post_variants.csv")
df.assign(system=args.system, split=args.split).to_csv(out, mode="a", header=not out.exists(), index=False)
