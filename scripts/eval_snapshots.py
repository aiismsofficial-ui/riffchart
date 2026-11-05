"""Pick a checkpoint by decoded chart quality (val loss doesn't predict it): chart the first N songs of each
split with every checkpoint at production decoding, score with the harness, print one table per split.

    python scripts/eval_snapshots.py v1=data/checkpoints/chart/full_v1_last/best.pt \
        v2s30k=data/checkpoints/chart/full_v2/step30000.pt ... [--splits val_<tag>:25,val:20]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+", help="name=path")
ap.add_argument("--splits", default="val:20", help="split:songs,... e.g. val_<tag>:25,val:20")
ap.add_argument("--source", default="extra")
ap.add_argument("--temperature", default="0.7")
ap.add_argument("--seed", default="0")
ap.add_argument("--diffs", default=None, help="e.g. expert (faster); outputs get an _E suffix")
args = ap.parse_args()
py = sys.executable
splits = [(s.split(":")[0], s.split(":")[1]) for s in args.splits.split(",")]
names = []
for spec in args.ckpts:
    name, ck = spec.split("=", 1)
    for split, n in splits:
        sysname = f"snap_{name}_{split}" + (f"_s{args.seed}" if args.seed != "0" else "") + ("_E" if args.diffs else "")
        names.append((sysname, split))
        print(f"{time.strftime('%H:%M:%S')} {name} on {split} ({n} songs)", flush=True)
        subprocess.run([py, "-m", "riffchart.model.generate", "--ckpt", ck, "--split", split, "--limit", n,
                        "--name", sysname, "--temperature", args.temperature, "--min-gap", "auto",
                        "--source", args.source, "--seed", args.seed] + (["--diffs", args.diffs] if args.diffs else []),
                       check=True)
        subprocess.run([py, "-m", "riffchart.eval.harness", sysname, "--split", split], check=True,
                       stdout=subprocess.DEVNULL)
for split, _ in splits:
    systems = [s for s, sp in names if sp == split]
    out = subprocess.run([py, "-m", "riffchart.eval.harness", *systems, "--split", split, "--compare"],
                         check=True, capture_output=True, text=True).stdout
    print(f"\n## {split}\n{out}", flush=True)
    Path("data/eval").mkdir(exist_ok=True)
    (Path("data/eval") / f"snapshots_{split}.md").write_text(out, encoding="utf-8")
