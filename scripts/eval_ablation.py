"""Score ablation runs: chart Expert on the first N validation songs, harness metrics + val loss.

    python scripts/eval_ablation.py A1_base A2_mixonly ... [--n 15]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--pos-bias", type=float, default=0.0)
ap.add_argument("--min-gap", type=float, default=0.0)
ap.add_argument("--tag", default="")
args = ap.parse_args()
py = sys.executable
rows = []
for run in args.runs:
    ck = Path("data/checkpoints/chart") / run / "best.pt"
    name = f"abl_{run}" + (f"_t{args.temperature}" if args.temperature else "") +         (f"_pb{args.pos_bias}" if args.pos_bias else "") + (f"_gap{args.min_gap}" if args.min_gap else "") + args.tag
    shutil.rmtree(Path("data/outputs") / name, ignore_errors=True)
    subprocess.run([py, "-m", "riffchart.model.generate", "--ckpt", str(ck), "--split", "val", "--limit", str(args.n),
                    "--name", name, "--diffs", "expert", "--temperature", str(args.temperature),
                    "--pos-bias", str(args.pos_bias), "--min-gap", str(args.min_gap)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([py, "-m", "riffchart.eval.harness", name, "--split", "val"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    s = json.loads((Path("data/eval") / name / "summary_val.json").read_text())["expert"]
    import torch
    st = torch.load(ck, map_location="cpu", weights_only=False)
    rows.append({"run": name, "val_loss": st.get("val", {}).get("loss"), "pos_acc": st.get("val", {}).get("pos_acc"), "step": st["step"],
                 **{k: s[k] for k in ("F1@25", "F1@50", "P@50", "R@50", "fret_exact", "chord_agree", "contour",
                                      "sustain_acc", "hopo_f1", "density_ratio", "pv_total")}})
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
out = Path("data/eval/ablations.csv")
df.to_csv(out, mode="a", header=not out.exists(), index=False)
