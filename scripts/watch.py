"""Live progress view of every running job: latest line of each recently written log in logs/ plus GPU load.

    python scripts/watch.py [--hours 3] [--every 5]
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=3.0, help="show logs written within this many hours")
ap.add_argument("--every", type=float, default=5.0)
ap.add_argument("--idle", type=int, default=4, help="how many finished jobs to keep showing")
args = ap.parse_args()
LOGS = Path(__file__).resolve().parents[1] / "logs"


def last_line(p: Path) -> str:
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - 4096))
        txt = f.read().decode("utf-8", "replace")
    lines = [s.strip() for s in txt.replace("\r", "\n").split("\n") if s.strip()]
    return lines[-1] if lines else ""


def gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        u, m, t, c = [x.strip() for x in out.split(",")]
        return f"GPU {u}% busy | {int(m) / 1024:.1f} / {int(t) / 1024:.0f} GB | {c} C"
    except Exception:
        return "GPU: n/a"


while True:
    now = time.time()
    logs = sorted((p for p in LOGS.glob("*.log") if now - p.stat().st_mtime < args.hours * 3600),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    os.system("cls" if os.name == "nt" else "clear")
    print(f"RiffChart jobs  {time.strftime('%H:%M:%S')}   {gpu()}\n")
    if not logs:
        print("  no log written in the last", args.hours, "h")
    active = [p for p in logs if now - p.stat().st_mtime < 120]
    logs = active + [p for p in logs if p not in active][: args.idle]    # running jobs first
    for p in logs:
        age = now - p.stat().st_mtime
        state = "ACTIVE" if age < 120 else f"idle {age / 60:.0f} min"
        print(f"  {p.stem:<24} [{state}]")
        print(f"      {last_line(p)[-150:]}\n")
    time.sleep(args.every)
