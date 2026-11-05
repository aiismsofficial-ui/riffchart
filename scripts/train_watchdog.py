"""Watch a training log and exit (with a reason) as soon as something needs attention.

    python scripts/train_watchdog.py logs/v2_4_train.log [--every 60]

Exits on: finished, crash (Traceback/Error), auto-rollback after divergence, grad-norm blow-up, slowdown,
validation collapse, or no log output for 10 minutes (hung). Writes a heartbeat to logs/<name>_watchdog.log
so the job monitor shows it.
"""
import argparse
import re
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--every", type=float, default=60)
ap.add_argument("--hang_min", type=float, default=10)
ap.add_argument("--gn_max", type=float, default=50.0)
ap.add_argument("--slow", type=float, default=0.4, help="s/step considered too slow (3 lines in a row)")
args = ap.parse_args()
log = Path(args.log)
hb = log.with_name(log.stem + "_watchdog.log")
STEP = re.compile(r"^step (\d+) loss ([\d.]+) gn ([\d.]+) ([\d.]+)s/step")
MEM = re.compile(r"mem ([\d.]+)/([\d.]+)GB")
VAL = re.compile(r"^step (\d+) val (.*)")


def stop(msg: str, code: int) -> None:
    line = time.strftime("%H:%M:%S ") + "WATCHDOG STOP: " + msg
    print(line, flush=True)
    hb.open("a", encoding="utf-8").write(line + "\n")
    sys.exit(code)


best, rollbacks_seen, high_gn = float("inf"), 0, 0
while True:
    txt = log.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    if "Traceback" in txt or re.search(r"\b\w*Error\b", txt):
        stop("crash: " + next((l for l in reversed(lines) if "Error" in l), "Traceback"), 2)
    rb = [l for l in lines if "diverged, rolled back" in l]
    if len(rb) > rollbacks_seen:
        stop("auto-rollback fired: " + rb[-1], 3)
    vals = [(int(m[1]), m[2]) for l in lines if (m := VAL.match(l))]
    if vals and '"pos_acc": 0.0' in vals[-1][1] and vals[-1][0] > 2500:
        stop(f"validation collapsed at step {vals[-1][0]}: {vals[-1][1]}", 4)
    steps = [(int(m[1]), float(m[2]), float(m[3])) for l in lines if (m := STEP.match(l))]
    speeds = [float(m[4]) for l in lines if (m := STEP.match(l))]
    if len(speeds) >= 6 and all(v > args.slow for v in speeds[-3:]):
        stop(f"training slowed to {speeds[-3:]} s/step (GPU memory spill?) "
             + (MEM.search(lines[-1])[0] if MEM.search(lines[-1]) else ""), 7)
    recent = [s for s in steps[-5:] if s[0] > 3000]
    if len(recent) >= 3 and all(g > args.gn_max for _, _, g in recent[-3:]):
        stop(f"grad norm blowing up: {[(s, g) for s, _, g in recent[-3:]]}", 5)
    if lines and lines[-1].startswith("done:"):
        stop("training finished: " + lines[-1], 0)
    age = (time.time() - log.stat().st_mtime) / 60
    if age > args.hang_min:
        stop(f"no log output for {age:.0f} min (hung or killed)", 6)
    last = steps[-1] if steps else None
    hb.open("a", encoding="utf-8").write(
        time.strftime("%H:%M:%S ") + (f"ok  step {last[0]} loss {last[1]:.3f} gn {last[2]:.2f} {speeds[-1]:.2f}s/step" if last else "ok  starting")
        + (f"  | last val {vals[-1][1]}" if vals else "") + "\n")
    time.sleep(args.every)
