"""Score the pitch-following fret pass (post/contour.py) on already-charted songs.

    python scripts/eval_contour.py [--system fl_ladder] [--split val]

Takes each song's chart from data/outputs/<system>, its Demucs guitar stem from data/stems/<id>/guitar.ogg,
applies variants of the pass (then the playability fixer) and scores them with the harness.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from autochart.baselines.common import split_songs, write_song_folder  # noqa: E402
from autochart.formats.loader import load_song  # noqa: E402
from autochart.formats.song import Song  # noqa: E402
from autochart.paths import project_path  # noqa: E402
from autochart.post.contour import follow_pitch, track_pitch  # noqa: E402
from autochart.post.playability import fix_track  # noqa: E402

VARIANTS = {"off": None, "default": {}, "conf5": {"min_conf": 0.5}, "conf7": {"min_conf": 0.7},
            "agree5": {"agree_thr": 0.5}, "run3": {"min_run": 3}}
ap = argparse.ArgumentParser()
ap.add_argument("--system", default="fl_ladder")
ap.add_argument("--split", default="val")
args = ap.parse_args()
rules = yaml.safe_load(open(project_path("configs/eval.yaml"), encoding="utf-8"))["playability"]
src = Path("data/outputs") / args.system
rows = [r for r in split_songs("data", args.split).itertuples() if (src / r.song_id / "notes.chart").exists()]
changed = {v: 0 for v in VARIANTS}
total = 0
t0 = time.perf_counter()
for i, r in enumerate(rows, 1):
    song, _, _ = load_song(src / r.song_id)
    y, sr = sf.read(f"data/stems/{r.song_id}/guitar.ogg", dtype="float32", always_2d=True)
    pt = track_pitch(y.mean(1), sr)
    total += sum(len(t) for t in song.tracks.values())
    for v, cfg in VARIANTS.items():
        tracks = {}
        for d, tr in song.tracks.items():
            if cfg is not None:
                tr, n = follow_pitch(song, tr, pt, 0.0, cfg)
                changed[v] += n
                tr = fix_track(song, tr, rules[d], audio_s=float(r.audio_s))
            tracks[d] = tr
        out = Song(song.resolution, song.tempo, tracks, song.sections)
        write_song_folder(Path("data/outputs") / f"ct_{v}" / r.song_id, out, str(r.name), str(r.artist), "autochart",
                          float(r.audio_s))
    print(f"[{i}/{len(rows)}] {r.name[:40]} ({time.perf_counter() - t0:.0f}s)", flush=True)
py = sys.executable
KEYS = ("F1@50", "fret_exact", "single_fret", "contour", "hopo_f1", "pv_total")
print(f"\n{'variant':<9} {'diff':<7} " + " ".join(f"{k:>11}" for k in KEYS) + "   frets changed")
for v in VARIANTS:
    subprocess.run([py, "-m", "autochart.eval.harness", f"ct_{v}", "--split", args.split], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summ = json.loads((Path("data/eval") / f"ct_{v}" / f"summary_{args.split}.json").read_text())
    for d in ("expert", "hard", "medium", "easy"):
        s = summ.get(d, {})
        print(f"{v:<9} {d:<7} " + " ".join(f"{s.get(k, float('nan')):11.3f}" for k in KEYS) +
              (f"   {changed[v] / max(total, 1):.1%}" if d == "expert" else ""), flush=True)
