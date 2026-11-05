"""Greedy vs sampling for Fretformer 2, on validation songs (features cached by probe_fretformer.py)."""
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from autochart.baselines.common import split_songs, write_song_folder
from autochart.baselines.fretformer import Fretformer, tokens_to_song
from autochart.eval.harness import score_system, summarize

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cfg = yaml.safe_load(open("configs/eval.yaml"))
rows = list(split_songs("data", "val").itertuples())[:N]
ff = Fretformer()
for name, temp, top_p in (("greedy", 0.0, 1.0), ("t07", 0.7, 0.95), ("t10_p09", 1.0, 0.9)):
    out = Path(cfg["outputs_root"]) / f"ffsamp_{name}"
    shutil.rmtree(out, ignore_errors=True)
    for r in rows:
        z = np.load(Path("data/cache/fretformer") / f"{r.song_id}.npz"); feats = {k: z[k] for k in z.files}
        t = ff.generate(feats, temperature=temp, top_p=top_p, beat_mode="count_minus1")
        song = tokens_to_song(t, feats["beats"], feats["downbeats"], "count_minus1", "mask", "forced")
        write_song_folder(out / r.song_id, song, str(r.name), str(r.artist), name, float(r.audio_s))
    s = summarize(score_system(f"ffsamp_{name}", cfg, "val"))
    for d in ("expert", "hard"):
        e = s.get(d, {})
        print(f"{name:8s} {d:6s} F1@50 {e.get('F1@50', float('nan')):.3f} fret {e.get('fret_exact', float('nan')):.3f} "
              f"contour {e.get('contour', float('nan')):.3f} density {e.get('density_ratio', float('nan')):.3f} pv {e.get('pv_total', float('nan')):.2f}")
