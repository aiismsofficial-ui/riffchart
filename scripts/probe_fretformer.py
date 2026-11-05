"""Pick Fretformer 2 decoding conventions the manifest leaves open, on VALIDATION songs only.

Variants: beat_idx convention (count-1 vs count), chord-token order (bitmask vs by size), HOPO
modifier meaning (final type vs .chart forced flag). Writes data/outputs/ffprobe_<variant>/ and
scores them on the val split.
"""
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from autochart.baselines.common import split_songs, write_song_folder
from autochart.baselines.fretformer import Fretformer, tokens_to_song
from autochart.eval.harness import score_system, summarize
import yaml

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cfg = yaml.safe_load(open("configs/eval.yaml"))
rows = list(split_songs("data", "val").itertuples())[:N]
ff = Fretformer()
results = []
toks = {}
for r in rows:
    cache = Path("data/cache/fretformer") / f"{r.song_id}.npz"
    if cache.exists():
        z = np.load(cache); feats = {k: z[k] for k in z.files}
    else:
        feats = ff.features(r); cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{k: np.asarray(v) for k, v in feats.items()})
    for bm in ("count_minus1", "count"):
        toks[(r.song_id, bm)] = (ff.generate(feats, beat_mode=bm), feats)
for bm, co, hm in itertools.product(("count_minus1",), ("mask",), ("type", "forced")):
    name = f"ffprobe_{bm}_{co}_{hm}"
    out = Path(cfg["outputs_root"]) / name
    shutil.rmtree(out, ignore_errors=True)
    for r in rows:
        t, feats = toks[(r.song_id, bm)]
        song = tokens_to_song(t, feats["beats"], feats["downbeats"], bm, co, hm)
        write_song_folder(out / r.song_id, song, str(r.name), str(r.artist), name, float(r.audio_s))
    s = summarize(score_system(name, cfg, "val"))
    e = s.get("expert", {})
    results.append((name, e.get("F1@50"), e.get("fret_exact"), e.get("contour"), e.get("type_acc"), e.get("hopo_f1")))
for r in results:
    print("%-40s F1@50 %.3f fret %.3f contour %.3f type_acc %.3f hopo_f1 %.3f" % r)
