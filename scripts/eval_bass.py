"""Bass charts without a bass model: the guitar chart model run on the bass track, scored against human bass
charts on held-out songs (test + val splits, never trained on).

    python scripts/eval_bass.py [--songs 30]

Per song: the mix is separated and beat-tracked once (RiffCharter.analyse), then charted as
  guitar      the guitar chart itself (what copying the guitar part to bass would give)
  bass        the bass track through the chart model (+ the usual post-processing)
  bass_sparse the same with the density control at -0.5
  bass_single bass, then every chord made a single note (its lowest fret), as bass parts mostly are
and Expert / Hard / Medium / Easy are compared with the human bass part (eval/metrics.py). Summary per variant:
F1@50, precision, density vs human, chord rate, fret contour. Results in data/eval/bass/.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riffchart.audio.io import load_mix  # noqa: E402
from riffchart.baselines.common import mix_files  # noqa: E402
from riffchart.eval.metrics import Events, evaluate  # noqa: E402
from riffchart.formats.loader import load_song  # noqa: E402
from riffchart.formats.song import popcount  # noqa: E402
from riffchart.product.core import OUT_SR, RiffCharter, Style  # noqa: E402

ORDER = ("expert", "hard", "medium", "easy")
OUT = Path("data/eval/bass")


def singles(song):
    s = copy.deepcopy(song)
    for tr in s.tracks.values():
        m = tr.mask.astype(np.int64)
        chord = popcount(tr.mask) > 1
        low = m & -m                                   # lowest set bit
        tr.mask = np.where(chord, low, m).astype(tr.mask.dtype)
    return s


def bass_songs(n: int) -> pd.DataFrame:
    s = pd.read_parquet("data/audit/songs.parquet")
    s = s[s.split.isin(["test", "val"]) & (s.audio_s > 90) & (s.audio_s < 360)]
    keep = []
    for r in s.itertuples():
        try:
            b, _, _ = load_song(r.path, "bass")
        except Exception:
            continue
        ex = b.tracks.get("expert")
        if ex is not None and len(ex) >= 100:
            keep.append(r.song_id)
        if len(keep) >= n:
            break
    return s[s.song_id.isin(keep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=30)
    ap.add_argument("--only-reduce", action="store_true",
                    help="just bass_reduce (Expert from the model, Hard/Medium/Easy by the reducer), into rows_reduce")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    songs = bass_songs(a.songs)
    print(f"{len(songs)} held-out songs with a human bass chart", flush=True)
    ac = RiffCharter()
    rows, t0 = [], time.perf_counter()
    for i, r in enumerate(songs.itertuples(), 1):
        ref, _, _ = load_song(r.path, "bass")
        off = r.delay_s + r.align_correction_ms / 1000.0
        an = ac.analyse(load_mix(mix_files(r), OUT_SR, mono=False))
        if a.only_reduce:
            pc = dict(ac.post_cfg, lower="reduce")
            br, _ = ac.chart_part(an, an["stems"]["bass"], difficulties=["expert"], post_cfg=pc,
                                  lower=["hard", "medium", "easy"])
            variants = {"bass_reduce": singles(br)}
        else:
            g, _, _ = ac.chart_analysis(an)
            b, _ = ac.chart_part(an, an["stems"]["bass"])
            bs, _ = ac.chart_part(an, an["stems"]["bass"], style=Style(density=-0.5))
            variants = {"guitar": g, "bass": b, "bass_sparse": bs, "bass_single": singles(b)}
        for v, est in variants.items():
            for d in ORDER:
                rt = ref.tracks.get(d)
                if rt is None or len(rt) < 20:
                    continue
                r_ev = Events.from_track(ref, rt, off)
                et = est.tracks.get(d)
                e_ev = Events.from_track(est, et) if et is not None and len(et) else Events.empty()
                row = {"song_id": r.song_id, "variant": v, "difficulty": d}
                row.update(evaluate(r_ev, e_ev, float(r_ev.t.max() - r_ev.t.min())))
                rows.append(row)
        el = time.perf_counter() - t0
        print(f"[{i}/{len(songs)}] {str(r.artist)[:20]} - {str(r.name)[:30]}  "
              f"(about {el / i * (len(songs) - i) / 60:.0f} min left)", flush=True)
        pd.DataFrame(rows).to_parquet(OUT / ("rows_reduce.parquet" if a.only_reduce else "rows.parquet"))
    df = pd.DataFrame(rows)
    cols = ["F1@50", "P@50", "R@50", "density_ratio", "chord_rate_ref", "chord_rate_est", "fret_exact", "contour"]
    summ = df.groupby(["variant", "difficulty"])[cols].mean().round(3)
    summ = summ.reindex([(v, d) for v in ("guitar", "bass", "bass_sparse", "bass_single", "bass_reduce") for d in ORDER
                         if (v, d) in summ.index])
    print(summ.to_string(), flush=True)
    (OUT / ("summary_reduce.json" if a.only_reduce else "summary.json")).write_text(json.dumps({f"{v}/{d}": row.to_dict() for (v, d), row in summ.iterrows()},
                                                 indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
