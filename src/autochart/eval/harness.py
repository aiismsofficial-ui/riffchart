"""Score a system's charts against the human test charts.

A "system" is a folder data/outputs/<system>/<song_id>/ containing notes.chart (or notes.mid) and
optionally song.ini. The special system "human" scores the reference charts against themselves
(useful for the playability baseline).

    python -m autochart.eval.harness <system> [<system> ...] [--split test]
    python -m autochart.eval.harness --compare sysA sysB ...   # markdown table of saved results

Reference audio time = chart time + song.ini delay + align_correction_ms (Phase 1).
Estimate audio time  = chart time + its own song.ini delay.
Expert is scored on every test song; Hard/Medium/Easy only where the human chart is a real
reduction (`real_<diff>`), since copies of Expert would reward not reducing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np
import pandas as pd
import yaml

from ..formats.loader import load_song
from ..formats.song import Song
from ..formats.textio import song_delay_s
from .metrics import Events, evaluate
from .playability import check
from ..paths import load_config

HEADLINE = ["F1@25", "F1@50", "P@50", "R@50", "fret_exact", "chord_agree", "contour", "sustain_acc",
            "hopo_f1", "tap_f1", "density_ratio", "pv_total"]


def tempo_scores(ref: Song, ref_off: float, est: Song, est_off: float, lo: float, hi: float, tol: float) -> dict:
    rb = ref.tempo.beat_times(hi - ref_off + 2) + ref_off
    eb = est.tempo.beat_times(hi - est_off + 2) + est_off
    rd = ref.tempo.downbeat_times(hi - ref_off + 2) + ref_off
    ed = est.tempo.downbeat_times(hi - est_off + 2) + est_off
    win = lambda x: x[(x >= lo) & (x <= hi)]  # noqa: E731
    rb, eb, rd, ed = win(rb), win(eb), win(rd), win(ed)
    if len(rb) < 2 or len(eb) < 2:
        return {"beat_f1": np.nan, "downbeat_f1": np.nan, "beat_f1_metrical": np.nan}
    f = mir_eval.beat.f_measure(rb, eb, tol)
    # allow half / double time (metrical-level errors)
    half = eb[::2]
    double = np.sort(np.concatenate([eb, (eb[:-1] + eb[1:]) / 2]))
    fm = max(f, mir_eval.beat.f_measure(rb, half, tol), mir_eval.beat.f_measure(rb, double, tol))
    fd = mir_eval.beat.f_measure(rd, ed, tol) if len(rd) >= 2 and len(ed) >= 2 else np.nan
    return {"beat_f1": f, "downbeat_f1": fd, "beat_f1_metrical": fm}


def score_system(system: str, cfg: dict, split: str) -> pd.DataFrame:
    root = Path(cfg["data_root"])
    songs = pd.read_parquet(root / "audit" / "songs.parquet")
    songs = songs[songs.split == split]
    out_root = Path(cfg["outputs_root"]) / system
    rows = []
    for s in songs.itertuples():
        ref, ref_ini, _ = load_song(s.path)
        ref_off = s.delay_s + s.align_correction_ms / 1000.0
        if system == "human":
            est, est_off = ref, ref_off
        else:
            folder = out_root / s.song_id
            try:
                est, est_ini, _ = load_song(folder)
                est_off = song_delay_s(est_ini)
            except (FileNotFoundError, OSError):
                rows.append({"song_id": s.song_id, "difficulty": "expert", "missing": True})
                continue
        audio_s = float(s.audio_s)
        first_tick_ref = None
        for d in cfg["difficulties"]:
            rt = ref.tracks.get(d)
            if rt is None or len(rt) == 0 or (d != "expert" and not getattr(s, f"real_{d}")):
                continue
            r_ev = Events.from_track(ref, rt, ref_off)
            et = est.tracks.get(d)
            e_ev = Events.from_track(est, et, est_off) if et is not None and len(et) else Events.empty()
            span = float(r_ev.t.max() - r_ev.t.min()) if len(r_ev) > 1 else audio_s
            row = {"song_id": s.song_id, "difficulty": d, "missing": False, "audio_s": audio_s}
            row.update(evaluate(r_ev, e_ev, span))
            row.update(check(e_ev, cfg["playability"][d], audio_s))
            if d == "expert":
                first_tick_ref = (float(r_ev.t.min()), float(r_ev.t.max()))
                row.update(tempo_scores(ref, ref_off, est, est_off, first_tick_ref[0] - 1.0,
                                        first_tick_ref[1] + 0.5, cfg["tempo"]["beat_tolerance_s"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    ev = root / "eval" / system
    ev.mkdir(parents=True, exist_ok=True)
    df.to_csv(ev / f"per_song_{split}.csv", index=False)
    summary = summarize(df)
    (ev / f"summary_{split}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return df


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    ok = df[~df.missing.astype(bool)] if "missing" in df else df
    out["missing_songs"] = int(df.missing.astype(bool).sum()) if "missing" in df else 0
    for d, g in ok.groupby("difficulty"):
        m = g.drop(columns=["song_id", "difficulty", "missing"], errors="ignore").mean(numeric_only=True)
        out[d] = {k: float(v) for k, v in m.items()}
        # ratios are skewed by a few sparse references; the median is the headline value
        out[d]["density_ratio"] = float(g.density_ratio.median())
        out[d]["songs"] = int(len(g))
    return out


def compare(systems: list[str], cfg: dict, split: str) -> str:
    root = Path(cfg["data_root"]) / "eval"
    lines = []
    for d in cfg["difficulties"]:
        cols = HEADLINE + (["beat_f1", "beat_f1_metrical", "downbeat_f1"] if d == "expert" else [])
        lines += [f"\n**{d.capitalize()}**\n", "| system | songs | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for sysname in systems:
            p = root / sysname / f"summary_{split}.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            if d not in s:
                continue
            v = s[d]
            cells = [f"{v.get(c, float('nan')):.3f}" for c in cols]
            lines.append(f"| {sysname} | {v['songs']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--split", default=None)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    split = args.split or cfg["split"]
    if args.compare:
        print(compare(args.systems, cfg, split))
        return
    for sysname in args.systems:
        df = score_system(sysname, cfg, split)
        print(sysname, json.dumps(summarize(df).get("expert", {}), default=float)[:600])


if __name__ == "__main__":
    main()
