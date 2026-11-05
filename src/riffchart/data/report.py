"""Phase 1 dataset report: markdown tables + histograms in data/reports/phase1/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DIFFS = ("expert", "hard", "medium", "easy")


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def _genre(g: str) -> str:
    g = str(g or "").strip().lower()
    return g if g else "(none)"


def run_report(cfg: dict) -> dict:
    root = Path(cfg["data_root"])
    df = pd.read_parquet(root / "audit" / "songs.parquet")
    out = root / "reports" / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    use = df[df.usable]
    s: dict = {}
    s["folders"] = len(df)
    s["parsed"] = int((df.status != "parse_error").sum())
    s["audio_ok"] = int((df.status == "ok").sum())
    s["usable"] = len(use)
    s["hours_all"] = float(df.audio_s.fillna(0).sum() / 3600)
    s["hours_usable"] = float(use.audio_s.sum() / 3600)
    s["detector_bias_ms"] = float(df.detector_bias_ms.iloc[0])
    s["corrected"] = int((use.align_correction_ms != 0).sum())

    reasons = df.drop_reasons.fillna("").str.split(",").explode()
    reasons = reasons[reasons != ""]
    by_reason = reasons.value_counts().rename_axis("reason").reset_index(name="songs")
    primary = df.loc[~df.usable, "drop_reasons"].str.split(",").str[0].value_counts()
    by_reason["primary"] = by_reason.reason.map(primary).fillna(0).astype(int)

    diff_rows = []
    for d in DIFFS:
        real = use if d == "expert" else use[use[f"real_{d}"]]
        diff_rows.append({"difficulty": d, "all_songs": int((df[f"n_{d}"].fillna(0) > 0).sum()),
                          "usable_songs": int((use[f"n_{d}"].fillna(0) > 0).sum()),
                          "real_reductions": len(real) if d != "expert" else "-",
                          "real_notes": int(real[f"n_{d}"].fillna(0).sum()),
                          "median_nps": float(real[f"nps_{d}"].median()),
                          "chord_frac": float(real[f"chord_frac_{d}"].mean()),
                          "hopo_frac": float(real[f"hopo_frac_{d}"].mean()),
                          "tap_frac": float(real[f"tap_frac_{d}"].mean()),
                          "open_frac": float(real[f"open_frac_{d}"].mean()),
                          "sustain_frac": float(real[f"sustain_frac_{d}"].mean())})
    charters = use.charter.fillna("").str.replace(r"<[^>]+>", "", regex=True).str.strip().replace("", "(none)")
    charter_df = charters.value_counts().head(10).rename_axis("charter").reset_index(name="usable_songs")
    diff_df = pd.DataFrame(diff_rows)
    all4 = int(use[[f"n_{d}" for d in DIFFS]].fillna(0).gt(0).all(1).sum())

    split_rows = []
    from .split import split_names
    for name in split_names(cfg):
        x = df[df.split == name]
        split_rows.append({"split": name, "songs": len(x), "artists": x.artist_key.nunique(),
                           "hours": float(x.audio_s.sum() / 3600),
                           "real_E/M/H": int(x[["real_easy", "real_medium", "real_hard"]].all(1).sum()),
                           "guitar_stem": int(x.has_guitar_stem.sum())})
    split_df = pd.DataFrame(split_rows)

    fmt = use.format.value_counts().to_dict()
    dev = df.loc[df.status == "ok", "off_dev_ms"]
    align = {f"p{p}": float(np.nanpercentile(dev.abs(), p)) for p in (50, 75, 90, 95, 99)}
    s.update(format_usable=fmt, all_four_diffs_usable=all4, abs_offset_dev_ms=align,
             guitar_stem_usable=int(use.has_guitar_stem.sum()),
             median_hit50_usable=float(use.hit50.median()), median_drift_ms_usable=float(use.drift_range_ms.median()))

    # ---- plots
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    g = use.genre.map(_genre).value_counts().head(20)[::-1]
    ax[0, 0].barh(g.index, g.values, color="#4a7bd0")
    ax[0, 0].set_title("Genre (top 20, usable)")
    ax[0, 1].hist(use.bpm_median.clip(40, 280), bins=48, color="#4a7bd0")
    ax[0, 1].set_title("Median BPM (usable)")
    ax[0, 2].hist(use.audio_s / 60, bins=40, color="#4a7bd0")
    ax[0, 2].set_title("Song length (min)")
    data = [(use if d == "expert" else use[use[f"real_{d}"]])[f"nps_{d}"].dropna().clip(0, 20) for d in DIFFS]
    ax[1, 0].boxplot(data, tick_labels=[d.capitalize() for d in DIFFS], showfliers=False)
    ax[1, 0].set_title("Notes/sec by difficulty (real reductions only)")
    ax[1, 1].hist(use.nps_expert.clip(0, 20), bins=40, color="#4a7bd0")
    ax[1, 1].set_title("Expert notes/sec (usable)")
    ax[1, 2].hist(dev.clip(-150, 150), bins=60, color="#4a7bd0")
    ax[1, 2].set_title("Alignment deviation from library median (ms)")
    fig.tight_layout()
    fig.savefig(out / "histograms.png", dpi=110)
    plt.close(fig)

    md = [
        "# Phase 1 dataset report", "",
        f"- Song folders found: **{s['folders']}**, parsed OK: **{s['parsed']}**, audio OK: **{s['audio_ok']}**",
        f"- Usable: **{s['usable']}** songs, **{s['hours_usable']:.1f} h** of audio (of {s['hours_all']:.1f} h total)",
        f"- Usable by format: {fmt}; with a shipped guitar stem: {s['guitar_stem_usable']}",
        f"- Usable songs with all four difficulties: {all4}",
        f"- Onset-detector latency (library median offset): {s['detector_bias_ms']:.1f} ms; "
        f"songs with a constant-offset correction applied: {s['corrected']}",
        f"- |offset deviation| percentiles (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in align.items()),
        f"- Median notes-on-onset rate (±50 ms) for usable songs: {s['median_hit50_usable']:.2f}; median drift range {s['median_drift_ms_usable']:.1f} ms",
        "", "## Drop reasons (a song can have several)", "", _md(by_reason), "",
        "## Difficulties", "",
        "`real_reductions` = songs whose lower difficulty has <98% of the Expert note count; the rest are",
        "copies of Expert. Density/flag columns use Expert for expert and real reductions otherwise.", "",
        _md(diff_df), "", "## Top charters (usable songs)", "", _md(charter_df), "",
        "## Splits (artist-disjoint)", "", _md(split_df), "",
        "![histograms](histograms.png)", "",
    ]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print("\n".join(md))
    return s
