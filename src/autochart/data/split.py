"""Usability flags + artist-disjoint train/val/test split.

Alignment offset per song: total_off = lag_ms + hit_med_off_ms - delay_ms (audio onset minus chart
note, after song.ini delay). The library median of total_off is onset-detector latency, not chart
error, so deviations are measured from it. Constant offsets are corrected (stored as
`align_correction_ms`, to be added to note times); songs with drift or poor fit are dropped.

The test split is frozen: once data/splits/split.parquet exists every song already assigned keeps its
split (unless --force). New songs from an extra library with tag T: artists already in train -> train;
artists in val/test -> val_T/test_T (never train, so no artist leaks); unseen artists are split
artist-disjointly into train / val_T / test_T (`val_songs` / `test_songs` of that library's entry in
`extra_libraries`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dedup import norm_text

BASE_SPLITS = ("train", "val", "test")
CORRECT_MIN_MS = 25.0      # only correct offsets beyond the estimation noise
CORRECT_MAX_MS = 250.0     # beyond this the estimate is ambiguous (beat periodicity) -> drop
# Charts made by other autocharters must never become training targets.
AUTO_CHARTER_RE = r"herochart ai|autochart|auto-chart|auto chart|tab hero|tabhero|fretformer|audio2hero|audio2chart|\bai\b"


def split_names(cfg: dict) -> list[str]:
    """train / val / test plus the held-out val_<tag> / test_<tag> of every extra library."""
    return list(BASE_SPLITS) + [f"{p}_{lib['tag']}" for lib in cfg.get("extra_libraries") or [] for p in ("val", "test")]


def is_live(split: pd.Series) -> pd.Series:
    """Songs that get stems / features: every split except "dropped"."""
    return split.notna() & (split != "dropped")


def flag_songs(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    a = cfg["audit"]
    df = df.copy()
    df["total_off_ms"] = df.lag_ms + df.hit_med_off_ms.fillna(0) - df.delay_s * 1000
    ok = df.status == "ok"
    bias = float(df.loc[ok & (df.hit50 > 0.7), "total_off_ms"].median())
    df["detector_bias_ms"] = bias
    df["off_dev_ms"] = df.total_off_ms - bias
    reasons = pd.Series([[] for _ in range(len(df))], index=df.index)

    def add(mask, why):
        for i in df.index[mask.fillna(False)]:
            reasons[i].append(why)

    add(df.status != "ok", "status")
    add(ok & (df.n_expert.fillna(0) == 0), "no_expert")
    add(ok & (df.hit50 < a["min_hit50"]), "poor_fit")
    add(ok & (df.local_ratio < a["min_local_ratio"]) & (df.hit50 < 0.7), "ambiguous_alignment")
    add(ok & (df.off_dev_ms.abs() > CORRECT_MAX_MS), "large_offset")
    add(ok & (df.drift_range_ms > a["max_drift_ms"]), "tempo_drift")
    add(ok & (df.chart_end_past_audio_s > 2.0), "chart_past_audio")
    charter = df.get("charter", pd.Series("", index=df.index)).fillna("").str.lower()
    add(charter.str.contains(AUTO_CHARTER_RE), "auto_generated_chart")
    add(ok & ((df.bpm_min < a["min_bpm"]) | (df.bpm_max > a["max_bpm"])) & (df.hit50 < 0.7), "broken_tempo")
    df["drop_reasons"] = reasons.map(lambda r: ",".join(r))
    # Lower difficulties that are (near-)copies of Expert carry no reduction signal; >=98% of the
    # Expert note count is treated as a copy (one charter copies Expert into every difficulty).
    # <15% of Expert means an unfinished chart (e.g. 26 Hard notes vs 1700 Expert), also not real.
    for d in ("hard", "medium", "easy"):
        n = df.get(f"n_{d}", pd.Series(0, index=df.index)).fillna(0)
        ne = df.get("n_expert", pd.Series(0, index=df.index)).fillna(0)
        df[f"real_{d}"] = (n >= 0.15 * ne) & (n > 0) & (n < 0.98 * ne)
    df["usable"] = reasons.map(len) == 0
    corr = (df.off_dev_ms.abs() > CORRECT_MIN_MS) & df.usable
    # lag = audio - chart, so notes move by +deviation to land on the audio
    df["align_correction_ms"] = np.where(corr, df.off_dev_ms, 0.0)
    return df


def primary_artist(a: str) -> str:
    n = norm_text(a)
    return n if n else "unknown"


def assign_split(df: pd.DataFrame, cfg: dict, prev: pd.DataFrame | None) -> pd.Series:
    s = cfg["split"]
    split = pd.Series("train", index=df.index)
    artist = df.artist.map(primary_artist)
    if prev is not None:
        return assign_new(df, cfg, prev, artist)
    rng = np.random.default_rng(s["seed"])
    usable = df.usable
    counts = artist[usable].value_counts()
    real = (df.real_easy & df.real_medium & df.real_hard)[usable].groupby(artist[usable]).sum()
    eligible = [a for a in counts.index if counts[a] <= s["max_artist_songs_eval"]]
    rng.shuffle(eligible)
    taken: set[str] = set()
    for name, target, real_target in (("test", s["test_songs"], s["test_real_reduction_songs"]),
                                      ("val", s["val_songs"], s["val_real_reduction_songs"])):
        n = n_real = 0
        # pass 1: artists with genuinely reduced E/M/H charts, so Phase 5 has human references
        for a in eligible:
            if n_real >= real_target or n >= target:
                break
            if a not in taken and real.get(a, 0) > 0:
                taken.add(a)
                split[artist == a] = name
                n += counts[a]
                n_real += real[a]
        # pass 2: random artists until the song target is met
        for a in eligible:
            if n >= target:
                break
            if a not in taken:
                taken.add(a)
                split[artist == a] = name
                n += counts[a]
    return split


def assign_new(df: pd.DataFrame, cfg: dict, prev: pd.DataFrame, artist: pd.Series) -> pd.Series:
    s = cfg["split"]
    old = dict(zip(prev.song_id, prev.split))
    split = df.song_id.map(old)
    live = prev[prev.split.isin(BASE_SPLITS)]
    known = dict(zip(live.artist_key, live.split))
    lib = df.library if "library" in df else pd.Series("main", index=df.index)
    held = lambda base, tag: base if tag == "main" or base == "train" else f"{base}_{tag}"   # noqa: E731
    new = split.isna()
    base = artist[new].map(known)
    split[new] = [held(b, t) if isinstance(b, str) else None for b, t in zip(base, lib[new])]
    # unseen artists: artist-disjoint draw per library, eligible artists have few songs (as for the original)
    fresh = new & split.isna()
    split[fresh] = "train"
    for entry in cfg.get("extra_libraries") or []:
        mine = fresh & (lib == entry["tag"])
        counts = artist[mine & df.usable].value_counts()
        eligible = [a for a in counts.index if counts[a] <= s["max_artist_songs_eval"]]
        np.random.default_rng(s["seed"] + 1).shuffle(eligible)
        for name, target in ((f"test_{entry['tag']}", entry.get("test_songs", 0)),
                             (f"val_{entry['tag']}", entry.get("val_songs", 0))):
            n = 0
            while eligible and n < target:
                a = eligible.pop()
                split[mine & (artist == a)] = name
                n += counts[a]
    return split.fillna("train")


def run_split(cfg: dict, force: bool = False) -> None:
    root = Path(cfg["data_root"])
    audit = pd.read_parquet(root / "audit" / "audit_raw.parquet")
    dedup = pd.read_parquet(root / "audit" / "dedup.parquet")
    df = audit.merge(dedup, on="song_id", how="left")
    df["dup_keep"] = df.dup_keep.fillna(True).astype(bool)
    df = flag_songs(df, cfg)
    dupe = ~df.dup_keep
    df.loc[dupe, "drop_reasons"] = df.loc[dupe, "drop_reasons"].map(lambda r: (r + ",duplicate").strip(","))
    df.loc[dupe, "usable"] = False
    df["artist_key"] = df.artist.fillna("").map(primary_artist)

    out = root / "splits"
    out.mkdir(parents=True, exist_ok=True)
    prev_path = out / "split.parquet"
    prev = pd.read_parquet(prev_path) if prev_path.exists() and not force else None
    df["split"] = assign_split(df, cfg, prev)
    df.loc[~df.usable, "split"] = "dropped"
    df.to_parquet(root / "audit" / "songs.parquet")
    df[["song_id", "path", "artist_key", "split", "align_correction_ms"]].to_parquet(prev_path)
    for name in split_names(cfg):
        sel = df[df.split == name]
        (out / f"{name}.txt").write_text("\n".join(f"{r.song_id}\t{r.path}" for r in sel.itertuples()) + "\n", encoding="utf-8")
    (out / "split_meta.json").write_text(json.dumps({"seed": cfg["split"]["seed"], "frozen_from_previous": prev is not None,
                                                     "detector_bias_ms": float(df.detector_bias_ms.iloc[0])}, indent=2))
    print(df.split.value_counts().to_string())
    train = set(df[df.split == "train"].artist_key)
    for name in split_names(cfg)[1:]:
        print(f"artist leakage {name}<->train:", len(set(df[df.split == name].artist_key) & train))
