"""Fit the diff_guitar estimator on human-rated charts and compare it with the notes-per-second baseline.

    python scripts/fit_difficulty.py                 # features (cached), cross-validated report, writes the model
    python scripts/fit_difficulty.py --no-write      # report only

Ratings come from the charts' own song.ini (0-6; -1 / missing / out of range skipped), usable de-duplicated
songs from the audit. Evaluation is artist-grouped 5-fold cross-validation, so no artist is in both halves.
Charts are only read."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from riffchart.formats.loader import load_song  # noqa: E402
from riffchart.product.difficulty import FEATURES, chart_features  # noqa: E402

CACHE = ROOT / "data" / "reports" / "difficulty_features.parquet"
MODEL = ROOT / "configs" / "difficulty.yaml"
OLD_CUTS = [2.6, 3.5, 4.05, 4.75, 5.8, 7.5]


def rated_songs() -> pd.DataFrame:
    s = pd.read_parquet(ROOT / "data" / "audit" / "songs.parquet")
    s["rating"] = pd.to_numeric(s.diff_guitar, errors="coerce")
    keep = s.usable & (s.dup_keep != False) & s.rating.between(0, 6)  # noqa: E712
    return s.loc[keep, ["song_id", "path", "artist_key", "library", "rating", "split"]].reset_index(drop=True)


def features(df: pd.DataFrame) -> pd.DataFrame:
    if CACHE.exists():
        c = pd.read_parquet(CACHE)
        if set(df.song_id) <= set(c.song_id) and set(FEATURES) <= set(c.columns):
            return df.merge(c[["song_id", *FEATURES]], on="song_id")
    rows, t0 = [], time.time()
    for i, r in enumerate(df.itertuples()):
        try:
            song, _, _ = load_song(Path(r.path))
            f = chart_features(song)
        except Exception as e:                       # unreadable chart: skip
            print(f"skip {r.path}: {e}", file=sys.stderr)
            f = None
        if f is not None:
            rows.append({"song_id": r.song_id, **f})
        if (i + 1) % 200 == 0:
            print(f"  features {i + 1}/{len(df)} ({time.time() - t0:.0f} s)", flush=True)
    c = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    c.to_parquet(CACHE)
    return df.merge(c, on="song_id")


def fit_linear(X: np.ndarray, y: np.ndarray, alpha: float = 3.0) -> dict:
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / std
    A = np.c_[np.ones(len(Z)), Z]
    reg = alpha * np.eye(A.shape[1])
    reg[0, 0] = 0
    w = np.linalg.solve(A.T @ A + reg, A.T @ y)
    s = A @ w
    return {"mean": mean, "std": std, "intercept": w[0], "weights": w[1:], "cuts": fit_cuts(s, y)}


def fit_cuts(score: np.ndarray, y: np.ndarray) -> list[float]:
    """Cut points between levels that maximise exact agreement (one cut at a time, kept increasing)."""
    cuts = [k + 0.5 for k in range(6)]
    for _ in range(3):
        for k in range(6):
            lo = cuts[k - 1] + 0.05 if k else score.min() - 1
            hi = cuts[k + 1] - 0.05 if k < 5 else score.max() + 1
            cand = np.linspace(lo, hi, 60)
            acc = [(np.searchsorted(np.array(cuts[:k] + [c] + cuts[k + 1:]), score) == y).mean() for c in cand]
            cuts[k] = float(cand[int(np.argmax(acc))])
    return cuts


def predict(m: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = m["intercept"] + ((X - m["mean"]) / m["std"]) @ m["weights"]
    return np.searchsorted(np.array(m["cuts"]), s), s


def report(name: str, y: np.ndarray, p: np.ndarray, s: np.ndarray) -> dict:
    d = {"exact": float((p == y).mean()), "within1": float((np.abs(p - y) <= 1).mean()),
         "mae": float(np.abs(p - y).mean()), "spearman": float(spearmanr(s, y).statistic),
         "bias": float((p - y).mean())}
    print(f"{name:28s} exact {d['exact']:.1%}  within 1 {d['within1']:.1%}  MAE {d['mae']:.2f}  "
          f"Spearman {d['spearman']:.2f}  bias {d['bias']:+.2f}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    df = features(rated_songs())
    X, y = df[list(FEATURES)].to_numpy(np.float64), df.rating.to_numpy(int)
    print(f"{len(df)} rated charts, {df.artist_key.nunique()} artists; ratings "
          + " ".join(f"{k}:{(y == k).sum()}" for k in range(7)))
    # artist-grouped 5 folds
    rng = np.random.default_rng(0)
    artists = df.artist_key.unique()
    fold_of = dict(zip(artists, rng.permutation(len(artists)) % 5))
    fold = df.artist_key.map(fold_of).to_numpy()
    pb, sb, pl, sl, pg, sg = (np.zeros(len(y)) for _ in range(6))
    from sklearn.ensemble import HistGradientBoostingRegressor
    for k in range(5):
        tr, te = fold != k, fold == k
        nps = X[te, FEATURES.index("nps")]
        pb[te], sb[te] = np.searchsorted(OLD_CUTS, nps), nps
        m = fit_linear(X[tr], y[tr])
        pl[te], sl[te] = predict(m, X[te])
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)
        g.fit(X[tr], y[tr])
        sg[te] = g.predict(X[te])
        pg[te] = np.clip(np.round(sg[te]), 0, 6)
    res = {"n": len(y), "baseline_nps": report("baseline (notes/s cuts)", y, pb, sb),
           "linear": report("linear, 12 features", y, pl, sl),
           "boosted_trees": report("gradient boosted trees", y, pg, sg)}
    for lib in sorted(df.library.unique()):
        sel = (df.library == lib).to_numpy()
        print(f"  [{lib}] n={sel.sum()}")
        report("    baseline", y[sel], pb[sel], sb[sel])
        report("    linear", y[sel], pl[sel], sl[sel])
    m = fit_linear(X, y)
    print("weights (per standard deviation):")
    for f, w in sorted(zip(FEATURES, m["weights"]), key=lambda t: -abs(t[1])):
        print(f"  {f:16s} {w:+.3f}")
    (ROOT / "data" / "reports" / "difficulty_fit.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not args.no_write:
        out = {"features": list(FEATURES), "mean": [round(float(v), 6) for v in m["mean"]],
               "std": [round(float(v), 6) for v in m["std"]], "intercept": round(float(m["intercept"]), 6),
               "weights": [round(float(v), 6) for v in m["weights"]], "cuts": [round(c, 4) for c in m["cuts"]]}
        MODEL.write_text("# diff_guitar estimator (product/difficulty.py), fitted by scripts/fit_difficulty.py on\n"
                         f"# {len(y)} human-rated charts. Linear score on standardised Expert-chart features, cut into 0-6.\n"
                         + yaml.safe_dump(out, sort_keys=False, width=120), encoding="utf-8")
        print(f"wrote {MODEL}")


if __name__ == "__main__":
    main()
