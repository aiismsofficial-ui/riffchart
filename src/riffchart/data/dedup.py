"""Duplicate detection: same recording charted more than once.

Candidates are all pairs with similar audio duration. Each pair is scored by the best mean frame
cosine similarity of key-centred chroma sequences (1 frame/s) over lags of +-15 s; pairs above
`chroma_corr_min` are the same recording. Metadata (normalised artist + title) matches are reported
alongside so different recordings of one song can be seen too. Groups come from union-find; one
song per group is kept (most difficulties, then best alignment, then not under an "old" folder).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

MAX_LAG = 15


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"<[^>]+>", "", s)  # CH color tags
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    s = re.sub(r"\b(feat|ft|featuring)\b.*$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"^the ", "", s)


def _prep(fp: np.ndarray) -> np.ndarray:
    x = fp.astype(np.float32)
    x = x - x.mean(0, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def chroma_similarity(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG) -> float:
    s = a @ b.T
    best = -1.0
    for k in range(-max_lag, max_lag + 1):
        d = np.diagonal(s, offset=k)
        if len(d) >= 30:
            best = max(best, float(d.mean()))
    return best


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def find_duplicates(df: pd.DataFrame, chroma_dir: Path, dur_tol: float, thr: float,
                    frozen_eval: set[str] | None = None, prefer: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frozen_eval: song ids already in the frozen val/test split; they always win their group so
    the evaluation sets never change. Otherwise songs from libraries listed in `prefer` win."""
    df = df.reset_index(drop=True)
    fps = {}
    for i, sid in enumerate(df.song_id):
        p = chroma_dir / f"{sid}.npy"
        if p.exists():
            fps[i] = _prep(np.load(p))
    dur = df.audio_s.to_numpy()
    na = df.artist.map(norm_text).to_numpy()
    nt = df.name.map(norm_text).to_numpy()
    order = np.argsort(dur)
    pairs = []
    for ii, i in enumerate(order):
        for j in order[ii + 1 :]:
            if dur[j] - dur[i] > dur_tol:
                break
            meta = nt[i] == nt[j] and nt[i] != "" and (na[i] == na[j] or na[i] == "" or na[j] == "")
            sim = chroma_similarity(fps[i], fps[j]) if i in fps and j in fps else np.nan
            if meta or (sim == sim and sim >= thr * 0.8):
                pairs.append((int(i), int(j), sim, meta))
    # metadata matches outside the duration window (different recordings / edits)
    key = pd.Series([f"{a}|{t}" for a, t in zip(na, nt)])
    for _, idx in key.groupby(key).groups.items():
        idx = list(idx)
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = sorted((idx[x], idx[y]))
                if abs(dur[i] - dur[j]) > dur_tol:
                    sim = chroma_similarity(fps[i], fps[j]) if i in fps and j in fps else np.nan
                    pairs.append((i, j, sim, True))
    pdf = pd.DataFrame(pairs, columns=["i", "j", "chroma_sim", "meta_match"])
    pdf["same_recording"] = pdf.chroma_sim >= thr
    pdf["a"] = df.song_id.to_numpy()[pdf.i] if len(pdf) else []
    pdf["b"] = df.song_id.to_numpy()[pdf.j] if len(pdf) else []

    uf = _UF(len(df))
    for r in pdf[pdf.same_recording].itertuples():
        uf.union(r.i, r.j)
    groups = np.array([uf.find(i) for i in range(len(df))])
    n_diff = df[[f"n_{d}" for d in ("expert", "hard", "medium", "easy")]].fillna(0).gt(0).sum(1).to_numpy()
    old = df.path.str.lower().str.contains(r"[\\/]old ").to_numpy()
    lib = df.library.to_numpy() if "library" in df else np.array(["main"] * len(df))
    frozen = df.song_id.isin(frozen_eval or set()).to_numpy()
    score = n_diff * 10 + df.hit50.fillna(0).to_numpy() - old * 5 + np.isin(lib, list(prefer or ())) * 50 + frozen * 1000
    keep = np.ones(len(df), bool)
    dup_of = np.array([""] * len(df), dtype=object)
    for g in np.unique(groups):
        members = np.where(groups == g)[0]
        if len(members) > 1:
            best = members[np.argmax(score[members])]
            for m in members:
                if m != best:
                    keep[m] = False
                    dup_of[m] = df.song_id[best]
    out = pd.DataFrame({"song_id": df.song_id, "dup_group": groups, "dup_keep": keep, "dup_of": dup_of,
                        "song_key": key})
    return out, pdf


def run_dedup(cfg: dict) -> None:
    root = Path(cfg["data_root"]) / "audit"
    df = pd.read_parquet(root / "audit_raw.parquet")
    ok = df[df.status == "ok"].copy()
    d = cfg["dedup"]
    prev = Path(cfg["data_root"]) / "splits" / "split.parquet"
    frozen = set()
    if prev.exists():
        p = pd.read_parquet(prev)
        frozen = set(p.song_id[p.split.isin(["val", "test"])])
    prefer = {lib["tag"] for lib in cfg.get("extra_libraries") or [] if lib.get("prefer")}
    res, pairs = find_duplicates(ok, root / "chroma", d["duration_tol_s"], d["chroma_corr_min"], frozen, prefer)
    res.to_parquet(root / "dedup.parquet")
    pairs.to_csv(root / "dedup_pairs.csv", index=False)
    n_dup = int((~res.dup_keep).sum())
    print(f"pairs examined: {len(pairs)}  same-recording pairs: {int(pairs.same_recording.sum()) if len(pairs) else 0}"
          f"  songs dropped as duplicates: {n_dup}")
