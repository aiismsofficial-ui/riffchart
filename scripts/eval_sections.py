"""Score automatic sections (product/structure.py) against human charts that have sections and stems.

    python scripts/eval_sections.py <songs.json> [--split dev|test] [--n 40]

songs.json: [[folder, n_sections], ...] (song folders with vocals.ogg and named sections; read only).
Boundaries: precision / recall / F1 of section starts within +-1 bar. Labels: share of bars whose category
(intro, verse, prechorus, chorus, bridge, solo, instrumental, outro) matches the human chart's. Baselines: a
boundary every 8 bars, and every bar "verse". Bar features are cached in data/cache/sections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from riffchart.audio.io import load_audio  # noqa: E402
from riffchart.formats.loader import load_song  # noqa: E402
from riffchart.paths import PROJECT  # noqa: E402
from riffchart.product import structure as st  # noqa: E402

CACHE = PROJECT / "data/cache/sections"
SR = 22050


def features(folder: Path):
    key = hashlib.md5(str(folder).encode()).hexdigest()[:16]
    p = CACHE / f"{key}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    song, _, _ = load_song(folder)
    stems = [f for f in folder.iterdir() if f.suffix.lower() in (".ogg", ".opus", ".mp3")
             and not f.stem.lower().startswith("preview") and not f.stem.lower().startswith("crowd")]
    mix, voc, gtr = None, None, None
    for f in stems:
        a, _ = load_audio(f, SR, mono=True)
        mix = a if mix is None else (np.pad(mix, (0, max(0, len(a) - len(mix)))) +
                                     np.pad(a, (0, max(0, len(mix) - len(a)))))
        if f.stem.lower() == "vocals":
            voc = a
        if f.stem.lower() == "guitar":
            gtr = a
    end_s = len(mix) / SR
    ticks, secs = st.bar_times(song, end_s)
    keep = secs < end_s - 1.0
    ticks, secs = ticks[keep], secs[keep]
    f = st.bar_features(mix, SR, secs, end_s, voc, gtr, None)
    # human sections -> bar index of their start
    human = []
    for t, name in song.sections:
        i = int(np.searchsorted(ticks, t, side="right") - 1)
        if 0 <= i < len(ticks) and (not human or human[-1][0] != i):
            human.append((i, name))
    out = (f, ticks, secs, human)
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(out))
    return out


def bar_labels(starts_names, n):
    lab = [""] * n
    s = sorted(starts_names)
    for k, (i, name) in enumerate(s):
        j = s[k + 1][0] if k + 1 < len(s) else n
        for b in range(max(i, 0), min(j, n)):
            lab[b] = st.category(name)
    first = s[0][0] if s else n
    for b in range(min(first, n)):
        lab[b] = "intro"
    return lab


def prf(pred, ref, tol=1):
    pred, ref = [p for p in pred if p > 0], [r for r in ref if r > 0]
    hit_p = sum(any(abs(p - r) <= tol for r in ref) for p in pred)
    hit_r = sum(any(abs(p - r) <= tol for p in pred) for r in ref)
    P = hit_p / max(len(pred), 1)
    R = hit_r / max(len(ref), 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("songs")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    rows = json.load(open(a.songs))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rows))
    pick = order[: a.n] if a.split == "dev" else order[200: 200 + a.n]
    res = {"ours": [], "every8": [], "lab": [], "lab_verse": []}
    for k, i in enumerate(pick):
        folder = Path(rows[i][0])
        try:
            f, ticks, secs, human = features(folder)
        except Exception as e:
            print("skip", folder.name[:50], e)
            continue
        n = len(ticks)
        starts = st.boundaries(f)
        names = st.name_sections(f, starts, None, secs)
        ref = [b for b, _ in human]
        res["ours"].append(prf(starts, ref))
        res["every8"].append(prf(list(range(0, n, 8)), ref))
        hl = bar_labels(human, n)
        pl = bar_labels(list(zip(starts, names)), n)
        res["lab"].append(np.mean([x == y for x, y in zip(hl, pl)]))
        res["lab_verse"].append(np.mean([x == "verse" for x in hl]))
        if k < a.show:
            print(folder.name[:60])
            print("  human:", [(b, nm) for b, nm in human][:24])
            print("  ours :", list(zip(starts, names))[:24])
    o, e = np.array(res["ours"]), np.array(res["every8"])
    print(f"{a.split} songs {len(o)}  boundaries P/R/F ours {o.mean(0).round(3)}  every-8-bars {e.mean(0).round(3)}")
    print(f"labels per bar: ours {np.mean(res['lab']):.3f}  all-verse {np.mean(res['lab_verse']):.3f}")


if __name__ == "__main__":
    main()
