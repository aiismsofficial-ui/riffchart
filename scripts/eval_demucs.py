"""Pick the Demucs model for guitar stems using songs that ship a real guitar.ogg.

mix = sum of all shipped mix stems, reference = guitar.ogg. Scores SDR (dB) of candidate guitar
estimates and wall time per song. Usage: python scripts/eval_demucs.py [--n 12]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import yaml

from autochart.audio.io import load_audio, load_mix
from autochart.audio.separate import Separator
from autochart.data.scan import scan_library
from autochart.paths import load_config


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    return float(10 * np.log10((ref**2).sum() / (((ref - est) ** 2).sum() + 1e-9) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    songs = [s for s in scan_library(cfg["library_root"]) if s.has_guitar_stem and len(s.mix_files) >= 2]
    rng = np.random.default_rng(0)
    songs = [songs[i] for i in rng.choice(len(songs), min(args.n, len(songs)), replace=False)]
    rows = []
    for model_name in ("htdemucs_6s", "htdemucs_ft"):
        sep = Separator(model_name)
        for s in songs:
            mix = load_mix(s.mix_files, sep.sr, mono=False)
            ref = load_audio(s.audio["guitar"], sep.sr, mono=False)[0]
            if ref.shape[1] == 1:
                ref = np.repeat(ref, 2, 1)
            torch.cuda.synchronize()
            t = time.perf_counter()
            st = sep(mix)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            r = {"song": s.path.name, "model": model_name, "sec": dt, "audio_s": len(mix) / sep.sr}
            if "guitar" in st:
                r["sdr_guitar"] = sdr(ref, st["guitar"])
                r["sdr_guitar+other"] = sdr(ref, st["guitar"] + st["other"])
                r["sdr_guitar+other+piano"] = sdr(ref, st["guitar"] + st["other"] + st["piano"])
            else:
                r["sdr_other"] = sdr(ref, st["other"])
            r["sdr_mix_baseline"] = sdr(ref, mix)
            rows.append(r)
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        del sep
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(df.groupby("model").mean(numeric_only=True).round(2).to_string())
    df.to_csv("logs/eval_demucs.csv", index=False)


if __name__ == "__main__":
    main()
