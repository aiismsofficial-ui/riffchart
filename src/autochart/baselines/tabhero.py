"""Tab Hero baseline (MattGroho/tab-hero, MIT), run from its own source tree.

Input is our cached htdemucs_6s guitar stem, matching Tab Hero's training preprocessing
(docs/data_format.md). The checkpoint is a pickle: it is loaded with torch.load(weights_only=True)
only (it has no non-default globals), never with the repo's weights_only=False loader. Tab Hero
outputs millisecond timestamps with no tempo map and its decoder drops HOPO/tap modifiers, so the
written chart uses a constant 120 BPM grid (480 ticks/beat) and CH natural note types; its tempo
metrics are not meaningful and are reported as n/a.

    python -m autochart.baselines.tabhero [--split test] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from ..formats.hopo import build_track, chart_hopo_threshold
from ..formats.song import DIFFICULTIES, Song, TempoMap
from .common import load_stem, split_songs, write_song_folder
from ..paths import load_config

SRC = Path("data/third_party/tab-hero-src/src")
CKPT = Path("data/third_party/tab-hero/best_model.pt")
RES = 480


def load_generator(device: str = "cuda"):
    import types

    # tab_hero/__init__.py reads installed package metadata; register the package path without it.
    pkg = types.ModuleType("tab_hero")
    pkg.__path__ = [str((SRC / "tab_hero").resolve())]
    sys.modules.setdefault("tab_hero", pkg)
    from tab_hero.inference.generator import ChartGenerator
    from tab_hero.model.chart_transformer import ChartTransformer

    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    config = ck.get("config") or ChartGenerator._infer_config_from_state_dict(sd)
    model = ChartTransformer(**config)
    model.load_state_dict(sd)
    return ChartGenerator(model=model, device=device)


def generate(gen, wav22: np.ndarray, difficulty: str, temperature: float = 1.0) -> list[dict]:
    """Mirror of ChartGenerator.generate but fed with an in-memory waveform."""
    w = torch.from_numpy(wav22).unsqueeze(0).float()
    w = w / (w.abs().max() + 1e-8)
    dur_ms = w.shape[-1] / gen.audio_processor.sample_rate * 1000.0
    codes = gen.audio_processor.encode(w).unsqueeze(0)
    tok = gen.tokenizer
    ranges = {"time": (tok._time_token_start, tok._time_token_end), "fret": (tok._fret_token_start, tok._fret_token_end),
              "mod": (tok._mod_token_start, tok._mod_token_end), "dur": (tok._dur_token_start, tok._dur_token_end)}
    diff_id = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}[difficulty]
    with torch.no_grad():
        out = gen.model.generate(codes, difficulty_id=torch.tensor([diff_id], device=gen.device),
                                 instrument_id=torch.tensor([0], device=gen.device), temperature=temperature,
                                 top_k=50, top_p=0.95, vocab_ranges=ranges,
                                 audio_frame_rate=gen.audio_processor.frame_rate)
    notes = gen._decode_tokens(out[0].cpu().tolist())
    return [n for n in notes if n["timestamp_ms"] <= dur_ms]


def notes_to_song(by_diff: dict[str, list[dict]]) -> Song:
    tm = TempoMap(RES, np.array([0]), np.array([120.0]))
    thr = chart_hopo_threshold(RES)
    tracks = {}
    for d, notes in by_diff.items():
        raw = []
        for n in notes:
            tick = int(round(float(tm.sec_to_tick(n["timestamp_ms"] / 1000.0))))
            sus = int(round(n["duration_ms"] / 1000.0 * 2 * RES)) if n["duration_ms"] else 0
            sus = sus if sus >= RES // 3 else 0  # Tab Hero exports .mid, where CH cuts sustains < res/3
            for f in n["frets"]:
                raw.append((tick, int(f), sus))  # fret 5 = open, same bit as our mask
        tracks[d] = build_track(d, raw, thr, midi=False)
    return Song(RES, tm, tracks, source_format="chart")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default="tabhero")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = Path(cfg["data_root"])
    out = Path(cfg["outputs_root"]) / args.name
    gen = load_generator()
    torch.manual_seed(0)
    for r in tqdm(list(split_songs(root, args.split).itertuples())[: args.limit], desc=args.name):
        if (out / r.song_id / "notes.chart").exists():
            continue
        wav = load_stem(root, r.song_id, "guitar", gen.audio_processor.sample_rate)
        # ~30 s per difficulty: lower difficulties only where the human chart is a real reduction
        # (the only songs the harness scores them on)
        diffs = [d for d in DIFFICULTIES if d == "expert" or getattr(r, f"real_{d}")]
        song = notes_to_song({d: generate(gen, wav, d) for d in diffs})
        write_song_folder(out / r.song_id, song, str(r.name), str(r.artist), args.name, float(r.audio_s))


if __name__ == "__main__":
    main()
