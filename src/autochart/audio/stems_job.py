"""Batch Demucs separation over the usable songs, resumable.

Every song is separated from its full mix, including songs that ship a guitar.ogg: at inference
time the model only ever sees Demucs output, so training on Demucs stems avoids a train/test
mismatch. Shipped stems stay in the library as references.

Output per song: data/stems/<song_id>/{guitar,other}.ogg (stereo, model sample rate), where
other = every non-guitar/bass/drums/vocals source (other + piano for htdemucs_6s).
Files are written to *.tmp and renamed, so an interrupted run resumes cleanly.

    python -m autochart.audio.stems_job [--limit N] [--splits train,val,test]   (default: every split)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from .io import load_mix
from .separate import Separator
from ..paths import load_config

KEEP_AS_IS = ("guitar",)
RHYTHM_SECTION = ("bass", "drums", "vocals")


def _write(path: Path, data: np.ndarray, sr: int) -> None:
    tmp = path.with_suffix(".tmp.ogg")
    # libsndfile 1.2.2 segfaults on large single OGG/Vorbis writes; write in blocks instead.
    with sf.SoundFile(str(tmp), "w", sr, data.shape[1], format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(data), 32768):
            f.write(data[i : i + 32768])
    os.replace(tmp, path)


def _write_song(d: Path, guitar: np.ndarray, other: np.ndarray, sr: int) -> None:
    _write(d / "guitar.ogg", guitar, sr)
    _write(d / "other.ogg", other, sr)  # written last: its existence marks the song complete


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--splits", default="", help="comma list; default: all (data.split.split_names)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = Path(cfg["data_root"])
    songs = pd.read_parquet(root / "audit" / "songs.parquet")
    from ..data.split import split_names
    songs = songs[songs.split.isin(args.splits.split(",") if args.splits else split_names(cfg))]
    # test split first so evaluation can start early, then the rest
    songs = songs.assign(_o=songs.split.map({"test": 0, "val": 1, "train": 2})).sort_values(["_o", "song_id"])
    out_root = root / "stems"
    todo = [r for r in songs.itertuples() if not (out_root / r.song_id / "other.ogg").exists()]
    if args.limit:
        todo = todo[: args.limit]
    st = cfg["stems"]
    print(f"{len(songs)} songs in splits, {len(todo)} to separate with {st['model']}", flush=True)
    if not todo:
        return
    sep = Separator(st["model"], overlap=st["overlap"], shifts=st["shifts"], segment=st.get("segment"))
    manifest = open(out_root.joinpath("manifest.jsonl") if out_root.exists() else _mk(out_root), "a", encoding="utf-8")

    def load(r):
        files = [Path(r.path) / f for f in json.loads(r.audio_files)]
        return r, load_mix(files, sep.sr, mono=False)

    t0 = time.time()
    # One writer thread: each song's stems are written in a single task (guitar, then other.ogg as the
    # completion marker), and at most 3 songs may queue so stems never pile up in RAM.
    with ThreadPoolExecutor(2) as loader, ThreadPoolExecutor(1) as writer:
        futs = [loader.submit(load, r) for r in todo[:2]]
        nxt = 2
        pending = []
        for k in range(len(todo)):
            r, mix = futs[k].result()
            if nxt < len(todo):
                futs.append(loader.submit(load, todo[nxt]))
                nxt += 1
            futs[k] = None
            if mix.ndim == 1 or mix.shape[1] == 1:
                mix = np.repeat(mix.reshape(-1, 1), 2, 1)
            t = time.time()
            stems = sep(mix)
            gpu_s = time.time() - t
            other = sum(v for n, v in stems.items() if n not in KEEP_AS_IS + RHYTHM_SECTION)
            d = out_root / r.song_id
            d.mkdir(parents=True, exist_ok=True)
            guitar = stems["guitar"] if "guitar" in stems else other
            del stems
            pending.append(writer.submit(_write_song, d, guitar, other, sep.sr))
            while len(pending) > 3:
                pending.pop(0).result()
            manifest.write(json.dumps({"song_id": r.song_id, "model": st["model"], "gpu_s": round(gpu_s, 2),
                                       "audio_s": round(len(mix) / sep.sr, 1)}) + "\n")
            manifest.flush()
            el = time.time() - t0
            eta = el / (k + 1) * (len(todo) - k - 1)
            print(f"[{k + 1}/{len(todo)}] {r.song_id} gpu {gpu_s:.1f}s  elapsed {el / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)
        for p in pending:
            p.result()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p / "manifest.jsonl"


if __name__ == "__main__":
    main()
