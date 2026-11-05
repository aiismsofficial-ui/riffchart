"""Apply post-processing to every chart of a system's output folder and write a new system.

    python -m autochart.post.cli <system> <new_system> [--split test] [--set key=value ...]
e.g. --set lower=reduce --set repeats.enabled=true --set repeats.thr=0.9 --set fix=false
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from ..baselines.common import split_songs, write_song_folder
from ..formats.loader import load_song
from ..formats.textio import song_delay_s
from ..model.dataset import Song as SongData
from .pipeline import postprocess


def set_key(cfg: dict, item: str) -> None:
    key, val = item.split("=", 1)
    *path, last = key.split(".")
    d = cfg
    for k in path:
        d = d.setdefault(k, {})
    d[last] = yaml.safe_load(val)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("new_system")
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default="configs/post.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--grid", default="ft", help="beat grid the chart was generated on (repeats)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for item in args.set:
        set_key(cfg, item)
    src, dst = Path("data/outputs") / args.system, Path("data/outputs") / args.new_system
    for r in tqdm(list(split_songs("data", args.split).itertuples()), desc=args.new_system, mininterval=10):
        if not (src / r.song_id / "notes.chart").exists():
            continue
        song, ini, _ = load_song(src / r.song_id)
        sd = SongData(r.song_id)
        beats, downs = sd.grids[args.grid]
        lower = [d for d in ("hard", "medium", "easy") if getattr(r, f"real_{d}")]
        song = postprocess(song, cfg, audio_s=float(r.audio_s), mel=sd.mel, beats=beats, downbeats=downs,
                           lower=lower)
        write_song_folder(dst / r.song_id, song, str(r.name), str(r.artist), "autochart", float(r.audio_s),
                          song_delay_s(ini))


if __name__ == "__main__":
    main()
