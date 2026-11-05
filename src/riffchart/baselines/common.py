"""Shared helpers for baseline systems: test-song iteration, audio/stem loading, folder writing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..audio.io import load_audio, load_mix
from ..formats.chart_writer import write_chart_text
from ..formats.song import Song


def split_songs(data_root: str | Path, split: str = "test") -> pd.DataFrame:
    s = pd.read_parquet(Path(data_root) / "audit" / "songs.parquet")
    return s[s.split == split]


def mix_files(row) -> list[Path]:
    return [Path(row.path) / f for f in json.loads(row.audio_files)]


def load_full_mix(row, sr: int) -> np.ndarray:
    return load_mix(mix_files(row), sr)


def load_stem(data_root: str | Path, song_id: str, stem: str, sr: int, mono: bool = True) -> np.ndarray | None:
    p = Path(data_root) / "stems" / song_id / f"{stem}.ogg"
    if not p.exists():
        return None
    return load_audio(p, sr, mono)[0]


def write_song_folder(folder: Path, song: Song, name: str, artist: str, charter: str, audio_s: float,
                      delay_s: float = 0.0) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notes.chart").write_text(write_chart_text(song, name=name, artist=artist, charter=charter),
                                        encoding="utf-8")
    ini = ["[Song]", f"name = {name}", f"artist = {artist}", f"charter = {charter}", f"delay = {int(round(delay_s * 1000))}",
           f"song_length = {int(audio_s * 1000)}", "diff_guitar = -1"]
    (folder / "song.ini").write_text("\n".join(ini) + "\n", encoding="utf-8")
