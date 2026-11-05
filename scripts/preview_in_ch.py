"""Open a generated chart in Clone Hero with a bot player for a quick visual check.

Copies the chart folder plus the song's audio into a preview folder (generated charts don't carry
audio), then runs the documented CH preview command line
(https://wiki.clonehero.net/books/clone-hero-manual/page/chart-previews-via-the-command-line):

    "Clone Hero.exe" --song "<folder>" --player Guitar,<Difficulty> [-i]

Usage:
    python scripts/preview_in_ch.py <chart_folder | system/song_id> [--difficulty expert]
                                    [--exe "C:/.../Clone Hero.exe"] [--audio <song folder>] [--dry-run]
The exe path can also come from configs/preview.yaml (`clone_hero_exe`) or $CLONE_HERO_EXE.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml

AUDIO_EXTS = (".ogg", ".opus", ".mp3", ".wav")


def find_exe(cli: str | None) -> Path | None:
    from riffchart.product.clonehero import find_exe as _find
    return _find(cli)


def resolve_chart(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    p = Path("data/outputs") / arg
    if p.is_dir():
        return p
    raise SystemExit(f"chart folder not found: {arg}")


def source_audio(chart: Path, audio: str | None) -> list[Path]:
    if audio:
        return [f for f in Path(audio).iterdir() if f.suffix.lower() in AUDIO_EXTS]
    songs = pd.read_parquet("data/audit/songs.parquet")
    row = songs[songs.song_id == chart.name]
    if row.empty:
        raise SystemExit("cannot infer the source song; pass --audio <song folder>")
    r = row.iloc[0]
    return [Path(r.path) / f for f in json.loads(r.audio_files)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chart")
    ap.add_argument("--difficulty", default="expert", choices=["expert", "hard", "medium", "easy"])
    ap.add_argument("--exe")
    ap.add_argument("--audio")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    chart = resolve_chart(args.chart)
    dest = Path("data/preview") / f"{chart.parent.name}_{chart.name}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(chart, dest, ignore=shutil.ignore_patterns("tokens.json"))
    for f in source_audio(chart, args.audio):  # keep stem names (song/guitar/rhythm...) so CH mixes them
        shutil.copy2(f, dest / f.name.lower())
    cmd = ["--song", str(dest.resolve()), "--player", f"Guitar,{args.difficulty.capitalize()}", "-i"]
    exe = find_exe(args.exe)
    if exe is None or args.dry_run:
        print("Preview folder:", dest.resolve())
        print("Command:", " ".join(f'"{c}"' if " " in c else c for c in ["Clone Hero.exe"] + cmd))
        if exe is None:
            print("Clone Hero.exe not found: pass --exe or set clone_hero_exe in configs/preview.yaml")
        return
    subprocess.Popen([str(exe)] + cmd, cwd=str(exe.parent))
    print("Launched", exe, "with", dest.resolve())


if __name__ == "__main__":
    main()
