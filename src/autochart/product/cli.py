"""autochart: chart songs for Clone Hero / YARG.

    autochart "Artist - Song.mp3" --out "D:/Clone Hero/Songs/autochart"
    autochart "D:/music/album" --out <songs folder> --difficulties expert,hard --style density=-0.3,chords=low

Inputs are audio files or folders (searched recursively). Each song becomes <out>/<Artist - Title>/
with notes.chart, song.ini, song.ogg and guitar.ogg. Songs whose folder already has a notes.chart are
skipped (re-run a batch to resume it), unless --overwrite.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

from ..formats.song import DIFFICULTIES
from ..paths import PROJECT
from .meta import GENERIC

AUDIO_EXT = {".mp3", ".ogg", ".opus", ".flac", ".wav"}       # what libsndfile decodes
STAGE_LABEL = {"load": "loading", "separate": "separating guitar", "beats": "tracking beats",
               "chart": "charting", "post": "polishing", "write": "writing"}


def find_audio(inputs: list[str]) -> tuple[list[Path], list[Path]]:
    files, unsupported = [], []
    for s in inputs:
        p = Path(s)
        cands = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in cands:
            if not f.is_file():
                continue
            if f.suffix.lower() in AUDIO_EXT:
                files.append(f)
            elif p.is_file() or f.suffix.lower() in {".m4a", ".aac", ".wma", ".mp4"}:
                unsupported.append(f)
    return files, unsupported


def display_name(f: Path) -> str:
    return f.parent.name if f.stem.lower() in GENERIC else f.stem


def parse_diffs(spec: str) -> list[str]:
    if spec == "all":
        return list(DIFFICULTIES)
    out = [d.strip().lower() for d in spec.split(",") if d.strip()]
    bad = [d for d in out if d not in DIFFICULTIES]
    if bad or not out:
        raise argparse.ArgumentTypeError(f"difficulties: 'all' or a comma list of {', '.join(DIFFICULTIES)}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autochart", description="Chart songs for Clone Hero / YARG (5-fret guitar).")
    ap.add_argument("inputs", nargs="+", help="audio files or folders")
    ap.add_argument("--out", required=True, help="songs folder to write into (e.g. your Clone Hero Songs folder)")
    ap.add_argument("--difficulties", type=parse_diffs, default=list(DIFFICULTIES), help="all (default) or e.g. expert,hard")
    ap.add_argument("--style", default="", help="density=-1..1 (sparser/denser), chords=auto|low|mid|high")
    ap.add_argument("--no-stems", action="store_true", help="write the full mix as song.ogg, no separate guitar.ogg")
    ap.add_argument("--overwrite", action="store_true", help="re-chart songs that already have a folder")
    ap.add_argument("--config", default="configs/product.yaml")
    args = ap.parse_args(argv)

    from .core import Style
    try:
        style = Style.parse(args.style)
    except ValueError as e:
        ap.error(str(e))
    files, unsupported = find_audio(args.inputs)
    for f in unsupported:
        print(f"skipping {f.name}: unsupported format (convert to mp3/ogg/flac/wav)", file=sys.stderr)
    if not files:
        print("no audio files found", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT / "logs" / "autochart.log"
    log_path.parent.mkdir(exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")

    def note(msg: str) -> None:
        tqdm.write(msg)
        log.write(time.strftime("%H:%M:%S ") + msg + "\n"); log.flush()

    print("loading models...", flush=True)
    from .core import AutoCharter
    ac = AutoCharter(args.config)
    ok = skipped = failed = 0
    t_all = time.perf_counter()
    bar = tqdm(files, unit="song", dynamic_ncols=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for i, f in enumerate(bar):
        def on_stage(name, f=f, i=i):
            bar.set_description(f"{display_name(f)[:40]} | {STAGE_LABEL[name]}")
            log.write(time.strftime("%H:%M:%S ") + f"[{i + 1}/{len(files)}] {display_name(f)[:60]}: "
                      f"{STAGE_LABEL[name]}\n")
            log.flush()
        try:
            r = ac.chart_file(f, out, args.difficulties, style, write_stems=not args.no_stems,
                              overwrite=args.overwrite, on_stage=on_stage)
        except Exception as e:                                    # one bad file must not stop a batch
            failed += 1
            note(f"FAILED {display_name(f)}: {e}")
            log.write(traceback.format_exc()); log.flush()
            continue
        if r is None:
            skipped += 1
            note(f"skip   {display_name(f)} (already charted, use --overwrite)")
            continue
        ok += 1
        notes = " ".join(f"{d[0].upper()}{r.n_notes[d]}" for d in DIFFICULTIES if d in r.n_notes)
        note(f"done   {r.folder.name}  {sum(r.seconds.values()):.0f}s  ({notes}, diff {r.diff_guitar})")
    bar.close()
    note(f"finished: {ok} charted, {skipped} skipped, {failed} failed in {time.perf_counter() - t_all:.0f}s -> {out}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
