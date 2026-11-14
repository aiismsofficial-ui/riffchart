"""Crash sweep: run the editor / export features on every chart in a library, reading only.

For each song folder: load it, the editor's JSON round trip, Easier from Expert, notes.mid (with the
lyrics as unpitched vocals), and a .sng (packed and read back on every Nth song). Outputs go to --out and are
deleted as it goes; the library is never written to. Failures are listed in --out/sweep.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from riffchart.formats.loader import find_chart_file, load_song  # noqa: E402
from riffchart.formats.midi_parser import parse_midi  # noqa: E402
from riffchart.formats.midi_writer import write_midi  # noqa: E402
from riffchart.paths import project_path  # noqa: E402
from riffchart.post.pipeline import regenerate_lower  # noqa: E402
from riffchart.product import vocals as voc  # noqa: E402
from riffchart.product.sng import read_sng, song_files, write_sng  # noqa: E402
from riffchart.ui.chartjson import json_to_song, song_to_json  # noqa: E402


def folders(root: Path):
    for ini in sorted(root.rglob("song.ini")):
        if find_chart_file(ini.parent):
            yield ini.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sng-every", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(project_path("configs/post.yaml"), encoding="utf-8"))
    fails, n, t0 = [], 0, time.time()
    for i, f in enumerate(folders(Path(a.library))):
        n += 1
        step = "load"
        try:
            song, ini, _ = load_song(f)
            step = "json"
            s2 = json_to_song(json.loads(json.dumps(song_to_json(song, ini))))
            step = "lower"
            if song.tracks.get("expert") is not None and len(song.tracks["expert"]):
                regenerate_lower(copy.deepcopy(s2), cfg)
            step = "midi"
            if abs(song.chart_offset_s) < 1e-6 and song.tracks.get("expert") is not None:
                notes, phrases = voc.vocal_notes(song, None)
                try:
                    p = write_midi(song, out / "x.mid", ini, vocals=(notes, phrases))
                    parse_midi(p)
                    p.unlink()
                except ValueError as e:                  # refused on purpose (e.g. a 1 BPM marker)
                    print(f"refused midi: {f.name}: {e}", flush=True)
            step = "sng"
            names = [x.name for x in song_files(f)]
            if i % a.sng_every == 0:
                p = write_sng(f, out / "x.sng", ini)
                meta, files = read_sng(p)
                bad = [x for x in names if files.get(x) != (f / x).read_bytes()]
                p.unlink()
                if bad:
                    raise AssertionError(f"sng contents differ: {bad}")
        except Exception as e:
            fails.append({"folder": str(f), "step": step, "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()[-1500:]})
            print(f"FAIL {step}: {f.name}: {type(e).__name__}: {e}", flush=True)
        if n % 25 == 0:
            print(f"{n} songs, {len(fails)} failures, {time.time() - t0:.0f}s", flush=True)
    json.dump(fails, open(out / "sweep.json", "w", encoding="utf-8"), indent=1)
    print(f"done: {n} songs, {len(fails)} failures, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
