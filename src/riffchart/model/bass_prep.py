"""Bass training data (resumable, per song), next to the guitar features in data/features/model.

  data/features/bass/mel/<id>.npy     fp16 (T, 2, 128): log-mel of [Demucs bass stem, full mix], same settings
                                      as the guitar features (the mix channel is copied from model/mel)
  data/features/bass/notes/<id>.npz   per difficulty: onset/end seconds (audio time), mask, type
  data/features/bass/songs.parquet    the songs with a bass chart: song_id, split, library, real_* flags

The beat grids are shared with the guitar features (data/features/model/grid). Songs come from the audited
library (usable, live split) and need an Expert bass part of at least MIN_NOTES notes. A lower difficulty
counts as a real human reduction (`real_<diff>`) when it has under 98% of Expert's notes, as for guitar.

    python -m riffchart.model.bass_prep [--limit N]
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soxr
import torch

from ..audio.io import load_mix
from ..baselines.common import mix_files
from ..data.split import is_live
from ..formats.loader import load_song
from ..formats.song import DIFFICULTIES
from ..paths import PROJECT
from .data_prep import ROOT as GUITAR_ROOT
from .data_prep import SR, Mel, song_notes

ROOT = PROJECT / "data/features/bass"
MIN_NOTES = 50


def bass_rows() -> pd.DataFrame:
    s = pd.read_parquet(PROJECT / "data/audit/songs.parquet")
    return s[is_live(s.split)]


def main(argv=None) -> None:
    from ..audio.separate import Separator

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    for sub in ("mel", "notes"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    rows = bass_rows()
    if args.limit:
        rows = rows.head(args.limit)
    table_p = ROOT / "songs.parquet"
    done_rows = pd.read_parquet(table_p).to_dict("records") if table_p.exists() else []
    seen = {r["song_id"] for r in done_rows}
    skip_p = ROOT / "no_bass.txt"
    skipped = set(skip_p.read_text().split()) if skip_p.exists() else set()
    todo = [r for r in rows.itertuples() if r.song_id not in seen and r.song_id not in skipped]
    print(f"{len(rows)} live songs, {len(seen)} done, {len(skipped)} without bass, {len(todo)} to check", flush=True)

    sep = Separator(device="cuda")
    mel = Mel()
    q: queue.Queue = queue.Queue(maxsize=4)

    def reader():                                  # parse + decode on a thread while the GPU separates
        for r in todo:
            try:
                notes, song, _ = song_notes(r, instrument="bass")
                n_ex = len(notes.get("expert_on", []))
                if n_ex < MIN_NOTES:
                    q.put((r, None, None, None))
                    continue
                mix = load_mix(mix_files(r), sep.sr, mono=False)
                q.put((r, notes, n_ex, mix))
            except (ValueError, FileNotFoundError):  # no PART BASS / no bass section
                q.put((r, None, None, None))
            except Exception as e:
                print(f"  {r.song_id}: {type(e).__name__}: {e}", flush=True)
                q.put((r, None, None, None))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    t0, k, new = time.time(), 0, 0
    while (item := q.get()) is not None:
        r, notes, n_ex, mix = item
        k += 1
        if notes is None:
            skipped.add(r.song_id)
            if k % 50 == 0:
                skip_p.write_text("\n".join(sorted(skipped)))
            continue
        with torch.no_grad():
            bass = sep(mix.astype(np.float32))["bass"].mean(axis=1)
        bass = soxr.resample(bass, sep.sr, SR).astype(np.float32)
        gm = np.load(GUITAR_ROOT / "mel" / f"{r.song_id}.npy", mmap_mode="r")          # [guitar, mix]
        bm = mel(bass)
        T = min(len(bm), len(gm))
        np.save(ROOT / "mel" / f"{r.song_id}.npy", np.stack([bm[:T], np.asarray(gm[:T, 1], np.float32)], 1)
                .astype(np.float16))
        np.savez(ROOT / "notes" / f"{r.song_id}.npz", **notes)
        rec = {"song_id": r.song_id, "split": r.split, "library": getattr(r, "library", "main"),
               "chart_file": r.chart_file, "charter": r.charter, "n_expert": n_ex}
        for d in ("hard", "medium", "easy"):
            n = len(notes.get(f"{d}_on", []))
            rec[f"n_{d}"] = n
            rec[f"real_{d}"] = bool(0 < n < 0.98 * n_ex)
        done_rows.append(rec)
        new += 1
        if new % 25 == 0:
            pd.DataFrame(done_rows).to_parquet(table_p)
            skip_p.write_text("\n".join(sorted(skipped)))
        if k % 50 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(todo)} checked, {len(done_rows)} with bass  {el:.0f}s  "
                  f"(~{el / k * (len(todo) - k) / 60:.0f} min left)", flush=True)
    pd.DataFrame(done_rows).to_parquet(table_p)
    skip_p.write_text("\n".join(sorted(skipped)))
    df = pd.DataFrame(done_rows)
    print(f"done in {time.time() - t0:.0f}s: {len(df)} songs with bass; by split "
          f"{df.split.value_counts().to_dict()}; real hard/medium/easy "
          f"{int(df.real_hard.sum())}/{int(df.real_medium.sum())}/{int(df.real_easy.sum())}", flush=True)


if __name__ == "__main__":
    main()
