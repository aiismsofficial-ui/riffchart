"""Signal-processing baseline (no learning): the floor every model must beat.

Expert:
  tempo     Beat This beats/downbeats on the full mix -> tempo map (one BPM per beat)
  onsets    librosa onset detection on the Demucs guitar stem, snapped to a 1/4-beat grid
  frets     dominant CQT pitch at each onset, ranked within a +-8 s window into 5 lanes
  chords    strongest onsets (top 20% onset strength) become 2-note adjacent-fret chords
  sustains  RMS stays above 40% of the onset level for >= 1/2 beat -> sustain to the decay point
  types     natural CH HOPO rules only (the writer resolves them)
Lower difficulties are rule-based reductions: Hard keeps 8th-note positions, Medium 8ths with
single notes only, Easy quarter notes on G-Y-B lanes.

    python -m riffchart.baselines.heuristic [--split test]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import yaml
from tqdm import tqdm

from ..formats.hopo import build_track, chart_hopo_threshold
from ..formats.song import Song
from ..timing.beats import track_beats
from ..timing.tempo_map import beats_to_tempo_map
from .common import load_full_mix, load_stem, split_songs, write_song_folder
from ..paths import load_config

SR, HOP, RES = 22050, 256, 192
GRID = RES // 4  # 16th notes


def expert_notes(y: np.ndarray, tm) -> list[tuple[int, int, int]]:
    fps = SR / HOP
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    on_f = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, delta=0.07, wait=2)
    if len(on_f) == 0:
        return []
    on_s = on_f / fps
    ticks = np.round(tm.sec_to_tick(on_s) / GRID).astype(int) * GRID
    ticks, first = np.unique(ticks, return_index=True)
    on_f, strength = on_f[first], env[on_f[first]]

    C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("E2"), n_bins=48))
    pitch_bin = np.array([C[:, min(f + 2, C.shape[1] - 1)].argmax() for f in on_f], float)
    times = on_f / fps
    frets = np.zeros(len(ticks), int)
    for i, t in enumerate(times):
        w = pitch_bin[np.abs(times - t) < 8.0]
        q = np.quantile(w, [0.2, 0.4, 0.6, 0.8]) if len(w) > 4 else np.array([1e9] * 4)
        frets[i] = int(np.searchsorted(q, pitch_bin[i], side="right"))
    strong = strength >= np.quantile(strength, 0.8)

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    notes = []
    for i, (tk, f) in enumerate(zip(ticks, frets)):
        nxt = ticks[i + 1] if i + 1 < len(ticks) else tk + 4 * RES
        start = on_f[i]
        stop = int(tm.tick_to_sec(nxt) * fps)
        seg = rms[start:stop]
        sus = 0
        if len(seg) > 2:
            above = seg >= 0.4 * seg[: 3].max()
            run = int(np.argmin(above)) if not above.all() else len(seg)
            end_tick = float(tm.sec_to_tick((start + run) / fps))
            length = int((end_tick - tk) // GRID * GRID)
            if length >= RES // 2:
                sus = min(length, int(nxt - tk - GRID))
        if strong[i]:
            a = min(f, 3)
            notes += [(int(tk), a, max(sus, 0)), (int(tk), a + 1, max(sus, 0))]
        else:
            notes.append((int(tk), int(f), max(sus, 0)))
    return notes


def reduce(notes: list[tuple[int, int, int]], diff: str) -> list[tuple[int, int, int]]:
    by_tick: dict[int, list[tuple[int, int, int]]] = {}
    for n in notes:
        by_tick.setdefault(n[0], []).append(n)
    step = {"hard": RES // 2, "medium": RES // 2, "easy": RES}[diff]
    out = []
    for tk, ns in sorted(by_tick.items()):
        if tk % step:
            continue
        if diff == "hard":
            out += ns
        else:
            f = min(n[1] for n in ns)
            if diff == "easy":
                f = min(f, 3)
            out.append((tk, f, ns[0][2]))
    return out


def chart_song(y_guitar: np.ndarray, y_mix: np.ndarray, cache: Path, timing: str = "phase2") -> Song:
    if timing == "phase2":  # Phase 2 setup: pretrained Beat This, one BPM per beat
        beats, downbeats = track_beats(y_mix, SR, cache)
        tm = beats_to_tempo_map(beats, downbeats, RES)
    else:  # Phase 3 defaults: fine-tuned tracker, smoothed map
        from ..timing.pipeline import tempo_map_from_audio
        tm = tempo_map_from_audio(y_mix, SR, cache=cache.parent.parent / "beats_finetuned" / cache.name)
    notes = expert_notes(y_guitar, tm)
    thr = chart_hopo_threshold(RES)
    tracks = {"expert": build_track("expert", notes, thr, midi=False)}
    for d in ("hard", "medium", "easy"):
        tracks[d] = build_track(d, reduce(notes, d), thr, midi=False)
    return Song(RES, tm, tracks, source_format="chart")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default="heuristic")
    ap.add_argument("--timing", default="phase2", choices=["phase2", "phase3"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = Path(cfg["data_root"])
    out = Path(cfg["outputs_root"]) / args.name
    for r in tqdm(list(split_songs(root, args.split).itertuples()), desc=args.name):
        if (out / r.song_id / "notes.chart").exists():
            continue
        y_mix = load_full_mix(r, SR)
        y_g = load_stem(root, r.song_id, "guitar", SR)
        if y_g is None:
            y_g = y_mix  # stems not ready: fall back to the mix
        song = chart_song(y_g, y_mix, root / "cache" / "beats" / f"{r.song_id}.npz", args.timing)
        write_song_folder(out / r.song_id, song, str(r.name), str(r.artist), args.name, float(r.audio_s))


if __name__ == "__main__":
    main()
