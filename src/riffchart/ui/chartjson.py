"""Chart <-> JSON for the editor, and waveform peaks for the highway.

Ticks are authoritative (as in formats.song); the page converts to seconds with the same tempo map.
Note rows are [tick, mask, sustain, ntype] with mask bit0..4 = G R Y B O, bit5 = open and
ntype 0 strum / 1 HOPO / 2 tap (the *final* type; forced flags are recomputed when the chart is written).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ..formats.hopo import chart_hopo_threshold
from ..formats.song import DIFFICULTIES, GuitarTrack, Song, TempoMap

PEAKS_PER_S = 200


def song_to_json(song: Song, ini: dict[str, str] | None = None, audio_s: float | None = None) -> dict:
    ini = ini or {}
    tm = song.tempo
    try:
        delay = float(ini.get("delay", 0) or 0) / 1000.0
    except ValueError:
        delay = 0.0
    tracks = {}
    for d in DIFFICULTIES:
        tr = song.tracks.get(d)
        if tr is None:
            continue
        tracks[d] = {
            "notes": [[int(t), int(m), int(s), int(n)] for t, m, s, n in zip(tr.tick, tr.mask, tr.sustain, tr.ntype)],
            "star_power": [[int(t), int(n)] for t, n in tr.star_power],
            "solos": [[int(a), int(b)] for a, b in tr.solos],
        }
    end = audio_s or float(tm.tick_to_sec(song.last_tick())) + 4.0
    try:
        preview = float(ini.get("preview_start_time", "")) / 1000.0
    except ValueError:
        preview = None
    return {
        "resolution": int(song.resolution),
        "offset_s": float(song.chart_offset_s) + delay,     # audio time = chart time + offset_s
        "audio_s": float(end),
        "preview_s": preview,                               # song-list preview start (audio time), or None
        "hopo_threshold": int(chart_hopo_threshold(song.resolution, ini)),
        "tempo": [[int(t), float(b)] for t, b in zip(tm.ticks, tm.bpm)],
        "ts": [[int(t), int(n), int(d)] for t, n, d in zip(tm.ts_ticks, tm.ts_num, tm.ts_den)],
        "sections": [[int(t), s] for t, s in song.sections],
        "tracks": tracks,
        "meta": {k: ini.get(k, "") for k in ("name", "artist", "album", "genre", "year", "charter")},
    }


def json_to_song(d: dict) -> Song:
    """Inverse of song_to_json (used when the editor saves). Notes are sorted and one chord per tick."""
    res = int(d["resolution"])
    tempo = sorted((int(t), float(b)) for t, b in d["tempo"])
    ts = sorted((int(t), int(n), int(dd)) for t, n, dd in d.get("ts") or [[0, 4, 4]])
    tm = TempoMap(res, np.array([t for t, _ in tempo]), np.array([b for _, b in tempo]),
                  np.array([t for t, _, _ in ts]), np.array([n for _, n, _ in ts]), np.array([x for _, _, x in ts]))
    tracks = {}
    for diff, t in (d.get("tracks") or {}).items():
        if diff not in DIFFICULTIES:
            continue
        by_tick: dict[int, list[int]] = {}
        for tick, mask, sus, nt in t.get("notes", []):
            tick, mask = int(tick), int(mask) & 0x3F
            if mask == 0 or tick < 0:
                continue
            prev = by_tick.get(tick)
            if prev:                                        # merge duplicates into one chord
                prev[0] |= mask
                prev[1] = max(prev[1], int(sus))
            else:
                by_tick[tick] = [mask, max(0, int(sus)), int(nt)]
        ticks = np.array(sorted(by_tick), np.int64)
        rows = [by_tick[k] for k in ticks]
        mask = np.array([r[0] for r in rows], np.uint8)
        if len(mask):                                       # open can't be part of a chord
            mask = np.where((mask & 0x20) & (mask & 0x1F), mask & 0x1F, mask).astype(np.uint8)
        tr = GuitarTrack(diff, ticks, mask, np.array([r[1] for r in rows], np.int64),
                         np.array([min(max(r[2], 0), 2) for r in rows], np.uint8),
                         np.zeros(len(ticks), bool), np.zeros(len(ticks), bool))
        tr.star_power = sorted((int(a), int(b)) for a, b in t.get("star_power", []) if int(b) > 0)
        tr.solos = sorted((int(a), int(b)) for a, b in t.get("solos", []) if int(b) > int(a))
        tracks[diff] = tr
    sections = sorted((int(t), str(s)) for t, s in d.get("sections", []))
    return Song(res, tm, tracks, sections, chart_offset_s=0.0)


def peaks(path: Path, cache_dir: Path | None = None, per_s: int = PEAKS_PER_S) -> dict:
    """RMS amplitude envelope of an audio file, per_s values per second as uint8 (base64).
    Cached next to cache_dir keyed by the file's size and mtime."""
    path = Path(path)
    st = path.stat()
    key = f"{path.name}.{st.st_size}.{int(st.st_mtime)}.{per_s}.rms.json"
    if cache_dir is not None:
        hit = Path(cache_dir) / key
        if hit.exists():
            return json.loads(hit.read_text(encoding="utf-8"))
    with sf.SoundFile(str(path)) as f:
        sr, hop = f.samplerate, max(1, f.samplerate // per_s)
        chunks, carry = [], np.zeros(0, np.float32)
        for block in f.blocks(blocksize=hop * 2000, dtype="float32", always_2d=True):
            x = np.concatenate([carry, (block ** 2).mean(1)])        # power; RMS per bucket shows dynamics
            n = len(x) // hop
            chunks.append(np.sqrt(x[: n * hop].reshape(n, hop).mean(1)))
            carry = x[n * hop:]
        if len(carry):
            chunks.append(np.array([np.sqrt(carry.mean())], np.float32))
    env = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    top = float(np.percentile(env, 99.5)) if len(env) else 1.0
    q = np.clip(env / max(top, 1e-4) * 255, 0, 255).astype(np.uint8)
    out = {"name": path.stem, "per_s": per_s, "duration_s": len(env) / per_s,
           "data": base64.b64encode(q.tobytes()).decode("ascii")}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        for old in Path(cache_dir).glob(f"{path.name}.*.json"):
            old.unlink(missing_ok=True)
        (Path(cache_dir) / key).write_text(json.dumps(out), encoding="utf-8")
    return out
