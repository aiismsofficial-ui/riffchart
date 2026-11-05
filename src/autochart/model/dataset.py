"""Training windows for the charting model.

A sample = (song, difficulty, grid source, window start). The grid source is the human tempo map,
the fine-tuned tracker or the pretrained tracker (Phase 3 maps), so the model learns to chart on
imperfect inference-time grids; human notes are snapped onto whichever grid was drawn. Expert
comes from every song; lower difficulties only from real human reductions (Phase 1 `real_*`).

Features: the (T, 2, 128) log-mel is sampled at the centre of every 48th of a beat between the
window's beats (linear interpolation), giving (2, 128, n_beats * 48).
Augmentation (train only): log-gain, random spectral tilt (EQ), guitar/mix balance and occasional
channel dropout, beat-time jitter applied to features and targets alike.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..formats.song import DIFFICULTIES, popcount
from ..timing.grid import GRID, MODEL_POSITIONS, MUSICAL_POSITIONS
from .data_prep import FPS, ROOT
from .grid_events import beat_time, downbeat_flags, notes_to_units
from .tokenizer import Event, encode_window, source_token, style_tokens

GRIDS = ("human", "ft", "pt")


def load_split(split: str) -> pd.DataFrame:
    s = pd.read_parquet("data/audit/songs.parquet")
    return s[s.split == split]


def positions_for(cfg: dict) -> np.ndarray:
    return MODEL_POSITIONS if cfg["data"].get("positions", "musical24") == "model16" else MUSICAL_POSITIONS


class Song:
    def __init__(self, sid: str):
        self.sid = sid
        self.mel = np.load(ROOT / "mel" / f"{sid}.npy", mmap_mode="r")
        pf = ROOT / "pitch" / f"{sid}.npy"
        self.pitch = np.load(pf, mmap_mode="r") if pf.exists() else None     # (T, 64) uint8, model/pitch_feat.py
        g = np.load(ROOT / "grid" / f"{sid}.npz")
        self.grids = {k: (g[f"{k}_b"], g[f"{k}_d"]) for k in GRIDS}
        self.audio_s = float(g["audio_s"])
        n = np.load(ROOT / "notes" / f"{sid}.npz")
        self.notes = {d: (n[f"{d}_on"], n[f"{d}_end"], n[f"{d}_mask"], n[f"{d}_type"])
                      for d in DIFFICULTIES if f"{d}_on" in n.files}


def window_features(mel: np.ndarray, beats: np.ndarray, start_beat: int, n_beats: int) -> np.ndarray:
    units = start_beat * GRID + np.arange(n_beats * GRID) + 0.5
    f = beat_time(units, beats) * FPS
    i0 = np.floor(f).astype(np.int64)
    w = (f - i0).astype(np.float32)[:, None, None]
    T = len(mel)
    valid = (i0 >= 0) & (i0 + 1 < T)
    i0c, i1c = np.clip(i0, 0, T - 1), np.clip(i0 + 1, 0, T - 1)
    lo, hi = min(i0c.min(), i1c.min()), max(i0c.max(), i1c.max()) + 1
    block = np.asarray(mel[lo:hi], np.float32)
    x = block[i0c - lo] * (1 - w) + block[i1c - lo] * w        # (F, 2, 128)
    x[~valid] = np.log(1e-6)
    return x.transpose(1, 2, 0)                                 # (2, 128, F)


def pitch_window(pitch: np.ndarray | None, beats: np.ndarray, start_beat: int, n_beats: int) -> np.ndarray:
    """Pitch input (model/pitch_feat.py) on the same beat grid as window_features: (1, 128, F), log units.
    None (no pitch available) gives silence."""
    from .pitch_feat import LO, dequant, to_bands
    if pitch is None:
        return np.full((1, 128, n_beats * GRID), LO, np.float32)
    x = window_features(pitch[:, None, :], beats, start_beat, n_beats)   # (1, 64, F), quantised scale
    x = np.maximum(dequant(np.maximum(x, 0)), LO)
    return to_bands(x.transpose(0, 2, 1)).transpose(0, 2, 1)              # (1, 128, F)


def model_window(mel, pitch, beats, start_beat: int, n_beats: int, use_pitch: bool) -> np.ndarray:
    x = window_features(mel, beats, start_beat, n_beats)
    if use_pitch:
        x = np.concatenate([x, pitch_window(pitch, beats, start_beat, n_beats)], 0)
    return x


def input_stats(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Normalisation (mean, std), each (C, 128, 1): mel channels, plus the pitch channel if used."""
    st = np.load(ROOT / "mel_stats.npz")
    mean, std = st["mean"][:, :, None], st["std"][:, :, None]
    if cfg["data"].get("pitch"):
        ps = np.load(ROOT / "pitch_stats.npz")
        mean = np.concatenate([mean, ps["mean"][:, :, None]], 0)
        std = np.concatenate([std, ps["std"][:, :, None]], 0)
    return mean.astype(np.float32), std.astype(np.float32)


def select_input(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Channel selection: 'both' (guitar stem + mix, default) or 'mix' only."""
    return x[1:2] if cfg["data"].get("input", "both") == "mix" else x


class WindowDataset(torch.utils.data.IterableDataset):
    def __init__(self, split: str, cfg: dict, train: bool = True, seed: int = 0, song_limit: int | None = None):
        self.cfg, self.train, self.seed = cfg, train, seed
        rows = load_split(split)
        if song_limit:
            rows = rows.sample(min(song_limit, len(rows)), random_state=0)
        self.rows = rows
        lib = rows["library"] if "library" in rows else pd.Series("main", index=rows.index)
        self.source = dict(zip(rows.song_id, lib.map(lambda x: "main" if x == "main" else "extra")))
        self.by_diff = {"expert": list(rows.song_id)}
        for d in ("hard", "medium", "easy"):
            self.by_diff[d] = list(rows[rows[f"real_{d}"]].song_id)
        self.mean, self.std = input_stats(cfg)
        self.use_pitch = bool(cfg["data"].get("pitch"))

    def sample(self, rng: np.random.Generator, songs: dict):
        c, d_cfg = self.cfg["data"], self.cfg["data"]["difficulty_weights"]
        diffs = [d for d in DIFFICULTIES if self.by_diff[d]]
        p = np.array([d_cfg[d] for d in diffs], float)
        diff = diffs[rng.choice(len(diffs), p=p / p.sum())]
        sid = self.by_diff[diff][rng.integers(len(self.by_diff[diff]))]
        if sid not in songs:
            songs[sid] = Song(sid)
        s = songs[sid]
        gp = c["grid_probs"] if self.train else {"human": 0, "ft": 1, "pt": 0}
        gname = GRIDS[rng.choice(3, p=np.array([gp[g] for g in GRIDS], float) / sum(gp.values()))]
        beats, downs = s.grids[gname]
        if self.train and rng.random() < c.get("metrical_aug", 0.0):
            # show the grid at double or half tempo; targets stay at the human audio times, so the
            # model learns audio density instead of "notes per beat" (trackers make octave errors)
            if rng.random() < 0.5:
                beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
            else:
                beats = beats[rng.integers(2)::2]
        if self.train and c["beat_jitter_s"] > 0:
            beats = beats + rng.normal(0, c["beat_jitter_s"], len(beats))
            beats = np.maximum.accumulate(beats + np.arange(len(beats)) * 1e-4)
        on, end, mask, ntype = s.notes[diff]
        u, m, t, sus = notes_to_units(on, end, mask, ntype, beats, positions_for(self.cfg))
        n_ctx, n_tgt = c["context_beats"], c["target_beats"]
        first_b, last_b = int(u.min() // GRID), int(u.max() // GRID)
        start = int(rng.integers(first_b - n_ctx - n_tgt // 2, last_b - n_ctx - n_tgt // 2 + 1))
        nb = n_ctx + n_tgt
        x = model_window(s.mel, s.pitch, beats, start, nb, self.use_pitch)
        x = (x - self.mean) / self.std
        rel = u - start * GRID
        sel = (rel >= 0) & (rel < nb * GRID + GRID)       # one beat past the end for SUS_NEXT lookups
        events = [Event(int(a), int(b), int(cc), int(dd)) for a, b, cc, dd in zip(rel[sel], m[sel], t[sel], sus[sel])]
        bi = np.arange(start, start + nb)
        flags = downbeat_flags(beats, downs)
        down = np.where((bi >= 0) & (bi < len(beats)), flags[np.clip(bi, 0, len(beats) - 1)], (bi % 4) == 0)
        tgt = [e for e in events if n_ctx * GRID <= e.unit < nb * GRID]
        chord_share = float((popcount(np.array([e.mask for e in tgt], np.uint8)) > 1).mean()) if tgt else 0.0
        drop = self.train and rng.random() < c["style_dropout"]
        style = style_tokens(len(tgt), n_tgt, chord_share, drop=drop or not self.cfg["model_style"])
        if self.cfg.get("model_source"):
            src_drop = self.train and rng.random() < c["style_dropout"]
            style = style + [source_token("auto" if src_drop else self.source[sid])]
        ids, bidx, pos, loss = encode_window(events, down, n_ctx, n_tgt, diff, style)
        if self.train:
            x = self.augment(x, rng)
        return select_input(x, self.cfg).astype(np.float32), ids, bidx, pos, loss

    def augment(self, x, rng):
        a = self.cfg["augment"]
        x = x + rng.uniform(-a["gain"], a["gain"]) / self.std                        # log gain
        tilt = np.linspace(-1, 1, x.shape[1])[None, :, None]
        x = x + rng.uniform(-a["tilt"], a["tilt"]) * tilt / self.std + \
            rng.uniform(-a["tilt"], a["tilt"]) * np.cos(np.pi * tilt)[..., :] / self.std
        bal = rng.uniform(-a["balance"], a["balance"])
        x[0] += bal / self.std[0]                                                   # guitar vs mix
        if len(x) > 2:
            x[2] += bal / self.std[2]                                               # pitch comes from the guitar
        r = rng.random()
        if r < a["drop_guitar"]:
            x[0] = (np.log(1e-6) - self.mean[0]) / self.std[0]
            if len(x) > 2:
                from .pitch_feat import LO
                x[2] = (LO - self.mean[2]) / self.std[2]
        elif r < a["drop_guitar"] + a["drop_mix"]:
            x[1] = (np.log(1e-6) - self.mean[1]) / self.std[1]
        return x

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id * 1000 if wi else 0))
        songs: dict = {}
        while True:
            try:
                yield self.sample(rng, songs)
            except (ValueError, IndexError):
                continue


def collate(batch, max_tokens: int, pad_multiple: int = 128):
    xs, ids, bidx, pos, loss = zip(*batch)
    L = min(max(len(i) for i in ids), max_tokens)
    L = min(-(-L // pad_multiple) * pad_multiple, max_tokens)   # few distinct shapes -> static compile
    B = len(batch)
    out_ids = torch.zeros(B, L, dtype=torch.long)
    out_b = torch.zeros(B, L, dtype=torch.long)
    out_p = torch.zeros(B, L, dtype=torch.long)
    out_l = torch.zeros(B, L, dtype=torch.bool)
    for i in range(B):
        n = min(len(ids[i]), L)
        out_ids[i, :n] = torch.from_numpy(ids[i][:n])
        out_b[i, :n] = torch.from_numpy(bidx[i][:n])
        out_p[i, :n] = torch.from_numpy(pos[i][:n])
        out_l[i, :n] = torch.from_numpy(loss[i][:n])
    return torch.from_numpy(np.stack(xs)), out_ids, out_b, out_p, out_l


def compute_mel_stats(n_songs: int = 200) -> None:
    rows = load_split("train").sample(n_songs, random_state=0)
    acc = []
    for sid in rows.song_id:
        m = np.load(ROOT / "mel" / f"{sid}.npy", mmap_mode="r")
        acc.append(np.asarray(m[:: 20], np.float32))
    a = np.concatenate(acc)                     # (N, 2, 128)
    np.savez(ROOT / "mel_stats.npz", mean=a.mean(0), std=a.std(0) + 1e-3)
