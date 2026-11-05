"""Phase 4 data preparation (resumable, per song).

  data/features/model/mel/<id>.npy     fp16 (T, 2, 128): log-mel of [Demucs guitar stem, full mix]
                                       sr 22050, n_fft 2048, hop 256 (86.13 fps), 30 Hz-11 kHz
  data/features/model/grid/<id>.npz    beat/downbeat times (audio s) from three sources:
                                       human (chart tempo map), ft (fine-tuned Beat This), pt
                                       (pretrained final0); tracker grids go through the Phase 3
                                       tempo-map builder (smooth 20 ms, raw TS) so they look like
                                       inference-time grids
  data/features/model/notes/<id>.npz   per difficulty: onset/end seconds (audio time), mask, type

    python -m riffchart.model.data_prep [--workers 6]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from ..audio.io import load_audio
from ..baselines.common import load_full_mix
from ..formats.loader import load_song
from ..formats.song import DIFFICULTIES
from ..data.split import is_live
from ..paths import PROJECT
from ..timing.tempo_map import beats_to_tempo_map

SR, N_FFT, HOP, N_MELS = 22050, 2048, 256, 128
FPS = SR / HOP
ROOT = PROJECT / "data/features/model"


class Mel(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.m = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP, f_min=30, f_max=11025,
                                                      n_mels=N_MELS, power=2.0, norm="slaney", mel_scale="slaney").to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, y: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(y).to(self.device)
        return torch.log(self.m(x) + 1e-6).T.cpu().numpy()   # (T, 128)


def song_notes(row) -> dict:
    song, _, _ = load_song(row.path)
    off = row.delay_s + row.align_correction_ms / 1000.0
    out = {}
    for d in DIFFICULTIES:
        tr = song.tracks.get(d)
        if tr is None or len(tr) == 0:
            continue
        on = song.tempo.tick_to_sec(tr.tick) + off
        sus = np.where(tr.sustain < song.resolution / 16, 0, tr.sustain)
        end = song.tempo.tick_to_sec(tr.tick + sus) + off
        out[f"{d}_on"], out[f"{d}_end"] = on, np.where(sus > 0, end, on)
        out[f"{d}_mask"], out[f"{d}_type"] = tr.mask, tr.ntype
    return out, song, off


def human_grid(song, off: float, audio_s: float) -> tuple[np.ndarray, np.ndarray]:
    end = audio_s - off + 2
    return song.tempo.beat_times(end) + off, song.tempo.downbeat_times(end) + off


def tracker_grid(model, beatmel: np.ndarray, audio_s: float) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor

    with torch.no_grad():   # fp32 only (rotary cache, see Phase 3 notes)
        p = split_predict_aggregate(torch.from_numpy(beatmel.astype(np.float32)).cuda(), 1500, 6, "keep_first", model)
    b, d = Postprocessor("minimal", 50)(p["beat"].float(), p["downbeat"].float())
    tm = beats_to_tempo_map(np.asarray(b), np.asarray(d), 192, mode="smooth", tol_s=0.02, ts_mode="raw")
    return tm.beat_times(audio_s + 2), tm.downbeat_times(audio_s + 2)


def main(argv=None) -> None:
    from beat_this.inference import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    songs = pd.read_parquet("data/audit/songs.parquet")
    songs = songs[is_live(songs.split)]
    for sub in ("mel", "grid", "notes"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    mel = Mel()
    ft = load_model("data/checkpoints/beat_finetune/best.ckpt", "cuda")
    pt = load_model("final0", "cuda")

    todo = [r for r in songs.itertuples() if not (ROOT / "notes" / f"{r.song_id}.npz").exists()]
    print(f"{len(songs)} songs, {len(todo)} to prepare", flush=True)

    def load(r):
        g = load_audio(Path("data/stems") / r.song_id / "guitar.ogg", SR)[0]
        m = load_full_mix(r, SR)
        n = min(len(g), len(m))
        return r, g[:n], m[:n]

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for k, (r, g, m) in enumerate(ex.map(load, todo)):
            feats = np.stack([mel(g), mel(m)], 1).astype(np.float16)       # (T, 2, 128)
            np.save(ROOT / "mel" / f"{r.song_id}.npy", feats)
            audio_s = len(m) / SR
            notes, song, off = song_notes(r)
            hb, hd = human_grid(song, off, audio_s)
            bm = np.load(Path("data/features/beatmel") / f"{r.song_id}.npy")
            fb, fd = tracker_grid(ft, bm, audio_s)
            pb, pd_ = tracker_grid(pt, bm, audio_s)
            np.savez(ROOT / "grid" / f"{r.song_id}.npz", human_b=hb, human_d=hd, ft_b=fb, ft_d=fd, pt_b=pb, pt_d=pd_,
                     audio_s=audio_s)
            np.savez(ROOT / "notes" / f"{r.song_id}.npz", **notes)   # written last = song complete
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
