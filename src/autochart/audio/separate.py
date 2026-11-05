"""Demucs source separation (in-process API, no torchaudio I/O)."""
from __future__ import annotations

import numpy as np
import torch


class Separator:
    def __init__(self, model_name: str = "htdemucs_6s", device: str = "cuda", overlap: float = 0.25,
                 shifts: int = 0, segment: float | None = None):
        from demucs.pretrained import get_model

        self.model = get_model(model_name)
        self.model.to(device).eval()
        self.device = device
        self.overlap, self.shifts, self.segment = overlap, shifts, segment
        self.sr = self.model.samplerate
        self.sources = list(self.model.sources)

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray) -> dict[str, np.ndarray]:
        """wav: (n, 2) float32 at self.sr. Returns {source: (n, 2) float32}."""
        from demucs.apply import apply_model

        x = torch.from_numpy(np.ascontiguousarray(wav.T)).to(self.device)
        ref = x.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        x = (x - mean) / std
        kw = {"segment": self.segment} if self.segment else {}
        out = apply_model(self.model, x[None], device=self.device, shifts=self.shifts, split=True,
                          overlap=self.overlap, progress=False, **kw)[0]
        out = out * std + mean
        return {name: out[i].T.float().cpu().numpy() for i, name in enumerate(self.sources)}
