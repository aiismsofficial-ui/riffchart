"""Beat trackers: Beat This! checkpoints, an ensemble, and our fine-tuned checkpoint."""
from __future__ import annotations

import numpy as np
import torch

_CACHE: dict = {}


def _a2f(checkpoint: str, device: str = "cuda"):
    from beat_this.inference import Audio2Frames

    if (checkpoint, device) not in _CACHE:
        _CACHE[(checkpoint, device)] = Audio2Frames(checkpoint_path=checkpoint, device=device)
    return _CACHE[(checkpoint, device)]


def _post(beat_logits: torch.Tensor, down_logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.model.postprocessor import Postprocessor

    b, d = Postprocessor("minimal", 50)(beat_logits, down_logits)
    return np.asarray(b, float), np.asarray(d, float)


def refine(times: np.ndarray, logits: torch.Tensor, fps: int = 50) -> np.ndarray:
    """Sub-frame peak refinement: parabolic fit on the logits at frame-1, frame, frame+1.
    Beat This's minimal postprocessing returns frame-quantised times (20 ms steps, +-10 ms jitter)."""
    lg = logits.float().cpu().numpy()
    f = np.clip(np.round(np.asarray(times) * fps).astype(int), 1, len(lg) - 2)
    a, b, c = lg[f - 1], lg[f], lg[f + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    return (f + np.clip(off, -0.5, 0.5)) / fps


def get_tracker(name: str, ckpt: str | None = None, device: str = "cuda"):
    do_refine = name.endswith("_refined")
    name = name.removesuffix("_refined")
    if name == "beatthis":
        ckpts = ["final0"]
    elif name == "beatthis_ens":
        ckpts = ["final0", "final1", "final2"]
    elif name == "finetuned":
        ckpts = [ckpt or "data/checkpoints/beat_finetune/best.ckpt"]
    elif name == "finetuned_ens":  # fine-tuned + pretrained final0, logit average
        ckpts = [ckpt or "data/checkpoints/beat_finetune/best.ckpt", "final0"]
    elif name == "finetuned_ens4":
        ckpts = [ckpt or "data/checkpoints/beat_finetune/best.ckpt", "final0", "final1", "final2"]
    else:
        raise ValueError(name)

    def run(y: np.ndarray, sr: int):
        logits = [_a2f(c, device)(y, sr) for c in ckpts]
        b = torch.stack([torch.as_tensor(l[0]) for l in logits]).mean(0)
        d = torch.stack([torch.as_tensor(l[1]) for l in logits]).mean(0)
        bt, dt = _post(b, d)
        if do_refine:
            bt, dt = refine(bt, b), refine(dt, d)
        return bt, dt

    return run
