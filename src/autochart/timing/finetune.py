"""Fine-tune Beat This! on the human charts' tempo maps (Phase 3).

    python -m autochart.timing.finetune prepare           # log-mel features + beat targets (resumable)
    python -m autochart.timing.finetune train [--resume]  # TensorBoard logs in <out_dir>/tb

Targets: the human chart's beats/downbeats in audio time (chart time + ini delay +
align_correction), supervised only inside the charted region (first note - pre_s ... last note +
post_s); outside it the loss mask is 0 because charts don't say where beats are there.
Runs in fp32/TF32 (never bf16 autocast, see train()). The saved checkpoint keeps Beat This's format ({"state_dict": {"model.*"}, "hyper_parameters"}),
so beat_this.inference.Audio2Frames(checkpoint_path=...) loads it directly.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mir_eval
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from ..baselines.common import load_full_mix
from ..formats.loader import load_song
from ..paths import load_config

FPS = 50
SR = 22050


# ------------------------------------------------------------------ prepare
def prepare(cfg: dict) -> None:
    import pandas as pd
    from beat_this.preprocessing import LogMelSpect

    root = Path(cfg["data_root"])
    feat_dir = Path(cfg["features"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    songs = pd.read_parquet(root / "audit" / "songs.parquet")
    from ..data.split import is_live
    songs = songs[is_live(songs.split)]
    mel = LogMelSpect(device="cuda")

    todo = [r for r in songs.itertuples() if not (feat_dir / f"{r.song_id}.npy").exists()]
    print(f"{len(songs)} songs, {len(todo)} need features", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        for k, (r, y) in enumerate(ex.map(lambda r: (r, load_full_mix(r, SR)), todo)):
            with torch.no_grad():
                s = mel(torch.from_numpy(y).cuda()).half().cpu().numpy()
            np.save(feat_dir / f"{r.song_id}.npy", s)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)

    tg = {}
    rc = cfg["region"]
    for r in songs.itertuples():
        song, _, _ = load_song(r.path)
        off = r.delay_s + r.align_correction_ms / 1000.0
        on = song.tempo.tick_to_sec(song.tracks["expert"].tick) + off
        lo, hi = on.min() - rc["pre_s"], on.max() + rc["post_s"]
        beats = song.tempo.beat_times(hi - off + 1) + off
        downs = song.tempo.downbeat_times(hi - off + 1) + off
        tg[r.song_id] = np.concatenate([[lo, hi, len(beats)], beats, downs]).astype(np.float64)
    np.savez(cfg["targets"], **tg)
    print(f"targets for {len(tg)} songs", flush=True)


def unpack(t: np.ndarray):
    lo, hi, nb = t[0], t[1], int(t[2])
    return lo, hi, t[3 : 3 + nb], t[3 + nb :]


# ------------------------------------------------------------------ data
class CropDataset(torch.utils.data.IterableDataset):
    def __init__(self, ids: list[str], cfg: dict, seed: int):
        self.ids, self.cfg, self.seed = ids, cfg, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) + int(time.time()))
        tg = np.load(self.cfg["targets"])
        feat = Path(self.cfg["features"])
        tr, au = self.cfg["train"], self.cfg["augment"]
        L = tr["crop_frames"]
        while True:
            sid = self.ids[rng.integers(len(self.ids))]
            x = np.load(feat / f"{sid}.npy", mmap_mode="r")
            lo, hi, beats, downs = unpack(tg[sid])
            stretch = 1.0 + rng.uniform(-au["tempo_range"], au["tempo_range"])
            src_len = int(round(L * stretch))                      # frames read before resampling
            lo_f, hi_f = max(int(lo * FPS), 0), min(int(hi * FPS), len(x))
            if hi_f - lo_f < 200:
                continue
            start = int(rng.integers(max(lo_f - src_len // 2, 0), max(hi_f - src_len // 2, lo_f - src_len // 2 + 1)))
            start = min(max(start, 0), max(len(x) - src_len, 0))
            seg = torch.from_numpy(np.asarray(x[start : start + src_len], np.float32))
            if len(seg) < src_len:
                seg = F.pad(seg, (0, 0, 0, src_len - len(seg)))
            seg = F.interpolate(seg.T[None], size=L, mode="linear", align_corners=False)[0].T  # (L, 128)
            # targets in output frames
            def frames(ts):
                f = np.round((ts * FPS - start) / stretch).astype(int)
                return f[(f >= 0) & (f < L)]
            yb, yd, mask = torch.zeros(L), torch.zeros(L), torch.zeros(L)
            yb[frames(beats)] = 1
            yd[frames(downs)] = 1
            m0 = int(max((lo * FPS - start) / stretch, 0))
            m1 = int(min((hi * FPS - start) / stretch, L))
            m1 = min(m1, int((len(x) - start) / stretch))
            if m1 - m0 < 100:
                continue
            mask[m0:m1] = 1
            for _ in range(au["time_masks"]):
                w = int(rng.integers(0, au["time_mask_max"] + 1)); s = int(rng.integers(0, L - w + 1))
                seg[s : s + w] = 0
            for _ in range(au["freq_masks"]):
                w = int(rng.integers(0, au["freq_mask_max"] + 1)); s = int(rng.integers(0, 128 - w + 1))
                seg[:, s : s + w] = 0
            yield seg, yb, yd, mask


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, ids: list[str], cfg: dict) -> dict:
    from beat_this.inference import split_predict_aggregate
    from beat_this.model.postprocessor import Postprocessor

    model.eval()
    tg = np.load(cfg["targets"])
    pp = Postprocessor("minimal", FPS)
    fb, fd = [], []
    for sid in ids:
        x = torch.from_numpy(np.load(Path(cfg["features"]) / f"{sid}.npy").astype(np.float32)).cuda()
        pred = split_predict_aggregate(x, 1500, 6, "keep_first", model)
        b, d = pp(pred["beat"].float(), pred["downbeat"].float())
        lo, hi, rb, rd = unpack(tg[sid])
        w = lambda a: np.asarray(a)[(np.asarray(a) >= lo) & (np.asarray(a) <= hi)]  # noqa: E731
        fb.append(mir_eval.beat.f_measure(w(rb), w(b)))
        fd.append(mir_eval.beat.f_measure(w(rd), w(d)) if len(w(rd)) > 1 and len(w(d)) > 1 else 0.0)
    model.train()
    return {"beat_f1": float(np.mean(fb)), "downbeat_f1": float(np.mean(fd))}


# ------------------------------------------------------------------ train
def train(cfg: dict, resume: bool) -> None:
    import pandas as pd
    from beat_this.inference import load_checkpoint, load_model
    from beat_this.model.loss import ShiftTolerantBCELoss

    tr = cfg["train"]
    torch.manual_seed(tr["seed"])
    # fp32 + TF32 matmuls, NOT bf16 autocast: rotary-embedding-torch caches its sin/cos tables in the
    # dtype of the first call, and bf16 angles over 1500 positions destroy the positional signal
    # (pretrained val beat F1 0.858 -> 0.756). See NOTES.md, Phase 3.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    root = Path(cfg["data_root"])
    songs = pd.read_parquet(root / "audit" / "songs.parquet")
    train_ids = list(songs[songs.split == "train"].song_id)
    val_ids = list(songs[songs.split == "val"].song_id)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    hparams = load_checkpoint(cfg["init_checkpoint"], "cpu")["hyper_parameters"]
    model = load_model(cfg["init_checkpoint"], "cuda").train()
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tr["warmup_steps"])
                                              * 0.5 * (1 + math.cos(math.pi * min(s / tr["steps"], 1.0))))
    lb = ShiftTolerantBCELoss(tr["pos_weights"]["beat"], tr["tolerance"]).cuda()
    ld = ShiftTolerantBCELoss(tr["pos_weights"]["downbeat"], tr["tolerance"]).cuda()
    step, best = 0, -1.0
    last = out / "last.pt"
    if resume and last.exists():
        st = torch.load(last, map_location="cuda", weights_only=True)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best = st["step"], st["best"]
        print(f"resumed at step {step}, best {best:.4f}", flush=True)
    else:
        base = evaluate(model, val_ids, cfg)
        print(f"step 0 (pretrained final0): val {base}", flush=True)
        (out / "baseline_val.json").write_text(json.dumps(base))

    tb = SummaryWriter(str(out / "tb"))
    dl = torch.utils.data.DataLoader(CropDataset(train_ids, cfg, tr["seed"] + step), batch_size=tr["batch_size"],
                                     num_workers=tr["num_workers"], persistent_workers=True)
    t0 = time.time()
    for x, yb, yd, mask in dl:
        if step >= tr["steps"]:
            break
        x, yb, yd, mask = x.cuda(), yb.cuda(), yd.cuda(), mask.cuda()
        pred = model(x)
        loss_b = lb(pred["beat"].float(), yb, mask)
        loss_d = ld(pred["downbeat"].float(), yd, mask)
        loss = loss_b + loss_d
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 50 == 0:
            tb.add_scalar("train/loss_beat", loss_b.item(), step)
            tb.add_scalar("train/loss_downbeat", loss_d.item(), step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} ({loss_b.item():.4f}/{loss_d.item():.4f}) "
                  f"{(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % tr["eval_every"] == 0 or step == tr["steps"]:
            m = evaluate(model, val_ids, cfg)
            score = m["beat_f1"] + m["downbeat_f1"]
            for k, v in m.items():
                tb.add_scalar(f"val/{k}", v, step)
            print(f"step {step} val {m}", flush=True)
            if score > best:
                best = score
                torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                            "hyper_parameters": hparams, "step": step, "val": m}, out / "best.ckpt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best}, last)
    tb.close()
    print(f"done, best val score {best:.4f}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train"])
    ap.add_argument("--config", default="configs/beat_finetune.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "prepare":
        prepare(cfg)
    else:
        train(cfg, args.resume)


if __name__ == "__main__":
    main()
