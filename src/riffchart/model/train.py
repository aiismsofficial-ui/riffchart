"""Train the charting model.

    python -m riffchart.model.train --config configs/model_small.yaml [--resume] [--steps N]

Logs: <out_root>/<run>/tb (TensorBoard), checkpoints last.pt (resume) and best.pt (lowest val loss).
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from .data_prep import ROOT
from .dataset import WindowDataset, collate, compute_mel_stats
from .net import ChartTransformer, ModelConfig
from .tokenizer import V
from ..paths import load_config


def build_model(cfg: dict) -> ChartTransformer:
    m = dict(cfg["model"])
    m.setdefault("vocab", V.SIZE)      # checkpoints store theirs (models before SRC tokens are smaller)
    m["in_channels"] = (1 if cfg["data"].get("input", "both") == "mix" else 2) + (1 if cfg["data"].get("pitch") else 0)
    return ChartTransformer(ModelConfig(**m))


def init_from(model: torch.nn.Module, ckpt: str, device: str) -> None:
    """Start from another run's weights. Input channels the old model lacks (e.g. the pitch channel)
    get zero weights in the first conv, so the new model starts out computing exactly the old one."""
    old = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    new = model.state_dict()
    for k, v in old.items():
        if k not in new:
            continue
        if new[k].shape == v.shape:
            new[k] = v
        elif k == "stem.net.0.weight" and new[k].shape[1] > v.shape[1] and new[k].shape[0] == v.shape[0]:
            w = torch.zeros_like(new[k]); w[:, : v.shape[1]] = v; new[k] = w
        else:
            raise ValueError(f"init_from: {k} {tuple(v.shape)} -> {tuple(new[k].shape)}")
    model.load_state_dict(new)
    print(f"initialised from {ckpt}", flush=True)


def fixed_val_batches(cfg: dict) -> list:
    ds = WindowDataset("val", cfg, train=False, seed=12345)
    it = iter(ds)
    items = [next(it) for _ in range(cfg["train"]["val_windows"])]
    bs = cfg["train"]["batch_size"]
    return [collate(items[i : i + bs], cfg["data"]["max_tokens"]) for i in range(0, len(items), bs)]


def loss_fn(logits, ids, loss_mask):
    # predict token t+1 from position t
    lg, tgt, m = logits[:, :-1], ids[:, 1:], loss_mask[:, 1:]
    ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
    return (ce * m).sum() / m.sum().clamp(min=1), ce, m


@torch.no_grad()
def evaluate(model, batches, device) -> dict:
    model.eval()
    tot, n, onset_hit, onset_n = 0.0, 0, 0, 0
    for x, ids, b, p, lm in batches:
        x, ids, b, p, lm = (t.to(device, non_blocking=True) for t in (x, ids, b, p, lm))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, ids, b, p)
        loss, ce, m = loss_fn(logits, ids, lm)
        tot += float((ce * m).sum()); n += int(m.sum())
        # next-token accuracy on position tokens (a proxy for timing)
        pred = logits[:, :-1].argmax(-1)
        tgt = ids[:, 1:]
        is_pos = m & (tgt >= V.POS) & (tgt < V.HOPO)
        onset_hit += int(((pred == tgt) & is_pos).sum()); onset_n += int(is_pos.sum())
    model.train()
    return {"loss": tot / max(n, 1), "pos_acc": onset_hit / max(onset_n, 1)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_small.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--run")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.steps:
        cfg["train"]["steps"] = args.steps
    if args.run:
        cfg["run"] = args.run
    tr = cfg["train"]
    torch.manual_seed(tr["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(cfg["out_root"]) / cfg["run"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    if not (ROOT / "mel_stats.npz").exists():
        compute_mel_stats()

    device = "cuda"
    # Windows spills to system RAM when the card is full (5x slower, v2 run 2); a hard cap makes the caching
    # allocator free and reuse its blocks instead
    torch.cuda.set_per_process_memory_fraction(tr.get("mem_fraction", 0.9))
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.1f}M", flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tr["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tr["lr"], betas=tuple(tr["betas"]),
                            fused=True)

    def lr_at(s):
        if s < tr["warmup_steps"]:
            return (s + 1) / tr["warmup_steps"]
        q = min((s - tr["warmup_steps"]) / max(tr["steps"] - tr["warmup_steps"], 1), 1.0)
        return tr["min_lr_ratio"] + (1 - tr["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * q))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, best = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best = st["step"], st["best"]
        print(f"resumed at step {step} (best val {best:.4f})", flush=True)

    if step == 0 and tr.get("init_from"):
        init_from(model, tr["init_from"], device)
    fwd = model
    if tr.get("compile"):
        try:
            fwd = torch.compile(model, dynamic=False)
        except Exception as e:  # noqa: BLE001
            print("torch.compile unavailable, running eager:", repr(e)[:200], flush=True)

    val = fixed_val_batches(cfg)
    ds = WindowDataset("train", cfg, train=True, seed=tr["seed"] + step, song_limit=cfg["data"].get("train_song_limit"))
    dl = torch.utils.data.DataLoader(ds, batch_size=tr["batch_size"], num_workers=tr["num_workers"],
                                     collate_fn=functools.partial(collate, max_tokens=cfg["data"]["max_tokens"]),
                                     persistent_workers=True, prefetch_factor=4, pin_memory=True)
    tb = SummaryWriter(str(out / "tb"))
    t0, seen_tok = time.time(), 0
    t_log, step_log = t0, step
    gn_hist, skipped = collections.deque(maxlen=200), 0
    model.train()
    compiled_ok = fwd is not model
    for x, ids, b, p, lm in dl:
        if step >= tr["steps"]:
            break
        x, ids, b, p, lm = (t.to(device, non_blocking=True) for t in (x, ids, b, p, lm))
        # gradient accumulation: same batch and loss as one big step, a fraction of the activation memory
        n_tok = lm[:, 1:].sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for mb in zip(*(t.chunk(tr.get("grad_accum", 1)) for t in (x, ids, b, p, lm))):
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = fwd(*mb[:4])
            except torch.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001  (e.g. no triton on Windows)
                if compiled_ok:
                    print("compiled forward failed, falling back to eager:", repr(e)[:300], flush=True)
                    fwd, compiled_ok = model, False
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = fwd(*mb[:4])
                else:
                    raise
            _, ce, m = loss_fn(logits, mb[1], mb[4])
            part = (ce * m).sum() / n_tok
            part.backward()
            loss = loss + part.detach()
        m = lm[:, 1:]
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        # spike guard: skip updates whose grad norm is non-finite or far above the recent median
        # (v2 run 1 diverged at step ~8.6k after the grad norm crept up; one bad step can wreck Adam state)
        g = float(gn)
        if step > tr["warmup_steps"] and (not math.isfinite(g) or g > tr.get("spike_factor", 10.0) * max(np.median(gn_hist), 0.1)):
            skipped += 1
            tb.add_scalar("train/skipped_steps", skipped, step)
            if skipped % 50 == 0:
                print(f"step {step}: skipped {skipped} spiking updates (gn {g:.1f})", flush=True)
        else:
            gn_hist.append(g)
            opt.step()
        sched.step(); step += 1
        seen_tok += int(m.sum())
        if step % tr["log_every"] == 0:
            el = time.time() - t0
            tb.add_scalar("train/loss", loss.item(), step)
            tb.add_scalar("train/grad_norm", float(gn), step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
            if step % (tr["log_every"] * 4) == 0:
                now = time.time()          # recent speed (a cumulative average hides slowdowns and resumes)
                print(f"step {step} loss {loss.item():.4f} gn {float(gn):.2f} {(now - t_log) / (step - step_log):.3f}s/step "
                      f"{seen_tok / el:.0f} tgt-tok/s mem {torch.cuda.memory_allocated() / 2**30:.1f}/"
                      f"{torch.cuda.memory_reserved() / 2**30:.1f}GB", flush=True)
                t_log, step_log = now, step
        if step % tr["eval_every"] == 0 or step == tr["steps"]:
            m_val = evaluate(model, val, device)
            for k, v in m_val.items():
                tb.add_scalar(f"val/{k}", v, step)
            print(f"step {step} val {json.dumps({k: round(v, 4) for k, v in m_val.items()})}", flush=True)
            diverged = math.isfinite(best) and not m_val["loss"] < tr.get("diverge_factor", 1.5) * best
            if diverged and (out / "last.pt").exists():
                # diverged: go back to the last good state (weights + Adam) at half the learning rate
                st = torch.load(out / "last.pt", map_location=device, weights_only=True)
                model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
                sched.base_lrs = [lr * 0.5 for lr in sched.base_lrs]
                gn_hist.clear()
                print(f"step {step}: val loss {m_val['loss']:.3f} vs best {best:.3f} -> diverged, rolled back to "
                      f"step {st['step']} weights, lr x0.5", flush=True)
                continue
            if m_val["loss"] < best:
                best = m_val["loss"]
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "val": m_val}, out / "best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "best": best, "cfg": cfg}, out / "last.pt")
            keep = tr.get("keep_every")       # val loss doesn't predict chart quality (Phase 4): keep snapshots
            if keep and step % keep == 0:
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step, "val": m_val},
                           out / f"step{step}.pt")
    tb.close()
    print(f"done: best val loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
