"""Fretformer 2 (Autochart) baseline, re-implemented locally from the published ONNX graphs.

Weights: Dvaro/autochart-models `fretformer-v2/` (CC BY-NC-SA 4.0; used here only for local,
non-commercial benchmarking, never redistributed). Every stage follows `manifest.json`:
  stems   htdemucs_6s, piano folded into other; channels full, drums, bass, vocals, guitar, other
  beats   fine-tuned Beat This (ONNX mel + core, 1500-frame chunks, border 6, keep_first, minimal
          postprocessing) + smoother: remove_spurious (0.6), fill_missed (1.7), prepend_lead_in
  mel     transcriber mel graph per channel (absolute dB) -> shared reference (max over channels)
          -> clamp [-80, 0] -> (x + 40) / 40; 32 linearly interpolated frames per beat
  encoder ONNX; decoder: PyTorch port of decoder_step (weights read from the ONNX initialisers,
          parity-checked against onnxruntime) with a preallocated KV cache, so decoding is not
          O(L^2) in copies. All four difficulties decode as one batch.
Details the manifest leaves open (beat_idx convention, chord-token order, HOPO modifier meaning,
sampling) were picked on validation songs, see `--probe` and NOTES.md.

    python -m riffchart.baselines.fretformer [--split test] [--probe]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from onnx import numpy_helper
from tqdm import tqdm

from ..audio.separate import Separator
from ..formats.hopo import build_track, chart_hopo_threshold
from ..formats.song import DIFFICULTIES, Song
from ..timing.tempo_map import beats_to_tempo_map
from .common import load_full_mix, split_songs, write_song_folder

import soxr
from ..paths import load_config

FF_DIR = Path("data/third_party/fretformer-v2")
SR = 22050
RES = 192
V = dict(PAD=0, BOS=1, EOS=2, BEAT=3, DIFF=4, POS=8, NOTE=56, OPEN=87, HOPO=88, TAP=89, SUS=90,
         INTENSITY=102, DESC=110, SOLO_ON=151, SECTION_ON=155, STRUCT_LO=150, VOCAB=167)
SUS_TICKS = [12, 24, 36, 48, 72, 96, 144, 192, 288, 384, 576, 768]   # in 48-per-beat ticks
TPB = 48


# --------------------------------------------------------------------------- decoder port
class Decoder(torch.nn.Module):
    """PyTorch port of decoder_step_fp32.onnx (pre-norm TransformerDecoder, 10 layers)."""

    def __init__(self, onnx_path: Path, device: str = "cuda"):
        super().__init__()
        m = onnx.load(str(onnx_path))
        w = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in m.graph.initializer}
        self.L, self.H, self.D = 10, 8, 64
        # self-attn q/k/v and cross-attn q projections (x @ W + b), in graph order
        mm = [1949, 2080, 2091, 2122] + [2133 + 11 * i for i in range(36)]
        p = {}
        for layer in range(self.L):
            for j, name in enumerate(("q", "k", "v", "cq")):
                idx = mm[layer * 4 + j]
                p[f"{layer}.{name}.w"] = w[f"onnx::MatMul_{idx}"]
                p[f"{layer}.{name}.b"] = w[f"onnx::Add_{idx + 5}"]
            for k in ("self_attn.out_proj", "multihead_attn.out_proj", "linear1", "linear2", "norm1", "norm2", "norm3"):
                p[f"{layer}.{k}.w"] = w[f"decoder_layers.{layer}.{k}.weight"]
                p[f"{layer}.{k}.b"] = w[f"decoder_layers.{layer}.{k}.bias"]
        for k in ("token_embedding.weight", "token_position_embedding.weight", "output_head.weight", "output_head.bias"):
            p[k] = w[k]
        p["decoder_norm.w"], p["decoder_norm.b"] = w["decoder_norm.weight"], w["decoder_norm.bias"]
        self.p = {k: v.to(device) for k, v in p.items()}
        self.device = device

    def reset(self, batch: int, max_len: int):
        shape = (self.L, batch, self.H, max_len, self.D)
        self.k = torch.zeros(shape, device=self.device)
        self.v = torch.zeros(shape, device=self.device)
        self.t = 0

    def _ln(self, x, n):
        return torch.nn.functional.layer_norm(x, (512,), self.p[f"{n}.w"], self.p[f"{n}.b"], 1e-5)

    @torch.inference_mode()
    def step(self, token, step_id, beat_idx, mem, prefix_bias, cross_k, cross_v, mem_invalid):
        p, B = self.p, token.shape[0]
        x = p["token_embedding.weight"][token] + p["token_position_embedding.weight"][step_id] + mem[beat_idx] + prefix_bias
        t = self.t
        for i in range(self.L):
            h = self._ln(x, f"{i}.norm1")
            q = (h @ p[f"{i}.q.w"] + p[f"{i}.q.b"]).view(B, self.H, 1, self.D)
            self.k[i, :, :, t] = (h @ p[f"{i}.k.w"] + p[f"{i}.k.b"]).view(B, self.H, self.D)
            self.v[i, :, :, t] = (h @ p[f"{i}.v.w"] + p[f"{i}.v.b"]).view(B, self.H, self.D)
            att = torch.softmax((q * 0.125) @ self.k[i, :, :, : t + 1].transpose(-1, -2), -1)
            o = (att @ self.v[i, :, :, : t + 1]).reshape(B, 512)
            x = x + torch.nn.functional.linear(o, p[f"{i}.self_attn.out_proj.w"], p[f"{i}.self_attn.out_proj.b"])
            h = self._ln(x, f"{i}.norm2")
            q = (h @ p[f"{i}.cq.w"] + p[f"{i}.cq.b"]).view(B, self.H, 1, self.D)
            s = (q * 0.125) @ cross_k[i].transpose(-1, -2).unsqueeze(0)            # (B,H,1,n)
            s = s.masked_fill(mem_invalid.view(1, 1, 1, -1), float("-inf"))
            o = (torch.softmax(s, -1) @ cross_v[i].unsqueeze(0)).reshape(B, 512)
            x = x + torch.nn.functional.linear(o, p[f"{i}.multihead_attn.out_proj.w"], p[f"{i}.multihead_attn.out_proj.b"])
            h = self._ln(x, f"{i}.norm3")
            h = torch.relu(torch.nn.functional.linear(h, p[f"{i}.linear1.w"], p[f"{i}.linear1.b"]))
            x = x + torch.nn.functional.linear(h, p[f"{i}.linear2.w"], p[f"{i}.linear2.b"])
        self.t += 1
        x = self._ln(x, "decoder_norm")
        return torch.nn.functional.linear(x, p["output_head.weight"], p["output_head.bias"])


# --------------------------------------------------------------------------- front end
def _session(name: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(FF_DIR / name), providers=providers)


class Fretformer:
    def __init__(self, device: str = "cuda"):
        self.beat_mel = _session("beat_this_mel_fp32.onnx")
        self.beat_core = _session("beat_this_core_fp32.onnx")
        self.tr_mel = _session("transcriber_mel_fp32.onnx")
        self.encoder = _session("encoder_fp32.onnx")
        self.dec = Decoder(FF_DIR / "decoder_step_fp32.onnx", device)
        tables = np.load(FF_DIR / "prefix_tables.npz")
        self.desc = [torch.from_numpy(tables[f"descriptor_embedding_{k}"]).float().to(device) for k in range(5)]
        self.sep = Separator("htdemucs_6s")
        self.device = device

    # ---- beats (manifest stages.beat)
    def beats(self, mono22: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from beat_this.inference import split_predict_aggregate
        from beat_this.model.postprocessor import Postprocessor

        mel = self.beat_mel.run(None, {"wav": mono22.astype(np.float32)})[0]
        n = len(mel)
        if n < 1500:
            mel = np.pad(mel, ((0, 1500 - n), (0, 0)))
        core = self.beat_core

        class _M(torch.nn.Module):
            def forward(self, x):
                x = x.cpu().numpy().astype(np.float32)
                if x.shape[1] != 1500:
                    x = np.pad(x, ((0, 0), (0, 1500 - x.shape[1]), (0, 0)))
                b, d = core.run(None, {"mel": x})
                return {"beat": torch.from_numpy(b), "downbeat": torch.from_numpy(d)}

        pred = split_predict_aggregate(torch.from_numpy(mel), 1500, 6, "keep_first", _M())
        b, d = Postprocessor("minimal", 50)(pred["beat"][:n], pred["downbeat"][:n])
        return smooth_beats(np.asarray(b, float)), np.asarray(d, float)

    # ---- features
    def features(self, row) -> dict:
        mix44 = load_full_mix_stereo(row)
        stems = self.sep(mix44)
        stems["other"] = stems["other"] + stems.pop("piano")
        full22 = load_full_mix(row, SR)
        chans = {"full": full22}
        for name in ("drums", "bass", "vocals", "guitar", "other"):
            chans[name] = soxr.resample(stems[name], 44100, SR).mean(1).astype(np.float32)
        beats, downbeats = self.beats(full22)
        dur = len(full22) / SR
        beats = beats[beats <= dur]
        n_beats = min(len(beats) - 1, 1280)
        mels = [self.tr_mel.run(None, {"wav": chans[c]})[0] for c in ("full", "drums", "bass", "vocals", "guitar", "other")]
        ref = max(float(m.max()) for m in mels)
        mels = [(np.clip(m - ref, -80.0, 0.0) + 40.0) / 40.0 for m in mels]
        bm = beat_windows(mels, beats[: n_beats + 1])
        return {"beat_mel": bm, "beats": beats[: n_beats + 1], "downbeats": downbeats, "dur": dur}

    # ---- decoding
    def generate(self, feats: dict, difficulties=DIFFICULTIES, temperature: float = 0.0, top_p: float = 1.0,
                 beat_mode: str = "count_minus1", seed: int = 0, max_tokens: int = 8192) -> dict[str, list[int]]:
        bm = feats["beat_mel"][None]
        n = bm.shape[1]
        mem, ck, cv, inv = self.encoder.run(None, {"beat_mel": bm.astype(np.float32), "beat_mask": np.ones((1, n), bool)})
        dev = self.device
        mem, ck, cv = (torch.from_numpy(a).to(dev) for a in (mem, ck, cv))
        inv = torch.from_numpy(inv).to(dev).view(-1)
        B = len(difficulties)
        prefix = [[V["BOS"], V["DIFF"] + DIFFICULTIES.index(d)] + [V["DESC"] + k * 8 + 7 for k in range(5)]
                  + [V["SOLO_ON"], V["SECTION_ON"]] for d in difficulties]
        bias = sum(self.desc[k][7] for k in range(5)).view(1, 512).expand(B, 512).contiguous()
        self.dec.reset(B, max_tokens)
        g = torch.Generator(device=dev).manual_seed(seed)
        seqs = [list(p) for p in prefix]
        beats_seen = torch.zeros(B, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        P = len(prefix[0])
        tok = torch.tensor([p[0] for p in prefix], device=dev)
        for t in range(max_tokens - 1):
            if beat_mode == "count_minus1":
                bidx = (beats_seen - 1).clamp(0, n - 1)
            else:  # "count"
                bidx = beats_seen.clamp(0, n - 1)
            step = torch.full((B,), t, dtype=torch.long, device=dev)
            logits = self.dec.step(tok, step, bidx, mem, bias, ck, cv, inv)
            if t + 1 < P:
                tok = torch.tensor([p[t + 1] for p in prefix], device=dev)
                continue
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                pr = torch.softmax(logits / temperature, -1)
                if top_p < 1:
                    sp, si = pr.sort(-1, descending=True)
                    keep = sp.cumsum(-1) - sp < top_p
                    pr = torch.zeros_like(pr).scatter(-1, si, sp * keep)
                nxt = torch.multinomial(pr, 1, generator=g).view(-1)
            # a row that ran out of beats must end
            nxt = torch.where((beats_seen >= n) & (nxt == V["BEAT"]), torch.full_like(nxt, V["EOS"]), nxt)
            nxt = torch.where(done, torch.full_like(nxt, V["PAD"]), nxt)
            for b in range(B):
                if not done[b]:
                    seqs[b].append(int(nxt[b]))
            beats_seen += (nxt == V["BEAT"]).long()
            done |= nxt == V["EOS"]
            if bool(done.all()):
                break
            tok = nxt
        return {d: s[P:] for d, s in zip(difficulties, seqs)}


def smooth_beats(b: np.ndarray, min_ratio: float = 0.6, max_ratio: float = 1.7) -> np.ndarray:
    """remove_spurious_beats -> fill_missed_beats -> prepend_lead_in (manifest smoother order)."""
    if len(b) < 3:
        return b
    med = float(np.median(np.diff(b)))
    keep = [b[0]]
    for x in b[1:]:
        if x - keep[-1] >= min_ratio * med:
            keep.append(x)
    b = np.array(keep)
    out = [b[0]]
    for x in b[1:]:
        gap = x - out[-1]
        if gap > max_ratio * med:
            k = int(round(gap / med))
            out += list(out[-1] + gap * np.arange(1, k) / k)
        out.append(x)
    b = np.array(out)
    p0 = b[1] - b[0]
    n_lead = int(np.floor(b[0] / p0 + 1e-6))
    lead = b[0] - p0 * np.arange(n_lead, 0, -1)
    return np.concatenate([lead[lead >= -1e-6].clip(0), b])


def beat_windows(mels: list[np.ndarray], beats: np.ndarray) -> np.ndarray:
    n_frames = mels[0].shape[1]
    t0, t1 = beats[:-1, None], beats[1:, None]
    tt = t0 + (np.arange(32)[None] + 0.5) / 32 * (t1 - t0)       # (n_beats, 32)
    f = tt / (512 / SR)
    i0 = np.clip(np.floor(f).astype(int), 0, n_frames - 1)
    i1 = np.clip(i0 + 1, 0, n_frames - 1)
    fr = (f - np.floor(f)).astype(np.float32)
    out = np.stack([m[:, i0] * (1 - fr) + m[:, i1] * fr for m in mels], 0)  # (6, 80, n_beats, 32)
    return out.transpose(2, 0, 1, 3).astype(np.float32)                     # (n_beats, 6, 80, 32)


def load_full_mix_stereo(row) -> np.ndarray:
    from ..audio.io import load_mix
    from .common import mix_files
    return load_mix(mix_files(row), 44100, mono=False)


# --------------------------------------------------------------------------- tokens -> chart
def tokens_to_notes(tokens: list[int], beat_mode: str, chord_order: str = "mask", hopo_mode: str = "type"):
    """Returns [(tick48, mask, sustain48, kind)], kind in {'strum','hopo','tap'}."""
    # Stream per note: POS, [HOPO|TAP], NOTE, [SUS]  (the modifier precedes the note it marks)
    beat = 0 if beat_mode == "count" else -1
    pos = 0
    kind = "strum"
    notes = []
    for tk in tokens:
        if tk == V["EOS"]:
            break
        if tk == V["BEAT"]:
            beat += 1
            pos = 0
        elif V["POS"] <= tk < V["NOTE"]:
            pos = tk - V["POS"]
            kind = "strum"
        elif tk in (V["HOPO"], V["TAP"]):
            kind = "hopo" if tk == V["HOPO"] else "tap"
        elif V["NOTE"] <= tk <= V["OPEN"]:
            mask = 32 if tk == V["OPEN"] else combo_mask(tk - V["NOTE"], chord_order)
            notes.append([max(beat, 0) * TPB + pos, mask, 0, kind])
            kind = "strum"
        elif V["SUS"] <= tk < V["INTENSITY"] and notes:
            notes[-1][2] = SUS_TICKS[tk - V["SUS"]]
    return notes


_COMBOS_BY_SIZE = sorted(range(1, 32), key=lambda m: (bin(m).count("1"), m))


def combo_mask(i: int, order: str) -> int:
    return i + 1 if order == "mask" else _COMBOS_BY_SIZE[i]


def tokens_to_song(tok: dict[str, list[int]], beats: np.ndarray, downbeats: np.ndarray, beat_mode: str,
                   chord_order: str = "mask", hopo_mode: str = "type") -> Song:
    tm = beats_to_tempo_map(beats, downbeats, RES)
    lead = int(round(float(tm.sec_to_tick(beats[0])) / RES))  # beat 0 of the model = this chart beat
    thr = chart_hopo_threshold(RES)
    tracks = {}
    for d, seq in tok.items():
        notes, chart_forced, taps = [], set(), set()
        raw = tokens_to_notes(seq, beat_mode, chord_order, hopo_mode)
        for t48, mask, s48, kind in raw:
            tick = (lead * TPB + t48) * (RES // TPB)
            for f in range(6):
                if mask >> f & 1:
                    notes.append((tick, f, s48 * (RES // TPB)))
            if kind == "tap":
                taps.add(tick)
        tr = build_track(d, notes, thr, midi=False, tap_ticks=taps)
        if hopo_mode == "type":  # modifier = final note type: force wherever natural differs
            want = {(lead * TPB + t48) * (RES // TPB): k for t48, _, _, k in raw}
            chart_forced = {int(t) for t, nt in zip(tr.tick, tr.ntype)
                            if nt != 2 and (nt == 1) != (want.get(int(t)) == "hopo")}
            tr = build_track(d, notes, thr, midi=False, tap_ticks=taps, chart_forced_ticks=chart_forced)
        else:  # modifier = forced flag
            forced = set((lead * TPB + t48) * (RES // TPB) for t48, _, _, k in raw if k == "hopo")
            tr = build_track(d, notes, thr, midi=False, tap_ticks=taps, chart_forced_ticks=forced)
        tracks[d] = tr
    return Song(RES, tm, tracks, source_format="chart")


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default="fretformer2")
    ap.add_argument("--beat-mode", default="count_minus1")
    ap.add_argument("--chord-order", default="mask")
    ap.add_argument("--hopo-mode", default="forced")   # Phase 2: forced-flag reading matched references (type_acc 0.784 vs 0.708)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--save-tokens", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = Path(cfg["data_root"])
    out = Path(cfg["outputs_root"]) / args.name
    rows = list(split_songs(root, args.split).itertuples())[: args.limit]
    ff = Fretformer()
    for r in tqdm(rows, desc=args.name):
        if (out / r.song_id / "notes.chart").exists():
            continue
        cache = root / "cache" / "fretformer" / f"{r.song_id}.npz"
        if cache.exists():
            z = np.load(cache)
            feats = {k: z[k] for k in z.files}
        else:
            feats = ff.features(r)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **{k: np.asarray(v) for k, v in feats.items()})
        toks = ff.generate(feats, temperature=args.temperature, top_p=args.top_p, beat_mode=args.beat_mode)
        song = tokens_to_song(toks, feats["beats"], feats["downbeats"], args.beat_mode, args.chord_order, args.hopo_mode)
        write_song_folder(out / r.song_id, song, str(r.name), str(r.artist), args.name, float(r.audio_s))
        if args.save_tokens:
            (out / r.song_id / "tokens.json").write_text(json.dumps(toks))


if __name__ == "__main__":
    main()
