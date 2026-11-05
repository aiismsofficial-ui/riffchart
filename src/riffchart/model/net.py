"""Encoder-decoder charting transformer (Phase 4).

Encoder  input (B, C=2, 128 mels, beats*48 frames), beat-synchronous (one frame per 48th of a beat)
         -> conv stem (freq downsampling, time stride 3) -> 16 tokens per beat
         -> + shared beat-index embedding + sub-beat embedding -> pre-norm transformer (RoPE)
Decoder  token embedding + shared beat-index embedding + 48th-position embedding
         + "aligned memory" (mean encoder state of the token's beat, projected)
         -> causal pre-norm transformer (RoPE self-attention, cross-attention) -> logits
Blocks   RMSNorm, SwiGLU, F.scaled_dot_product_attention. RoPE angles are computed and applied in
         fp32 even under bf16 autocast (Phase 3 lesson: low-precision rotary tables wreck positions).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import V


@dataclass
class ModelConfig:
    d_model: int = 384
    n_heads: int = 6
    enc_layers: int = 4
    dec_layers: int = 6
    ff_mult: float = 8 / 3
    dropout: float = 0.1
    in_channels: int = 2
    n_mels: int = 128
    frames_per_beat: int = 48
    tokens_per_beat: int = 16
    max_beats: int = 64
    vocab: int = V.SIZE
    max_tokens: int = 2048
    qk_norm: bool = False        # RMS-normalise queries/keys per head (stabilises bigger models)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.w


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """x (B, H, T, D), pos (T,) or (B, T). Rotation in fp32."""
    d = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.float()[..., None] * inv                       # (..., T, D/2)
    if ang.dim() == 2:
        ang = ang[None, None]
    else:
        ang = ang[:, None]
    cos, sin = ang.cos(), ang.sin()
    xf = x.float()
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, d: int, h: int, dropout: float, cross: bool = False, qk_norm: bool = False):
        super().__init__()
        self.h, self.dh, self.cross = h, d // h, cross
        self.qn, self.kn = (RMSNorm(d // h), RMSNorm(d // h)) if qk_norm else (nn.Identity(), nn.Identity())
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def forward(self, x, ctx=None, pos_q=None, pos_k=None, causal=False, kv_cache=None, mask=None):
        q = self.qn(self.split(self.q(x)))
        if self.cross:
            if kv_cache is not None and "k" in kv_cache:
                k, v = kv_cache["k"], kv_cache["v"]
            else:
                k, v = (self.split(t) for t in self.kv(ctx).chunk(2, -1))
                k = self.kn(k)
                if kv_cache is not None:
                    kv_cache["k"], kv_cache["v"] = k, v
        else:
            k, v = (self.split(t) for t in self.kv(x).chunk(2, -1))
            q, k = rope(q, pos_q), rope(self.kn(k), pos_q)
            if kv_cache is not None:
                if "k" in kv_cache:
                    k = torch.cat([kv_cache["k"], k], 2)
                    v = torch.cat([kv_cache["v"], v], 2)
                kv_cache["k"], kv_cache["v"] = k, v
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0,
                                             is_causal=causal and kv_cache is None and mask is None)
        B, H, T, D = out.shape
        return self.o(out.transpose(1, 2).reshape(B, T, H * D))


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float, dropout: float):
        super().__init__()
        hid = int(d * mult / 64 + 0.5) * 64
        self.w12 = nn.Linear(d, 2 * hid, bias=False)
        self.w3 = nn.Linear(hid, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(self.drop(F.silu(a) * b))


class EncLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1, self.att = RMSNorm(c.d_model), Attention(c.d_model, c.n_heads, c.dropout, qk_norm=c.qk_norm)
        self.n2, self.ff = RMSNorm(c.d_model), SwiGLU(c.d_model, c.ff_mult, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, pos):
        x = x + self.drop(self.att(self.n1(x), pos_q=pos))
        return x + self.drop(self.ff(self.n2(x)))


class DecLayer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n1, self.self_att = RMSNorm(c.d_model), Attention(c.d_model, c.n_heads, c.dropout, qk_norm=c.qk_norm)
        self.n2, self.cross_att = RMSNorm(c.d_model), Attention(c.d_model, c.n_heads, c.dropout, cross=True, qk_norm=c.qk_norm)
        self.n3, self.ff = RMSNorm(c.d_model), SwiGLU(c.d_model, c.ff_mult, c.dropout)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mem, pos, cache=None):
        x = x + self.drop(self.self_att(self.n1(x), pos_q=pos, causal=True,
                                        kv_cache=None if cache is None else cache["self"]))
        x = x + self.drop(self.cross_att(self.n2(x), ctx=mem, kv_cache=None if cache is None else cache["cross"]))
        return x + self.drop(self.ff(self.n3(x)))


class ConvStem(nn.Module):
    """(B, C, 128, F) -> (B, F / (frames_per_beat / tokens_per_beat), d); frequency 128 -> 16."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        ch = [c.in_channels, 32, 64, 96, 128]
        layers = []
        for i in range(4):
            stride = (2, c.frames_per_beat // c.tokens_per_beat if i == 0 else 1) if i < 3 else (1, 1)
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, stride=stride, padding=1), nn.GroupNorm(8, ch[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(128 * (c.n_mels // 8), c.d_model)

    def forward(self, x):
        h = self.net(x)                                 # (B, 128, 16, F/3)
        B, C, Fq, T = h.shape
        return self.proj(h.permute(0, 3, 1, 2).reshape(B, T, C * Fq))


class ChartTransformer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.stem = ConvStem(c)
        self.beat_emb = nn.Embedding(c.max_beats + 1, c.d_model)
        self.sub_emb = nn.Embedding(c.tokens_per_beat, c.d_model)
        self.enc = nn.ModuleList([EncLayer(c) for _ in range(c.enc_layers)])
        self.enc_norm = RMSNorm(c.d_model)
        self.tok_emb = nn.Embedding(c.vocab, c.d_model)
        self.pos48_emb = nn.Embedding(c.frames_per_beat, c.d_model)
        self.align = nn.Linear(c.d_model, c.d_model, bias=False)
        self.dec = nn.ModuleList([DecLayer(c) for _ in range(c.dec_layers)])
        self.dec_norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, feats: torch.Tensor):
        """feats (B, C, 128, n_beats*48) -> memory (B, n_beats*16, d), beat summary (B, n_beats, d)."""
        x = self.stem(feats)
        B, T, D = x.shape
        tpb = self.c.tokens_per_beat
        n_beats = T // tpb
        idx = torch.arange(T, device=x.device)
        x = x + self.beat_emb(idx // tpb) + self.sub_emb(idx % tpb)
        for layer in self.enc:
            x = layer(x, idx)
        x = self.enc_norm(x)
        beat_mem = x[:, : n_beats * tpb].view(B, n_beats, tpb, D).mean(2)
        return x, beat_mem

    def embed_tokens(self, ids, bidx, pos48, beat_mem):
        bidx = bidx.clamp(0, beat_mem.shape[1] - 1)
        aligned = torch.gather(beat_mem, 1, bidx[..., None].expand(-1, -1, beat_mem.shape[-1]))
        return self.tok_emb(ids) + self.beat_emb(bidx) + self.pos48_emb(pos48) + self.align(aligned)

    def forward(self, feats, ids, bidx, pos48):
        mem, beat_mem = self.encode(feats)
        x = self.embed_tokens(ids, bidx, pos48, beat_mem)
        tpos = torch.arange(ids.shape[1], device=ids.device)
        for layer in self.dec:
            x = layer(x, mem, tpos)
        return self.head(self.dec_norm(x))

    # ---- incremental decoding
    def new_cache(self):
        return [{"self": {}, "cross": {}} for _ in self.dec]

    def step(self, ids, bidx, pos48, start: int, mem, beat_mem, cache, key_valid):
        """Feed tokens ids (B, t) at positions start..start+t-1 and return the last logits.
        key_valid (B, start+t) marks non-padding keys (prefixes are left-padded; RoPE only sees
        relative distances, so padding does not shift anything)."""
        x = self.embed_tokens(ids, bidx, pos48, beat_mem)
        T = ids.shape[1]
        tpos = torch.arange(start, start + T, device=ids.device)
        kv = key_valid[:, None, None, :]
        if T > 1:
            causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
            mask = kv[..., start:] & causal[None, None]
        else:
            mask = kv
        for layer, c in zip(self.dec, cache):
            h = layer.n1(x)
            x = x + layer.self_att(h, pos_q=tpos, kv_cache=c["self"], mask=mask)
            x = x + layer.cross_att(layer.n2(x), ctx=mem, kv_cache=c["cross"])
            x = x + layer.ff(layer.n3(x))
        return self.head(self.dec_norm(x[:, -1]))
