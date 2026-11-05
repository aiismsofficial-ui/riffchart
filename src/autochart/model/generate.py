"""Chart a song with a trained model: sliding windows + grammar-constrained decoding.

Windows of `target_beats` (32) with the previous `context_beats` (16) of *generated* tokens as the
prefix, so patterns carry across windows. The four difficulties decode as one batch (left-padded
prefixes, masked). Grammar (per row):
    beat marker (forced: BAR on downbeats, else BEAT; EOS after the last beat of the window)
    -> POS (strictly increasing within the beat) -> [HOPO|TAP] -> NOTE -> [SUS] -> ...

    python -m autochart.model.generate --ckpt data/checkpoints/chart/<run>/best.pt --split test
           [--name <system>] [--temperature 0] [--top-p 1] [--limit N]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from ..baselines.common import split_songs, write_song_folder
from ..formats.hopo import build_track, chart_hopo_threshold
from ..formats.song import DIFFICULTIES, Song
from ..timing.grid import GRID, MUSICAL_POSITIONS
from ..timing.tempo_map import beats_to_tempo_map
from ..paths import project_path
from .data_prep import ROOT
from .dataset import Song as SongData, model_window, select_input, window_features
from .grid_events import beat_time, downbeat_flags
from .net import ChartTransformer, ModelConfig
from .tokenizer import V, Event, decode, encode_beats, resolve_sustains, source_token, style_tokens

POS_TOK = np.arange(V.POS, V.HOPO)
NOTE_TOK = np.arange(V.NOTE, V.OPEN + 1)
SUS_TOK = np.arange(V.SUS_NEXT, V.SRC)          # not the SRC tokens after them


class Charter:
    def __init__(self, ckpt: str, device: str = "cuda"):
        st = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = st["cfg"]
        self.cfg["model"]["vocab"] = st["model"]["tok_emb.weight"].shape[0]   # v1 predates the SRC tokens
        from .train import build_model
        self.model = build_model(self.cfg).to(device).eval()
        self.model.load_state_dict(st["model"])
        from .dataset import input_stats
        self.use_pitch = bool(self.cfg["data"].get("pitch"))
        self.mean, self.std = input_stats(self.cfg)
        self.device = device
        d = self.cfg["data"]
        self.n_ctx, self.n_tgt = d["context_beats"], d["target_beats"]
        from .dataset import positions_for
        self.pos_allowed = np.isin(MUSICAL_POSITIONS, positions_for(self.cfg))

    def _allowed(self, state, last_pos, at_last_beat, next_marker, pos_ok=None):
        allow = np.zeros(V.SIZE, bool)
        end_tok = V.EOS if at_last_beat else next_marker
        if state in ("beat", "note", "sus"):
            ok = (MUSICAL_POSITIONS > last_pos) & self.pos_allowed
            if pos_ok is not None:
                ok &= pos_ok
            allow[POS_TOK[ok]] = True
            allow[end_tok] = True
            if state == "note":
                allow[SUS_TOK] = True
        elif state == "pos":
            allow[[V.HOPO, V.TAP]] = True
            allow[NOTE_TOK] = True
        elif state == "type":
            allow[NOTE_TOK] = True
        return allow

    @torch.no_grad()
    def chart(self, mel, beats, downbeats, difficulties=DIFFICULTIES, temperature=0.0, top_p=1.0, seed=0,
              style=None, audio_s: float | None = None, pos_bias: float = 0.0,
              min_gap_s: dict | None = None, style_fn=None, source: str = "extra",
              pitch: np.ndarray | None = None, start_beat: int = 0, end_beat: int | None = None,
              prefix: dict[str, list[Event]] | None = None) -> dict[str, list[Event]]:
        """pos_bias: added to note-start (POS) logits; < 0 thins the chart, > 0 densifies it. A float applies
        to every difficulty, a {difficulty: bias} dict per difficulty.
        min_gap_s: optional per-difficulty minimum time between chord onsets, enforced by the grammar.
        style_fn(difficulty, first_beat, n_beats) -> [DENS, CHORD] tokens per window (overrides style).
        source: main | extra | auto (training library style), for models trained with `model_source` (ignored otherwise).
        start_beat / end_beat / prefix: chart only beats [start_beat, end_beat), continuing from the chords in
        `prefix` (song units, as returned here) as the left context; only the new chords are returned."""
        n_ctx, n_tgt, nb = self.n_ctx, self.n_tgt, self.n_ctx + self.n_tgt
        flags = downbeat_flags(beats, downbeats)
        audio_s = audio_s if audio_s is not None else len(mel) / 86.1328125
        # only beats inside the audio: past the end the grid is extrapolated and there is nothing to chart
        total = int(np.searchsorted(beats, audio_s - 0.05))
        if end_beat is not None:
            total = min(total, int(end_beat))
        g = torch.Generator(device=self.device).manual_seed(seed)
        events = {d: [e for e in (prefix or {}).get(d, []) if e.unit < start_beat * GRID] for d in difficulties}
        style = style or [V.DENS + V.N_DENS - 1, V.CHORD + V.N_CHORD - 1]
        B = len(difficulties)
        if isinstance(pos_bias, dict):
            pb = torch.tensor([float(pos_bias.get(d, 0.0)) for d in difficulties], device=self.device)[:, None]
            pos_bias = pb if bool((pb != 0).any()) else 0.0
        for T0 in range(int(start_beat), total, n_tgt):
            s = T0 - n_ctx
            x = (model_window(mel, pitch, beats, s, nb, self.use_pitch) - self.mean) / self.std
            x = torch.from_numpy(select_input(x, self.cfg).astype(np.float32))[None].to(self.device)
            mem, beat_mem = self.model.encode(x)
            mem, beat_mem = mem.expand(B, -1, -1), beat_mem.expand(B, -1, -1)
            bi = np.arange(s, s + nb)
            down = np.where((bi >= 0) & (bi < len(beats)), flags[np.clip(bi, 0, len(beats) - 1)], bi % 4 == 0)
            n_this = min(n_tgt, total - T0)
            prefixes = []
            for d in difficulties:
                ctx = [Event(e.unit - s * GRID, e.mask, e.ntype, e.sus) for e in events[d] if s * GRID <= e.unit < T0 * GRID]
                ids, bidx, pos = encode_beats(ctx, n_ctx, down, 0)
                st = list(style_fn(d, T0, n_this) if style_fn else style)
                if self.cfg.get("model_source"):
                    st.append(source_token(source))
                head = [V.BOS, V.DIFF + DIFFICULTIES.index(d)] + st
                prefixes.append((head + ids + [V.SEP], [0] * len(head) + bidx + [n_ctx], [0] * len(head) + pos + [0]))
            L = max(len(p[0]) for p in prefixes)
            ids_t = torch.zeros(B, L, dtype=torch.long); b_t = torch.zeros_like(ids_t); p_t = torch.zeros_like(ids_t)
            valid = torch.zeros(B, L + 2048, dtype=torch.bool)
            for i, (a, bb, pp) in enumerate(prefixes):
                n = len(a)
                ids_t[i, L - n:] = torch.tensor(a); b_t[i, L - n:] = torch.tensor(bb); p_t[i, L - n:] = torch.tensor(pp)
                valid[i, L - n:] = True
            valid[:, L:] = True
            ids_t, b_t, p_t, valid = (t.to(self.device) for t in (ids_t, b_t, p_t, valid))
            cache = self.model.new_cache()
            logits = self.model.step(ids_t, b_t, p_t, 0, mem, beat_mem, cache, valid[:, :L])
            # decoding state per row
            state = ["need"] * B
            beat = [n_ctx - 1] * B
            lastp = [-1] * B
            curp = [0] * B
            last_t = [-1e9] * B                       # time of the previous chord onset (s)
            gaps = [(min_gap_s or {}).get(d, 0.0) for d in difficulties]
            out = [[] for _ in range(B)]
            done = [False] * B
            t = L
            last_beat = n_ctx + n_this - 1
            while not all(done) and t < L + 2000:
                nxt = np.zeros(B, np.int64)
                lg = logits.float()
                # one grammar mask for the whole batch, one sampling op, one device->host copy
                allow = np.zeros((B, V.SIZE), bool)
                forced = np.full(B, -1, np.int64)
                for i in range(B):
                    if done[i]:
                        forced[i] = V.PAD
                    elif state[i] == "need":
                        forced[i] = V.BAR if down[beat[i] + 1] else V.BEAT
                    else:
                        nm = V.BAR if beat[i] + 1 < nb and down[min(beat[i] + 1, nb - 1)] else V.BEAT
                        pos_ok = None
                        if gaps[i] > 0 and state[i] in ("beat", "note", "sus"):
                            cand_t = beat_time((s + beat[i]) * GRID + MUSICAL_POSITIONS, beats)
                            pos_ok = cand_t - last_t[i] >= gaps[i]
                        allow[i] = self._allowed(state[i], lastp[i], beat[i] >= last_beat, nm, pos_ok)
                allow[forced >= 0, V.PAD] = True                     # keep rows finite; overridden below
                if isinstance(pos_bias, torch.Tensor) or pos_bias:
                    lg[:, V.POS:V.HOPO] += pos_bias
                allow = allow[:, : lg.shape[-1]]                     # older checkpoints: smaller vocab
                li = lg.masked_fill(~torch.from_numpy(allow).to(self.device), float("-inf"))
                if temperature <= 0:
                    pick = li.argmax(-1)
                else:
                    pr = torch.softmax(li / temperature, -1)
                    if top_p < 1:
                        sp, si = pr.sort(-1, descending=True)
                        keep = sp.cumsum(-1) - sp < top_p
                        pr = torch.zeros_like(pr).scatter(-1, si, sp * keep)
                    pick = torch.multinomial(pr, 1, generator=g).view(-1)
                nxt = np.where(forced >= 0, forced, pick.cpu().numpy())
                nb_t, np_t = np.zeros(B, np.int64), np.zeros(B, np.int64)
                for i in range(B):
                    tok = int(nxt[i])
                    if done[i]:
                        nb_t[i], np_t[i] = beat[i], 0
                        continue
                    out[i].append(tok)
                    if tok in (V.BEAT, V.BAR):
                        beat[i] += 1; lastp[i] = -1; curp[i] = 0; state[i] = "beat"
                    elif tok == V.EOS:
                        done[i] = True
                    elif V.POS <= tok < V.HOPO:
                        curp[i] = int(MUSICAL_POSITIONS[tok - V.POS]); lastp[i] = curp[i]; state[i] = "pos"
                        last_t[i] = float(beat_time((s + beat[i]) * GRID + curp[i], beats))
                    elif tok in (V.HOPO, V.TAP):
                        state[i] = "type"
                    elif V.NOTE <= tok <= V.OPEN:
                        state[i] = "note"
                    else:
                        state[i] = "sus"
                    nb_t[i], np_t[i] = min(beat[i], nb - 1), curp[i]
                if all(done):
                    break
                tok_t = torch.from_numpy(nxt).to(self.device)[:, None]
                logits = self.model.step(tok_t, torch.from_numpy(nb_t).to(self.device)[:, None],
                                         torch.from_numpy(np_t).to(self.device)[:, None], t, mem, beat_mem, cache,
                                         valid[:, : t + 1])
                t += 1
            for i, d in enumerate(difficulties):
                for e in decode(out[i], beat0=n_ctx):
                    events[d].append(Event(e.unit + s * GRID, e.mask, e.ntype, e.sus))
        out = {}
        for d, ev in events.items():
            ev = [e for e in ev if beat_time(e.unit, beats) < audio_s - 0.05 and e.unit >= start_beat * GRID]
            out[d] = resolve_sustains(ev, gap_units=0)
        return out


# Human lower / Expert chord-count ratio, median over train songs with a genuine reduction
# (Phase 1 real_*: 92 / 77 / 61 songs).
LOWER_RATIO = {"hard": 0.78, "medium": 0.56, "easy": 0.43}


def chart_guided(ch: "Charter", mel, beats, downbeats, difficulties, ratio=None, **kw) -> dict[str, list[Event]]:
    """Expert first, then the lower difficulties with their density token set per window from the
    generated Expert density times the human ratio (the AUTO token over-densifies them, Phase 5)."""
    ratio = ratio or LOWER_RATIO
    out = ch.chart(mel, beats, downbeats, ["expert"], **kw)
    ex_units = np.array([e.unit for e in out["expert"]])
    lower = [d for d in difficulties if d != "expert"]
    if lower:
        def style_fn(d, b0, n):
            k = int(((ex_units >= b0 * GRID) & (ex_units < (b0 + n) * GRID)).sum())
            return style_tokens(round(k * ratio[d]), n, 0.0)[:1] + [V.CHORD + V.N_CHORD - 1]
        out.update(ch.chart(mel, beats, downbeats, lower, style_fn=style_fn, **kw))
    return {d: out[d] for d in difficulties}


def events_to_song(events: dict[str, list[Event]], beats, downbeats, res: int = 192, tempo=None) -> Song:
    """tempo: an existing TempoMap whose beat k (tick k * res) is beats[k]; default: one built from the beats."""
    if tempo is None:
        tm = beats_to_tempo_map(beats, downbeats, res, mode="per_beat")
        n_lead = int(round(float(tm.sec_to_tick(beats[0])) / res))
    else:
        tm, res, n_lead = tempo, tempo.resolution, 0
    thr = chart_hopo_threshold(res)
    k = res / GRID
    tracks = {}
    for d, evs in events.items():
        notes, taps, want = [], set(), {}
        evs = sorted(evs, key=lambda e: e.unit)
        for i, e in enumerate(evs[:-1]):       # a sustain never runs past the next chord (SUS_k can overshoot)
            e.sus = min(e.sus, evs[i + 1].unit - e.unit) if e.sus > 0 else e.sus
        for e in evs:
            tick = int(round((n_lead * GRID + e.unit) * k))
            if tick < 0:
                continue
            for f in range(6):
                if e.mask >> f & 1:
                    notes.append((tick, f, int(round(max(e.sus, 0) * k))))
            if e.ntype == 2:
                taps.add(tick)
            want[tick] = e.ntype
        tr = build_track(d, notes, thr, midi=False, tap_ticks=taps)
        forced = {int(t) for t, nt in zip(tr.tick, tr.ntype) if nt != 2 and (nt == 1) != (want.get(int(t)) == 1)}
        tracks[d] = build_track(d, notes, thr, midi=False, tap_ticks=taps, chart_forced_ticks=forced)
    return Song(res, tm, tracks, source_format="chart")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--name")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--grid", default="ft")
    ap.add_argument("--all-diffs", action="store_true", help="chart every difficulty even without a real human reduction")
    ap.add_argument("--diffs", default=None, help="comma list, e.g. expert (ablations)")
    ap.add_argument("--pos-bias", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (repeat runs to measure sampling noise)")
    ap.add_argument("--guided", action="store_true", help="lower difficulties conditioned on the Expert density")
    ap.add_argument("--source", default="extra", choices=["main", "extra", "auto"])
    ap.add_argument("--require-lower", action="store_true", help="skip songs without a real human reduction")
    ap.add_argument("--min-gap", default="0", help="min onset gap (s) for all difficulties, or 'auto' for the "
                    "per-difficulty playability minimum (configs/eval.yaml min_ioi_s, calibrated on train)")
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    ch = Charter(args.ckpt)
    if args.min_gap == "auto":
        rules = yaml.safe_load(open(project_path("configs/eval.yaml")))["playability"]
        gaps = {d: float(rules[d]["min_ioi_s"]) for d in DIFFICULTIES}
    else:
        gaps = {d: float(args.min_gap) for d in DIFFICULTIES} if float(args.min_gap) else None
    name = args.name or Path(args.ckpt).parent.name
    out = Path("data/outputs") / name
    rows = list(split_songs("data", args.split).itertuples())[: args.limit]
    t0 = time.time()
    for r in tqdm(rows, desc=name, mininterval=10):
        if (out / r.song_id / "notes.chart").exists():
            continue
        sd = SongData(r.song_id)
        beats, downs = sd.grids[args.grid]
        diffs = [d for d in DIFFICULTIES if args.all_diffs or d == "expert" or getattr(r, f"real_{d}")]
        if args.require_lower and len(diffs) == 1:
            continue
        if args.diffs:
            diffs = [d for d in diffs if d in args.diffs.split(",")]
        kw = dict(temperature=args.temperature, top_p=args.top_p, audio_s=sd.audio_s, pos_bias=args.pos_bias,
                  min_gap_s=gaps, source=args.source, seed=args.seed)
        kw["pitch"] = sd.pitch
        ev = chart_guided(ch, sd.mel, beats, downs, diffs, **kw) if args.guided else \
            ch.chart(sd.mel, beats, downs, diffs, **kw)
        song = events_to_song(ev, beats, downs)
        write_song_folder(out / r.song_id, song, str(r.name), str(r.artist), "autochart", float(r.audio_s))
    print(f"{len(rows)} songs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
