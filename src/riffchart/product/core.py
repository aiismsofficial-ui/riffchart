"""Audio file -> Clone Hero song folder.

Stages (models load once per process, so batch mode pays the start-up cost once):
  load      decode + resample to 44.1 kHz stereo
  separate  Demucs htdemucs_6s -> guitar stem (Phase 1 settings)
  beats     fine-tuned Beat This on the mix -> smoothed tempo map (Phase 3 defaults)
  chart     beat-synchronous mel (guitar stem + mix) -> chart transformer, grammar-constrained sampling
  post      playability fixes + star power (configs/post.yaml)
  write     notes.chart, song.ini, song.ogg (+ guitar.ogg)
"""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import yaml

from ..audio.io import FFMPEG_EXT, load_audio
from ..formats.chart_writer import write_chart_text
from ..formats.song import DIFFICULTIES, Song
from ..formats.textio import parse_ini
from ..model.data_prep import SR as MEL_SR, Mel
from ..model.generate import Charter, events_to_song
from ..model.tokenizer import CHORD_EDGES, V
from ..paths import project_path
from ..post.pipeline import postprocess
from ..timing.pipeline import DEFAULTS as TEMPO_DEFAULTS
from ..timing.tempo_map import beats_to_tempo_map
from ..timing.trackers import get_tracker
from .follow import MODES as FOLLOW_MODES, playable_part
from .difficulty import estimate as estimate_difficulty, load_model as load_difficulty_model
from .meta import SongMeta, cover_art, read_meta
from .visuals import CH_VIDEO_OK, VIDEO_FILES, convert_video, find_art, find_first, refresh_visuals, save_video

OUT_SR = 44100
STAGES = ("load", "separate", "beats", "chart", "post", "write")


@dataclass
class Style:
    """User style controls. density: -1 (sparser) .. +1 (denser), added to note-start logits.
    chords: auto | low | mid | high, the chord-share conditioning token."""
    density: float = 0.0
    chords: str = "auto"
    follow: str | None = None        # guitar | auto | all (product/follow.py); None = configs/product.yaml

    @staticmethod
    def parse(spec: str | None) -> "Style":
        st = Style()
        for item in filter(None, (spec or "").split(",")):
            k, _, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if k == "density":
                st.density = float(v)
                if not -1 <= st.density <= 1:
                    raise ValueError("density must be between -1 and 1")
            elif k == "follow":
                if v not in FOLLOW_MODES:
                    raise ValueError(f"follow must be one of {', '.join(FOLLOW_MODES)}")
                st.follow = v
            elif k == "chords":
                if v not in CHORD_BIN:
                    raise ValueError(f"chords must be one of {', '.join(CHORD_BIN)}")
                st.chords = v
            else:
                raise ValueError(f"unknown style key {k!r} (use density=..., chords=..., follow=...)")
        return st

    def tokens(self) -> list[int]:
        return [V.DENS + V.N_DENS - 1, V.CHORD + CHORD_BIN[self.chords]]


# chord share bins (tokenizer CHORD_EDGES 0.1 / 0.25 / 0.4 / 0.55); the last bin is AUTO
CHORD_BIN = {"auto": len(CHORD_EDGES) + 1, "low": 1, "mid": 2, "high": 4}
# density: post/density.py thins or adds notes deterministically. Only "denser" also biases decoding, and
# only Expert's (note-start logits +1.5 at +1), the one way to add notes to Expert (~+5-10% on v2); biasing
# the lower difficulties too made them as dense as Expert (Creep, 2026-11-04).
DENSER_BIAS = 1.5


@dataclass
class Result:
    folder: Path
    meta: SongMeta
    audio_s: float
    n_notes: dict[str, int]
    diff_guitar: int
    seconds: dict[str, float] = field(default_factory=dict)
    filled_s: float = 0.0            # seconds where other instruments were followed (product/follow.py)


def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s).strip().rstrip(".")
    return s[:120] or "song"


def write_ogg(path: Path, data: np.ndarray, sr: int, quality: float) -> None:
    tmp = path.with_suffix(".tmp.ogg")
    # libsndfile compression_level: 0 = best quality, 1 = smallest
    with sf.SoundFile(str(tmp), "w", sr, data.shape[1], format="OGG", subtype="VORBIS",
                      compression_level=1.0 - quality) as f:
        for i in range(0, len(data), 32768):          # chunked: large single Vorbis writes crash (Phase 1)
            f.write(data[i : i + 32768])
    tmp.replace(path)


def _takes_two_args(fn) -> bool:
    import inspect
    try:
        inspect.signature(fn).bind("stage", 0.5)
        return True
    except (TypeError, ValueError):
        return False


def preview_start(mix_mono: np.ndarray, sr: int, downbeats: np.ndarray, audio_s: float) -> float:
    """Start of the loudest 30 s stretch beginning on a downbeat in the 15-70% part of the song
    (choruses are usually the loudest sections)."""
    hop = sr // 10
    n = len(mix_mono) // hop
    rms = np.sqrt((mix_mono[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    c = np.concatenate([[0], np.cumsum(rms)])
    win = min(300, n)
    cands = [d for d in downbeats if 0.15 * audio_s <= d <= min(0.70 * audio_s, audio_s - 30)]
    if not cands:
        return max(0.0, min(0.3 * audio_s, audio_s - 30))
    score = [c[min(int(d * 10) + win, n)] - c[int(d * 10)] for d in cands]
    return float(cands[int(np.argmax(score))])


def estimate_diff(n_expert: int, first_s: float, last_s: float, cuts: list[float]) -> int:
    nps = n_expert / max(last_s - first_s, 1.0)
    return int(np.searchsorted(cuts, nps))


def song_ini(meta: SongMeta, audio_s: float, preview_s: float, diff: int) -> str:
    lines = ["[song]", f"name = {meta.title}", f"artist = {meta.artist}"]
    if meta.album:
        lines.append(f"album = {meta.album}")
    if meta.genre:
        lines.append(f"genre = {meta.genre}")
    if meta.year:
        lines.append(f"year = {meta.year}")
    lines += ["charter = RiffChart", f"song_length = {int(audio_s * 1000)}",
              f"preview_start_time = {int(preview_s * 1000)}", f"diff_guitar = {diff}", "delay = 0",
              "loading_phrase = Charted automatically by RiffChart"]
    return "\n".join(lines) + "\n"


class RiffCharter:
    def __init__(self, config: str | Path = "configs/product.yaml", device: str | None = None):
        # NVIDIA (CUDA) and AMD (ROCm builds of PyTorch) GPUs both show up as "cuda"; otherwise the CPU
        device = device or self.cfg_device(config) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.cfg = yaml.safe_load(open(project_path(config), encoding="utf-8"))
        self.post_cfg = yaml.safe_load(open(project_path(self.cfg["post_config"]), encoding="utf-8"))
        rules = yaml.safe_load(open(project_path(self.post_cfg.get("eval_config", "configs/eval.yaml")),
                                    encoding="utf-8"))["playability"]
        self.min_gap = {d: float(rules[d]["min_ioi_s"]) for d in DIFFICULTIES}
        torch.backends.cuda.matmul.allow_tf32 = True
        from ..audio.separate import Separator
        s = self.cfg["separation"]
        self.sep = Separator(s["model"], device=device, overlap=s["overlap"], shifts=s["shifts"])
        self.charter = Charter(str(project_path(self.cfg["chart_ckpt"])), device=device)
        self.mel = Mel(device)
        self.tracker = get_tracker(self.cfg["beat_tracker"], device=device)

    def guitar_difficulty(self, song: Song) -> int:
        """diff_guitar 0-6: the fitted estimator (configs/difficulty.yaml) if present, else notes/s cuts."""
        path = self.cfg.get("difficulty_model")
        if path and project_path(path).exists():
            if not hasattr(self, "_diff_model"):
                self._diff_model = load_difficulty_model(project_path(path))
            return estimate_difficulty(song, self._diff_model)
        ex = song.tracks.get("expert")
        on = song.tempo.tick_to_sec(ex.tick) if ex is not None and len(ex) else np.zeros(2)
        return estimate_diff(len(on), float(on.min()), float(on.max()), self.cfg["diff_guitar_nps_cuts"])

    @staticmethod
    def cfg_device(config) -> str | None:
        """`device:` in the product config (auto | cuda | cpu); auto / missing = pick automatically."""
        d = (yaml.safe_load(open(project_path(config), encoding="utf-8")) or {}).get("device", "auto")
        return None if d in (None, "auto") else str(d)

    def chart_file(self, path: str | Path, out_root: str | Path, difficulties=DIFFICULTIES,
                   style: Style | None = None, write_stems: bool | None = None, overwrite: bool = False,
                   on_stage=None, meta: SongMeta | None = None, folder_name: str | None = None) -> Result | None:
        """Returns None when the output folder already exists and overwrite is False.
        meta / folder_name override the names read from the file (re-charting a song in place)."""
        path, style = Path(path), style or Style()
        stems = self.cfg["audio"]["write_stems"] if write_stems is None else write_stems
        meta = meta or read_meta(path)
        folder = Path(out_root) / (folder_name or safe_name(f"{meta.artist} - {meta.title}"))
        if (folder / "notes.chart").exists() and not overwrite:
            return None
        if (folder / "notes.chart").exists():          # keep the chart being replaced (edits included)
            hist = folder / ".riffchart" / "history"
            hist.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / "notes.chart", hist / time.strftime("%Y%m%d-%H%M%S-rechart-notes.chart"))
        sec: dict[str, float] = {}
        t = time.perf_counter()

        from .. import progress
        takes_frac = on_stage is not None and _takes_two_args(on_stage)

        def stage(name):
            if on_stage:
                on_stage(name)
            # the slow stages report how far they are (separation, the chart model's windows)
            progress.set_hook((lambda f, n=name: on_stage(n, f)) if takes_frac else None)

        def done(name):
            nonlocal t
            now = time.perf_counter()
            sec[name] = now - t
            t = now

        stage("load")
        mix, _ = load_audio(path, OUT_SR, mono=False)
        done("load")
        an = self.analyse(mix, stage=stage, done=done)
        stage("chart")
        song, play, _ = self.chart_analysis(an, difficulties, style)
        done("chart")
        mix, audio_s, mono, downs = an["mix"], an["audio_s"], an["mono"], an["downs"]

        stage("write")
        folder.mkdir(parents=True, exist_ok=True)
        q = self.cfg["audio"]["vorbis_quality"]
        if stems:
            g44 = play.audio if self.sep.sr == OUT_SR else soxr.resample(play.audio, self.sep.sr, OUT_SR)
            k = min(len(mix), len(g44))
            write_ogg(folder / "guitar.ogg", g44[:k].astype(np.float32), OUT_SR, q)
            write_ogg(folder / "song.ogg", (mix[:k] - g44[:k]).astype(np.float32), OUT_SR, q)
        else:
            write_ogg(folder / "song.ogg", mix.astype(np.float32), OUT_SR, q)
            (folder / "guitar.ogg").unlink(missing_ok=True)
        diff = self.guitar_difficulty(song)
        prev = preview_start(mono, OUT_SR, downs, audio_s)
        ini = song_ini(meta, audio_s, prev, diff)
        old_ini = parse_ini(folder / "song.ini") if (folder / "song.ini").exists() else {}
        video = None
        if self.cfg.get("keep_video", True) and Path(path).suffix.lower() in FFMPEG_EXT:
            try:
                video = save_video(path, folder)           # a music video becomes the background video
            except Exception:
                video = None
            if video and video["codec"] not in CH_VIDEO_OK and self.cfg.get("convert_video", True):
                try:                                       # VP9 / AV1 / HEVC: Clone Hero can't play it
                    convert_video(folder)
                except Exception:                          # keep the copy; Song details offers it again
                    pass
        start = video["start_ms"] if video else (
            old_ini.get("video_start_time") if find_first(folder, VIDEO_FILES) else None)
        if start not in (None, "", 0, "0"):
            ini += f"video_start_time = {start}\n"
        (folder / "song.ini").write_text(ini, encoding="utf-8")
        if not find_art(folder):
            art = cover_art(path)                          # the cover from the file's tags, if it has one
            if art:
                (folder / ("album.png" if art[:4] == b"\x89PNG" else "album.jpg")).write_bytes(art)
        try:
            refresh_visuals(folder)                        # background and highway (unless the user chose)
        except Exception:
            pass
        (folder / "notes.chart").write_text(
            write_chart_text(song, name=meta.title, artist=meta.artist, charter="RiffChart"), encoding="utf-8")
        done("write")
        return Result(folder, meta, audio_s, {k: len(v) for k, v in song.tracks.items()}, diff, sec, play.filled_s)

    def analyse(self, mix: np.ndarray, stage=None, done=None) -> dict:
        """Audio at OUT_SR -> stems, tempo map, model inputs (everything charting needs, style-independent)."""
        stage, done = stage or (lambda _: None), done or (lambda _: None)
        mix = np.repeat(mix[:, :1], 2, 1) if mix.shape[1] == 1 else mix[:, :2]
        audio_s = len(mix) / OUT_SR
        stage("separate")
        stems = self.sep(soxr.resample(mix, OUT_SR, self.sep.sr) if self.sep.sr != OUT_SR else mix)
        done("separate")
        stage("beats")
        mono = mix.mean(1)
        m22 = soxr.resample(mono, OUT_SR, MEL_SR).astype(np.float32)
        beats_raw, downs_raw = self.tracker(m22, MEL_SR)
        o = TEMPO_DEFAULTS
        tm = beats_to_tempo_map(beats_raw, downs_raw, o["resolution"], mode=o["mode"], tol_s=o["tol_s"],
                                ts_mode=o["ts_mode"])
        done("beats")
        return {"mix": mix, "audio_s": audio_s, "mono": mono, "m22": m22, "stems": stems,
                "beats": tm.beat_times(audio_s + 2), "downs": tm.downbeat_times(audio_s + 2)}

    def chart_analysis(self, an: dict, difficulties=DIFFICULTIES, style: Style | None = None):
        """-> (post-processed Song, Playable, model features)."""
        style = style or Style()
        follow = style.follow or self.cfg.get("follow", "guitar")
        play = playable_part(an["stems"], self.sep.sr, follow)
        g22 = soxr.resample(play.model_in.mean(1), self.sep.sr, MEL_SR).astype(np.float32)
        m22 = an["m22"]
        n = min(len(m22), len(g22))
        feats = np.stack([self.mel(g22[:n]), self.mel(m22[:n])], 1).astype(np.float16)
        pitch = None
        if self.charter.use_pitch:                     # pitch-aware model: CQT of what the player follows
            from ..model.pitch_feat import cqt_uint8
            pitch = cqt_uint8(g22[:n])[: len(feats)]
        d = self.cfg["decoding"]
        diffs = [x for x in DIFFICULTIES if x in difficulties]
        beats, downs, audio_s = an["beats"], an["downs"], an["audio_s"]
        ev = self.charter.chart(feats, beats, downs, diffs, d["temperature"], d["top_p"], seed=d["seed"],
                                style=style.tokens(), audio_s=audio_s,
                                pos_bias={"expert": DENSER_BIAS * max(style.density, 0.0)},
                                min_gap_s=self.min_gap if d["min_gap"] == "auto" else None, pitch=pitch)
        song = events_to_song(ev, beats, downs)
        song = postprocess(song, self.post_cfg, audio_s=audio_s, mel=feats, beats=beats, downbeats=downs,
                           density=style.density)
        return song, play, feats
