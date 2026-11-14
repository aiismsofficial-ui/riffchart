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

import logging
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
from .follow import MODES as FOLLOW_MODES, playable_part, INSTRUMENT_FILES, instrument_parts
from .difficulty import estimate as estimate_difficulty, load_model as load_difficulty_model
from .meta import SongMeta, cover_art, read_meta
from .visuals import VIDEO_FILES, convert_video, find_art, find_first, refresh_visuals, save_video, wants_convert

log = logging.getLogger(__name__)

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
    lyric_words: int = 0             # words of lyrics written to the chart (product/lyrics.py)


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


def song_ini(meta: SongMeta, audio_s: float, preview_s: float, diff: int, part_diffs: dict | None = None) -> str:
    """part_diffs: {"diff_bass": n, "diff_keys": n} for the extra parts the chart has."""
    lines = ["[song]", f"name = {meta.title}", f"artist = {meta.artist}"]
    if meta.album:
        lines.append(f"album = {meta.album}")
    if meta.genre:
        lines.append(f"genre = {meta.genre}")
    if meta.year:
        lines.append(f"year = {meta.year}")
    lines += ["charter = RiffChart", f"song_length = {int(audio_s * 1000)}",
              f"preview_start_time = {int(preview_s * 1000)}", f"diff_guitar = {diff}"]
    for k, v in (part_diffs or {}).items():
        lines.append(f"{k} = {v}")
    lines += ["delay = 0",
              "loading_phrase = Charted automatically by RiffChart"]
    return "\n".join(lines) + "\n"


def _set_ini(path: Path, key: str, value: str) -> None:
    """Set one song.ini key, keeping every other line as it is."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else ["[song]"]
    out = [ln for ln in lines if not ("=" in ln and ln.split("=", 1)[0].strip().lower() == key)]
    out.insert(1 if out and out[0].strip().startswith("[") else 0, f"{key} = {value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _sync_mid(folder: Path) -> None:
    """notes.mid follows notes.chart when the song has pitched vocals (product/vocals.py)."""
    try:
        from .vocals import sync_mid
        sync_mid(folder)
    except Exception as e:
        log.warning("notes.mid not updated for %s: %s", Path(folder).name, e)


class RiffCharter:
    def __init__(self, config: str | Path = "configs/product.yaml", device: str | None = None):
        # NVIDIA (CUDA) and AMD (ROCm builds of PyTorch) GPUs both show up as "cuda"; otherwise the CPU
        device = device or self.cfg_device(config) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.config_path = config
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
        from .bass import PARTS
        split = self.cfg["audio"].get("split_instruments", True)
        # each extra part is charted from what its own track will hold (keys: minus what the guitar part took)
        own = instrument_parts(an["stems"], play, self.sep.sr) if split else {}
        part_diffs = {}
        for part, spec in PARTS.items():
            audio = own.get(spec["file"][:-4]) if split else an["stems"].get(spec["stem"])
            if audio is None:
                continue
            stage(part)
            tracks = self.part_tracks(part, an, audio, song)
            if tracks:
                song.parts[part] = tracks
                part_diffs[spec["ini"]] = self.guitar_difficulty(Song(song.resolution, song.tempo, tracks))
            done(part)
        mix, audio_s, mono, downs = an["mix"], an["audio_s"], an["mono"], an["downs"]

        stage("write")
        folder.mkdir(parents=True, exist_ok=True)
        q = self.cfg["audio"]["vorbis_quality"]
        # every track written under a temporary name first, then swapped in: a failure while writing (full
        # disk, out of memory) leaves a re-charted song's old tracks as they were
        new: dict[str, Path] = {}
        try:
            if stems:
                parts = {"guitar": play.audio}
                if split:
                    parts.update(own)
                k = len(mix)
                rest = mix.astype(np.float32).copy()
                for name, a in parts.items():
                    a = a if self.sep.sr == OUT_SR else soxr.resample(a, self.sep.sr, OUT_SR)
                    k = min(k, len(a))
                    parts[name] = a
                rest = rest[:k]
                for name, a in parts.items():              # song.ogg = what's left: every file sums to the mix
                    rest -= a[:k]
                    new[f"{name}.ogg"] = folder / f".{name}.new.ogg"
                    write_ogg(new[f"{name}.ogg"], a[:k].astype(np.float32), OUT_SR, q)
                new["song.ogg"] = folder / ".song.new.ogg"
                write_ogg(new["song.ogg"], rest, OUT_SR, q)
            else:
                new["song.ogg"] = folder / ".song.new.ogg"
                write_ogg(new["song.ogg"], mix.astype(np.float32), OUT_SR, q)
        except BaseException:                              # (t is the stage timer: other names here)
            for tmp_file in new.values():
                tmp_file.unlink(missing_ok=True)
                tmp_file.with_suffix(".tmp.ogg").unlink(missing_ok=True)
            raise
        for f in ("guitar.ogg",) + tuple(f"{n}.ogg" for n in INSTRUMENT_FILES.values()):
            if f not in new:
                (folder / f).unlink(missing_ok=True)       # re-charting: no stems left from the last settings
        for name, tmp_file in new.items():
            tmp_file.replace(folder / name)
        diff = self.guitar_difficulty(song)
        prev = preview_start(mono, OUT_SR, downs, audio_s)
        ini = song_ini(meta, audio_s, prev, diff, part_diffs)
        old_ini = parse_ini(folder / "song.ini") if (folder / "song.ini").exists() else {}
        video = None
        if self.cfg.get("keep_video", True) and Path(path).suffix.lower() in FFMPEG_EXT:
            try:
                video = save_video(path, folder)           # a music video becomes the background video
            except Exception:
                video = None
            if video and self.cfg.get("convert_video", True) and wants_convert(
                    video, self.cfg.get("video_webm", True)):
                try:                                       # VP9 / AV1 / HEVC (and H.264 mp4): VP8 webm
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
        done("write")
        words = self.add_lyrics(song, folder, meta, an, audio_s, stage, done)
        if not song.sections and self.cfg.get("sections", "auto") == "auto":
            song.sections = self.auto_sections(song, folder, an, play)
        (folder / "notes.chart").write_text(
            write_chart_text(song, name=meta.title, artist=meta.artist, charter="RiffChart"), encoding="utf-8")
        _sync_mid(folder)
        return Result(folder, meta, audio_s, {k: len(v) for k, v in song.tracks.items()}, diff, sec, play.filled_s,
                      words)

    def update_song(self, folder: str | Path, on_stage=None) -> dict:
        """Bring a library song up to what a fresh chart gets now, without touching its notes:
          instruments  bass / drums / vocals / keys tracks split out of song.ogg (the band without the guitar
                       track), when the song has none yet; song.ogg becomes what's left, so the mix is unchanged.
                       The old song.ogg is kept in .riffchart/song-before-update.ogg.
          lyrics       looked up and timed to the singing when the song has none (Settings: lyrics online)
          video        a music video that isn't VP8 webm yet converted to it (video_webm setting)
          background   the cover background and an animated background made again at the sharpness setting
        Returns {"instruments": [files written], "lyrics": words or 0, "background": bool, "notes": [...]}
        (+ "bass" / "keys" note counts and "video": True when they were done)."""
        from . import lyrics
        from .visuals import bg_anim, bg_mode, make_bg_anim, refresh_background
        from ..formats.loader import load_song
        from .. import progress
        folder = Path(folder)
        stage = on_stage or (lambda *_: None)
        notes: list[str] = []
        out = {"instruments": [], "lyrics": 0, "background": False, "notes": notes}
        song_ogg = folder / "song.ogg"
        ini = parse_ini(folder / "song.ini") if (folder / "song.ini").exists() else {}
        has_parts = [f"{n}.ogg" for n in INSTRUMENT_FILES.values() if (folder / f"{n}.ogg").exists()]
        want_parts = self.cfg["audio"].get("split_instruments", True) and not has_parts and song_ogg.exists()
        lmode = lyrics.mode(self.config_path)
        named = bool(ini.get("artist") and ini.get("name"))
        from .update import wants_lyrics
        want_lyrics = wants_lyrics(folder, ini) and (
            (lmode == "online" and (named or lyrics.can_transcribe())) or (lmode == "local" and lyrics.can_transcribe()))
        if has_parts:
            notes.append("already has its own instrument tracks")
        elif not song_ogg.exists():
            notes.append("no song.ogg to split")
        stems = None
        if want_parts:                                      # (lyrics separate the whole song: lyrics_audio)
            stage("load")
            mix, _ = load_audio(song_ogg, OUT_SR, mono=False)
            mix = np.repeat(mix[:, :1], 2, 1) if mix.shape[1] == 1 else mix[:, :2]
            stage("separate")
            progress.set_hook((lambda f: stage("separate", f)) if on_stage else None)
            try:
                stems = self.sep(soxr.resample(mix, OUT_SR, self.sep.sr) if self.sep.sr != OUT_SR else mix)
            finally:
                progress.set_hook(None)
        if want_parts and stems is not None:
            stage("write")
            q = self.cfg["audio"]["vorbis_quality"]
            ref = float(np.sqrt(np.mean(mix ** 2))) or 1e-9
            parts = {}
            for name, file in INSTRUMENT_FILES.items():
                if name not in stems:
                    continue
                a = stems[name].astype(np.float32)
                a = a if self.sep.sr == OUT_SR else soxr.resample(a, self.sep.sr, OUT_SR)
                if float(np.sqrt(np.mean(a ** 2))) < 0.02 * ref:      # too quiet to be its own track
                    continue
                parts[file] = a
            if parts:
                k = min([len(mix)] + [len(a) for a in parts.values()])
                rest = mix[:k].astype(np.float32).copy()
                keep = folder / ".riffchart" / "song-before-update.ogg"
                keep.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(song_ogg, keep)
                tmp = {f"{file}.ogg": folder / f".{file}.update.ogg" for file in parts}
                tmp["song.ogg"] = folder / ".song.update.ogg"
                try:                                        # all written first, then swapped in together: a
                    for file, a in parts.items():           # full disk never leaves bass.ogg beside a song.ogg
                        rest -= a[:k]                       # that still has the bass in it
                        write_ogg(tmp[f"{file}.ogg"], a[:k], OUT_SR, q)
                    write_ogg(tmp["song.ogg"], rest, OUT_SR, q)
                except BaseException:
                    for t in tmp.values():
                        t.unlink(missing_ok=True)
                        t.with_suffix(".tmp.ogg").unlink(missing_ok=True)
                    raise
                done: list[str] = []
                try:                                        # a file the game or a browser has open can't be
                    for name, t in tmp.items():             # replaced on Windows: then undo the swap, or the
                        t.replace(folder / name)            # parts would play twice (in song.ogg and their own)
                        done.append(name)
                except BaseException:
                    for name in done:
                        if name == "song.ogg":
                            shutil.copy2(keep, song_ogg)
                        else:
                            (folder / name).unlink(missing_ok=True)
                    for t in tmp.values():
                        t.unlink(missing_ok=True)
                    raise
                out["instruments"] += [n for n in done if n != "song.ogg"]
            else:
                notes.append("no instrument loud enough for its own track")
        from . import bass as bass_mod
        for part, spec in bass_mod.PARTS.items():
            if not (bass_mod.enabled(self.config_path, part) and (folder / "notes.chart").exists()
                    and bass_mod.ours(ini) and not bass_mod.in_chart(folder, part)):
                continue
            if (folder / spec["file"]).exists():
                stage(part)
                try:
                    out[part] = self._update_part(folder, ini, stems, part)
                    if not out[part]:
                        notes.append(f"the {part} part is too quiet to chart")
                except Exception as e:                      # the rest of the update still counts
                    log.warning("%s chart skipped for %s: %s", part, folder.name, e)
                    notes.append(f"{part} chart skipped ({type(e).__name__})")
            elif part == "bass":
                notes.append("no bass track to chart")
        if want_lyrics:
            stage("lyrics")
            try:
                whole = stems if stems is not None and not (folder / "guitar.ogg").exists() else None
                voc, vsr, lmix = self.lyrics_audio(folder, whole, mix if whole is not None else None)
                audio_s = len(voc) / vsr
                lyrics.make(folder, ini.get("artist", ""), ini.get("name", ""), voc, vsr, audio_s,
                            online=lmode == "online", device=self.device, mix=lmix)
                from . import chartfile
                with chartfile.lock(folder):
                    song, _, chart_path = load_song(folder)
                    if chart_path.suffix.lower() == ".chart":
                        words = lyrics.attach(song, folder, ini, audio_s)
                        if words:
                            chartfile.rewrite(folder, chart_path, song, ini, "lyrics")
                            out["lyrics"] = words
                            _sync_mid(folder)
                    else:
                        notes.append("lyrics are only written into .chart songs")
                if not out["lyrics"] and not notes[-1:] == ["lyrics are only written into .chart songs"]:
                    notes.append("no lyrics found" + (" online" if lmode == "online" and not lyrics.can_transcribe() else ""))
            except Exception as e:                          # offline, LRCLIB down: the rest still counts
                log.warning("lyrics skipped for %s: %s", folder.name, e)
                notes.append(f"lyrics skipped ({type(e).__name__})")
        elif lyrics.load(folder) is not None:
            notes.append("already has lyrics")
        from .update import settings as update_settings
        from .visuals import video_info
        if update_settings()[2] and not bg_anim(folder) and wants_convert(video_info(folder)):
            stage("video")                                 # the music video as VP8 webm (plays everywhere)
            try:
                convert_video(folder, (lambda f: stage("video", f)) if on_stage else None)
                out["video"] = True
            except Exception as e:                         # the .mp4 stays; Song details offers it again
                log.warning("video conversion skipped for %s: %s", folder.name, e)
                notes.append(f"video conversion skipped ({type(e).__name__})")
        stage("visuals")
        try:
            anim = bg_anim(folder)
            if bg_mode(folder) == "auto":
                refresh_background(folder, wait=True)      # also redraws an animated background over it
                out["background"] = True
            elif anim:
                make_bg_anim(folder, anim)
                out["background"] = True
        except Exception as e:
            log.warning("background skipped for %s: %s", folder.name, e)
            notes.append(f"background skipped ({type(e).__name__})")
        from .update import mark
        mark(folder, out)
        return out

    def auto_sections(self, song, folder: Path, an: dict, play) -> list[tuple[int, str]]:
        """Intro / Verse / Chorus / ... names at the section starts (product/structure.py). Never fails a chart."""
        from . import lyrics, structure
        try:
            r = lambda a: a if self.sep.sr == OUT_SR else soxr.resample(a, self.sep.sr, OUT_SR)
            return structure.auto_sections(song, an["mix"], OUT_SR, vocals=r(an["stems"]["vocals"]),
                                           guitar=r(play.audio), lines=lyrics.load(folder))
        except Exception as e:
            log.warning("sections skipped for %s: %s", folder.name, e)
            return []

    def add_lyrics(self, song, folder: Path, meta, an: dict, audio_s: float, stage, done) -> int:
        """Lyrics timed to the singing (`lyrics: online` looks them up on LRCLIB, else transcribes the vocal track
        here when Whisper is installed; `local` only transcribes). A song re-charted in place
        keeps the lyrics it already has. Never fails the chart: lyrics are a bonus."""
        from . import lyrics
        try:
            lmode = lyrics.mode(self.config_path)                               # Settings may change it
            if lyrics.load(folder) is None and lmode != "off":
                stage("lyrics")
                mix = an["mix"] if self.sep.sr == OUT_SR else soxr.resample(an["mix"], OUT_SR, self.sep.sr)
                lyrics.make(folder, meta.artist, meta.title, an["stems"]["vocals"], self.sep.sr, audio_s,
                            online=lmode == "online", device=self.device, mix=mix)
                done("lyrics")
            return lyrics.attach(song, folder, audio_s=audio_s)
        except Exception as e:                             # offline, LRCLIB down, odd data
            log.warning("lyrics skipped for %s: %s", folder.name, e)
            return 0

    def analyse(self, mix: np.ndarray, stage=None, done=None) -> dict:
        """Audio at OUT_SR -> stems, tempo map, model inputs (everything charting needs, style-independent)."""
        stage, done = stage or (lambda _: None), done or (lambda _: None)
        mix = np.repeat(mix[:, :1], 2, 1) if mix.shape[1] == 1 else mix[:, :2]
        audio_s = len(mix) / OUT_SR
        if audio_s < 5:                                    # an empty or broken file decodes to (almost) nothing
            raise ValueError(f"the file holds only {audio_s:.1f} s of audio: it may be empty or damaged")
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
        song, feats = self.chart_part(an, play.model_in, difficulties, style)
        return song, play, feats

    def lyrics_audio(self, folder: Path, stems: dict | None = None, mix: np.ndarray | None = None):
        """(vocals, sr, mix at sr): the singing to time or write down lyrics from, separated from the WHOLE song.
        Not from vocals.ogg / song.ogg alone: where the guitar is quiet the guitar track borrows the melodic
        parts, vocals included (follow.py), so those files are missing whole verses. stems / mix: an existing
        separation of the whole song (at self.sep.sr / OUT_SR), reused when given."""
        if stems is None:
            files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in (".ogg", ".opus", ".mp3", ".wav")
                     and f.stem.lower() in {"song", "guitar"} | set(INSTRUMENT_FILES.values())]
            from ..audio.io import load_mix
            mix = load_mix(files, OUT_SR, mono=False)
            mix = np.repeat(mix[:, :1], 2, 1) if mix.shape[1] == 1 else mix[:, :2]
            stems = self.sep(soxr.resample(mix, OUT_SR, self.sep.sr) if self.sep.sr != OUT_SR else mix)
        vsr = self.sep.sr
        return stems["vocals"], vsr, (soxr.resample(mix, OUT_SR, vsr) if vsr != OUT_SR else mix)

    def _update_part(self, folder: Path, ini: dict, stems: dict | None, part: str = "bass") -> int:
        """A bass (or keys) chart for a charted song from its bass.ogg (keys.ogg), on the chart's own tempo map
        (the beats aren't tracked again); written into notes.chart (old one in history). Returns Expert's note
        count, 0 = none."""
        from .bass import PARTS
        spec = PARTS[part]
        from ..audio.io import load_mix
        from ..formats.loader import load_song
        from ..formats.textio import song_delay_s
        song, _, chart_path = load_song(folder)
        if chart_path.suffix.lower() != ".chart":            # a notes.mid the game reads first: never write
            raise RuntimeError("the song's notes.mid is read before notes.chart, so a part can't be added")
        files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in (".ogg", ".opus", ".mp3", ".wav")
                 and f.stem.lower() in {"song", "guitar"} | set(INSTRUMENT_FILES.values())]
        mix = load_mix(files, OUT_SR, mono=False)                   # the whole song: every track summed
        if stems is not None and part == "bass" and "bass" in stems:     # just separated; keys come from the
            audio = stems["bass"]                                         # file (the guitar part's share is out)
        else:
            audio, _ = load_audio(folder / spec["file"], self.sep.sr, mono=False)
        shift = song.chart_offset_s + song_delay_s(ini)             # audio time = chart time + shift
        audio_s = len(mix) / OUT_SR
        an = {"mix": mix, "audio_s": audio_s, "m22": soxr.resample(mix.mean(1), OUT_SR, MEL_SR).astype(np.float32),
              "beats": song.tempo.beat_times(audio_s + 2) + shift, "downs": song.tempo.downbeat_times(audio_s + 2) + shift}
        tracks = self.part_tracks(part, an, audio, song, shift_s=shift)
        if not tracks or not len(tracks.get("expert", [])):
            return 0
        from . import chartfile
        with chartfile.lock(folder):                 # the chart as it is now: an editor save made while the
            now, _, now_path = load_song(folder)    # part was charted is kept
            if now_path != chart_path or not chartfile.same_timing(now, song):
                raise RuntimeError("the chart's tempo map changed while the part was charted: run it again")
            if any(len(t) for t in now.parts.get(part, {}).values()):
                return len(now.parts[part].get("expert", []))      # added meanwhile (another job, the editor)
            now.parts[part] = tracks
            chartfile.rewrite(folder, chart_path, now, ini, part)
            _set_ini(folder / "song.ini", spec["ini"], str(self.guitar_difficulty(Song(song.resolution, song.tempo, tracks))))
            _sync_mid(folder)
        return len(tracks["expert"])

    def part_tracks(self, part: str, an: dict, audio: np.ndarray, onto: Song, shift_s: float = 0.0) -> dict | None:
        """A bass / keys chart from that part's track (product/bass.py), on `onto`'s tempo map; None when the
        setting is off or the part is too quiet to chart."""
        from . import bass as bs
        c = self.cfg.get(part, {})
        if not bs.enabled(self.config_path, part) or not bs.loud_enough(audio, an["mix"]):
            return None
        song, _ = self.chart_part(an, audio, style=Style(density=float(c.get("density", 0.0))))
        if c.get("singles", False):
            tracks = bs.singles(song.tracks)
        else:
            tracks = song.tracks
            for tr in tracks.values():
                bs.fix_same_fret_hopos(tr)
        return bs.retime(tracks, song, onto, shift_s)

    def chart_part(self, an: dict, part: np.ndarray, difficulties=DIFFICULTIES, style: Style | None = None,
                   post_cfg: dict | None = None, lower: list[str] | None = None):
        """The chart model on `part` (audio at the separator's rate: the guitar, or the bass track for a bass
        chart) with the full mix as its second input -> (post-processed Song, model features)."""
        style = style or Style()
        g22 = soxr.resample(part.mean(1) if part.ndim == 2 else part, self.sep.sr, MEL_SR).astype(np.float32)
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
        song = postprocess(song, post_cfg or self.post_cfg, audio_s=audio_s, mel=feats, beats=beats,
                           downbeats=downs, density=style.density, lower=lower)
        return song, feats
