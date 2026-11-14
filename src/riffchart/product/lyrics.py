"""Lyrics for charted songs, shown by Clone Hero and YARG.

Text and timing come from LRCLIB (lrclib.net, a free community lyrics database; only the artist, title and
length are sent, and only when `lyrics: online` is set). Many entries are timed word by word; others only by
line, and then the words are spread over the line by length. The user's file is often a different cut from the
one the lyrics were timed on (a video intro, a radio edit), so the whole set is shifted to fit the separated
vocal track: we try every shift in +-20 s and keep the one that puts the word starts on vocal onsets.

The result is kept in seconds of the song's own audio in `.riffchart/lyrics.json`, and turned into .chart
events (`phrase_start`, `lyric <word>`, `phrase_end`) whenever the chart is written, so tempo edits and sync
nudges in the editor never move the words away from the singing.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .covers import USER_AGENT

API = "https://lrclib.net/api"
TIMEOUT = 15
FILE = "lyrics.json"
MAX_SHIFT_S = 20.0
HOP_S = 0.01


@dataclass
class Line:
    start: float
    end: float
    words: list[tuple[float, str]] = field(default_factory=list)


# ---------------------------------------------------------------- fetching

def _get(path: str, params: dict) -> object | None:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def fetch(artist: str, title: str, duration_s: float | None = None) -> dict | None:
    """The best LRCLIB record with timed lyrics for this song, or None."""
    params = {"artist_name": artist, "track_name": title}
    if duration_s:
        params["duration"] = int(round(duration_s))
    rec = _get("get", params)
    if isinstance(rec, dict) and _timed(rec):
        return rec
    found: list = []
    for q in ({"track_name": title, "artist_name": artist}, {"q": f"{artist} {title}"}, {"track_name": title}):
        res = _get("search", q)
        if isinstance(res, list) and res:
            found = res
            break
    best, best_score = None, -1e9
    for r in found:
        if r.get("instrumental") or not _timed(r):
            continue
        same_artist = _norm(r.get("artistName", "")) == _norm(artist)
        same_title = _norm(r.get("trackName", "")) == _norm(title)
        close = bool(duration_s and r.get("duration") and abs(float(r["duration"]) - duration_s) <= 3)
        if not same_title or not (same_artist or close):  # a title alone matches too many songs
            continue
        score = 3 * same_artist + 2 * close + bool(r.get("hasWordSync"))
        if duration_s and r.get("duration"):
            score -= abs(float(r["duration"]) - duration_s) / 10
        if score > best_score:
            best, best_score = r, score
    if best is None:
        return None
    if best.get("hasWordSync") and not best.get("lyricsfile"):
        best = _get("get/" + str(best["id"]), {}) or best
    return best


def _timed(rec: dict) -> bool:
    return bool(rec.get("lyricsfile") or rec.get("syncedLyrics"))


# ---------------------------------------------------------------- parsing

def parse(rec: dict) -> list[Line]:
    """Lines with word times in seconds. Word-timed `lyricsfile` first, then line-timed `syncedLyrics`."""
    lines: list[Line] = []
    if rec.get("lyricsfile"):
        import yaml
        try:
            doc = yaml.safe_load(rec["lyricsfile"]) or {}
        except yaml.YAMLError:
            doc = {}
        for ln in doc.get("lines") or []:
            words = [(w["start_ms"] / 1000, str(w.get("text", "")).strip()) for w in ln.get("words") or []
                     if "start_ms" in w and str(w.get("text", "")).strip()]
            if not words and ln.get("text") and "start_ms" in ln:
                words = _spread(ln["start_ms"] / 1000, ln.get("end_ms", ln["start_ms"] + 3000) / 1000, ln["text"])
            if words:
                end = ln.get("end_ms")
                lines.append(Line(words[0][0], end / 1000 if end else words[-1][0] + 0.5, words))
    if not lines and rec.get("syncedLyrics"):
        stamps = []
        for raw in rec["syncedLyrics"].splitlines():
            m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", raw)
            if m:
                stamps.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
        for i, (t, text) in enumerate(stamps):
            if not text:
                continue
            nxt = stamps[i + 1][0] if i + 1 < len(stamps) else t + 4.0
            end = min(nxt, t + max(1.0, 0.33 * len(text.split()) + 0.5))
            lines.append(Line(t, end, _spread(t, end, text)))
    return lines


def _spread(start: float, end: float, text: str) -> list[tuple[float, str]]:
    """Words of a line timed only by its start: spaced by their length over the line."""
    words = text.split()
    if not words:
        return []
    weight = np.array([len(w) + 2 for w in words], float)
    starts = start + (end - start) * np.concatenate([[0], np.cumsum(weight)[:-1]]) / weight.sum()
    return [(float(t), w) for t, w in zip(starts, words)]


# ---------------------------------------------------------------- fitting to the vocal track

def vocal_onsets(vocals: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """(onset strength, singing activity) at HOP_S, from the separated vocal stem."""
    x = vocals.mean(axis=1) if vocals.ndim == 2 else vocals
    hop = int(sr * HOP_S)
    n = len(x) // hop
    if n < 10:
        return np.zeros(1), np.zeros(1)
    e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-10)
    db = 20 * np.log10(e)
    db = np.convolve(db, np.ones(3) / 3, mode="same")
    onset = np.maximum(0.0, np.diff(db, prepend=db[0]))
    onset = np.maximum.reduce([np.roll(onset, k) for k in (-3, -2, -1, 0, 1, 2, 3)])   # +-30 ms tolerance
    active = (db > np.percentile(db, 60) - 6).astype(float)
    onset /= onset.max() + 1e-9
    return onset, active


def fit_shift(lines: list[Line], vocals: np.ndarray, sr: int, max_shift: float = MAX_SHIFT_S) -> tuple[float, float]:
    """(shift in seconds to add to every lyric time, confidence z-score of that shift). `vocals` must be the
    separated vocal stem: on a full mix the drums' onsets win and the fit lands anywhere."""
    starts = np.array([t for ln in lines for t, _ in ln.words])
    if len(starts) < 8:
        return 0.0, 0.0
    onset, active = vocal_onsets(vocals, sr)
    n = len(onset)
    shifts = np.arange(-max_shift, max_shift + HOP_S / 2, HOP_S)
    idx = np.round(starts / HOP_S).astype(int)[None, :] + np.round(shifts / HOP_S).astype(int)[:, None]
    inside = (idx >= 0) & (idx < n)
    idx = np.clip(idx, 0, n - 1)
    hit = np.where(inside, onset[idx] + 0.5 * active[idx], 0.0)
    score = hit.mean(axis=1) * inside.mean(axis=1)       # words pushed off the song's ends count as misses
    best = int(score.argmax())
    z = float((score[best] - np.median(score)) / (score.std() + 1e-9))
    return float(shifts[best]), z


def shifted(lines: list[Line], s: float) -> list[Line]:
    return [Line(ln.start + s, ln.end + s, [(t + s, w) for t, w in ln.words]) for ln in lines]


# ---------------------------------------------------------------- storing and writing

def save(folder: Path, lines: list[Line], info: dict) -> None:
    d = Path(folder) / ".riffchart"
    d.mkdir(parents=True, exist_ok=True)
    data = {**info, "lines": [{"start": round(ln.start, 3), "end": round(ln.end, 3),
                               "words": [[round(t, 3), w] for t, w in ln.words]} for ln in lines]}
    (d / FILE).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load(folder: Path) -> list[Line] | None:
    p = Path(folder) / ".riffchart" / FILE
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return [Line(float(ln["start"]), float(ln["end"]), [(float(t), str(w)) for t, w in ln["words"]])
                for ln in d.get("lines", [])]
    except (ValueError, KeyError, TypeError):
        return None


_CH_SPECIAL = re.compile(r'["#^*%/+$_§=\[\]{}<>]')


def _word(w: str) -> str:
    """Clone Hero reads some symbols as markup (- joins syllables, = is a hyphen, # ^ * are vocal marks)."""
    w = _CH_SPECIAL.sub("", w).strip()
    return w.rstrip("-") if w.endswith("-") else w


def to_events(lines: list[Line], tempo, offset_s: float = 0.0, end_s: float | None = None) -> list[tuple[int, str]]:
    """.chart [Events] entries. offset_s: audio time = chart time + offset_s."""
    ev: list[tuple[int, str]] = []
    last_end = -1
    for ln in lines:
        words = [(t, _word(w)) for t, w in ln.words]
        words = [(t, w) for t, w in words if w and (end_s is None or t < end_s)]
        if not words:
            continue
        tick = lambda s: int(round(float(tempo.sec_to_tick(max(0.0, s - offset_s)))))
        start = max(tick(min(ln.start, words[0][0])), last_end + 1)
        end = max(tick(ln.end), tick(words[-1][0]) + 1)
        ev.append((start, "phrase_start"))
        prev = start - 1
        for t, w in words:
            k = max(tick(t), prev + 1, start)
            ev.append((k, f"lyric {w}"))
            prev = k
        end = max(end, prev + 1)
        ev.append((end, "phrase_end"))
        last_end = end
    return ev


def summary(folder: Path) -> dict | None:
    """{words, source} of the folder's saved lyrics, for the song details panel."""
    p = Path(folder) / ".riffchart" / FILE
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return {"words": sum(len(ln.get("words", [])) for ln in d.get("lines", [])), "source": d.get("source", ""),
            "word_sync": bool(d.get("word_sync")), "shift_s": d.get("shift_s", 0)}


def remove(folder: Path) -> None:
    (Path(folder) / ".riffchart" / FILE).unlink(missing_ok=True)


def attach(song, folder: Path, ini: dict | None = None, audio_s: float | None = None) -> int:
    """Put the folder's saved lyrics on `song` as events, for the chart writer. Returns the word count."""
    lines = load(folder)
    if not lines:
        return 0
    delay = 0.0
    try:
        delay = float((ini or {}).get("delay", 0) or 0) / 1000
    except ValueError:
        pass
    song.lyrics = to_events(lines, song.tempo, song.chart_offset_s + delay, audio_s)
    return sum(1 for _, e in song.lyrics if e.startswith("lyric "))


# ---------------------------------------------------------------- transcribing on this PC (no lyrics online)
MIN_WORDS = 20            # fewer words than this from the vocal track: an instrumental, or noise; don't keep it


MODES = ("online", "local", "off")   # online: lrclib.net, else transcribe here; local: only transcribe here


def mode(config: str = "configs/product.yaml") -> str:
    from ..paths import load_config
    m = str(load_config(config).get("lyrics", "online"))
    return m if m in MODES else "online"


def sung_enough(vocals: np.ndarray, mix: np.ndarray | None) -> bool:
    """False for an instrumental: separation leaves a vocal track 14 dB+ below the mix (product/structure.py)."""
    if mix is None:
        return True
    def p95(a):
        a = a.mean(1) if a.ndim == 2 else a
        n = len(a) // 2048 * 2048
        r = np.sqrt((a[:n].reshape(-1, 2048) ** 2).mean(1) + 1e-10)
        return float(np.percentile(20 * np.log10(r), 95))
    return p95(vocals) - p95(mix) > -14.0


def can_transcribe() -> bool:
    """openai-whisper installed (the optional `transcribe` extra)."""
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(vocals: np.ndarray, sr: int, model: str = "turbo", device: str | None = None) -> list[Line]:
    """Words and their times from the separated vocal track with Whisper, run here (the model is downloaded
    once to data/checkpoints/whisper). Segments that look like silence or a hallucinated loop are dropped."""
    import soxr
    import torch
    import whisper
    from ..paths import project_path
    y = vocals.mean(1) if vocals.ndim == 2 else vocals
    y16 = soxr.resample(y.astype(np.float32), sr, 16000) if sr != 16000 else y.astype(np.float32)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu" and model in ("turbo", "large", "large-v3", "large-v3-turbo", "medium"):
        model = "small"                                  # the big ones take many minutes a song without a GPU
    m = whisper.load_model(model, device=dev, download_root=str(project_path("data/checkpoints/whisper")))
    try:
        r = m.transcribe(y16, word_timestamps=True, condition_on_previous_text=False, fp16=dev == "cuda",
                         no_speech_threshold=0.6, logprob_threshold=-1.0, compression_ratio_threshold=2.4,
                         verbose=None)
    finally:
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    lines: list[Line] = []
    for seg in r.get("segments", []):
        if seg.get("no_speech_prob", 0) > 0.6 or seg.get("avg_logprob", 0) < -1.2 or \
                seg.get("compression_ratio", 1) > 2.4:
            continue
        words = [(float(w["start"]), w["word"].strip()) for w in seg.get("words", [])
                 if w["word"].strip() and w.get("probability", 1.0) > 0.2]
        if not words:
            continue
        text = " ".join(w for _, w in words).lower()
        if len(lines) >= 3 and all(" ".join(w for _, w in ln.words).lower() == text for ln in lines[-3:]):
            continue                                     # the same line a 4th time running: a loop
        lines.append(Line(min(float(seg["start"]), words[0][0]), float(seg["end"]), words))
    return lines


def make(folder: Path, artist: str, title: str, vocals: np.ndarray, sr: int, audio_s: float,
         online: bool = True, local: bool = True, device: str | None = None, mix: np.ndarray | None = None) -> dict:
    """Look the lyrics up and fit them to the vocals; with none online (or no names to look up), transcribe
    the vocal track on this PC when Whisper is installed. Saves them in the song folder; returns a report."""
    rec = fetch(artist, title, audio_s) if online and artist and title else None
    lines = parse(rec) if rec is not None else []
    if not lines:
        if not (local and can_transcribe()):
            return {"found": False}
        if not sung_enough(vocals, mix):
            return {"found": False, "instrumental": True}
        lines = transcribe(vocals, sr, device=device)
        n = sum(len(ln.words) for ln in lines)
        if n < MIN_WORDS:
            return {"found": False, "transcribed_words": n}
        info = {"source": "whisper", "word_sync": True, "shift_s": 0.0}
        save(folder, lines, info)
        return {"found": True, "words": n, **info}
    # Same length as the database's recording: the same cut, so only a small slip is possible, and searching
    # wide only finds chance matches (seen: +13.8 s on a song that needed -0.2 s). A different length (a video
    # intro, a radio edit) needs the wide search and a clearer winner. Unsure: keep the database's timing.
    same_cut = bool(rec.get("duration")) and abs(float(rec["duration"]) - audio_s) <= 3.0
    s, z = fit_shift(lines, vocals, sr, 3.0 if same_cut else MAX_SHIFT_S)
    if z < (2.5 if same_cut else 3.5):
        s = 0.0
    lines = shifted(lines, s)
    info = {"source": "lrclib", "id": rec.get("id"), "word_sync": bool(rec.get("hasWordSync")),
            "shift_s": round(s, 3), "fit_z": round(z, 2)}
    save(folder, lines, info)
    return {"found": True, "words": sum(len(ln.words) for ln in lines), **info}
