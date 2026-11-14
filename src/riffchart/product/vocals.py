"""Pitched vocals for YARG: the chart's timed lyrics become sung notes, at the pitch the singer hits in
vocals.ogg, in notes.mid's PART VOCALS (only .mid can carry vocals; YARG sings them, Clone Hero shows the
lyrics).

  .riffchart/vocal_pitch.npz  CREPE pitch track of vocals.ogg (made once, when the user turns this on)
  .riffchart/mid.json         marks notes.mid as made by RiffChart from notes.chart: notes.chart stays the
                              song's source (the editor and every re-write use it) and notes.mid is written
                              again after each change (sync_mid). Turning vocals off removes that notes.mid.

One note per word, from the word's start to the next word (or the phrase end); its pitch is the median of the
clearly voiced frames in the first 0.6 s. No clear pitch -> "#" (spoken: the word shows without a pitch line).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

PITCH = ".riffchart/vocal_pitch.npz"
MARK = ".riffchart/mid.json"
VOICED = 0.5
LOOK_S = 0.6


def enabled(folder: Path) -> bool:
    return (Path(folder) / PITCH).exists()


def ours(folder: Path) -> bool:
    """notes.mid was written by RiffChart from notes.chart (so notes.chart is the one to read and edit)."""
    return (Path(folder) / MARK).exists() and (Path(folder) / "notes.chart").exists()


def analyse(folder: Path, device: str = "cuda") -> None:
    """Pitch track of vocals.ogg, saved for every later notes.mid write."""
    from ..audio.io import load_audio
    from ..post.contour import track_pitch
    folder = Path(folder)
    if not (folder / "vocals.ogg").exists():
        raise FileNotFoundError("this song has no vocals.ogg: bring it up to date (or re-chart it) first")
    y, sr = load_audio(folder / "vocals.ogg", None, mono=True)
    pt = track_pitch(y, sr, device=device, model="full")
    p = folder / PITCH
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, midi=pt.midi.astype(np.float32), periodicity=pt.periodicity.astype(np.float32),
                        frame_s=np.float32(pt.frame_s))


def vocal_notes(song, pitch: dict | None, ini: dict | None = None) -> tuple[list, list]:
    """(notes [(start, end, word, midi or None)], phrases [(start, end)]) in ticks, from song.lyrics events.
    ini: song.ini, for its delay (audio time = chart time + Offset + delay, as lyrics.attach places words)."""
    from ..formats.textio import song_delay_s
    delay = song_delay_s(ini or {})
    notes, phrases = [], []
    words: list[tuple[int, str]] = []
    start = None

    def close(end: int):
        nonlocal words
        if not words:
            return
        end = max(end, words[-1][0] + 1)
        phrases.append((start if start is not None else words[0][0], end))
        for i, (t, w) in enumerate(words):
            nxt = words[i + 1][0] if i + 1 < len(words) else end
            notes.append((t, max(nxt, t + 1), w, _pitch(song, pitch, t, nxt, delay)))
        words = []

    for t, ev in sorted(song.lyrics, key=lambda e: (e[0], 0 if e[1] == "phrase_end" else 1 if e[1] == "phrase_start" else 2)):
        if ev == "phrase_start":
            close(t)
            start = t
        elif ev == "phrase_end":
            close(t)
            start = None
        elif ev.startswith("lyric "):
            w = ev[6:].strip()
            if w:
                words.append((t, w))
    close(words[-1][0] + song.resolution if words else 0)
    return notes, phrases


def _pitch(song, pitch: dict | None, a: int, b: int, delay_s: float = 0.0) -> float | None:
    if pitch is None:
        return None
    sa = float(song.tempo.tick_to_sec(np.array([a]))[0]) + song.chart_offset_s + delay_s
    sb = float(song.tempo.tick_to_sec(np.array([b]))[0]) + song.chart_offset_s + delay_s
    fs = float(pitch["frame_s"])
    f0, f1 = int((sa + 0.03) / fs), int(min(sb, sa + LOOK_S) / fs)
    f1 = min(f1, len(pitch["midi"]))
    if f1 <= f0:
        return None
    per = pitch["periodicity"][f0:f1]
    ok = per >= VOICED
    if ok.sum() < 3:
        return None
    return float(np.median(pitch["midi"][f0:f1][ok]))


def sync_mid(folder: Path) -> str:
    """After notes.chart changed: write notes.mid again (vocals on), or remove the one RiffChart made (off).
    Returns "written" / "removed" / ""."""
    from ..formats.chart_parser import parse_chart
    from ..formats.midi_writer import write_midi
    from ..formats.textio import parse_ini
    folder = Path(folder)
    chart, mid, mark = folder / "notes.chart", folder / "notes.mid", folder / MARK
    if enabled(folder) and chart.exists():
        ini = parse_ini(folder / "song.ini") if (folder / "song.ini").exists() else {}
        song = parse_chart(chart, ini)
        if not song.lyrics:
            return remove_mid(folder)
        with np.load(folder / PITCH) as z:
            pitch = {k: z[k] for k in z.files}
        new = folder / "notes.mid.new"
        try:
            if song.raw_sections:                              # drums, 6-fret...: the game reads notes.mid
                raise ValueError("the chart has parts a RiffChart notes.mid can't hold ("   # first, so they'd vanish
                                 + ", ".join(sorted(song.raw_sections)[:3]) + ")")
            write_midi(song, new, ini, name=ini.get("name", ""), vocals=vocal_notes(song, pitch, ini))
        except BaseException as e:
            new.unlink(missing_ok=True)
            new.with_suffix(".mid.tmp").unlink(missing_ok=True)     # write_midi's own temporary file
            if isinstance(e, ValueError):                      # can't be a .mid (offset, very slow tempo):
                remove_mid(folder)                             # never leave an older notes.mid to be played
            raise
        if mid.exists() and not mark.exists():                 # someone else's notes.mid: keep a copy
            bak = folder / "notes.mid.bak"
            mid.replace(bak if not bak.exists() else folder / time.strftime("notes.mid.%Y%m%d-%H%M%S.bak"))
        new.replace(mid)
        mark.parent.mkdir(parents=True, exist_ok=True)
        mark.write_text(json.dumps({"written": time.time()}), encoding="utf-8")
        return "written"
    return remove_mid(folder)


def refresh(folder: Path) -> str:
    """sync_mid when notes.chart changed after RiffChart last wrote notes.mid (an edit in Moonscraper, say), so
    the game, which reads notes.mid first, never gets a stale chart. Returns what sync_mid did, or ""."""
    folder = Path(folder)
    if not ours(folder):
        return ""
    mid = folder / "notes.mid"
    if mid.exists() and mid.stat().st_mtime >= (folder / "notes.chart").stat().st_mtime:
        return ""
    return sync_mid(folder)


def remove_mid(folder: Path) -> str:
    folder = Path(folder)
    if (folder / MARK).exists():
        (folder / "notes.mid").unlink(missing_ok=True)
        (folder / MARK).unlink(missing_ok=True)
        return "removed"
    return ""


def disable(folder: Path) -> None:
    (Path(folder) / PITCH).unlink(missing_ok=True)
    remove_mid(folder)


def summary(folder: Path) -> dict | None:
    """{notes, pitched} for Song details, or None when off."""
    if not enabled(folder) or not (Path(folder) / "notes.mid").exists():
        return None
    try:
        import mido
        m = mido.MidiFile(str(Path(folder) / "notes.mid"))
        tr = next((t for t in m.tracks if any(x.type == "track_name" and x.name == "PART VOCALS" for x in t)), None)
        if tr is None:
            return None
        words = [x.text for x in tr if x.type == "lyrics"]
        return {"notes": len(words), "pitched": sum(1 for w in words if not w.endswith("#"))}
    except Exception:
        return None
