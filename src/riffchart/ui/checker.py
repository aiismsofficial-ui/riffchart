"""Playability issues for the editor: the evaluation checker (eval/playability.py, configs/eval.yaml
thresholds) with each violation located at a note and explained in words."""
from __future__ import annotations

import numpy as np
import yaml

from ..eval.metrics import Events
from ..eval.playability import find_issues
from ..formats.song import DIFFICULTIES, Song
from ..paths import project_path

# error: can't be played as written in Clone Hero; warning: playable but awkward or odd
SEVERITY = {"open_chord": "error", "stacked": "error", "chord_size": "error", "too_fast": "error",
            "sustain_overlap": "error", "bad_hopo": "error", "out_of_audio": "error",
            "chord_span": "warning", "fast_jump": "warning", "tiny_sustain": "warning", "ts_off_bar": "warning"}
TITLE = {"open_chord": "Open note in a chord", "stacked": "Stacked notes", "chord_size": "Chord too big",
         "chord_span": "Chord too wide", "too_fast": "Too fast", "fast_jump": "Fast jump",
         "sustain_overlap": "Sustain runs into the next note", "tiny_sustain": "Tiny sustain",
         "bad_hopo": "HOPO on the same frets", "out_of_audio": "Outside the audio",
         "ts_off_bar": "Time signature off the bar line"}
FRETS = "GRYBO"


def load_rules(path: str = "configs/eval.yaml") -> dict:
    return yaml.safe_load(open(project_path(path), encoding="utf-8"))["playability"]


def _frets(mask: int) -> str:
    return "open" if mask & 32 else "+".join(FRETS[f] for f in range(5) if mask >> f & 1)


def describe(rule: str, i: int, ev: Events, r: dict) -> str:
    ms = lambda x: f"{x * 1000:.0f} ms"                                     # noqa: E731
    if rule in ("stacked", "too_fast", "fast_jump") and i > 0:
        gap = ev.t[i] - ev.t[i - 1]
        if rule == "stacked":
            return f"{ms(gap)} after the previous note: they sound as one; merge them into a chord or delete one"
        if rule == "too_fast":
            return f"{ms(gap)} after the previous note (minimum {ms(r['min_ioi_s'])} on this difficulty)"
        return (f"{_frets(int(ev.mask[i - 1]))} to {_frets(int(ev.mask[i]))} in {ms(gap)} "
                f"(jumps of {r['jump_frets']}+ frets need {ms(r['jump_ioi_s'])})")
    if rule == "sustain_overlap" and i + 1 < len(ev):
        over = ev.t[i] + ev.sus[i] - ev.t[i + 1]
        return f"the tail ends {ms(over)} after the next note starts: shorten it (X clears, Shift+Down shortens)"
    if rule == "chord_size":
        return f"{_frets(int(ev.mask[i]))}: at most {r['max_chord']} frets on this difficulty"
    if rule == "chord_span":
        return f"{_frets(int(ev.mask[i]))} spans more than {r['max_span']} lanes"
    if rule == "tiny_sustain":
        return f"{ms(ev.sus[i])} tail (under {ms(r['min_sustain_s'])}): make it longer or remove it"
    if rule == "bad_hopo":
        return f"{_frets(int(ev.mask[i]))} again: nothing to hammer on or pull off; make it a strum (H)"
    if rule == "open_chord":
        return "open notes can't be combined with frets"
    if rule == "out_of_audio":
        return "this note is before the start or after the end of the audio"
    return ""


def timing_issues(song: Song, offset_s: float) -> list[dict]:
    """Song-wide timing problems (shown in every difficulty's list). A time signature has to start on a bar line
    of the one before it: Clone Hero and YARG draw the bars from there (Moonscraper's validator checks it too)."""
    tm, out = song.tempo, []
    ts = list(zip(tm.ts_ticks.tolist(), tm.ts_num.tolist(), tm.ts_den.tolist()))
    for (t0, n0, d0), (t1, n1, d1) in zip(ts, ts[1:]):
        bar = song.resolution * 4 / d0 * n0
        rel = (t1 - t0) / bar
        if abs(rel - round(rel)) > 1e-6:
            beats = (t1 - t0) / (song.resolution * 4 / d0) % n0
            out.append({"rule": "ts_off_bar", "tick": int(t1), "t": round(float(tm.tick_to_sec(t1)) + offset_s, 4),
                        "severity": SEVERITY["ts_off_bar"], "title": TITLE["ts_off_bar"],
                        "detail": f"{n1}/{d1} starts {beats:g} beats into a {n0}/{d0} bar; move it to a bar line "
                                  "(or change the one before it)"})
    return out


def song_issues(song: Song, offset_s: float, audio_s: float | None, rules: dict) -> dict:
    out = {}
    timing = timing_issues(song, offset_s)
    for d in DIFFICULTIES:
        tr = song.tracks.get(d)
        if tr is None or not len(tr):
            out[d] = {"issues": list(timing), "errors": 0, "warnings": len(timing)}
            continue
        ev = Events.from_track(song, tr, offset_s)
        r = rules[d]
        items = [dict(x) for x in timing]
        for rule, i in find_issues(ev, r, audio_s):
            items.append({"rule": rule, "tick": int(tr.tick[i]), "t": round(float(ev.t[i]), 4),
                          "severity": SEVERITY[rule], "title": TITLE[rule], "detail": describe(rule, i, ev, r)})
        items.sort(key=lambda x: (x["tick"], x["severity"] != "error"))
        out[d] = {"issues": items, "errors": sum(x["severity"] == "error" for x in items),
                  "warnings": sum(x["severity"] == "warning" for x in items)}
    return out
