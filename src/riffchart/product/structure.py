"""Automatic section names (Intro, Verse 1, Chorus 1, Guitar Solo, ...) from the audio, bar by bar.

1. Bars come from the chart's tempo map. Each bar gets chroma (harmony) and MFCC (sound) means, its loudness,
   how much of it is sung (the timed lyrics when there are any, else vocals.ogg's level) and how loud the
   guitar track is.
2. Boundaries: novelty of the bar self-similarity matrix (checkerboard kernel) plus changes in singing and
   loudness, peaks at least MIN_BARS apart; sections longer than MAX_BARS are split at their strongest point.
3. Sections that sound alike (mean features, cosine) are grouped.
4. Names: before the first singing -> Intro, after the last -> Outro; sung groups: the one whose lyrics repeat
   most (or, without lyrics, that comes back most and is loudest) -> Chorus, a short sung group right before a
   chorus -> Pre-Chorus, a sung group heard once late in the song -> Bridge, the rest -> Verse; unsung sections
   in the middle -> Guitar Solo when the guitar is up front, else Interlude. Repeats are numbered.
Tuned on 40 and scored on 80 other human charts with stems (scripts/eval_sections.py): section starts within a
bar F1 0.63 (a start every 8 bars: 0.25), bars named the same kind of part 49% (all "verse": 25%).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

SR = 22050
HOP = 512
MIN_BARS = 4
MAX_BARS = 16
KERNEL = 6                 # bars each side of the checkerboard
PEAK_Z = 1.0              # novelty peak threshold (z-score)
GROUP_DIST = 0.4          # cosine distance to call two sections the same part
SUNG = 0.35                # fraction of a bar with singing to call it sung
SOLO_GTR_DB = 0.0          # guitar this much louder (vs the song's median) in an unsung part -> solo
VOCAL_REL_DB = -14.0       # vocal track's loud parts vs the mix's: below this the song counts as instrumental
MIN_SUNG_SHARE = 0.12      # ... and so does one sung in fewer than this share of its sections
INTRO_MAX_BARS = 16        # an intro starts within the first 16 bars (or 15% of the song)
MFCC_W = 2.0               # timbre vs harmony in the bar features
PHRASE = 0                 # >0: sections start on an n-bar grid (phase picked per song); 0 = any bar (scored best)


def bar_times(song, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(bar start ticks, their seconds in the audio) up to end_s."""
    tm, res = song.tempo, song.resolution
    ticks = []
    ts = list(zip(tm.ts_ticks.tolist(), tm.ts_num.tolist(), tm.ts_den.tolist()))
    end_tick = float(tm.sec_to_tick(np.array([max(0.0, end_s - song.chart_offset_s)]))[0])
    for k, (start, num, den) in enumerate(ts):
        stop = ts[k + 1][0] if k + 1 < len(ts) else end_tick
        bar = res * 4 / den * num
        t = float(start)
        while t < stop - 1e-6:
            ticks.append(int(round(t)))
            t += bar
    ticks = np.array(ticks, np.int64)
    secs = tm.tick_to_sec(ticks) + song.chart_offset_s
    return ticks, secs


def _frames_to_bars(x: np.ndarray, bars_s: np.ndarray, end_s: float) -> np.ndarray:
    """Mean of frame features (..., n) in each bar -> (nbars, ...)."""
    n = x.shape[-1]
    edges = np.append(bars_s, end_s)
    fi = np.clip((edges * SR / HOP).astype(int), 0, n)
    out = []
    for a, b in zip(fi[:-1], fi[1:]):
        b = max(b, a + 1)
        out.append(x[..., a:min(b, n)].mean(-1) if a < n else np.zeros(x.shape[:-1]))
    return np.array(out)


def _rms_db(y: np.ndarray) -> np.ndarray:
    import librosa
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    return 20 * np.log10(r + 1e-5)


def bar_features(mix: np.ndarray, sr: int, bars_s: np.ndarray, end_s: float, vocals: np.ndarray | None = None,
                 guitar: np.ndarray | None = None, lines: list | None = None) -> dict:
    import librosa
    import soxr

    def mono22(a):
        a = a.mean(1) if a.ndim == 2 else a
        return soxr.resample(a.astype(np.float32), sr, SR) if sr != SR else a.astype(np.float32)
    y = mono22(mix)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=14, hop_length=HOP)[1:]
    nb = len(bars_s)
    ch = _frames_to_bars(chroma, bars_s, end_s)
    ch /= np.linalg.norm(ch, axis=1, keepdims=True) + 1e-8
    mf = _frames_to_bars(mfcc, bars_s, end_s)
    mf = (mf - mf.mean(0)) / (mf.std(0) + 1e-6)
    loud = _frames_to_bars(_rms_db(y)[None], bars_s, end_s)[:, 0]
    sung = np.zeros(nb)
    edges = np.append(bars_s, end_s)
    has_vocals = bool(lines)
    if vocals is not None:                      # the vocal track says best where there's singing
        v = _rms_db(mono22(vocals))
        on = (v > np.percentile(v, 95) - 18).astype(float)
        sung = _frames_to_bars(on[None], bars_s, end_s)[:, 0]
        # separation always leaves something in the vocal track: an instrumental's is far below the mix
        has_vocals = has_vocals or float(np.percentile(v, 95) - np.percentile(_rms_db(y), 95)) > VOCAL_REL_DB
    elif lines:                                 # lyrics only: a line timed as a whole can stretch over a solo,
        for ln in lines:                        # so it counts for about as long as its words take to sing
            ws = [float(t) for t, _ in ln.words] or [float(ln.start)]
            a = min(ws)
            b = min(float(ln.end), a + 0.45 * max(len(ws), 1) + 0.6)
            for i in range(nb):
                ov = min(b, edges[i + 1]) - max(a, edges[i])
                if ov > 0:
                    sung[i] += ov / max(edges[i + 1] - edges[i], 1e-3)
        sung = np.clip(sung, 0, 1)
    gtr = None
    if guitar is not None:
        gtr = _frames_to_bars(_rms_db(mono22(guitar))[None], bars_s, end_s)[:, 0]
    return {"chroma": ch, "mfcc": mf, "loud": loud, "sung": sung, "guitar": gtr, "has_vocals": has_vocals}


def _feat(f: dict) -> np.ndarray:
    """Per-bar vector, centred on the song's average so that cosine compares what differs between parts."""
    c = f["chroma"] - f["chroma"].mean(0)
    c /= np.sqrt((c ** 2).sum(1).mean()) + 1e-8
    m = f["mfcc"] / np.sqrt(f["mfcc"].shape[1])
    x = np.concatenate([c, MFCC_W * m], 1)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def boundaries(f: dict) -> list[int]:
    """Section starts (bar indices, 0 first)."""
    x = _feat(f)
    n = len(x)
    if n < 2 * MIN_BARS:
        return [0]
    ssm = x @ x.T
    k = KERNEL
    g = np.outer(np.r_[-np.ones(k), np.ones(k)], np.r_[-np.ones(k), np.ones(k)])
    pad = np.pad(ssm, k, mode="edge")
    nov = np.array([(pad[i:i + 2 * k, i:i + 2 * k] * g).sum() for i in range(n)])
    nov = (nov - nov.mean()) / (nov.std() + 1e-8)
    sung = f["sung"] > SUNG
    vchange = np.r_[0, np.abs(np.diff(sung.astype(float)))]
    # a sung part starts where singing starts, give or take the bar before
    vchange = np.maximum(vchange, np.r_[vchange[1:], 0] * 0.5)
    lch = np.r_[0, np.abs(np.diff(f["loud"]))]
    lch = (lch - lch.mean()) / (lch.std() + 1e-8)
    score = nov + 1.2 * vchange + 0.3 * lch
    score[0] = -np.inf
    ok = np.ones(n, bool)
    if PHRASE:
        # the phrase grid's phase: where the song's changes line up best (singing starts weigh most)
        phase = max(range(PHRASE), key=lambda p: np.maximum(score[p::PHRASE][1 if p == 0 else 0:], 0).sum())
        ok = (np.arange(n) - phase) % PHRASE == 0
    cand = [i for i in range(1, n) if ok[i] and score[i] > PEAK_Z]
    picked = []
    for i in sorted(cand, key=lambda i: -score[i]):
        if all(abs(i - j) >= MIN_BARS for j in picked) and i >= MIN_BARS // 2 and n - i >= MIN_BARS // 2:
            picked.append(i)
    starts = sorted([0] + picked)
    out = []                                         # split sections that run on too long
    for a, b in zip(starts, starts[1:] + [n]):
        out.append(a)
        while b - a > MAX_BARS:
            inner = range(a + MIN_BARS, b - MIN_BARS + 1)
            if not len(inner):
                break
            best = max(inner, key=lambda i: score[i] + (2.0 if ok[i] else 0))
            out.append(best)
            a = best
    return sorted(set(out))


def _norm_line(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def name_sections(f: dict, starts: list[int], lines: list | None = None, bars_s: np.ndarray | None = None) -> list[str]:
    n = len(f["sung"])
    segs = list(zip(starts, starts[1:] + [n]))
    x = _feat(f)
    vec = np.array([x[a:b].mean(0) for a, b in segs])
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    sung = np.array([f["sung"][a:b].mean() for a, b in segs]) > SUNG
    loud = np.array([f["loud"][a:b].mean() for a, b in segs])
    # groups of alike sections (sung and unsung never share a group)
    group = [-1] * len(segs)
    reps: list[np.ndarray] = []
    rep_sung: list[bool] = []
    for i, v in enumerate(vec):
        best, bd = -1, GROUP_DIST
        for g, r in enumerate(reps):
            d = 1 - float(v @ r)
            if rep_sung[g] == sung[i] and d < bd:
                best, bd = g, d
        if best < 0:
            reps.append(v.copy()); rep_sung.append(bool(sung[i])); best = len(reps) - 1
        group[i] = best
    count = Counter(group)
    # lyric repetition per section: share of its lines whose words come back elsewhere in the song
    rep_score = np.zeros(len(segs))
    if lines and bars_s is not None:
        texts = Counter(_norm_line(" ".join(w for _, w in ln.words)) for ln in lines)
        edges = np.append(bars_s, np.inf)
        for i, (a, b) in enumerate(segs):
            inside = [ln for ln in lines if edges[a] <= (ln.start + ln.end) / 2 < edges[b]]
            if inside:
                rep_score[i] = np.mean([texts[_norm_line(" ".join(w for _, w in ln.words))] > 1 for ln in inside])
    has_v = f["has_vocals"] and sung.mean() >= MIN_SUNG_SHARE
    first_sung = int(np.argmax(sung)) if has_v else None
    last_sung = len(segs) - 1 - int(np.argmax(sung[::-1])) if has_v else None
    kinds = [""] * len(segs)
    if has_v:
        sung_groups = sorted({group[i] for i in range(len(segs)) if sung[i]})
        def gscore(g):
            idx = [i for i in range(len(segs)) if group[i] == g]
            return (rep_score[idx].mean() if lines else 0) * 2 + (count[g] > 1) * 0.6 + \
                (loud[idx].mean() - loud[sung].mean()) / 6.0
        chorus = max(sung_groups, key=gscore) if sung_groups else None
        if chorus is not None and count[chorus] < 2 and not (lines and rep_score[[i for i in range(len(segs)) if group[i] == chorus]].mean() > 0.5):
            chorus = None                            # a chorus comes back
        for i in range(len(segs)):
            if sung[i]:
                if group[i] == chorus:
                    kinds[i] = "chorus"
                elif count[group[i]] == 1 and i > len(segs) * 0.45 and chorus is not None and any(group[j] == chorus for j in range(i + 1, len(segs))):
                    kinds[i] = "bridge"                  # heard once, late, and a chorus still to come
                else:
                    kinds[i] = "verse"
            elif i < first_sung and segs[i][0] < max(INTRO_MAX_BARS, 0.15 * n):
                kinds[i] = "intro"
            elif i > last_sung:
                kinds[i] = "outro"
            else:
                kinds[i] = "instrumental"
        for i in range(len(segs) - 1):               # short verse-like part straight into a chorus
            a, b = segs[i]
            if kinds[i] == "verse" and kinds[i + 1] == "chorus" and b - a <= 8 and i > 0 and kinds[i - 1] == "verse":
                kinds[i] = "prechorus"
    else:                                            # no singing to go by: intro, parts by group, outro
        letters = {}
        for i in range(len(segs)):
            if len(segs) > 2 and i in (0, len(segs) - 1):
                kinds[i] = "intro" if i == 0 else "outro"
            else:
                kinds[i] = f"part{letters.setdefault(group[i], len(letters))}"
    gtr = f.get("guitar")
    if gtr is not None:
        med = float(np.median(gtr))
        for i, (a, b) in enumerate(segs):
            if kinds[i] == "instrumental" and gtr[a:b].mean() - med > SOLO_GTR_DB:
                kinds[i] = "solo"
    return _numbered(kinds)


NAMES = {"intro": "Intro", "verse": "Verse", "prechorus": "Pre-Chorus", "chorus": "Chorus", "bridge": "Bridge",
         "instrumental": "Interlude", "solo": "Guitar Solo", "outro": "Outro"}


def _numbered(kinds: list[str]) -> list[str]:
    total = Counter(kinds)
    seen: Counter = Counter()
    out = []
    for k in kinds:
        seen[k] += 1
        if k.startswith("part"):
            name = f"Part {chr(65 + int(k[4:]) % 26)}"
        else:
            name = NAMES[k]
        if total[k] > 1:
            name += f" {seen[k]}"
        out.append(name)
    return out


def auto_sections(song, mix: np.ndarray, sr: int, vocals: np.ndarray | None = None, guitar: np.ndarray | None = None,
                  lines: list | None = None) -> list[tuple[int, str]]:
    """[(tick, name)] for the chart, from the audio (mix, optional stems) and the timed lyrics."""
    end_s = len(mix) / sr
    ticks, secs = bar_times(song, end_s)
    keep = secs < end_s - 1.0
    ticks, secs = ticks[keep], secs[keep]
    if len(ticks) < 2 * MIN_BARS:
        return []
    f = bar_features(mix, sr, secs, end_s, vocals, guitar, lines)
    starts = boundaries(f)
    names = name_sections(f, starts, lines, secs)
    return [(int(ticks[s]), n) for s, n in zip(starts, names)]


# ---------------------------------------------------------------- scoring against human charts
def category(name: str) -> str:
    n = name.lower().replace("_", " ")
    for pat, k in (("pre ?-?chorus", "prechorus"), ("chorus", "chorus"), ("verse", "verse"), ("intro", "intro"),
                   ("outro|ending|end\\b|coda", "outro"), ("solo", "solo"), ("bridge", "bridge")):
        if re.search(pat, n):
            return k
    return "instrumental"
