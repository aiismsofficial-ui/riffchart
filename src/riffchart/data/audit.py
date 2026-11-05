"""Per-song audit: parse chart, decode audio, check onset/note alignment, collect stats.

Alignment: a smoothed impulse train of chart note onsets (chart time) is cross-correlated with the
audio's onset-strength envelope. lag = audio_time - chart_time at the correlation peak.
  lag_wide   best lag in +-wide_lag_s with no offset applied (used to verify the delay sign)
  lag        best lag within +-local_lag_s of the song.ini delay (YARG: audio = chart + delay)
  peak_z     correlation peak height in std units of the lag curve (low = chart doesn't fit audio)
  drift      spread of per-window lags (tempo-map drift / broken tempo maps)
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal

from ..audio.io import load_mix
from ..formats.loader import load_song
from ..formats.song import DIFFICULTIES, Song, popcount
from ..formats.textio import ini_float, song_delay_s
from .scan import SongFolder


def onset_envelope(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_mels=96, fmax=sr // 2, center=True)
    return env.astype(np.float32)


def note_train(times: np.ndarray, n_frames: int, fps: float, sigma: float) -> np.ndarray:
    x = np.zeros(n_frames, np.float32)
    idx = np.round(times * fps).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    np.add.at(x, idx, 1.0)
    return scipy.ndimage.gaussian_filter1d(x, sigma)


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def xcorr_lags(env: np.ndarray, train: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation c[k] = sum env[t+k] * train[t] for k in [-max_lag, max_lag]."""
    full = scipy.signal.correlate(_z(env), _z(train), mode="full", method="fft") / len(env)
    zero = len(train) - 1
    lags = np.arange(-max_lag, max_lag + 1)
    idx = np.clip(zero + lags, 0, len(full) - 1)
    return lags, full[idx]


def _peak(lags: np.ndarray, c: np.ndarray, lo: int, hi: int) -> tuple[float, float, float]:
    """(lag frames with parabolic refinement, peak value, peak z-score over the searched range)."""
    m = (lags >= lo) & (lags <= hi)
    lg, cc = lags[m], c[m]
    i = int(np.argmax(cc))
    off = 0.0
    if 0 < i < len(cc) - 1:
        a, b, d = cc[i - 1], cc[i], cc[i + 1]
        den = a - 2 * b + d
        off = 0.5 * (a - d) / den if den != 0 else 0.0
    z = (cc[i] - np.median(cc)) / (cc.std() + 1e-9)
    return float(lg[i] + off), float(cc[i]), float(z)


def hit_stats(note_s: np.ndarray, peak_s: np.ndarray, tol: float = 0.05) -> tuple[float, float, float]:
    """Fraction of notes with an onset peak within +-tol, and median / IQR of signed offsets (ms)."""
    if len(peak_s) == 0 or len(note_s) == 0:
        return 0.0, float("nan"), float("nan")
    i = np.clip(np.searchsorted(peak_s, note_s), 1, len(peak_s) - 1)
    left, right = peak_s[i - 1], peak_s[i]
    nearest = np.where(np.abs(note_s - left) < np.abs(right - note_s), left, right)
    d = nearest - note_s
    hit = np.abs(d) <= tol
    if not hit.any():
        return 0.0, float("nan"), float("nan")
    q1, q2, q3 = np.percentile(d[hit] * 1e3, [25, 50, 75])
    return float(hit.mean()), float(q2), float(q3 - q1)


def reference_track(song: Song):
    for d in DIFFICULTIES:
        tr = song.tracks.get(d)
        if tr is not None and len(tr) >= 20:
            return d, tr
    return None, None


def audit_song(sf: SongFolder, cfg: dict, out_dir: Path) -> dict:
    row: dict = {"song_id": sf.song_id, "path": str(sf.path), "chart_file": sf.chart.name,
                 "has_guitar_stem": sf.has_guitar_stem, "audio_files": [p.name for p in sf.mix_files],
                 "status": "ok", "error": ""}
    try:
        song, ini, _ = load_song(sf.path)
    except Exception as e:  # noqa: BLE001
        row.update(status="parse_error", error=repr(e)[:300])
        return row

    row.update(
        format=song.source_format, resolution=song.resolution,
        name=ini.get("name", song.metadata.get("name", sf.path.name)),
        artist=ini.get("artist", song.metadata.get("artist", "")),
        album=ini.get("album", ""), genre=ini.get("genre", song.metadata.get("genre", "")),
        year=ini.get("year", ""), charter=ini.get("charter", ini.get("frets", "")),
        ini_song_length_ms=ini_float(ini, "song_length"), diff_guitar=ini_float(ini, "diff_guitar"),
        delay_s=song_delay_s(ini), chart_offset_s=song.chart_offset_s,
        n_tempo=len(song.tempo.ticks), bpm_min=float(song.tempo.bpm.min()), bpm_max=float(song.tempo.bpm.max()),
        n_ts=len(song.tempo.ts_ticks), n_sections=len(song.sections), parse_warnings=len(song.warnings),
    )
    # tempo-weighted median BPM over the charted span
    last = max(song.last_tick(), 1)
    edges = np.append(song.tempo.ticks, max(last, song.tempo.ticks[-1] + 1))
    w = np.clip(np.diff(edges), 0, None)
    order = np.argsort(song.tempo.bpm)
    cw = np.cumsum(w[order])
    row["bpm_median"] = float(song.tempo.bpm[order][np.searchsorted(cw, cw[-1] / 2)]) if cw[-1] > 0 else float(song.tempo.bpm[0])

    for d in DIFFICULTIES:
        tr = song.tracks.get(d)
        n = len(tr) if tr is not None else 0
        row[f"n_{d}"] = n
        if n:
            on, sus = song.times(tr)
            span = max(on[-1] - on[0], 1e-3)
            row[f"nps_{d}"] = n / span
            row[f"chord_frac_{d}"] = float((popcount(tr.mask) > 1).mean())
            row[f"hopo_frac_{d}"] = float((tr.ntype == 1).mean())
            row[f"tap_frac_{d}"] = float((tr.ntype == 2).mean())
            row[f"open_frac_{d}"] = float(((tr.mask & 32) > 0).mean())
            row[f"sustain_frac_{d}"] = float((tr.sustain > 0).mean())
            row[f"n_sp_{d}"] = len(tr.star_power)
            row[f"first_note_s_{d}"] = float(on[0])
            row[f"last_note_s_{d}"] = float(on[-1] + sus[-1])

    ref_diff, ref = reference_track(song)
    row["ref_diff"] = ref_diff
    if ref is None:
        row.update(status="no_notes")
        return row

    a = cfg["audit"]
    sr, hop = a["sr"], a["hop"]
    fps = sr / hop
    try:
        y = load_mix(sf.mix_files, sr)
    except Exception as e:  # noqa: BLE001
        row.update(status="audio_error", error=repr(e)[:300])
        return row
    dur = len(y) / sr
    rms = float(np.sqrt(np.mean(y**2))) if len(y) else 0.0
    row.update(audio_s=dur, audio_rms=rms, audio_peak=float(np.abs(y).max()) if len(y) else 0.0)
    if dur < 10 or rms < 1e-4:
        row.update(status="audio_silent_or_short")
        return row

    # chroma fingerprint for dedup (1 frame/sec, 12 bins)
    hop_fp = sr // 4
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_fp, n_fft=4096)
    nsec = chroma.shape[1] // 4
    if nsec > 0:
        fp = chroma[:, : nsec * 4].reshape(12, nsec, 4).mean(-1).T.astype(np.float16)
        (out_dir / "chroma").mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "chroma" / f"{sf.song_id}.npy", fp)

    # guitar stem envelope when available (cleaner for guitar notes), else full mix
    env_src = y
    if sf.has_guitar_stem:
        try:
            env_src = load_mix([sf.audio["guitar"]] + ([sf.audio["rhythm"]] if "rhythm" in sf.audio else []), sr)
        except Exception:  # noqa: BLE001
            env_src = y
    env = onset_envelope(env_src, sr, hop)
    on, _ = song.times(ref)
    train = note_train(on, len(env), fps, a["note_sigma_frames"])

    delay_f = row["delay_s"] * fps
    wide = int(max(a["wide_lag_s"], abs(row["delay_s"]) + 1.5) * fps)
    lags, c = xcorr_lags(env, train, wide)
    lw, _, _ = _peak(lags, c, -wide, wide)
    loc = int(a["local_lag_s"] * fps)
    prior = a["expected_latency_ms"] / 1e3 * fps
    one_s = int(fps)

    def hyp(center: float) -> tuple[float, float, float]:
        """Lag near center + detector latency. Beat periodicity gives near-equal peaks one
        subdivision apart, so among local maxima within +-local we take the one closest to the
        prior unless another is clearly stronger (> 1/peak_tie of it). Also returns the ratio of
        the chosen peak to the best peak within +-1 s."""
        c0 = int(round(center + prior))
        m = (lags >= c0 - loc) & (lags <= c0 + loc)
        lg, cc = lags[m], c[m]
        pk, _ = scipy.signal.find_peaks(cc)
        if len(pk) == 0:
            pk = np.array([int(np.argmax(cc))])
        best = cc[pk].max()
        strong = pk[cc[pk] >= a["peak_tie"] * best] if best > 0 else pk[cc[pk] == best]
        i = int(strong[np.argmin(np.abs(lg[strong] - c0))])
        lp, cp, _ = _peak(lg, cc, int(lg[max(i - 1, 0)]), int(lg[min(i + 1, len(lg) - 1)]))
        _, c1, _ = _peak(lags, c, c0 - one_s, c0 + one_s)
        return lp, cp, cp / c1 if c1 > 0 else 0.0

    lp, cp, ratio = hyp(delay_f)
    ln, cn, _ = hyp(-delay_f)
    peaks, _ = scipy.signal.find_peaks(env, prominence=env.std() * 0.5)
    hit50, med_off, iqr_off = hit_stats(on + lp / fps, peaks / fps)
    hit50_neg, _, _ = hit_stats(on + ln / fps, peaks / fps)
    row.update(lag_wide_ms=lw / fps * 1e3, lag_ms=lp / fps * 1e3, peak_corr=cp, local_ratio=ratio,
               hit50=hit50, hit_med_off_ms=med_off, hit_iqr_ms=iqr_off,
               lag_neg_delay_ms=ln / fps * 1e3, corr_neg_delay=cn, hit50_neg_delay=hit50_neg,
               chart_end_past_audio_s=row.get(f"last_note_s_{ref_diff}", 0) + row["delay_s"] - dur)

    # drift: per-window lags searched only +-100 ms around the global lag (beat periodicity makes
    # wider windows ambiguous)
    win = int(a["window_s"] * fps)
    w_rad = int(0.1 * fps)
    wl = []
    shifted = on + lp / fps
    for s0 in range(0, len(env) - win // 2, win):
        sel = (shifted >= s0 / fps) & (shifted < (s0 + win) / fps)
        if sel.sum() < a["min_window_notes"]:
            continue
        seg_env = env[s0 : s0 + win]
        seg_tr = note_train(shifted[sel] - s0 / fps, len(seg_env), fps, a["note_sigma_frames"])
        l2, c2 = xcorr_lags(seg_env, seg_tr, w_rad)
        p, cpk, _ = _peak(l2, c2, -w_rad, w_rad)
        if cpk > 0.15:
            wl.append((lp + p) / fps * 1e3)
    row["n_windows"] = len(wl)
    if wl:
        wl = np.array(wl)
        row["drift_range_ms"] = float(np.percentile(wl, 90) - np.percentile(wl, 10)) if len(wl) >= 3 else float(np.ptp(wl))
        row["drift_std_ms"] = float(wl.std())
        row["window_lags_ms"] = [round(float(v), 1) for v in wl]
    return row


def audit_worker(args) -> dict:
    sf, cfg, out_dir = args
    rows_dir = out_dir / "rows"
    out = rows_dir / f"{sf.song_id}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    try:
        row = audit_song(sf, cfg, out_dir)
    except Exception as e:  # noqa: BLE001
        row = {"song_id": sf.song_id, "path": str(sf.path), "status": "crash",
               "error": "".join(traceback.format_exception_only(e))[:300]}
    rows_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, default=float), encoding="utf-8")
    return row
