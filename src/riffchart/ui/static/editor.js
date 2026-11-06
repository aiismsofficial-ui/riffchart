"use strict";
// riffchart editor, milestone 2: view + playback. Vertical highway, future at the top, strikeline near
// the bottom. Ticks are authoritative; seconds come from the chart's tempo map (same rules as Python).
const $ = (s) => document.querySelector(s);
const DIFFS = ["expert", "hard", "medium", "easy"];
const FRET_COLORS = ["#22c55e", "#ef4444", "#eab308", "#3b82f6", "#f97316"];
const OPEN_COLOR = "#a855f7", SP_COLOR = "#67e8f9";
const TYPE_NAMES = ["strum", "HOPO", "tap"];
const songId = new URLSearchParams(location.search).get("song");

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
let toastTimer;
function toast(msg, bad = false) {
  const t = $("#toast"); t.textContent = msg; t.classList.toggle("bad", bad); t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), bad ? 6000 : 3000);
}
function bsearch(arr, x) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; } return lo; }
function fmtTime(s) {
  const neg = s < 0; s = Math.abs(s);
  const m = Math.floor(s / 60), r = s - m * 60;
  return `${neg ? "-" : ""}${m}:${r.toFixed(3).padStart(6, "0")}`;
}
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ---------------------------------------------------------------- tempo map
class Tempo {
  constructor(res, tempo, ts) {
    this.res = res;
    this.ticks = tempo.map((x) => x[0]); this.bpm = tempo.map((x) => x[1]);
    if (!this.ticks.length || this.ticks[0] !== 0) { this.ticks.unshift(0); this.bpm.unshift(120); }
    this.spt = this.bpm.map((b) => 60 / (b * res));
    this.sec = [0];
    for (let i = 1; i < this.ticks.length; i++) this.sec.push(this.sec[i - 1] + (this.ticks[i] - this.ticks[i - 1]) * this.spt[i - 1]);
    this.ts = (ts && ts.length ? ts : [[0, 4, 4]]).slice();
    if (this.ts[0][0] !== 0) this.ts.unshift([0, 4, 4]);
  }
  seg(tick) { let i = bsearch(this.ticks, tick + 1e-9) - 1; return Math.max(0, Math.min(i, this.ticks.length - 1)); }
  tickToSec(t) { const i = this.seg(t); return this.sec[i] + (t - this.ticks[i]) * this.spt[i]; }
  secToTick(s) { let i = bsearch(this.sec, s + 1e-12) - 1; i = Math.max(0, Math.min(i, this.sec.length - 1)); return this.ticks[i] + (s - this.sec[i]) / this.spt[i]; }
  bpmAt(t) { return this.bpm[this.seg(t)]; }
  tsAt(t) { let k = 0; for (let i = 0; i < this.ts.length; i++) if (this.ts[i][0] <= t) k = i; return this.ts[k]; }
  // grid lines from tick a to b: {tick, kind: 2 bar / 1 beat / 0 sub, bar, beat}
  lines(a, b, division) {
    const out = [], sub = this.res * 4 / division;
    for (let k = 0; k < this.ts.length; k++) {
      const [start, num, den] = this.ts[k];
      const stop = k + 1 < this.ts.length ? this.ts[k + 1][0] : Infinity;
      if (stop <= a || start > b) continue;
      const beat = this.res * 4 / den, bar = beat * num;
      const barBase = this.barIndexAt(start);
      const n0 = Math.max(0, Math.floor((a - start) / sub)), from = start + n0 * sub;
      for (let t = from; t <= Math.min(b, stop - 1e-6); t += sub) {
        const rel = t - start, bi = Math.floor(rel / bar + 1e-9);
        const isBar = Math.abs(rel - bi * bar) < 1e-6, isBeat = Math.abs(rel / beat - Math.round(rel / beat)) < 1e-6;
        out.push({ tick: Math.round(t), kind: isBar ? 2 : isBeat ? 1 : 0, bar: barBase + bi + 1,
          beat: Math.floor((rel - bi * bar) / beat + 1e-9) + 1 });
      }
    }
    return out;
  }
  barIndexAt(tick) {      // completed bars before `tick` (tick at a TS change)
    let n = 0;
    for (let k = 0; k < this.ts.length; k++) {
      const [start, num, den] = this.ts[k];
      if (start >= tick) break;
      const stop = Math.min(k + 1 < this.ts.length ? this.ts[k + 1][0] : Infinity, tick);
      n += Math.ceil((stop - start) / (this.res * 4 / den * num) - 1e-9);
    }
    return n;
  }
  barBeat(tick) {
    const [start, num, den] = this.tsAt(tick);
    const beat = this.res * 4 / den, bar = beat * num, rel = tick - start;
    const bi = Math.floor(rel / bar + 1e-9);
    return { bar: this.barIndexAt(start) + bi + 1, beat: (rel - bi * bar) / beat + 1, num, den };
  }
}

// audio playback: audio.js (Player)

// ---------------------------------------------------------------- state
const S = {
  chart: null, tempo: null, player: null, peaks: [], diff: "expert", view: "single",
  pps: 260, division: 16, hover: null, lastT: 0, tracks: {}, lines: [], sections: [],
};
const hw = $("#hw"), ov = $("#ov"), ctx = hw.getContext("2d"), octx = ov.getContext("2d");
let W = 0, H = 0, OW = 0, OH = 0, dpr = 1, colors = {};

function resize() {
  dpr = window.devicePixelRatio || 1;
  const r = hw.getBoundingClientRect(), o = ov.getBoundingClientRect();
  W = r.width; H = r.height; OW = o.width; OH = o.height;
  hw.width = Math.round(W * dpr); hw.height = Math.round(H * dpr);
  ov.width = Math.round(OW * dpr); ov.height = Math.round(OH * dpr);
  colors = { bg: cssVar("--bg"), panel: cssVar("--panel"), panel2: cssVar("--panel2"), line: cssVar("--line"),
    text: cssVar("--text"), muted: cssVar("--muted"), accent: cssVar("--accent") };
  drawOverview();
}
window.addEventListener("resize", resize);

function prepTrack(t) {
  const n = t.notes.length, on = new Float64Array(n), off = new Float64Array(n), tick = new Float64Array(n);
  const off_s = S.chart.offset_s;
  t.notes.forEach((x, i) => { tick[i] = x[0]; on[i] = S.tempo.tickToSec(x[0]) + off_s; off[i] = S.tempo.tickToSec(x[0] + x[2]) + off_s; });
  const sp = t.star_power.map(([a, l]) => [a, a + l]);
  // which notes are in star power
  const inSp = new Uint8Array(n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    while (k < sp.length && sp[k][1] <= tick[i]) k++;
    inSp[i] = k < sp.length && sp[k][0] <= tick[i] && tick[i] < sp[k][1] ? 1 : 0;
  }
  return { ...t, on, off, tick, inSp,
    spSec: sp.map(([a, b]) => [S.tempo.tickToSec(a) + off_s, S.tempo.tickToSec(b) + off_s]),
    soloSec: t.solos.map(([a, b]) => [S.tempo.tickToSec(a) + off_s, S.tempo.tickToSec(b) + off_s]) };
}

// ---------------------------------------------------------------- layout
function layout() {
  const gutter = 64, wave = Math.min(140, Math.max(60, W * 0.12));
  const diffs = S.view === "all" ? DIFFS.filter((d) => S.tracks[d]) : [S.diff];
  // 150: labels on the right; a floating panel (problems, re-chart) takes more room when open
  const panel = [$("#issues"), $("#rechart")].some((el) => el && !el.hidden) ? Math.min(360, W * 0.3) : 0;
  const avail = W - gutter - wave - 24 - 150 - panel;
  const per = S.view === "all" ? Math.min(300, (avail - (diffs.length - 1) * 16) / diffs.length) : Math.min(420, avail);
  const hwW = Math.max(150, per);
  const total = diffs.length * hwW + (diffs.length - 1) * 16;
  let x = gutter + wave + 12 + Math.max(0, (avail - total) / 2);
  const hws = diffs.map((d) => { const h = { diff: d, x, w: hwW, lane: hwW / 5 }; x += hwW + 16; return h; });
  return { gutter, wave, waveX: gutter + 4, hws, strikeY: H * 0.82, rightX: x };
}
const yOf = (t, now, L) => L.strikeY - (t - now) * S.pps;
const tOf = (y, now, L) => now + (L.strikeY - y) / S.pps;

// ---------------------------------------------------------------- drawing
function draw() {
  const now = S.player ? S.player.time() : 0;
  if (S.player) scheduleClicks(now);
  if (S.player && typeof rechartFrame === "function") rechartFrame(now);
  S.lastT = now;
  updateClock(now);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, W, H);
  if (!S.chart) return requestAnimationFrame(draw);
  const L = layout();
  const t0 = tOf(H + 40, now, L), t1 = tOf(-40, now, L);
  const off = S.chart.offset_s;
  const k0 = S.tempo.secToTick(t0 - off), k1 = S.tempo.secToTick(t1 - off);
  const lines = S.tempo.lines(Math.max(0, k0), k1, S.division);

  drawWave(L, now, t0, t1);
  for (const h of L.hws) drawHighway(h, L, now, t0, t1, lines);
  // gutter: bar numbers and times
  ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
  for (const ln of lines) {
    if (ln.kind !== 2) continue;
    const y = yOf(S.tempo.tickToSec(ln.tick) + off, now, L);
    ctx.fillStyle = colors.text; ctx.fillText(String(ln.bar), L.gutter - 10, y - 6);
    ctx.fillStyle = colors.muted; ctx.fillText(fmtTime(S.tempo.tickToSec(ln.tick) + off).slice(0, -1), L.gutter - 10, y + 7);
  }
  // right labels: sections, tempo and time-signature changes
  ctx.textAlign = "left";
  const rx = L.rightX - 4;
  for (const [tick, name] of S.chart.sections) {
    const y = yOf(S.tempo.tickToSec(tick) + off, now, L);
    if (y < -20 || y > H + 20) continue;
    ctx.fillStyle = colors.accent; ctx.fillRect(rx, y - 0.5, 8, 1);
    ctx.font = "600 12px system-ui, sans-serif"; ctx.fillText(name, rx + 12, y);
  }
  ctx.font = "11px system-ui, sans-serif";
  S.chart.tempo.forEach(([tick, bpm], i) => {
    const y = yOf(S.tempo.tickToSec(tick) + off, now, L);
    if (y < -20 || y > H + 20) return;
    ctx.fillStyle = colors.muted; ctx.fillText(`♩ ${bpm.toFixed(bpm % 1 ? 1 : 0)}`, rx + 12, y + 13);
  });
  for (const [tick, num, den] of S.chart.ts) {
    const y = yOf(S.tempo.tickToSec(tick) + off, now, L);
    if (y < -20 || y > H + 20) continue;
    ctx.fillStyle = colors.text; ctx.fillText(`${num}/${den}`, rx + 12, y + 26);
  }
  // strikeline across everything
  ctx.strokeStyle = colors.accent; ctx.lineWidth = 2; ctx.globalAlpha = 0.9;
  ctx.beginPath(); ctx.moveTo(L.waveX, L.strikeY); ctx.lineTo(L.rightX - 12, L.strikeY); ctx.stroke(); ctx.globalAlpha = 1;
  drawOverviewCursor(now, L);
  requestAnimationFrame(draw);
}

function drawWave(L, now, t0, t1) {
  const x = L.waveX, w = L.wave - 8, cx = x + w / 2;
  ctx.fillStyle = colors.panel; ctx.fillRect(x, 0, w, H);
  const layers = [{ p: S.peaks.find((p) => p.name === "song"), c: colors.muted, a: 0.55 },
    { p: S.peaks.find((p) => p.name === "guitar"), c: colors.accent, a: 0.85 }];
  for (const { p, c, a } of layers) {
    if (!p || !p.bytes) continue;
    ctx.fillStyle = c; ctx.globalAlpha = a;
    const step = 2;
    for (let y = 0; y < H; y += step) {
      const ta = tOf(y + step, now, L), tb = tOf(y, now, L);
      let i0 = Math.floor(ta * p.per_s), i1 = Math.ceil(tb * p.per_s);
      if (i1 < 0 || i0 >= p.bytes.length) continue;
      i0 = Math.max(0, i0); i1 = Math.min(p.bytes.length - 1, Math.max(i1, i0));
      let m = 0; for (let i = i0; i <= i1; i++) if (p.bytes[i] > m) m = p.bytes[i];
      const hw2 = (m / 255) * (w / 2 - 2);
      ctx.fillRect(cx - hw2, y, hw2 * 2, step);
    }
  }
  ctx.globalAlpha = 1;
  const pv = S.chart.preview_s;
  if (pv != null) {                                        // song-list preview: a teal bracket on the left edge
    const ya = yOf(pv, now, L), yb = yOf(pv + PREVIEW_LEN, now, L);
    if (ya >= 0 && yb <= H) {
      ctx.fillStyle = PREVIEW_COLOR; ctx.globalAlpha = 0.9;
      ctx.fillRect(x, Math.max(0, yb), 3, Math.min(H, ya) - Math.max(0, yb));
      if (ya <= H) { ctx.fillRect(x, ya - 1, w, 2); ctx.font = "600 10px system-ui"; ctx.textAlign = "left"; ctx.fillText("PREVIEW", x + 6, ya - 5); }
      ctx.globalAlpha = 1;
    }
  }
}
const PREVIEW_LEN = 30, PREVIEW_COLOR = "#2dd4bf";

function drawHighway(h, L, now, t0, t1, lines) {
  const tr = S.tracks[h.diff], off = S.chart.offset_s;
  ctx.fillStyle = colors.panel; ctx.fillRect(h.x, 0, h.w, H);
  if (!tr) return;
  // solos
  for (const [a, b] of tr.soloSec) {
    if (b < t0 || a > t1) continue;
    const ya = yOf(a, now, L), yb = yOf(b, now, L);
    ctx.fillStyle = "rgba(59,130,246,0.10)"; ctx.fillRect(h.x, yb, h.w, ya - yb);
    ctx.fillStyle = "#3b82f6"; ctx.font = "600 11px system-ui"; ctx.textAlign = "left";
    ctx.fillText("SOLO", h.x + 4, Math.min(ya - 8, H - 8));
  }
  // star power phrases: bars on both edges
  for (const [a, b] of tr.spSec) {
    if (b < t0 || a > t1) continue;
    const ya = yOf(a, now, L), yb = yOf(b, now, L);
    ctx.fillStyle = SP_COLOR; ctx.globalAlpha = 0.7;
    ctx.fillRect(h.x - 4, yb, 3, ya - yb); ctx.fillRect(h.x + h.w + 1, yb, 3, ya - yb); ctx.globalAlpha = 1;
  }
  // grid
  for (const ln of lines) {
    const y = yOf(S.tempo.tickToSec(ln.tick) + off, now, L);
    ctx.fillStyle = ln.kind === 0 ? colors.muted : colors.text;
    ctx.globalAlpha = ln.kind === 2 ? 0.6 : ln.kind === 1 ? 0.28 : 0.1;
    ctx.fillRect(h.x, Math.round(y) - (ln.kind === 2 ? 1 : 0), h.w, ln.kind === 2 ? 2 : 1);
  }
  ctx.globalAlpha = 1;
  // lane dividers
  ctx.fillStyle = colors.line;
  for (let i = 1; i < 5; i++) ctx.fillRect(Math.round(h.x + i * h.lane), 0, 1, H);
  // fret targets at the strikeline
  const r = Math.min(h.lane * 0.36, 17);
  for (let f = 0; f < 5; f++) {
    const cx = h.x + (f + 0.5) * h.lane;
    ctx.strokeStyle = FRET_COLORS[f]; ctx.lineWidth = 2.5; ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.arc(cx, L.strikeY, r, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // notes: visible range by onset, plus sustains that started earlier
  const i0 = Math.max(0, bsearch(tr.on, t0 - 30)), i1 = bsearch(tr.on, t1 + 0.1);
  for (let i = i0; i < i1; i++) {                 // sustains first, gems on top
    if (tr.off[i] <= tr.on[i] + 1e-6 || tr.off[i] < t0) continue;
    const m = tr.notes[i][1], ya = yOf(tr.on[i], now, L), yb = yOf(tr.off[i], now, L);
    const sw = Math.max(4, r * 0.42);
    if (m & 32) {                              // open sustain: faint band with solid edges
      const c = tr.inSp[i] ? SP_COLOR : OPEN_COLOR;
      ctx.fillStyle = c; ctx.globalAlpha = 0.16; ctx.fillRect(h.x + 6, yb, h.w - 12, ya - yb);
      ctx.globalAlpha = 0.8; ctx.fillRect(h.x + 6, yb, 3, ya - yb); ctx.fillRect(h.x + h.w - 9, yb, 3, ya - yb);
      ctx.globalAlpha = 1; continue;
    }
    for (let f = 0; f < 5; f++) if (m & (1 << f)) {
      ctx.fillStyle = tr.inSp[i] ? SP_COLOR : FRET_COLORS[f]; ctx.globalAlpha = 0.85;
      ctx.fillRect(h.x + (f + 0.5) * h.lane - sw / 2, yb, sw, ya - yb);
    }
    ctx.globalAlpha = 1;
  }
  const hov = S.hover && S.hover.diff === h.diff ? S.hover.i : -1;
  for (let i = Math.max(0, bsearch(tr.on, t0 - 0.1)); i < i1; i++) {
    const y = yOf(tr.on[i], now, L), [, m, , nt] = tr.notes[i];
    const past = tr.on[i] < now - 0.001 && S.player.playing;
    ctx.globalAlpha = past ? 0.35 : 1;
    if (m & 32) drawOpen(h, y, r, nt, tr.inSp[i], i === hov);
    else for (let f = 0; f < 5; f++) if (m & (1 << f)) drawGem(h.x + (f + 0.5) * h.lane, y, r, f, nt, tr.inSp[i], i === hov);
  }
  ctx.globalAlpha = 1;
  drawEditOverlay(h, L, now, t0, t1);
  if (typeof drawRechartRange === "function") drawRechartRange(h, L, now);
  // label
  if (S.view === "all") {
    ctx.font = "600 11px system-ui"; ctx.textAlign = "center";
    const label = h.diff.toUpperCase(), tw = ctx.measureText(label).width + 14;
    ctx.fillStyle = colors.panel2; roundRect(h.x + h.w / 2 - tw / 2, 4, tw, 18, 9); ctx.fill();
    ctx.fillStyle = colors.text; ctx.fillText(label, h.x + h.w / 2, 13);
  }
}

function drawGem(cx, cy, r, f, nt, sp, hover) {
  const col = sp ? SP_COLOR : FRET_COLORS[f];
  const rr = nt === 1 ? r * 0.82 : r;
  ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2);
  if (nt === 2) {                               // tap: dark centre, coloured ring
    ctx.fillStyle = "#111"; ctx.fill(); ctx.lineWidth = rr * 0.42; ctx.strokeStyle = col;
    ctx.beginPath(); ctx.arc(cx, cy, rr * 0.78, 0, Math.PI * 2); ctx.stroke();
  } else {
    ctx.fillStyle = col; ctx.fill();
    ctx.lineWidth = nt === 1 ? 3 : 2; ctx.strokeStyle = nt === 1 ? "#ffffff" : "rgba(0,0,0,.55)"; ctx.stroke();
    if (nt === 0) { ctx.beginPath(); ctx.arc(cx, cy, rr * 0.42, 0, Math.PI * 2); ctx.fillStyle = "rgba(255,255,255,.85)"; ctx.fill(); }
  }
  if (hover) { ctx.lineWidth = 2; ctx.strokeStyle = colors.text; ctx.beginPath(); ctx.arc(cx, cy, rr + 4, 0, Math.PI * 2); ctx.stroke(); }
}
function drawOpen(h, y, r, nt, sp, hover) {
  const hh = Math.max(6, r * 0.55);
  ctx.fillStyle = sp ? SP_COLOR : OPEN_COLOR;
  roundRect(h.x + 4, y - hh / 2, h.w - 8, hh, hh / 2); ctx.fill();
  if (nt === 1) { ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke(); }
  if (hover) { ctx.lineWidth = 2; ctx.strokeStyle = colors.text; roundRect(h.x + 1, y - hh / 2 - 3, h.w - 2, hh + 6, hh); ctx.stroke(); }
}
function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h); }

// ---------------------------------------------------------------- overview (bottom = start, top = end)
let ovImage = null;
function drawOverview() {
  if (!S.chart || !OW) return;
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.fillStyle = colors.panel; octx.fillRect(0, 0, OW, OH);
  const dur = S.chart.audio_s, tr = S.tracks[S.view === "all" ? "expert" : S.diff] || S.tracks.expert;
  if (tr) {
    const bins = Math.max(1, Math.floor(OH / 2)), cnt = new Float32Array(bins);
    for (const t of tr.on) { const b = Math.floor((t / dur) * bins); if (b >= 0 && b < bins) cnt[b]++; }
    const mx = Math.max(1, ...cnt);
    octx.fillStyle = colors.accent;
    for (let b = 0; b < bins; b++) {
      const w = (cnt[b] / mx) * (OW - 12);
      octx.globalAlpha = 0.35 + 0.5 * (cnt[b] / mx);
      octx.fillRect(6, OH - (b + 1) * (OH / bins), w, OH / bins);
    }
    octx.globalAlpha = 1;
    octx.fillStyle = SP_COLOR;
    for (const [a, b] of tr.spSec) octx.fillRect(OW - 4, OH - (b / dur) * OH, 3, ((b - a) / dur) * OH);
  }
  octx.fillStyle = colors.text; octx.font = "600 9px system-ui"; octx.textAlign = "left";
  for (const [tick, name] of S.chart.sections) {
    const y = OH - ((S.tempo.tickToSec(tick) + S.chart.offset_s) / dur) * OH;
    octx.globalAlpha = 0.8; octx.fillRect(0, y, OW, 1); octx.globalAlpha = 1;
  }
  if (S.chart.preview_s != null) {
    const pa = OH - (S.chart.preview_s / dur) * OH, pb = OH - ((S.chart.preview_s + PREVIEW_LEN) / dur) * OH;
    octx.fillStyle = PREVIEW_COLOR; octx.globalAlpha = 0.9; octx.fillRect(0, pb, 3, pa - pb); octx.globalAlpha = 1;
  }
  if (typeof drawOverviewIssues === "function") drawOverviewIssues(octx, OW, OH, dur);
  if (typeof drawOverviewRechart === "function") drawOverviewRechart(octx, OW, OH, dur);
  ovImage = octx.getImageData(0, 0, ov.width, ov.height);
}
function drawOverviewCursor(now, L) {
  if (!ovImage) return;
  octx.putImageData(ovImage, 0, 0);
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const dur = S.chart.audio_s;
  const ta = tOf(H, now, L), tb = tOf(0, now, L);
  const ya = OH - (ta / dur) * OH, yb = OH - (tb / dur) * OH;
  octx.fillStyle = colors.text; octx.globalAlpha = 0.12; octx.fillRect(0, yb, OW, ya - yb);
  octx.globalAlpha = 1; octx.fillStyle = colors.accent; octx.fillRect(0, OH - (now / dur) * OH - 1, OW, 2);
}

// ---------------------------------------------------------------- clock, clicks
function updateClock(now) {
  $("#time").textContent = fmtTime(now);
  if (!S.tempo) return;
  const tick = S.tempo.secToTick(now - S.chart.offset_s);
  const bb = S.tempo.barBeat(Math.max(0, tick));
  const sec = S.chart.sections.filter(([t]) => t <= tick).pop();
  $("#pos").textContent = `Bar ${bb.bar} · beat ${Math.floor(bb.beat)} · ♩ ${S.tempo.bpmAt(tick).toFixed(1)} · ${bb.num}/${bb.den}` + (sec ? ` · ${sec[1]}` : "");
}
// note clap / metronome. At normal speed clicks are scheduled ~0.2 s ahead on the audio clock, so they
// land exactly on the music; when slowed (media playback) they fire as the playhead crosses.
let clickUpTo = null;
function scheduleClicks(now) {
  const P = S.player;
  const clap = $("#clap").checked, metro = $("#metro").checked;
  if (!P.playing || (!clap && !metro)) { clickUpTo = null; return; }
  const buffered = P.mode === "buffer";
  const head = buffered ? now + P.delay + 0.2 : now;                // audio position to schedule up to
  if (clickUpTo === null || head < clickUpTo - 0.05 || head - clickUpTo > 1.0) { clickUpTo = buffered ? now + P.delay : now; return; }
  const a = clickUpTo, b = head;
  if (b <= a) return;
  clickUpTo = b;
  const at = (t) => (buffered ? P.ctxTimeOf(t) : undefined);
  if (clap) {
    const tr = S.tracks[S.view === "all" ? "expert" : S.diff];
    if (tr) for (let i = bsearch(tr.on, a + 1e-9); i < tr.on.length && tr.on[i] <= b; i++) P.click(2200, 0.22, 0.025, at(tr.on[i]));
  }
  if (metro) {
    const off = S.chart.offset_s, ka = S.tempo.secToTick(a - off), kb = S.tempo.secToTick(b - off);
    for (const ln of S.tempo.lines(Math.max(0, ka), kb, 4)) {
      if (ln.kind === 0 || ln.tick <= ka) continue;
      P.click(ln.kind === 2 ? 1500 : 1000, 0.18, 0.04, at(S.tempo.tickToSec(ln.tick) + off));
    }
  }
}

// ---------------------------------------------------------------- input
function ensureAudioCtx() { if (S.player) S.player.resume(); }
function togglePlay() {
  ensureAudioCtx();
  if (!S.player.buffers) { toast("Audio is still loading…"); return; }
  if (S.player.playing) S.player.pause(); else S.player.play();
  $("#play").innerHTML = S.player.playing ? "&#10074;&#10074;" : "&#9654;";
}
function seekBy(dt) { S.player.seek(S.player.time() + dt); }
function gridStep(dir) {   // move to the next/previous grid line
  const off = S.chart.offset_s, tick = S.tempo.secToTick(S.player.time() - off), step = S.chart.resolution * 4 / S.division;
  const n = dir > 0 ? Math.floor(tick / step + 1e-6) + 1 : Math.ceil(tick / step - 1e-6) - 1;
  S.player.seek(S.tempo.tickToSec(Math.max(0, n * step)) + off);
}
function barStep(dir) {
  const off = S.chart.offset_s, tick = S.tempo.secToTick(S.player.time() - off);
  const bars = S.tempo.lines(Math.max(0, tick - S.chart.resolution * 64), tick + S.chart.resolution * 64, 4).filter((l) => l.kind === 2);
  const target = dir > 0 ? bars.find((l) => l.tick > tick + 1) : bars.filter((l) => l.tick < tick - 1).pop();
  if (target) S.player.seek(S.tempo.tickToSec(target.tick) + off);
}
function setZoom(pps) { S.pps = Math.max(80, Math.min(1600, pps)); $("#zoom").value = S.pps; savePrefs(); }
function setDiff(d) {
  if (d === "all") S.view = "all"; else { S.view = "single"; S.diff = d; }
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.diff === (S.view === "all" ? "all" : S.diff)));
  drawOverview(); savePrefs();
  if (typeof renderIssues === "function") renderIssues();
}
function setSpeed(r) { S.player.setRate(r); $("#speed").value = String(r); }

$("#play").addEventListener("click", togglePlay);
$("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.disabled) setDiff(b.dataset.diff); });
$("#speed").addEventListener("change", (e) => setSpeed(parseFloat(e.target.value)));
$("#grid").addEventListener("change", (e) => { S.division = parseInt(e.target.value, 10); savePrefs(); });
$("#zoom").addEventListener("input", (e) => setZoom(parseFloat(e.target.value)));
$("#vsong").addEventListener("input", applyVolumes);
$("#vguitar").addEventListener("input", applyVolumes);
$("#delay").addEventListener("change", (e) => {
  const ms = Math.max(0, Math.min(1000, parseFloat(e.target.value) || 0));
  e.target.value = ms; if (S.player) S.player.setDelay(ms); savePrefs();
});
["clap", "metro"].forEach((id) => $("#" + id).addEventListener("change", () => { ensureAudioCtx(); savePrefs(); }));
function applyVolumes() {
  if (!S.player) return;
  S.player.names.forEach((name, i) => S.player.setVolume(i, parseFloat(name.toLowerCase().startsWith("guitar") ? $("#vguitar").value : $("#vsong").value)));
  savePrefs();
}

// toolbar controls must not keep keyboard focus: Space would press them again instead of playing
document.addEventListener("pointerup", (e) => {
  const el = e.target.closest("button, a, input[type=checkbox], input[type=range]");
  if (el) setTimeout(() => el.blur(), 0);
});

hw.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) { setZoom(S.pps * Math.exp(-e.deltaY * 0.0015)); return; }
  const dy = e.deltaMode === 1 ? e.deltaY * 40 : e.deltaY;
  seekBy(-dy / S.pps);
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (e.target.closest("input[type=text], select") || !S.player || e.ctrlKey || e.metaKey) return;
  const k = e.key;
  if ((e.shiftKey || e.altKey) && k.startsWith("Arrow")) return;      // edit.js: move / sustain
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "ArrowUp") { e.preventDefault(); gridStep(1); }
  else if (k === "ArrowDown") { e.preventDefault(); gridStep(-1); }
  else if (k === "PageUp") { e.preventDefault(); barStep(1); }
  else if (k === "PageDown") { e.preventDefault(); barStep(-1); }
  else if (k === "Home") { e.preventDefault(); S.player.seek(0); }
  else if (k === "End") { e.preventDefault(); S.player.seek(S.chart.audio_s - 1); }
  else if (k === "+" || k === "=") setZoom(S.pps * 1.2);
  else if (k === "-" || k === "_") setZoom(S.pps / 1.2);
  else if (k === "[" || k === "]") {
    const opts = [...$("#speed").options].map((o) => parseFloat(o.value));
    const i = opts.indexOf(S.player.rate), j = Math.max(0, Math.min(opts.length - 1, (i < 0 ? opts.length - 1 : i) + (k === "]" ? 1 : -1)));
    setSpeed(opts[j]); toast(`Speed ${Math.round(opts[j] * 100)}%`);
  }
  else if (k === "v" || k === "V") setDiff(S.view === "all" ? S.diff : "all");
  else if ((k === "i" || k === "I") && !e.altKey && !e.shiftKey && S.chart) showDetails();
  else if ((k === "c" || k === "C") && !e.altKey) { $("#clap").checked = !$("#clap").checked; ensureAudioCtx(); savePrefs(); toast(`Note clap ${$("#clap").checked ? "on" : "off"}`); }
  else if ((k === "m" || k === "M") && !e.altKey) { $("#metro").checked = !$("#metro").checked; ensureAudioCtx(); savePrefs(); toast(`Metronome ${$("#metro").checked ? "on" : "off"}`); }
  else if (e.altKey && "1234".includes(k)) { e.preventDefault(); setDiff(DIFFS[+k - 1]); }
});

// overview: click / drag to jump
let ovDrag = false;
function ovSeek(e) { const r = ov.getBoundingClientRect(); const f = 1 - (e.clientY - r.top) / r.height; S.player.seek(Math.max(0, Math.min(1, f)) * S.chart.audio_s); }
ov.addEventListener("pointerdown", (e) => { ovDrag = true; ov.setPointerCapture(e.pointerId); ovSeek(e); });
ov.addEventListener("pointermove", (e) => { if (ovDrag) ovSeek(e); });
ov.addEventListener("pointerup", () => { ovDrag = false; });

// hover tooltip
hw.addEventListener("pointermove", (e) => {
  if (!S.chart) return;
  const r = hw.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const L = layout(), now = S.player.time();
  const h = L.hws.find((h) => x >= h.x && x < h.x + h.w);
  S.hover = null;
  const tip = $("#tip");
  if (h && S.tracks[h.diff]) {
    const tr = S.tracks[h.diff], t = tOf(y, now, L), tol = 14 / S.pps;
    const i = bsearch(tr.on, t - tol);
    let best = -1, bd = Infinity;
    for (let j = i; j < tr.on.length && tr.on[j] <= t + tol; j++) { const d = Math.abs(tr.on[j] - t); if (d < bd) { bd = d; best = j; } }
    if (best >= 0) {
      S.hover = { diff: h.diff, i: best };
      const [tick, m, sus, nt] = tr.notes[best];
      const frets = m & 32 ? "open" : ["G", "R", "Y", "B", "O"].filter((_, f) => m & (1 << f)).join("+");
      const bb = S.tempo.barBeat(tick);
      tip.innerHTML = `<b>${esc(frets)}</b> · ${TYPE_NAMES[nt]}${tr.inSp[best] ? " · star power" : ""}<br>` +
        `${fmtTime(tr.on[best])} · bar ${bb.bar} beat ${bb.beat.toFixed(2)}<br>` +
        `tick ${tick}${sus ? ` · sustain ${(tr.off[best] - tr.on[best]).toFixed(2)} s` : ""}`;
      tip.hidden = false;
      tip.style.left = Math.min(x + 16, W - 220) + "px"; tip.style.top = Math.max(4, y - 50) + "px";
      return;
    }
  }
  tip.hidden = true;
});
hw.addEventListener("pointerleave", () => { S.hover = null; $("#tip").hidden = true; });

// ---------------------------------------------------------------- prefs
const PREF_KEY = "riffchart.editor", OLD_PREF_KEY = "autochart.editor";   // old name: read once
function savePrefs() {
  try { localStorage.setItem(PREF_KEY, JSON.stringify({ pps: S.pps, division: S.division, diff: S.diff, view: S.view,
    vsong: $("#vsong").value, vguitar: $("#vguitar").value, delay: $("#delay").value, clap: $("#clap").checked, metro: $("#metro").checked })); } catch { /* ignore */ }
}
function loadPrefs() {
  let p; try { p = JSON.parse(localStorage.getItem(PREF_KEY) || localStorage.getItem(OLD_PREF_KEY) || "null"); } catch { p = null; }
  if (!p) return;
  S.pps = p.pps || S.pps; S.division = p.division || S.division; S.diff = p.diff || S.diff; S.view = p.view || S.view;
  $("#zoom").value = S.pps; $("#grid").value = String(S.division);
  if (p.vsong) $("#vsong").value = p.vsong; if (p.vguitar) $("#vguitar").value = p.vguitar;
  if (p.delay) $("#delay").value = p.delay;
  $("#clap").checked = !!p.clap; $("#metro").checked = !!p.metro;
}

// ---------------------------------------------------------------- load
async function load() {
  if (!songId) { $("#loading").textContent = "No song selected. Go back and pick one."; return; }
  loadPrefs();
  try {
    const base = `/api/songs/${encodeURIComponent(songId)}`;
    const [chart, peaks] = await Promise.all([fetch(base + "/chart").then((r) => { if (!r.ok) throw new Error(r.status === 404 ? "song not found" : r.statusText); return r.json(); }),
      fetch(base + "/peaks").then((r) => (r.ok ? r.json() : []))]);
    S.chart = chart;
    S.tempo = new Tempo(chart.resolution, chart.tempo, chart.ts);
    initEdit(chart);                                    // edit.js: editable copy + prepared tracks
    S.peaks = peaks.filter((p) => p.data).map((p) => ({ ...p, bytes: Uint8Array.from(atob(p.data), (c) => c.charCodeAt(0)) }));
    const stems = new URLSearchParams(location.search).get("noaudio") ? [] : chart.stems;   // noaudio=1: view only
    S.player = new Player(stems.map((s) => `${base}/file/${encodeURIComponent(s)}`), stems, chart.audio_s);
    S.player.onprogress = (d, n) => { $("#loading").textContent = `Loading audio… ${d}/${n}`; };
    $("#loading").textContent = "Loading audio…";
    try { await S.player.ready; } catch (e) { toast("Audio could not be loaded: " + e.message, true); S.player.buffers = []; }
    S.player.setDelay(parseFloat($("#delay").value) || 0);
    applyVolumes();
    document.title = `${chart.title} · RiffChart editor`;
    $("#title").textContent = chart.title; $("#artist").textContent = chart.artist;
    showPreview();
    if (!S.tracks[S.diff].notes.length) S.diff = DIFFS.find((d) => S.tracks[d].notes.length) || "expert";
    updateInfo();
    if (!chart.stems.length) toast("This song has no audio files; playback is silent.", true);
    setDiff(S.view === "all" ? "all" : S.diff);
    $("#loading").hidden = true;
    resize();
    // deep links: editor.html?song=..&t=<seconds>&view=all|expert|hard|medium|easy
    const q = new URLSearchParams(location.search);
    if (q.get("view")) setDiff(q.get("view"));
    const t0 = parseFloat(q.get("t"));
    S.player.seek(Number.isFinite(t0) ? t0 : Math.max(0, (S.tracks[S.diff]?.on[0] ?? 0) - 1.5));
  } catch (e) { $("#loading").textContent = "Could not load this song: " + e.message; }
}
resize();
requestAnimationFrame(draw);
load();

// song-list preview start (song.ini preview_start_time): set at the playhead, snapped to the nearest bar line
function showPreview() { $("#pvval").textContent = S.chart.preview_s == null ? "auto" : fmtTime(S.chart.preview_s).replace(/\.\d+$/, ""); }
async function setPreview() {
  if (!S.chart) return;
  const now = S.player.time(), tick = S.tempo.secToTick(now - S.chart.offset_s);
  const bars = S.tempo.lines(tick - S.chart.resolution * 16, tick + S.chart.resolution * 16, 4).filter((l) => l.kind === 2);
  let t = now;
  if (bars.length) {
    const near = bars.reduce((a, b) => (Math.abs(b.tick - tick) < Math.abs(a.tick - tick) ? b : a));
    const ts = S.tempo.tickToSec(near.tick) + S.chart.offset_s;
    if (Math.abs(ts - now) < 1.5) t = ts;
  }
  t = Math.max(0, Math.min(t, S.chart.audio_s - 1));
  try {
    const r = await fetch(`/api/songs/${encodeURIComponent(songId)}/preview`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ start_s: t }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    const was = S.chart.preview_s;
    S.chart.preview_s = j.preview_s; showPreview(); drawOverview();
    toast(`Song-list preview now starts at ${fmtTime(j.preview_s).replace(/\.\d+$/, "")}` +
      (was != null ? ` (was ${fmtTime(was).replace(/\.\d+$/, "")})` : "") + (j.export_stale ? ". Export again to update Clone Hero." : "."));
  } catch (e) { toast("Could not set the preview: " + e.message, true); }
}
$("#pvset").addEventListener("click", setPreview);
$("#pvhear").addEventListener("click", () => {
  if (!S.chart || S.chart.preview_s == null) return;
  S.player.seek(S.chart.preview_s);
  if (!S.player.playing) togglePlay();
});

// song details (details.js): title / artist / album / cover, saved to song.ini
function showDetails() {
  openDetails(songId, { onSaved: (info) => {
    S.chart.title = info.title; S.chart.artist = info.artist;
    $("#title").textContent = info.title; $("#artist").textContent = info.artist;
    document.title = `${info.title} · RiffChart editor`;
  } });
}
$("#details").addEventListener("click", showDetails);
