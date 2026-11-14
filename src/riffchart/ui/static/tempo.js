"use strict";
// Tempo map editor (loaded after edit.js). Notes are stored in ticks (beats), so every change here moves notes
// with their beats: fixing the tempo is how a chart is lined up with the audio.
//  - Tempo map panel: every BPM and time-signature marker; type a new value, delete, add at the playhead.
//  - Drag a beat line on the waveform (while the panel is open): that beat moves to where you drop it. The tempo
//    between the previous marker and that beat is stretched to fit, and one is added at the beat so the next
//    marker stays where it was in the song (nothing after it moves).
//  - Tap tempo: tap the button (or B) along with the music; the average of the last taps can be used for the
//    marker at the playhead.
// Everything is one undo step per change (a whole drag is one step).
const TP = { open: false, drag: null, hover: null, taps: [] };
const MIN_BPM = 10, MAX_TBPM = 2000, GRAB_PX = 6;

const tempoList = () => S.chart.tempo.map((x) => x.slice()).sort((a, b) => a[0] - b[0]);
function fmtBar(tick) {
  const b = S.tempo.barBeat(tick);
  const beat = Math.round(b.beat * 100) / 100;
  return `bar ${b.bar}` + (Math.abs(beat - 1) > 1e-6 ? ` beat ${+beat.toFixed(2)}` : "");
}
function beatTick(tick) {           // the beat line nearest to `tick`
  const [start, , den] = S.tempo.tsAt(tick), beat = S.chart.resolution * 4 / den;
  return Math.max(0, Math.round(start + Math.round((tick - start) / beat) * beat));
}
function barTick(tick) {            // the bar line nearest to `tick`
  const [start, num, den] = S.tempo.tsAt(tick), bar = S.chart.resolution * 4 / den * num;
  return Math.max(0, Math.round(start + Math.round((tick - start) / bar) * bar));
}
function playTick() { return S.tempo.secToTick(Math.max(0, S.player.time() - off())); }

// ---------------------------------------------------------------- panel
function renderTempo() {
  const ol = $("#tlist");
  const rows = [];
  for (const [t, b] of tempoList()) rows.push({ t, kind: "bpm", v: b });
  for (const [t, n, d] of S.chart.ts) rows.push({ t, kind: "ts", v: `${n}/${d}` });
  rows.sort((a, b) => a.t - b.t || (a.kind === "ts" ? -1 : 1));
  ol.innerHTML = "";
  for (const r of rows) {
    const li = document.createElement("li");
    const time = fmtTime(S.tempo.tickToSec(r.t) + off());
    li.innerHTML = `<span class="ttl">${esc(fmtBar(r.t))}</span><span class="when">${time}</span>
      <label>${r.kind === "bpm" ? "BPM" : "Time sig."} <input type="text" inputmode="decimal" value="${r.kind === "bpm" ? +(+r.v).toFixed(3) : r.v}"></label>
      ${r.t === 0 ? "" : `<button class="ghost small icon" title="Delete this marker" aria-label="Delete">✕</button>`}`;
    const inp = li.querySelector("input");
    inp.addEventListener("change", () => (r.kind === "bpm" ? setBpm(r.t, inp.value) : setTs(r.t, inp.value)));
    inp.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") inp.blur(); });
    li.querySelector(".ttl").addEventListener("click", () => S.player.seek(S.tempo.tickToSec(r.t) + off()));
    const del = li.querySelector("button");
    if (del) del.addEventListener("click", () => removeMarker(r.t, r.kind));
    ol.appendChild(li);
  }
  showTaps();
}
function setBpm(tick, text) {
  const b = parseFloat(String(text).replace(",", "."));
  if (!(b >= MIN_BPM && b <= MAX_TBPM)) { toast(`BPM must be between ${MIN_BPM} and ${MAX_TBPM}`, true); return renderTempo(); }
  edit(`tempo ${+b.toFixed(3)} at ${fmtBar(tick)}`, () => {
    const tempo = tempoList().filter(([t]) => t !== tick);
    tempo.push([tick, Math.round(b * 1000) / 1000]);
    setTempo(tempo.sort((x, y) => x[0] - y[0]));
  });
  renderTempo();
}
function setTs(tick, text) {
  const m = String(text).trim().match(/^(\d{1,2})\s*\/\s*(\d{1,2})$/);
  const n = m ? +m[1] : 0, d = m ? +m[2] : 0;
  if (!(n >= 1 && n <= 32 && [1, 2, 4, 8, 16, 32].includes(d))) { toast("Time signature like 4/4, 3/4, 7/8 (the bottom number 1, 2, 4, 8, 16 or 32)", true); return renderTempo(); }
  edit(`time signature ${n}/${d} at ${fmtBar(tick)}`, () => {
    const ts = S.chart.ts.filter(([t]) => t !== tick).concat([[tick, n, d]]).sort((x, y) => x[0] - y[0]);
    setTiming({ ...timing(), ts });
  });
  renderTempo();
}
function removeMarker(tick, kind) {
  edit(kind === "bpm" ? `remove tempo at ${fmtBar(tick)}` : `remove time signature at ${fmtBar(tick)}`, () => {
    if (kind === "bpm") setTempo(tempoList().filter(([t]) => t !== tick));
    else setTiming({ ...timing(), ts: S.chart.ts.filter(([t]) => t !== tick) });
  });
  renderTempo();
}
function addBpmHere(bpm) {
  const t = beatTick(playTick());
  setBpm(t, bpm ?? S.tempo.bpmAt(t));
  toast(`Tempo marker at ${fmtBar(t)}`);
}
function addTsHere() {
  const t = barTick(playTick()), [, n, d] = S.tempo.tsAt(t);
  if (S.chart.ts.some(([x]) => x === t)) return toast("There's already a time signature on that bar: change it in the list");
  setTs(t, `${n}/${d}`);
  toast(`Time signature at ${fmtBar(t)}: type the new one in the list`);
}

// ---------------------------------------------------------------- tap tempo
function tap() {
  const now = performance.now() / 1000;
  if (TP.taps.length && now - TP.taps[TP.taps.length - 1] > 2) TP.taps = [];
  TP.taps.push(now);
  if (TP.taps.length > 16) TP.taps.shift();
  showTaps();
}
function tapBpm() {
  const n = TP.taps.length;
  if (n < 4) return null;
  return 60 * (n - 1) / (TP.taps[n - 1] - TP.taps[0]);
}
function showTaps() {
  const b = tapBpm(), use = $("#ttapuse");
  $("#ttapval").textContent = b ? `${b.toFixed(1)} BPM (${TP.taps.length} taps)` : TP.taps.length ? `keep tapping… (${TP.taps.length})` : "tap along with the beat";
  use.disabled = !b;
}

// ---------------------------------------------------------------- dragging a beat line on the waveform
function inWave(x, L) { return x >= L.waveX && x <= L.waveX + L.wave - 8; }
function beatNear(y, L) {           // beat line within GRAB_PX of y -> tick
  const t = beatTick(Math.max(0, tickAtY(y, L)));
  return Math.abs(yAtTick(t, L) - y) <= GRAB_PX && t > 0 ? t : null;
}
function startDrag(tick) {
  const tempo = tempoList();
  if (!tempo.length || tempo[0][0] !== 0) tempo.unshift([0, S.tempo.bpmAt(0)]);
  const prev = tempo.filter(([t]) => t < tick).pop();
  const next = tempo.find(([t]) => t > tick);
  TP.drag = { tick, base: snapshot(), tempo, a: prev[0], aSec: S.tempo.tickToSec(prev[0]),
    n: next ? next[0] : null, nSec: next ? S.tempo.tickToSec(next[0]) : null, bpmAfter: S.tempo.bpmAt(tick), moved: false };
}
function dragTo(sec) {              // put beat TP.drag.tick at chart time `sec`
  const d = TP.drag, res = S.chart.resolution;
  const lo = d.aSec + (d.tick - d.a) / res * 60 / MAX_TBPM, hiA = d.aSec + (d.tick - d.a) / res * 60 / MIN_BPM;
  let hi = hiA;
  if (d.n != null) hi = Math.min(hi, d.nSec - (d.n - d.tick) / res * 60 / MAX_TBPM);
  sec = Math.max(lo, Math.min(hi, sec));
  const bpmA = (d.tick - d.a) / res * 60 / (sec - d.aSec);
  const bpmB = d.n != null ? (d.n - d.tick) / res * 60 / (d.nSec - sec) : d.bpmAfter;
  const tempo = d.tempo.filter(([t]) => t !== d.a && t !== d.tick);
  tempo.push([d.a, Math.round(bpmA * 1000) / 1000], [d.tick, Math.round(Math.min(MAX_TBPM, Math.max(MIN_BPM, bpmB)) * 1000) / 1000]);
  setTempo(tempo.sort((x, y) => x[0] - y[0]));
  d.moved = true;
  rebuildAll();
}
window.addEventListener("pointerdown", (e) => {
  if (!TP.open || e.button !== 0 || e.target !== hw || !S.chart) return;
  const r = hw.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, L = layout();
  if (!inWave(x, L)) return;
  const t = beatNear(y, L);
  if (t == null) return;
  e.stopImmediatePropagation(); e.preventDefault();
  startDrag(t);
  hw.setPointerCapture(e.pointerId);
}, true);
window.addEventListener("pointermove", (e) => {
  if (!TP.open || !S.chart) return;
  const r = hw.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, L = layout();
  if (TP.drag) {
    e.stopImmediatePropagation();
    dragTo(tOf(y, S.player.time(), L) - off());
    return;
  }
  const over = e.target === hw && inWave(x, L) ? beatNear(y, L) : null;
  TP.hover = over;
  hw.style.cursor = over != null ? "ns-resize" : "";
}, true);
window.addEventListener("pointerup", (e) => {
  if (!TP.drag) return;
  e.stopImmediatePropagation();
  const d = TP.drag;
  TP.drag = null;
  if (d.moved) {
    E.undo.push({ snap: d.base, label: `move the beat at ${fmtBar(d.tick)}` }); E.redo = [];
    if (E.undo.length > 200) E.undo.shift();
    markDirty(); showSync(); renderTempo();
  }
}, true);

// drawn by editor.js's draw() each frame
function tempoOverlay(L, now, lines) {
  if (!TP.open) return;
  const x = L.waveX, w = L.wave - 8;
  for (const ln of lines) {
    if (ln.kind === 0) continue;
    const y = yAtTick(ln.tick, L);
    const hot = TP.drag ? ln.tick === TP.drag.tick : ln.tick === TP.hover;
    ctx.fillStyle = hot ? colors.accent : colors.text;
    ctx.globalAlpha = hot ? 1 : ln.kind === 2 ? 0.55 : 0.25;
    ctx.fillRect(x, y - (hot ? 1.5 : 0.5), w, hot ? 3 : 1);
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = colors.accent;
  for (const [t] of S.chart.tempo) {                 // tempo markers: a small triangle on the waveform's edge
    const y = yAtTick(t, L);
    if (y < -10 || y > H + 10) continue;
    ctx.beginPath(); ctx.moveTo(x + w, y); ctx.lineTo(x + w - 7, y - 5); ctx.lineTo(x + w - 7, y + 5); ctx.fill();
  }
  if (TP.drag) {
    ctx.font = "600 11px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    const y = yAtTick(TP.drag.tick, L);
    ctx.fillText(`${S.tempo.bpmAt(Math.max(0, TP.drag.tick - 1)).toFixed(2)} BPM`, x + 4, y - 10);
  }
}

// ---------------------------------------------------------------- open / close
function toggleTempo() {
  const p = $("#tempo");
  p.hidden = !p.hidden;
  TP.open = !p.hidden;
  if (TP.open) {
    $("#issues").hidden = true; $("#history").hidden = true;
    if (typeof closeRechart === "function" && !$("#rechart").hidden) closeRechart();
    renderTempo();
  } else { TP.hover = null; hw.style.cursor = ""; }
}
$("#tempobtn").addEventListener("click", toggleTempo);
$("#tclose").addEventListener("click", () => { if (TP.open) toggleTempo(); });
$("#taddbpm").addEventListener("click", () => addBpmHere());
$("#taddts").addEventListener("click", addTsHere);
$("#ttap").addEventListener("click", tap);
$("#ttapuse").addEventListener("click", () => { const b = tapBpm(); if (b) addBpmHere(Math.round(b * 100) / 100); });
window.addEventListener("keydown", (e) => {
  if (!TP.open || e.ctrlKey || e.metaKey || e.altKey || (e.target.matches && e.target.matches("input, select, textarea"))) return;
  if (e.key === "b" || e.key === "B") { e.preventDefault(); e.stopImmediatePropagation(); tap(); }
}, true);
// after Undo / Redo the list may be out of date
function tempoChanged() { if (TP.open) renderTempo(); }
