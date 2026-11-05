"use strict";
// riffchart editor, milestone 3: editing. Loaded after editor.js and uses its globals (S, layout, ...).
//
// Editable notes are chords, one row per tick: [tick, mask, sustain, flag]
//   mask bit0..4 = G R Y B O, bit5 = open; flag 0 = natural, 1 = forced (HOPO<->strum flip), 2 = tap.
// The final type follows Clone Hero .chart rules from the neighbours (like Moonscraper), so moving a note
// changes HOPOs around it the same way the game will. Saving sends final types; the server's chart
// writer turns them back into N 5 / N 6 flags.

const FRET_KEYS = { "1": 0, "2": 1, "3": 2, "4": 3, "5": 4 };
const E = { undo: [], redo: [], dirty: false, sel: { diff: null, ticks: new Set() }, clip: null, drag: null,
  ghost: null, box: null, savedAt: null, saving: false };

// ---------------------------------------------------------------- note model
const popc = (m) => { let n = 0; for (; m; m &= m - 1) n++; return n; };
function naturalHopo(notes, thr) {
  const out = new Uint8Array(notes.length);
  for (let i = 1; i < notes.length; i++) {
    const [t, m] = notes[i], [pt, pm] = notes[i - 1];
    const differs = popc(pm) === 1 ? m !== pm : true;
    out[i] = popc(m) === 1 && differs && t - pt <= thr ? 1 : 0;
  }
  return out;
}
function finalTypes(notes) {
  const nat = naturalHopo(notes, S.chart.hopo_threshold);
  return notes.map((n, i) => (n[3] === 2 ? 2 : (nat[i] ^ (n[3] === 1)) ? 1 : 0));
}
function initEdit(chart) {
  S.edit = {};
  for (const d of DIFFS) {
    const t = chart.tracks[d] || { notes: [], star_power: [], solos: [] };
    const nat = naturalHopo(t.notes, chart.hopo_threshold);
    S.edit[d] = { notes: t.notes.map(([tick, m, sus, nt], i) => [tick, m, sus, nt === 2 ? 2 : (nt === 1) !== !!nat[i] ? 1 : 0]),
      star_power: t.star_power.map((x) => x.slice()), solos: t.solos.map((x) => x.slice()) };
    rebuild(d);
  }
  E.clean = contentSnap();
  updateButtons();
  if (typeof scheduleCheck === "function") scheduleCheck(0);
}
function normalize(tr) {
  const by = new Map();
  for (const n of tr.notes) {
    let [t, m, s, f] = n;
    t = Math.max(0, Math.round(t)); m &= 0x3f; s = Math.max(0, Math.round(s));
    if (!m) continue;
    const p = by.get(t);
    if (p) { p[1] |= m; p[2] = Math.max(p[2], s); } else by.set(t, [t, m, s, f]);
  }
  tr.notes = [...by.values()].sort((a, b) => a[0] - b[0]);
  for (const n of tr.notes) if ((n[1] & 32) && (n[1] & 31)) n[1] &= 31;       // open can't be in a chord
  tr.star_power = tr.star_power.filter(([, l]) => l > 0).sort((a, b) => a[0] - b[0]);
  tr.solos = tr.solos.filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0]);
}
function rebuild(d) {
  const tr = S.edit[d], types = finalTypes(tr.notes);
  S.tracks[d] = prepTrack({ notes: tr.notes.map((n, i) => [n[0], n[1], n[2], types[i]]), star_power: tr.star_power, solos: tr.solos });
}
function rebuildAll() {
  for (const d of DIFFS) rebuild(d);
  S.hover = null; $("#tip").hidden = true; drawOverview(); updateInfo();
  if (typeof scheduleCheck === "function") scheduleCheck();
}

// ---------------------------------------------------------------- undo / redo / dirty
const snapshot = () => JSON.stringify({ edit: S.edit, sections: S.chart.sections, sel: [E.sel.diff, [...E.sel.ticks]] });
function restore(snap) {
  const o = JSON.parse(snap);
  S.edit = o.edit; S.chart.sections = o.sections;
  E.sel = { diff: o.sel[0], ticks: new Set(o.sel[1]) };
  rebuildAll();
}
function edit(label, fn) {
  const before = snapshot();
  const changed = fn();
  if (changed === false) return false;
  E.undo.push({ snap: before, label }); if (E.undo.length > 200) E.undo.shift();
  E.redo = [];
  for (const d of DIFFS) normalize(S.edit[d]);
  rebuildAll(); markDirty(); return true;
}
function undo() {
  const u = E.undo.pop(); if (!u) return toast("Nothing to undo");
  E.redo.push({ snap: snapshot(), label: u.label }); restore(u.snap); markDirty(); toast("Undo: " + u.label);
}
function redo() {
  const r = E.redo.pop(); if (!r) return toast("Nothing to redo");
  E.undo.push({ snap: snapshot(), label: r.label }); restore(r.snap); markDirty(); toast("Redo: " + r.label);
}
const contentSnap = () => JSON.stringify({ edit: S.edit, sections: S.chart.sections });
function markDirty() { E.dirty = contentSnap() !== E.clean; updateButtons(); }   // undo back to saved = clean
function updateButtons() {
  $("#save").disabled = !E.dirty || E.saving; $("#undo").disabled = !E.undo.length; $("#redo").disabled = !E.redo.length;
  $("#savestate").textContent = E.saving ? "Saving…" : E.dirty ? "Unsaved changes" : E.savedAt ? `Saved ${E.savedAt}` : "";
  document.title = `${E.dirty ? "• " : ""}${S.chart ? S.chart.title : ""} · RiffChart editor`;
}
window.addEventListener("beforeunload", (e) => { if (E.dirty) { e.preventDefault(); e.returnValue = ""; } });

function updateInfo() {
  const n = DIFFS.map((d) => `${d[0].toUpperCase()}${S.edit ? S.edit[d].notes.length : 0}`).join(" ");
  const sel = E.sel.ticks.size ? ` · ${E.sel.ticks.size} selected` : "";
  $("#info").textContent = `${n} notes${sel}`;
}

// ---------------------------------------------------------------- save
function chartPayload() {      // what the server needs to write or check the chart
  const tracks = {};
  for (const d of DIFFS) {
    const tr = S.edit[d], types = finalTypes(tr.notes);
    tracks[d] = { notes: tr.notes.map((n, i) => [n[0], n[1], n[2], types[i]]), star_power: tr.star_power, solos: tr.solos };
  }
  return { resolution: S.chart.resolution, tempo: S.chart.tempo, ts: S.chart.ts, sections: S.chart.sections, tracks,
    offset_s: S.chart.offset_s, audio_s: S.chart.stems.length ? S.chart.audio_s : null };
}
async function save() {
  if (E.saving || !S.chart) return;
  E.saving = true; updateButtons();
  const sent = contentSnap();
  const body = chartPayload();
  try {
    const r = await fetch(`/api/songs/${encodeURIComponent(songId)}/chart`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    E.clean = sent; E.dirty = contentSnap() !== sent; E.savedAt = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    toast("Saved notes.chart (the previous version is kept in the song's .riffchart/history folder)");
  } catch (e) { toast("Save failed: " + e.message, true); }
  E.saving = false; updateButtons();
}

// ---------------------------------------------------------------- geometry helpers
const off = () => S.chart.offset_s;
const tickAtY = (y, L) => S.tempo.secToTick(tOf(y, S.player.time(), L) - off());
const yAtTick = (tick, L) => yOf(S.tempo.tickToSec(tick) + off(), S.player.time(), L);
function snapTick(tick) {
  const [start] = S.tempo.tsAt(tick), step = S.chart.resolution * 4 / S.division;
  return Math.max(0, Math.round(start + Math.round((tick - start) / step) * step));
}
const stepTicks = () => S.chart.resolution * 4 / S.division;
function hwAt(x, L) { return L.hws.find((h) => x >= h.x - 6 && x < h.x + h.w + 6); }
function laneAt(x, h) { return Math.max(0, Math.min(4, Math.floor((x - h.x) / h.lane))); }
function curDiff() { return S.view === "all" ? (E.sel.diff || S.diff) : S.diff; }
function playheadTick() { return snapTick(S.tempo.secToTick(S.player.time() - off())); }
function findChord(d, tick) { const ns = S.edit[d].notes; const i = bsearch(ns.map((n) => n[0]), tick); return ns[i] && ns[i][0] === tick ? ns[i] : null; }

function hitGem(h, x, y, L) {      // -> {tick, fret} of the gem under the pointer (fret 5 = open)
  const tr = S.edit[h.diff], r = Math.min(h.lane * 0.36, 17) + 4, lane = laneAt(x, h);
  let best = null, bd = Infinity;
  const t = tickAtY(y, L), span = Math.abs(tickAtY(y - r, L) - t) + 1;
  for (const n of tr.notes) {
    if (n[0] < t - span * 2) continue;
    if (n[0] > t + span * 2) break;
    const dy = Math.abs(yAtTick(n[0], L) - y);
    if (dy > r) continue;
    if (n[1] & 32) { if (dy < bd) { bd = dy; best = { tick: n[0], fret: 5 }; } continue; }
    if (n[1] & (1 << lane)) { const cx = h.x + (lane + 0.5) * h.lane; if (Math.abs(cx - x) <= r && dy < bd) { bd = dy; best = { tick: n[0], fret: lane }; } }
  }
  return best;
}
function hitTail(h, x, y, L) {     // end of a sustain tail -> tick of its chord
  const tr = S.edit[h.diff], lane = laneAt(x, h);
  for (const n of tr.notes) {
    if (!n[2]) continue;
    if (!((n[1] & 32) || (n[1] & (1 << lane)))) continue;
    if (Math.abs(yAtTick(n[0] + n[2], L) - y) <= 6) return n[0];
  }
  return null;
}

// ---------------------------------------------------------------- selection
function select(d, ticks, add = false) {
  if (!add || E.sel.diff !== d) E.sel = { diff: d, ticks: new Set() };
  for (const t of ticks) E.sel.ticks.add(t);
  updateInfo();
}
function toggleSel(d, tick) {
  if (E.sel.diff !== d) E.sel = { diff: d, ticks: new Set() };
  if (E.sel.ticks.has(tick)) E.sel.ticks.delete(tick); else E.sel.ticks.add(tick);
  updateInfo();
}
function clearSel() { E.sel = { diff: E.sel.diff, ticks: new Set() }; updateInfo(); }
const selNotes = () => (E.sel.diff ? S.edit[E.sel.diff].notes.filter((n) => E.sel.ticks.has(n[0])) : []);
function needSel() { if (!E.sel.ticks.size) { toast("Select some notes first (click, Shift+click or drag a box)"); return false; } return true; }

// ---------------------------------------------------------------- operations
function toggleFret(d, tick, fret) {        // fret 0..4, or 5 = open
  const tr = S.edit[d], bit = 1 << fret, ex = tr.notes.find((n) => n[0] === tick);
  if (!ex) tr.notes.push([tick, bit, 0, 0]);
  else if (fret === 5) ex[1] = ex[1] & 32 ? 0 : 32;
  else { ex[1] = (ex[1] & 32) ? bit : ex[1] ^ bit; }
}
function moveSelection(dt, dl, base) {
  const d = E.sel.diff, tr = S.edit[d];
  if (base) tr.notes = base.notes.map((n) => n.slice());
  const sel = tr.notes.filter((n) => E.sel.ticks.has(n[0]));
  if (!sel.length) return false;
  // keep frets on the highway: clamp the lane shift over the whole selection
  let lo = 4, hi = 0;
  for (const n of sel) if (!(n[1] & 32)) for (let f = 0; f < 5; f++) if (n[1] & (1 << f)) { lo = Math.min(lo, f); hi = Math.max(hi, f); }
  dl = Math.max(-lo, Math.min(4 - hi, dl));
  dt = Math.max(-Math.min(...sel.map((n) => n[0])), dt);
  const moved = sel.map(([t, m, s, f]) => [t + dt, m & 32 ? m : dl >= 0 ? (m << dl) & 31 : m >> -dl, s, f]);
  const newTicks = new Set(moved.map((n) => n[0]));
  // landing on an existing chord merges into it (normalize ORs the frets), nothing is lost
  tr.notes = tr.notes.filter((n) => !E.sel.ticks.has(n[0])).concat(moved);
  E.sel.ticks = newTicks;
  return dt !== 0 || dl !== 0;
}
function setSustain(ticks, fn) {
  const tr = S.edit[E.sel.diff || curDiff()];
  for (const n of tr.notes) if (ticks.has(n[0])) n[2] = Math.max(0, Math.round(fn(n)));
}
function toggleFlag(flag) {
  if (!needSel()) return;
  const sel = selNotes(), allOn = sel.every((n) => n[3] === flag);
  edit(flag === 2 ? "tap" : "HOPO/strum", () => { for (const n of sel) n[3] = allOn ? 0 : flag; });
}
function forceType() {   // H: flip the current final type of each selected chord (strum <-> HOPO)
  if (!needSel()) return;
  const d = E.sel.diff, tr = S.edit[d];
  edit("HOPO/strum", () => {
    const nat = naturalHopo(tr.notes, S.chart.hopo_threshold);
    tr.notes.forEach((n, i) => { if (E.sel.ticks.has(n[0])) n[3] = n[3] === 2 ? (nat[i] ? 0 : 1) : n[3] === 1 ? 0 : 1; });
  });
}
function toggleRange(kind) {          // star power phrase or solo over the selection
  if (!needSel()) return;
  const d = E.sel.diff, tr = S.edit[d], ticks = [...E.sel.ticks];
  const a = Math.min(...ticks), b = Math.max(...ticks);
  if (kind === "sp") {
    const hit = tr.star_power.filter(([s, l]) => s <= b && s + l > a);
    edit(hit.length ? "remove star power" : "add star power", () => {
      if (hit.length) tr.star_power = tr.star_power.filter((p) => !hit.includes(p));
      else tr.star_power.push([a, b - a + 1]);
    });
    toast(hit.length ? "Star power phrase removed" : "Star power phrase added");
  } else {
    const hit = tr.solos.filter(([s, e]) => s <= b && e >= a);
    edit(hit.length ? "remove solo" : "add solo", () => {
      if (hit.length) tr.solos = tr.solos.filter((p) => !hit.includes(p));
      else tr.solos.push([a, Math.max(b, a + 1)]);
    });
    toast(hit.length ? "Solo removed" : "Solo added");
  }
}
function deleteSel() {
  if (!needSel()) return;
  const d = E.sel.diff;
  edit("delete", () => { S.edit[d].notes = S.edit[d].notes.filter((n) => !E.sel.ticks.has(n[0])); E.sel.ticks = new Set(); });
}
function copySel(cut) {
  if (!needSel()) return;
  const sel = selNotes(), t0 = Math.min(...sel.map((n) => n[0])), tr = S.edit[E.sel.diff];
  const t1 = Math.max(...sel.map((n) => n[0]));
  E.clip = { notes: sel.map(([t, m, s, f]) => [t - t0, m, s, f]),
    sp: tr.star_power.filter(([s, l]) => s >= t0 && s <= t1).map(([s, l]) => [s - t0, l]) };
  if (cut) deleteSel();
  toast(`${cut ? "Cut" : "Copied"} ${sel.length} note${sel.length > 1 ? "s" : ""}`);
}
function paste() {
  if (!E.clip) return toast("Nothing copied yet");
  const d = curDiff(), at = playheadTick(), tr = S.edit[d];
  edit("paste", () => {
    const notes = E.clip.notes.map(([t, m, s, f]) => [t + at, m, s, f]), ticks = new Set(notes.map((n) => n[0]));
    tr.notes = tr.notes.filter((n) => !ticks.has(n[0])).concat(notes);
    for (const [s, l] of E.clip.sp) tr.star_power.push([s + at, l]);
    E.sel = { diff: d, ticks };
  });
  toast(`Pasted ${E.clip.notes.length} at the playhead`);
}
function sectionAtPlayhead() {
  const tick = S.tempo.secToTick(S.player.time() - off());
  const near = S.chart.sections.find(([t]) => Math.abs(t - tick) < S.chart.resolution / 2);
  const at = near ? near[0] : playheadTick();
  const name = prompt(near ? "Rename section (leave empty to delete it):" : "New section name at the playhead:", near ? near[1] : "");
  if (name === null) return;
  edit("section", () => {
    S.chart.sections = S.chart.sections.filter(([t]) => t !== at);
    if (name.trim()) S.chart.sections.push([at, name.trim()]);
    S.chart.sections.sort((a, b) => a[0] - b[0]);
  });
}

// ---------------------------------------------------------------- mouse
function pointerPos(e) { const r = hw.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

hw.addEventListener("contextmenu", (e) => e.preventDefault());
hw.addEventListener("pointerdown", (e) => {
  if (!S.chart || !S.edit) return;
  const { x, y } = pointerPos(e), L = layout(), h = hwAt(x, L);
  if (!h) return;
  if (S.view === "all" && E.sel.diff !== h.diff && !e.shiftKey) E.sel = { diff: h.diff, ticks: new Set() };
  const gem = hitGem(h, x, y, L);
  if (e.button === 2) {                         // right click: delete that fret / open note
    if (gem) edit("delete note", () => {
      const tr = S.edit[h.diff], n = tr.notes.find((n) => n[0] === gem.tick);
      n[1] = gem.fret === 5 ? 0 : n[1] & ~(1 << gem.fret);
      if (!n[1]) E.sel.ticks.delete(gem.tick);
    });
    return;
  }
  if (e.button !== 0) return;
  hw.setPointerCapture(e.pointerId);
  const tail = hitTail(h, x, y, L);
  if (tail !== null && !gem) {
    if (!E.sel.ticks.has(tail) || E.sel.diff !== h.diff) select(h.diff, [tail]);
    E.drag = { kind: "sustain", h, base: snapshot(), tick: tail, ticks: new Set(E.sel.ticks), changed: false,
      sus0: new Map(S.edit[h.diff].notes.filter((n) => E.sel.ticks.has(n[0])).map((n) => [n[0], n[2]])) };
    return;
  }
  if (gem) {
    if (e.shiftKey || e.ctrlKey) toggleSel(h.diff, gem.tick);
    else if (!(E.sel.diff === h.diff && E.sel.ticks.has(gem.tick))) select(h.diff, [gem.tick]);
    E.drag = { kind: "move", h, x0: x, y0: y, lane0: laneAt(x, h), tick0: gem.tick, base: snapshot(),
      notes0: JSON.parse(JSON.stringify(S.edit[h.diff])), sel0: new Set(E.sel.ticks), active: false, changed: false };
    return;
  }
  E.drag = { kind: "empty", h, x0: x, y0: y, add: e.shiftKey || e.ctrlKey, active: false };
});

hw.addEventListener("pointermove", (e) => {
  if (!S.chart || !S.edit) return;
  const { x, y } = pointerPos(e), L = layout(), dr = E.drag;
  if (!dr) {                                    // ghost note + cursor
    const h = hwAt(x, L);
    E.ghost = null;
    if (h && !hitGem(h, x, y, L)) {
      const tail = hitTail(h, x, y, L);
      hw.style.cursor = tail !== null ? "ns-resize" : "crosshair";
      if (tail === null) E.ghost = { diff: h.diff, tick: snapTick(tickAtY(y, L)), lane: laneAt(x, h) };
    } else hw.style.cursor = h ? "grab" : "default";
    return;
  }
  if (dr.kind === "sustain") {
    const end = snapTick(tickAtY(y, L)), d = dr.h.diff, grabbed = dr.sus0.get(dr.tick);
    const delta = (end - dr.tick) - grabbed;
    for (const n of S.edit[d].notes) if (dr.sus0.has(n[0])) n[2] = Math.max(0, dr.sus0.get(n[0]) + delta);
    dr.changed = true; rebuild(d);
  } else if (dr.kind === "move") {
    if (!dr.active && Math.hypot(x - dr.x0, y - dr.y0) < 4) return;
    dr.active = true; hw.style.cursor = "grabbing";
    const dt = snapTick(tickAtY(y, L)) - snapTick(tickAtY(dr.y0, L)) , dl = laneAt(x, dr.h) - dr.lane0;
    E.sel.ticks = new Set(dr.sel0);
    dr.changed = moveSelection(dt, dl, dr.notes0);
    normalize(S.edit[dr.h.diff]);
    rebuild(dr.h.diff);
  } else if (dr.kind === "empty") {
    if (!dr.active && Math.hypot(x - dr.x0, y - dr.y0) < 5) return;
    dr.active = true;
    E.box = { h: dr.h, x0: dr.x0, y0: dr.y0, x1: x, y1: y };
  }
});

hw.addEventListener("pointerup", (e) => {
  const dr = E.drag; E.drag = null;
  if (!dr || !S.edit) return;
  const { x, y } = pointerPos(e), L = layout();
  if (dr.kind === "sustain" || dr.kind === "move") {
    if (dr.changed) {
      E.undo.push({ snap: dr.base, label: dr.kind === "move" ? "move" : "sustain" }); E.redo = [];
      for (const d of DIFFS) normalize(S.edit[d]);
      rebuildAll(); markDirty();
    }
    hw.style.cursor = "grab";
  } else if (dr.kind === "empty") {
    if (dr.active) {                            // box select
      const b = E.box; E.box = null;
      const ta = tickAtY(Math.max(b.y0, b.y1), L), tb = tickAtY(Math.min(b.y0, b.y1), L);
      const la = laneAt(Math.min(b.x0, b.x1), dr.h), lb = laneAt(Math.max(b.x0, b.x1), dr.h);
      let lanes = 0; for (let f = la; f <= lb; f++) lanes |= 1 << f;
      const hits = S.edit[dr.h.diff].notes.filter((n) => n[0] >= ta && n[0] <= tb && ((n[1] & lanes) || (n[1] & 32))).map((n) => n[0]);
      select(dr.h.diff, hits, dr.add);
    } else if (dr.add) {
      clearSel();
    } else {                                    // click on empty highway: add a note
      const tick = snapTick(tickAtY(y, L)), lane = laneAt(x, dr.h);
      edit("add note", () => { toggleFret(dr.h.diff, tick, lane); E.sel = { diff: dr.h.diff, ticks: new Set([tick]) }; });
    }
  }
});
hw.addEventListener("pointerleave", () => { E.ghost = null; });

// ---------------------------------------------------------------- keyboard
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input[type=text], select") || !S.edit) return;
  const k = e.key, ctrl = e.ctrlKey || e.metaKey, step = stepTicks();
  if (ctrl) {
    const lk = k.toLowerCase();
    if (lk === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
    else if (lk === "y" || (lk === "z" && e.shiftKey)) { e.preventDefault(); redo(); }
    else if (lk === "s") { e.preventDefault(); save(); }
    else if (lk === "c") { e.preventDefault(); copySel(false); }
    else if (lk === "x") { e.preventDefault(); copySel(true); }
    else if (lk === "v") { e.preventDefault(); paste(); }
    else if (lk === "a") { e.preventDefault(); const d = curDiff(); select(d, S.edit[d].notes.map((n) => n[0])); }
    return;
  }
  if (k in FRET_KEYS || k === "0" || k === "6") {
    if (e.altKey) return;                        // Alt+1..4 switches difficulty (editor.js)
    const fret = k in FRET_KEYS ? FRET_KEYS[k] : 5;
    if (E.sel.ticks.size) {
      const sel = selNotes();
      edit("change frets", () => { for (const n of sel) toggleFret(E.sel.diff, n[0], fret); });
    } else {
      const d = curDiff(), tick = playheadTick();
      edit("add note", () => { toggleFret(d, tick, fret); });
    }
    e.preventDefault(); return;
  }
  if (k === "Delete" || k === "Backspace") { e.preventDefault(); deleteSel(); }
  else if (k === "Escape") { clearSel(); }
  else if (k === "h" || k === "H") forceType();
  else if (k === "t" || k === "T") toggleFlag(2);
  else if (k === "p" || k === "P") toggleRange("sp");
  else if (k === "l" || k === "L") toggleRange("solo");
  else if (k === "n" || k === "N") { e.preventDefault(); sectionAtPlayhead(); }
  else if (e.altKey && (k === "ArrowUp" || k === "ArrowDown")) {
    e.preventDefault(); if (!needSel()) return;
    edit("move", () => moveSelection(k === "ArrowUp" ? step : -step, 0));
  } else if (e.shiftKey && (k === "ArrowUp" || k === "ArrowDown")) {
    e.preventDefault(); if (!needSel()) return;
    edit("sustain", () => setSustain(E.sel.ticks, (n) => n[2] + (k === "ArrowUp" ? step : -step)));
  } else if (k === "ArrowLeft" || k === "ArrowRight") {
    e.preventDefault(); if (!needSel()) return;
    edit("shift frets", () => moveSelection(0, k === "ArrowRight" ? 1 : -1));
  } else if (k === "x" || k === "X") {
    if (!needSel()) return;
    edit("clear sustain", () => setSustain(E.sel.ticks, () => 0));
  }
});

$("#save").addEventListener("click", save);
$("#undo").addEventListener("click", undo);
$("#redo").addEventListener("click", redo);

// ---------------------------------------------------------------- overlay drawing (called by drawHighway)
function drawEditOverlay(h, L, now) {
  if (!S.edit) return;
  if (typeof drawIssueMarks === "function") drawIssueMarks(h, L);
  const r = Math.min(h.lane * 0.36, 17);
  if (E.sel.diff === h.diff && E.sel.ticks.size) {
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2.5;
    for (const n of S.edit[h.diff].notes) {
      if (!E.sel.ticks.has(n[0])) continue;
      const y = yAtTick(n[0], L);
      if (y < -30 || y > H + 30) continue;
      ctx.setLineDash([4, 3]);
      if (n[1] & 32) { roundRect(h.x + 1, y - r * 0.55, h.w - 2, r * 1.1, r * 0.5); ctx.stroke(); }
      else for (let f = 0; f < 5; f++) if (n[1] & (1 << f)) { ctx.beginPath(); ctx.arc(h.x + (f + 0.5) * h.lane, y, r + 5, 0, Math.PI * 2); ctx.stroke(); }
      ctx.setLineDash([]);
      if (n[2]) { ctx.fillStyle = "#fff"; const ye = yAtTick(n[0] + n[2], L); ctx.fillRect(h.x + 8, ye - 1, h.w - 16, 2); }
    }
  }
  if (E.ghost && E.ghost.diff === h.diff && !E.drag) {
    const y = yAtTick(E.ghost.tick, L), cx = h.x + (E.ghost.lane + 0.5) * h.lane;
    ctx.globalAlpha = 0.35; ctx.fillStyle = FRET_COLORS[E.ghost.lane];
    ctx.beginPath(); ctx.arc(cx, y, r, 0, Math.PI * 2); ctx.fill(); ctx.globalAlpha = 1;
  }
  if (E.box && E.box.h.diff === h.diff) {
    const b = E.box, x = Math.min(b.x0, b.x1), y = Math.min(b.y0, b.y1);
    ctx.fillStyle = "rgba(245,165,36,0.12)"; ctx.fillRect(x, y, Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
    ctx.strokeStyle = colors.accent; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
  }
}
