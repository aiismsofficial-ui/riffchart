"use strict";
// autochart editor, milestone 5: re-chart a section. Loaded after edit.js and uses its globals.
//
// Pick a range of bars (selection, the named section at the playhead, or nudge by bars), ask the server
// for a new take (the AI charts just those bars again, continuing from the notes before them), audition
// it with the section looping, flip between takes and the original, keep the one you like.
// Takes hold final note types per difficulty for the range; switching takes only replaces notes inside
// the range. The first take you apply adds one undo step ("re-chart section"), so Ctrl+Z goes back.

const R = { open: false, a: 0, b: 0, bars: [], takes: [], cur: 0, busy: false, undoPushed: false, n: 0 };
const RC_KEY = "autochart.rechart";

// ---------------------------------------------------------------- bars and range
function rcBars() {
  const end = Math.ceil(S.tempo.secToTick(S.chart.audio_s - off()));
  const ls = S.tempo.lines(0, end + S.chart.resolution * 16, 4).filter((l) => l.kind === 2);
  return ls.map((l) => ({ tick: Math.round(l.tick), bar: l.bar }));
}
function barAtOrBefore(tick) { let k = 0; for (let i = 0; i < R.bars.length && R.bars[i].tick <= tick; i++) k = i; return k; }
function barAfter(tick) { for (let i = 0; i < R.bars.length; i++) if (R.bars[i].tick > tick) return i; return R.bars.length - 1; }
function setRange(ia, ib) {     // bar indexes, ib exclusive
  ia = Math.max(0, Math.min(ia, R.bars.length - 2));
  ib = Math.max(ia + 1, Math.min(ib, R.bars.length - 1));
  const a = R.bars[ia].tick, b = R.bars[ib].tick;
  if (a === R.a && b === R.b) return;
  if (R.takes.length > 1 && R.cur !== 0) toast(`Kept take ${R.cur} in bars ${R.bars[barAtOrBefore(R.a)].bar}–${R.bars[barAtOrBefore(R.b) - 1].bar}`);
  R.a = a; R.b = b; R.takes = []; R.cur = 0; R.undoPushed = false;
  renderRechart();
}
function rangeFromSelection() {
  if (!E.sel.ticks.size) { toast("Select some notes first, or use the section at the playhead"); return false; }
  const ts = [...E.sel.ticks];
  setRange(barAtOrBefore(Math.min(...ts)), barAfter(Math.max(...ts)));
  return true;
}
function rangeAtPlayhead() {
  const tick = S.tempo.secToTick(S.player.time() - off());
  const secs = S.chart.sections;
  let i = -1;
  for (let k = 0; k < secs.length; k++) if (secs[k][0] <= tick + 1) i = k;
  if (i >= 0) {
    const ia = barAtOrBefore(secs[i][0]);
    const ib = i + 1 < secs.length ? barAtOrBefore(secs[i + 1][0] - 1) + 1 : R.bars.length - 1;
    if (ib - ia <= 32) return setRange(ia, ib);
  }
  const ia = barAtOrBefore(tick);
  setRange(ia, ia + 8);
}
const secOfTick = (t) => S.tempo.tickToSec(t) + off();

// ---------------------------------------------------------------- takes
function rangeNotes(d) {          // current notes of d inside the range, with final types
  const ns = S.edit[d].notes, types = finalTypes(ns);
  return ns.map((n, i) => [n[0], n[1], n[2], types[i]]).filter((n) => n[0] >= R.a && n[0] < R.b);
}
function originalTake() {
  const notes = {};
  for (const d of DIFFS) notes[d] = rangeNotes(d);
  return { label: "Original", notes, info: "the notes that were there" };
}
function rcBusyness(notes, orig, diffs) {   // how busy a take is next to the original, on its hardest difficulty
  const d = DIFFS.find((x) => diffs.includes(x)), n = notes[d].length, m = orig[d].length;
  if (!m) return n ? "new notes" : "";
  const r = n / m;
  const word = r >= 1.35 ? "much busier" : r >= 1.1 ? "busier" : r > 0.9 ? "similar amount of notes" : r > 0.7 ? "sparser" : "much sparser";
  return `${word} (${d[0].toUpperCase() + d.slice(1)} ${n} vs ${m})`;
}
function counts(notes) { return DIFFS.map((d) => `${d[0].toUpperCase()}${notes[d].length}`).join(" · "); }

function applyTake(i) {
  const t = R.takes[i];
  if (!t) return;
  if (!R.undoPushed) { E.undo.push({ snap: snapshot(), label: "re-chart section" }); E.redo = []; R.undoPushed = true; }
  for (const d of DIFFS) {
    const tr = S.edit[d];
    const outside = tr.notes.filter((n) => n[0] < R.a || n[0] >= R.b);
    const ins = t.notes[d].map((n) => n.slice());
    const before = outside.filter((n) => n[0] < R.a).pop();
    if (before && ins.length && before[2] > 0) before[2] = Math.min(before[2], ins[0][0] - before[0]);
    const isNew = new Set(ins.map((n) => n[0]));
    const all = outside.concat(ins).sort((x, y) => x[0] - y[0]);
    const nat = naturalHopo(all, S.chart.hopo_threshold);   // final type -> flag against the new neighbours
    tr.notes = all.map((n, k) => (isNew.has(n[0]) ? [n[0], n[1], n[2], n[3] === 2 ? 2 : (n[3] === 1) !== !!nat[k] ? 1 : 0] : n));
    normalize(tr);
  }
  E.sel = { diff: E.sel.diff, ticks: new Set() };
  R.cur = i;
  rebuildAll(); markDirty(); renderRechart();
}

function rcOptions() {
  return { difficulties: [...document.querySelectorAll("#rcdiffs input:checked")].map((x) => x.value),
    density: parseFloat($("#rcdensity").value), chords: $("#rcchords").value, follow: $("#rcfollow").value };
}
function rcStyleNote(o) {
  const bits = [];
  if (Math.abs(o.density) >= 0.05) bits.push(o.density < 0 ? "sparser" : "denser");
  if (o.chords !== "auto") bits.push({ low: "fewer chords", mid: "some chords", high: "more chords" }[o.chords]);
  if (o.follow !== "auto") bits.push(o.follow === "guitar" ? "guitar only" : "whole melody");
  if (o.difficulties.length < 4) bits.push(o.difficulties.map((d) => d[0].toUpperCase() + d.slice(1)).join("+") + " only");
  return bits.join(", ");
}

async function newTake() {
  if (R.busy || !R.open) return;
  const o = rcOptions();
  if (!o.difficulties.length) return toast("Tick at least one difficulty", true);
  if (!R.takes.length) R.takes = [originalTake()];
  R.busy = true; renderRechart();
  const status = (s) => { $("#rcstatus").textContent = s; };
  try {
    const st = await fetch(`/api/songs/${encodeURIComponent(songId)}/section`).then((r) => r.json()).catch(() => ({}));
    status(st.busy ? "Waiting for the song being charted in the library to finish…"
      : !st.analysed ? "Listening to the song first (first take only, about 20–60 s)…"
      : st.model !== "ready" ? "Loading the AI (first take only)…" : "Charting…");
    const t0 = performance.now();
    const seed = Math.floor(Math.random() * 1e9);
    const r = await fetch(`/api/songs/${encodeURIComponent(songId)}/section`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chart: chartPayload(), start: R.a, end: R.b, seed, options: o }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    if (j.start !== R.a || j.end !== R.b) throw new Error("the section changed while charting; try again");
    const notes = {};
    for (const d of DIFFS) notes[d] = j.notes[d] || R.takes[0].notes[d];
    R.n += 1;
    const busy = rcBusyness(notes, R.takes[0].notes, o.difficulties);
    R.takes.push({ label: `Take ${R.takes.length}`, notes, info: [busy, rcStyleNote(o)].filter(Boolean).join(" · "), seed });
    applyTake(R.takes.length - 1);
    status(`Took ${((performance.now() - t0) / 1000).toFixed(1)} s. Listen, then try another take or keep it.`);
    audition();
  } catch (e) { status(""); toast("Re-chart failed: " + e.message, true); }
  R.busy = false; renderRechart();
}

function audition() {             // play from one bar before the section
  const ia = barAtOrBefore(R.a);
  S.player.seek(secOfTick(R.bars[Math.max(0, ia - 1)].tick));
  if (!S.player.playing) togglePlay();
}

// ---------------------------------------------------------------- panel
function renderRechart() {
  if (!R.open) return;
  const ia = barAtOrBefore(R.a), ib = barAtOrBefore(R.b);
  $("#rcfrom").textContent = R.bars[ia].bar;
  $("#rcto").textContent = R.bars[Math.max(ia, ib - 1)].bar;
  const names = S.chart.sections.filter(([t], k, arr) => t < R.b && (k + 1 >= arr.length || arr[k + 1][0] > R.a)).map(([, n]) => n);
  const a = secOfTick(R.a), b = secOfTick(R.b);
  $("#rcwhat").textContent = `${names.length ? names.slice(0, 3).join(", ") + (names.length > 3 ? "…" : "") + " · " : ""}${ib - ia} bar${ib - ia > 1 ? "s" : ""} · ${fmtTime(a).slice(0, -4)}–${fmtTime(b).slice(0, -4)}`;
  $("#rctake").disabled = R.busy;
  $("#rctake").textContent = R.busy ? "Charting…" : R.takes.length > 1 ? "Another take" : "New take";
  const ol = $("#rctakes");
  ol.innerHTML = R.takes.map((t, i) => `<li data-i="${i}" class="${i === R.cur ? "cur" : ""}">
    <span class="dot"></span><span class="ttl">${esc(t.label)}</span><span class="when">${counts(t.notes)}</span>
    ${t.info ? `<span class="det">${esc(t.info)}</span>` : ""}</li>`).join("");
  $("#rcorig").disabled = !R.takes.length || R.cur === 0;
  $("#rckeep").disabled = R.takes.length < 2;
  document.querySelectorAll("#rechart .rcbound button, #rcsel, #rchere").forEach((x) => (x.disabled = R.busy));
  rcStyleSummary();
  drawOverview();
}
function openRechart() {
  if (!S.edit) return;
  R.bars = rcBars();
  if (R.bars.length < 2) return toast("This chart has no bars to re-chart", true);
  $("#issues").hidden = true;
  R.open = true; R.a = R.b = -1; R.takes = []; R.cur = 0; R.undoPushed = false;
  $("#rechart").hidden = false;
  if (!(E.sel.ticks.size && rangeFromSelection())) rangeAtPlayhead();
  renderRechart();
}
function closeRechart() {
  if (!R.open) return;
  R.open = false; $("#rechart").hidden = true; $("#rcstatus").textContent = ""; drawOverview();
  if (R.takes.length > 1 && R.cur !== 0) toast(`Kept take ${R.cur} (Ctrl+Z goes back to the original)`);
}
function rcStyleSummary() { $("#rcstylenote").textContent = `(${rcStyleNote({ ...rcOptions(), difficulties: DIFFS }) || "default"})`; }
function saveRcPrefs() {
  rcStyleSummary();
  try { localStorage.setItem(RC_KEY, JSON.stringify({ density: $("#rcdensity").value, chords: $("#rcchords").value, follow: $("#rcfollow").value, loop: $("#rcloop").checked })); } catch { /* private mode */ }
}
function loadRcPrefs() {
  let o; try { o = JSON.parse(localStorage.getItem(RC_KEY) || "null"); } catch { o = null; }
  if (!o) return;
  $("#rcdensity").value = o.density ?? 0; $("#rcchords").value = o.chords ?? "auto"; $("#rcfollow").value = o.follow ?? "auto";
  $("#rcloop").checked = o.loop ?? true;
  $("#rcdensout").textContent = rcDensLabel($("#rcdensity").value);
}
function rcDensLabel(v) {
  v = parseFloat(v);
  if (Math.abs(v) < 0.05) return "normal";
  return v < 0 ? `sparser (about ${Math.round((1 + 0.5 * v) * 100)}%)` : `denser (up to +${Math.round(60 * v)}% below Expert)`;
}

$("#rcbtn").addEventListener("click", () => (R.open ? closeRechart() : openRechart()));
$("#rcclose").addEventListener("click", closeRechart);
$("#rckeep").addEventListener("click", closeRechart);
$("#rcorig").addEventListener("click", () => { applyTake(0); audition(); });
$("#rctake").addEventListener("click", newTake);
$("#rcsel").addEventListener("click", rangeFromSelection);
$("#rchere").addEventListener("click", rangeAtPlayhead);
$("#rctakes").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-i]");
  if (li && !R.busy) { applyTake(+li.dataset.i); audition(); }
});
document.querySelectorAll("#rechart .rcbound button").forEach((x) => x.addEventListener("click", () => {
  const [side, d] = [x.dataset.adj[0], parseInt(x.dataset.adj.slice(1), 10)];
  const ia = barAtOrBefore(R.a), ib = barAtOrBefore(R.b);
  if (side === "a") setRange(ia + d, ib); else setRange(ia, ib + d);
}));
$("#rcdensity").addEventListener("input", (e) => { $("#rcdensout").textContent = rcDensLabel(e.target.value); saveRcPrefs(); });
["rcchords", "rcfollow", "rcloop"].forEach((id) => $("#" + id).addEventListener("change", saveRcPrefs));
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input[type=text], select") || e.ctrlKey || e.metaKey || e.altKey || !S.edit) return;
  if (e.key === "r" || e.key === "R") {
    e.preventDefault();
    if (e.shiftKey) { if (!R.open) openRechart(); newTake(); } else if (R.open) closeRechart(); else openRechart();
  }
});
loadRcPrefs();
rcStyleSummary();

// ---------------------------------------------------------------- drawing and looping (called by editor.js)
function drawRechartRange(h, L, now) {
  if (!R.open || R.a < 0) return;
  const on = rcOptions().difficulties.includes(h.diff);
  const ya = yOf(secOfTick(R.a), now, L), yb = yOf(secOfTick(R.b), now, L);
  if (ya < -10 && yb < -10) return;
  if (yb > H + 10) return;
  ctx.fillStyle = colors.accent;
  ctx.globalAlpha = on ? 0.07 : 0.025; ctx.fillRect(h.x, yb, h.w, ya - yb);
  ctx.globalAlpha = on ? 0.9 : 0.35;
  ctx.fillRect(h.x - 2, ya - 1, h.w + 4, 2); ctx.fillRect(h.x - 2, yb - 1, h.w + 4, 2);
  ctx.fillRect(h.x - 8, yb, 3, ya - yb);
  ctx.globalAlpha = 1;
}
function drawOverviewRechart(c, w, h, dur) {
  if (!R.open || R.a < 0) return;
  const ya = h - (secOfTick(R.a) / dur) * h, yb = h - (secOfTick(R.b) / dur) * h;
  c.fillStyle = colors.accent; c.globalAlpha = 0.25; c.fillRect(0, yb, w, ya - yb); c.globalAlpha = 1;
}
function rechartFrame(now) {      // loop the section, unless you scrolled or jumped away from it
  const last = R.lastNow ?? now; R.lastNow = now;
  if (!R.open || R.a < 0 || !S.player.playing || !$("#rcloop").checked) return;
  const end = secOfTick(R.b) + 0.25;
  if (now > end && last <= end && now - last < 0.5) {
    const ia = barAtOrBefore(R.a);
    S.player.seek(secOfTick(R.bars[Math.max(0, ia - 1)].tick));
    R.lastNow = null;
  }
}
