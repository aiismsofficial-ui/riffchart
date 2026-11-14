"use strict";
// Lyrics (loaded after edit.js). The song's lyrics are lines of words timed in audio seconds (S.lyrics, from
// .riffchart/lyrics.json or the chart's own lyric events), so tempo edits never move them. Save writes them
// into notes.chart (and notes.mid when the song has pitched vocals).
//  - Lyrics panel: every line; change its words, delete it, add one at the playhead, shift them all.
//  - While the panel is open the words are drawn on the waveform: drag one to where it's sung.
//  - Tap timing: play the song and press W (or the button) as each word starts; the highlighted word takes
//    that time and the next one is up. Words keep their order (a word can't pass its neighbours).
// Each change is one undo step (a whole drag is one step). "Hel- lo" keeps the "-": Clone Hero joins them.
const LY = { open: false, drag: null, hover: null, tap: null };
const LY_GAP = 0.02, LY_GRAB = 8;

const lyrLines = () => S.lyrics || [];
function allWords() {                       // [{li, wi, t}] in time order
  const out = [];
  lyrLines().forEach((ln, li) => ln.words.forEach((w, wi) => out.push({ li, wi, t: w[0] })));
  return out;
}
function fixLine(ln) {                      // line bounds follow its words
  ln.words.sort((a, b) => a[0] - b[0]);
  if (!ln.words.length) return;
  ln.start = Math.min(ln.start, ln.words[0][0]);
  if (ln.start < ln.words[0][0] - 2) ln.start = ln.words[0][0];
  ln.end = Math.max(ln.end, ln.words[ln.words.length - 1][0] + 0.2);
}
function tidyLyrics() {
  S.lyrics = lyrLines().filter((ln) => ln.words.length);
  S.lyrics.forEach(fixLine);
  S.lyrics.sort((a, b) => a.start - b.start);
}
function spread(start, end, text) {         // words timed only by the line: spaced by their length
  const words = text.split(/\s+/).filter(Boolean);
  const wt = words.map((w) => w.length + 2), sum = wt.reduce((a, b) => a + b, 0);
  let acc = 0;
  return words.map((w, i) => { const t = start + (end - start) * acc / sum; acc += wt[i]; return [Math.round(t * 1000) / 1000, w]; });
}
function wordBounds(li, wi) {               // a word stays between its neighbours (across lines too)
  const ws = allWords(), i = ws.findIndex((w) => w.li === li && w.wi === wi);
  return [i > 0 ? ws[i - 1].t + LY_GAP : 0, i < ws.length - 1 ? ws[i + 1].t - LY_GAP : (S.chart.audio_s || 1e9)];
}

// ---------------------------------------------------------------- panel
function renderLyrics() {
  const ol = $("#llist");
  if (!ol) return;
  const src = { chart: "the chart", lrclib: "lrclib.net", whisper: "written down from the singing" }[S.lyricsSource] || "";
  $("#lsrc").textContent = lyrLines().length ? `${allWords().length} words${src ? " · from " + src : ""}` : "";
  if (!lyrLines().length) {
    ol.innerHTML = `<li class="none">No lyrics yet. Add a line at the playhead, or use Find lyrics in the song's details.</li>`;
    return;
  }
  ol.innerHTML = "";
  lyrLines().forEach((ln, li) => {
    const el = document.createElement("li");
    const tap = LY.tap && LY.tap.li === li;
    el.className = tap ? "cur" : "";
    el.innerHTML = `<span class="ttl">${fmtTime(ln.start)}</span>
      <button class="ghost small icon" title="Delete this line" aria-label="Delete">✕</button>
      <input type="text" spellcheck="true" aria-label="Words of the line at ${fmtTime(ln.start)}">`;
    const inp = el.querySelector("input");
    inp.value = ln.words.map((w) => w[1]).join(" ");
    inp.addEventListener("change", () => setLineText(li, inp.value));
    inp.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") inp.blur(); });
    el.querySelector(".ttl").addEventListener("click", () => S.player.seek(Math.max(0, ln.start - 0.5)));
    el.querySelector("button").addEventListener("click", () => removeLine(li));
    ol.appendChild(el);
  });
}
function setLineText(li, text) {
  const ln = lyrLines()[li], words = String(text).split(/\s+/).filter(Boolean);
  if (!ln) return;
  if (words.join(" ") === ln.words.map((w) => w[1]).join(" ")) return;
  edit(words.length ? "change lyrics" : "delete a lyric line", () => {
    if (!words.length) S.lyrics.splice(li, 1);
    else if (words.length === ln.words.length) words.forEach((w, i) => { ln.words[i][1] = w; });   // same count: keep the times
    else retext(ln, words);
    tidyLyrics();
  });
  renderLyrics();
}
// Words kept at the start and end of the line keep their times; only the changed part in between is spaced out
// again, over the time the words it replaces took (or the gap they go in).
function retext(ln, words) {
  const old = ln.words, n = Math.min(old.length, words.length);
  let p = 0, s = 0;
  while (p < n && old[p][1] === words[p]) p++;
  while (s < n - p && old[old.length - 1 - s][1] === words[words.length - 1 - s]) s++;
  const midOld = old.slice(p, old.length - s), midNew = words.slice(p, words.length - s);
  const next = s ? old[old.length - s][0] : null;
  let a, b;
  if (midOld.length) { a = midOld[0][0]; b = next != null ? next : Math.max(midOld[midOld.length - 1][0] + 0.3, a + 0.3 * midNew.length); }
  else { a = p ? old[p - 1][0] + 0.15 : ln.start; b = next != null ? next : a + 0.3 * midNew.length; }
  if (next != null && b - a < 0.05 * midNew.length) a = Math.max(p ? old[p - 1][0] + LY_GAP : 0, b - 0.05 * midNew.length);
  ln.words = [...old.slice(0, p), ...(midNew.length ? spread(a, b, midNew.join(" ")) : []), ...old.slice(old.length - s)];
  if (!s && !midNew.length && ln.words.length) ln.end = Math.min(ln.end, ln.words[ln.words.length - 1][0] + 1.0);
  if (p === 0 && ln.words.length) ln.start = ln.words[0][0];
}
function removeLine(li) {
  edit("delete a lyric line", () => { S.lyrics.splice(li, 1); });
  if (LY.tap) LY.tap = null;
  renderLyrics();
}
function addLineHere() {
  const text = prompt(`New lyric line at ${fmtTime(S.player.time())}:`, "");
  if (!text || !text.trim()) return;
  const t = S.player.time(), n = text.trim().split(/\s+/).length;
  edit("add a lyric line", () => {
    S.lyrics = lyrLines();
    S.lyrics.push({ start: t, end: t + 0.35 * n + 0.3, words: spread(t, t + 0.35 * n, text.trim()) });
    tidyLyrics();
  });
  renderLyrics();
}
function shiftLyrics(sec) {
  if (!lyrLines().length) return;
  edit(`lyrics ${sec > 0 ? "later" : "earlier"} by ${Math.abs(sec * 1000)} ms`, () => {
    const d = Math.max(sec, -Math.min(...lyrLines().map((ln) => ln.start)));
    for (const ln of S.lyrics) { ln.start += d; ln.end += d; for (const w of ln.words) w[0] = Math.round((w[0] + d) * 1000) / 1000; }
  });
  renderLyrics();
}

// ---------------------------------------------------------------- tap timing
function toggleTap() {
  if (LY.tap) { LY.tap = null; showTap(); renderLyrics(); return; }
  const ws = allWords();
  if (!ws.length) return toast("No lyrics to time yet", true);
  const now = S.player.time(), next = ws.find((w) => w.t >= now - 0.25) || ws[0];
  LY.tap = { li: next.li, wi: next.wi };
  showTap(); renderLyrics();
  toast("Tap timing: play the song and press W as each highlighted (yellow) word starts.");
}
function tapWord() {
  if (!LY.tap) return toggleTap();
  const { li, wi } = LY.tap, ln = lyrLines()[li];
  if (!ln || !ln.words[wi]) { LY.tap = null; showTap(); return; }
  const [lo] = wordBounds(li, wi), t = Math.round(Math.max(lo, S.player.time()) * 1000) / 1000;
  edit(`time "${ln.words[wi][1]}"`, () => {
    const w = ln.words[wi], old = w[0];
    w[0] = t;
    if (wi === 0) ln.start = t;
    // later words of the line that the tap passed are pushed after it; the rest keep their times
    for (let i = wi + 1; i < ln.words.length; i++) if (ln.words[i][0] <= t) ln.words[i][0] = t + LY_GAP * (i - wi); else break;
    ln.end = Math.max(ln.end, ln.words[ln.words.length - 1][0] + 0.2);
    return old !== t;
  });
  const ws = allWords(), i = ws.findIndex((w) => w.li === li && w.wi === wi);
  LY.tap = i >= 0 && i < ws.length - 1 ? { li: ws[i + 1].li, wi: ws[i + 1].wi } : null;
  if (!LY.tap) toast("That was the last word.");
  showTap(); renderLyrics();
}
function showTap() {
  $("#ltap").textContent = LY.tap ? "Stop tapping" : "Tap timing";
  $("#ltap").classList.toggle("ghost", !LY.tap);
}

// ---------------------------------------------------------------- dragging words on the waveform
function wordAt(x, y, L) {
  if (x < L.waveX || x > L.waveX + L.wave) return null;
  const now = S.player.time();
  let best = null, bd = LY_GRAB;
  lyrLines().forEach((ln, li) => ln.words.forEach((w, wi) => {
    const d = Math.abs(yOf(w[0], now, L) - y);
    if (d < bd) { bd = d; best = { li, wi }; }
  }));
  return best;
}
window.addEventListener("pointerdown", (e) => {
  if (!LY.open || e.button !== 0 || e.target !== hw || !S.chart) return;
  const r = hw.getBoundingClientRect(), L = layout(), hit = wordAt(e.clientX - r.left, e.clientY - r.top, L);
  if (!hit) return;
  e.stopImmediatePropagation(); e.preventDefault();
  LY.drag = { ...hit, base: snapshot(), bounds: wordBounds(hit.li, hit.wi), moved: false };
  hw.setPointerCapture(e.pointerId);
}, true);
window.addEventListener("pointermove", (e) => {
  if (!LY.open || !S.chart) return;
  const r = hw.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, L = layout();
  if (LY.drag) {
    e.stopImmediatePropagation();
    const d = LY.drag, ln = S.lyrics[d.li], [lo, hi] = d.bounds;
    const t = Math.round(Math.max(lo, Math.min(hi, tOf(y, S.player.time(), L))) * 1000) / 1000;
    ln.words[d.wi][0] = t;
    if (d.wi === 0) ln.start = t;
    fixLine(ln);
    d.moved = true;
    return;
  }
  LY.hover = e.target === hw ? wordAt(x, y, L) : null;
  if (!TP.open) hw.style.cursor = LY.hover ? "ns-resize" : "";
}, true);
window.addEventListener("pointerup", (e) => {
  if (!LY.drag) return;
  e.stopImmediatePropagation();
  const d = LY.drag;
  LY.drag = null;
  if (d.moved) {
    E.undo.push({ snap: d.base, label: `move "${S.lyrics[d.li].words[d.wi][1]}"` }); E.redo = [];
    if (E.undo.length > 200) E.undo.shift();
    markDirty(); renderLyrics();
  }
}, true);

// drawn by editor.js's draw() each frame: the words on the waveform while the panel is open
function lyricsOverlay(L, now) {
  if (!LY.open || !lyrLines().length) return;
  const x = L.waveX, w = L.wave - 8;
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  lyrLines().forEach((ln, li) => {
    const ya = yOf(ln.start, now, L), yb = yOf(ln.end, now, L);
    if (ya < -20 && yb < -20 || ya > H + 20 && yb > H + 20) return;
    ctx.fillStyle = colors.accent; ctx.globalAlpha = 0.12;
    ctx.fillRect(x, yb, w, ya - yb);                       // the line's span
    ctx.globalAlpha = 1;
    ln.words.forEach((wd, wi) => {
      const y = yOf(wd[0], now, L);
      if (y < -20 || y > H + 20) return;
      const hot = (LY.drag && LY.drag.li === li && LY.drag.wi === wi) || (LY.hover && LY.hover.li === li && LY.hover.wi === wi);
      const tap = LY.tap && LY.tap.li === li && LY.tap.wi === wi;
      ctx.fillStyle = tap ? "#facc15" : hot ? colors.accent : colors.text;
      ctx.fillRect(x, y - (hot || tap ? 1 : 0.5), 10, hot || tap ? 2 : 1);
      ctx.font = `${hot || tap ? 700 : 500} 11px system-ui, sans-serif`;
      const label = wd[1].length > 16 ? wd[1].slice(0, 15) + "…" : wd[1];
      ctx.globalAlpha = 0.8; ctx.fillStyle = colors.bg;                    // readable over the waveform
      ctx.fillRect(x + 11, y - 7, Math.min(w - 12, ctx.measureText(label).width + 4), 14);
      ctx.globalAlpha = 1;
      ctx.fillStyle = tap ? "#facc15" : hot ? colors.accent : colors.text;
      ctx.fillText(wd[1].length > 16 ? wd[1].slice(0, 15) + "…" : wd[1], x + 13, y);
    });
  });
}

// ---------------------------------------------------------------- open / close
function toggleLyrics() {
  const p = $("#lyricspanel");
  p.hidden = !p.hidden;
  LY.open = !p.hidden;
  if (LY.open) {
    $("#issues").hidden = true; $("#history").hidden = true; $("#reports").hidden = true;
    if (typeof TP !== "undefined" && TP.open) toggleTempo();
    if (typeof closeRechart === "function" && !$("#rechart").hidden) closeRechart();
    renderLyrics();
  } else { LY.hover = null; LY.tap = null; showTap(); hw.style.cursor = ""; }
}
function closeLyrics() { if (LY.open) toggleLyrics(); }
$("#lyrbtn").addEventListener("click", toggleLyrics);
$("#lclose").addEventListener("click", closeLyrics);
$("#ladd").addEventListener("click", addLineHere);
$("#ltap").addEventListener("click", toggleTap);
$("#learly").addEventListener("click", (e) => shiftLyrics(e.shiftKey ? -0.2 : -0.05));
$("#llate").addEventListener("click", (e) => shiftLyrics(e.shiftKey ? 0.2 : 0.05));
window.addEventListener("keydown", (e) => {
  if (!LY.open || e.ctrlKey || e.metaKey || e.altKey || (e.target.matches && e.target.matches("input, select, textarea"))) return;
  if (e.key === "w" || e.key === "W") { e.preventDefault(); e.stopImmediatePropagation(); tapWord(); }
}, true);
// after Undo / Redo the list may be out of date
function lyricsChanged() { if (LY.tap && !(lyrLines()[LY.tap.li] || { words: [] }).words[LY.tap.wi]) LY.tap = null; if (LY.open) renderLyrics(); }
