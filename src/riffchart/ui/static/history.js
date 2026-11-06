"use strict";
// Chart history (loaded after edit.js): the versions earlier saves and re-charts replaced. Opening one puts it
// in the editor as an ordinary edit, so Undo goes back and Save restores it (the current chart then goes to
// the history too, so nothing is ever lost).
const HI = { list: [], open: null, loading: false };

function fmtWhen(sec, secs = false) {
  const d = new Date(sec * 1000), now = new Date();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: secs ? "2-digit" : undefined });
  if (d.toDateString() === now.toDateString()) return `today ${time}`;
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return `yesterday ${time}`;
  return `${d.toLocaleDateString([], { day: "numeric", month: "short", year: d.getFullYear() === now.getFullYear() ? undefined : "numeric" })} ${time}`;
}
function countsHtml(notes) {
  if (notes.error) return `<span class="down">can't be read</span>`;
  return DIFFS.map((d) => {
    const n = notes[d] || 0, cur = S.edit ? S.edit[d].notes.length : n, diff = n - cur;
    const delta = diff ? ` <span class="${diff > 0 ? "up" : "down"}">${diff > 0 ? "+" : ""}${diff}</span>` : "";
    return `${d[0].toUpperCase()} <b>${n}</b>${delta}`;
  }).join(" · ");
}
function renderHistory() {
  const ol = $("#hlist");
  if (HI.loading) { ol.innerHTML = `<li class="none">Loading…</li>`; return; }
  if (!HI.list.length) {
    ol.innerHTML = `<li class="none">No earlier versions yet. Every Save and Re-chart keeps the version it replaces.</li>`;
    return;
  }
  ol.innerHTML = "";
  const seen = new Map();                                   // same minute twice: show seconds
  for (const v of HI.list) seen.set(fmtWhen(v.made), (seen.get(fmtWhen(v.made)) || 0) + 1);
  for (const v of HI.list) {
    const when = fmtWhen(v.made, seen.get(fmtWhen(v.made)) > 1);
    const li = document.createElement("li");
    li.classList.toggle("cur", HI.open === v.name);
    li.title = `Replaced by ${v.reason === "rechart" ? "a re-chart" : "a save"} ${fmtWhen(v.replaced)}`;
    li.innerHTML = `<span class="ttl">${when}</span>
      <span class="when">${v.reason === "rechart" ? "before a re-chart" : "before a save"}</span>
      <span class="hcount">${countsHtml(v.notes)}</span>`;
    if (!v.notes.error) li.addEventListener("click", () => openVersion(v));
    ol.appendChild(li);
  }
}
async function refreshHistory() {
  HI.loading = !HI.list.length; renderHistory();
  try {
    HI.list = await fetch(`/api/songs/${encodeURIComponent(songId)}/history`).then((r) => { if (!r.ok) throw new Error(r.statusText); return r.json(); });
  } catch (e) { toast("Could not list earlier versions: " + e.message, true); }
  HI.loading = false; renderHistory();
}
async function openVersion(v) {
  if (HI.open === v.name) return;
  try {
    const r = await fetch(`/api/songs/${encodeURIComponent(songId)}/history/${encodeURIComponent(v.name)}`);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    edit(`open the version from ${fmtWhen(v.made)}`, () => {
      S.edit = editTracks(j);
      S.chart.sections = j.sections;
      setTiming({ tempo: j.tempo, ts: j.ts, res: j.resolution, thr: j.hopo_threshold });
      E.sel = { diff: null, ticks: new Set() };
    });
    HI.open = v.name;
    if (typeof showSync === "function") showSync();
    renderHistory();
    toast(`Showing the version from ${fmtWhen(v.made)}. Save to make it the chart again, or Undo to go back.`);
  } catch (e) { toast("Could not open that version: " + e.message, true); }
}
function toggleHistory() {
  const p = $("#history");
  p.hidden = !p.hidden;
  if (!p.hidden) {
    $("#issues").hidden = true;
    if (typeof closeRechart === "function" && !$("#rechart").hidden) closeRechart();
    refreshHistory();
  }
}
$("#histbtn").addEventListener("click", toggleHistory);
$("#hclose").addEventListener("click", () => { $("#history").hidden = true; });
// called by edit.js: after Undo / Redo the "open" marker may be wrong; after a save the list has grown
function historyChanged(saved) {
  HI.open = null;
  if ($("#history").hidden) return;
  if (saved) refreshHistory(); else renderHistory();
}
