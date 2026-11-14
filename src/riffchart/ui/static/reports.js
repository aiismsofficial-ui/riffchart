"use strict";
// Report a spot (loaded after edit.js): F flags the place under the playhead where the chart goes wrong, with a
// short note. Spots are kept with the song (.riffchart/reports.json), drawn as red flags beside the highway and
// listed in the Reported spots panel; the library's "Feedback file" puts every song's spots in one small file
// (names, times, your notes and the notes around each spot; no audio) to send to the developer.
const RP = { list: [] };

async function loadReports() {
  try {
    RP.list = await fetch(`/api/songs/${encodeURIComponent(songId)}/reports`).then((r) => (r.ok ? r.json() : []));
  } catch { RP.list = []; }
  renderReports();
}
function excerptAt(d, tick) {         // the chosen difficulty's notes within 4 beats either side
  const span = S.chart.resolution * 4, ns = S.edit[d].notes, types = finalTypes(ns);
  return ns.map((n, i) => [n[0], n[1], n[2], types[i]]).filter((n) => Math.abs(n[0] - tick) <= span);
}
async function reportSpot() {
  if (!S.edit) return;
  const time = S.player.time(), tick = Math.max(0, Math.round(S.tempo.secToTick(Math.max(0, time - off()))));
  const d = curDiff(), where = `${fmtTime(time)} (${fmtBar(tick)}, ${d})`;
  const note = prompt(`Report this spot: ${where}\n\nWhat's wrong here? (for example: "notes off the beat", "missed riff", "chords too hard")`, "");
  if (note === null) return;
  try {
    const r = await fetch(`/api/songs/${encodeURIComponent(songId)}/reports`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ time_s: time, tick, difficulty: d, note, excerpt: excerptAt(d, tick) }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    RP.list.push(j); renderReports();
    toast(`Spot reported at ${where}. The library's Feedback file collects them.`);
  } catch (e) { toast("Couldn't save the spot: " + e.message, true); }
}
async function removeReport(id) {
  try {
    RP.list = await fetch(`/api/songs/${encodeURIComponent(songId)}/reports/${id}`, { method: "DELETE" }).then((r) => r.json());
    renderReports();
  } catch (e) { toast(e.message, true); }
}
function renderReports() {
  const btn = $("#rptbtn");
  btn.textContent = RP.list.length ? `⚑ Reported (${RP.list.length})` : "⚑ Report a spot";
  const ol = $("#rlist");
  if (!ol) return;
  if (!RP.list.length) { ol.innerHTML = `<li class="none">Nothing reported yet. Press F where the chart goes wrong.</li>`; return; }
  ol.innerHTML = "";
  for (const r of RP.list.slice().sort((a, b) => a.time_s - b.time_s)) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="ttl">${fmtTime(r.time_s)} · ${esc(r.difficulty)}</span>
      <button class="ghost small icon" title="Delete this spot" aria-label="Delete">✕</button>
      <span class="hcount">${esc(r.note || "(no note)")}</span>`;
    li.querySelector(".ttl").addEventListener("click", () => S.player.seek(r.time_s));
    li.querySelector("button").addEventListener("click", () => removeReport(r.id));
    ol.appendChild(li);
  }
}
function toggleReports() {
  const p = $("#reports");
  p.hidden = !p.hidden;
  if (!p.hidden) {
    $("#issues").hidden = true; $("#history").hidden = true;
    if (typeof TP !== "undefined" && TP.open) toggleTempo();
    if (typeof closeRechart === "function" && !$("#rechart").hidden) closeRechart();
    renderReports();
  }
}
// drawn by editor.js's draw() each frame: a red flag beside the highway at each reported spot
function reportsOverlay(L, now) {
  if (!RP.list.length) return;
  ctx.font = "600 11px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
  for (const r of RP.list) {
    const y = yOf(r.time_s, now, L);
    if (y < -20 || y > H + 20) continue;
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(L.waveX, y - 1, L.rightX - L.waveX - 12, 2);
    ctx.fillText(`⚑ ${r.note ? r.note.slice(0, 28) : "reported"}`, L.rightX - 4 + 12, y - 12);
  }
}
$("#rptbtn").addEventListener("click", toggleReports);
$("#rclose2").addEventListener("click", () => { $("#reports").hidden = true; });
$("#rhere").addEventListener("click", reportSpot);
window.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || (e.target.matches && e.target.matches("input, select, textarea"))) return;
  if (e.key === "f" || e.key === "F") { e.preventDefault(); e.stopImmediatePropagation(); reportSpot(); }
}, true);
loadReports();
