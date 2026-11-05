"use strict";
// riffchart editor, milestone 4: live playability check. After every edit (debounced) the chart is sent
// to /api/check, which runs the evaluation checker (configs/eval.yaml thresholds) and returns each
// problem located at a note. Problems show on the highway, in the overview, on the difficulty tabs and in
// a list; "." and "," jump between them.

const C = { result: null, timer: null, seq: 0, pending: false, cur: -1 };
const ERR = "#ef4444", WARN = "#eab308";
const LABEL = { expert: "Expert", hard: "Hard", medium: "Medium", easy: "Easy" };

function scheduleCheck(delay = 300) {
  clearTimeout(C.timer);
  C.pending = true;
  C.timer = setTimeout(runCheck, delay);
}
async function runCheck() {
  if (!S.edit) return;
  const seq = ++C.seq;
  try {
    const r = await fetch("/api/check", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(chartPayload()) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const res = await r.json();
    if (seq !== C.seq) return;                      // a newer check is on its way
    C.result = res; C.pending = false;
    for (const d of DIFFS) res[d].issues.forEach((x) => { x.sec = S.tempo.tickToSec(x.tick) + S.chart.offset_s; });
    renderIssues(); drawOverview();
  } catch (e) {
    C.pending = false;
    $("#issuesbtn").textContent = "Check failed"; $("#issuesbtn").className = "ghost small issuesbtn err";
    $("#issuesbtn").title = "Playability check failed: " + e.message;
  }
}

const showWarn = () => $("#iwarn").checked;
function visibleIssues(d) {
  const r = C.result && C.result[d];
  return r ? r.issues.filter((x) => x.severity === "error" || showWarn()) : [];
}

function renderIssues() {
  if (!C.result) return;
  const d = curDiff(), r = C.result[d], btn = $("#issuesbtn");
  const e = r.errors, w = r.warnings;
  btn.className = "ghost small issuesbtn " + (e ? "err" : w && showWarn() ? "warn" : "ok");
  btn.textContent = e || (w && showWarn())
    ? `⚠ ${e} problem${e === 1 ? "" : "s"}${w ? ` · ${w} warning${w === 1 ? "" : "s"}` : ""}` : "✓ Playable";
  btn.title = "Playability check (same rules as the evaluation). Click for the list; . and , jump to the next / previous.";
  // badges on the difficulty tabs
  document.querySelectorAll("#tabs button").forEach((b) => {
    const dd = b.dataset.diff;
    if (dd === "all") return;
    const n = C.result[dd] ? C.result[dd].errors : 0;
    b.innerHTML = LABEL[dd] + (n ? `<span class="cnt" title="${n} playability problem${n > 1 ? "s" : ""}">${n}</span>` : "");
  });
  const items = visibleIssues(d);
  $("#ititle").textContent = `${LABEL[d]}: ${items.length ? `${items.length} to look at` : "no problems"}`;
  $("#ilist").innerHTML = items.length ? items.map((x, i) => {
    const bb = S.tempo.barBeat(x.tick);
    return `<li class="${x.severity}${i === C.cur ? " cur" : ""}" data-i="${i}"><span class="dot"></span>` +
      `<span class="ttl">${esc(x.title)}</span><span class="when">${fmtTime(x.sec).slice(0, -2)} · bar ${bb.bar}</span>` +
      `<span class="det">${esc(x.detail)}</span></li>`;
  }).join("") : `<li class="none">Nothing unplayable in ${LABEL[d]}.${C.result[d].warnings && !showWarn() ? " (Warnings hidden.)" : ""}</li>`;
}

function gotoIssue(i) {
  const d = curDiff(), items = visibleIssues(d);
  if (!items.length) { toast(`No playability problems in ${LABEL[d]}`); return; }
  C.cur = (i + items.length) % items.length;
  const x = items[C.cur];
  S.player.seek(x.sec - 0.6);
  select(d, [x.tick]);
  renderIssues();
  const li = document.querySelector(`#ilist li[data-i="${C.cur}"]`);
  if (li) li.scrollIntoView({ block: "nearest" });
  toast(`${C.cur + 1}/${items.length} · ${x.title}: ${x.detail}`, x.severity === "error");
}
function stepIssue(dir) {   // next / previous relative to the playhead
  const d = curDiff(), items = visibleIssues(d);
  if (!items.length) return gotoIssue(0);
  const now = S.player.time() + 0.6;                // the playhead sits 0.6 s before a visited problem
  let i;
  if (dir > 0) { i = items.findIndex((x) => x.sec > now + 0.01); if (i < 0) i = 0; }
  else { i = -1; items.forEach((x, k) => { if (x.sec < now - 0.01) i = k; }); if (i < 0) i = items.length - 1; }
  gotoIssue(i);
}

function toggleIssues() {
  $("#issues").hidden = !$("#issues").hidden;
  if (!$("#issues").hidden && typeof closeRechart === "function") closeRechart();   // same spot on screen
  renderIssues();
}
$("#issuesbtn").addEventListener("click", toggleIssues);
$("#iclose").addEventListener("click", () => { $("#issues").hidden = true; });
$("#iwarn").addEventListener("change", () => { C.cur = -1; renderIssues(); drawOverview(); });
$("#ilist").addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) gotoIssue(+li.dataset.i); });
$("#tabs").addEventListener("click", () => setTimeout(() => { C.cur = -1; renderIssues(); }, 0));
window.addEventListener("keydown", (e) => {
  if (e.target.closest("input[type=text], select") || e.ctrlKey || e.metaKey || e.altKey || !S.edit) return;
  if (e.key === "." || e.code === "Period") { e.preventDefault(); stepIssue(1); }
  else if (e.key === "," || e.code === "Comma") { e.preventDefault(); stepIssue(-1); }
  else if (e.key === "i" || e.key === "I") toggleIssues();
});

// ---------------------------------------------------------------- drawing hooks
function drawIssueMarks(h, L) {
  const items = visibleIssues(h.diff);
  if (!items.length) return;
  const now = S.player.time();
  ctx.font = "700 11px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const x of items) {
    const y = yOf(x.sec, now, L);
    if (y < -20 || y > H + 20) continue;
    const col = x.severity === "error" ? ERR : WARN;
    ctx.fillStyle = col; ctx.globalAlpha = 0.13; ctx.fillRect(h.x, y - 13, h.w, 26); ctx.globalAlpha = 1;
    ctx.fillRect(h.x - 9, y - 13, 3, 26);
    ctx.beginPath(); ctx.arc(h.x - 18, y, 8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#111"; ctx.fillText("!", h.x - 18, y + 0.5);
  }
  // the sustain-overlap case: draw the overlapping part of the tail in red
  const tr = S.tracks[h.diff];
  for (const x of items) {
    if (x.rule !== "sustain_overlap") continue;
    const i = bsearch(tr.on, x.sec - 1e-6);
    if (i + 1 >= tr.on.length) continue;
    const ya = yOf(tr.on[i + 1], now, L), yb = yOf(tr.off[i], now, L);
    if (ya < -20 && yb < -20) continue;
    ctx.strokeStyle = ERR; ctx.lineWidth = 3; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(h.x + 4, ya); ctx.lineTo(h.x + h.w - 4, ya); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = ERR; ctx.globalAlpha = 0.25; ctx.fillRect(h.x, yb, h.w, ya - yb); ctx.globalAlpha = 1;
  }
}
function drawOverviewIssues(c, w, h, dur) {
  const d = S.view === "all" ? "expert" : S.diff;
  for (const x of visibleIssues(d)) {
    c.fillStyle = x.severity === "error" ? ERR : WARN;
    c.fillRect(0, h - (x.sec / dur) * h - 1.5, 10, 3);
  }
}
