"use strict";
const $ = (s) => document.querySelector(s);
const AUDIO = [".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".flac", ".wav", ".mp4", ".mkv", ".mka", ".wma"];
const DIFFS = ["expert", "hard", "medium", "easy"];
let songs = [], jobs = [], status = null, lastSongsKey = "";

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

let toastTimer;
function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg; t.classList.toggle("bad", bad); t.classList.remove("interactive"); t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), bad ? 6000 : 3500);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* empty */ }
  if (!r.ok) throw Object.assign(new Error((body && body.detail) || r.statusText), { status: r.status });
  return body;
}
const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data ?? {}) });

// ---------- options
function options() {
  return {
    difficulties: [...document.querySelectorAll("input[name=diff]:checked")].map((x) => x.value),
    density: parseFloat($("#density").value), chords: $("#chords").value, follow: $("#follow").value,
    stems: $("#stems").checked, overwrite: $("#overwrite").checked,
  };
}
function densLabel(v) {   // matches post/density.py: -1 keeps 50% of notes, +1 adds up to 60% below Expert
  v = parseFloat(v);
  if (Math.abs(v) < 0.05) return "normal";
  if (v < 0) return `sparser: keeps about ${Math.round((1 + 0.5 * v) * 100)}% of the notes`;
  return `denser: up to +${Math.round(60 * v)}% notes on Hard–Easy, a little more on Expert`;
}
const OPT_KEY = "riffchart.options", OLD_OPT_KEY = "autochart.options";   // old name: read once
function saveOptions() { try { localStorage.setItem(OPT_KEY, JSON.stringify(options())); } catch { /* private mode */ } }
function loadOptions() {
  let o; try { o = JSON.parse(localStorage.getItem(OPT_KEY) || localStorage.getItem(OLD_OPT_KEY) || "null"); } catch { o = null; }
  if (!o) return;
  document.querySelectorAll("input[name=diff]").forEach((x) => (x.checked = (o.difficulties || DIFFS).includes(x.value)));
  $("#density").value = o.density ?? 0; $("#chords").value = o.chords ?? "auto"; $("#follow").value = o.follow ?? "auto";
  $("#stems").checked = o.stems ?? true; // overwrite is deliberately not remembered
}
$("#density").addEventListener("input", (e) => { $("#densout").textContent = densLabel(e.target.value); updateSetSum(); });
document.querySelector(".opts").addEventListener("change", () => { saveOptions(); updateSetSum(); });
function updateSetSum() {
  const o = options(), all = o.difficulties.length === 4;
  const chords = { auto: "chords follow the song", low: "fewer chords", mid: "some chords", high: "more chords" }[o.chords];
  $("#setsum").textContent = [all ? "all difficulties" : o.difficulties.map((d) => d[0].toUpperCase() + d.slice(1)).join(", ") || "no difficulty!",
    densLabel(o.density) === "normal" ? "normal density" : densLabel(o.density).split(":")[0], chords,
    { auto: "guitar + other instruments where it's quiet", guitar: "guitar only", all: "whole song" }[o.follow],
    o.stems ? "separate guitar track" : "one audio track"].concat(o.overwrite ? ["re-chart existing"] : []).join(" · ");
}

function checkOptions() {
  if (!options().difficulties.length) { toast("Pick at least one difficulty", true); return false; }
  return true;
}

// ---------- adding songs
async function uploadFiles(list) {
  // skip a Clone Hero folder's background video (video.mp4 / video.webm); a single dropped file always counts
  const isBgVideo = (n) => list.length > 1 && /^(video|background|bg)\.[^.]+$/i.test(n);
  const files = list.filter((f) => AUDIO.some((e) => f.file.name.toLowerCase().endsWith(e)) && !isBgVideo(f.file.name));
  const ignored = list.length - files.length;
  if (!files.length) { toast(ignored ? "No supported audio files there (mp3, m4a, ogg, opus, webm, flac, wav, mp4)" : "Nothing to add", true); return; }
  if (!checkOptions()) return;
  let added = 0;
  toast(`Uploading ${files.length} file${files.length > 1 ? "s" : ""}…`);
  for (let i = 0; i < files.length; i += 8) {            // a few at a time keeps memory low for big folders
    const fd = new FormData();
    for (const f of files.slice(i, i + 8)) fd.append("files", f.file, f.path || f.file.name);
    fd.append("options", JSON.stringify(options()));
    try {
      const r = await api("/api/jobs/upload", { method: "POST", body: fd });
      added += r.added.length;
      refreshJobs();
    } catch (e) { toast("Upload failed: " + e.message, true); return; }
  }
  toast(`Added ${added} song${added === 1 ? "" : "s"} to the queue` + (ignored ? ` (${ignored} non-audio file${ignored > 1 ? "s" : ""} ignored)` : ""));
}

function readEntry(entry, prefix = "") {
  return new Promise((resolve) => {
    if (entry.isFile) {
      entry.file((file) => resolve([{ file, path: prefix + file.name }]), () => resolve([]));
    } else if (entry.isDirectory) {
      const reader = entry.createReader(), all = [];
      const next = () => reader.readEntries(async (batch) => {
        if (!batch.length) {
          const nested = await Promise.all(all.map((e) => readEntry(e, prefix + entry.name + "/")));
          resolve(nested.flat());
        } else { all.push(...batch); next(); }
      }, () => resolve([]));
      next();
    } else resolve([]);
  });
}

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", async (e) => {
  e.preventDefault();
  const items = [...(e.dataTransfer.items || [])].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  const list = items.length ? (await Promise.all(items.map((x) => readEntry(x)))).flat()
    : [...e.dataTransfer.files].map((file) => ({ file, path: file.name }));
  uploadFiles(list);
});
drop.addEventListener("click", (e) => { if (e.target.tagName !== "INPUT" && e.target.tagName !== "LABEL") $("#pick").click(); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#pick").click(); } });
window.addEventListener("dragover", (e) => e.preventDefault());           // a missed drop must not open the file
window.addEventListener("drop", (e) => e.preventDefault());
$("#pick").addEventListener("change", (e) => { uploadFiles([...e.target.files].map((file) => ({ file, path: file.name }))); e.target.value = ""; });

$("#pathform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const raw = $("#pathin").value.trim();
  if (!raw || !checkOptions()) return;
  try {
    const r = await post("/api/jobs/paths", { paths: raw.split(/\r?\n|;/), options: options() });
    if (r.added.length) { toast(`Added ${r.added.length} song${r.added.length > 1 ? "s" : ""} to the queue`); $("#pathin").value = ""; }
    if (r.skipped.length) toast(r.skipped.slice(0, 3).join(" · ") + (r.skipped.length > 3 ? ` (+${r.skipped.length - 3} more)` : ""), true);
    else if (!r.added.length) toast("No audio files found there", true);
    refreshJobs();
  } catch (err) { toast(err.message, true); }
});

// ---------- queue
function fmtDur(s) { s = Math.max(0, Math.round(s)); return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`; }

function editorUrl(id) { return `editor.html?song=${encodeURIComponent(id)}`; }
const seenDone = new Set();
let firstJobsLoaded = false;

function renderJobs() {
  $("#queuecard").hidden = !jobs.length;
  const n = (st) => jobs.filter((j) => j.status === st).length;
  const queued = n("queued"), running = n("running");
  // ~30-40 s per song on this PC, +15 s model loading on the first song
  const eta = (queued + running * 0.5) * 35 + (status && status.model !== "ready" && (queued || running) ? 15 : 0);
  $("#queuesum").textContent = (queued || running)
    ? `${running ? "1 charting, " : ""}${queued} waiting · about ${fmtDur(eta)} left` : `${n("done")} done`;
  $("#jobs").innerHTML = jobs.map((j) => {
    const pct = Math.round(j.progress * 100);
    let msg = j.status === "running" ? (j.stage_label || j.message || "starting") + ` · ${pct}%`
      : j.status === "queued" ? (j.message === "restarted" ? "waiting (restarted)" : "waiting") : j.message;
    if (j.status === "running" && j.message && j.stage_label === "") msg = j.message;
    const ready = j.song && (j.status === "done" || j.status === "skipped");
    const acts = j.status === "queued" ? `<button class="ghost small" data-cancel="${j.id}">Cancel</button>`
      : j.status === "running" ? ""
      : (j.status === "failed" ? `<button class="ghost small" data-retry="${j.id}">Retry</button>` : "") +
        (ready ? `<a class="btn small" href="${editorUrl(j.song)}">Open in editor</a>` : "") +
        `<button class="ghost small icon" data-cancel="${j.id}" title="Remove from this list" aria-label="Remove from this list">✕</button>`;
    const label = j.target ? `Re-charting ${j.name}` : j.name;
    return `<li class="job ${j.status}"><div class="name">${esc(label)}<span class="tag ${j.status}">${j.status}</span></div>
      <div class="acts">${acts}</div><div class="bar"><i style="width:${j.status === "queued" ? 0 : pct}%"></i></div>
      <div class="msg">${esc(msg)}</div></li>`;
  }).join("");
  // a song just finished while the page is open: say so, with a way straight into the editor
  for (const j of jobs) {
    if (j.status !== "done" || seenDone.has(j.id)) continue;
    seenDone.add(j.id);
    if (firstJobsLoaded) toastHtml(`<b>${esc(j.name)}</b> is ready · <a href="${editorUrl(j.song)}">Open in editor</a>`);
  }
  firstJobsLoaded = true;
}

function toastHtml(html) {
  const t = $("#toast");
  t.innerHTML = html; t.classList.remove("bad"); t.classList.add("show", "interactive");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show", "interactive"), 9000);
}

$("#jobs").addEventListener("click", async (e) => {
  const b = e.target.closest("button"); if (!b) return;
  try {
    if (b.dataset.cancel) await api(`/api/jobs/${b.dataset.cancel}`, { method: "DELETE" });
    if (b.dataset.retry) await post(`/api/jobs/${b.dataset.retry}/retry`);
  } catch (err) { toast(err.message, true); }
  refreshJobs();
});
$("#clearjobs").addEventListener("click", async () => { await post("/api/jobs/clear"); refreshJobs(); });

// ---------- songs
function when(ts) {
  const d = new Date(ts * 1000), now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  return sameDay ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : d.toLocaleDateString();
}
function fmtLen(s) { if (!s) return "–"; s = Math.round(s); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }
const DLABEL = { expert: "Expert", hard: "Hard", medium: "Medium", easy: "Easy" };

function renderSongs() {
  if (document.querySelector(".menulist:not([hidden])")) return;       // don't yank an open menu away
  const q = $("#filter").value.trim().toLowerCase();
  const list = songs.filter((s) => !q || `${s.title} ${s.artist}`.toLowerCase().includes(q));
  $("#songcount").textContent = songs.length ? `${songs.length}` : "";
  $("#nosongs").hidden = songs.length > 0;
  const chExe = status && status.clone_hero && status.clone_hero.exe;
  const busy = new Map(jobs.filter((j) => j.target && (j.status === "running" || j.status === "queued")).map((j) => [j.target, j]));
  $("#songs").innerHTML = list.map((s) => {
    const notes = DIFFS.map((d) => `<span title="${DLABEL[d]}">${s.notes[d] ?? "–"}</span>`).join(" / ");
    const exp = s.exported ? `<span class="badge ${s.export_stale ? "stale" : ""}" title="${s.export_stale ? "Changed since the last export: export again to update Clone Hero" : "In your Clone Hero folder"}">${s.export_stale ? "● changed since export" : "✓ in Clone Hero"}</span>` : "";
    const j = busy.get(s.id);
    const state = j ? `<span class="tag running">${j.status === "running" ? `re-charting ${Math.round(j.progress * 100)}%` : "re-chart queued"}</span>` : "";
    const id = esc(s.id);
    return `<tr data-id="${id}">
      <td><a class="songlink" href="${editorUrl(s.id)}" title="Open in the editor"><span class="t">${esc(s.title)}</span><span class="a">${esc(s.artist)}</span></a>${exp}${state}${s.error ? `<div class="err">${esc(s.error)}</div>` : ""}</td>
      <td class="num">${notes}</td><td class="num">${fmtLen(s.length_s)}</td><td class="num">${esc(s.diff_guitar ?? "")}</td>
      <td class="when">${when(s.charted)}</td>
      <td><div class="acts">
        <a class="btn small" href="${editorUrl(s.id)}">Edit</a>
        <button class="ghost small" data-test="${id}" data-diff="expert" ${chExe ? 'title="Open in Clone Hero on Expert (the bot plays it)"' : 'disabled title="Clone Hero.exe not found (configs/preview.yaml)"'}>▶ Clone Hero</button>
        <button class="ghost small" data-export="${id}" data-again="${s.exported ? 1 : ""}" title="Copy to your Clone Hero songs folder">${s.exported ? "Export again" : "Export"}</button>
        <div class="menu"><button class="ghost small icon" data-menu="${id}" aria-label="More actions" aria-haspopup="true">⋯</button>
          <div class="menulist" hidden>
            <button data-rechart="${id}" ${j ? "disabled" : ""}>Re-chart with other settings…</button>
            <button data-rename="${id}">Rename…</button>
            ${DIFFS.slice(1).map((d) => `<button data-test="${id}" data-diff="${d}" ${chExe ? "" : "disabled"}>Test in Clone Hero on ${DLABEL[d]}</button>`).join("")}
            <button data-reveal="${id}">Open song folder</button>
            <button data-remove="${id}" class="danger" ${j ? "disabled" : ""}>Remove from library…</button>
          </div></div>
      </div></td></tr>`;
  }).join("");
}
$("#filter").addEventListener("input", renderSongs);

function closeMenus() {
  const open = document.querySelector(".menulist:not([hidden])");
  document.querySelectorAll(".menulist").forEach((m) => (m.hidden = true));
  if (open) renderSongs();                                              // catch up on skipped refreshes
}
document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) closeMenus(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenus(); });
window.addEventListener("scroll", closeMenus, { passive: true });
window.addEventListener("resize", closeMenus);

$("#songs").addEventListener("click", async (e) => {
  const b = e.target.closest("button"); if (!b || b.disabled) return;
  if (b.dataset.menu) {
    const m = b.nextElementSibling, open = m.hidden;
    closeMenus();
    if (open) {                      // fixed position: the table's scroll box would clip it
      m.hidden = false;
      const r = b.getBoundingClientRect(), h = m.offsetHeight, w = m.offsetWidth;
      m.style.left = Math.max(8, r.right - w) + "px";
      m.style.top = (r.bottom + 4 + h > window.innerHeight ? Math.max(8, r.top - 4 - h) : r.bottom + 4) + "px";
    }
    return;
  }
  closeMenus();
  const song = (id) => songs.find((s) => s.id === id);
  const enc = (id) => encodeURIComponent(id);
  try {
    if (b.dataset.export) {
      if (b.dataset.again && !confirm("Replace the copy in your Clone Hero folder with this version?")) return;
      const r = await post(`/api/songs/${enc(b.dataset.export)}/export`, { overwrite: !!b.dataset.again });
      toast(`Exported to ${r.folder}. In Clone Hero, rescan songs to see it.`);
      refreshSongs(true);
    } else if (b.dataset.test) {
      await post(`/api/songs/${enc(b.dataset.test)}/test`, { difficulty: b.dataset.diff });
      toast(`Starting Clone Hero on ${DLABEL[b.dataset.diff]}… (the bot plays it; press Esc in Clone Hero to quit)`);
    } else if (b.dataset.reveal) {
      await post(`/api/songs/${enc(b.dataset.reveal)}/reveal`);
    } else if (b.dataset.rechart) {
      openRechart(song(b.dataset.rechart));
    } else if (b.dataset.rename) {
      const s = song(b.dataset.rename);
      const title = prompt("Song title:", s.title); if (title === null) return;
      const artist = prompt("Artist:", s.artist); if (artist === null) return;
      await post(`/api/songs/${enc(s.id)}/meta`, { title, artist });
      toast("Renamed. Export again to update the copy in Clone Hero."); refreshSongs(true);
    } else if (b.dataset.remove) {
      const s = song(b.dataset.remove);
      if (!confirm(`Remove "${s.title}" from your library?\n\nIt is moved to data/ui/trash (not deleted). A copy you exported to Clone Hero stays there.`)) return;
      await api(`/api/songs/${enc(s.id)}`, { method: "DELETE" });
      toast(`Removed "${s.title}" (moved to data/ui/trash)`); refreshSongs(true);
    }
  } catch (err) { toast(err.message, true); }
});

// ---------- re-chart dialog
let rechartId = null;
function openRechart(s) {
  rechartId = s.id;
  $("#rc-title").textContent = `Re-chart “${s.title}”`;
  const o = options();
  document.querySelectorAll("input[name=rcdiff]").forEach((x) => (x.checked = o.difficulties.includes(x.value)));
  $("#rc-density").value = o.density; $("#rc-densout").textContent = densLabel(o.density);
  $("#rc-chords").value = o.chords; $("#rc-follow").value = o.follow;
  $("#rechart").showModal();
}
$("#rc-density").addEventListener("input", (e) => { $("#rc-densout").textContent = densLabel(e.target.value); });
$("#rc-cancel").addEventListener("click", () => $("#rechart").close());
$("#rc-go").addEventListener("click", async () => {
  const difficulties = [...document.querySelectorAll("input[name=rcdiff]:checked")].map((x) => x.value);
  if (!difficulties.length) { toast("Pick at least one difficulty", true); return; }
  const opts = { difficulties, density: parseFloat($("#rc-density").value), chords: $("#rc-chords").value,
    follow: $("#rc-follow").value, stems: true };
  try {
    await post(`/api/songs/${encodeURIComponent(rechartId)}/rechart`, { options: opts });
    $("#rechart").close();
    toast("Re-charting… about 30 seconds. Your current chart is kept in the song's history.");
    refreshJobs(); refreshSongs(true);
  } catch (err) { toast(err.message, true); }
});

// ---------- status + export folder
function renderStatus() {
  if (!status) return;
  const m = status.model, ch = status.clone_hero;
  const mLabel = { ready: "model ready", loading: "loading model…", "not loaded": "model loads on first song", failed: "model failed to load" }[m] || m;
  const mCls = { ready: "ok", loading: "busy", failed: "bad" }[m] || "";
  $("#status").innerHTML = `<span class="pill ${mCls}">${mLabel}</span>` +
    `<span class="pill ${ch.exe ? "ok" : "bad"}">${ch.exe ? "Clone Hero found" : "Clone Hero not found"}</span>`;
  const reg = ch.export_registered;
  $("#exportbar").innerHTML = `Export folder <code>${esc(status.export_dir)}</code>` +
    (reg ? `<span class="badge">✓ added to Clone Hero</span>`
      : `<button class="ghost small" id="register">Add it to Clone Hero's song folders</button>`);
  const btn = $("#register");
  if (btn) btn.addEventListener("click", async () => {
    try {
      await post("/api/clonehero/register");
      toast("Added. Next time you open Clone Hero, scan songs once to pick it up.");
      refreshStatus();
    } catch (err) { toast(err.message, true); }
  });
}

// ---------- polling
async function refreshStatus() {
  try {
    const had = status && status.clone_hero.exe;
    status = await api("/api/status"); renderStatus();
    if (!!had !== !!status.clone_hero.exe) renderSongs();
  } catch { /* server restarting */ }
}
async function refreshJobs() {
  try {
    const prevDone = jobs.filter((j) => j.status === "done").length;
    jobs = await api("/api/jobs"); renderJobs();
    if (jobs.filter((j) => j.status === "done").length !== prevDone) refreshSongs(true);
  } catch { /* server restarting */ }
}
async function refreshSongs(force = false) {
  try {
    const s = await api("/api/songs");
    const key = JSON.stringify(s);
    if (force || key !== lastSongsKey) { songs = s; lastSongsKey = key; renderSongs(); }
  } catch { /* server restarting */ }
}

loadOptions();
$("#densout").textContent = densLabel($("#density").value);
updateSetSum();
refreshStatus(); refreshJobs(); refreshSongs();
let tick = 0;
setInterval(() => {
  tick++;
  const busy = jobs.some((j) => j.status === "running" || j.status === "queued");
  if (busy || tick % 5 === 0) refreshJobs();
  if (tick % 5 === 0) refreshStatus();
  if (tick % 10 === 0) refreshSongs();
}, 1000);
