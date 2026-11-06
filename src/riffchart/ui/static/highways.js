"use strict";
// Highway designer: edit a design, see it live (flat and on a scrolling 3D highway), save it as a preset,
// and use it on all songs, for new songs, or in Clone Hero itself. The server renders every preview, so what
// you see is exactly the image the songs get.
const $ = (s) => document.querySelector(s);
const enc = encodeURIComponent;
const BUILTIN = [["riffchart", "RiffChart"], ["cover", "From each song's cover"], ["none", "None (the player's own)"]];
const STARTERS = [
  { name: "RiffChart amber", top: "#16181e", bottom: "#0e0f13", rail: "#f5a524", divider: "#30343e", glow: 50 },
  { name: "Neon", top: "#0b1026", bottom: "#05060f", rail: "#22e4ff", divider: "#1d2a4a", glow: 90 },
  { name: "Synthwave", top: "#2a0b3d", bottom: "#0d0418", rail: "#ff3ec8", divider: "#4a1d5e", glow: 80,
    text_color: "#ff9a3e" },
  { name: "Crimson", top: "#1e0d0f", bottom: "#0d0607", rail: "#e3242b", divider: "#3e1f22", glow: 60 },
  { name: "Ice", top: "#1a2430", bottom: "#0c1218", rail: "#cfe8ff", divider: "#33475a", glow: 40 },
  { name: "Toxic", top: "#0f1a0b", bottom: "#060a05", rail: "#7cff2e", divider: "#25381c", glow: 70 },
  { name: "Stealth", top: "#141414", bottom: "#0a0a0a", rail: "#5a5a5a", divider: "#262626", glow: 0, rail_width: 8 },
  { name: "Gold", top: "#1c1608", bottom: "#0c0a04", rail: "#ffd24a", divider: "#3d331a", glow: 65, rail_width: 18 },
];

let blank = {}, designs = [], libDefault = "riffchart", chOk = false;
// what's on the page: the design, which saved preset it is (id), and its pattern image
let cur = {}, curId = null, patFile = null, patSaved = false, patRemoved = false, dirty = false;
let renderTimer = null, renderSeq = 0, flatUrl = null;

let toastTimer;
function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg; t.classList.toggle("bad", bad); t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), bad ? 6000 : 4000);
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* not json */ }
  if (!r.ok) throw new Error((body && body.detail) || r.statusText);
  return body;
}
const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data ?? {}) });
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------- form <-> design
const fields = () => document.querySelectorAll("[data-k]");
function toForm() {
  $("#name").value = cur.name;
  for (const el of fields()) {
    const v = cur[el.dataset.k];
    if (el.type === "checkbox") el.checked = !!v; else el.value = v;
    const out = el.parentElement.querySelector("output");
    if (out) out.textContent = el.dataset.k.endsWith("width") || el.dataset.k === "text_size" ? `${v}px` : `${v}%`;
  }
  const hasPat = !!patFile || (patSaved && !patRemoved);
  $("#patname").textContent = patFile ? patFile.name : hasPat ? "saved image" : "none";
  $("#patrm").hidden = !hasPat;
  $("#delete").hidden = !curId;
  for (const b of ["#applyall", "#makedefault", "#install"]) $(b).disabled = false;
  $("#install").hidden = !chOk;
  $("#dirty").textContent = dirty ? "● unsaved changes" : curId ? "saved" : "not saved yet";
  document.querySelectorAll(".hw-item").forEach((b) => b.classList.toggle("on", b.dataset.id === curId));
}
function fromForm() {
  cur.name = $("#name").value.trim() || "My highway";
  for (const el of fields()) {
    const k = el.dataset.k;
    cur[k] = el.type === "checkbox" ? el.checked : el.type === "range" ? Number(el.value) : el.value;
  }
}
function changed() { fromForm(); dirty = true; toForm(); schedule(); }

function load(design, id = null, hasPattern = false) {
  if (dirty && !confirm("Discard the unsaved changes to this design?")) return false;
  cur = { ...blank, ...design }; delete cur.id; delete cur.has_pattern; delete cur.mtime;
  curId = id; patFile = null; patSaved = hasPattern; patRemoved = false; dirty = false;
  toForm(); schedule(0);
  return true;
}

// ---------- preview
function formData(withId) {
  const fd = new FormData();
  fd.append("design", JSON.stringify(withId && curId ? { ...cur, id: curId } : cur));
  if (patFile) fd.append("pattern", patFile);
  else if (patSaved && !patRemoved && curId) fd.append("pattern_from", curId);
  return fd;
}
function schedule(ms = 140) { clearTimeout(renderTimer); renderTimer = setTimeout(render, ms); }
async function render() {
  const seq = ++renderSeq;
  try {
    const r = await fetch("/api/highways/render", { method: "POST", body: formData(false) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const blob = await r.blob();
    if (seq !== renderSeq) return;                         // a newer preview is on its way
    if (flatUrl) URL.revokeObjectURL(flatUrl);
    flatUrl = URL.createObjectURL(blob);
    $("#flat").src = flatUrl;
    $("#road").style.backgroundImage = `url("${flatUrl}")`;
  } catch (e) { toast(`Preview failed: ${e.message}`, true); }
}

// ---------- lists
function thumb(src, name, id, onClick, extra = "") {
  const b = document.createElement("button");
  b.type = "button"; b.className = "hw-item"; b.dataset.id = id || "";
  b.innerHTML = `<span class="hw-thumb" style="${src}"></span><span class="hw-iname">${esc(name)}${extra}</span>`;
  b.addEventListener("click", onClick);
  return b;
}
function starterStyle(s) {
  const d = { ...blank, ...s };
  return `background: linear-gradient(90deg, ${d.rail} 0 6%, transparent 6% 94%, ${d.rail} 94%), linear-gradient(${d.top}, ${d.bottom}, ${d.top})`;
}
function renderLists() {
  const box = $("#designs");
  box.innerHTML = designs.length ? "" : `<p class="muted small">None yet. Pick a starting point below, change it, then Save.</p>`;
  for (const d of designs) {
    const tag = libDefault === `design:${d.id}` ? ` <em class="muted">· new songs</em>` : "";
    box.appendChild(thumb(`background-image:url('/api/highways/${enc(d.id)}/image?v=${d.mtime}')`, d.name, d.id,
      () => load(d, d.id, d.has_pattern), tag));
  }
  const st = $("#starters");
  st.innerHTML = "";
  for (const s of STARTERS) st.appendChild(thumb(starterStyle(s), s.name, "", () => load({ ...s, name: `My ${s.name.replace(/^RiffChart /, "")}` })));
  const sel = $("#libdefault");
  sel.innerHTML = BUILTIN.map(([v, n]) => `<option value="${v}">${n}</option>`).join("")
    + (designs.length ? `<optgroup label="Your designs">${designs.map((d) => `<option value="design:${esc(d.id)}">${esc(d.name)}</option>`).join("")}</optgroup>` : "");
  sel.value = libDefault;
  toForm();
}
async function refresh() {
  const r = await api("/api/highways");
  blank = r.blank; designs = r.designs; libDefault = r.default; chOk = r.clone_hero;
  renderLists();
}

// ---------- save, delete, use
async function save(asCopy = false) {
  fromForm();
  const fd = formData(!asCopy);
  if (patRemoved && !patFile) fd.append("remove_pattern", "true");
  const btns = document.querySelectorAll(".hw-actions button");
  btns.forEach((b) => (b.disabled = true));
  try {
    const r = await api("/api/highways", { method: "POST", body: fd });
    dirty = false;
    curId = r.design.id; cur.name = r.design.name; patFile = null; patSaved = r.design.has_pattern; patRemoved = false;
    await refresh();
    const n = r.updated.length;
    toast(n ? `Saved. ${n} song${n > 1 ? "s" : ""} using it ${n > 1 ? "were" : "was"} redrawn${r.exported.length ? `, ${r.exported.length} updated in Clone Hero` : ""}.`
      : "Saved. Use it below, or pick it in a song's details.");
    return true;
  } catch (e) { toast(e.message, true); return false; }
  finally { btns.forEach((b) => (b.disabled = false)); }
}
async function needSaved() {
  if (curId && !dirty) return true;
  return save();
}

$("#save").addEventListener("click", () => save());
$("#saveas").addEventListener("click", () => { if (curId) cur.name = `${$("#name").value.trim()} copy`.slice(0, 40); save(true); });
$("#delete").addEventListener("click", async () => {
  if (!curId || !confirm(`Delete the design “${cur.name}”?\n\nSongs using it keep their highway.`)) return;
  try {
    await api(`/api/highways/${enc(curId)}`, { method: "DELETE" });
    curId = null; patSaved = false; dirty = true;
    await refresh();
    toast("Deleted. It's still on the page: Save brings it back.");
  } catch (e) { toast(e.message, true); }
});
$("#applyall").addEventListener("click", async () => {
  if (!(await needSaved())) return;
  if (!confirm(`Give every song the highway “${cur.name}”?\n\nSongs with their own highway image are left alone, and so are ` +
    "Clone Hero copies you haven't exported yet. Copies that were up to date are updated.\n\nNew songs get it too.")) return;
  try {
    const r = await post("/api/highways/apply", { highway: `design:${curId}`, songs: "all", export: true, default: true });
    libDefault = r.default;
    renderLists();
    toast(`${r.updated.length} songs now use “${cur.name}”${r.exported.length ? ` (${r.exported.length} updated in Clone Hero: rescan songs there)` : ""}` +
      `${r.skipped_own.length ? `. ${r.skipped_own.length} with their own highway left alone` : ""}.`);
  } catch (e) { toast(e.message, true); }
});
$("#makedefault").addEventListener("click", async () => {
  if (!(await needSaved())) return;
  await setDefault(`design:${curId}`);
});
$("#install").addEventListener("click", async () => {
  if (!(await needSaved())) return;
  try {
    const r = await post(`/api/highways/${enc(curId)}/install`);
    toast(`Installed as ${r.installed.split(/[\\/]/).pop()}. In Clone Hero, pick it as your profile's highway.`);
  } catch (e) { toast(e.message, true); }
});
async function setDefault(v) {
  try {
    libDefault = (await post("/api/highways/default", { highway: v })).default;
    renderLists();
    const name = v.startsWith("design:") ? `“${designs.find((d) => `design:${d.id}` === v)?.name}”` : $("#libdefault").selectedOptions[0].text;
    toast(`Songs you chart from now on get ${name}. Existing songs keep theirs.`);
  } catch (e) { toast(e.message, true); }
}
$("#libdefault").addEventListener("change", (e) => setDefault(e.target.value));

// ---------- inputs
for (const el of fields()) el.addEventListener("input", changed);
$("#name").addEventListener("input", () => { cur.name = $("#name").value; dirty = true; toForm(); });
$("#patpick").addEventListener("change", (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  if (f.size > 10 * 1024 * 1024) { toast("That image is too big (10 MB max)", true); return; }
  patFile = f; patRemoved = false;
  if (cur.pattern_opacity < 10) cur.pattern_opacity = 35;
  dirty = true; toForm(); schedule(0);
});
$("#patrm").addEventListener("click", () => { patFile = null; patRemoved = true; dirty = true; toForm(); schedule(0); });
$("#new").addEventListener("click", () => load({ ...blank, name: "My highway" }));
$("#scroll").addEventListener("change", (e) => $("#road").classList.toggle("still", !e.target.checked));
window.addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });

(async () => {
  try {
    await refresh();
    const want = new URLSearchParams(location.search).get("design");
    const d = designs.find((x) => x.id === want) || designs[0];
    if (d) load(d, d.id, d.has_pattern); else load({ ...blank, name: "My highway" });
  } catch (e) { toast(e.message, true); }
})();
