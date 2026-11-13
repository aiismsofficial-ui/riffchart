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
  { name: "Gold", top: "#1c1608", bottom: "#0c0a04", rail: "#ffd24a", divider: "#3d331a", glow: 65, rail_width: 18,
    rail_style: "chrome" },
  // animated
  { name: "Neon sparks", top: "#0b1026", bottom: "#05060f", rail: "#22e4ff", divider: "#1d2a4a", glow: 85,
    rail_style: "neon", anim: "sparks", anim_color: "#7df3ff", anim_strength: 75, anim_speed: 1 },
  { name: "Synthwave grid", top: "#2a0b3d", bottom: "#0d0418", rail: "#ff3ec8", divider: "#4a1d5e", glow: 80,
    rail_style: "neon", anim: "grid", anim_color: "#22e4ff", anim_strength: 70, anim_speed: 2, vignette: 40 },
  { name: "Storm", top: "#121826", bottom: "#070a10", rail: "#9fc6ff", divider: "#24324a", glow: 55,
    anim: "lightning", anim_color: "#cfe6ff", anim_strength: 80, anim_seconds: 6 },
  { name: "Inferno", top: "#1e0a06", bottom: "#0b0403", rail: "#ff5a1f", divider: "#3e1a12", glow: 70,
    anim: "fire", anim_color: "#ff6a00", anim_strength: 75, anim_speed: 2, center_glow: 15 },
  { name: "Aurora", top: "#06141a", bottom: "#03080b", rail: "#4dffb8", divider: "#123a35", glow: 50,
    anim: "aurora", anim_color: "#38ff9c", anim_strength: 70, anim_seconds: 8 },
  { name: "Warp drive", top: "#0a0a14", bottom: "#040408", rail: "#b48cff", divider: "#262040", glow: 60,
    anim: "warp", anim_color: "#d9c8ff", anim_strength: 80, anim_speed: 2, fret_tint: 25 },
  { name: "Fret lanes", top: "#141418", bottom: "#08080b", rail: "#e8e8e8", divider: "#3a3a44", glow: 30,
    fret_tint: 70, lane_style: "glow", anim: "sheen", anim_color: "#ffffff", anim_strength: 45 },
  { name: "Rainbow pulse", top: "#101018", bottom: "#06060a", rail: "#ff3e3e", divider: "#2a2a36", glow: 90,
    rail_style: "neon", anim: "hue", anim_strength: 60, anim_seconds: 6 },
];

let blank = {}, designs = [], libDefault = "riffchart", chOk = false;
// what's on the page: the design, which saved preset it is (id), and its pattern image
let cur = {}, curId = null, patFile = null, patSaved = false, patRemoved = false, dirty = false;
let renderTimer = null, renderSeq = 0, flatUrl = null, vidUrl = null, videoTimer = null, videoSeq = 0;
const UNITS = { rail_width: "px", lane_width: "px", text_size: "px", anim_speed: "×", anim_seconds: " s" };

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
    if (out) out.textContent = el.dataset.k === "anim_speed" ? `${v}×` : `${v}${UNITS[el.dataset.k] || "%"}`;
  }
  const hasPat = !!patFile || (patSaved && !patRemoved);
  $("#patname").textContent = patFile ? patFile.name : hasPat ? "saved image" : "none";
  $("#patrm").hidden = !hasPat;
  $("#delete").hidden = !curId;
  for (const b of ["#applyall", "#makedefault", "#install"]) $(b).disabled = false;
  $("#install").hidden = !chOk;
  $(".hw-animopts").hidden = !cur.anim || cur.anim === "none";
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
function schedule(ms = 140) {
  clearTimeout(renderTimer); renderTimer = setTimeout(render, ms);
  clearTimeout(videoTimer); videoTimer = setTimeout(renderVideo, Math.max(ms, 450));
}
// animated designs: a small preview video, drawn repeating down the 3D highway on a canvas
async function renderVideo() {
  const seq = ++videoSeq, vid = $("#flatvid");
  if (!cur.anim || cur.anim === "none") {
    vid.hidden = true; vid.removeAttribute("src"); $("#flat").hidden = false; $("#animstate").textContent = "";
    showRoad(null); return;
  }
  $("#animstate").textContent = "Rendering the animation…";
  try {
    const r = await fetch("/api/highways/render_video", { method: "POST", body: formData(false) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const blob = await r.blob();
    if (seq !== videoSeq) return;
    if (vidUrl) URL.revokeObjectURL(vidUrl);
    vidUrl = URL.createObjectURL(blob);
    vid.src = vidUrl; vid.hidden = false; $("#flat").hidden = true;
    await vid.play().catch(() => {});
    $("#animstate").textContent = `${cur.anim_seconds} s loop. Clone Hero gets it at full size (512 × 1024, 30 fps).`;
    showRoad(vid);
  } catch (e) { if (seq === videoSeq) $("#animstate").textContent = `Animation preview failed: ${e.message}`; }
}
let roadSrc = null, roadRaf = 0;
function showRoad(video) {
  const road = $("#road");
  let cv = road.querySelector("canvas");
  roadSrc = video;
  cancelAnimationFrame(roadRaf);
  if (!video) { if (cv) cv.hidden = true; return; }
  if (!cv) { cv = document.createElement("canvas"); cv.width = 240; cv.height = 900; road.appendChild(cv); }
  cv.hidden = false;
  const g = cv.getContext("2d"), t0 = performance.now();
  const draw = (now) => {
    if (roadSrc !== video) return;
    const tile = 480, off = $("#scroll").checked ? ((now - t0) / 2400 * tile) % tile : 0;
    if (video.readyState >= 2) for (let y = off - tile; y < cv.height; y += tile) g.drawImage(video, 0, y, 240, tile);
    roadRaf = requestAnimationFrame(draw);
  };
  roadRaf = requestAnimationFrame(draw);
}
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
    const moving = d.anim && d.anim !== "none" ? ` <em class="muted">· animated</em>` : "";
    box.appendChild(thumb(`background-image:url('/api/highways/${enc(d.id)}/image?v=${d.mtime}')`, d.name, d.id,
      () => load(d, d.id, d.has_pattern), moving + tag));
  }
  const st = $("#starters");
  st.innerHTML = "";
  for (const s of STARTERS) st.appendChild(thumb(starterStyle(s), s.name, "", () => load({ ...s, name: `My ${s.name.replace(/^RiffChart /, "")}` }),
    s.anim ? ` <em class="muted">· animated</em>` : ""));
  const sel = $("#libdefault");
  sel.innerHTML = BUILTIN.map(([v, n]) => `<option value="${v}">${n}</option>`).join("")
    + (designs.length ? `<optgroup label="Your designs">${designs.map((d) => `<option value="design:${esc(d.id)}">${esc(d.name)}</option>`).join("")}</optgroup>` : "");
  sel.value = libDefault;
  toForm();
}
async function refresh() {
  const r = await api("/api/highways");
  blank = r.blank; designs = r.designs; libDefault = r.default; chOk = r.clone_hero;
  const sel = $("#anim");
  if (!sel.options.length) sel.innerHTML = Object.entries(r.effects).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
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
    const parts = r.installed.split(/[\\/]/);
    toast(/highway\.webm$/i.test(r.installed)
      ? `Installed in Clone Hero's Video Highways as “${parts[parts.length - 2]}”. Pick it as your profile's highway there.`
      : `Installed as ${parts.pop()}. In Clone Hero, pick it as your profile's highway.`);
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

// ---------- animated backgrounds
let bgSel = null, bgUrl = null;
async function initBackgrounds() {
  const r = await api("/api/backgrounds");
  const box = $("#bgpresets");
  for (const p of r.presets) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = p.name; b.dataset.id = p.id;
    b.addEventListener("click", () => pickBg(p, b));
    box.appendChild(b);
  }
  $("#bginstall").hidden = !r.clone_hero;
}
async function pickBg(p, btn) {
  bgSel = p;
  document.querySelectorAll("#bgpresets button").forEach((x) => x.classList.toggle("on", x === btn));
  $("#bgstate").textContent = "Rendering a preview…";
  $("#bgall").disabled = $("#bginstall").disabled = false;
  try {
    const r = await fetch("/api/backgrounds/preview", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset: p.id }) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    if (bgUrl) URL.revokeObjectURL(bgUrl);
    bgUrl = URL.createObjectURL(await r.blob());
    if (bgSel !== p) return;
    $("#bgvid").src = bgUrl; $("#bgvid").play().catch(() => {});
    $("#bgstate").textContent = `${p.name}: an 8-second loop (1280 × 720 in Clone Hero). On a song it plays over that song's own background.`;
  } catch (e) { $("#bgstate").textContent = `Preview failed: ${e.message}`; }
}
$("#bgall").addEventListener("click", async () => {
  if (!bgSel || !confirm(`Give every song without a music video the animated background “${bgSel.name}”?

` +
    "Each takes about 15 seconds to make (they're made one after another in the background).")) return;
  try {
    const r = await post("/api/backgrounds/apply", { preset: bgSel.id });
    toast(`Making “${bgSel.name}” for ${r.started.length} songs${r.skipped_video.length ? ` (${r.skipped_video.length} with a music video left alone)` : ""}. ` +
      "Then Save & update in Clone Hero, or Export, to copy them there.");
  } catch (e) { toast(e.message, true); }
});
$("#bginstall").addEventListener("click", async () => {
  if (!bgSel) return;
  $("#bginstall").disabled = true;
  try {
    const r = await post(`/api/backgrounds/${enc(bgSel.id)}/install`);
    toast(`Installed as ${r.installed.split(/[\\/]/).pop()} in Clone Hero's Video Backgrounds. Pick it in Clone Hero's settings.`);
  } catch (e) { toast(e.message, true); }
  finally { $("#bginstall").disabled = false; }
});

(async () => {
  try {
    initBackgrounds().catch((e) => toast(e.message, true));
    await refresh();
    const want = new URLSearchParams(location.search).get("design");
    const d = designs.find((x) => x.id === want) || designs[0];
    if (d) load(d, d.id, d.has_pattern); else load({ ...blank, name: "My highway" });
  } catch (e) { toast(e.message, true); }
})();
