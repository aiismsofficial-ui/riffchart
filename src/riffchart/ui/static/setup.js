"use strict";
// First-run setup / Settings: Clone Hero's settings file, the songs (export) folder, protected folders and
// Clone Hero's song video / background / highway options.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const norm = (p) => String(p || "").replace(/[\\/]+$/, "").replace(/\//g, "\\").toLowerCase();
let S = null, prot = [];                // prot: [{path, on}]

let toastTimer;
function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg; t.classList.toggle("bad", bad); t.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), bad ? 7000 : 4000);
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* not json */ }
  if (!r.ok) throw Object.assign(new Error((body && body.detail) || r.statusText), { status: r.status });
  return body;
}

function render() {
  const ch = S.clone_hero, first = !S.done;
  $("#intro").hidden = !first;
  $("#heading").textContent = first ? "Set up RiffChart" : "Settings";
  $("#back").hidden = first;
  $("#save").textContent = first ? "Save and start" : "Save";
  $("#chstate").innerHTML = ch.found
    ? `✓ Found. ${ch.exe ? "Clone Hero itself is installed too, so songs can open straight in it." : "(Clone Hero.exe wasn't found, so “open in Clone Hero” won't work; everything else does.)"}`
      + (ch.running ? ` <b class="warnc">Clone Hero is running: close it before saving to change its settings.</b>` : "")
    : "✗ Not found here. Install and start Clone Hero once, or paste where its settings.ini is. Without it RiffChart still charts and exports.";
  showReg();
  // protected folders: what's protected now, plus Clone Hero's other song folders (ticked on first run)
  if (!prot.length) {
    const seen = new Set();
    for (const p of S.protected) { prot.push({ path: p, on: true }); seen.add(norm(p)); }
    for (const p of ch.folders) if (!seen.has(norm(p)) && norm(p) !== norm(S.export_dir)) { prot.push({ path: p, on: first, ch: true }); seen.add(norm(p)); }
  }
  renderProt();
  const box = $("#opts");
  box.innerHTML = "";
  for (const [k, label] of Object.entries(ch.labels)) {
    const v = ch.options[k], on = v === null || v === undefined ? true : !!v;
    box.insertAdjacentHTML("beforeend", `<label class="su-check"><input type="checkbox" data-opt="${k}" ${on || first ? "checked" : ""} ${ch.found ? "" : "disabled"}>
      ${esc(label)} <span class="muted">${v === 0 ? "(off now)" : v === 1 ? "(on now)" : "(Clone Hero's default: on)"}</span></label>`);
  }
}
function showReg() {
  const ch = S.clone_hero, ex = norm($("#export").value);
  const already = ch.folders.some((p) => norm(p) === ex);
  $("#regrow").hidden = already || !ch.found;
  $("#regstate").textContent = already ? "✓ Clone Hero already plays songs from this folder." : !ch.found ? "" : "";
}
function renderProt() {
  const box = $("#protlist");
  box.innerHTML = prot.length ? "" : `<p class="muted">None. Add your chart collections so they're never changed.</p>`;
  prot.forEach((p, i) => {
    box.insertAdjacentHTML("beforeend", `<label class="su-check"><input type="checkbox" data-i="${i}" ${p.on ? "checked" : ""}>
      <code>${esc(p.path)}</code>${p.ch ? ` <span class="muted">(one of Clone Hero's song folders)</span>` : ""}</label>`);
  });
  box.querySelectorAll("input").forEach((el) => el.addEventListener("change", () => { prot[+el.dataset.i].on = el.checked; }));
}

$("#export").addEventListener("input", showReg);
$("#addprot").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("#protin").value.trim().replace(/^"|"$/g, "");
  if (!v) return;
  if (!prot.some((p) => norm(p.path) === norm(v))) prot.push({ path: v, on: true });
  $("#protin").value = ""; renderProt();
});
$("#save").addEventListener("click", async () => {
  const options = {};
  document.querySelectorAll("[data-opt]").forEach((el) => { options[el.dataset.opt] = el.checked; });
  const body = { export_dir: $("#export").value.trim().replace(/^"|"$/g, ""),
    clone_hero_settings: $("#settings").value.trim().replace(/^"|"$/g, "") || null,
    protected: prot.filter((p) => p.on).map((p) => p.path),
    register: !$("#regrow").hidden && $("#register").checked, options };
  if (!body.export_dir) { toast("Choose a folder for your finished songs", true); $("#export").focus(); return; }
  if (prot.some((p) => p.on && norm(body.export_dir).startsWith(norm(p.path) + "\\"))) {
    toast("The songs folder can't be inside a folder RiffChart never changes", true); return;
  }
  const first = !S.done;
  $("#save").disabled = true; $("#savestate").textContent = "Saving…";
  try {
    const r = await api("/api/setup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const bits = ["Saved"];
    if (r.registered) bits.push("the songs folder is now in Clone Hero");
    if (r.options_changed) bits.push("Clone Hero's song options are set");
    toast(bits.join(", ") + ".");
    S = r; prot = []; render();
    if (first) setTimeout(() => { location.href = "./"; }, 900);
  } catch (e) {
    toast(e.status === 409 ? "Close Clone Hero first, then Save again (it overwrites its settings when it exits)." : e.message, true);
  } finally { $("#save").disabled = false; $("#savestate").textContent = ""; }
});

(async () => {
  try {
    S = await api("/api/setup");
    $("#settings").value = S.clone_hero.settings;
    $("#export").value = S.done ? S.export_dir : S.default_export;
    render();
  } catch (e) { toast(e.message, true); }
})();
