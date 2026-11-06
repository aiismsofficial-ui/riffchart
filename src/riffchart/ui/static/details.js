// Song details panel (library and editor): title, artist, album, genre, year, guitar difficulty and cover,
// saved into song.ini / album.png, optionally straight into the Clone Hero folder.
// Uses the page's toast(msg, bad). Call openDetails(songId, { onSaved(info) }).
(() => {
  const DIFF_NAMES = ["0 (easiest)", "1", "2", "3", "4", "5", "6 (hardest)"];
  const FIELDS = ["title", "artist", "album", "genre", "year"];
  const enc = encodeURIComponent;
  let dlg, cur = null, onSaved = null, pendingArt = null, removeArt = false, artUrl = null;

  async function call(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
      throw new Error(msg);
    }
    return r.json();
  }

  function build() {
    dlg = document.createElement("dialog");
    dlg.className = "dialog details";
    dlg.innerHTML = `
      <h3>Song details</h3>
      <p class="muted">What Clone Hero and YARG show for this song. Fix anything the app got wrong.</p>
      <div class="dt-body">
        <div class="dt-art">
          <div class="dt-cover" tabindex="0" title="Drop an image here, or click to choose one"><img alt="" hidden><span>No cover</span></div>
          <div class="dt-artbtns"><label class="link">Choose image…<input type="file" accept="image/png,image/jpeg" hidden></label>
            <button type="button" class="linkbtn" data-k="rmart">Remove</button></div>
        </div>
        <div class="dt-fields">
          <label class="wide">Song title<input type="text" data-f="title" maxlength="200" required></label>
          <label class="wide">Artist<input type="text" data-f="artist" maxlength="200"></label>
          <label class="wide">Album<input type="text" data-f="album" maxlength="200"></label>
          <label>Genre<input type="text" data-f="genre" maxlength="100" list="dt-genres"></label>
          <label>Year<input type="text" data-f="year" maxlength="4" inputmode="numeric" pattern="[0-9]{0,4}"></label>
          <label class="wide" title="The difficulty Clone Hero shows in the song list. Estimated from the Expert chart.">
            Guitar difficulty<select data-f="diff_guitar"><option value="">not set</option>
            ${DIFF_NAMES.map((n, i) => `<option value="${i}">${n}</option>`).join("")}</select></label>
        </div>
      </div>
      <datalist id="dt-genres">${["Rock", "Metal", "Alternative", "Punk", "Pop", "Pop Punk", "Emo", "Post-Hardcore",
        "Metalcore", "Hard Rock", "Indie", "Electronic", "Classical", "Video Game", "Country", "Blues", "Jazz", "Funk",
        "Hip Hop", "R&B", "Soundtrack", "Other"].map((g) => `<option value="${g}">`).join("")}</datalist>
      <p class="dt-state muted"></p>
      <div class="dlgbtns"><button type="button" class="ghost" data-k="cancel">Cancel</button>
        <button type="button" class="ghost" data-k="save">Save</button>
        <button type="button" data-k="export">Save &amp; update in Clone Hero</button></div>`;
    document.body.appendChild(dlg);
    // keep the editor's shortcuts (space, R, number keys…) out of the form; Esc still closes the dialog
    dlg.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && e.target.matches("input[type=text]")) { e.preventDefault(); save(false); }
    });
    dlg.querySelector('[data-k="cancel"]').addEventListener("click", () => dlg.close());
    dlg.querySelector('[data-k="save"]').addEventListener("click", () => save(false));
    dlg.querySelector('[data-k="export"]').addEventListener("click", () => save(true));
    dlg.querySelector('[data-k="rmart"]').addEventListener("click", () => { pendingArt = null; removeArt = true; showArt(null); });
    const pick = dlg.querySelector(".dt-art input[type=file]");
    pick.addEventListener("change", () => { if (pick.files[0]) setArt(pick.files[0]); pick.value = ""; });
    const cover = dlg.querySelector(".dt-cover");
    cover.addEventListener("click", () => pick.click());
    cover.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick.click(); } });
    cover.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); cover.classList.add("over"); });
    cover.addEventListener("dragleave", () => cover.classList.remove("over"));
    cover.addEventListener("drop", (e) => {
      e.preventDefault(); e.stopPropagation(); cover.classList.remove("over");
      const f = [...e.dataTransfer.files].find((x) => /^image\/(png|jpeg)$/.test(x.type));
      if (f) setArt(f); else toast("Drop a PNG or JPEG image", true);
    });
    dlg.addEventListener("close", () => { if (artUrl) URL.revokeObjectURL(artUrl); artUrl = null; });
  }

  function setArt(file) {
    if (!/^image\/(png|jpeg)$/.test(file.type)) { toast("The cover must be a PNG or JPEG image", true); return; }
    if (file.size > 10 * 1024 * 1024) { toast("That image is too big (10 MB max)", true); return; }
    pendingArt = file; removeArt = false;
    if (artUrl) URL.revokeObjectURL(artUrl);
    artUrl = URL.createObjectURL(file);
    showArt(artUrl);
  }

  function showArt(src) {
    const img = dlg.querySelector(".dt-cover img"), ph = dlg.querySelector(".dt-cover span");
    img.hidden = !src; ph.hidden = !!src;
    if (src) img.src = src; else img.removeAttribute("src");
    dlg.querySelector('[data-k="rmart"]').hidden = !src;
  }

  function fill(info) {
    cur = info; pendingArt = null; removeArt = false;
    for (const f of FIELDS) dlg.querySelector(`[data-f="${f}"]`).value = info[f] ?? "";
    const d = dlg.querySelector('[data-f="diff_guitar"]');
    d.value = /^[0-6]$/.test(String(info.diff_guitar ?? "")) ? String(info.diff_guitar) : "";
    showArt(info.art ? `/api/songs/${enc(info.id)}/file/${info.art}` : null);
    dlg.querySelector('[data-k="export"]').textContent = info.exported ? "Save & update in Clone Hero" : "Save & export to Clone Hero";
    dlg.querySelector(".dt-state").textContent = !info.exported ? "Not in your Clone Hero folder yet."
      : info.export_stale ? "● The copy in Clone Hero is older than this song: saving with “update” refreshes it."
      : "✓ The copy in Clone Hero is up to date.";
  }

  async function save(exportToo) {
    const body = { export: exportToo };
    for (const f of FIELDS) body[f] = dlg.querySelector(`[data-f="${f}"]`).value.trim();
    body.diff_guitar = dlg.querySelector('[data-f="diff_guitar"]').value;
    if (!body.title) { toast("The song title can't be empty", true); dlg.querySelector('[data-f="title"]').focus(); return; }
    if (body.year && !/^\d{1,4}$/.test(body.year)) { toast("The year should be a number like 1999", true); return; }
    const btns = dlg.querySelectorAll(".dlgbtns button");
    btns.forEach((b) => (b.disabled = true));
    try {
      const base = `/api/songs/${enc(cur.id)}`;
      if (pendingArt) {
        const fd = new FormData(); fd.append("file", pendingArt);
        await call(`${base}/art`, { method: "POST", body: fd });
      } else if (removeArt) {
        await call(`${base}/art`, { method: "DELETE" });
      }
      const info = await call(`${base}/meta`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      dlg.close();
      toast(exportToo ? "Saved and updated in Clone Hero. Rescan songs in Clone Hero to see the change."
        : info.exported ? "Saved. The copy in Clone Hero is now out of date: use “Save & update” or Export again." : "Saved.");
      if (onSaved) onSaved(info);
    } catch (err) {
      toast(err.message, true);
    } finally {
      btns.forEach((b) => (b.disabled = false));
    }
  }

  window.openDetails = async (songId, opts = {}) => {
    if (!dlg) build();
    onSaved = opts.onSaved || null;
    try {
      fill(await call(`/api/songs/${enc(songId)}`));
    } catch (err) { toast(err.message, true); return; }
    dlg.showModal();
    dlg.querySelector('[data-f="title"]').focus();
  };
})();
