// Song details panel (library and editor): title, artist, album, genre, year, guitar difficulty, cover,
// background and music video, saved into the song folder, optionally straight into the Clone Hero folder.
// Uses the page's toast(msg, bad). Call openDetails(songId, { onSaved(info) }).
(() => {
  const DIFF_NAMES = ["0 (easiest)", "1", "2", "3", "4", "5", "6 (hardest)"];
  const FIELDS = ["title", "artist", "album", "genre", "year"];
  const ONLINE_OK = "riffchart.coverSearchOk";
  const enc = encodeURIComponent;
  let dlg, cur = null, onSaved = null;
  // changes wait for Save: cover (file, url or remove), background (mode or own file), video removal
  let pendingArt = null, pendingArtUrl = null, removeArt = false, artUrl = null;
  let bgMode = "auto", bgFile = null, bgUrl = null, removeVideo = false;

  async function call(path, opts = {}) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
      throw new Error(msg);
    }
    return r.json();
  }
  const q = (s) => dlg.querySelector(s);

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
          <button type="button" class="ghost small dt-find" data-k="find">Find cover online…</button>
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
      <div class="dt-online" hidden>
        <div class="dt-onhead"><b>Covers found online</b> <span class="muted dt-onstate"></span>
          <button type="button" class="linkbtn" data-k="closefind">Close</button></div>
        <div class="dt-grid"></div>
      </div>
      <div class="dt-visual">
        <div class="dt-bgprev" title="Shown behind the highway in Clone Hero and YARG"><img alt="" hidden><span>No background</span></div>
        <div class="dt-bgopts">
          <label>Background <select data-k="bgmode">
            <option value="auto">Made from the cover</option><option value="own">My own image…</option><option value="none">None</option>
          </select></label>
          <input type="file" data-k="bgfile" accept="image/png,image/jpeg" hidden>
          <div class="dt-video muted"></div>
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
    q('[data-k="cancel"]').addEventListener("click", () => dlg.close());
    q('[data-k="save"]').addEventListener("click", () => save(false));
    q('[data-k="export"]').addEventListener("click", () => save(true));
    q('[data-k="rmart"]').addEventListener("click", () => { pendingArt = pendingArtUrl = null; removeArt = true; showArt(null); });
    q('[data-k="find"]').addEventListener("click", findOnline);
    q('[data-k="closefind"]').addEventListener("click", () => (q(".dt-online").hidden = true));
    const pick = q(".dt-art input[type=file]");
    pick.addEventListener("change", () => { if (pick.files[0]) setArt(pick.files[0]); pick.value = ""; });
    const cover = q(".dt-cover");
    cover.addEventListener("click", () => pick.click());
    cover.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick.click(); } });
    cover.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); cover.classList.add("over"); });
    cover.addEventListener("dragleave", () => cover.classList.remove("over"));
    cover.addEventListener("drop", (e) => {
      e.preventDefault(); e.stopPropagation(); cover.classList.remove("over");
      const f = [...e.dataTransfer.files].find((x) => /^image\/(png|jpeg)$/.test(x.type));
      if (f) setArt(f); else toast("Drop a PNG or JPEG image", true);
    });
    const mode = q('[data-k="bgmode"]'), bgPick = q('[data-k="bgfile"]');
    mode.addEventListener("change", () => {
      if (mode.value === "own") { bgPick.click(); mode.value = bgMode; return; }   // stays until a file is picked
      bgMode = mode.value; bgFile = null; showBg();
    });
    bgPick.addEventListener("change", () => {
      const f = bgPick.files[0]; bgPick.value = "";
      if (!f) return;
      if (!/^image\/(png|jpeg)$/.test(f.type)) { toast("The background must be a PNG or JPEG image", true); return; }
      bgMode = "own"; bgFile = f; mode.value = "own"; showBg();
    });
    dlg.addEventListener("close", () => {
      for (const u of [artUrl, bgUrl]) if (u) URL.revokeObjectURL(u);
      artUrl = bgUrl = null;
    });
  }

  function setArt(file) {
    if (!/^image\/(png|jpeg)$/.test(file.type)) { toast("The cover must be a PNG or JPEG image", true); return; }
    if (file.size > 10 * 1024 * 1024) { toast("That image is too big (10 MB max)", true); return; }
    pendingArt = file; pendingArtUrl = null; removeArt = false;
    if (artUrl) URL.revokeObjectURL(artUrl);
    artUrl = URL.createObjectURL(file);
    showArt(artUrl);
  }

  function coverSrc() {          // what the cover will be after Save (for the background preview)
    if (removeArt) return null;
    if (pendingArt) return artUrl;
    if (pendingArtUrl) return pendingArtUrl.thumb;
    return cur.art ? `/api/songs/${enc(cur.id)}/file/${cur.art}` : null;
  }

  function showArt(src) {
    const img = q(".dt-cover img"), ph = q(".dt-cover span");
    img.hidden = !src; ph.hidden = !!src;
    if (src) img.src = src; else img.removeAttribute("src");
    q('[data-k="rmart"]').hidden = !src;
    showBg();
  }

  function showBg() {
    const box = q(".dt-bgprev"), img = box.querySelector("img"), ph = box.querySelector("span");
    let src = null, blur = false;
    if (bgMode === "own") {
      if (bgFile) { if (bgUrl) URL.revokeObjectURL(bgUrl); bgUrl = URL.createObjectURL(bgFile); src = bgUrl; }
      else if (cur.background.file) src = `/api/songs/${enc(cur.id)}/file/${cur.background.file}`;
    } else if (bgMode === "auto") {
      const unchangedCover = !removeArt && !pendingArt && !pendingArtUrl;
      src = unchangedCover && cur.background.mode === "auto" && cur.background.file
        ? `/api/songs/${enc(cur.id)}/file/${cur.background.file}` : coverSrc();
      blur = src && !(unchangedCover && cur.background.mode === "auto" && cur.background.file);
    }
    box.classList.toggle("blur", !!blur);           // preview of a background that Save will make
    img.hidden = !src; ph.hidden = !!src;
    ph.textContent = bgMode === "auto" && !src ? "Add a cover to make one" : "No background";
    if (src) img.src = src; else img.removeAttribute("src");
  }

  function showVideo() {
    const v = cur.video, el = q(".dt-video");
    if (!v) { el.innerHTML = "No music video. Drop a music video file (mp4, webm) to chart it with one."; return; }
    el.innerHTML = removeVideo ? `Music video will be removed. <button type="button" class="linkbtn" data-k="keepvid">Keep it</button>`
      : `Music video: ${v.file} (${v.mb} MB) plays behind the highway. <button type="button" class="linkbtn" data-k="rmvid">Remove</button>`;
    const b = el.querySelector("button");
    b.addEventListener("click", () => { removeVideo = !removeVideo; showVideo(); });
  }

  async function findOnline() {
    if (!localStorage.getItem(ONLINE_OK)) {
      const ok = confirm("Find covers online?\n\nRiffChart sends this song's artist, title and album (never the audio) to " +
        "Apple's iTunes Search and MusicBrainz, and shows the covers they have. Covers are copyrighted: fine for your " +
        "own game, but think twice before sharing song folders.\n\nAllow this from now on?");
      if (!ok) return;
      localStorage.setItem(ONLINE_OK, "1");
    }
    const box = q(".dt-online"), grid = q(".dt-grid"), state = q(".dt-onstate");
    box.hidden = false; grid.innerHTML = ""; state.textContent = "searching…";
    const v = (f) => encodeURIComponent(q(`[data-f="${f}"]`).value.trim());
    try {
      const r = await call(`/api/covers/search?artist=${v("artist")}&title=${v("title")}&album=${v("album")}`);
      state.textContent = r.results.length ? `${r.results.length} found · click one to use it` : "nothing found";
      if (r.errors.length) state.textContent += ` (${r.errors.join("; ")})`;
      for (const c of r.results) {
        const b = document.createElement("button");
        b.type = "button"; b.className = "dt-cand"; b.title = `${c.label}${c.year ? ` (${c.year})` : ""} · ${c.source}`;
        b.innerHTML = `<img alt="" loading="lazy"><span>${c.source === "Apple Music" ? "Apple" : "MB"}</span>`;
        const img = b.querySelector("img");
        img.addEventListener("error", () => b.remove());       // no cover for that release
        img.src = c.thumb;
        b.addEventListener("click", () => {
          grid.querySelectorAll(".dt-cand").forEach((x) => x.classList.toggle("on", x === b));
          pendingArtUrl = c; pendingArt = null; removeArt = false;
          showArt(c.thumb);
        });
        grid.appendChild(b);
      }
    } catch (err) { state.textContent = ""; toast(`Cover search failed: ${err.message}`, true); }
  }

  function fill(info) {
    cur = info; pendingArt = pendingArtUrl = bgFile = null; removeArt = removeVideo = false;
    bgMode = info.background.mode;
    for (const f of FIELDS) q(`[data-f="${f}"]`).value = info[f] ?? "";
    const d = q('[data-f="diff_guitar"]');
    d.value = /^[0-6]$/.test(String(info.diff_guitar ?? "")) ? String(info.diff_guitar) : "";
    q('[data-k="bgmode"]').value = bgMode;
    q(".dt-online").hidden = true;
    showArt(info.art ? `/api/songs/${enc(info.id)}/file/${info.art}` : null);
    showVideo();
    q('[data-k="export"]').textContent = info.exported ? "Save & update in Clone Hero" : "Save & export to Clone Hero";
    q(".dt-state").textContent = !info.exported ? "Not in your Clone Hero folder yet."
      : info.export_stale ? "● The copy in Clone Hero is older than this song: saving with “update” refreshes it."
      : "✓ The copy in Clone Hero is up to date.";
  }

  async function save(exportToo) {
    const body = { export: exportToo };
    for (const f of FIELDS) body[f] = q(`[data-f="${f}"]`).value.trim();
    body.diff_guitar = q('[data-f="diff_guitar"]').value;
    if (!body.title) { toast("The song title can't be empty", true); q('[data-f="title"]').focus(); return; }
    if (body.year && !/^\d{1,4}$/.test(body.year)) { toast("The year should be a number like 1999", true); return; }
    const btns = dlg.querySelectorAll(".dlgbtns button");
    btns.forEach((b) => (b.disabled = true));
    const json = (o) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(o) });
    try {
      const base = `/api/songs/${enc(cur.id)}`;
      if (bgMode !== cur.background.mode && bgMode !== "own") await call(`${base}/background`, json({ mode: bgMode }));
      if (bgFile) {
        const fd = new FormData(); fd.append("file", bgFile);
        await call(`${base}/background/file`, { method: "POST", body: fd });
      }
      if (pendingArt) {
        const fd = new FormData(); fd.append("file", pendingArt);
        await call(`${base}/art`, { method: "POST", body: fd });
      } else if (pendingArtUrl) {
        await call(`${base}/art/url`, json({ url: pendingArtUrl.full }));
      } else if (removeArt) {
        await call(`${base}/art`, { method: "DELETE" });
      }
      if (removeVideo) await call(`${base}/video`, { method: "DELETE" });
      const info = await call(`${base}/meta`, json(body));
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
    q('[data-f="title"]').focus();
  };
})();
