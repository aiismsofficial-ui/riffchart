// Game view (G): the chart played back on a tilted highway, the way Clone Hero shows it. Read-only.
// Our own drawing (not Clone Hero's art): the song's highway.png and background, the editor's Note speed, gems,
// HOPO / tap / open / star power looks, sustains, beat lines and the strike line.
//
// Perspective fitted to Clone Hero recordings (2026-11-27): a point s highway-lengths ahead of the strike line
// (s = seconds ahead / seconds visible) is drawn at scale k = D / (s + D), D = 0.69, between a horizon at 13.5 %
// of the height and the strike line at 91 %; the far end (s = 1) then sits at 45 % and is 0.41 as wide.

const GV = {
  canvas: null, g: null, hwImg: null, bgImg: null, songId: null, flash: new Float64Array(6),
  D: 0.69, HORIZON: 0.135, STRIKE: 0.91, NEAR_S: -0.075,
  TILE: 0.3,                     // highway width / length, for how often the highway picture repeats
};

function gameInit() {
  const cv = document.createElement("canvas");
  cv.id = "game"; cv.hidden = true; cv.setAttribute("aria-label", "Game view: the chart on a tilted highway");
  $("#stage").insertBefore(cv, $("#ov"));
  GV.canvas = cv; GV.g = cv.getContext("2d");
  cv.addEventListener("wheel", (e) => {                   // scroll the song like the flat view; no zoom here
    e.preventDefault();
    if (!S.player) return;
    if (e.ctrlKey || e.metaKey) { setZoom(S.ns * Math.exp(-e.deltaY * 0.0015)); return; }
    const dy = e.deltaMode === 1 ? e.deltaY * 40 : e.deltaY;
    seekBy(-dy / 400 * nsSeconds(S.ns));
  }, { passive: false });
  cv.addEventListener("click", () => { if (S.player) togglePlay(); });
}

function gameImages() {
  const id = new URLSearchParams(location.search).get("song");
  if (!id || GV.songId === id) return;
  GV.songId = id;
  const load = (names, key) => {
    const [n, ...rest] = names;
    if (!n) return;
    const img = new Image();
    img.onload = () => { GV[key] = img; };
    img.onerror = () => load(rest, key);
    img.src = `/api/songs/${encodeURIComponent(id)}/file/${n}?v=${Date.now()}`;
  };
  load(["highway.png"], "hwImg");
  load(["background.jpg", "background.png"], "bgImg");
}

function setGame(on) {
  if (!GV.canvas) gameInit();
  S.game = !!on;
  GV.canvas.hidden = !S.game; hw.style.visibility = S.game ? "hidden" : "";
  $("#gamebtn").classList.toggle("on", S.game);
  $("#tip").hidden = true;
  if (S.game) { gameImages(); toast("Game view: what Clone Hero shows at this note speed. Read-only; G to edit again"); }
}

// ---------------------------------------------------------------- geometry
function gameGeom() {
  const cw = GV.canvas.clientWidth, ch = GV.canvas.clientHeight;
  const yh = ch * GV.HORIZON, ys = ch * GV.STRIKE;
  const half = Math.min(ch * 0.34, cw * 0.44);            // half the highway's width at the strike line
  return { cw, ch, yh, ys, half, cx: cw / 2 };
}
const gk = (s) => GV.D / (s + GV.D);
const gy = (G, s) => G.yh + (G.ys - G.yh) * gk(s);
const gx = (G, s, u) => G.cx + u * G.half * gk(s);        // u: -1 (left edge) .. 1 (right edge)
const laneU = (f) => -0.8 + f * 0.4;                       // lane centres

// ---------------------------------------------------------------- frame
function gameFrame(now) {
  const cv = GV.canvas, g = GV.g;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const G = gameGeom(), vis = nsSeconds(S.ns);
  // background: the song's picture, darkened, else plain
  g.fillStyle = "#0b0d12"; g.fillRect(0, 0, w, h);
  if (GV.bgImg) {
    const im = GV.bgImg, sc = Math.max(w / im.width, h / im.height);
    g.globalAlpha = 0.55; g.drawImage(im, (w - im.width * sc) / 2, (h - im.height * sc) / 2, im.width * sc, im.height * sc);
    g.globalAlpha = 1;
  }
  if (!S.chart) return;
  const tr = S.tracks[S.diff] || S.tracks[DIFFS.find((d) => S.tracks[d])];
  const sOf = (t) => (t - now) / vis;
  const sFar = 1, yTop = gy(G, sFar);

  // highway surface, row by row (texture fixed to the song's time so it moves with the notes)
  const im = GV.hwImg, rows = 2, tileS = im ? (im.height / im.width) * GV.TILE * 2 : 1;
  for (let y = Math.floor(yTop); y < h; y += rows) {
    const k = (y + rows / 2 - G.yh) / (G.ys - G.yh), s = GV.D / k - GV.D;
    if (s < GV.NEAR_S - 0.2) break;
    const half = G.half * k;
    const fade = Math.min(1, (sFar - s) / 0.12);          // the far end fades in
    g.globalAlpha = Math.max(0, fade);
    if (im) {
      let v = (((now + s * vis) / vis) / tileS) % 1; if (v < 0) v += 1;
      const sy = (1 - v) * im.height;
      const k2 = (y + rows * 1.5 - G.yh) / (G.ys - G.yh), ds = Math.abs(s - (GV.D / k2 - GV.D));
      const sh = Math.max(1, Math.min(im.height / 4, ds / tileS * im.height));
      g.drawImage(im, 0, Math.max(0, Math.min(im.height - sh, sy - sh)), im.width, sh, G.cx - half, y, half * 2, rows + 0.5);
    } else {
      g.fillStyle = "#1b1f27"; g.fillRect(G.cx - half, y, half * 2, rows + 0.5);
    }
  }
  g.globalAlpha = 1;
  // lane lines and rails
  const sNear = GV.NEAR_S - 0.2;
  const line = (u, wdt, col, a) => {
    g.strokeStyle = col; g.globalAlpha = a; g.lineWidth = wdt;
    g.beginPath(); g.moveTo(gx(G, sFar, u), yTop); g.lineTo(gx(G, sNear, u), gy(G, sNear)); g.stroke();
  };
  for (let i = 1; i < 5; i++) line(-1 + i * 0.4, 1.2, "#ffffff", 0.16);
  line(-1, 4, "#c9ced8", 0.85); line(1, 4, "#c9ced8", 0.85);
  g.globalAlpha = 1;
  // beat lines
  const off = S.chart.offset_s, tA = now + GV.NEAR_S * vis, tB = now + vis;
  const lines = S.tempo.lines(Math.max(0, S.tempo.secToTick(tA - off)), S.tempo.secToTick(tB - off), 1);
  for (const ln of lines) {
    const s = sOf(S.tempo.tickToSec(ln.tick) + off);
    if (s < GV.NEAR_S || s > sFar) continue;
    const y = gy(G, s), x0 = gx(G, s, -1), x1 = gx(G, s, 1), k = gk(s);
    g.fillStyle = "#ffffff"; g.globalAlpha = (ln.kind === 2 ? 0.55 : 0.22) * Math.min(1, (sFar - s) / 0.12);
    const th = (ln.kind === 2 ? 3 : 1.5) * k;
    g.fillRect(x0, y - th / 2, x1 - x0, th);
  }
  g.globalAlpha = 1;
  // strike line: a bar and the five fret rings
  const ky = gk(0), laneW = G.half * 0.4 * ky;
  g.fillStyle = "rgba(255,255,255,.18)"; g.fillRect(gx(G, 0, -1), G.ys - 2, G.half * 2 * ky, 4);
  for (let f = 0; f < 5; f++) gameRing(g, gx(G, 0, laneU(f)), G.ys, laneW * 0.44, FRET_COLORS[f]);
  if (!tr) return;

  // sustains (under the gems): from the gem (or the strike line while held) to the end
  const i0 = Math.max(0, bsearch(tr.on, now - 30)), i1 = bsearch(tr.on, now + vis + 0.05);
  for (let i = i0; i < i1; i++) {
    if (tr.off[i] <= tr.on[i] + 1e-6 || tr.off[i] < now) continue;
    const m = tr.notes[i][1], sA = Math.max(sOf(tr.on[i]), tr.on[i] <= now ? 0 : -1), sB = Math.min(sFar, sOf(tr.off[i]));
    if (sB <= sA) continue;
    const held = tr.on[i] <= now;
    const lanes = m & 32 ? [-1] : [0, 1, 2, 3, 4].filter((f) => m & (1 << f));
    for (const f of lanes) {
      const u = f < 0 ? 0 : laneU(f), hw2 = f < 0 ? 0.9 : 0.055;
      const col = tr.inSp[i] ? SP_COLOR : f < 0 ? OPEN_COLOR : FRET_COLORS[f];
      g.fillStyle = col; g.globalAlpha = f < 0 ? 0.28 : held ? 0.95 : 0.8;
      g.beginPath();
      g.moveTo(gx(G, sA, u - hw2), gy(G, sA)); g.lineTo(gx(G, sB, u - hw2), gy(G, sB));
      g.lineTo(gx(G, sB, u + hw2), gy(G, sB)); g.lineTo(gx(G, sA, u + hw2), gy(G, sA)); g.closePath(); g.fill();
    }
  }
  g.globalAlpha = 1;
  // gems, far to near; a gem is gone once it reaches the strike line (the "bot" hits it)
  for (let i = i1 - 1; i >= i0; i--) {
    const s = sOf(tr.on[i]);
    if (s > sFar || tr.on[i] < now - 0.005) continue;
    const [, m, , nt] = tr.notes[i], k = gk(s), y = gy(G, s), a = Math.min(1, (sFar - s) / 0.1);
    g.globalAlpha = a;
    if (m & 32) gameOpen(g, G, s, y, k, nt, tr.inSp[i]);
    else for (let f = 0; f < 5; f++) if (m & (1 << f)) gameGem(g, gx(G, s, laneU(f)), y, G.half * 0.4 * k * 0.44, f, nt, tr.inSp[i]);
  }
  g.globalAlpha = 1;
  // hit flashes on the strike line
  if (S.player.playing) {
    for (let i = Math.max(0, bsearch(tr.on, now - 0.15)); i < tr.on.length && tr.on[i] <= now; i++) {
      const m = tr.notes[i][1], age = (now - tr.on[i]) / 0.15;
      const lanes = m & 32 ? [0, 1, 2, 3, 4] : [0, 1, 2, 3, 4].filter((f) => m & (1 << f));
      for (const f of lanes) gameFlash(g, gx(G, 0, laneU(f)), G.ys, laneW, age, m & 32 ? OPEN_COLOR : FRET_COLORS[f]);
    }
    for (let i = Math.max(0, bsearch(tr.on, now - 30)); i < tr.on.length && tr.on[i] <= now; i++) {
      if (tr.off[i] <= now || tr.off[i] <= tr.on[i] + 1e-6) continue;  // held sustains keep a small glow
      const m = tr.notes[i][1];
      for (let f = 0; f < 5; f++) if (m & (1 << f)) gameFlash(g, gx(G, 0, laneU(f)), G.ys, laneW * 0.7, 0.5, FRET_COLORS[f]);
    }
  }
  // corner label
  g.font = "600 12px system-ui, sans-serif"; g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.fillStyle = "rgba(255,255,255,.7)";
  g.fillText(`Game view · Note speed ${$("#nsval").textContent} · ${S.diff[0].toUpperCase()}${S.diff.slice(1)} · G to edit`, 12, h - 12);
}

// ---------------------------------------------------------------- pieces (our own 5-fret look)
function gameRing(g, x, y, r, col) {
  g.globalAlpha = 1;
  g.fillStyle = "#111318"; g.beginPath(); g.ellipse(x, y + r * 0.12, r * 1.08, r * 0.5, 0, 0, Math.PI * 2); g.fill();
  g.lineWidth = r * 0.22; g.strokeStyle = col;
  g.beginPath(); g.ellipse(x, y, r, r * 0.45, 0, 0, Math.PI * 2); g.stroke();
  g.fillStyle = "rgba(255,255,255,.12)"; g.beginPath(); g.ellipse(x, y - r * 0.05, r * 0.55, r * 0.22, 0, 0, Math.PI * 2); g.fill();
}
function gameGem(g, x, y, r, f, nt, sp) {
  const col = sp ? SP_COLOR : FRET_COLORS[f];
  if (nt === 1) r *= 0.84;                                  // HOPO: smaller
  const th = r * 0.32;                                      // puck thickness
  // body (side), base ellipse, then the top
  g.fillStyle = shade(col, -0.45);
  g.beginPath(); g.ellipse(x, y + th, r, r * 0.45, 0, 0, Math.PI); g.lineTo(x - r, y); g.ellipse(x, y, r, r * 0.45, 0, Math.PI, 0, true); g.closePath(); g.fill();
  g.fillStyle = nt === 2 ? "#15171c" : col;                 // tap: dark top
  g.beginPath(); g.ellipse(x, y, r, r * 0.45, 0, 0, Math.PI * 2); g.fill();
  if (nt === 2) { g.lineWidth = r * 0.2; g.strokeStyle = col; g.stroke(); }
  // cap: strum = small white centre on the colour; HOPO = big white top with a colour ring
  g.fillStyle = nt === 2 ? shade(col, 0.2) : "#ffffff";
  const cr = nt === 1 ? 0.72 : nt === 2 ? 0.34 : 0.42;
  g.beginPath(); g.ellipse(x, y - r * 0.06, r * cr, r * cr * 0.45, 0, 0, Math.PI * 2); g.fill();
  if (sp) { g.globalAlpha *= 0.5; g.strokeStyle = "#ffffff"; g.lineWidth = 2; g.beginPath(); g.ellipse(x, y, r * 1.12, r * 0.52, 0, 0, Math.PI * 2); g.stroke(); g.globalAlpha = Math.min(1, g.globalAlpha * 2); }
}
function gameOpen(g, G, s, y, k, nt, sp) {
  const x0 = gx(G, s, -0.94), x1 = gx(G, s, 0.94), hh = G.half * 0.07 * k;
  g.fillStyle = shade(sp ? SP_COLOR : OPEN_COLOR, -0.4);
  g.beginPath(); g.roundRect ? g.roundRect(x0, y - hh / 2 + hh * 0.5, x1 - x0, hh, hh / 2) : g.rect(x0, y, x1 - x0, hh); g.fill();
  g.fillStyle = sp ? SP_COLOR : OPEN_COLOR;
  g.beginPath(); g.roundRect ? g.roundRect(x0, y - hh / 2, x1 - x0, hh, hh / 2) : g.rect(x0, y - hh / 2, x1 - x0, hh); g.fill();
  if (nt === 1) { g.fillStyle = "rgba(255,255,255,.85)"; g.fillRect(x0 + hh, y - hh * 0.12, x1 - x0 - hh * 2, hh * 0.24); }
}
function gameFlash(g, x, y, r, age, col) {
  const a = Math.max(0, 1 - age), rad = r * (0.6 + age * 0.6);
  const gr = g.createRadialGradient(x, y, 0, x, y, rad);
  gr.addColorStop(0, `rgba(255,255,255,${0.9 * a})`); gr.addColorStop(0.4, hexA(col, 0.6 * a)); gr.addColorStop(1, hexA(col, 0));
  g.fillStyle = gr; g.globalAlpha = 1;
  g.beginPath(); g.ellipse(x, y - r * 0.15, rad, rad * 0.75, 0, 0, Math.PI * 2); g.fill();
}
// shade() and hexA() are in editor.js

$("#gamebtn").addEventListener("click", () => setGame(!S.game));
window.addEventListener("keydown", (e) => {
  if (e.target.closest && e.target.closest("input, select, textarea") || e.ctrlKey || e.metaKey || e.altKey || !S.chart) return;
  if (e.key === "g" || e.key === "G") { e.preventDefault(); setGame(!S.game); }
  else if (e.key === "Escape" && S.game) setGame(false);
});
