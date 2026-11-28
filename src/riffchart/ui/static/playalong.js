// Play along (in the Game view): play the chart yourself with the keyboard or a guitar controller.
// Keyboard: frets 1-5, strum Up / Down (Moonscraper's default keys). Controller: the browser's Gamepad API
// (Xbox-style guitar: A green, B red, Y yellow, X blue, LB orange; strum on the D-pad or the stick).
// Rules close to Clone Hero's: a +-70 ms window; chords need exactly their frets, single notes allow lower frets
// held (anchoring); HOPOs need no strum while the streak is going, taps never do; an open note is a strum with no
// frets. A miss or a strum with nothing to hit breaks the streak and mutes the part's track until the next hit.

const PA = { on: false, keys: [false, false, false, false, false], pad: [false, false, false, false, false],
  padStrum: false, hit: new Set(), miss: new Set(), next: 0, streak: 0, best: 0, score: 0, hits: 0, misses: 0,
  overstrums: 0, lastHit: -1, lastNow: 0, flashes: [], muted: false, from: 0 };
const PA_WINDOW = 0.07;
const PA_KEYS = { Digit1: 0, Digit2: 1, Digit3: 2, Digit4: 3, Digit5: 4, Numpad1: 0, Numpad2: 1, Numpad3: 2, Numpad4: 3, Numpad5: 4 };
const PA_STRUM = new Set(["ArrowUp", "ArrowDown"]);

const paTrack = () => S.tracks[S.diff] || S.tracks[DIFFS.find((d) => S.tracks[d])];
const paHeld = () => PA.keys.reduce((m, v, f) => m | ((v || PA.pad[f]) ? 1 << f : 0), 0);
const paMult = () => Math.min(4, 1 + Math.floor(PA.streak / 10));

function paStart() {
  if (!S.chart || !paTrack()) return;
  if (!S.game) setGame(true);
  Object.assign(PA, { on: true, hit: new Set(), miss: new Set(), streak: 0, best: 0, score: 0, hits: 0, misses: 0,
    overstrums: 0, lastHit: -1, heldAtHit: -1, flashes: [], muted: false });
  const t = Math.max(0, S.player.time() - 2);            // two seconds to get ready
  S.player.seek(t);
  PA.from = t; PA.lastNow = t; paResync(t);
  $("#playbtn").setAttribute("aria-pressed", "true"); $("#gamebtn").setAttribute("aria-pressed", "false");
  if (!S.player.playing) togglePlay();
  toast("Play along: frets 1-5, strum Up / Down (or a guitar controller). Space pauses, Esc stops");
}
function paStop() {
  if (!PA.on) return;
  PA.on = false;
  if (S.player.playing) togglePlay();
  paUnmute();
  $("#playbtn").setAttribute("aria-pressed", "false"); $("#gamebtn").setAttribute("aria-pressed", String(!!S.game));
}
function paResync(now) {                                  // after a seek: start judging from here
  const tr = paTrack();
  PA.next = tr ? bsearch(tr.on, now - PA_WINDOW) : 0;
}
function paMute() {
  if (PA.muted || !S.player) return;
  PA.muted = true;
  S.player.names.forEach((n, i) => { if (n.toLowerCase().startsWith(PART)) S.player.setVolume(i, 0.12); });
}
function paUnmute() { if (PA.muted) { PA.muted = false; applyVolumes(); } }

function paMatches(mask, held) {
  if (mask & 32) return held === 0;                       // open: no frets
  const chord = (mask & (mask - 1)) !== 0;
  if (chord) return held === mask;
  return (held & mask) !== 0 && (held & ~((mask << 1) - 1)) === 0;   // the note's fret, plus any lower ones
}
function paHit(i, now) {
  const tr = paTrack(), m = tr.notes[i][1];
  PA.hit.add(i); PA.hits++; PA.streak++; PA.best = Math.max(PA.best, PA.streak);
  PA.score += 50 * paMult() * Math.max(1, m & 31 ? [0, 1, 2, 3, 4].filter((f) => m & (1 << f)).length : 1);
  PA.lastHit = i; PA.heldAtHit = paHeld();
  PA.flashes.push({ m, t: now });
  paUnmute();
}
function paBreak() { PA.streak = 0; paMute(); }
// the next note still to be judged, if it's inside the window
function paCandidate(now) {
  const tr = paTrack();
  while (PA.next < tr.on.length && (PA.hit.has(PA.next) || PA.miss.has(PA.next))) PA.next++;
  const i = PA.next;
  return i < tr.on.length && Math.abs(tr.on[i] - now) <= PA_WINDOW ? i : -1;
}
function paStrum() {
  if (!PA.on || !S.player.playing) return;
  const now = S.player.time(), tr = paTrack(), i = paCandidate(now);
  if (i >= 0 && paMatches(tr.notes[i][1], paHeld())) return paHit(i, now);
  // a strum just after a HOPO / tap you already hit by fretting is fine (Clone Hero allows it too)
  if (PA.lastHit >= 0 && tr.notes[PA.lastHit][3] !== 0 && now - tr.on[PA.lastHit] < PA_WINDOW * 1.5) return;
  PA.overstrums++; paBreak();
}
// HOPOs and taps: hit by the frets alone (no strum) when the right frets are held inside the window. That counts
// on a fret press / release and also when the note arrives with the frets already down (pressed a little early),
// as long as the hand changed since the last hit (holding one fret can't hit a run by itself). A HOPO needs the
// streak going and the note before it hit; a tap always works.
function paFretHit(now) {
  const tr = paTrack(), i = paCandidate(now);
  if (i < 0) return;
  const nt = tr.notes[i][3], held = paHeld();
  if (!(nt === 2 || (nt === 1 && PA.streak > 0 && PA.lastHit === i - 1))) return;
  if (held === PA.heldAtHit && PA.lastHit >= 0) return;
  if (paMatches(tr.notes[i][1], held)) paHit(i, now);
}
function paFrets() {                                      // a fret pressed or released
  if (!PA.on || !S.player.playing) return;
  paFretHit(S.player.time());
}

// ---------------------------------------------------------------- controller mapping
// Guitars differ (button numbers, strum bar on a hat switch or a stick), so each controller gets a mapping, saved
// per controller in this browser. An input is {b: button} or {a: axis, v: value} (hat switches report fixed values).
// "Controller setup" (C, or on its own the first time a non-standard controller is used) asks for each fret and
// both strum directions and records what the guitar sends.
const PAD_KEY = "riffchart.padmap.";
const PAD_STEPS = ["GREEN fret", "RED fret", "YELLOW fret", "BLUE fret", "ORANGE fret", "strum DOWN", "strum UP"];
const PAD_DEFAULT = { frets: [{ b: 0 }, { b: 1 }, { b: 3 }, { b: 2 }, { b: 4 }],
  strum: [{ b: 12 }, { b: 13 }, { a: 1, v: -1 }, { a: 1, v: 1 }] };
function padMap(p) {
  try { const m = JSON.parse(localStorage.getItem(PAD_KEY + p.id) || "null"); if (m && m.frets) return m; } catch { /* none */ }
  return null;
}
function padOn(p, inp) {
  if (!inp) return false;
  if (inp.b != null) return !!(p.buttons[inp.b] && p.buttons[inp.b].pressed);
  const v = p.axes[inp.a];
  return v != null && Math.abs(v - inp.v) < 0.2;
}
function padStartSetup(p) {
  PA.cal = { step: 0, base: p ? { b: p.buttons.map((x) => x.pressed), ax: p.axes.slice() } : null, wait: false,
    id: p ? p.id : null, map: { frets: [], strum: [] } };   // "nothing pressed" = how it is right now
  if (S.player.playing) togglePlay();
}
function padSetupFrame(p) {                              // one step: wait for a new input, then for its release
  const c = PA.cal, b = p.buttons.map((x) => x.pressed), ax = p.axes.slice();
  if (!c.base) { c.base = { b, ax }; return; }
  const newBtn = b.findIndex((v, k) => v && !c.base.b[k]);
  const newAx = ax.findIndex((v, k) => Math.abs(v - c.base.ax[k]) > 0.4);
  if (c.wait) {                                          // released: back to how it was
    if (newBtn < 0 && newAx < 0) c.wait = false;
    return;
  }
  let inp = null;
  if (newBtn >= 0) inp = { b: newBtn };
  else if (newAx >= 0) inp = { a: newAx, v: Math.round(ax[newAx] * 1000) / 1000 };
  if (!inp) return;
  if (c.step < 5) c.map.frets.push(inp); else c.map.strum.push(inp);
  c.step++; c.wait = true;
  if (c.step >= PAD_STEPS.length) {
    try { localStorage.setItem(PAD_KEY + c.id, JSON.stringify(c.map)); } catch { /* still used this session */ }
    PA.padMaps = Object.assign(PA.padMaps || {}, { [c.id]: c.map });
    PA.cal = null;
    toast("Controller set up. Press Space to play (C sets it up again)");
  }
}

function paPoll() {                                       // guitar controllers: frets and strum edges
  const pads = navigator.getGamepads ? [...navigator.getGamepads()].filter(Boolean) : [];
  if (!pads.length) return;
  const p = pads[0];
  PA.padInfo = { id: p.id, mapping: p.mapping || "non-standard",                     // shown in the corner: for mapping help
    down: p.buttons.map((x, k) => (x.pressed ? k : -1)).filter((k) => k >= 0),
    axes: p.axes.map((v, k) => (Math.abs(v) > 0.5 && Math.abs(v) <= 1.01 ? `${k}:${v.toFixed(2)}` : "")).filter(Boolean) };
  if (PA.cal) { padSetupFrame(p); return; }
  let map = (PA.padMaps && PA.padMaps[p.id]) || padMap(p);
  if (!map) {
    if (p.mapping !== "standard" && !PA.askedSetup) { PA.askedSetup = true; padStartSetup(p); return; }
    map = PAD_DEFAULT;
  }
  const frets = map.frets.map((inp) => padOn(p, inp));
  const strum = map.strum.some((inp) => padOn(p, inp));
  const changed = frets.some((v, f) => v !== PA.pad[f]);
  PA.pad = frets;
  if (changed) paFrets();
  if (strum && !PA.padStrum) paStrum();
  PA.padStrum = strum;
}

// each Game view frame (gameview.js): poll the controller, mark misses, follow seeks
function paFrame(now) {
  if (!PA.on) return;
  paPoll();
  const tr = paTrack();
  if (!tr) return;
  if (Math.abs(now - PA.lastNow) > 0.5) paResync(now);     // jumped (wheel, overview)
  PA.lastNow = now;
  if (!S.player.playing) return;
  paFretHit(now);                                         // a HOPO / tap arriving with the frets already down
  while (PA.next < tr.on.length && tr.on[PA.next] < now - PA_WINDOW) {
    const i = PA.next++;
    if (PA.hit.has(i) || PA.miss.has(i)) continue;
    PA.miss.add(i); PA.misses++; paBreak();
  }
}

// drawing helpers used by gameview.js
function paFlashes(g, G, now, laneW) {
  PA.flashes = PA.flashes.filter((x) => now - x.t < 0.18 && now >= x.t - 0.05);
  for (const x of PA.flashes) {
    const age = Math.max(0, (now - x.t) / 0.18), lanes = x.m & 32 ? [0, 1, 2, 3, 4] : [0, 1, 2, 3, 4].filter((f) => x.m & (1 << f));
    for (const f of lanes) gameFlash(g, gx(G, 0, laneU(f)), G.ys, laneW, age, x.m & 32 ? OPEN_COLOR : FRET_COLORS[f]);
  }
  const held = paHeld();                                  // frets held down: the ring lights up
  for (let f = 0; f < 5; f++) if (held & (1 << f)) {
    g.globalAlpha = 0.55; g.fillStyle = FRET_COLORS[f];
    g.beginPath(); g.ellipse(gx(G, 0, laneU(f)), G.ys, laneW * 0.4, laneW * 0.18, 0, 0, Math.PI * 2); g.fill();
  }
  g.globalAlpha = 1;
}
function paHud(g, w, h) {
  const judged = PA.hits + PA.misses, pct = judged ? Math.round(PA.hits / judged * 100) : 100;
  g.textBaseline = "alphabetic";
  g.textAlign = "right"; g.fillStyle = "#ffffff";
  g.font = "700 30px system-ui, sans-serif"; g.fillText(PA.score.toLocaleString(), w - 24, 48);
  g.font = "600 15px system-ui, sans-serif"; g.fillStyle = "rgba(255,255,255,.8)";
  g.fillText(`${pct}% · streak ${PA.streak}`, w - 24, 72);
  const mult = paMult();
  g.font = "800 22px system-ui, sans-serif"; g.fillStyle = ["#ffffff", "#ffffff", "#f5a524", "#22c55e", "#67e8f9"][mult];
  g.fillText(`×${mult}`, w - 24, 102);
  g.textAlign = "right"; g.font = "500 12px system-ui, sans-serif"; g.fillStyle = "rgba(255,255,255,.55)";
  const pi = PA.padInfo;
  g.fillText(pi ? `Controller: ${pi.id.slice(0, 48)} (${pi.mapping}) · buttons ${pi.down.join(" ") || "-"}${pi.axes.length ? " · axes " + pi.axes.join(" ") : ""}`
    : "No controller seen yet: press any button on it", w - 24, h - 12);
  if (PA.cal) {                                          // controller setup prompt
    const bw = 420, bh = 170, x = (w - bw) / 2, y = h * 0.22;
    g.globalAlpha = 0.94; g.fillStyle = "#151821"; g.beginPath(); g.roundRect ? g.roundRect(x, y, bw, bh, 14) : g.rect(x, y, bw, bh); g.fill();
    g.globalAlpha = 1; g.textAlign = "center"; g.fillStyle = "rgba(255,255,255,.7)";
    g.font = "600 13px system-ui, sans-serif"; g.fillText(`CONTROLLER SETUP · step ${PA.cal.step + 1} of ${PAD_STEPS.length}`, w / 2, y + 36);
    g.fillStyle = "#ffffff"; g.font = "700 30px system-ui, sans-serif";
    g.fillText(PA.cal.wait ? "Let go…" : `Press ${PAD_STEPS[PA.cal.step]}`, w / 2, y + 88);
    g.font = "500 13px system-ui, sans-serif"; g.fillStyle = "rgba(255,255,255,.6)";
    g.fillText("One at a time · Esc cancels", w / 2, y + 130);
    return;
  }
  if (!S.player.playing && judged) {                     // paused or the song ended: results
    const bw = 360, bh = 200, x = (w - bw) / 2, y = h * 0.22;
    g.globalAlpha = 0.92; g.fillStyle = "#151821"; g.beginPath(); g.roundRect ? g.roundRect(x, y, bw, bh, 14) : g.rect(x, y, bw, bh); g.fill();
    g.globalAlpha = 1; g.textAlign = "center"; g.fillStyle = "#ffffff";
    g.font = "700 40px system-ui, sans-serif"; g.fillText(`${pct}%`, w / 2, y + 62);
    g.font = "500 15px system-ui, sans-serif"; g.fillStyle = "rgba(255,255,255,.85)";
    g.fillText(`${PA.hits} hit · ${PA.misses} missed · ${PA.overstrums} overstrum${PA.overstrums === 1 ? "" : "s"}`, w / 2, y + 96);
    g.fillText(`Best streak ${PA.best} · Score ${PA.score.toLocaleString()}`, w / 2, y + 122);
    g.font = "500 13px system-ui, sans-serif"; g.fillStyle = "rgba(255,255,255,.6)";
    g.fillText("Space to keep playing · P to start over from here · C controller setup · Esc to stop", w / 2, y + 162);
  }
}

// keys: while playing along they belong to the game (no editor shortcuts), except Space (pause) and Esc (stop)
window.addEventListener("keydown", (e) => {
  if (!PA.on || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target.closest && e.target.closest("input, select, textarea")) return;
  if (e.code in PA_KEYS) {
    e.preventDefault(); e.stopImmediatePropagation();
    if (!e.repeat && !PA.keys[PA_KEYS[e.code]]) { PA.keys[PA_KEYS[e.code]] = true; paFrets(); }
  } else if (PA_STRUM.has(e.key)) {
    e.preventDefault(); e.stopImmediatePropagation();
    if (!e.repeat) paStrum();
  } else if (e.key === "Escape") {
    e.preventDefault(); e.stopImmediatePropagation();
    if (PA.cal) { PA.cal = null; toast("Controller setup cancelled"); } else paStop();
  } else if (e.key === "c" || e.key === "C") {
    e.preventDefault(); e.stopImmediatePropagation();
    const pads = navigator.getGamepads ? [...navigator.getGamepads()].filter(Boolean) : [];
    if (pads.length) padStartSetup(pads[0]); else toast("No controller seen yet: press a button on it first", true);
  } else if (e.key === "p" || e.key === "P") {
    e.preventDefault(); e.stopImmediatePropagation(); paStart();
  } else if (e.key === " " && PA.cal) {
    e.preventDefault(); e.stopImmediatePropagation();
  } else if (e.key !== " ") {
    e.stopImmediatePropagation();                       // no other shortcuts mid-song
  }
}, true);
window.addEventListener("keyup", (e) => {
  if (!PA.on || !(e.code in PA_KEYS)) return;
  e.stopImmediatePropagation();
  PA.keys[PA_KEYS[e.code]] = false; paFrets();
}, true);
window.addEventListener("blur", () => { PA.keys = [false, false, false, false, false]; });
$("#playbtn").addEventListener("click", () => (PA.on ? paStop() : paStart()));
$("#gamebtn").addEventListener("click", () => { if (PA.on) paStop(); });   // back to watching
