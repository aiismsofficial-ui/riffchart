"use strict";
// Editor playback. The stems are downloaded once and decoded into memory (Web Audio), then every stem
// plays from the same AudioContext clock: sample-locked together, no network during playback, and the
// highway follows what is *heard* (getOutputTimestamp includes the output device latency), minus an
// optional extra "audio delay" for setups that under-report it (Bluetooth headphones).
//
// At 100% speed the in-memory buffers play directly. Slower speeds use <audio> elements on the same
// in-memory data (blob URLs, still no network) because only those keep the pitch when slowed down.

class Player {
  constructor(urls, names, duration) {
    this.names = names; this.duration = duration; this.rate = 1; this.playing = false;
    this._t = 0; this.delay = 0; this.mode = null;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    this.gains = names.map(() => { const g = this.ctx.createGain(); g.connect(this.ctx.destination); return g; });
    this.vol = names.map(() => 1);
    this.buffers = null; this.blobs = null; this.els = null; this.sources = [];
    this.onprogress = null;
    this.ready = this.load(urls);
  }

  async load(urls) {
    let done = 0;
    const raw = await Promise.all(urls.map(async (u) => {
      const r = await fetch(u);
      if (!r.ok) throw new Error(`${u.split("/").pop()}: ${r.status}`);
      const b = await r.arrayBuffer();
      done++; if (this.onprogress) this.onprogress(done, urls.length);
      return b;
    }));
    this.blobs = raw.map((b) => URL.createObjectURL(new Blob([b])));
    // decodeAudioData detaches its input, so give it copies
    this.buffers = await Promise.all(raw.map((b) => this.ctx.decodeAudioData(b.slice(0))));
    this.duration = Math.max(this.duration || 0, ...this.buffers.map((b) => b.duration));
  }

  resume() { if (this.ctx.state === "suspended") this.ctx.resume(); }

  // context time currently coming out of the speakers
  heardCtxTime() {
    const ts = this.ctx.getOutputTimestamp ? this.ctx.getOutputTimestamp() : null;
    if (ts && ts.contextTime > 0 && ts.performanceTime > 0) {
      return ts.contextTime + Math.max(0, performance.now() - ts.performanceTime) / 1000;
    }
    return this.ctx.currentTime - (this.ctx.outputLatency || this.ctx.baseLatency || 0);
  }

  time() {
    if (!this.playing) return this._t;
    let t;
    if (this.mode === "buffer") {
      t = this.off0 + (this.heardCtxTime() - this.ctx0) * this.rate - this.delay;
    } else {
      const m = this.els[0], now = performance.now();
      const pred = this.anchorT + (now - this.anchorPerf) / 1000 * this.rate;
      if (m && !m.paused && !m.seeking) {
        const err = m.currentTime - pred;
        if (Math.abs(err) > 0.08) { this.anchorT = m.currentTime; this.anchorPerf = now; }
        else this.anchorT += err * 0.05;
      }
      t = this.anchorT + (now - this.anchorPerf) / 1000 * this.rate - this.delay - (this.ctx.outputLatency || 0);
    }
    if (t >= this.duration) { this.pause(); this._t = this.duration; return this.duration; }
    return t;
  }

  play() {
    if (this.playing || !this.buffers) return;
    this.resume();
    const t = Math.max(-2, Math.min(this._t, this.duration));
    if (this.rate === 1) {
      this.mode = "buffer";
      this.ctx0 = this.ctx.currentTime + 0.04;                  // a little lead so every stem starts together
      this.off0 = t;                                            // audio position rendered at context time ctx0
      this.sources = this.buffers.map((buf, i) => {
        const s = this.ctx.createBufferSource();
        s.buffer = buf; s.connect(this.gains[i]);
        const pos = this.off0;
        if (pos >= 0) s.start(this.ctx0, Math.min(pos, buf.duration));
        else s.start(this.ctx0 - pos, 0);                       // before the song starts: wait
        return s;
      });
    } else {
      this.mode = "media";
      if (!this.els) {
        this.els = this.blobs.map((u) => { const a = new Audio(u); a.preservesPitch = true; a.preload = "auto"; return a; });
        this.els[0].addEventListener("ended", () => this.pause());
      }
      this.els.forEach((a, i) => { a.volume = this.vol[i]; a.playbackRate = this.rate; a.currentTime = Math.max(0, t); });
      this.anchorT = t; this.anchorPerf = performance.now();
      Promise.all(this.els.map((a) => a.play())).catch((e) => toast("Audio could not start: " + e.message, true));
      this._sync = setInterval(() => {                         // keep the stems together while slowed
        const m = this.els[0].currentTime;
        for (const a of this.els.slice(1)) if (!a.seeking && Math.abs(a.currentTime - m) > 0.05) a.currentTime = m;
      }, 500);
    }
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    this._t = this.time();
    this.playing = false;
    if (this.mode === "buffer") for (const s of this.sources) { try { s.stop(); } catch { /* not started */ } }
    else if (this.els) { this.els.forEach((a) => a.pause()); clearInterval(this._sync); }
    this.sources = [];
  }

  seek(t) {
    t = Math.max(-2, Math.min(t, this.duration));
    if (this.playing) { this.pause(); this._t = t; this.play(); } else this._t = t;
  }

  setRate(r) {
    const was = this.playing;
    if (was) this.pause();
    this.rate = r;
    if (was) this.play();
  }

  setVolume(i, v) {
    this.vol[i] = v;
    this.gains[i].gain.value = v;
    if (this.els && this.els[i]) this.els[i].volume = v;
  }

  setDelay(ms) {
    const was = this.playing, t = this.time();
    if (was) this.pause();
    this.delay = ms / 1000; this._t = t;
    if (was) this.play();
  }

  // context time at which audio position `t` is rendered (buffer mode): a click scheduled there plays
  // exactly with that moment of the song, whatever the output latency
  ctxTimeOf(t) { return this.ctx0 + (t - this.off0) / this.rate; }

  click(freq, gain, len, when) {
    const c = this.ctx, o = c.createOscillator(), g = c.createGain();
    const t = Math.max(when ?? c.currentTime, c.currentTime);
    o.frequency.value = freq; o.type = "square";
    g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + len);
    o.connect(g).connect(c.destination); o.start(t); o.stop(t + len + 0.01);
  }
}
