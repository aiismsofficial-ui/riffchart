"""Background charting queue for the app: one worker thread, models loaded once on first use.

Jobs are kept in <workspace>/../jobs.json so a restart resumes queued songs (a song that was mid-chart
is queued again). Progress lines go to logs/ui.log, which the job monitor (scripts/watch.py) shows.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..formats.song import DIFFICULTIES
from ..paths import PROJECT

STAGE_LABEL = {"load": "loading audio", "separate": "separating guitar", "beats": "tracking beats",
               "chart": "charting", "post": "polishing", "write": "writing files"}


@dataclass
class Options:
    difficulties: list[str] = field(default_factory=lambda: list(DIFFICULTIES))
    density: float = 0.0
    chords: str = "auto"
    stems: bool = True
    overwrite: bool = False
    follow: str = "auto"         # guitar | auto | all (product/follow.py)

    @staticmethod
    def parse(d: dict | None) -> "Options":
        d = d or {}
        o = Options(
            difficulties=[x for x in DIFFICULTIES if x in d.get("difficulties", DIFFICULTIES)],
            density=float(d.get("density", 0.0)), chords=str(d.get("chords", "auto")),
            stems=bool(d.get("stems", True)), overwrite=bool(d.get("overwrite", False)),
            follow=str(d.get("follow", "auto")))
        from ..product.core import CHORD_BIN
        from ..product.follow import MODES
        if not o.difficulties:
            raise ValueError("pick at least one difficulty")
        if not -1 <= o.density <= 1:
            raise ValueError("density must be between -1 and 1")
        if o.chords not in CHORD_BIN:
            raise ValueError(f"chords must be one of {', '.join(CHORD_BIN)}")
        if o.follow not in MODES:
            raise ValueError(f"follow must be one of {', '.join(MODES)}")
        return o


@dataclass
class Job:
    id: str
    source: str                  # audio file path on this PC
    name: str                    # display name
    options: Options
    upload: bool = False         # source is our upload copy (deleted after charting)
    target: str = ""             # re-chart: workspace folder to replace (names kept)
    meta: dict = field(default_factory=dict)   # re-chart: title/artist/album/genre/year to keep
    status: str = "queued"       # queued | running | done | skipped | failed | cancelled
    stage: str = ""
    progress: float = 0.0        # 0..1
    message: str = ""
    song: str = ""               # workspace folder name when done
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage_label"] = STAGE_LABEL.get(self.stage, self.stage)
        return d


class JobQueue:
    def __init__(self, workspace: Path, stage_weights: dict[str, float], make_charter: Callable,
                 state_file: Path | None = None, log_file: Path | None = None):
        self.workspace = Path(workspace)
        self.weights = stage_weights
        self.make_charter = make_charter      # () -> object with chart_file(...) like RiffCharter
        self.state_file = state_file or self.workspace.parent / "jobs.json"
        self.log_file = log_file or PROJECT / "logs" / "ui.log"
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.charter = None
        self.gpu = threading.Lock()           # one model user at a time (song jobs, section re-charts)
        self.load_lock = threading.Lock()
        self.model_state = "not loaded"       # not loaded | loading | ready | failed
        self.stop = False
        self._load()
        self.thread = threading.Thread(target=self._run, daemon=True, name="chart-worker")

    def start(self) -> None:
        self.thread.start()
        if any(j.status == "queued" for j in self.jobs.values()):
            self.wake.set()

    # --- persistence / logging
    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            for d in json.loads(self.state_file.read_text(encoding="utf-8")):
                d.pop("stage_label", None)
                d["options"] = Options(**d["options"])
                j = Job(**d)
                if j.status == "running":            # interrupted by a restart: chart it again
                    j.status, j.stage, j.progress, j.message = "queued", "", 0.0, "restarted"
                self.jobs[j.id] = j
        except Exception:                            # a damaged state file must not stop the app
            self.state_file.rename(self.state_file.with_suffix(".bad.json"))

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps([j.to_dict() for j in self.jobs.values()], indent=1), encoding="utf-8")
        tmp.replace(self.state_file)

    def log(self, msg: str) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")

    # --- API
    def add(self, source: Path, name: str, options: Options, upload: bool = False, target: str = "",
            meta: dict | None = None) -> Job:
        j = Job(uuid.uuid4().hex[:12], str(source), name, options, upload, target=target, meta=meta or {})
        with self.lock:
            self.jobs[j.id] = j
            self._save()
        self.log(f"queued {name} ({self.counts()})")
        self.wake.set()
        return j

    def list(self) -> list[dict]:
        with self.lock:
            return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: j.created)]

    def counts(self) -> str:
        n = {s: 0 for s in ("queued", "running", "done")}
        for j in self.jobs.values():
            if j.status in n:
                n[j.status] += 1
        return f"{n['done']} done, {n['running']} running, {n['queued']} queued"

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job, or remove a finished one from the list. A running song finishes first."""
        with self.lock:
            j = self.jobs.get(job_id)
            if j is None or j.status == "running":
                return False
            if j.status == "queued":
                self._drop_upload(j)
            del self.jobs[job_id]
            self._save()
            return True

    def retry(self, job_id: str) -> bool:
        with self.lock:
            j = self.jobs.get(job_id)
            if j is None or j.status not in ("failed", "skipped"):
                return False
            if not Path(j.source).exists():
                j.message = "the audio file is gone; add it again"
                return False
            if j.status == "skipped":
                j.options.overwrite = True
            j.status, j.stage, j.progress, j.message, j.created = "queued", "", 0.0, "", time.time()
            self._save()
        self.wake.set()
        return True

    def clear_finished(self) -> int:
        with self.lock:
            gone = [k for k, j in self.jobs.items() if j.status in ("done", "skipped", "failed", "cancelled")]
            for k in gone:
                del self.jobs[k]
            self._save()
            return len(gone)

    def shutdown(self) -> None:
        self.stop = True
        self.wake.set()

    # --- worker
    def _drop_upload(self, j: Job) -> None:
        if j.upload:
            d = Path(j.source).parent
            shutil.rmtree(d, ignore_errors=True)

    def _next(self) -> Job | None:
        with self.lock:
            q = [j for j in self.jobs.values() if j.status == "queued"]
            if not q:
                return None
            j = min(q, key=lambda j: j.created)
            j.status, j.started, j.progress, j.message = "running", time.time(), 0.0, ""
            self._save()
            return j

    def _run(self) -> None:
        while not self.stop:
            j = self._next()
            if j is None:
                self.wake.wait(2.0)
                self.wake.clear()
                continue
            self._chart(j)

    def _ensure_charter(self, j: Job | None = None) -> None:
        with self.load_lock:
            if self.charter is not None:
                return
            self.model_state = "loading"
            if j is not None:
                j.stage, j.message = "", "loading models (first song only)"
            self.log("loading models")
            try:
                self.charter = self.make_charter()
                self.model_state = "ready"
            except Exception:
                self.model_state = "failed"
                raise

    def run_on_model(self, fn: Callable):
        """fn(charter) on the caller's thread, with the models loaded and no song job using them."""
        self._ensure_charter()
        with self.gpu:
            return fn(self.charter)

    def _chart(self, j: Job) -> None:
        from ..product.core import Style
        order = list(self.weights)
        done_w = {s: sum(self.weights[x] for x in order[: i]) for i, s in enumerate(order)}

        def on_stage(name: str) -> None:
            j.stage, j.message = name, ""
            j.progress = min(done_w.get(name, j.progress), 0.99)
            self.log(f"{j.name[:60]}: {STAGE_LABEL.get(name, name)} [{self.counts()}]")

        try:
            self._ensure_charter(j)
            o = j.options
            kw = {}
            if j.target:
                from ..product.meta import SongMeta
                kw = {"folder_name": j.target, "meta": SongMeta(**{k: j.meta.get(k, "") for k in
                                                                   ("title", "artist", "album", "genre", "year")})}
            with self.gpu:
                r = self.charter.chart_file(j.source, self.workspace, o.difficulties,
                                            Style(o.density, o.chords, o.follow), write_stems=o.stems,
                                            overwrite=o.overwrite or bool(j.target), on_stage=on_stage, **kw)
            if r is None:
                j.status, j.message = "skipped", "already in your library (tick 're-chart' to redo it)"
                from ..product.core import safe_name
                from ..product.meta import read_meta
                m = read_meta(j.source)
                j.song = j.target or safe_name(f"{m.artist} - {m.title}")
            else:
                j.status, j.song = "done", r.folder.name
                j.message = f"{sum(r.seconds.values()):.0f} s, Expert {r.n_notes.get('expert', 0)} notes"
                if getattr(r, "filled_s", 0) >= 5:
                    j.message += f"; followed other instruments for {r.filled_s:.0f} s where the guitar is quiet"
            self._drop_upload(j)
            self.log(f"{j.status} {j.name[:60]} {j.message} [{self.counts()}]")
        except Exception as e:
            j.status, j.message = "failed", f"{type(e).__name__}: {e}"
            self.log(f"FAILED {j.name[:60]}: {j.message}\n{traceback.format_exc()}")
        j.progress, j.finished = (1.0 if j.status != "failed" else j.progress), time.time()
        with self.lock:
            self._save()
