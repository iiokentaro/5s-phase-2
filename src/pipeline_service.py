"""Server-side manager for pipeline runs: one at a time, streamed over SSE.

Holds the authoritative state of the current (or last) run so that a browser
reloaded mid-run can redraw the whole chain from `/api/pipeline/status` and then
resume the event stream from exactly where it left off, with no gap and no
duplicate. Standard library only -- the repository has no web dependency and
this feature does not add one.
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pipeline_steps
from aadt_estimation import AadtParams
from poi_params import PoiParams

JOURNAL_PATH = REPO_ROOT / "outputs" / ".pipeline_run.json"
JOURNAL_MIN_INTERVAL_S = 1.0
EVENT_HISTORY = 3000
LOG_TAIL = 500
SUBSCRIBER_QUEUE_SIZE = 2000
TERM_GRACE_S = 5.0

_LOCK = threading.RLock()
_RUN: "RunState | None" = None


def _now_ms() -> int:
    return int(time.time() * 1000)


class StepState:
    def __init__(self, spec: dict):
        self.id = spec["id"]
        self.label_en = spec["label_en"]
        self.label_ja = spec["label_ja"]
        self.group = spec.get("group")
        self.optional = spec.get("optional", False)
        self.status = "pending"
        self.started_ms: int | None = None
        self.elapsed_ms: int | None = None
        self.reason: str | None = spec.get("skip_reason")
        self.artifacts: list = []
        self.log: deque = deque(maxlen=LOG_TAIL)

    def snapshot(self, with_log: bool = True) -> dict:
        out = {
            "id": self.id, "label_en": self.label_en, "label_ja": self.label_ja,
            "group": self.group, "optional": self.optional, "status": self.status,
            "started_ms": self.started_ms, "elapsed_ms": self.elapsed_ms,
            "reason": self.reason, "artifacts": self.artifacts,
        }
        if with_log:
            out["log_tail"] = list(self.log)
        return out


class RunState:
    def __init__(self, preset: str, steps: list[dict]):
        self.run_id = uuid.uuid4().hex
        self.preset = preset
        self.status = "running"
        self.started_ms = _now_ms()
        self.elapsed_ms: int | None = None
        self.pid: int | None = None
        self.detached = False
        self.reload_pmtiles: str | None = None
        self.seq = 0
        self.steps = [StepState(s) for s in steps]
        self._by_id = {s.id: s for s in self.steps}
        self.events: deque = deque(maxlen=EVENT_HISTORY)
        self.subscribers: set = set()
        self._journal_written = 0.0
        self._proc: subprocess.Popen | None = None
        self._cancelling = False

    # -- snapshot -----------------------------------------------------------
    def snapshot(self, with_log: bool = True) -> dict:
        return {
            "run_id": self.run_id, "preset": self.preset, "status": self.status,
            "started_ms": self.started_ms,
            "elapsed_ms": self.elapsed_ms if self.elapsed_ms is not None
                          else _now_ms() - self.started_ms,
            "server_ms": _now_ms(),
            "last_seq": self.seq,
            "detached": self.detached,
            "reload_pmtiles": self.reload_pmtiles,
            "counts": self.counts(),
            "steps": [s.snapshot(with_log) for s in self.steps],
        }

    def counts(self) -> dict:
        out = {"ok": 0, "skipped": 0, "failed": 0, "pending": 0, "running": 0}
        for s in self.steps:
            key = s.status if s.status in out else "failed"
            out[key] += 1
        return out

    # -- event fan-out ------------------------------------------------------
    def publish(self, event: dict) -> None:
        with _LOCK:
            self.seq += 1
            event = {**event, "seq": self.seq, "run_id": self.run_id}
            self._apply(event)
            self.events.append(event)
            dead = []
            for q in self.subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)  # slow client; it will resync via /status
            for q in dead:
                self.subscribers.discard(q)
            self._write_journal()

    def _apply(self, event: dict) -> None:
        """Keep `steps` a correct snapshot at all times, so /status never has to
        replay history to answer."""
        kind = event.get("event")
        step = self._by_id.get(event.get("step_id"))
        if kind == "step_start" and step:
            step.status = "running"
            step.started_ms = event.get("started_ms") or _now_ms()
            step.elapsed_ms = 0
            step.reason = None
        elif kind == "step_progress" and step:
            step.elapsed_ms = event.get("elapsed_ms")
        elif kind == "step_log" and step:
            step.log.append({"stream": event.get("stream", "stdout"), "line": event.get("line", "")})
        elif kind == "step_done" and step:
            step.status = event.get("status", "ok")
            step.elapsed_ms = event.get("elapsed_ms", 0)
            step.reason = event.get("reason")
            step.artifacts = event.get("artifacts") or []
        elif kind == "run_done":
            self.status = event.get("status", "failed")
            self.elapsed_ms = event.get("elapsed_ms")
            self.reload_pmtiles = event.get("reload_pmtiles")
            for s in self.steps:
                if s.status in ("pending", "running"):
                    s.status = "skipped"
                    s.reason = s.reason or "run ended"

    def _write_journal(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._journal_written < JOURNAL_MIN_INTERVAL_S:
            return
        self._journal_written = now
        try:
            JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = self.snapshot(with_log=False)
            payload["pid"] = self.pid
            tmp = JOURNAL_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, JOURNAL_PATH)
        except OSError:
            pass  # the journal is a convenience, never a reason to fail a run

    # -- subscribers --------------------------------------------------------
    def subscribe(self, from_seq: int) -> tuple[list, "queue.Queue"]:
        """Snapshot the backlog and register the queue under one lock, so no
        event can slip through the gap between the two."""
        with _LOCK:
            backlog = [e for e in self.events if e["seq"] > from_seq]
            q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
            if self.status != "running":
                q.put_nowait(None)  # nothing more will ever arrive
            else:
                self.subscribers.add(q)
            return backlog, q

    def unsubscribe(self, q) -> None:
        with _LOCK:
            self.subscribers.discard(q)

    def close_subscribers(self) -> None:
        with _LOCK:
            for q in list(self.subscribers):
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
            self.subscribers.clear()


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------

def status() -> dict | None:
    with _LOCK:
        return _RUN.snapshot() if _RUN else _load_journal()


def is_running() -> bool:
    with _LOCK:
        return bool(_RUN and _RUN.status == "running")


def _load_journal() -> dict | None:
    """After a server restart the in-memory state is gone; the journal still has
    the last chain, which is more useful to show than an empty panel."""
    try:
        data = json.loads(JOURNAL_PATH.read_text())
    except (OSError, ValueError):
        return None
    if data.get("status") == "running":
        pid = data.get("pid")
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            data["detached"] = True  # cannot reattach to its output fd
        else:
            data["status"] = "interrupted"
            for step in data.get("steps", []):
                if step.get("status") in ("pending", "running"):
                    step["status"] = "skipped"
                    step["reason"] = step.get("reason") or "server stopped"
    data["last_seq"] = data.get("last_seq", 0)
    return data


def start(preset: str, severity: str = pipeline_steps.DEFAULT_SEVERITY,
          poi_params: dict | None = None, aadt_params: dict | None = None) -> dict:
    """Spawn the runner. Raises RuntimeError if a run is already in flight.

    `severity`, `poi_params` and `aadt_params` are validated again here,
    because this function is also reachable from tests and from the CLI. A bad
    value raises ValueError.
    """
    global _RUN
    if severity not in pipeline_steps.SEVERITY_CHOICES:
        raise ValueError(f"unknown severity {severity}")
    params = PoiParams.from_dict(poi_params)
    aadt = AadtParams.from_dict(aadt_params)
    with _LOCK:
        if is_running():
            raise RuntimeError("a pipeline run is already in progress")
        specs = pipeline_steps.describe(preset)
        run = RunState(preset, specs)
        _RUN = run

    read_fd, write_fd = os.pipe()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, str(REPO_ROOT / "src" / "pipeline_runner.py"),
           "--preset", preset, "--severity", severity, "--poi-params", params.to_json(),
           "--aadt-params", aadt.to_json(), "--event-fd", str(write_fd)]
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        pass_fds=(write_fd,),
        start_new_session=True,  # its own process group, so killpg reaches osmium/tippecanoe too
    )
    os.close(write_fd)
    run.pid = proc.pid
    run._proc = proc
    run._write_journal(force=True)

    threading.Thread(target=_read_events, args=(run, read_fd), daemon=True).start()
    threading.Thread(target=_read_stdout, args=(run, proc), daemon=True).start()
    threading.Thread(target=_wait, args=(run, proc), daemon=True).start()
    return {"run_id": run.run_id, "last_seq": run.seq, "preset": preset, "severity": severity,
            "poi_params": params.to_dict(), "aadt_params": aadt.to_dict()}


def _read_events(run: RunState, read_fd: int) -> None:
    with os.fdopen(read_fd, "r") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "run_summary":
                run._summary = event  # consumed by _wait once the process exits
                continue
            run.publish(event)


def _read_stdout(run: RunState, proc: subprocess.Popen) -> None:
    """Anything written straight to the real fd 1/2 -- GDAL/GEOS C-level warnings
    that never pass through sys.stdout -- still belongs on the chain."""
    for line in proc.stdout:
        run.publish({"event": "step_log", "step_id": None,
                     "stream": "raw", "line": line.rstrip()})


def _wait(run: RunState, proc: subprocess.Popen) -> None:
    while proc.poll() is None:
        _enforce_timeout(run)
        time.sleep(1.0)
    code = proc.wait()
    summary = getattr(run, "_summary", None)

    if run._cancelling:
        status = "cancelled"
    elif summary:
        status = summary.get("status", "failed")
    elif code == 0:
        status = "ok"
    elif code < 0:
        status = "failed"
    else:
        status = "failed"

    tiles = next((s for s in run.steps if s.id == "build_tiles"), None)
    reload_url = None
    if tiles and tiles.status == "ok":
        reload_url = f"segments_priority.pmtiles?v={run.run_id}"

    if code < 0 and not run._cancelling:
        run.publish({"event": "step_log", "step_id": None, "stream": "stderr",
                     "line": f"runner terminated by signal {-code}"})

    run.publish({
        "event": "run_done", "status": status,
        "elapsed_ms": (summary or {}).get("elapsed_ms", _now_ms() - run.started_ms),
        "counts": run.counts(), "reload_pmtiles": reload_url,
        "exit_code": code,
    })
    run._write_journal(force=True)
    run.close_subscribers()


def _enforce_timeout(run: RunState) -> None:
    by_id = {s.id: s for s in pipeline_steps.STEPS}
    for step in run.steps:
        if step.status != "running" or not step.started_ms:
            continue
        spec = by_id.get(step.id)
        limit = spec.timeout_s if spec else 1800
        if (_now_ms() - step.started_ms) / 1000 > limit:
            run.publish({"event": "step_log", "step_id": step.id, "stream": "stderr",
                         "line": f"step exceeded its {limit}s budget -- terminating"})
            cancel(run.run_id, timeout=True)
            return


def cancel(run_id: str | None = None, timeout: bool = False) -> bool:
    with _LOCK:
        run = _RUN
        if run is None or run.status != "running":
            return False
        if run_id and run_id != run.run_id:
            raise ValueError("run_id does not match the active run")
        run._cancelling = not timeout
        proc = run._proc
        pid = run.pid

    if proc is None:  # detached orphan recovered from the journal
        if pid:
            _killpg(pid)
        return True

    _killpg(proc.pid)
    deadline = time.monotonic() + TERM_GRACE_S
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        _killpg(proc.pid, signal.SIGKILL)
    return True


def _killpg(pid: int, sig=signal.SIGTERM) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def shutdown() -> None:
    """Never leave a multi-GB pyrosm process behind when the server exits."""
    with _LOCK:
        run = _RUN
    if run and run.status == "running" and run.pid:
        _killpg(run.pid)
        time.sleep(0.3)
        _killpg(run.pid, signal.SIGKILL)
