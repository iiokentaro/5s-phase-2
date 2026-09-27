"""Child process that executes one pipeline preset and reports progress.

Started by pipeline_service with an extra inherited file descriptor; every
structured event and every captured line of output goes down that one fd as
JSON lines, so their relative order is exactly the order they happened in.

Running in a separate process, outside the web server, makes three things
possible: a segfault in GEOS/GDAL loses the run instead
of the server, SIGTERM to the process group actually interrupts a multi-minute
geopandas call, and the ~290 print() calls scattered through src/ can be
redirected wholesale without touching the server's own stdout.

Usage (normally invoked by the server, not by hand)
---------------------------------------------------
    python src/pipeline_runner.py --preset quick --event-fd 3
"""

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pipeline_steps
from pipeline_steps import BUILD_STEP_IDS, Ctx, steps_for
from aadt_estimation import DEFAULT as DEFAULT_AADT_PARAMS
from aadt_estimation import AadtParams
from poi_params import DEFAULT as DEFAULT_POI_PARAMS
from poi_params import PoiParams

PROGRESS_INTERVAL_S = 10.0


def _now_ms() -> int:
    return int(time.time() * 1000)


class EventWriter:
    """Serialises JSON events onto the event fd. Every emit is line-buffered and
    flushed, so the parent sees a step start before the logs that follow it."""

    def __init__(self, stream):
        self._stream = stream
        self._lock = threading.Lock()

    def emit(self, event: str, **data) -> None:
        payload = json.dumps({"event": event, "ts": _now_ms(), **data}, default=str)
        with self._lock:
            try:
                self._stream.write(payload + "\n")
                self._stream.flush()
            except (BrokenPipeError, ValueError):
                pass  # parent is gone; the step itself will fail soon enough


class LineSplitter(io.TextIOBase):
    """File-like sink that turns writes into one event per complete line."""

    def __init__(self, emit_line, stream_name: str):
        self._emit = emit_line
        self._stream = stream_name
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line, self._stream)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._emit(self._buf, self._stream)
            self._buf = ""

    def writable(self) -> bool:
        return True


class Reporter:
    """Owns the current-step cursor so redirected output is attributed correctly."""

    def __init__(self, events: EventWriter):
        self.events = events
        self.current_id: str | None = None
        self.current_started: float | None = None
        self._stop = threading.Event()
        self._watchdog = threading.Thread(target=self._tick, daemon=True)

    def start_watchdog(self) -> None:
        self._watchdog.start()

    def stop_watchdog(self) -> None:
        self._stop.set()

    def _tick(self) -> None:
        # Several build sub-steps print nothing for minutes; without this a
        # reconnecting browser would have no authoritative elapsed time to show.
        while not self._stop.wait(PROGRESS_INTERVAL_S):
            step_id, started = self.current_id, self.current_started
            if step_id and started:
                self.events.emit("step_progress", step_id=step_id,
                                 elapsed_ms=int((time.monotonic() - started) * 1000))

    def log(self, line: str, stream: str = "stdout") -> None:
        self.events.emit("step_log", step_id=self.current_id, stream=stream, line=line)

    def step_start(self, step_id: str, label_en: str, label_ja: str, group=None) -> None:
        self.current_id = step_id
        self.current_started = time.monotonic()
        self.events.emit("step_start", step_id=step_id, label_en=label_en,
                         label_ja=label_ja, group=group, started_ms=_now_ms())

    def step_done(self, step_id: str, status: str, elapsed_ms: int,
                  reason: str | None = None, artifacts=None) -> None:
        self.current_id = None
        self.current_started = None
        self.events.emit("step_done", step_id=step_id, status=status,
                         elapsed_ms=elapsed_ms, reason=reason,
                         artifacts=artifacts or [])

    def skip(self, step_id: str, reason: str) -> None:
        self.events.emit("step_done", step_id=step_id, status="skipped",
                         elapsed_ms=0, reason=reason, artifacts=[])


def _make_progress(reporter: Reporter):
    """The callback build_v_safe.build() drives its sub-step spans through."""

    @contextmanager
    def progress(step_id, label_en, label_ja, group=None):
        full_id = f"{group}.{step_id}" if group else step_id
        started = time.monotonic()
        reporter.step_start(full_id, label_en, label_ja, group)
        try:
            yield
        except BaseException:
            reporter.step_done(full_id, "failed", int((time.monotonic() - started) * 1000),
                               reason=traceback.format_exc(limit=1).strip().splitlines()[-1])
            raise
        reporter.step_done(full_id, "ok", int((time.monotonic() - started) * 1000))

    return progress


def _run_build_block(ctx: Ctx, reporter: Reporter) -> None:
    """build() owns the frame across its sub-steps, so it runs as one call
    and reports each span through the progress callback."""
    from build_v_safe import build
    ctx.gdf, rural, score = build(progress=_make_progress(reporter), poi_params=ctx.state.get("poi_params"))
    ctx.state["rural_thresholds"] = rural
    ctx.state["score_thresholds"] = score


def run_preset(preset: str, reporter: Reporter,
               severity: str = pipeline_steps.DEFAULT_SEVERITY,
               poi_params: PoiParams = DEFAULT_POI_PARAMS,
               aadt_params: AadtParams = DEFAULT_AADT_PARAMS) -> str:
    steps = steps_for(preset)
    ctx = Ctx(log=lambda line: reporter.log(str(line)))
    # Unlike severity, these change what the build computes (full/complete only;
    # quick reads the committed parquet and ignores them).
    ctx.state["poi_params"] = poi_params
    reporter.log(f"POI parameters: {poi_params.to_json()}")
    # Which Elvik (2019) severity coefficients sens_exp_model reports. It rides
    # in ctx.state like rural_thresholds and score_thresholds do; it selects
    # what is reported, never what the build computes.
    ctx.state["severity"] = severity
    # Which scale factors turn probe counts into AADT on the deliverable. Like
    # severity, this selects what is written, never what the build computes, so
    # quick honours it too.
    ctx.state["aadt_params"] = aadt_params
    reporter.log(f"AADT scale factors: {aadt_params.to_json()}")

    skipped: set[str] = set()
    failed_hard = False
    any_failed = False
    pending = list(steps)

    while pending:
        step = pending.pop(0)

        if failed_hard:
            reporter.skip(step.id, "upstream failed")
            continue

        blocked = next((n for n in step.needs if n in skipped), None)
        if blocked:
            skipped.add(step.id)
            reporter.skip(step.id, f"depends on {blocked}")
            continue

        # build()'s sub-steps are not individually callable: the frame lives in
        # build()'s locals. The first of them triggers the whole call, which
        # reports the rest itself.
        if step.id in BUILD_STEP_IDS:
            rest = [s for s in pending if s.id in BUILD_STEP_IDS]
            pending = [s for s in pending if s.id not in BUILD_STEP_IDS]
            try:
                _run_build_block(ctx, reporter)
            except BaseException:
                reporter.log(traceback.format_exc(), "stderr")
                for s in rest:
                    reporter.skip(s.id, "upstream failed")
                failed_hard = any_failed = True
            continue

        try:
            reason = step.check()
        except Exception as exc:
            reason = f"check failed: {exc}"
        if reason:
            skipped.add(step.id)
            reporter.skip(step.id, reason)
            continue

        reporter.step_start(step.id, step.label_en, step.label_ja, step.group)
        started = time.monotonic()
        before = len(ctx.artifacts)
        try:
            step.run(ctx)
        except BaseException:
            reporter.log(traceback.format_exc(), "stderr")
            last = traceback.format_exc(limit=1).strip().splitlines()[-1]
            reporter.step_done(step.id, "failed", int((time.monotonic() - started) * 1000),
                               reason=last)
            any_failed = True
            if not step.optional:
                failed_hard = True
            else:
                skipped.add(step.id)  # dependents cannot use what was never written
        else:
            reporter.step_done(step.id, "ok", int((time.monotonic() - started) * 1000),
                               artifacts=ctx.artifacts[before:])

    if failed_hard:
        return "failed"
    return "partial" if any_failed else "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", required=True, choices=pipeline_steps.ALL_PRESETS)
    parser.add_argument("--event-fd", type=int, required=True)
    parser.add_argument("--severity", default=pipeline_steps.DEFAULT_SEVERITY,
                        choices=sorted(pipeline_steps.SEVERITY_CHOICES),
                        help="Elvik (2019) severity coefficients for the exponential-model step")
    parser.add_argument("--poi-params", default=None,
                        help="JSON object of POI parameters (src/poi_params.py)")
    parser.add_argument("--aadt-params", default=None,
                        help="JSON object of AADT scale factors (src/aadt_estimation.py)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)  # every module in src/ resolves data paths relative to the repo root

    import matplotlib
    matplotlib.use("Agg")  # no GUI backend is available in this process

    events = EventWriter(os.fdopen(args.event_fd, "w", buffering=1))
    reporter = Reporter(events)
    started = time.monotonic()

    sink_out = LineSplitter(reporter.log, "stdout")
    sink_err = LineSplitter(reporter.log, "stderr")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = sink_out, sink_err
    # force=True: prefilter_pbf, poi_isochrone and exposure_signals configure
    # logging themselves, and without it their handlers would keep the real fds.
    logging.basicConfig(level=logging.INFO, stream=sink_out,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)

    reporter.start_watchdog()
    try:
        status = run_preset(args.preset, reporter, severity=args.severity,
                            poi_params=PoiParams.from_json(args.poi_params),
                            aadt_params=AadtParams.from_json(args.aadt_params))
    except KeyboardInterrupt:
        status = "cancelled"
    except BaseException:
        sink_err.write(traceback.format_exc() + "\n")
        status = "failed"
    finally:
        reporter.stop_watchdog()
        sink_out.flush()
        sink_err.flush()
        sys.stdout, sys.stderr = real_stdout, real_stderr

    events.emit("run_summary", status=status,
                elapsed_ms=int((time.monotonic() - started) * 1000))
    return 0 if status in ("ok", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
