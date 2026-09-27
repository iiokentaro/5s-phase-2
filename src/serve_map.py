"""Serve the MapLibre + PMTiles map (docs/) on localhost, optionally with the
pipeline-run API behind it.

PMTiles fetches tiles with HTTP Range requests, which Python's built-in
`http.server` ignores (it always answers 200 with the full file). This handler
adds single-range `bytes=start-end` support so docs/index.html works locally.

With `--enable-pipeline` the same server also exposes `/api/pipeline/*`, which
the sidebar in docs/index.html uses to run the data pipeline and stream its
progress back over Server-Sent Events. That API executes repository code, so it
is off unless asked for, it is bound to the loopback interface only, and every
request is checked against the Host/Origin allowlist below (a browser on any
website can otherwise reach 127.0.0.1).

Usage
-----
    python src/serve_map.py                      # http://localhost:8000/
    python src/serve_map.py --port 8080
    python src/serve_map.py --enable-pipeline    # map + pipeline GUI
"""

import argparse
import atexit
import functools
import json
import os
import queue
import re
import signal
import sys
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")

sys.path.insert(0, str(REPO_ROOT / "src"))

# Requiring a header no cross-site form or simple fetch() can set forces a CORS
# preflight, which this server never answers -- so another site cannot POST here.
CSRF_HEADER = "X-5S-Pipeline"
SSE_HEARTBEAT_S = 15.0


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        m = RANGE_RE.match(self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()

        size = os.path.getsize(path)
        start_s, end_s = m.groups()
        if start_s:
            start = int(start_s)
            end = min(int(end_s), size - 1) if end_s else size - 1
        else:  # suffix range: last N bytes
            start = max(size - int(end_s or 0), 0)
            end = size - 1
        if start > end or start >= size:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        f = open(path, "rb")
        f.seek(start)
        self._range_remaining = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._range_remaining))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)
        self._range_remaining = None


class ApiHandler(RangeRequestHandler):
    """Adds /api/pipeline/* in front of the static file handler.

    send_head() is deliberately untouched: PMTiles range serving is the one thing
    this server already does correctly and must keep doing (tests/test_range_requests.py).
    """

    pipeline_enabled = False

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._api_get()
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            return self._api_post()
        self.send_error(HTTPStatus.NOT_FOUND)

    # -- helpers ------------------------------------------------------------
    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _allowed_origins(self):
        port = self.server.server_address[1]
        return {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host") in (f"localhost:{port}", f"127.0.0.1:{port}")

    def _gate(self, *, post: bool) -> bool:
        """Returns False (having already answered) when the request must not run."""
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"error": "host not allowed"})
            return False
        if not self.pipeline_enabled:
            self._json(HTTPStatus.FORBIDDEN, {
                "error": "pipeline disabled",
                "hint": "restart the server with --enable-pipeline",
            })
            return False
        if post:
            if self.headers.get(CSRF_HEADER) != "1":
                self._json(HTTPStatus.FORBIDDEN, {"error": f"missing {CSRF_HEADER} header"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in self._allowed_origins():
                self._json(HTTPStatus.FORBIDDEN, {"error": "origin not allowed"})
                return False
        return True

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    # -- GET ----------------------------------------------------------------
    def _api_get(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/pipeline/status":
            # Answered even when disabled, so the sidebar can explain itself.
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "host not allowed"})
            if not self.pipeline_enabled:
                return self._json(HTTPStatus.OK, {
                    "enabled": False, "running": False, "run": None,
                    "hint": "restart the server with --enable-pipeline",
                })
            import pipeline_service
            return self._json(HTTPStatus.OK, {
                "enabled": True,
                "running": pipeline_service.is_running(),
                "run": pipeline_service.status(),
            })

        if not self._gate(post=False):
            return None

        if parsed.path == "/api/pipeline/steps":
            import pipeline_steps
            preset = (query.get("preset") or ["quick"])[0]
            if preset not in pipeline_steps.ALL_PRESETS:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"unknown preset {preset}"})
            return self._json(HTTPStatus.OK, {
                "preset": preset,
                "presets": pipeline_steps.PRESET_META,
                "steps": pipeline_steps.describe(preset),
            })

        if parsed.path == "/api/pipeline/events":
            return self._sse(query)

        return self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

    # -- POST ---------------------------------------------------------------
    def _api_post(self):
        parsed = urlparse(self.path)
        if not self._gate(post=True):
            return None
        import pipeline_service

        if parsed.path == "/api/pipeline/run":
            import pipeline_steps
            body = self._body()  # the request body can only be read once
            preset = body.get("preset", "quick")
            severity = body.get("severity", pipeline_steps.DEFAULT_SEVERITY)
            if preset not in pipeline_steps.ALL_PRESETS:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"unknown preset {preset}"})
            if severity not in pipeline_steps.SEVERITY_CHOICES:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"unknown severity {severity}"})
            from aadt_estimation import AadtParams
            from poi_params import PoiParams
            try:
                PoiParams.from_dict(body.get("poi_params"))
                AadtParams.from_dict(body.get("aadt_params"))
            except ValueError as exc:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            try:
                started = pipeline_service.start(preset, severity=severity,
                                                 poi_params=body.get("poi_params"),
                                                 aadt_params=body.get("aadt_params"))
            except RuntimeError as exc:
                return self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            return self._json(HTTPStatus.ACCEPTED, started)

        if parsed.path == "/api/pipeline/cancel":
            try:
                ok = pipeline_service.cancel(self._body().get("run_id"))
            except ValueError as exc:
                return self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            return self._json(HTTPStatus.ACCEPTED, {"cancelled": ok})

        return self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

    # -- SSE ----------------------------------------------------------------
    def _sse(self, query):
        import pipeline_service

        run = pipeline_service._RUN
        if run is None:
            return self._json(HTTPStatus.NOT_FOUND, {"error": "no run to stream"})
        wanted = (query.get("run_id") or [None])[0]
        if wanted and wanted != run.run_id:
            return self._json(HTTPStatus.CONFLICT, {"error": "run_id is not the current run"})

        # The browser resends the last id it saw on an automatic reconnect; an
        # explicit from_seq covers the first connect after a page reload.
        last_event_id = self.headers.get("Last-Event-ID")
        from_seq = 0
        for candidate in (last_event_id, (query.get("from_seq") or ["0"])[0]):
            try:
                from_seq = int(candidate)
                break
            except (TypeError, ValueError):
                continue

        backlog, q = run.subscribe(from_seq)

        # HTTP/1.0 (the default here) ends a response at connection close, so an
        # open-ended body needs neither Content-Length nor chunked encoding.
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            self._write_frame(": open")
            for event in backlog:
                self._send_event(event)
            while True:
                try:
                    event = q.get(timeout=SSE_HEARTBEAT_S)
                except queue.Empty:
                    # Proves the connection is alive during the silent minutes of
                    # refine_influenced_segments, and lets a closed tab be noticed.
                    self._write_frame(f": hb {int(time.time() * 1000)}")
                    continue
                if event is None:
                    break
                self._send_event(event)
                if event.get("event") == "run_done":
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            run.unsubscribe(q)
        return None

    def _send_event(self, event):
        self._write_frame(
            f"id: {event['seq']}\n"
            f"event: {event.get('event', 'message')}\n"
            f"data: {json.dumps(event, default=str)}"
        )

    def _write_frame(self, text: str) -> None:
        self.wfile.write((text + "\n\n").encode())
        self.wfile.flush()


def make_handler(enable_pipeline: bool, directory: Path = DOCS_DIR):
    return functools.partial(
        type("BoundApiHandler", (ApiHandler,), {"pipeline_enabled": enable_pipeline}),
        directory=str(directory),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--enable-pipeline", action="store_true",
                        help="expose /api/pipeline/* so the sidebar can run the pipeline")
    args = parser.parse_args()

    # Loopback only, and deliberately not configurable: with --enable-pipeline
    # this process will run repository code on request. Do not add a --host flag.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.enable_pipeline))
    url = f"http://localhost:{args.port}/"
    print(f"Serving {DOCS_DIR} at {url} (Ctrl+C to stop)")
    if args.enable_pipeline:
        print("Pipeline API enabled at /api/pipeline/* (loopback only)")
        import pipeline_service
        atexit.register(pipeline_service.shutdown)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(sig)

            def handler(signum, frame, _previous=previous):
                pipeline_service.shutdown()
                if callable(_previous):
                    _previous(signum, frame)
                else:
                    raise KeyboardInterrupt

            signal.signal(sig, handler)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
