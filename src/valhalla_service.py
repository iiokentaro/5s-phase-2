"""Start and stop the Valhalla container that serves one country's isochrones.

valhalla/docker-compose.yml has one profile per country, both published on
port 8003, so only one country can be served at a time. `serving(country)`
makes port 8003 answer for `country` for the duration of a with-block:

  1. If port 8003 already answers for `country` (someone started it by hand),
     use it as-is and leave it running afterwards.
  2. Otherwise stop the other country's profile, `docker compose --profile
     <country> up -d`, and wait until /status answers and /locate finds a
     road next to one of that country's POIs. The first start builds the
     routing tiles from the full .osm.pbf (30 min to 2 hours per country).
  3. On leaving the block, `docker compose --profile <country> down`, also
     when the block raised.

Anything that stops this (no docker, daemon down, the wait timing out) raises
RuntimeError: an isochrone that cannot be built must fail the run.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_DIR = REPO_ROOT / "valhalla"
BASE_URL = "http://localhost:8003"
COUNTRIES = ("thailand", "maharashtra")
# The first start builds the routing tiles; Thailand took up to 1.5 hours.
READY_TIMEOUT_S = 3 * 3600
POLL_INTERVAL_S = 15.0
LOG_EVERY_S = 300.0
HTTP_TIMEOUT_S = 5.0


def _compose(profile: str, *args: str) -> list[str]:
    return ["docker", "compose", "--profile", profile, *args]


def _run(cmd: list[str], log: Callable[[str], None]) -> None:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=COMPOSE_DIR, capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip():
            log("  " + line)
    if proc.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed with exit code {proc.returncode}")


def docker_problem() -> str | None:
    """Why docker cannot be used here, or None."""
    if shutil.which("docker") is None:
        return "docker is not installed (install Docker Desktop)"
    proc = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return "the Docker daemon is not running (start Docker Desktop)"
    return None


def serves(probe: tuple[float, float], base_url: str = BASE_URL) -> bool:
    """Whether the Valhalla at base_url has a routable edge near probe (lon, lat)."""
    lon, lat = probe
    try:
        resp = requests.post(f"{base_url}/locate",
                             json={"locations": [{"lat": lat, "lon": lon}], "costing": "pedestrian"},
                             timeout=HTTP_TIMEOUT_S)
        if resp.status_code != 200:
            return False
        body = resp.json()
        return bool(body and body[0].get("edges"))
    except (requests.RequestException, ValueError, IndexError, AttributeError):
        return False


def _status_ok(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/status", timeout=HTTP_TIMEOUT_S).status_code == 200
    except requests.RequestException:
        return False


def _wait_ready(country: str, probe: tuple[float, float], log: Callable[[str], None],
                base_url: str, timeout_s: float) -> None:
    started = time.monotonic()
    last_log = started
    while True:
        if _status_ok(base_url) and serves(probe, base_url):
            log(f"[{country}] Valhalla ready after {int(time.monotonic() - started)} s")
            return
        now = time.monotonic()
        if now - started > timeout_s:
            raise RuntimeError(
                f"Valhalla for {country} was not ready after {int(timeout_s)} s "
                f"(check `docker compose --profile {country} logs` in valhalla/)"
            )
        if now - last_log >= LOG_EVERY_S:
            log(f"[{country}] waiting for Valhalla ({int(now - started)} s; "
                "the first start builds the routing tiles)")
            last_log = now
        time.sleep(POLL_INTERVAL_S)


@contextmanager
def serving(country: str, probe: tuple[float, float], log: Callable[[str], None] = print,
            base_url: str = BASE_URL, timeout_s: float = READY_TIMEOUT_S):
    """Port 8003 answers for `country` inside the block. `probe` is a (lon, lat)
    in that country next to a road, such as one of its POIs."""
    if serves(probe, base_url):
        log(f"[{country}] Valhalla already serving on {base_url}; using it as-is")
        yield
        return

    problem = docker_problem()
    if problem:
        raise RuntimeError(f"cannot start Valhalla for {country}: {problem}")
    for other in COUNTRIES:
        if other != country:
            _run(_compose(other, "down"), log)
    _run(_compose(country, "up", "-d"), log)
    try:
        _wait_ready(country, probe, log, base_url, timeout_s)
        yield
    finally:
        _run(_compose(country, "down"), log)
