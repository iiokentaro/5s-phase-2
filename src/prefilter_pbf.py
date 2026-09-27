"""osmium tags-filter pre-filter in front of the pyrosm PBF parse.

Why this exists:
  `exposure_signals.extract_raw` and `road_separation.extract_road_network`
  both hand a country-wide `.osm.pbf` (Thailand 323.7 MB) to pyrosm, which
  must walk the whole file to decide what its `custom_filter` keeps. That took
  ~55 min for Thailand / ~22 min for Maharashtra. `osmium tags-filter` (C++)
  drops the non-matching objects up front in seconds, and pyrosm then parses a
  file one to two orders of magnitude smaller.

★ Superset principle (the correctness argument) ★
  pyrosm re-applies its own `custom_filter` to whatever file it is given. So a
  filtered PBF that is a strict SUPERSET of what pyrosm keeps yields identical
  output (the extras are dropped by pyrosm), while a SUBSET is silent data
  loss. FILTER_EXPRESSIONS below is therefore deliberately generous, and one
  shared filtered file per country covers both consumers.

★ nwr/ is mandatory ★
  The committed baselines contain relation-sourced rows: 863 POI relations in
  Thailand, 26 in Maharashtra, plus one relation each in osm_roads_thailand and
  the pedestrian-way outputs. An `n/`+`w/`-only filter would silently lose them.

★ Never pass -t/--remove-tags, never pass -R/--omit-referenced ★
  `road_separation._is_access_controlled_way` / `_is_divided_way` /
  `_is_grade_separated_way` read `motorroad` / `lanes:divided` /
  `dual_carriageway` / `bridge` / `tunnel` / `layer` out of
  pyrosm's catch-all `tags` JSON column, which only survives because
  tags-filter keeps every tag on a matching object. And default reference
  completion (relation -> member ways -> their nodes) is what gives ways and
  relations their geometry.

★ pyrosm does the bbox crop ★
  osmium's extract strategies clip at the boundary with different semantics
  than pyrosm's `bounding_box=`, which would break the superset guarantee at
  the bbox edge.

★ Hard dependency, by choice ★
  A missing osmium raises, so a 55-minute full parse is always a choice: the
  opt-out is `--no-prefilter` on the two extractor CLIs.

Equivalence against the committed `data/processed/osm_*.parquet` baselines is
verified by `src/verify_prefilter_equivalence.py`.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #
OSMIUM_BIN = "osmium"
FILTER_VERSION = 1              # bump whenever FILTER_EXPRESSIONS changes
FILTERED_SUFFIX = ".5s-filtered.osm.pbf"
META_SUFFIX = ".5s-filtered.meta.json"
OSMIUM_TIMEOUT = 1800           # seconds

INSTALL_HINT = (
    "install it with `brew install osmium-tool` (macOS) or "
    "`apt install osmium-tool` (Debian/Ubuntu), or pass --no-prefilter to parse "
    "the full pbf instead (slow: ~55 min for Thailand)"
)

# Union of both pyrosm consumers' needs, deliberately a superset.
#   exposure_signals.CUSTOM_FILTER : amenity=school/marketplace/hospital,
#                                    shop=*, highway=bus_stop/footway/path/
#                                    pedestrian/steps/crossing
#   road_separation.ROAD_HIGHWAY_VALUES : highway=motorway/trunk/primary/secondary
FILTER_EXPRESSIONS = [
    # --- exposure_signals.CUSTOM_FILTER ---
    "nwr/amenity=school,marketplace,hospital",
    "nwr/shop",  # key-existence match, mirroring pyrosm's `"shop": True`
    "nwr/highway=bus_stop,footway,path,pedestrian,steps,crossing",
    # split_signals also matches footway=crossing. Such an object can only reach
    # crossings_all via the highway= clause above, so this line is not required
    # for equivalence -- it costs nothing and decouples this filter from that
    # implicit coupling.
    "nwr/footway=crossing",
    # --- road_separation.ROAD_HIGHWAY_VALUES (+ _link variants as slack) ---
    "nwr/highway=motorway,trunk,primary,secondary",
    "nwr/highway=motorway_link,trunk_link,primary_link,secondary_link",
]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def _strip_pbf_suffix(source: Path) -> str:
    """'thailand-260621.osm.pbf' -> 'thailand-260621' (handles the double suffix)."""
    name = source.name
    for suffix in (".osm.pbf", ".pbf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return source.stem


def filtered_path(source_pbf: str | Path) -> Path:
    source = Path(source_pbf)
    return source.with_name(_strip_pbf_suffix(source) + FILTERED_SUFFIX)


def meta_path(source_pbf: str | Path) -> Path:
    source = Path(source_pbf)
    return source.with_name(_strip_pbf_suffix(source) + META_SUFFIX)


# --------------------------------------------------------------------------- #
# osmium
# --------------------------------------------------------------------------- #

def _require_osmium() -> str:
    resolved = shutil.which(OSMIUM_BIN)
    if resolved is None:
        raise RuntimeError(f"`{OSMIUM_BIN}` not found on PATH -- {INSTALL_HINT}")
    return resolved


def _osmium_version(binary: str) -> str:
    try:
        out = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
        return out.stdout.splitlines()[0].strip() if out.stdout else "unknown"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #

def _is_fresh(source: Path, dest: Path, meta: Path) -> bool:
    """The cached filtered PBF is reusable only if every recorded fact still holds.

    A missing/unparsable sidecar counts as stale, which is what makes the
    write-then-rename-then-sidecar ordering in build_filtered_pbf safe: a crash
    between the rename and the sidecar write costs a harmless re-filter, never a
    trusted-but-wrong cache. The 324 MB source is checked by stat, which is
    far cheaper than a hash.
    """
    if not dest.exists() or not meta.exists():
        return False
    try:
        recorded = json.loads(meta.read_text())
    except (OSError, ValueError) as exc:
        log.warning("unreadable prefilter sidecar %s (%s) -> re-filtering", meta.name, exc)
        return False

    src_stat = source.stat()
    checks = {
        "source_size": src_stat.st_size,
        "source_mtime_ns": src_stat.st_mtime_ns,
        "filter_version": FILTER_VERSION,
        "expressions": FILTER_EXPRESSIONS,
        "output_size": dest.stat().st_size,
    }
    for key, expected in checks.items():
        if recorded.get(key) != expected:
            log.info("prefilter cache stale (%s changed) -> re-filtering", key)
            return False
    return True


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_filtered_pbf(source_pbf: str | Path) -> Path:
    """Run osmium tags-filter unconditionally and return the filtered path."""
    source = Path(source_pbf)
    dest = filtered_path(source)
    meta = meta_path(source)
    binary = _require_osmium()
    version = _osmium_version(binary)

    # Write to a pid-suffixed temp file and only os.replace() (atomic on the
    # same filesystem) after a clean exit, so a truncated file can never occupy
    # the cache path. The pid suffix also keeps two concurrent runs (e.g.
    # exposure_signals.py and road_separation.py in parallel shells) from
    # corrupting each other's output.
    tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
    cmd = [
        binary, "tags-filter",
        "--overwrite",
        "--output-format", "pbf",
        "--output", str(tmp),
        str(source),
        *FILTER_EXPRESSIONS,
    ]

    log.info("osmium tags-filter: %s (%.1f MB) -> %s",
             source.name, source.stat().st_size / 1e6, dest.name)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=OSMIUM_TIMEOUT, check=False
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"osmium tags-filter failed (rc={proc.returncode}) on {source.name}: "
                f"{(proc.stderr or '').strip()[-2000:]}"
            )
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(
                f"osmium tags-filter produced no output for {source.name} "
                "(exit 0 but the file is missing or empty)"
            )
        os.replace(tmp, dest)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"osmium tags-filter timed out after {OSMIUM_TIMEOUT}s on {source.name}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"could not run osmium tags-filter on {source.name}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    out_size = dest.stat().st_size
    src_stat = source.stat()

    # Sidecar written AFTER the rename -- see _is_fresh.
    meta.write_text(json.dumps({
        "source": str(source),
        "source_size": src_stat.st_size,
        "source_mtime_ns": src_stat.st_mtime_ns,
        "filter_version": FILTER_VERSION,
        "expressions": FILTER_EXPRESSIONS,
        "osmium_version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_size": out_size,
        "elapsed_s": round(elapsed, 2),
    }, indent=2), encoding="utf-8")

    log.info("osmium tags-filter done in %.1fs: %.1f MB -> %.1f MB (%.1f%% of source)",
             elapsed, src_stat.st_size / 1e6, out_size / 1e6,
             100.0 * out_size / src_stat.st_size)
    if out_size >= src_stat.st_size:
        log.warning("filtered pbf is not smaller than the source -- the filter matched "
                    "everything; results stay correct (superset) but nothing was gained")
    return dest


def ensure_filtered_pbf(source_pbf: str | Path, *, force: bool = False) -> str:
    """Return the path to a tag-filtered copy of `source_pbf`, building it if needed.

    Idempotent: an existing filtered PBF whose sidecar still matches the source's
    size/mtime, FILTER_VERSION and FILTER_EXPRESSIONS is reused as-is.

    Raises
    ------
    FileNotFoundError : the source PBF is absent.
    RuntimeError      : osmium is not installed, or the filter run failed.
    """
    source = Path(source_pbf)
    if not source.exists():
        raise FileNotFoundError(f"source pbf not found: {source}")

    dest = filtered_path(source)
    if not force and _is_fresh(source, dest, meta_path(source)):
        log.info("using cached filtered pbf: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return str(dest)

    return str(build_filtered_pbf(source))
