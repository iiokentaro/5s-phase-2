"""Build docs/segments_priority.pmtiles from outputs/segments_priority.parquet.

The web map reads this PMTiles archive; rebuilding it is how a pipeline run
refreshes what the browser shows. Only the properties the map reads go into it
(INCLUDED_PROPERTIES). tippecanoe is an external binary; when it is absent this
module reports a skip reason, so the rest of a run still completes. tippecanoe
reads GeoJSON, so a parquet input is converted to a temporary GeoJSON file
first.

Usage
-----
    python src/build_tiles.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = REPO_ROOT / "outputs" / "segments_priority.parquet"
PMTILES_PATH = REPO_ROOT / "docs" / "segments_priority.pmtiles"
LAYER = "segments"
PMTILES_MAGIC = b"PMTiles"

# The properties docs/index.html reads. Every other column stays in
# outputs/segments_priority.parquet. A column added to the parquet enters
# the tiles once it is added to this list: each property kept is paid for once
# per feature (about 100k of them), and tippecanoe answers an over-budget tile by
# dropping features from the map. tests/test_build_tiles.py checks this list
# against what docs/index.html actually reads, in both directions.
#
# The three categorical values folded into score_explanation's text travel as
# the *_code integers safety_score.py writes (exposure_level_code,
# confidence_level_code); a repeated small int costs far less per feature than
# a repeated word, and index.html decodes them back to labels.
INCLUDED_PROPERTIES = [
    # styling and filters
    "priority_class",
    "misalignment",
    "review_track",
    # popup
    "segment_id",
    "score_explanation",
    "speed_limit",
    "v_safe",
    "v_safe_basis",
    "exposure_level_code",
    "confidence_level_code",
    "speedlimit_plausibility",
    "road_class",
    "land_use",
    "exp_delta_fatal_percent_uniform",
    "country",
]

# -zg guesses maxzoom 10, where a hand-picked -z14 produces 61 MB of detail
# the map never renders. --no-progress-indicator only silences the \r
# percentage lines (which would otherwise arrive as thousands of SSE log lines).
#
# MAX_TILE_BYTES: tippecanoe's default budget is 500,000 bytes per tile, and
# --drop-densest-as-needed pays for an over-budget tile by deleting features
# from it. With every parquet column in the tiles, the low-zoom tiles were cut
# to 12.8-15.2% of their features at the default, and the largest tile
# measured 2,327,792 bytes (zoom 4), so 2.5 MB let every tile through intact.
# With INCLUDED_PROPERTIES alone (then 14 properties) the largest tile measured
# 683,747 bytes, still above the default, so the budget stays as headroom. The
# cost is per-tile download and parse time in the browser. With the 15
# properties the archive is 16.1 MB.
MAX_TILE_BYTES = 2_500_000

TIPPECANOE_ARGS = [
    "-zg",
    *[arg for prop in INCLUDED_PROPERTIES for arg in ("-y", prop)],
    "-l", LAYER,
    "--maximum-tile-bytes", str(MAX_TILE_BYTES),
    "--drop-densest-as-needed",
    "--extend-zooms-if-still-dropping",
    "--no-progress-indicator",
    "--force",
]


def check() -> str | None:
    """Return a human-readable skip reason, or None when the step can run."""
    if shutil.which("tippecanoe") is None:
        return "tippecanoe not installed (`brew install tippecanoe`)"
    if not PARQUET_PATH.exists():
        return f"{PARQUET_PATH.relative_to(REPO_ROOT)} missing (run write_geo_outputs first)"
    return None


def _parquet_to_geojson(parquet_path: Path) -> Path:
    """Write a throwaway GeoJSON tippecanoe can read, in the pmtiles directory
    so the later os.replace() (in build_tiles) stays on one filesystem."""
    import geopandas as gpd

    gdf = gpd.read_parquet(parquet_path)
    fd, tmp_name = tempfile.mkstemp(suffix=".geojson", dir=PMTILES_PATH.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    gdf.to_file(tmp_path, driver="GeoJSON")
    return tmp_path


def build_tiles(input_path: str | Path = PARQUET_PATH,
                output_path: str | Path = PMTILES_PATH,
                log=print) -> str:
    """Run tippecanoe and atomically replace the served archive.

    The server may be streaming Range requests from the existing .pmtiles while
    this runs, so tippecanoe writes to a dot-prefixed temp file in the same
    directory and os.replace() swaps it in -- an in-flight range read never sees
    a half-written archive.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    tmp_geojson = _parquet_to_geojson(input_path) if input_path.suffix == ".parquet" else None
    tippecanoe_input = tmp_geojson or input_path

    # The temp name must keep the .pmtiles suffix: tippecanoe picks its output
    # format from the extension, and anything else (".tmp") silently gets an
    # MBTiles/SQLite archive instead -- which the map then serves as PMTiles and
    # renders as an empty layer.
    tmp_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}{output_path.suffix}")

    cmd = ["tippecanoe", "-o", str(tmp_path), *TIPPECANOE_ARGS, str(tippecanoe_input)]

    log(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            log(line.rstrip())
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"tippecanoe exited {code}")
        magic = tmp_path.read_bytes()[:len(PMTILES_MAGIC)]
        if magic != PMTILES_MAGIC:
            raise RuntimeError(
                f"tippecanoe wrote {magic!r}, not a PMTiles archive -- "
                "the output path must end in .pmtiles"
            )
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        if tmp_geojson is not None and tmp_geojson.exists():
            tmp_geojson.unlink()

    size = output_path.stat().st_size
    log(f"saved {output_path} ({size / 1e6:.1f} MB)")
    log("note: this file is tracked by git -- commit it to publish the updated map")
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(PARQUET_PATH))
    parser.add_argument("--output", default=str(PMTILES_PATH))
    args = parser.parse_args()

    reason = check()
    if reason:
        print(f"skipped: {reason}")
        return 1
    build_tiles(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
