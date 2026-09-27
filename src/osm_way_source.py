"""The single seam between this pipeline and whatever reads a .osm.pbf.

Why this module exists
----------------------
By pyrosm's behaviour, `pyrosm 0.9.1`'s `get_data_by_custom_criteria` returns
ONLY ways whose node count is exactly 2; ways with three or more nodes are
left out.
Measured against osmium on one Bangkok extract: 21,972 of 62,246 highway ways
returned (35.3% of ways, 791.9 km of 6,324.2 km = 12.5% by length). The ways it
does return are geometrically perfect, so nothing downstream looks wrong.

`verify_prefilter_equivalence.py` compares pyrosm output against pyrosm
output, so it cannot see this: self-consistency cannot detect what the thing
being compared with itself leaves out.

So this module holds two things that sit outside every extraction
implementation:

  * `WaySource`, a port with more than one implementation, so two independent
    readers can be diffed against each other (tests/test_osm_way_source_equivalence.py).
    Running two implementations side by side is what actually found the bug.
  * `assert_conservation`, which compares the extract against the SOURCE FILE's
    own declared counts. It runs on every extraction.

Adding a third implementation means implementing `WaySource` and adding it to
the equivalence test. Nothing else in the pipeline learns its name.
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import geopandas as gpd
import numpy as np
import shapely

OSMIUM_BIN = "osmium"
OSMIUM_TIMEOUT = 1800

# Tags promoted to their own column. Everything the pipeline branches on lives
# here, so no consumer has to parse the catch-all JSON on a hot path.
#
# ★ Promoting a tag MOVES it out of the catch-all `tags` dict ★
#   A consumer that read `tags` alone would see every promoted tag as absent --
#   for bridge/tunnel/layer, silently and in the unsafe direction, since
#   is_grade_separated exempts a segment from junction_speed_cap's 50 km/h cap.
#   Read through road_separation._tag_value, which checks the column first and
#   falls back to `tags`.
PROMOTED_TAGS = [
    "highway",
    "oneway",
    "motorroad",
    "bridge",
    "tunnel",
    "layer",
    "lanes:divided",
    "dual_carriageway",
    "foot",
    "access",
    "bicycle",
    "junction",
    "sidewalk",
    "cycleway",
    "service",
]

WAY_COLUMNS = ["osm_way_id", *PROMOTED_TAGS, "tags", "geometry"]

# --- conservation thresholds, calibrated on measured data --------------------
# `osmium export --geometry-types=linestring` drops closed ways tagged area=yes,
# so the LineString count sits just below fileinfo's way count: 62,246 vs 62,396
# on the Bangkok extract, a 0.24% gap. Anything below this floor means ways went
# missing for some other reason.
MIN_WAY_RATIO = 0.98
# Distinct vertex coordinates against fileinfo's referenced-node count: 245,366
# vs 247,420 on the same extract (0.99). The shortfall is nodes belonging only
# to the area ways above; the ceiling allows for two nodes sharing coordinates.
MIN_NODE_RATIO = 0.95
MAX_NODE_RATIO = 1.02
# The pattern of pyrosm's get_data_by_custom_criteria output. Real highway networks are nowhere near this two-node
# heavy: 35.3% on the Bangkok extract, 28.1% across Thailand (820,686 of
# 2,916,345). pyrosm scored 100%. A source that trips this is returning a
# node-pair subset, not the network, and the run must stop.
MAX_TWO_VERTEX_RATIO = 0.50


@runtime_checkable
class WaySource(Protocol):
    """Reads every `highway=*` way out of a .osm.pbf, with full geometry."""

    @property
    def name(self) -> str: ...

    def extract(self, pbf_path: str | Path) -> gpd.GeoDataFrame:
        """Returns WAY_COLUMNS in EPSG:4326, one row per OSM way, LineString
        only, with every node of the way present in the geometry."""
        ...


class ConservationError(AssertionError):
    """The extract does not account for what the source file declares."""


# --------------------------------------------------------------------------- #
# The oracle: the source file's own declared counts
# --------------------------------------------------------------------------- #

# Administrative boundaries. Provenance is recorded beside each file in
# data/raw/ (geoBoundaries ADM1 for Maharashtra, an ADM0 country outline for
# Thailand).
#
# ★ The outlines clip the way layer only after growing by CLIP_MARGIN_M ★
#   Two questions are kept apart:
#
#     "which ways must the layer keep?"  -> anything that could be attributed
#        to an ADB segment, which is a question about the segments. Answered
#        by the outline grown by CLIP_MARGIN_M below.
#     "which ways are this state's?"     -> answered by the bare outlines,
#        applied at reporting time in road_class_coverage.py.
#
#   Clipping the layer to the bare outline was measured and it LOSES matches:
#   28 Maharashtra ADB segments cross the state line, with vertices up to
#   1,221 m outside it, and clipping to the outline dropped 29 matched edges
#   across 6 segments. The state line is a fact about administration; whether
#   a road can be attributed to a segment is a fact about geometry.
BOUNDARY_PATHS = {
    "thailand": "data/raw/THAILAND.geojson",
    "maharashtra": "data/raw/INDIA-Maharashtra.geojson",
}

# How far outside the administrative boundary the clip must still reach.
#
#   1,221 m  furthest an ADB segment vertex sits outside Maharashtra's outline
#       5 m  segment_way_match's tolerance_distance_m -- a way that far from a
#            segment can still be attributed to it
#   ------
#   ~1.3 km minimum; 3 km is used, because the cost of being generous is disk
#   space and the cost of being tight is a silently missing match.
CLIP_MARGIN_M = 3000.0


def _clip_tag(clip_path: str | Path, margin_m: float) -> str:
    """A short filename fragment identifying the clip, including the boundary
    file's modification time and the margin, so changing either invalidates the
    cached extract."""
    p = Path(clip_path)
    return f"clip_{p.stem}_{p.stat().st_mtime_ns}_m{int(margin_m)}"


def buffered_boundary(clip_path: str | Path, margin_m: float,
                      out_dir: str | Path | None = None) -> Path:
    """The boundary grown by `margin_m`, written where osmium can read it.

    Buffering happens in a metric CRS and the result is written back in
    EPSG:4326, because osmium reads the polygon in degrees but a margin stated
    in degrees would be a different distance at different latitudes.
    """
    import geopandas as _gpd

    src = Path(clip_path)
    out = Path(out_dir or src.parent) / f"{src.stem}.buffer{int(margin_m)}m.geojson"
    if out.exists() and out.stat().st_mtime_ns >= src.stat().st_mtime_ns:
        return out
    gdf = _gpd.read_file(src)
    metric = gdf.to_crs(gdf.estimate_utm_crs())
    metric["geometry"] = metric.geometry.buffer(margin_m)
    metric.to_crs("EPSG:4326").to_file(out, driver="GeoJSON")
    return out


def highway_pbf(pbf_path: str | Path, out_dir: str | Path | None = None,
                clip_path: str | Path | None = None,
                margin_m: float = CLIP_MARGIN_M) -> Path:
    """`<stem>.highways[.clip_...].osm.pbf` -- the source restricted to highway
    ways (and, if `clip_path` is given, to that boundary polygon) plus the nodes
    they reference. Both the oracle and the osmium reader work from this file,
    so they cannot disagree about what the population is.

    ★ Why a boundary polygon and not a rectangle ★
      Without any clip this reads every highway way in the WHOLE source file.
      `western-zone-260621.osm.pbf` spans (67.30, 14.37, 80.91, 25.45) -- it is
      an India-west regional extract, not a Maharashtra one. Measured without a
      clip, 376 of 2,977 Maharashtra "motorway" ways are the Ahmedabad-Dholera
      Expressway, wholly in Gujarat, plus the Amritsar-Jamnagar (144) and
      Delhi-Vadodara (57) corridors.

      A rectangle around the ADB segments fixes the worst of that but is still
      loose: Maharashtra's bounding rectangle takes in parts of Gujarat, Madhya
      Pradesh, Karnataka and Telangana. Clipping to the ADM1 outline instead
      takes the way count from 981,400 to 862,186 and makes the per-class
      reference figures in road_class_coverage.py mean "this state's roads".

    ★ This does not change any matched result ★
      Measured: a rectangle-clipped extract gives all seven segment_way_match
      statistics byte-identical to the whole-file extract in both countries,
      with 560,695 fewer Maharashtra ways. A way that does not lie on an ADB
      segment never shares a vertex with one and is never within the 5 m
      tolerance of one, so it cannot be attributed. The clip buys runtime, file
      size, and reference statistics that describe the right area.

    `--strategy=complete_ways` (osmium's default) keeps a way whole if any of its
    nodes falls inside the boundary, so WAY_COLUMNS' "every node of the way
    present" contract still holds for a way crossing the state line -- which is
    what lets an ADB segment near the border still match.
    """
    src = Path(pbf_path)
    stem = src.name
    for suffix in (".osm.pbf", ".pbf"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    suffix = f".{_clip_tag(clip_path, margin_m)}" if clip_path is not None else ""
    out = Path(out_dir or src.parent) / f"{stem}.highways{suffix}.osm.pbf"
    if out.exists() and out.stat().st_mtime_ns >= src.stat().st_mtime_ns:
        return out

    # osmium infers the output format from the extension, so the temp file has
    # to keep a real one; `--output-format pbf` states it anyway, because the
    # pid infix would otherwise make the name unrecognisable.
    tag_tmp = out.with_name(f"{out.name}.{os.getpid()}.tags.tmp.osm.pbf")
    try:
        subprocess.run(
            [OSMIUM_BIN, "tags-filter", "--overwrite", "--output-format", "pbf",
             "-o", str(tag_tmp), str(src), "w/highway"],
            check=True, capture_output=True, text=True, timeout=OSMIUM_TIMEOUT,
        )
        if clip_path is None:
            tag_tmp.replace(out)
            return out

        polygon = buffered_boundary(clip_path, margin_m, out_dir=out.parent)
        clip_tmp = out.with_name(f"{out.name}.{os.getpid()}.clip.tmp.osm.pbf")
        try:
            subprocess.run(
                [OSMIUM_BIN, "extract", "--overwrite", "--output-format", "pbf",
                 "--strategy", "complete_ways",
                 "--polygon", str(polygon),
                 "-o", str(clip_tmp), str(tag_tmp)],
                check=True, capture_output=True, text=True, timeout=OSMIUM_TIMEOUT,
            )
            clip_tmp.replace(out)
        finally:
            clip_tmp.unlink(missing_ok=True)
    finally:
        tag_tmp.unlink(missing_ok=True)
    return out


def declared_counts(highway_pbf_path: str | Path) -> dict[str, int]:
    """What the file itself says it holds. This is the oracle -- it is read from
    the .pbf by a different tool than the one that built the extract."""
    res = subprocess.run(
        [OSMIUM_BIN, "fileinfo", "-e", str(highway_pbf_path)],
        check=True, capture_output=True, text=True, timeout=OSMIUM_TIMEOUT,
    )
    counts = {}
    for key, label in (("ways", "Number of ways"), ("nodes", "Number of nodes")):
        m = re.search(rf"{label}:\s*(\d+)", res.stdout)
        if m is None:
            raise ConservationError(f"osmium fileinfo did not report '{label}'")
        counts[key] = int(m.group(1))
    return counts


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #

def assert_conservation(ways: gpd.GeoDataFrame, highway_pbf_path: str | Path,
                        *, source_name: str = "?") -> dict:
    """Raise unless the extract accounts for the source file's own counts.

    Called on every extraction. The point is that it compares against the .pbf,
    never against another extract, so it stays valid when the reader is
    replaced -- which is the failure this whole module exists to prevent.
    """
    declared = declared_counts(highway_pbf_path)
    n_vertices = shapely.get_num_coordinates(ways.geometry.values)
    coords = shapely.get_coordinates(ways.geometry.values)
    distinct = len(np.unique(coords, axis=0)) if len(coords) else 0

    report = {
        "source": source_name,
        "declared_ways": declared["ways"],
        "declared_nodes": declared["nodes"],
        "extracted_ways": int(len(ways)),
        "total_vertices": int(n_vertices.sum()),
        "distinct_vertices": int(distinct),
        "two_vertex_ways": int((n_vertices == 2).sum()),
    }
    report["way_ratio"] = report["extracted_ways"] / max(declared["ways"], 1)
    report["node_ratio"] = report["distinct_vertices"] / max(declared["nodes"], 1)
    report["two_vertex_ratio"] = report["two_vertex_ways"] / max(len(ways), 1)

    problems = []
    if report["way_ratio"] < MIN_WAY_RATIO:
        problems.append(
            f"only {report['extracted_ways']:,} of the {declared['ways']:,} highway ways "
            f"the file declares came back ({report['way_ratio']:.1%}, floor {MIN_WAY_RATIO:.0%}) "
            f"-- ways are being dropped"
        )
    if report["way_ratio"] > 1.0:
        problems.append(
            f"{report['extracted_ways']:,} ways from a file declaring {declared['ways']:,} "
            f"-- the extract is not a subset of the source"
        )
    if not MIN_NODE_RATIO <= report["node_ratio"] <= MAX_NODE_RATIO:
        problems.append(
            f"{report['distinct_vertices']:,} distinct vertices against {declared['nodes']:,} "
            f"referenced nodes ({report['node_ratio']:.2f}, expected "
            f"{MIN_NODE_RATIO}-{MAX_NODE_RATIO}) -- geometry is truncated or duplicated"
        )
    if report["two_vertex_ratio"] > MAX_TWO_VERTEX_RATIO:
        problems.append(
            f"{report['two_vertex_ratio']:.1%} of ways have exactly two vertices "
            f"(ceiling {MAX_TWO_VERTEX_RATIO:.0%}; Thailand's true value is 28.1%) "
            f"-- the pattern of pyrosm get_data_by_custom_criteria output: the reader "
            f"is returning a node-pair subset, not the network"
        )
    if (n_vertices < 2).any():
        problems.append(f"{int((n_vertices < 2).sum()):,} ways have fewer than two vertices")
    dup = int(ways["osm_way_id"].duplicated().sum())
    if dup:
        problems.append(f"{dup:,} duplicate osm_way_id values")

    if problems:
        raise ConservationError(
            f"[{source_name}] extract does not conserve {Path(highway_pbf_path).name}:\n  - "
            + "\n  - ".join(problems)
        )
    return report


def get_source(name: str = "osmium") -> WaySource:
    """The pipeline asks for a source by name and never imports one directly."""
    if name == "osmium":
        from osm_way_source_osmium import OsmiumWaySource

        return OsmiumWaySource()
    if name == "pyrosm":
        from osm_way_source_pyrosm import PyrosmWaySource

        return PyrosmWaySource()
    raise ValueError(f"unknown way source {name!r} (known: osmium, pyrosm)")


def extract_ways(pbf_path: str | Path, *, source: str | WaySource = "osmium",
                 verify: bool = True,
                 clip_path: str | Path | None = None,
                 margin_m: float = CLIP_MARGIN_M) -> gpd.GeoDataFrame:
    """Extract, then prove the extract conserves the file. `verify=False` exists
    only for the equivalence test, which checks a known-deficient reader on
    purpose.

    `clip_path` should be the country's administrative boundary (see
    BOUNDARY_PATHS) for any source .pbf covering a wider area than the study
    population -- see highway_pbf's docstring.
    """
    src = get_source(source) if isinstance(source, str) else source
    hw_pbf = highway_pbf(pbf_path, clip_path=clip_path, margin_m=margin_m)
    ways = src.extract(hw_pbf)
    missing = [c for c in WAY_COLUMNS if c not in ways.columns]
    if missing:
        raise ConservationError(f"[{src.name}] missing contract columns: {missing}")
    if verify:
        assert_conservation(ways, hw_pbf, source_name=src.name)
    return ways[WAY_COLUMNS]


if __name__ == "__main__":
    import argparse
    import json
    import sys

    sys.path.insert(0, "src")
    from exposure_signals import PBF_PATHS

    ap = argparse.ArgumentParser(description="Extract and verify, without caching.")
    ap.add_argument("--country", choices=[*PBF_PATHS, "both"], default="both")
    ap.add_argument("--source", default="osmium", choices=["osmium", "pyrosm"])
    ap.add_argument("--no-clip", action="store_true",
                    help="skip the administrative-boundary clip (debugging only)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    for country in ([*PBF_PATHS] if args.country == "both" else [args.country]):
        src = get_source(args.source)
        clip = None if args.no_clip else BOUNDARY_PATHS[country]
        hw = highway_pbf(PBF_PATHS[country], clip_path=clip)
        ways = src.extract(hw)
        print(f"\n== {country} / {src.name}  clip={clip} ==")
        print(json.dumps(assert_conservation(ways, hw, source_name=src.name), indent=2))
