"""Persistence for the OSM way layer: read the cache, or build it and verify.

Extraction, the source contract and the conservation check live in
osm_way_source; this module decides where the layer is written and whether
what is on disk is still current.

The extract keeps every highway class. Which classes vote on which flag is
decided by the consumers, from the `highway` column.

★ Clipped to the administrative boundary PLUS a margin ★
  `western-zone-260621.osm.pbf` -- Maharashtra's source file -- covers
  (67.30, 14.37, 80.91, 25.45): an India-west regional extract. Built from
  the whole file, the layer would hold 2,977 "Maharashtra" motorway ways, 376
  of them the Ahmedabad-Dholera Expressway, which is wholly in Gujarat.

  Clipping to the bare state outline overcorrects. 28 Maharashtra ADB segments
  cross the state line, with vertices up to 1,221 m outside it, and clipping to
  the outline exactly drops 29 matched edges across 6 segments (measured).
  So the clip is the outline grown by osm_way_source.CLIP_MARGIN_M, which is
  chosen to cover that excursion plus segment_way_match's 5 m tolerance.

  "Which ways are this state's" is answered by the unbuffered outline,
  applied in road_class_coverage.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, "src")
from osm_way_source import (BOUNDARY_PATHS, WAY_COLUMNS, assert_conservation,
                            extract_ways, highway_pbf)

PROCESSED_DIR = Path("data/processed")


def ways_path(country: str) -> Path:
    return PROCESSED_DIR / f"osm_ways_{country}.parquet"


def meta_path(country: str) -> Path:
    return PROCESSED_DIR / f"osm_ways_{country}.meta.json"


def _fingerprint(pbf_path: str | Path, source: str,
                 clip_path: str | Path | None) -> dict:
    """Written after the parquet, so a crash between the two costs a rebuild.

    The boundary file's own modification time is part of the fingerprint, so
    replacing the boundary invalidates the cache.
    """
    st = Path(pbf_path).stat()
    clip = None
    if clip_path is not None:
        cst = Path(clip_path).stat()
        clip = {"path": str(clip_path), "size": cst.st_size, "mtime_ns": cst.st_mtime_ns}
    return {
        "source_pbf": str(pbf_path),
        "source_size": st.st_size,
        "source_mtime_ns": st.st_mtime_ns,
        "way_source": source,
        "columns": WAY_COLUMNS,
        "clip": clip,
    }


def is_fresh(country: str, pbf_path: str | Path, source: str,
            clip_path: str | Path | None) -> bool:
    if not (ways_path(country).exists() and meta_path(country).exists()):
        return False
    try:
        have = json.loads(meta_path(country).read_text())
    except ValueError:
        return False
    return all(have.get(k) == v for k, v in _fingerprint(pbf_path, source, clip_path).items())


def cache_ways(country: str, pbf_path: str | Path | None = None, *,
               source: str = "osmium", force: bool = False) -> gpd.GeoDataFrame:
    from exposure_signals import PBF_PATHS

    pbf_path = pbf_path or PBF_PATHS[country]
    clip_path = BOUNDARY_PATHS[country]
    if not force and is_fresh(country, pbf_path, source, clip_path):
        return gpd.read_parquet(ways_path(country))

    t0 = time.time()
    ways = extract_ways(pbf_path, source=source, clip_path=clip_path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ways.to_parquet(ways_path(country))
    meta_path(country).write_text(json.dumps(
        {**_fingerprint(pbf_path, source, clip_path), "rows": len(ways),
         "elapsed_s": round(time.time() - t0, 1)}, indent=2))
    print(f"[osm_ways] {country}: {len(ways):,} ways -> {ways_path(country)} "
          f"({time.time() - t0:.0f}s)")
    return ways


def load_ways(country: str, *, verify: bool = False) -> gpd.GeoDataFrame:
    """The pipeline's entry point. `verify=True` re-runs the conservation check
    against the source .pbf -- cheap relative to a build, and the only way to
    notice that a committed parquet predates a fix."""
    path = ways_path(country)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it with:\n"
            f"    python src/osm_ways.py --country {country}"
        )
    ways = gpd.read_parquet(path)
    if verify:
        from exposure_signals import PBF_PATHS

        assert_conservation(
            ways, highway_pbf(PBF_PATHS[country], clip_path=BOUNDARY_PATHS[country]),
            source_name="cache")
    return ways


if __name__ == "__main__":
    import argparse
    import warnings

    from exposure_signals import PBF_PATHS

    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description="Build the OSM way layer for a country.")
    ap.add_argument("--country", choices=[*PBF_PATHS, "both"], default="both")
    ap.add_argument("--source", default="osmium", choices=["osmium", "pyrosm"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for country in ([*PBF_PATHS] if args.country == "both" else [args.country]):
        ways = cache_ways(country, source=args.source, force=args.force)
        n = ways["highway"].value_counts()
        print(f"  top classes: {n.head(8).to_dict()}")
