"""The single entry point for POIs: OSM ∪ Overture Places, per country.

Every consumer (proximity join, school isochrone origins, influence-zone
fallback, map layer) reads POIs through `load_pois`, never a source file.
A new source is one more `PoiSource` implementation passed to `union_pois`;
no consumer changes.

Common schema (EPSG:4326):
  source, source_id, is_school, is_hospital, is_marketplace, is_shop,
  is_bus_stop, confidence (NaN for OSM), geometry

An Overture place is kept when its confidence is at or above
`overture_min_confidence` OR it is in the top `overture_top_percent` of its
country x category (poi_params.PoiParams). An Overture point with a
same-category OSM feature within DEDUPE_RADIUS_M (distance to a polygon is to
its edge, 0 inside) is the same place and is dropped; OSM is kept as-is. The
radius is small on purpose: two schools can stand 50 m apart, and each needs
its own isochrone.

The union depends only on the two confidence parameters; the speed caps and
isochrone minutes never change which POIs are in it.

The default-parameter union is cached to data/processed/pois_{country}.parquet
and rebuilt when an input is newer; any other parameter set is computed in
memory.

Usage (repo root)
-----
    .venv/bin/python src/poi_sources.py thailand maharashtra
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poi_categories import OSM_BOOL_COLS  # noqa: E402
from poi_params import DEFAULT, PoiParams  # noqa: E402

PROCESSED_DIR = Path("data/processed")
# Dominant UTM zone per country; the pipeline-wide metric CRS.
BUFFER_CRS = {"thailand": "EPSG:32647", "maharashtra": "EPSG:32643"}
DEDUPE_RADIUS_M = 10
COLUMNS = ["source", "source_id", *OSM_BOOL_COLS, "confidence", "geometry"]


class PoiSource(Protocol):
    name: str

    def load(self, country: str) -> gpd.GeoDataFrame:
        """POIs in the common schema."""


def poi_ids(pois: pd.DataFrame) -> pd.Series:
    """The stable id shared by POIs and their isochrones: "<source>:<source_id>"."""
    return pois["source"] + ":" + pois["source_id"]


class OsmPoiSource:
    name = "osm"

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir = Path(processed_dir)

    def path(self, country: str) -> Path:
        return self.processed_dir / f"osm_pois_{country}.parquet"

    def load(self, country: str) -> gpd.GeoDataFrame:
        raw = gpd.read_parquet(self.path(country))
        return self.tag(raw).reset_index(drop=True)

    @staticmethod
    def tag(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Common-schema rows for raw's POI rows, keeping raw's index."""
        out = gpd.GeoDataFrame(
            {
                "source": "osm",
                "source_id": raw["osm_type"].astype(str) + "/" + raw["id"].astype(str)
                if "osm_type" in raw.columns else raw["id"].astype(str),
                "is_school": raw["amenity"] == "school",
                "is_hospital": raw["amenity"] == "hospital",
                "is_marketplace": raw["amenity"] == "marketplace",
                "is_shop": raw["shop"].notna(),
                "is_bus_stop": raw["highway"] == "bus_stop",
                "confidence": np.nan,
            },
            geometry=raw.geometry,
            crs=raw.crs,
        )
        return out.loc[out[OSM_BOOL_COLS].any(axis=1), COLUMNS]


class OverturePoiSource:
    name = "overture"

    def __init__(self, min_confidence: float = DEFAULT.overture_min_confidence,
                 top_percent: float = DEFAULT.overture_top_percent,
                 processed_dir: Path = PROCESSED_DIR):
        self.min_confidence = min_confidence
        self.top_percent = top_percent
        self.processed_dir = Path(processed_dir)

    def path(self, country: str) -> Path:
        return self.processed_dir / f"overture_pois_{country}.parquet"

    def load_all(self, country: str) -> gpd.GeoDataFrame:
        """Every fetched Overture POI, before the confidence rule."""
        return self.tag(gpd.read_parquet(self.path(country)))

    def load(self, country: str) -> gpd.GeoDataFrame:
        return self.filter_confidence(self.load_all(country))

    @staticmethod
    def tag(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        out = gpd.GeoDataFrame(
            {
                "source": "overture",
                "source_id": raw["id"].astype(str),
                **{c: raw[c].astype(bool) if c in raw.columns else False for c in OSM_BOOL_COLS},
                "confidence": raw["confidence"].astype(float),
            },
            geometry=raw.geometry.values,
            crs=raw.crs,
        )
        return out[COLUMNS]

    def filter_confidence(self, pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Keep, per category, rows with confidence >= min_confidence OR within
        that category's top `top_percent`. Missing confidence is never kept."""
        keep = pd.Series(False, index=pois.index)
        thresholds = {}
        for col in OSM_BOOL_COLS:
            in_cat = pois[col] & pois["confidence"].notna()
            if not in_cat.any():
                continue
            t = float(pois.loc[in_cat, "confidence"].quantile(1 - self.top_percent / 100))
            thresholds[col] = t
            keep |= in_cat & ((pois["confidence"] >= self.min_confidence) | (pois["confidence"] >= t))
        out = pois[keep].reset_index(drop=True)
        out.attrs["top_percent_thresholds"] = thresholds
        return out


def union_pois(
    base: gpd.GeoDataFrame,
    extra: gpd.GeoDataFrame,
    metric_crs: str,
    radius_m: float = DEDUPE_RADIUS_M,
) -> gpd.GeoDataFrame:
    """base ∪ extra, dropping extra rows with a same-category base feature within radius_m."""
    extra_m = extra.to_crs(metric_crs)
    base_m = base.to_crs(metric_crs)
    duplicate = pd.Series(False, index=extra.index)
    for col in OSM_BOOL_COLS:
        e = extra_m[extra_m[col]]
        b = base_m[base_m[col]]
        if e.empty or b.empty:
            continue
        hits = gpd.sjoin(e[["geometry"]], b[["geometry"]], predicate="dwithin", distance=radius_m)
        duplicate[hits.index.unique()] = True
    kept = extra[~duplicate]
    if base.empty:
        return kept.reset_index(drop=True)
    if kept.empty:
        return base.reset_index(drop=True)
    return gpd.GeoDataFrame(pd.concat([base, kept.to_crs(base.crs)], ignore_index=True), crs=base.crs)


def _overture(params: PoiParams, processed_dir: Path) -> OverturePoiSource:
    return OverturePoiSource(params.overture_min_confidence, params.overture_top_percent, processed_dir)


def build_union(country: str, params: PoiParams = DEFAULT,
                processed_dir: Path = PROCESSED_DIR) -> tuple[gpd.GeoDataFrame, dict]:
    base = OsmPoiSource(processed_dir).load(country)
    extra = _overture(params, Path(processed_dir)).load(country)
    union = union_pois(base, extra, BUFFER_CRS[country])
    sources = {"osm": base, "overture": extra}
    meta = {
        "country": country,
        "overture_min_confidence": params.overture_min_confidence,
        "overture_top_percent": params.overture_top_percent,
        "overture_top_percent_thresholds": extra.attrs.get("top_percent_thresholds", {}),
        "dedupe_radius_m": DEDUPE_RADIUS_M,
        "n_rows": int(len(union)),
        "n_by_source_before_union": {k: {c: int(v[c].sum()) for c in OSM_BOOL_COLS} for k, v in sources.items()},
        "n_by_source_in_union": {
            k: {c: int(union.loc[union["source"] == k, c].sum()) for c in OSM_BOOL_COLS} for k in sources
        },
    }
    return union, meta


def cache_path(country: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    return Path(processed_dir) / f"pois_{country}.parquet"


def build_and_cache(country: str, processed_dir: Path = PROCESSED_DIR) -> gpd.GeoDataFrame:
    """Write the default-parameter union (the one committed with the repository)."""
    union, meta = build_union(country, DEFAULT, processed_dir)
    out = cache_path(country, processed_dir)
    union.to_parquet(out)
    out.with_name(f"pois_{country}_meta.json").write_text(json.dumps(meta, indent=2))
    _load_cached.cache_clear()
    print(f"[{country}] wrote {out}: {json.dumps(meta['n_by_source_in_union'])}", flush=True)
    return union


def _inputs(country: str, processed_dir: Path) -> list[Path]:
    return [OsmPoiSource(processed_dir).path(country), OverturePoiSource(processed_dir=processed_dir).path(country)]


def is_stale(country: str, processed_dir: Path = PROCESSED_DIR) -> bool:
    out = cache_path(country, processed_dir)
    if not out.exists():
        return True
    built = out.stat().st_mtime
    return any(p.exists() and p.stat().st_mtime > built for p in _inputs(country, processed_dir))


def _require_overture(country: str, processed_dir: Path) -> None:
    path = OverturePoiSource(processed_dir=processed_dir).path(country)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python src/fetch_overture_pois.py {country}`")


def _confidence_key(params: PoiParams) -> tuple[float, float]:
    return params.overture_min_confidence, params.overture_top_percent


@functools.lru_cache(maxsize=None)
def _load_cached(country: str, processed_dir: Path, confidence: tuple[float, float]) -> gpd.GeoDataFrame:
    _require_overture(country, processed_dir)
    if confidence != _confidence_key(DEFAULT):
        params = PoiParams(overture_min_confidence=confidence[0], overture_top_percent=confidence[1])
        return build_union(country, params, processed_dir)[0]
    if is_stale(country, processed_dir):
        build_and_cache(country, processed_dir=processed_dir)
    return gpd.read_parquet(cache_path(country, processed_dir))


def load_pois(country: str, params: PoiParams = DEFAULT, processed_dir: Path = PROCESSED_DIR) -> gpd.GeoDataFrame:
    """The POI union under `params`; the default one is read from its cache."""
    return _load_cached(country, Path(processed_dir), _confidence_key(params)).copy()


def poi_candidates(country: str, poi_type: str, processed_dir: Path = PROCESSED_DIR) -> gpd.GeoDataFrame:
    """Every POI of `poi_type` any parameter set could select: all OSM ones plus
    every fetched Overture one, before the confidence rule and the dedupe.
    Isochrones are built for this set once, so a changed confidence rule never
    needs Valhalla."""
    processed_dir = Path(processed_dir)
    _require_overture(country, processed_dir)
    col = f"is_{poi_type}"
    osm = OsmPoiSource(processed_dir).load(country)
    overture = OverturePoiSource(processed_dir=processed_dir).load_all(country)
    both = pd.concat([osm[osm[col]], overture[overture[col]].to_crs(osm.crs)], ignore_index=True)
    return gpd.GeoDataFrame(both, geometry="geometry", crs=osm.crs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("countries", nargs="+", choices=sorted(BUFFER_CRS))
    args = parser.parse_args(argv)
    for country in args.countries:
        build_and_cache(country)


if __name__ == "__main__":
    main()
