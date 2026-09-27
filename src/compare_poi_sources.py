"""Measure how much Overture Places POIs overlap the OSM POIs, per category.

Both sides are cut to a common area first: OSM (cropped to the load_target()
bbox at extraction) is kept inside the region polygon, and Overture (cropped
to the region polygon) is kept inside the load_target() bbox.

A POI counts as matched when the nearest POI of the same category on the
other side lies within d metres. Distance to an OSM polygon (e.g. a school
campus) is distance to its edge, 0 inside. As an auxiliary check, each
Overture->OSM pair gets a name similarity score (difflib ratio over the OSM
name / name:en / name:th against Overture names.primary).

Usage (repo root)
-----
    .venv/bin/python src/compare_poi_sources.py thailand maharashtra
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_overture_pois import BOUNDARIES, normalize_name  # noqa: E402
from poi_sources import BUFFER_CRS, OsmPoiSource  # noqa: E402
from schema import load_target  # noqa: E402

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")
DISTANCES_M = [25, 50, 100, 200]
CONFIDENCE_MIN = [0.0, 0.5, 0.6, 0.7]
NAME_MATCH_MIN = 0.6
CATEGORIES = ["is_school", "is_hospital", "is_marketplace", "is_shop"]
OSM_NAME_KEYS = ("name", "name:en", "name:th")


def _osm_names(tags: str) -> list[str]:
    try:
        d = json.loads(tags) if isinstance(tags, str) else {}
    except json.JSONDecodeError:
        return []
    return [n for n in (normalize_name(d.get(k)) for k in OSM_NAME_KEYS) if n]


def _name_similarity(ov_name: str, osm_names: list[str]) -> float:
    a = normalize_name(ov_name)
    if not a or not osm_names:
        return float("nan")
    return max(difflib.SequenceMatcher(None, a, b).ratio() for b in osm_names)


def load_common_area(country: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    region = gpd.read_file(BOUNDARIES[country]).to_crs("EPSG:4326").geometry.union_all()
    shapely.prepare(region)
    target = load_target()
    minx, miny, maxx, maxy = target[target["country"] == country].total_bounds

    raw = gpd.read_parquet(PROCESSED_DIR / f"osm_pois_{country}.parquet")
    osm = OsmPoiSource.tag(raw)
    osm = osm[~osm["is_bus_stop"] | osm[["is_school", "is_hospital", "is_marketplace", "is_shop"]].any(axis=1)]
    osm["tags"] = raw.loc[osm.index, "tags"]
    osm = osm[shapely.intersects(osm.geometry.values, region)]

    ov = gpd.read_parquet(PROCESSED_DIR / f"overture_pois_{country}.parquet")
    ov = ov.cx[minx:maxx, miny:maxy]

    crs = BUFFER_CRS[country]
    osm = osm.to_crs(crs).reset_index(drop=True)
    ov = ov.to_crs(crs).reset_index(drop=True)
    osm["osm_names"] = osm["tags"].map(_osm_names)
    return osm, ov


def _nearest(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per left row: distance to and index of the nearest right row within max(DISTANCES_M)."""
    j = gpd.sjoin_nearest(
        left[["geometry"]], right[["geometry"]], how="left",
        max_distance=max(DISTANCES_M), distance_col="dist",
    )
    j = j[~j.index.duplicated(keep="first")]
    return j[["index_right", "dist"]]


def compare(country: str) -> pd.DataFrame:
    osm_all, ov_all = load_common_area(country)
    subsets = {c: (osm_all[osm_all[c]], ov_all[ov_all[c]]) for c in CATEGORIES}
    school_ov = ov_all[ov_all["is_school"] & (ov_all["taxonomy_primary"] != "preschool")]
    subsets["is_school_excl_preschool"] = (osm_all[osm_all["is_school"]], school_ov)

    rows = []
    for cat, (osm, ov_cat) in subsets.items():
        for cmin in CONFIDENCE_MIN:
            ov = ov_cat[ov_cat["confidence"].fillna(0) >= cmin]
            if ov.empty or osm.empty:
                continue
            ov_near = _nearest(ov, osm)
            osm_near = _nearest(osm, ov)
            has_pair = ov_near["index_right"].notna()
            sim = pd.Series(float("nan"), index=ov_near.index)
            sim[has_pair] = [
                _name_similarity(ov.at[i, "name"], osm.at[int(r), "osm_names"])
                for i, r in ov_near.loc[has_pair, "index_right"].items()
            ]
            for d in DISTANCES_M:
                ov_hit = ov_near["dist"] <= d
                osm_hit = osm_near["dist"] <= d
                ov_hit_named = ov_hit & (sim >= NAME_MATCH_MIN)
                n_osm, n_ov = len(osm), len(ov)
                n_ov_new = int((~ov_hit).sum())
                rows.append({
                    "country": country,
                    "category": cat,
                    "confidence_min": cmin,
                    "distance_m": d,
                    "n_osm": n_osm,
                    "n_overture": n_ov,
                    "overture_matched_pct": round(100 * ov_hit.sum() / n_ov, 1),
                    "osm_matched_pct": round(100 * osm_hit.sum() / n_osm, 1),
                    "overture_matched_named_pct": round(100 * ov_hit_named.sum() / n_ov, 1),
                    "n_union_est": n_osm + n_ov_new,
                    "union_vs_osm_x": round((n_osm + n_ov_new) / n_osm, 2),
                })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("countries", nargs="+", choices=sorted(BOUNDARIES))
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    for country in args.countries:
        df = compare(country)
        out = OUTPUT_DIR / f"poi_overlap_{country}.csv"
        df.to_csv(out, index=False)
        print(f"\n=== {country} -> {out}")
        print(df.drop(columns=["country"]).to_string(index=False))


if __name__ == "__main__":
    main()
