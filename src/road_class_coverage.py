"""Reference measurements of the OSM way layer, per highway class.

`src/road_access_rules.json`'s `source.calibration_basis` justifies each
country diff by a figure in `outputs/road_class_coverage_{country}.json`
(`m1_inventory`), and this script writes that file. It reads
`data/processed/osm_ways_*.parquet`, the layer the pipeline uses, and is run
by hand.

Coverage of the ADB segments by OSM is reported by
`segment_way_match.py --json`, which measures the attribution the pipeline
uses.

A length ratio is not a coverage metric
---------------------------------------
M2 divides OSM class km by ADB network km. It exceeds 100% in both countries
-- 133.6% in Thailand and 284.0% in Maharashtra for the four arterial classes
-- because the .pbf covers a wider area than the ADB study population.
Coverage is a question about correspondence between two datasets, and only a
segment-to-way match answers it.

Usage (from the repo root):
    python src/road_class_coverage.py --country both --json outputs/road_class_coverage.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS, PBF_PATHS
from osm_way_source import BOUNDARY_PATHS
from road_separation import ROAD_HIGHWAY_VALUES, _tag_series
from schema import load_target

warnings.filterwarnings("ignore", category=UserWarning)

# Counted by M7. Every key here is a promoted column on the osm_ways layer, so
# `_tag_series` reads it without parsing the catch-all JSON.
TAG_PROBES = {
    "motorroad=yes": ("motorroad", {"yes"}),
    "foot=no": ("foot", {"no"}),
    "access=no|private": ("access", {"no", "private"}),
    "bicycle=no": ("bicycle", {"no"}),
    "lanes:divided=yes": ("lanes:divided", {"yes"}),
    "dual_carriageway=yes": ("dual_carriageway", {"yes"}),
    "sidewalk(any)": ("sidewalk", None),
    "junction(any)": ("junction", None),
}


def _km(metres) -> float:
    return float(np.nansum(metres)) / 1000.0


def m1_inventory(ways: gpd.GeoDataFrame, crs: str) -> pd.DataFrame:
    """Way count and km per `highway` value. Cited by road_access_rules.json."""
    lengths = ways.to_crs(crs).geometry.length.to_numpy()
    df = pd.DataFrame({"highway": ways["highway"].astype(str), "m": lengths})
    out = df.groupby("highway").agg(ways=("m", "size"), km=("m", _km))
    out["pct_of_all_km"] = 100 * out["km"] / out["km"].sum()
    return out.sort_values("km", ascending=False)


def m2_length_ratio(inventory: pd.DataFrame, adb_km: float) -> pd.DataFrame:
    """OSM class km over ADB network km. Reported so it can be read
    sceptically; see the module docstring."""
    out = inventory[["km"]].copy()
    out["pct_of_adb_km"] = 100 * out["km"] / adb_km
    return out


def m7_tag_inventory(ways: gpd.GeoDataFrame, crs: str) -> pd.DataFrame:
    """Presence of the tags the road-structure flags are derived from, by class.

    `foot=no` is the one worth reading: it is the pedestrian PROHIBITION, it
    is counted from explicit tags alone (the per-country defaults are left
    out), and with
    `bicycle=no` it is the `osm_vru_prohibited` basis of `is_access_controlled`.
    """
    lengths = ways.to_crs(crs).geometry.length.to_numpy()
    hw = ways["highway"].astype(str).to_numpy()
    frames = []
    for label, (key, values) in TAG_PROBES.items():
        series = _tag_series(ways, key)
        hit = (series.notna() if values is None
               else series.isin(list(values)).fillna(False)).to_numpy(dtype=bool)
        if not hit.any():
            continue
        agg = (pd.DataFrame({"highway": hw[hit], "m": lengths[hit]})
               .groupby("highway").agg(ways=("m", "size"), km=("m", _km)))
        agg["tag"] = label
        frames.append(agg.reset_index())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).set_index(["tag", "highway"]).sort_index()


def within_boundary(ways: gpd.GeoDataFrame, country: str) -> gpd.GeoDataFrame:
    """Just this country's own ways, for reporting.

    The layer osm_ways builds is deliberately wider than the administrative
    outline -- it is grown by osm_way_source.CLIP_MARGIN_M so that an ADB
    segment crossing a state line can still be matched, and 28 Maharashtra
    segments do cross with vertices up to 1,221 m outside. That margin is right
    for matching and wrong for a figure captioned "this state's roads", so the
    outline is applied here instead.
    """
    boundary = gpd.read_file(BOUNDARY_PATHS[country]).to_crs(ways.crs)
    poly = boundary.geometry.union_all()
    keep = shapely.intersects(ways.geometry.values, poly)
    return ways.loc[keep]


def run_country(country: str) -> dict:
    import osm_ways

    crs = BUFFER_CRS[country]
    full = osm_ways.load_ways(country)
    ways = within_boundary(full, country)
    print(f"[{country}] {len(full):,} ways in the layer, {len(ways):,} inside the "
          f"administrative boundary ({len(full) - len(ways):,} in the matching margin)")
    segs = load_target()
    segs = segs[segs["country"] == country]
    adb_km = _km(segs.to_crs(crs).geometry.length)

    print(f"\n{'=' * 78}\n{country.upper()}  --  {len(ways):,} OSM ways, "
          f"{len(segs):,} ADB segments, {adb_km:,.0f} km of ADB network\n{'=' * 78}")

    inventory = m1_inventory(ways, crs)
    ratio = m2_length_ratio(inventory, adb_km)
    show = inventory.join(ratio[["pct_of_adb_km"]])
    print("\n--- M1/M2  way inventory per class ---")
    print(show.head(25).to_string(float_format=lambda v: f"{v:,.1f}"))

    arterial_km = float(inventory.reindex(list(ROAD_HIGHWAY_VALUES))["km"].sum())
    print(f"\nthe four arterial classes "
          f"({'/'.join(ROAD_HIGHWAY_VALUES)}): {arterial_km:,.0f} km")
    print(f"  as a share of ADB network km: {100 * arterial_km / adb_km:.1f}% "
          f"-- above 100%, so this ratio does not measure coverage")
    print(f"all highway classes: {inventory['km'].sum():,.0f} km")

    print("\n--- M7  tags the road-structure flags are derived from ---")
    tags = m7_tag_inventory(ways, crs)
    if len(tags):
        print(tags.to_string(float_format=lambda v: f"{v:,.1f}"))

    return {
        "country": country,
        "osm_ways_in_layer": int(len(full)),
        "osm_ways_in_boundary": int(len(ways)),
        "adb_segments": int(len(segs)),
        "adb_km": round(adb_km, 1),
        "arterial_four_class_km": round(arterial_km, 1),
        "all_highway_km": round(float(inventory["km"].sum()), 1),
        "m1_inventory": show.round(1).to_dict(orient="index"),
        "m7_tags": (tags.round(1).reset_index().to_dict(orient="records")
                    if len(tags) else []),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--country", choices=[*PBF_PATHS, "both"], default="both")
    ap.add_argument("--json", default=None, help="also write the report here")
    args = ap.parse_args()

    countries = [*PBF_PATHS] if args.country == "both" else [args.country]
    reports = [run_country(c) for c in countries]

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
