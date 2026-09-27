"""Post-hoc V_safe cap near OSM junction nodes (signal-controlled
intersections and marked minor junctions), applied *after* `safe_speed.add_v_safe`
has already produced its is_access_controlled/is_divided/is_vru-based V_safe.

★ Why this is a separate, later step ★
Safe System guidance sets ~50 km/h as the survivable-impact threshold for the
side-impact (right-angle, vehicle-vehicle) crash type characteristic of
intersections -- a distinct physical conflict from the pedestrian (30) and
head-on (70) cases `safe_speed.py` covers. It is a *cap*: a segment already
at or below 50 keeps its value. Roads without access control are at 30 or 50
(safe_speed.py), so the cap changes only access-controlled segments
(70 / 80-100).

★ Exclusions: motorway and grade-separated segments ★
The cap fires regardless of `is_divided` (a median does not stop a
signal-controlled intersection from being an at-grade side-impact conflict
point) and regardless of plain access control: `motorroad=yes` roads can
carry at-grade junctions. But two cases mean a junction within
JUNCTION_BUFFER_M is *not* actually an at-grade conflict for this segment, so
the cap is excluded for them:
- `access_control_basis` in MOTORWAY_BASES (OSM highway=motorway, or the
  Overture motorway fallback): fully access-controlled -- any junction within
  the buffer is a grade-separated ramp/interchange, not an at-grade crossing
  motorway traffic itself passes through. A segment whose matched ways mix
  motorway and motorroad gets basis `osm_motorroad` and is not exempt.
- `is_grade_separated == True` (road_separation.py; the segment's matched
  OSM way(s) carry bridge/tunnel/layer!=0): a flyover/underpass passing
  near or over a junction node is not at the same grade as that junction.
Both exclusions gate `near_junction` itself (not just the cap), so an
excluded segment is also not pulled into segment_localization.py's Stage 2
influence-zone split on the junction's account (it can still split on
`is_vru`).

★ Tags used / data source ★
- `highway=traffic_signals` (node): signal-controlled intersection.
- `junction=yes` (way): a marked, non-roundabout junction.
`junction=roundabout` is excluded: a roundabout is itself a speed-reducing
design, and the cap targets the uncontrolled side-impact case.

Source: `data/external/osm_junctions_{country}.geojson`, an Overpass Turbo
export. Every row satisfies `highway=='traffic_signals' or junction=='yes'`;
some also carry `junction=roundabout`/`intersection` as a secondary tag, and
the row still counts because it matched on the primary one.

★ Granularity ★
build_v_safe.py runs this in two stages. Stage 1 runs it on whole segments
and marks near_junction; segment_localization.py then cuts near-junction
segments at the 300 m buffer boundary, and Stage 2 caps only the influenced
pieces. A segment too short to split (50 m minimum per part) is capped whole.
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS
from safe_speed import V_SAFE_TABLE

warnings.filterwarnings("ignore", category=UserWarning)

EXTERNAL_DIR = "data/external"
PROCESSED_DIR = "data/processed"
JUNCTION_BUFFER_M = 300
JUNCTION_V_SAFE_CAP = V_SAFE_TABLE["side_impact"]  # 50 km/h, single source of truth with safe_speed.py
COUNTRIES = ["thailand", "maharashtra"]
# road_separation.add_road_structure's access_control_basis values that mean a
# full motorway (no at-grade junctions). `osm_motorroad` is deliberately absent.
MOTORWAY_BASES = ["osm_motorway", "overture_motorway_fallback"]
# The pre-split exclusion keyed on Overture's road_class alone, so a
# `motorroad=yes` road was capped and an unmatched Overture motorway was exempt
# -- the opposite of both current rulings. Legacy comparison path only.
LEGACY_EXCLUDED_ROAD_CLASS = "motorway"


def extract_junctions(country: str) -> gpd.GeoDataFrame:
    """Read the Overpass Turbo export and keep the rows with
    `highway=='traffic_signals' or junction=='yes'`, in case the export is
    regenerated with a looser query."""
    gdf = gpd.read_file(f"{EXTERNAL_DIR}/osm_junctions_{country}.geojson")
    keep = (gdf["highway"] == "traffic_signals") | (gdf["junction"] == "yes")
    return gdf.loc[keep, ["highway", "junction", "geometry"]].copy()


def cache_junctions(country: str) -> gpd.GeoDataFrame:
    junctions = extract_junctions(country)
    junctions.to_parquet(f"{PROCESSED_DIR}/osm_junctions_{country}.parquet")
    return junctions


def load_cached_junctions(country: str) -> gpd.GeoDataFrame:
    cache_path = Path(f"{PROCESSED_DIR}/osm_junctions_{country}.parquet")
    if not cache_path.exists():
        return cache_junctions(country)
    return gpd.read_parquet(cache_path)


def add_junction_speed_cap(gdf: gpd.GeoDataFrame, include_legacy: bool = False) -> gpd.GeoDataFrame:
    """Cap v_safe to JUNCTION_V_SAFE_CAP (50) for segments within
    JUNCTION_BUFFER_M (300m) of a cached junction feature, excluding
    motorway and grade-separated segments (see module docstring). Must run
    after `safe_speed.add_v_safe` and `road_separation.add_road_structure`
    (for `is_grade_separated` / `access_control_basis`). Only ever lowers v_safe (min()), and only
    relabels collision_type/v_safe_basis for segments it actually changes.

    `include_legacy` applies the pre-split exclusion (road_class=='motorway')
    to the same proximity result, writing `near_junction_legacy` and capping
    the `*_legacy` trio.
    ★ `near_junction_legacy` must stay out of segment_localization's split mask.
    The legacy columns describe the rows the current rule produced; letting them
    choose the rows would make the comparison circular.
    """
    gdf = gdf.copy()
    near_junction_raw = pd.Series(False, index=gdf.index)

    for country in gdf["country"].unique():
        junctions = load_cached_junctions(country)
        mask = gdf["country"] == country
        if len(junctions) == 0 or not mask.any():
            continue
        crs = BUFFER_CRS[country]
        junctions_utm = junctions.to_crs(crs)[["geometry"]]
        segs = gdf.loc[mask, ["geometry"]].to_crs(crs)
        joined = gpd.sjoin(segs, junctions_utm, predicate="dwithin", distance=JUNCTION_BUFFER_M)
        near_junction_raw.loc[joined.index.unique()] = True

    grade_separated = gdf.get("is_grade_separated") == True  # noqa: E712

    # Motorway (any nearby junction is a grade-separated interchange) and
    # grade-separated segments (flyover/underpass) are excluded from
    # near_junction itself, so the Stage 2 influence-zone split
    # (segment_localization.py) also leaves them whole on the junction's account.
    exclude = (gdf.get("access_control_basis", pd.Series(pd.NA, index=gdf.index)).isin(MOTORWAY_BASES)
               | grade_separated)
    gdf["near_junction"] = near_junction_raw & ~exclude

    capped = gdf["near_junction"] & (gdf["v_safe"] > JUNCTION_V_SAFE_CAP)
    gdf.loc[capped, "v_safe"] = JUNCTION_V_SAFE_CAP
    gdf.loc[capped, "collision_type"] = "side_impact"
    gdf.loc[capped, "v_safe_basis"] = "side_impact:junction_buffer"

    if include_legacy:
        exclude_legacy = (gdf["road_class"] == LEGACY_EXCLUDED_ROAD_CLASS) | grade_separated
        gdf["near_junction_legacy"] = near_junction_raw & ~exclude_legacy

        capped_legacy = gdf["near_junction_legacy"] & (gdf["v_safe_legacy"] > JUNCTION_V_SAFE_CAP)
        gdf.loc[capped_legacy, "v_safe_legacy"] = JUNCTION_V_SAFE_CAP
        gdf.loc[capped_legacy, "collision_type_legacy"] = "side_impact"
        gdf.loc[capped_legacy, "v_safe_basis_legacy"] = "side_impact:junction_buffer"

    return gdf


if __name__ == "__main__":
    for country in COUNTRIES:
        print(f"--- {country} ---")
        junctions = cache_junctions(country)
        print(f"junction features: {len(junctions)}")
        if len(junctions):
            print(junctions.geometry.type.value_counts())
            print(junctions[["highway", "junction"]].apply(lambda s: s.value_counts(dropna=False).to_dict()))

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, _ = build()
    print()
    print(f"capped to {JUNCTION_V_SAFE_CAP}km/h near a junction: "
          f"{(target['v_safe_basis'] == 'side_impact:junction_buffer').sum()} / {len(target)}")
