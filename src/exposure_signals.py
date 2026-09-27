"""Extract OSM POI / pedestrian-way / crossing signals once per country and
cache to data/processed/, so the expensive pbf parse (tens of minutes over
a near-country-wide bbox) never has to re-run.

One pyrosm call per country pulls POIs, pedestrian ways and crossings in
one parse pass.

★ Grade-separated crossings are not at-grade VRU/vehicle conflict points.★
`highway=crossing` is a node tagged on a footway; the bridge/tunnel/layer
tag that says "this is a skywalk/underpass, not an at-grade crossing" lives
on the *footway way*, not the node. A crossing node sitting on a footway
tagged bridge=yes (common in Bangkok at busy intersections) means
pedestrians go OVER the road, not across it -- counting it as exposure
would be backwards. Any crossing node within 3 m of a grade-separated
pedestrian way is excluded.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyrosm import OSM
from shapely.geometry import Point

sys.path.insert(0, "src")
from prefilter_pbf import ensure_filtered_pbf
from poi_categories import (
    MAPILLARY_BOOL_COLS,
    MAPILLARY_JSON_PATHS,
    OSM_BOOL_COLS,
    SEGMENT_BOOL_COLS,
    mapillary_flags,
)
from poi_params import DEFAULT, PoiParams
from poi_sources import BUFFER_CRS, load_pois
from safe_speed import V_SAFE_TABLE
from poi_isochrone import load_selected_isochrones
from schema import load_target

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = "data/processed"

PBF_PATHS = {
    "thailand": "data/external/thailand-260621.osm.pbf",
    "maharashtra": "data/external/western-zone-260621.osm.pbf",
}

CUSTOM_FILTER = {
    "amenity": ["school", "marketplace", "hospital"],
    "shop": True,
    "highway": ["bus_stop", "footway", "path", "pedestrian", "steps", "crossing"],
}
TAGS_AS_COLUMNS = ["amenity", "shop", "highway", "bridge", "tunnel", "layer", "crossing", "footway"]

# Hand pyrosm an osmium-prefiltered copy of the country pbf (Thailand
# 55 min -> ~41 s). See src/prefilter_pbf.py for why this is
# output-identical, and src/verify_prefilter_equivalence.py for the proof.
# A missing osmium raises; set this False (or pass --no-prefilter) to parse
# the full pbf (the 55-minute path).
USE_PBF_PREFILTER = True

# Way tags that mean "grade-separated" (pedestrian path goes over/under the
# road, so a crossing on it is not an at-grade vehicle/VRU conflict point).
GRADE_SEPARATED_BRIDGE_VALUES = {"yes", "boardwalk", "viaduct", "movable", "construction"}
GRADE_SEPARATED_TUNNEL_VALUES = {"yes", "building_passage", "covered", "culvert"}


def resolve_pbf(country: str) -> str:
    """The pbf both extractors should parse: prefiltered unless opted out."""
    if not USE_PBF_PREFILTER:
        return PBF_PATHS[country]
    return ensure_filtered_pbf(PBF_PATHS[country])


def _country_bbox(country: str) -> list[float]:
    target = load_target()
    bounds = target[target["country"] == country].total_bounds
    return list(bounds)


def extract_raw(country: str, pbf_path: str | None = None) -> gpd.GeoDataFrame:
    """Parse the country pbf with pyrosm, cropped to the target bbox.

    The LineString ways it returns (the pedestrian ways) are those with exactly
    two nodes (Thailand 44,797 of 44,798), which follows from the behaviour of
    pyrosm's get_data_by_custom_criteria.

    `pbf_path` overrides which file is parsed (used by the prefilter wiring and
    by verify_prefilter_equivalence.py, which drives this at an explicit file
    without touching module state).
    """
    bbox = _country_bbox(country)
    osm = OSM(pbf_path or resolve_pbf(country), bounding_box=bbox)
    return osm.get_data_by_custom_criteria(
        custom_filter=CUSTOM_FILTER, tags_as_columns=TAGS_AS_COLUMNS, filter_type="keep"
    )


def _is_grade_separated(row) -> bool:
    bridge = row.get("bridge")
    tunnel = row.get("tunnel")
    layer = row.get("layer")
    if bridge in GRADE_SEPARATED_BRIDGE_VALUES:
        return True
    if tunnel in GRADE_SEPARATED_TUNNEL_VALUES:
        return True
    if layer is not None and str(layer) not in ("nan", "None"):
        try:
            if float(str(layer).split(";")[0]) != 0:
                return True
        except ValueError:
            pass
    return False


def split_signals(raw: gpd.GeoDataFrame, crs_for_buffer: str | int) -> dict[str, gpd.GeoDataFrame]:
    pois = raw[raw["amenity"].notna() | raw["shop"].notna() | (raw["highway"] == "bus_stop")].copy()

    pedestrian_ways = raw[
        raw["highway"].isin(["footway", "path", "pedestrian", "steps"]) & (raw.geometry.type == "LineString")
    ].copy()
    pedestrian_ways["grade_separated"] = pedestrian_ways.apply(_is_grade_separated, axis=1)

    crossings_all = raw[
        ((raw["highway"] == "crossing") | (raw["footway"] == "crossing")) & (raw.geometry.type == "Point")
    ].copy()

    crossings_at_grade = crossings_all
    grade_sep_ways = pedestrian_ways[pedestrian_ways["grade_separated"]]
    if len(grade_sep_ways) and len(crossings_all):
        buffered = grade_sep_ways.to_crs(crs_for_buffer).copy()
        buffered["geometry"] = buffered.geometry.buffer(3)
        crossings_proj = crossings_all.to_crs(crs_for_buffer)
        hits = gpd.sjoin(crossings_proj, buffered[["geometry"]], predicate="within")
        at_grade_mask = ~crossings_all.index.isin(hits.index.unique())
        crossings_at_grade = crossings_all[at_grade_mask].copy()

    n_excluded = len(crossings_all) - len(crossings_at_grade)
    print(f"  excluded {n_excluded} / {len(crossings_all)} crossing nodes as grade-separated (bridge/tunnel/layer)")

    return {"pois": pois, "pedestrian_ways": pedestrian_ways, "crossings_at_grade": crossings_at_grade}


def extract_and_cache(country: str) -> dict[str, gpd.GeoDataFrame]:
    raw = extract_raw(country)
    signals = split_signals(raw, BUFFER_CRS[country])
    for name, gdf in signals.items():
        gdf.to_parquet(f"{PROCESSED_DIR}/osm_{name}_{country}.parquet")
    return signals


def load_cached_signals(country: str) -> dict[str, gpd.GeoDataFrame]:
    return {
        name: gpd.read_parquet(f"{PROCESSED_DIR}/osm_{name}_{country}.parquet")
        for name in ("pois", "pedestrian_ways", "crossings_at_grade")
    }


# dwithin distance per segment land_use (metres, applied after UTM projection).
POI_BUFFER_M = {"URBAN": 200, "RURAL": 400}
CROSSING_BUFFER_M = 25


def _flatten_mapillary_json(country: str) -> gpd.GeoDataFrame:
    json_path = Path(MAPILLARY_JSON_PATHS[country])
    if not json_path.exists():
        return gpd.GeoDataFrame(
            columns=["source", "object_value", *OSM_BOOL_COLS, *MAPILLARY_BOOL_COLS, "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    with json_path.open() as f:
        by_objectid = json.load(f)

    rows: list[dict] = []
    for features in by_objectid.values():
        for feat in features:
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates")
            if not coords:
                continue
            object_value = feat["object_value"]
            flags = mapillary_flags(object_value)
            rows.append(
                {
                    "source": "mapillary",
                    "object_value": object_value,
                    **{col: False for col in OSM_BOOL_COLS},
                    **flags,
                    "geometry": Point(coords[0], coords[1]),
                }
            )

    if not rows:
        return gpd.GeoDataFrame(
            columns=["source", "object_value", *OSM_BOOL_COLS, *MAPILLARY_BOOL_COLS, "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def load_mapillary_pois(country: str) -> gpd.GeoDataFrame:
    cache_path = Path(f"{PROCESSED_DIR}/mapillary_pois_{country}.parquet")
    if cache_path.exists():
        return gpd.read_parquet(cache_path)

    gdf = _flatten_mapillary_json(country)
    gdf.to_parquet(cache_path)
    return gdf


def _combined_pois(country: str, params: PoiParams = DEFAULT) -> gpd.GeoDataFrame:
    """POI union (OSM ∪ Overture, see poi_sources) plus Mapillary points."""
    pois = load_pois(country, params)
    for col in MAPILLARY_BOOL_COLS:
        pois[col] = False
    map_pois = load_mapillary_pois(country)
    if len(map_pois) == 0:
        return pois
    if len(pois) == 0:
        return map_pois
    return pd.concat([pois, map_pois], ignore_index=True)


def _apply_dwithin_hits(
    gdf: gpd.GeoDataFrame,
    seg_idx: pd.Index,
    joined: gpd.GeoDataFrame,
) -> None:
    if joined.empty:
        return

    map_hits = joined[joined["source"] == "mapillary"]
    poi_hits = joined[joined["source"] != "mapillary"]

    if not poi_hits.empty:
        osm_agg = poi_hits.groupby(poi_hits.index)[OSM_BOOL_COLS].max()
        gdf.loc[osm_agg.index, OSM_BOOL_COLS] = osm_agg.astype(bool).values
        osm_cat = osm_agg.sum(axis=1).astype(int)
        gdf.loc[osm_cat.index, "osm_poi_category_count"] = osm_cat.values

    if not map_hits.empty:
        map_agg = map_hits.groupby(map_hits.index)[MAPILLARY_BOOL_COLS].max()
        gdf.loc[map_agg.index, MAPILLARY_BOOL_COLS] = map_agg.astype(bool).values
        map_vru = map_agg["map_is_pedestrian"] | map_agg["map_is_bicycle"] | map_agg["map_is_school"]
        gdf.loc[map_vru.index, "is_mapillary_vru"] = map_vru.values

    counts = joined.groupby(joined.index).size()
    gdf.loc[counts.index, "poi_count"] = counts.values

    # Combined segment-facing bool flags (OR across sources).
    gdf.loc[seg_idx, "is_school"] = (
        gdf.loc[seg_idx, "is_school"] | gdf.loc[seg_idx, "map_is_school"]
    )
    gdf.loc[seg_idx, "is_hospital"] = (
        gdf.loc[seg_idx, "is_hospital"] | gdf.loc[seg_idx, "map_is_hospital"]
    )
    gdf.loc[seg_idx, "is_pedestrian"] = gdf.loc[seg_idx, "map_is_pedestrian"]
    gdf.loc[seg_idx, "is_bicycle"] = gdf.loc[seg_idx, "map_is_bicycle"]


# The pre-split VRU mask's road classes, used only by the legacy comparison
# path. It masked trunk outright; the current mask asks `is_access_controlled`
# instead, which trunk satisfies only when OSM tags it motorroad=yes.
LEGACY_VRU_EXCLUDED_ROAD_CLASSES = ["motorway", "trunk"]


def vru_mask(gdf) -> pd.Series:
    """Rows where a nearby VRU detection is not an at-grade conflict *on this
    segment*: pedestrians and cyclists cannot legally enter an access-controlled
    road, so the detection belongs to the surrounding network. `is_divided`
    deliberately does not mask -- a median does not keep VRUs off the road."""
    if "is_access_controlled" not in gdf.columns:
        return pd.Series(False, index=gdf.index)
    return gdf["is_access_controlled"].fillna(False).astype(bool)


def legacy_vru_mask(gdf) -> pd.Series:
    """The pre-split mask: road class alone excluded trunk, whether or not
    anything about its carriageway was ever confirmed, and any `is_separated`
    segment was excluded too. Legacy comparison path only."""
    return (
        gdf["road_class"].isin(LEGACY_VRU_EXCLUDED_ROAD_CLASSES)
        | gdf["is_separated_legacy"].fillna(False).astype(bool)
    )


def _apply_isochrone_hits(
    gdf: gpd.GeoDataFrame,
    isochrones: gpd.GeoDataFrame,
    country: str,
    poi_type: str,
) -> None:
    """Overwrite is_{poi_type} with the isochrone intersection.

    URBAN: reset the dwithin-derived flag to isochrone-only.
    RURAL: keep the dwithin result (floor) and OR it with the isochrone.
    A Mapillary sign of the same type (map_is_school / map_is_hospital) marks
    the segment it was seen from, so it is ORed back after the URBAN reset.
    osm_poi_category_count (exposure axis) is left unchanged from the dwithin result.
    """
    col = f"is_{poi_type}"
    map_col = f"map_is_{poi_type}"
    mask_country = gdf["country"] == country
    if not mask_country.any():
        return

    for land_use in ("URBAN", "RURAL"):
        mask = mask_country & (gdf["land_use"] == land_use)
        if not mask.any():
            continue

        # URBAN: reset the dwithin-derived flag (replaced with isochrone-only)
        if land_use == "URBAN":
            gdf.loc[mask, col] = False
            if map_col in gdf.columns:
                gdf.loc[mask, col] = gdf.loc[mask, map_col].fillna(False).astype(bool)

        iso_lu = isochrones[isochrones["land_use"] == land_use]
        if iso_lu.empty:
            continue

        # intersects sjoin between the isochrone polygon and the segments
        iso_utm = iso_lu[["geometry"]].to_crs(BUFFER_CRS[country])
        segs_utm = gdf.loc[mask, ["geometry"]].to_crs(BUFFER_CRS[country])
        joined = gpd.sjoin(segs_utm, iso_utm, predicate="intersects")
        if not joined.empty:
            gdf.loc[joined.index.unique(), col] = True


# A Mapillary VRU detection (school-zone sign, crosswalk or bicycle marking)
# always caps at the pedestrian threshold; only the POI types are tunable.
MAPILLARY_VRU_SPEED_KMH = V_SAFE_TABLE["pedestrian"]


def _vru_speed_cap(gdf: gpd.GeoDataFrame, params: PoiParams) -> tuple[pd.Series, pd.Series]:
    """Per row: the lowest speed among the Mapillary VRU triggers present, and
    which trigger set it. Triggers are a Mapillary VRU detection and, when
    `params` enables hospitals, a Mapillary hospital sign at the hospital speed.
    NaN / None where nothing triggers.

    The POI types (OSM ∪ Overture) set V_safe in poi_speed_zones.py, on the
    final geometry after stage 2."""
    triggers = [("mapillary", gdf["is_mapillary_vru"], MAPILLARY_VRU_SPEED_KMH)]
    hospital = params.cap("hospital")
    if hospital.enabled and "map_is_hospital" in gdf.columns:
        triggers.append(("mapillary_hospital", gdf["map_is_hospital"], hospital.speed_kmh))
    cap = pd.Series(np.nan, index=gdf.index)
    source = pd.Series(None, index=gdf.index, dtype="object")
    for name, hit, speed in triggers:
        lower = hit.fillna(False).astype(bool) & ~(cap <= speed)
        cap[lower] = speed
        source[lower] = name
    return cap, source


def add_poi_proximity(
    gdf: gpd.GeoDataFrame,
    use_isochrone: bool = True,
    include_legacy: bool = False,
    params: PoiParams = DEFAULT,
) -> gpd.GeoDataFrame:
    """Attach POI proximity via UTM dwithin (no segment buffer polygons).

    POIs (OSM ∪ Overture, poi_sources.load_pois) and Mapillary map_features points are joined at 200 m (urban) or
    400 m (rural). Category bools are aggregated with max(); poi_count is the
    total hit count across both sources.

    `use_isochrone` is unused: the POI isochrones set V_safe in
    poi_speed_zones.py, after stage 2. is_{type} and osm_poi_category_count
    are the dwithin result (exposure axis and legacy comparison only).

    `is_mapillary_vru` (school-zone sign / crosswalk marking / bicycle marking
    detected within the dwithin distance) is the Mapillary-only sub-signal used
    by exposure_level. `is_vru` is the VRU trigger that drives V_safe here: a
    Mapillary VRU detection, or a Mapillary hospital sign when `params` enables
    hospitals. `vru_speed_cap` is the lowest speed among the triggers present
    (Mapillary 30 km/h; hospital sign at the hospital speed) and
    `vru_cap_source` names the one that set it. Both flags are forced to False
    (and the cap to NaN) for any
    segment with `is_access_controlled==True` (road_separation.py): pedestrians
    and cyclists cannot legally enter it, so a nearby detection belongs to the
    surrounding network. `is_divided` does not mask them -- VRUs can still be on
    a divided road.

    `include_legacy=True` additionally writes `is_mapillary_vru_legacy` /
    `is_vru_legacy` under the pre-split mask (road_class in motorway/trunk OR
    `is_separated_legacy`), for the comparison path. Both maskings are applied
    to the same unmasked values, computed once here: deriving the legacy pair
    from the already-masked columns would lose every detection the new mask
    drops and the old one keeps (a `motorroad=yes` primary, for instance).
    """
    gdf = gdf.copy()
    for col in SEGMENT_BOOL_COLS:
        gdf[col] = False
    for col in MAPILLARY_BOOL_COLS:
        gdf[col] = False
    gdf["poi_count"] = 0
    gdf["osm_poi_category_count"] = 0
    gdf["is_mapillary_vru"] = False
    gdf["is_vru"] = False
    gdf["vru_speed_cap"] = np.nan
    gdf["vru_cap_source"] = pd.Series(None, index=gdf.index, dtype="object")

    poi_cols = ["source", "geometry", *OSM_BOOL_COLS, *MAPILLARY_BOOL_COLS]

    for country in gdf["country"].unique():
        pois = _combined_pois(country, params)
        if len(pois) == 0:
            continue

        for land_use, dist_m in POI_BUFFER_M.items():
            mask = (gdf["country"] == country) & (gdf["land_use"] == land_use)
            if not mask.any():
                continue
            seg_idx = gdf.index[mask]
            segs = gdf.loc[mask, ["geometry"]].to_crs(BUFFER_CRS[country])
            pois_utm = pois[poi_cols].to_crs(BUFFER_CRS[country])
            joined = gpd.sjoin(segs, pois_utm, predicate="dwithin", distance=dist_m)
            _apply_dwithin_hits(gdf, seg_idx, joined)

    gdf["vru_speed_cap"], gdf["vru_cap_source"] = _vru_speed_cap(gdf, params)
    gdf["is_vru"] = gdf["vru_speed_cap"].notna()

    # Held before either mask is applied, so both can be derived from the same
    # evidence (see the include_legacy note in the docstring). The legacy rule
    # predates the per-type caps and always read Mapillary OR school.
    mapillary_vru_raw = gdf["is_mapillary_vru"].copy()
    vru_raw = gdf["is_mapillary_vru"] | gdf["is_school"]

    masked = vru_mask(gdf)
    gdf.loc[masked, "is_mapillary_vru"] = False
    gdf.loc[masked, "is_vru"] = False
    gdf.loc[masked, "vru_speed_cap"] = np.nan
    gdf.loc[masked, "vru_cap_source"] = None

    if include_legacy:
        if "is_separated_legacy" not in gdf.columns:
            raise ValueError(
                "include_legacy=True needs `is_separated_legacy`; run "
                "road_separation.add_road_structure(..., include_legacy=True) first"
            )
        legacy_masked = legacy_vru_mask(gdf)
        gdf["is_mapillary_vru_legacy"] = mapillary_vru_raw & ~legacy_masked
        gdf["is_vru_legacy"] = vru_raw & ~legacy_masked

    return gdf


add_poi_count = add_poi_proximity


def add_crossing_signal(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """has_crossing/crossing_count are urban signals (OSM crossing tagging is
    near-zero in rural areas). A rural 0 is not read as "low exposure"
    downstream (see exposure_level.apply_rural_safety_margin)."""
    gdf = gdf.copy()
    gdf["has_crossing"] = False
    gdf["crossing_count"] = 0

    for country in gdf["country"].unique():
        crossings = load_cached_signals(country)["crossings_at_grade"]
        mask = gdf["country"] == country
        if len(crossings) == 0 or not mask.any():
            continue
        crossings_utm = crossings.to_crs(BUFFER_CRS[country])

        segs = gdf.loc[mask, ["geometry"]].to_crs(BUFFER_CRS[country]).copy()
        segs["geometry"] = segs.geometry.buffer(CROSSING_BUFFER_M)
        joined = gpd.sjoin(crossings_utm[["geometry"]], segs, predicate="within")
        counts = joined.groupby("index_right").size()
        gdf.loc[counts.index, "crossing_count"] = counts.values
        gdf.loc[counts.index, "has_crossing"] = True

    return gdf


def _prefilter_cli_args(description: str) -> argparse.Namespace:
    """Shared --no-prefilter / --force-prefilter flags (also used by road_separation)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--no-prefilter", action="store_true",
                        help="parse the full pbf directly, skipping osmium tags-filter "
                             "(correct but slow: ~55 min for Thailand)")
    parser.add_argument("--force-prefilter", action="store_true",
                        help="rebuild the filtered pbf even if the cache is fresh")
    return parser.parse_args()


if __name__ == "__main__":
    _args = _prefilter_cli_args("Extract and cache OSM POI / pedestrian-way / crossing signals")
    if _args.no_prefilter:
        USE_PBF_PREFILTER = False
    elif _args.force_prefilter:
        for _c in PBF_PATHS:
            ensure_filtered_pbf(PBF_PATHS[_c], force=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for country in PBF_PATHS:
        print(f"--- {country} ---")
        signals = extract_and_cache(country)
        for name, gdf in signals.items():
            print(f"{name}: {len(gdf)}")
        print()
