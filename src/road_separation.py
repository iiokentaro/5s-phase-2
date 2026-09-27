"""Derive the road-structure flags from OSM road-network tags.

The flags answer independent questions and are judged separately; either,
both, or neither can be True. All default to False.

is_access_controlled -- pedestrians and cyclists cannot legally enter:
  highway == 'motorway'  OR  motorroad == 'yes'
  (road_access_join adds a third basis, 'osm_vru_prohibited', from the
  legal-access tags)

is_divided -- opposing traffic cannot meet, so head-on and angle crashes are
treated as impossible (VRUs may still be present, so this alone does not
exclude them):
  lanes:divided == 'yes'  OR  dual_carriageway == 'yes'
  OR  oneway present with any value other than no / 0 / false

is_grade_separated -- a matched way is a bridge, tunnel or non-zero layer.

segment_way_match attributes each metre of a segment to the OSM way it was
built from (vertex agreement), and road_structure_flags turns the matched
lengths into flags: is_access_controlled needs every matched metre, is_divided
`divided_min_length_frac` of them (src/match_rules.json).

One fallback: an Overture road_class == 'motorway' with
f85_speed >= MOTORWAY_FALLBACK_MIN_F85 counts as access-controlled where OSM
did not confirm it (basis 'overture_motorway_fallback'). is_divided keeps its
OSM value.

★ Legacy comparison path ★
`add_road_structure(..., include_legacy=True)` also aggregates the pre-split
`is_separated` rule (_is_separated_way_legacy) into `is_separated_legacy`, so
the old and new recommended speeds can be read off one table. It uses the
same attribution and adds only its own column.
"""

import json
import logging
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyrosm import OSM

sys.path.insert(0, "src")
from exposure_signals import (
    BUFFER_CRS,
    GRADE_SEPARATED_BRIDGE_VALUES,
    GRADE_SEPARATED_TUNNEL_VALUES,
    PBF_PATHS,
    _country_bbox,
    _prefilter_cli_args,
    resolve_pbf,
)
from schema import load_target

import osm_ways
import road_structure_flags
import segment_way_match

# Parent matching depends only on the ADB export and the way layer, and
# build_v_safe runs this step twice per country per build.
_MATCH_CACHE: dict = {}

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = "data/processed"
# The classes extract_road_network (the osm_roads_* cache) keeps.
ROAD_HIGHWAY_VALUES = ["motorway", "trunk", "primary", "secondary"]
# `oneway` values that mean "not one-way". Anything else present (yes, -1, 1,
# true, reversible, alternating) means traffic on the way flows one direction
# at a time, so it cannot meet opposing traffic head-on.
ONEWAY_ABSENT_VALUES = {"no", "0", "false"}
# Unmatched Overture motorway counts as access-controlled only at or above this
# F85 (km/h). Below it the motorway tag itself is suspect; 16/169 motorway
# segments fell below it, 13 of them with placeholder all-zero speeds.
MOTORWAY_FALLBACK_MIN_F85 = 50
# --- legacy comparison path (see _is_separated_way_legacy) ---
# The road classes the pre-split rule paired with oneway=yes. Kept apart from
# ROAD_HIGHWAY_VALUES: that one says what to extract, this one said what counted
# as separated.
LEGACY_SEPARATED_HIGHWAY_VALUES = ("motorway", "trunk")


def extract_road_network(country: str, pbf_path: str | None = None) -> gpd.GeoDataFrame:
    """See exposure_signals.extract_raw for what `pbf_path` is for."""
    bbox = _country_bbox(country)
    osm = OSM(pbf_path or resolve_pbf(country), bounding_box=bbox)
    roads = osm.get_data_by_custom_criteria(
        custom_filter={"highway": ROAD_HIGHWAY_VALUES},
        tags_as_columns=["highway", "oneway"],
        filter_type="keep",
    )
    return roads[roads.geometry.type == "LineString"].copy()


def _tags(row) -> dict:
    """pyrosm's catch-all `tags` column: a JSON string in the cached parquet,
    a dict straight out of pyrosm, or missing."""
    tags = row.get("tags")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except ValueError:
            return {}
    return tags if isinstance(tags, dict) else {}


def _tag_value(row, key):
    """A tag read from its own column when pyrosm promoted it (highway, oneway),
    else from `tags`. NaN/None/'' all mean absent."""
    value = row.get(key)
    if value is None or (isinstance(value, float) and value != value):
        value = _tags(row).get(key)
    if value is None or (isinstance(value, float) and value != value):
        return None
    value = str(value).strip()
    return value or None


def _access_control_basis_way(row) -> str | None:
    """'osm_motorway' / 'osm_motorroad' if the way is access-controlled, else None.
    motorway wins when both apply: it is the stronger claim (no at-grade junctions)."""
    if _tag_value(row, "highway") == "motorway":
        return "osm_motorway"
    if _tag_value(row, "motorroad") == "yes":
        return "osm_motorroad"
    return None


def _is_access_controlled_way(row) -> bool:
    return _access_control_basis_way(row) is not None


def _is_divided_way(row) -> bool:
    if _tag_value(row, "lanes:divided") == "yes":
        return True
    if _tag_value(row, "dual_carriageway") == "yes":
        return True
    oneway = _tag_value(row, "oneway")
    return oneway is not None and oneway.lower() not in ONEWAY_ABSENT_VALUES


def _is_separated_way_legacy(row) -> bool:
    """The pre-split `is_separated` rule, kept for the legacy comparison path.

    A way was separated if ANY of:
      (1) highway in {motorway, trunk} AND oneway == 'yes'
      (2) lanes:divided == 'yes'
      (3) dual_carriageway == 'yes'

    (1) is what mixed the two concepts this module now judges separately: it
    reads access control off the road class and carriageway separation off
    `oneway` in one test. Note it is strictly narrower than `_is_divided_way`
    on real data -- a one-way primary qualifies there and not here -- so the
    legacy flag comes out a subset of `is_divided` (verified on both countries'
    cached way sets).
    """
    if (_tag_value(row, "highway") in LEGACY_SEPARATED_HIGHWAY_VALUES
            and _tag_value(row, "oneway") == "yes"):
        return True
    if _tag_value(row, "lanes:divided") == "yes":
        return True
    return _tag_value(row, "dual_carriageway") == "yes"


def _is_grade_separated_way(row) -> bool:
    """A way is grade-separated (bridge/flyover or tunnel/underpass) if it
    carries bridge/tunnel/layer values meaning "physically above or below the
    surrounding network" -- same value sets exposure_signals.py uses for
    pedestrian-way crossings, reused here for junction_speed_cap.py's at-grade
    assumption (a flyover passing over an at-grade junction is not itself an
    at-grade side-impact conflict point).

    Read through `_tag_value`, never `_tags` directly: pyrosm MOVES a tag out
    of the catch-all `tags` dict into its own column once that key is named in
    `tags_as_columns`. Reading only `tags` would therefore return all-False the
    moment bridge/tunnel/layer are promoted -- silently, and in the unsafe
    direction, since `is_grade_separated` is what exempts a segment from
    junction_speed_cap's 50 km/h cap."""
    if _tag_value(row, "bridge") in GRADE_SEPARATED_BRIDGE_VALUES:
        return True
    if _tag_value(row, "tunnel") in GRADE_SEPARATED_TUNNEL_VALUES:
        return True
    layer = _tag_value(row, "layer")
    if layer is not None:
        try:
            if float(layer.split(";")[0]) != 0:
                return True
        except ValueError:
            pass
    return False


def _tag_series(roads: gpd.GeoDataFrame, key: str) -> pd.Series:
    """`_tag_value` for a whole column at once, same semantics: read the
    promoted column, fall back to the catch-all `tags`, and treat NaN/None/''
    as absent after stripping.

    The catch-all fallback is the expensive half, so it is narrowed twice
    before any per-row work happens: only rows whose promoted value is missing,
    and of those only rows whose `tags` JSON mentions the key at all -- a
    vectorised substring test. Without that narrowing this runs `_tags` on
    nearly every row, because most ways carry no `motorroad`, no
    `dual_carriageway` and no `layer`.
    """
    if key in roads.columns:
        out = roads[key].astype("string")
    else:
        out = pd.Series(pd.NA, index=roads.index, dtype="string")

    if "tags" in roads.columns:
        candidates = out.isna() & roads["tags"].notna()
        if candidates.any():
            mentions = roads.loc[candidates, "tags"].astype("string").str.contains(
                f'"{key}"', regex=False, na=False)
            probe = candidates[candidates].index[mentions.to_numpy()]
            if len(probe):
                found = roads.loc[probe].apply(lambda r: _tags(r).get(key), axis=1)
                out.loc[probe] = found.astype("string")

    out = out.str.strip()
    return out.where(out.ne(""), pd.NA)


def _annotate_vectorised(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Same predicates as the row-wise functions above, evaluated column-wise.

    The way layer has 2.9 million rows, and five `.apply(axis=1)` passes over
    it take minutes per build. tests/test_road_structure.py asserts the two
    paths agree, so the row-wise versions stay the readable definition of the
    rules.
    """
    # `==` on a nullable string column yields pd.NA where the value is absent,
    # and NA propagates through `|`, so every comparison is reduced to a plain
    # numpy bool here.
    def eq(series: pd.Series, value: str) -> np.ndarray:
        return (series == value).fillna(False).to_numpy(dtype=bool)

    def isin(series: pd.Series, values) -> np.ndarray:
        return series.isin(list(values)).fillna(False).to_numpy(dtype=bool)

    highway = _tag_series(roads, "highway")
    oneway = _tag_series(roads, "oneway")
    motorroad = _tag_series(roads, "motorroad")
    lanes_divided = _tag_series(roads, "lanes:divided")
    dual = _tag_series(roads, "dual_carriageway")

    basis = pd.Series(pd.NA, index=roads.index, dtype="object")
    basis[eq(motorroad, "yes")] = "osm_motorroad"
    basis[eq(highway, "motorway")] = "osm_motorway"  # the stronger claim wins ties
    roads["access_control_basis_way"] = basis
    roads["is_access_controlled_way"] = basis.notna().to_numpy()

    oneway_present = (oneway.notna().to_numpy()
                      & ~isin(oneway.str.lower(), ONEWAY_ABSENT_VALUES))
    divided = eq(lanes_divided, "yes") | eq(dual, "yes")
    roads["is_divided_way"] = divided | oneway_present
    roads["is_separated_way_legacy"] = (
        (isin(highway, LEGACY_SEPARATED_HIGHWAY_VALUES) & eq(oneway, "yes")) | divided)

    bridge = _tag_series(roads, "bridge")
    tunnel = _tag_series(roads, "tunnel")
    layer_num = pd.to_numeric(_tag_series(roads, "layer").str.split(";").str[0],
                              errors="coerce")
    roads["is_grade_separated_way"] = (
        isin(bridge, GRADE_SEPARATED_BRIDGE_VALUES)
        | isin(tunnel, GRADE_SEPARATED_TUNNEL_VALUES)
        | (layer_num.notna() & (layer_num != 0)).to_numpy(dtype=bool))
    return roads


def annotate_road_flags(roads: gpd.GeoDataFrame, *, vectorised: bool = True
                        ) -> gpd.GeoDataFrame:
    """Attach the way-level flags (shared by cache_road_network,
    verify_prefilter_equivalence.py and add_road_structure).

    `vectorised=False` runs the row-wise predicates, which are the definition
    the tests hold the fast path to."""
    if vectorised:
        return _annotate_vectorised(roads)
    roads["access_control_basis_way"] = roads.apply(_access_control_basis_way, axis=1)
    roads["is_access_controlled_way"] = roads["access_control_basis_way"].notna()
    roads["is_divided_way"] = roads.apply(_is_divided_way, axis=1)
    roads["is_grade_separated_way"] = roads.apply(_is_grade_separated_way, axis=1)
    roads["is_separated_way_legacy"] = roads.apply(_is_separated_way_legacy, axis=1)
    return roads


def cache_road_network(country: str) -> gpd.GeoDataFrame:
    roads = annotate_road_flags(extract_road_network(country))
    roads.to_parquet(f"{PROCESSED_DIR}/osm_roads_{country}.parquet")
    return roads


def _country_matches(country: str, rows: gpd.GeoDataFrame):
    """(row attribution, annotated ways, row lengths) for one country.

    The parent match is memoised per country: it depends only on the ADB export
    and the way layer, neither of which changes during a build, while
    build_v_safe runs this step twice (once on whole rows, once on the
    influence-zone children) for each country.
    """
    ways = _MATCH_CACHE.get(("ways", country))
    if ways is None:
        ways = annotate_road_flags(osm_ways.load_ways(country))
        _MATCH_CACHE[("ways", country)] = ways

    cached = _MATCH_CACHE.get(("parents", country))
    if cached is None:
        parents = load_target()
        parents["overture_segment_id"] = parents["segment_id"].astype(str)
        parents = parents[parents["country"] == country].reset_index(drop=True)
        edge_matches, _ = segment_way_match.match_parents(parents, ways, country)
        cached = (parents, edge_matches)
        _MATCH_CACHE[("parents", country)] = cached
    parents, edge_matches = cached

    local = rows.reset_index(drop=True)
    row_matches = segment_way_match.apportion_to_rows(edge_matches, parents, local, country)
    row_len = local.to_crs(BUFFER_CRS[country]).geometry.length.to_numpy()
    return row_matches, ways, row_len


def add_road_structure(gdf: gpd.GeoDataFrame, include_legacy: bool = False) -> gpd.GeoDataFrame:
    """Attach the road-structure flags by matching each row to the OSM ways it
    is made of.

    Orchestration only: segment_way_match decides which ways account for which
    metres of a row, road_structure_flags turns that into flags, and the
    Overture fallback below is the one rule that does not come from OSM.

    `include_legacy` also aggregates the pre-split rule into
    `is_separated_legacy` for build_v_safe.build()'s comparison path, from the
    same attribution.
    """
    gdf = gdf.copy()
    flags = road_structure_flags.empty_flags(gdf.index, include_legacy)
    for col in flags.columns:
        gdf[col] = flags[col]

    for country in gdf["country"].unique():
        mask = (gdf["country"] == country).to_numpy()
        rows = gdf.loc[mask]
        row_matches, ways, row_len = _country_matches(country, rows)
        derived = road_structure_flags.derive_flags(
            row_matches, ways, row_len, rows.index, include_legacy=include_legacy)
        for col in derived.columns:
            gdf.loc[mask, col] = derived[col].to_numpy()

    return _apply_overture_fallback(gdf)


def _apply_overture_fallback(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """RoadClass is an Overture estimate, and this is the one place it stands
    in for OSM: a motorway whose observed speed is consistent with a real
    motorway (a genuinely access-controlled road essentially never has F85 this
    low) counts as access-controlled when OSM did not say so.

    The gate is `is_access_controlled == False`. About 97% of rows match OSM,
    so a gate on `road_structure_confidence == 'low'` would miss the case the
    fallback is for: a motorway that matched OSM ways which fail to carry
    `highway=motorway` on every metre.

    Two consequences worth stating: this can overwrite `access_control_basis`
    on a row where OSM DID match and said "not access-controlled", and
    junction_speed_cap.MOTORWAY_BASES includes this basis, so those rows also
    become exempt from the 50 km/h junction cap.
    """
    fallback = (
        (~gdf["is_access_controlled"].fillna(False).astype(bool))
        & (gdf["road_class"] == "motorway")
        & (gdf["f85_speed"] >= MOTORWAY_FALLBACK_MIN_F85)
    )
    gdf.loc[fallback, "is_access_controlled"] = True
    gdf.loc[fallback, "access_control_basis"] = "overture_motorway_fallback"
    return gdf



if __name__ == "__main__":
    import exposure_signals

    _args = _prefilter_cli_args("Extract and cache the OSM road network, then derive the road-structure flags")
    if _args.no_prefilter:
        exposure_signals.USE_PBF_PREFILTER = False
    elif _args.force_prefilter:
        from prefilter_pbf import ensure_filtered_pbf
        for _c in PBF_PATHS:
            ensure_filtered_pbf(PBF_PATHS[_c], force=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for country in PBF_PATHS:
        print(f"--- {country} ---")
        roads = cache_road_network(country)
        print(f"road ways: {len(roads)}, is_access_controlled_way=True: "
              f"{roads['is_access_controlled_way'].sum()}, is_divided_way=True: {roads['is_divided_way'].sum()}")

    target = load_target()
    target = add_road_structure(target)
    print()
    print(target.groupby(["country", "is_access_controlled", "is_divided"]).size())
    print()
    print(target.groupby("country")["access_control_basis"].value_counts())
