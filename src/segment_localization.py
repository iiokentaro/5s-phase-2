"""Stage-2 refinement: clip long segments at influence-zone boundaries, so
that V_safe is localised to the parts of a segment that are near a Mapillary
VRU detection, a Mapillary hospital sign or a junction node. (The OSM ∪
Overture POI zones are applied later, in poi_speed_zones.py.)

Two stages (build_v_safe.py):
- Stage 1: a whole-segment spatial join flags is_vru and near_junction, and
  V_safe is set for the whole segment.
- Stage 2 (this module): for segments where Stage 1 raised either flag, the
  geometry is cut at the influence-zone boundary into an "influenced" and a
  "non-influenced" portion. The Stage 1 geometry signals then run again on the
  pieces; influenced pieces get the lower V_safe, the others keep the higher.

Performance:
  (1) The influence polygon is built only from the points (Mapillary VRU,
      hospital signs, junctions) within the influence radius of the target
      segments, found with gpd.sjoin(predicate="dwithin") before buffering.
  (2) The morphological closing (buffer +25 m / -25 m) is computed once per
      (country, land_use) group and reused for every segment in the group.
  Cutting needs an actual polygon, so the buffers and their union are built;
  the two steps above keep that small.

Sliver safety-side closing (Safe System precautionary principle):
  A non-influence gap shorter than MIN_PIECE_M between two influence zones is
  absorbed into the influence zone (lower V_safe), by morphological closing
  on the influence polygon (buffer out by MIN_PIECE_M/2, then back in). A gap
  too short to classify confidently as outside an influence zone is treated
  as inside it. PIPELINE.md describes this too.

sample_size_avg and sample_size_total are copied, not allocated:
  both count the probes that passed the segment. Cutting a road into an
  upstream and a downstream piece leaves the number of probes that passed
  each piece the same, so every child keeps the parent's value.

segment_id normalisation:
  All rows get segment_id converted to "{country}_{original_objectid}" (string).
  Children get "{country}_{parent_objectid}-{k}". This guarantees global
  uniqueness, since Thailand and Maharashtra both use OBJECTID sequences
  starting from 1.
"""

import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.ops import unary_union

sys.path.insert(0, "src")
from exposure_signals import (
    BUFFER_CRS,
    POI_BUFFER_M,
    load_mapillary_pois,
)
from poi_params import DEFAULT, PoiParams
from poi_isochrone import load_selected_isochrones
from junction_speed_cap import JUNCTION_BUFFER_M, load_cached_junctions

# Minimum length (metres) for an independent segment piece after clipping.
MIN_PIECE_M = 50

# Non-geometry columns copied verbatim from parent to each child.
# sample_size_avg and sample_size_total are copied as well (see the module docstring).
# All geometry-driven signal columns are re-computed by the Stage-1 functions.
INHERIT_COLS = [
    "segment_id", "country", "road_class", "land_use", "urban_pc",
    "speed_limit", "median_speed", "f85_speed",
    "analysis_status", "exclude_from_speedspi", "data_quality_flag",
]

_MAPILLARY_VRU_FLAGS = ["map_is_pedestrian", "map_is_bicycle", "map_is_school"]


def build_influence_polygon_near(
    target_segs_utm: gpd.GeoDataFrame,
    country: str,
    land_use: str,
    params: PoiParams = DEFAULT,
):
    """Build the UTM influence polygons from only the POIs/junctions near target_segs_utm.

    Uses gpd.sjoin(predicate="dwithin") to restrict buffering to the points
    within the influence radius of a target segment.

    target_segs_utm: segments projected to BUFFER_CRS[country].
    Returns a (buffer_poly, iso_poly) tuple, each the unary_union of the relevant
    geometries or None if none exist:
      - buffer_poly: Mapillary VRU buffers + junction buffers + Mapillary
        hospital-sign buffers. MIN_PIECE_M gating applies to it.
      - iso_poly: always None (poi_speed_zones.py applies the POI isochrones
        after stage 2).
    """
    crs = BUFFER_CRS[country]
    poi_radius = POI_BUFFER_M[land_use]
    segs_geom = target_segs_utm[["geometry"]]
    buffer_polys: list = []
    iso_polys: list = []

    # Mapillary VRU points pre-filtered to those near target segments.
    map_pois = load_mapillary_pois(country)
    if len(map_pois) > 0:
        vru_mask = pd.Series(False, index=map_pois.index)
        for flag in _MAPILLARY_VRU_FLAGS:
            if flag in map_pois.columns:
                vru_mask = vru_mask | (map_pois[flag] == True)  # noqa: E712
        vru_pts = map_pois[vru_mask]
        if len(vru_pts) > 0:
            vru_utm = vru_pts[["geometry"]].to_crs(crs)
            nearby = gpd.sjoin(vru_utm, segs_geom, predicate="dwithin", distance=poi_radius)
            local_vru = vru_utm.loc[nearby.index.unique()]
            if len(local_vru) > 0:
                buffer_polys.extend(local_vru.geometry.buffer(poi_radius).tolist())

    def add_point_buffers(points: gpd.GeoDataFrame) -> None:
        if len(points) == 0:
            return
        pts_utm = points[["geometry"]].to_crs(crs)
        nearby = gpd.sjoin(pts_utm, segs_geom, predicate="dwithin", distance=poi_radius)
        local = pts_utm.loc[nearby.index.unique()]
        if len(local) > 0:
            buffer_polys.extend(local.geometry.buffer(poi_radius).tolist())

    enabled = {c.poi_type for c in params.enabled_caps()}

    # A Mapillary hospital sign caps V_safe when hospitals are enabled
    # (exposure_signals._vru_speed_cap), so it joins as a buffer and the
    # MIN_PIECE_M gate applies.
    for poi_type in sorted(enabled):
        if poi_type == "hospital" and "map_is_hospital" in map_pois.columns:
            add_point_buffers(map_pois[map_pois["map_is_hospital"] == True])  # noqa: E712

    # Junction nodes pre-filtered to those near target segments.
    junctions = load_cached_junctions(country)
    if len(junctions) > 0:
        junc_utm = junctions[["geometry"]].to_crs(crs)
        nearby = gpd.sjoin(junc_utm, segs_geom, predicate="dwithin", distance=JUNCTION_BUFFER_M)
        local_junc = junc_utm.loc[nearby.index.unique()]
        if len(local_junc) > 0:
            buffer_polys.extend(local_junc.geometry.buffer(JUNCTION_BUFFER_M).tolist())

    buffer_poly = unary_union(buffer_polys) if buffer_polys else None
    iso_poly = unary_union(iso_polys) if iso_polys else None
    return buffer_poly, iso_poly


def _extract_linestrings(geom) -> list:
    """Normalise a Shapely geometry to a list of non-empty LineStrings."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    if isinstance(geom, GeometryCollection):
        result = []
        for g in geom.geoms:
            if isinstance(g, LineString) and not g.is_empty:
                result.append(g)
            elif isinstance(g, MultiLineString):
                result.extend(gg for gg in g.geoms if isinstance(gg, LineString) and not gg.is_empty)
        return result
    return []


def _make_child(parent_row: dict, child_geom_4326, child_len_m: float,
                parent_len_m: float, child_id: str) -> dict:
    """Inherit all parent columns and override geometry-specific fields."""
    child = dict(parent_row)
    child["geometry"] = child_geom_4326
    child["segment_id"] = child_id
    child["parent_section_id"] = parent_row["segment_id"]  # already normalised string
    child["shape_length"] = child_len_m
    # sample_size_avg and sample_size_total count the probes that passed the
    # segment: every piece of a road was passed by the same probes, so each
    # child keeps the parent's values (dict(parent_row) already copied them).
    return child


def refine_influenced_segments(gdf: gpd.GeoDataFrame, params: PoiParams = DEFAULT) -> gpd.GeoDataFrame:
    """Stage-2 refinement: split influenced segments at the influence-zone boundary.

    For every segment where is_vru or near_junction is True, clips the
    geometry into an "influenced" portion and a "non-influenced" portion.
    A segment is only split when *both* portions are at least MIN_PIECE_M long.
    The POI isochrones are applied later, in poi_speed_zones.py.

    Performance: the morphological closing (safety-side sliver absorption) is
    computed once per (country, land_use) group, and the influence polygon is
    built from the nearby points alone (dwithin pre-filter).

    segment_id normalisation: all rows get "{country}_{original_objectid}" to ensure
    global uniqueness (Thailand and Maharashtra share OBJECTID sequences from 1).
    Children get "{country}_{parent_objectid}-{k}".
    parent_section_id is set for split children; NA for all other rows.
    sample_size_avg and sample_size_total are copied to every child unchanged.
    All other geometry-driven columns are inherited and overwritten by the caller's
    Stage-1 re-run.
    """
    gdf = gdf.copy()

    # Normalise segment_id to globally unique "{country}_{objectid}" strings.
    gdf["segment_id"] = gdf["country"] + "_" + gdf["segment_id"].astype(str)
    if "parent_section_id" not in gdf.columns:
        gdf["parent_section_id"] = pd.NA

    target_mask = (gdf["is_vru"] == True) | (gdf["near_junction"] == True)  # noqa: E712
    if not target_mask.any():
        return gdf

    target_gdf = gdf.loc[target_mask]
    non_target_gdf = gdf.loc[~target_mask]

    split_parent_indices: set = set()
    new_child_dicts: list = []

    for (country, land_use), group in target_gdf.groupby(["country", "land_use"]):
        crs = BUFFER_CRS[country]

        # Project the whole group to UTM once.
        group_utm = group[["geometry"]].to_crs(crs)

        # Build influence polygons from nearby POIs/junctions only (dwithin pre-filter).
        # buffer_poly: Mapillary VRU / junction / hospital-sign buffers (MIN_PIECE_M
        # gating applies). iso_poly: always None (see the module docstring).
        buffer_poly, iso_poly = build_influence_polygon_near(group_utm, country, land_use, params)
        if buffer_poly is None and iso_poly is None:
            continue

        # Morphological closing (safety-side sliver absorption), once per group.
        closed_buffer_poly = (
            buffer_poly.buffer(MIN_PIECE_M / 2).buffer(-MIN_PIECE_M / 2)
            if buffer_poly is not None else None
        )
        cut_poly = unary_union([p for p in (closed_buffer_poly, iso_poly) if p is not None])

        for idx, row in group.iterrows():
            line_utm = group_utm.loc[idx, "geometry"]
            parent_len = line_utm.length
            if parent_len < 1e-6:
                continue

            influenced_parts = _extract_linestrings(line_utm.intersection(cut_poly))
            non_influenced_parts = _extract_linestrings(line_utm.difference(cut_poly))

            influenced_len = sum(p.length for p in influenced_parts)
            non_influenced_len = sum(p.length for p in non_influenced_parts)

            # Split only when both portions reach the minimum piece length.
            if influenced_len < MIN_PIECE_M or non_influenced_len < MIN_PIECE_M:
                continue

            split_parent_indices.add(idx)
            parent_dict = row.to_dict()
            parent_id_str = str(row["segment_id"])  # already "{country}_{objectid}"

            k = 0
            for part_utm in influenced_parts:
                if part_utm.length < 1e-9:
                    continue
                part_4326 = gpd.GeoSeries([part_utm], crs=crs).to_crs("EPSG:4326").iloc[0]
                new_child_dicts.append(
                    _make_child(parent_dict, part_4326, part_utm.length, parent_len,
                                f"{parent_id_str}-{k}")
                )
                k += 1

            for part_utm in non_influenced_parts:
                if part_utm.length < 1e-9:
                    continue
                part_4326 = gpd.GeoSeries([part_utm], crs=crs).to_crs("EPSG:4326").iloc[0]
                new_child_dicts.append(
                    _make_child(parent_dict, part_4326, part_utm.length, parent_len,
                                f"{parent_id_str}-{k}")
                )
                k += 1

    if not new_child_dicts:
        return gdf

    unsplit_target = target_gdf.loc[~target_gdf.index.isin(split_parent_indices)]
    children_gdf = gpd.GeoDataFrame(new_child_dicts, geometry="geometry", crs="EPSG:4326")

    n_parents = len(split_parent_indices)
    n_children = len(children_gdf)
    print(f"[segment_localization] split {n_parents} segments into {n_children} children "
          f"(net +{n_children - n_parents} rows)")

    combined = pd.concat([non_target_gdf, unsplit_target, children_gdf], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
