"""POI V_safe zones: set the V_safe that POIs (OSM ∪ Overture) impose, on the
final segment geometry after stage 2.

Every POI type `params` enables has a Valhalla walking isochrone per POI
(poi_isochrone.py; RURAL ones include the RURAL_FLOOR_M circle). URBAN and
RURAL segments use the same isochrones.

  ① filter_isochrones   drop the isochrones that touch no segment without
                         access control; saved per country and type.
  ② union_by_type       union the isochrones of one type.
  ③ union_by_speed      union the types that share a speed. Each connected
                         polygon of the result is one zone row, with one bool
                         reason column per type that is True when the zone
                         contains an isochrone of that type; saved per country
                         and speed.
  ④ apply_speed_zone    for each speed, highest first, overlay its zones on the
                         segments without access control:
                           - covered share >= FULL_SEGMENT_SHARE: the whole
                             segment takes the zone;
                           - 0 < share < FULL_SEGMENT_SHARE: the segment is cut
                             into the covered pieces and the rest, and only the
                             covered pieces take the zone.
                         "Take the zone": V_safe above the zone speed drops to
                         it; V_safe equal to it stays; either way the zone's
                         reason columns are ORed in. A segment whose V_safe is
                         already below the zone speed keeps its geometry and
                         reasons. Going from the highest speed down
                         means the lowest zone speed always wins.

Pieces keep every parent column (sample_size_avg and sample_size_total
included: they count the probes that passed the road, which is the same for
every piece), except
geometry, segment_id ("{parent}-p{speed}-{k}") and shape_length. The id a row
had when this step started is kept in `poi_parent_id`, which
sandwich_segments.py uses to join pieces back together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS
from poi_isochrone import load_selected_isochrones
from poi_params import DEFAULT, PoiParams
from segment_localization import _extract_linestrings

# A zone covering at least this share of a segment's length lowers the whole
# segment; below it, only the covered pieces.
FULL_SEGMENT_SHARE = 0.8

# The reason column each POI type sets.
REASON_COLS = {
    "school": "is_school_zone",
    "hospital": "is_near_hospital",
    "marketplace": "is_near_marketplace",
    "shop": "is_near_shop",
    "bus_stop": "is_near_bus_stop",
}

ZONE_DIR = Path("data/processed/poi_zones")

# Pieces shorter than this (m) are floating-point leftovers of the cut.
_MIN_PIECE_M = 1e-9


# --------------------------------------------------------------------------- #
# ① ② ③  zones
# --------------------------------------------------------------------------- #

def filter_isochrones(iso_utm: gpd.GeoDataFrame, segs_utm: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """① The isochrones that intersect at least one of `segs_utm` (same CRS)."""
    if iso_utm.empty or segs_utm.empty:
        return iso_utm.iloc[0:0]
    hits = gpd.sjoin(iso_utm[["geometry"]], segs_utm[["geometry"]], predicate="intersects")
    return iso_utm.loc[iso_utm.index.isin(hits.index.unique())]


def union_by_type(iso_utm: gpd.GeoDataFrame):
    """② One geometry: the union of every isochrone of one type."""
    return unary_union(iso_utm.geometry.values)


def _polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def union_by_speed(type_isochrones: dict[str, gpd.GeoDataFrame], crs) -> gpd.GeoDataFrame:
    """③ The zones of one speed: one row per connected polygon of the union of
    `type_isochrones` (poi_type -> filtered isochrones, all at that speed),
    with every REASON_COLS column. A type's column is True on a zone that
    contains one of its isochrones."""
    type_unions = [union_by_type(iso) for iso in type_isochrones.values() if not iso.empty]
    parts = _polygons(unary_union(type_unions)) if type_unions else []
    zones = gpd.GeoDataFrame({"geometry": parts}, geometry="geometry", crs=crs)
    for col in REASON_COLS.values():
        zones[col] = False
    for poi_type, iso in type_isochrones.items():
        if iso.empty or zones.empty:
            continue
        hits = gpd.sjoin(zones[["geometry"]], iso[["geometry"]], predicate="intersects")
        zones.loc[hits.index.unique(), REASON_COLS[poi_type]] = True
    return zones


def build_zones(country: str, segs_utm: gpd.GeoDataFrame, params: PoiParams,
                save: bool = True) -> dict[int, gpd.GeoDataFrame]:
    """speed_kmh -> zones (③) for one country, from the enabled POI types.
    `segs_utm` is the country's segments without access control, in
    BUFFER_CRS[country]. Writes ① and ③ under ZONE_DIR when `save`."""
    crs = BUFFER_CRS[country]
    by_speed: dict[int, dict[str, gpd.GeoDataFrame]] = {}
    for cap in params.enabled_caps():
        iso = load_selected_isochrones(country, cap.poi_type, params)
        iso_utm = iso.to_crs(crs)
        kept = filter_isochrones(iso_utm, segs_utm)
        print(f"[poi_zones] {country} {cap.poi_type}: kept {len(kept)} / {len(iso_utm)} isochrones "
              f"touching a segment without access control")
        if save:
            ZONE_DIR.mkdir(parents=True, exist_ok=True)
            out = ZONE_DIR / (f"{cap.poi_type}_{country}_u{cap.iso_min_urban}"
                              f"_r{cap.iso_min_rural}_filtered.parquet")
            _write(kept.to_crs("EPSG:4326"), out)
        by_speed.setdefault(cap.speed_kmh, {})[cap.poi_type] = kept

    zones_by_speed = {}
    for speed, type_iso in by_speed.items():
        zones = union_by_speed(type_iso, crs)
        zones_by_speed[speed] = zones
        print(f"[poi_zones] {country} {speed} km/h ({', '.join(sorted(type_iso))}): {len(zones)} zones")
        if save:
            _write(zones.to_crs("EPSG:4326"), ZONE_DIR / f"vsafe_{speed}kmh_{country}.parquet")
    return zones_by_speed


def _write(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Geometry only plus plain columns: any second geometry column (the
    isochrones carry origin/snapped points) is written as WKT."""
    out = gdf.copy()
    for col in out.columns:
        if col != out.geometry.name and isinstance(out[col].dtype, gpd.array.GeometryDtype):
            out[col] = out[col].to_wkt()
    out.to_parquet(path)


# --------------------------------------------------------------------------- #
# ④  overlay
# --------------------------------------------------------------------------- #

def _take_zone(row: dict, speed: int, reasons: dict[str, bool]) -> None:
    """Apply one zone to a row (or piece) in place. The caller has already
    checked that the row's V_safe is not below `speed`."""
    if row["v_safe"] > speed:
        row["v_safe"] = speed
        row["collision_type"] = "pedestrian"
        row["v_safe_basis"] = f"pedestrian:poi_zone_{speed}kmh"
    for col, hit in reasons.items():
        if hit:
            row[col] = True


def apply_speed_zone(segs: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, speed: int
                     ) -> tuple[gpd.GeoDataFrame, int]:
    """④ Overlay one speed's zones on `segs` (one country, projected to its
    BUFFER_CRS, index 0..n-1). Returns the new frame and how many rows were cut.

    Only rows with is_access_controlled False and v_safe >= speed are touched."""
    if segs.empty or zones.empty:
        return segs, 0
    reason_cols = list(REASON_COLS.values())
    candidates = segs[~segs["is_access_controlled"].fillna(False).astype(bool)
                      & (segs["v_safe"] >= speed)]
    hits = gpd.sjoin(candidates[["geometry"]], zones[["geometry"]], predicate="intersects")
    if hits.empty:
        return segs, 0

    replaced: set = set()
    new_rows: list[dict] = []
    n_cut = 0
    for idx, zone_idx in hits.groupby(level=0)["index_right"]:
        line = segs.at[idx, "geometry"]
        length = line.length
        if length < _MIN_PIECE_M:
            continue
        # Only zones the line actually runs through (not just touches) count,
        # both for the covered length and for the reasons.
        covering = [z for z in zone_idx.unique()
                    if sum(p.length for p in _extract_linestrings(line.intersection(zones.at[z, "geometry"])))
                    > _MIN_PIECE_M]
        if not covering:
            continue
        zone_geom = unary_union(zones.loc[covering, "geometry"].values)
        reasons = zones.loc[covering, reason_cols].any(axis=0).to_dict()
        inside = [p for p in _extract_linestrings(line.intersection(zone_geom)) if p.length > _MIN_PIECE_M]
        share = sum(p.length for p in inside) / length

        row = segs.loc[idx].to_dict()
        if share >= FULL_SEGMENT_SHARE:
            _take_zone(row, speed, reasons)
            new_rows.append(row)
            replaced.add(idx)
            continue

        outside = [p for p in _extract_linestrings(line.difference(zone_geom)) if p.length > _MIN_PIECE_M]
        for k, (piece, covered) in enumerate([(p, True) for p in inside] + [(p, False) for p in outside]):
            child = dict(row)
            child["geometry"] = piece
            child["segment_id"] = f"{row['segment_id']}-p{speed}-{k}"
            child["shape_length"] = piece.length
            if covered:
                _take_zone(child, speed, reasons)
            new_rows.append(child)
        replaced.add(idx)
        n_cut += 1

    if not replaced:
        return segs, 0
    kept = segs.drop(index=list(replaced))
    changed = gpd.GeoDataFrame(new_rows, geometry="geometry", crs=segs.crs)
    out = pd.concat([kept, changed], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=segs.crs), n_cut


def apply_poi_speed_zones(gdf: gpd.GeoDataFrame, params: PoiParams = DEFAULT,
                          save: bool = True) -> gpd.GeoDataFrame:
    """Run ① to ④ for every country in `gdf` (EPSG:4326, after stage 2).

    Adds the REASON_COLS columns (False where no zone applied) and
    `poi_parent_id` (the row's segment_id when this step started)."""
    gdf = gdf.copy()
    gdf["poi_parent_id"] = gdf["segment_id"].astype(str)
    for col in REASON_COLS.values():
        gdf[col] = False
    speeds = sorted({c.speed_kmh for c in params.enabled_caps()}, reverse=True)
    if not speeds:
        print("[poi_zones] no POI type caps V_safe; nothing to do")
        return gdf

    out = []
    for country in gdf["country"].unique():
        crs = BUFFER_CRS[country]
        sub = gdf[gdf["country"] == country]
        # Geometry is carried in UTM through the overlay; untouched rows keep
        # their original EPSG:4326 geometry so endpoints stay bit-identical.
        sub = sub.assign(_geom_4326=sub.geometry.values).reset_index(drop=True)
        sub_utm = sub.to_crs(crs)
        open_segs = sub_utm[~sub_utm["is_access_controlled"].fillna(False).astype(bool)]
        zones_by_speed = build_zones(country, open_segs, params, save=save)

        n_before = len(sub_utm)
        for speed in speeds:
            sub_utm, n_cut = apply_speed_zone(sub_utm, zones_by_speed[speed], speed)
            n_rows = {col: int(sub_utm[col].sum()) for col in REASON_COLS.values() if sub_utm[col].any()}
            print(f"[poi_zones] {country} {speed} km/h: cut {n_cut} segments; rows per reason {n_rows}")

        # Pieces were made in UTM: bring those back to EPSG:4326.
        cut = sub_utm["segment_id"] != sub_utm["poi_parent_id"]
        geom = sub_utm["_geom_4326"].copy()
        geom[cut.values] = sub_utm.loc[cut, "geometry"].to_crs("EPSG:4326").values
        sub_4326 = gpd.GeoDataFrame(sub_utm.drop(columns=["geometry", "_geom_4326"]),
                                    geometry=gpd.GeoSeries(geom.values, crs="EPSG:4326"))
        print(f"[poi_zones] {country}: {n_before} -> {len(sub_4326)} rows")
        out.append(sub_4326)

    result = pd.concat(out, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")
