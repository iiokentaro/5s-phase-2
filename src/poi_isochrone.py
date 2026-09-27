"""Valhalla pedestrian isochrones: the VRU zone of each POI type that caps V_safe

Every POI type whose speed cap is enabled in poi_params (schools by default)
gets its zone from a pedestrian isochrone: the area within an urban / rural
walking time of the POI, both minutes set per type in poi_params.

Design overview:
  1. poi_origins()     — normalise the type's POIs (OSM ∪ Overture) to a single point
  2. snap_origins()    — snap onto the nearest network (kept separate from the origin)
  3. build_and_cache() — parallel calls to the Valhalla /isochrone API →
                         corridor buffer → save as GeoParquet

Cache:
  data/processed/isochrones/{poi_type}_{country}_u{urban}_r{rural}.parquet, one
  file per type, country and pair of minutes, with a _meta.json beside it. A
  file already on disk and newer than the POI inputs (is_current) is reused,
  so Valhalla is started only for what is missing. valhalla_service.serving
  starts the right country's container for that.

Asymmetric design:
  URBAN  : isochrone only
  RURAL  : isochrone union RURAL_FLOOR_M circular buffer (safety-side floor)

Fallbacks (all safety-side):
  (1) snap_dist_m > MAX_SNAP_M (off-network)
  (2) Valhalla response error / timeout
  (3) area is still 0 after applying the corridor
  -> all fall back to the RURAL_FLOOR_M circular buffer (source='buffer_fallback')
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree

sys.path.insert(0, "src")
from poi_params import DEFAULT, PoiParams
from poi_sources import BUFFER_CRS, load_pois, poi_candidates, poi_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #
RURAL_FLOOR_M = 200           # rural safety-side floor circular-buffer radius (m)
ISOCHRONE_CORRIDOR_M = 15     # corridor width against degenerate geometry (m)
MAX_SNAP_M = 50               # maximum snap distance (m)
PARALLEL_WORKERS = 16         # number of parallel Valhalla API requests
VALHALLA_URL = "http://localhost:8003/isochrone"
REQUEST_TIMEOUT = 30          # per-request timeout (seconds)

PROCESSED_DIR = Path("data/processed")
ISOCHRONE_DIR = PROCESSED_DIR / "isochrones"
# The files whose POIs are the isochrone origins (poi_sources.poi_candidates).
POI_INPUTS = ("osm_pois", "overture_pois")


def cache_path(country: str, poi_type: str, minutes_urban: int, minutes_rural: int,
               iso_dir: Path = ISOCHRONE_DIR) -> Path:
    return Path(iso_dir) / f"{poi_type}_{country}_u{minutes_urban}_r{minutes_rural}.parquet"


def meta_path(parquet: Path) -> Path:
    return parquet.with_name(parquet.stem + "_meta.json")


def is_current(country: str, poi_type: str, minutes_urban: int, minutes_rural: int,
               iso_dir: Path = ISOCHRONE_DIR, processed_dir: Path = PROCESSED_DIR) -> bool:
    """The cache exists and was built after the POI files its origins come from."""
    iso = cache_path(country, poi_type, minutes_urban, minutes_rural, iso_dir)
    if not iso.exists():
        return False
    inputs = [Path(processed_dir) / f"{name}_{country}.parquet" for name in POI_INPUTS]
    return all(not p.exists() or iso.stat().st_mtime >= p.stat().st_mtime for p in inputs)


# --------------------------------------------------------------------------- #
# poi_origins
# --------------------------------------------------------------------------- #

def poi_origins(country: str, poi_type: str) -> gpd.GeoDataFrame:
    """Normalise every candidate POI of `poi_type` (poi_sources.poi_candidates:
    all OSM ones plus every fetched Overture one) to a single isochrone origin
    point. Building for the candidates, not for one parameter set's union, is
    what lets the confidence rule change without Valhalla (see
    load_selected_isochrones).

    - Point       : used as-is
    - Polygon etc.: representative_point() (guaranteed to be inside the polygon)

    poi_id is "<source>:<source_id>" (e.g. "osm:way/123", "overture:<GERS id>").

    Also fetches land_use (URBAN/RURAL) via sjoin_nearest against segments_v_safe.parquet.
    Defaults to RURAL if no segments exist (safety-side).

    Returns
    -------
    GeoDataFrame (geometry = origin_geometry, EPSG:4326)
      columns: poi_id, source, origin_geometry, land_use
    """
    pois = poi_candidates(country, poi_type).copy()

    if pois.empty:
        log.warning("%s: 0 %s POIs found", country, poi_type)
        return gpd.GeoDataFrame(
            columns=["poi_id", "source", "origin_geometry", "land_use"],
            geometry="origin_geometry",
            crs="EPSG:4326",
        )

    def _to_point(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "Point":
            return geom
        return geom.representative_point()

    pois["origin_geometry"] = pois["geometry"].apply(_to_point)
    pois = pois[pois["origin_geometry"].notna()].copy()
    pois["poi_id"] = poi_ids(pois)

    # Fetch land_use from segments_v_safe.parquet via sjoin_nearest
    segs_path = PROCESSED_DIR / "segments_v_safe.parquet"
    land_use_series = pd.Series("RURAL", index=pois.index, name="land_use")
    try:
        segs = gpd.read_parquet(segs_path, columns=["country", "land_use", "geometry"])
        segs_c = segs[segs["country"] == country][["land_use", "geometry"]].copy()
        if not segs_c.empty:
            pts = gpd.GeoDataFrame(
                {"poi_id": pois["poi_id"]},
                geometry=pois["origin_geometry"].values,
                crs="EPSG:4326",
                index=pois.index,
            )
            joined = gpd.sjoin_nearest(pts, segs_c, how="left")
            # sjoin_nearest can return duplicate rows for multiple equidistant matches; keep only the first
            joined = joined[~joined.index.duplicated(keep="first")]
            land_use_series = joined["land_use"].fillna("RURAL")
    except Exception as exc:
        log.warning("failed to fetch land_use (falling back to RURAL): %s", exc)

    pois["land_use"] = land_use_series.values

    gdf = gpd.GeoDataFrame(
        {"poi_id": pois["poi_id"].values,
         "source": pois["source"].values,
         "land_use": pois["land_use"].values,
         "origin_geometry": pois["origin_geometry"].values},
        geometry="origin_geometry",
        crs="EPSG:4326",
    )
    log.info("%s: %d pois (URBAN %d / RURAL %d)",
             country, len(gdf),
             (gdf["land_use"] == "URBAN").sum(),
             (gdf["land_use"] == "RURAL").sum())
    return gdf


# --------------------------------------------------------------------------- #
# snap_origins
# --------------------------------------------------------------------------- #

def snap_origins(origins: gpd.GeoDataFrame, country: str) -> gpd.GeoDataFrame:
    """Snap each POI point onto the nearest network (kept as a separate layer).

    Snap target: nearest point on osm_ways_{country}.parquet -- every
    `highway=*` way, footway/path/pedestrian/steps included.

    Added columns:
      snapped_geometry  : the snapped point (WGS84)
      snap_dist_m       : snap distance (m, after UTM projection)
      snapped           : snap_dist_m <= MAX_SNAP_M

    origin_geometry is kept (for QA / traceability).
    """
    origins = origins.copy()
    crs = BUFFER_CRS[country]

    # load the network to snap onto
    way_gdfs = []
    try:
        import osm_ways

        way_gdfs.append(osm_ways.load_ways(country)[["geometry"]])
    except Exception as exc:
        # Loud on purpose: snapping onto nothing leaves every origin where it
        # was, and the isochrones come out subtly wrong with no error. Build
        # the layer with:
        #     python src/osm_ways.py --country <country>
        log.warning("%s: could not load osm_ways (%s); origins will NOT be "
                    "snapped and the isochrones will be built from unsnapped "
                    "POI points", country, exc)

    if not way_gdfs:
        log.warning("%s: no road network found -> no snapping", country)
        origins["snapped_geometry"] = origins["origin_geometry"]
        origins["snap_dist_m"] = float("inf")
        origins["snapped"] = False
        return origins

    ways = pd.concat([g[["geometry"]] for g in way_gdfs], ignore_index=True)
    ways_utm = ways.to_crs(crs)
    way_geoms = ways_utm.geometry.tolist()
    tree = STRtree(way_geoms)

    # project POI points to UTM
    pois_utm = origins.set_geometry("origin_geometry").to_crs(crs)

    snap_pts_utm = []
    snap_dists = []

    log.info("%s: %d snap-target road segments, %d pois", country, len(way_geoms), len(pois_utm))
    for pt in pois_utm["origin_geometry"]:
        nearest_idx = tree.nearest(pt)
        nearest_geom = way_geoms[nearest_idx]
        snap_pt, _ = nearest_points(nearest_geom, pt)
        snap_dist = pt.distance(snap_pt)
        snap_pts_utm.append(snap_pt)
        snap_dists.append(snap_dist)

    # UTM snap points -> WGS84
    snap_gdf_utm = gpd.GeoDataFrame({"geometry": snap_pts_utm}, crs=crs)
    snap_gdf_wgs84 = snap_gdf_utm.to_crs("EPSG:4326")

    origins["snapped_geometry"] = snap_gdf_wgs84.geometry.values
    origins["snap_dist_m"] = snap_dists
    origins["snapped"] = [d <= MAX_SNAP_M for d in snap_dists]

    n_snapped = sum(origins["snapped"])
    log.info("%s: snap succeeded %d / %d (MAX_SNAP_M=%dm)",
             country, n_snapped, len(origins), MAX_SNAP_M)
    log.info("%s: snap_dist_m quantiles — p50=%.1fm p90=%.1fm p99=%.1fm",
             country,
             pd.Series(snap_dists).quantile(0.50),
             pd.Series(snap_dists).quantile(0.90),
             pd.Series(snap_dists).quantile(0.99))
    return origins


# --------------------------------------------------------------------------- #
# Valhalla API
# --------------------------------------------------------------------------- #

def _fetch_single_isochrone(
    pt: Point,
    minutes: float,
    valhalla_url: str,
) -> Polygon | None:
    """Call the Valhalla /isochrone API for a single point and return a Shapely Polygon.

    Returns None on failure (the caller decides the fallback).
    search_cutoff is set to MAX_SNAP_M to suppress mis-snapping.
    """
    payload = {
        "locations": [{"lat": pt.y, "lon": pt.x}],
        "costing": "pedestrian",
        "costing_options": {"pedestrian": {"use_ferry": 0, "use_living_streets": 1}},
        "contours": [{"time": minutes}],
        "polygons": True,
        "denoise": 1.0,
        "generalize": 0,  # vertex reduction is done via simplify() in to_zone
    }
    try:
        resp = requests.post(valhalla_url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        fc = resp.json()
        features = fc.get("features", [])
        if not features:
            return None
        geom_dict = features[0].get("geometry")
        if geom_dict is None:
            return None
        from shapely.geometry import shape
        geom = shape(geom_dict)
        return geom if not geom.is_empty else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# to_zone
# --------------------------------------------------------------------------- #

def to_zone(
    iso_geom,           # Polygon / LineString / None from Valhalla (WGS84)
    fallback_pt: Point,  # WGS84: snapped point or original POI point
    land_use: str,
    crs: str,           # UTM CRS
) -> tuple[Polygon, str]:
    """Normalise the Valhalla response geometry to a "corridor polygon with area > 0 (WGS84)".

    Returns
    -------
    (polygon_wgs84, source)
      source: 'isochrone' | 'buffer_fallback'
    """
    floor_pt_utm = (
        gpd.GeoDataFrame({"geometry": [fallback_pt]}, crs="EPSG:4326")
        .to_crs(crs)
        .geometry[0]
    )

    if iso_geom is None or iso_geom.is_empty:
        # full fallback: floor buffer
        zone_utm = floor_pt_utm.buffer(RURAL_FLOOR_M)
        zone_wgs84 = (
            gpd.GeoDataFrame({"geometry": [zone_utm]}, crs=crs)
            .to_crs("EPSG:4326")
            .geometry[0]
        )
        return zone_wgs84, "buffer_fallback"

    # project the Valhalla response to UTM and turn it into a corridor
    iso_utm = (
        gpd.GeoDataFrame({"geometry": [iso_geom]}, crs="EPSG:4326")
        .to_crs(crs)
        .geometry[0]
    )
    iso_utm = iso_utm.buffer(ISOCHRONE_CORRIDOR_M)

    # fall back if the area is still 0 after the corridor (e.g. collapsed to a point geometry)
    if iso_utm.is_empty or iso_utm.area <= 0:
        zone_utm = floor_pt_utm.buffer(RURAL_FLOOR_M)
        zone_wgs84 = (
            gpd.GeoDataFrame({"geometry": [zone_utm]}, crs=crs)
            .to_crs("EPSG:4326")
            .geometry[0]
        )
        return zone_wgs84, "buffer_fallback"

    # RURAL: isochrone union floor buffer (safety-side)
    if land_use == "RURAL":
        floor_utm = floor_pt_utm.buffer(RURAL_FLOOR_M)
        iso_utm = unary_union([iso_utm, floor_utm])

    # simplify by 2m (vertex reduction; minor impact on intersection-test precision)
    iso_utm = iso_utm.simplify(2)

    # convert back to WGS84
    zone_wgs84 = (
        gpd.GeoDataFrame({"geometry": [iso_utm]}, crs=crs)
        .to_crs("EPSG:4326")
        .geometry[0]
    )
    return zone_wgs84, "isochrone"


# --------------------------------------------------------------------------- #
# build_and_cache
# --------------------------------------------------------------------------- #

def _process_one_poi(args: tuple) -> dict:
    """Per-POI processing, executed in the thread pool."""
    row, valhalla_url, crs, minutes_by_land_use = args
    poi_id = row["poi_id"]
    land_use = row["land_use"]
    origin_pt = row["origin_geometry"]
    snapped = row["snapped"]
    snap_pt = row.get("snapped_geometry")
    snap_dist = row.get("snap_dist_m", float("inf"))
    minutes = minutes_by_land_use[land_use]

    # Valhalla is always queried with origin_pt.
    # Because Valhalla internally snaps against the whole PBF (including residential
    # streets and footways), an isochrone is often still obtained even when our
    # pre-snap failed.
    # snap_dist_m / snapped are kept only for QA / traceability.
    iso_geom = _fetch_single_isochrone(origin_pt, minutes, valhalla_url)

    # fallback point: the snap point if one exists, otherwise the origin
    fallback_pt = snap_pt if (snap_pt is not None) else origin_pt
    zone_geom, source = to_zone(iso_geom, fallback_pt, land_use, crs)

    return {
        "poi_id": poi_id,
        "poi_source": row["source"],
        "geometry": zone_geom,
        "origin_geometry": origin_pt,
        "snapped_geometry": snap_pt,
        "snap_dist_m": snap_dist,
        "snapped": snapped,
        "minutes": minutes,
        "land_use": land_use,
        "source": source,
    }


# More fallbacks than this means Valhalla is not answering for this country
# (wrong container, tiles still loading), not that a few POIs sit off-network.
# Writing the cache anyway would turn the whole zone into 200 m circles.
MAX_FALLBACK_SHARE = 0.5

OUTPUT_COLUMNS = ["poi_id", "poi_source", "geometry", "origin_geometry", "snapped_geometry",
                  "snap_dist_m", "snapped", "minutes", "land_use", "source"]


def build_and_cache(
    country: str,
    poi_type: str,
    minutes_urban: int,
    minutes_rural: int,
    valhalla_url: str = VALHALLA_URL,
    force: bool = False,
    log_fn=None,
) -> gpd.GeoDataFrame:
    """Generate the isochrones of every candidate `poi_type` POI and save them to
    GeoParquet (idempotent).

    Reads and returns the parquet if it already exists (force=True regenerates it).
    Needs a Valhalla serving `country` (valhalla_service.serving). Raises
    RuntimeError, writing nothing, when more than MAX_FALLBACK_SHARE of the POIs
    get no isochrone.

    Output schema:
      poi_id, poi_source, geometry(Polygon WGS84), origin_geometry, snapped_geometry,
      snap_dist_m, snapped, minutes, land_use, source
    """
    say = log_fn or (lambda msg: log.info("%s", msg))
    out_path = cache_path(country, poi_type, minutes_urban, minutes_rural)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        say(f"[{country}] {poi_type}: loading cached isochrones: {out_path}")
        return _read(out_path)

    crs = BUFFER_CRS[country]

    # Step 1: build POI origin points
    origins = poi_origins(country, poi_type)
    if origins.empty:
        say(f"[{country}] {poi_type}: 0 POIs found, saving an empty parquet")
        empty = gpd.GeoDataFrame(columns=OUTPUT_COLUMNS, geometry="geometry", crs="EPSG:4326")
        empty.to_parquet(out_path)
        return empty

    # Step 2: snap onto the network
    origins = snap_origins(origins, country)

    # Step 3: parallel Valhalla isochrone fetch + corridor conversion
    minutes_by_land_use = {"URBAN": minutes_urban, "RURAL": minutes_rural}
    args_list = [
        (row, valhalla_url, crs, minutes_by_land_use)
        for _, row in origins.iterrows()
    ]

    say(f"[{country}] {poi_type}: starting Valhalla isochrone fetch "
        f"({len(args_list)} POIs, {PARALLEL_WORKERS} workers, "
        f"urban {minutes_urban} min / rural {minutes_rural} min)")

    results = []
    fallback_count = 0
    report_every = max(500, len(args_list) // 20)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_process_one_poi, a): i for i, a in enumerate(args_list)}
        for done_count, future in enumerate(as_completed(futures), 1):
            try:
                r = future.result()
                results.append(r)
                if r["source"] == "buffer_fallback":
                    fallback_count += 1
            except Exception as exc:
                log.error("error processing %s POI: %s", poi_type, exc)
            if done_count % report_every == 0:
                say(f"  {done_count} / {len(args_list)} done ({fallback_count} fallbacks)")

    say(f"[{country}] {poi_type}: isochrone {len(results) - fallback_count} / "
        f"buffer_fallback {fallback_count}")
    if not results or fallback_count > MAX_FALLBACK_SHARE * len(results):
        raise RuntimeError(
            f"{country} {poi_type}: {fallback_count} of {len(results)} POIs got no isochrone; "
            f"is the Valhalla at {valhalla_url} serving {country}? Nothing was written."
        )

    # Step 4: build GeoDataFrame and save
    result_gdf = gpd.GeoDataFrame(results, geometry="geometry", crs="EPSG:4326")

    # origin_geometry / snapped_geometry columns hold WGS84 Points (separate from the geometry column)
    # GeoParquet doesn't support multiple geometry columns, so auxiliary columns are saved as WKT strings
    for col in ("origin_geometry", "snapped_geometry"):
        result_gdf[col] = result_gdf[col].apply(
            lambda g: g.wkt if g is not None else None
        )

    result_gdf.to_parquet(out_path)
    say(f"[{country}] {poi_type}: saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    meta = {
        "country": country,
        "poi_type": poi_type,
        "ISOCHRONE_MIN_URBAN": minutes_urban,
        "ISOCHRONE_MIN_RURAL": minutes_rural,
        "RURAL_FLOOR_M": RURAL_FLOOR_M,
        "ISOCHRONE_CORRIDOR_M": ISOCHRONE_CORRIDOR_M,
        "MAX_SNAP_M": MAX_SNAP_M,
        "PARALLEL_WORKERS": PARALLEL_WORKERS,
        "valhalla_url": valhalla_url,
        "n_pois": len(result_gdf),
        "n_pois_by_poi_source": {k: int(v) for k, v in result_gdf["poi_source"].value_counts().items()},
        "n_isochrone": len(result_gdf) - fallback_count,
        "n_buffer_fallback": fallback_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path(out_path).write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return result_gdf


def _read(path: Path) -> gpd.GeoDataFrame:
    iso = gpd.read_parquet(path)
    # Files written before the generalisation to every POI type name the id school_id.
    return iso.rename(columns={"school_id": "poi_id"})


def load_selected_isochrones(country: str, poi_type: str, params: PoiParams = DEFAULT,
                             iso_dir: Path = ISOCHRONE_DIR) -> gpd.GeoDataFrame:
    """The isochrones of the `poi_type` POIs in the POI union under `params`,
    at the minutes `params` sets for that type.

    Raises FileNotFoundError when that cache does not exist: the zone the user
    asked for cannot be computed, and a silent fallback would report a V_safe
    for a different zone. A selected POI with no isochrone -- the parquet
    predates a fetch that added it -- gets the RURAL_FLOOR_M circle, entered
    once per land_use so it reaches both tracks, and a warning: the zone is
    never silently dropped.
    """
    cap = params.cap(poi_type)
    path = cache_path(country, poi_type, cap.iso_min_urban, cap.iso_min_rural, iso_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: the {poi_type} isochrones for {country} at "
            f"{cap.iso_min_urban} min (urban) / {cap.iso_min_rural} min (rural) have not been built. "
            "Run the Full or Complete preset, which builds them with Valhalla."
        )
    iso = _read(path)
    pois = load_pois(country, params)
    pois = pois[pois[f"is_{poi_type}"]]
    ids = poi_ids(pois)
    selected = iso[iso["poi_id"].isin(set(ids))]
    missing = pois[~ids.isin(set(iso["poi_id"]))]
    if missing.empty:
        return selected
    log.warning("%s: %d selected %s POIs have no isochrone; using a %d m circle for them "
                "(rebuild with `python src/poi_isochrone.py --country %s --poi-type %s --force`)",
                country, len(missing), poi_type, RURAL_FLOOR_M, country, poi_type)
    crs = BUFFER_CRS[country]
    circles = missing.geometry.to_crs(crs).representative_point().buffer(RURAL_FLOOR_M).to_crs("EPSG:4326")
    fallback = pd.concat([
        gpd.GeoDataFrame({"poi_id": poi_ids(missing).values, "land_use": lu, "source": "buffer_fallback"},
                         geometry=circles.values, crs="EPSG:4326")
        for lu in ("URBAN", "RURAL")
    ], ignore_index=True)
    return gpd.GeoDataFrame(pd.concat([selected, fallback], ignore_index=True), geometry="geometry", crs="EPSG:4326")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    global MAX_SNAP_M
    from poi_params import DEFAULT_ISO_MIN_RURAL, DEFAULT_ISO_MIN_URBAN, POI_TYPES
    parser = argparse.ArgumentParser(
        description="Generate and cache a POI type's VRU zone using Valhalla pedestrian isochrones"
    )
    parser.add_argument(
        "--country",
        choices=["thailand", "maharashtra"],
        default=None,
        help="target country (both if omitted); Valhalla must be serving it",
    )
    parser.add_argument("--poi-type", choices=POI_TYPES, default="school")
    parser.add_argument("--minutes-urban", type=int, default=DEFAULT_ISO_MIN_URBAN)
    parser.add_argument("--minutes-rural", type=int, default=DEFAULT_ISO_MIN_RURAL)
    parser.add_argument(
        "--valhalla-url",
        default=VALHALLA_URL,
        help=f"Valhalla /isochrone endpoint (default: {VALHALLA_URL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate even if the parquet already exists",
    )
    parser.add_argument(
        "--max-snap-m",
        type=float,
        default=MAX_SNAP_M,
        help=f"maximum snap distance in m (default: {MAX_SNAP_M})",
    )
    args = parser.parse_args()
    MAX_SNAP_M = args.max_snap_m

    countries = [args.country] if args.country else ["thailand", "maharashtra"]
    for country in countries:
        log.info("=== %s ===", country)
        build_and_cache(country, args.poi_type, args.minutes_urban, args.minutes_rural,
                        valhalla_url=args.valhalla_url, force=args.force)


if __name__ == "__main__":
    main()
