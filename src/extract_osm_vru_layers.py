"""Extract the OSM layers used to judge VRU separation from one .osm.pbf.

Outputs, all GeoParquet in EPSG:4326, written to {out_dir}/{region}/ and
overwritten on every run:

vru-ways.parquet
    Ways with highway=footway/cycleway/pedestrian/path/bridleway. footway=*,
    cycleway=* and related tags stay as columns.
vru-attributes-on-roads.parquet
    Vehicle-road ways (ATTRIBUTE_ROAD_HIGHWAY_VALUES: ROAD_HIGHWAY_VALUES plus
    highway=service) carrying any sidewalk, sidewalk:*,
    cycleway or cycleway:* key, e.g. sidewalk=both, cycleway=track,
    cycleway:separation=*, cycleway:left:separation=*.
barriers.parquet
    Nodes and ways with barrier=guard_rail/fence/wall/kerb/bollard/hedge.
crossings.parquet
    Ways with footway=crossing or any crossing=* tag, and nodes with
    highway=crossing. A node that belongs to one of those ways is left out:
    the way already represents that crossing.
intersections-and-controls.parquet
    One row per node that is a traffic signal (highway=traffic_signals), a stop
    sign (highway=stop), or an at-grade intersection of vehicle roads.

Every row has osm_type, osm_id, a column per tag listed in LAYER_COLUMNS, the
complete tag set as a JSON string in `tags`, and geometry. Nodes are Points.
Ways are LineStrings, closed ones such as a fence ring included, except ways
tagged area=yes (pedestrian plazas, walled enclosures): osmium export always
builds those as areas, so they are MultiPolygons.

How an intersection is found
    Each vehicle-road way contributes, for each of its nodes, the number of road
    edges that meet there: 1 at either end of the way, 2 in the middle. A node
    whose edges sum to 3 or more across 2 or more ways is an intersection. Two
    ways joined end to end sum to 2 and are only a split point. A bridge over a
    road shares no node with it in OSM, so any node found this way is at grade.

Reading follows osm_way_source_osmium: osmium CLI does the reading, duckdb the
parsing. osmium tags-filter first cuts the file down twice, once to the features
of the five layers and once to the vehicle-road ways for the intersection count.

Usage (from the repository root):

    python src/extract_osm_vru_layers.py data/external/thailand-260621.osm.pbf \\
        data/processed/osm_vru_layers --region thailand
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import duckdb
import geopandas as gpd
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))

from osm_way_source import OSMIUM_BIN, OSMIUM_TIMEOUT  # noqa: E402
from osm_way_source_osmium import _strip_record_separator  # noqa: E402

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "install it with `brew install osmium-tool` (macOS) or "
    "`apt install osmium-tool` (Debian/Ubuntu)"
)

VRU_HIGHWAY_VALUES = ["footway", "cycleway", "pedestrian", "path", "bridleway"]
ROAD_HIGHWAY_VALUES = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    "living_street", "road",
]
BARRIER_VALUES = ["guard_rail", "fence", "wall", "kerb", "bollard", "hedge"]
CONTROL_VALUES = ["traffic_signals", "stop"]

# Everything the five layers can match. Prefix keys (`sidewalk:*`) and bare
# keys (`crossing`) are osmium tags-filter's key-only and prefix forms; the
# exact rule for each layer is applied again in LAYER_WHERE.
LAYER_FILTER = [
    "w/highway=" + ",".join(VRU_HIGHWAY_VALUES),
    "w/sidewalk", "w/sidewalk:*", "w/cycleway", "w/cycleway:*",
    "nw/barrier=" + ",".join(BARRIER_VALUES),
    "w/footway=crossing",
    "w/crossing",
    "n/highway=crossing," + ",".join(CONTROL_VALUES),
]
# vru-attributes-on-roads also takes service roads. The intersection count does
# not: parking aisles and driveways would turn every access point into a junction.
ATTRIBUTE_ROAD_HIGHWAY_VALUES = [*ROAD_HIGHWAY_VALUES, "service"]
# The way half of the crossings rule, used to list the nodes of crossing ways.
CROSSING_WAY_FILTER = ["w/footway=crossing", "w/crossing"]
ROAD_FILTER = ["w/highway=" + ",".join(ROAD_HIGHWAY_VALUES)]

# Every way becomes a LineString, except that osmium still builds area=yes ways
# as areas.
EXPORT_CONFIG = {"linear_tags": True, "area_tags": False}

LAYER_FILES = {
    "vru_ways": "vru-ways.parquet",
    "vru_attributes_on_roads": "vru-attributes-on-roads.parquet",
    "barriers": "barriers.parquet",
    "crossings": "crossings.parquet",
    "intersections_and_controls": "intersections-and-controls.parquet",
}

LAYER_COLUMNS = {
    "vru_ways": [
        "highway", "footway", "cycleway", "bicycle", "foot", "segregated",
        "surface", "access",
    ],
    "vru_attributes_on_roads": [
        "highway", "sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both",
        "cycleway", "cycleway:left", "cycleway:right", "cycleway:both",
        "cycleway:separation",
    ],
    "barriers": ["barrier", "kerb", "height"],
    "crossings": ["highway", "footway", "crossing", "crossing:markings", "crossing_ref"],
}


def _sql_list(values: Sequence[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


LAYER_WHERE = {
    "vru_ways": f"osm_type = 'way' AND highway IN {_sql_list(VRU_HIGHWAY_VALUES)}",
    "vru_attributes_on_roads": (
        f"osm_type = 'way' AND highway IN {_sql_list(ATTRIBUTE_ROAD_HIGHWAY_VALUES)} AND "
        "len(list_filter(json_keys(properties), k -> k IN ('sidewalk', 'cycleway') "
        "OR k LIKE 'sidewalk:%' OR k LIKE 'cycleway:%')) > 0"
    ),
    "barriers": f"barrier IN {_sql_list(BARRIER_VALUES)}",
    "crossings": (
        "(osm_type = 'way' AND (footway = 'crossing' OR crossing IS NOT NULL)) OR "
        "(osm_type = 'node' AND highway = 'crossing' "
        "AND osm_id NOT IN (SELECT node_id FROM crossing_way_nodes))"
    ),
}


# --------------------------------------------------------------------------- #
# osmium
# --------------------------------------------------------------------------- #

def _require_osmium() -> str:
    resolved = shutil.which(OSMIUM_BIN)
    if resolved is None:
        raise RuntimeError(f"`{OSMIUM_BIN}` not found on PATH -- {INSTALL_HINT}")
    return resolved


def _osmium(*args: str) -> None:
    try:
        subprocess.run(
            [OSMIUM_BIN, *args], check=True, capture_output=True, text=True,
            timeout=OSMIUM_TIMEOUT,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"osmium {args[0]} failed: {exc.stderr.strip()}") from exc


def _tags_filter(pbf: Path, expressions: Sequence[str], out: Path) -> None:
    """Referenced nodes are kept (no -R): the ways need them for geometry."""
    _osmium("tags-filter", str(pbf), *expressions, "--overwrite", "-o", str(out))


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

def _export_features(layer_pbf: Path, tmp: Path) -> Path:
    """osmium export -> newline-delimited GeoJSON without record separators.
    Untagged nodes are left out by osmium export, so only the matched nodes and
    tagged way nodes come through."""
    config = tmp / "export-config.json"
    config.write_text(json.dumps(EXPORT_CONFIG))
    seq = tmp / "features.geojsonseq"
    ndjson = tmp / "features.ndjson"
    _osmium("export", str(layer_pbf), "-c", str(config), "-f", "geojsonseq",
            "-a", "type,id", "--overwrite", "-o", str(seq))
    _strip_record_separator(seq, ndjson)
    seq.unlink()
    return ndjson


def _road_way_opl(road_pbf: Path, tmp: Path) -> Path:
    """One OPL line per road way: `w<id> T<tags> Nn<id>x<lon>y<lat>,...`.
    OPL escapes spaces and commas inside tags, so the fields split cleanly on
    spaces. Tagged nodes are written too (4 fields) and are dropped on read."""
    out = tmp / "road-ways.opl"
    _osmium("add-locations-to-ways", str(road_pbf), "-f", "opl,add_metadata=false",
            "--overwrite", "-o", str(out))
    return out


def _crossing_way_opl(layer_pbf: Path, tmp: Path) -> Path:
    """The crossing ways only (-R drops their nodes), one OPL line each:
    `w<id> T<tags> Nn<id>,n<id>,...`."""
    out = tmp / "crossing-ways.opl"
    _osmium("tags-filter", str(layer_pbf), *CROSSING_WAY_FILTER, "-R",
            "-f", "opl,add_metadata=false", "--overwrite", "-o", str(out))
    return out


def _load_crossing_way_nodes(con: duckdb.DuckDBPyConnection, opl: Path) -> None:
    con.execute(f"""
        CREATE TABLE crossing_way_nodes AS
        SELECT DISTINCT CAST(substr(unnest(string_split(substr(nodes, 2), ',')), 2)
                             AS BIGINT) AS node_id
        FROM read_csv(
            '{opl}', delim=' ', header=false, quote='', escape='',
            auto_detect=false, null_padding=true, max_line_size=16777216,
            columns={{'id': 'VARCHAR', 'tags': 'VARCHAR', 'nodes': 'VARCHAR'}}
        )
        WHERE id LIKE 'w%'
    """)


def _load_features(con: duckdb.DuckDBPyConnection, ndjson: Path) -> None:
    con.execute(f"""
        CREATE TABLE features AS
        SELECT properties->>'@type' AS osm_type,
               CAST(properties->>'@id' AS BIGINT) AS osm_id,
               properties->>'highway' AS highway,
               properties->>'footway' AS footway,
               properties->>'crossing' AS crossing,
               properties->>'barrier' AS barrier,
               json_merge_patch(properties, '{{"@type": null, "@id": null}}') AS properties,
               geometry
        FROM read_ndjson(
            '{ndjson}',
            columns={{'type': 'VARCHAR', 'geometry': 'JSON', 'properties': 'JSON'}}
        )
    """)


def _query_layer(con: duckdb.DuckDBPyConnection, layer: str) -> gpd.GeoDataFrame:
    selects = ",\n               ".join(
        "properties->>'{t}' AS \"{t}\"".format(t=tag) for tag in LAYER_COLUMNS[layer]
    )
    table = con.execute(f"""
        SELECT osm_type, osm_id,
               {selects},
               CAST(properties AS VARCHAR) AS tags,
               ST_AsWKB(ST_GeomFromGeoJSON(geometry)) AS wkb
        FROM features
        WHERE {LAYER_WHERE[layer]}
        ORDER BY osm_type, osm_id
    """).to_arrow_table()
    return _to_geodataframe(table)


def _query_intersections_and_controls(
    con: duckdb.DuckDBPyConnection, opl: Path
) -> gpd.GeoDataFrame:
    con.execute(f"""
        CREATE TABLE road_node_uses AS
        WITH ways AS (
            SELECT CAST(substr(id, 2) AS BIGINT) AS way_id,
                   regexp_extract(tags, '(?:^T|,)highway=([^,]*)', 1) AS highway,
                   string_split(substr(nodes, 2), ',') AS refs
            FROM read_csv(
                '{opl}', delim=' ', header=false, quote='', escape='',
                auto_detect=false, null_padding=true, max_line_size=16777216,
                columns={{'id': 'VARCHAR', 'tags': 'VARCHAR', 'nodes': 'VARCHAR',
                          'extra': 'VARCHAR'}}
            )
            WHERE id LIKE 'w%'
        ),
        uses AS (
            SELECT way_id, highway, len(refs) AS n,
                   unnest(refs) AS ref,
                   unnest(generate_series(1, len(refs))) AS pos
            FROM ways
        )
        SELECT way_id, highway,
               CAST(regexp_extract(ref, '^n(\\d+)', 1) AS BIGINT) AS node_id,
               TRY_CAST(regexp_extract(ref, 'x(-?[0-9.]+)', 1) AS DOUBLE) AS lon,
               TRY_CAST(regexp_extract(ref, 'y(-?[0-9.]+)', 1) AS DOUBLE) AS lat,
               CASE WHEN pos = 1 OR pos = n THEN 1 ELSE 2 END AS edges
        FROM uses
    """)
    con.execute("""
        CREATE TABLE road_nodes AS
        SELECT node_id,
               CAST(sum(edges) AS INTEGER) AS degree,
               CAST(count(DISTINCT way_id) AS INTEGER) AS way_count,
               list_sort(list(DISTINCT highway)) AS road_highways,
               any_value(lon) FILTER (WHERE lon IS NOT NULL) AS lon,
               any_value(lat) FILTER (WHERE lat IS NOT NULL) AS lat
        FROM road_node_uses
        GROUP BY node_id
    """)
    missing = con.execute("""
        SELECT count(*) FROM road_nodes
        WHERE degree >= 3 AND way_count >= 2 AND (lon IS NULL OR lat IS NULL)
    """).fetchone()[0]
    if missing:
        log.warning("%d intersection nodes have no location in the file and are dropped", missing)

    table = con.execute(f"""
        WITH controls AS (
            SELECT osm_id AS node_id, highway AS control,
                   CAST(properties AS VARCHAR) AS tags,
                   ST_GeomFromGeoJSON(geometry) AS geom
            FROM features
            WHERE osm_type = 'node' AND highway IN {_sql_list(CONTROL_VALUES)}
        ),
        intersections AS (
            SELECT node_id FROM road_nodes
            WHERE degree >= 3 AND way_count >= 2 AND lon IS NOT NULL AND lat IS NOT NULL
        ),
        kept AS (
            SELECT node_id FROM controls UNION SELECT node_id FROM intersections
        )
        SELECT 'node' AS osm_type,
               k.node_id AS osm_id,
               c.control,
               coalesce(r.degree >= 3 AND r.way_count >= 2, false) AS is_intersection,
               r.degree,
               r.way_count,
               r.road_highways,
               c.tags,
               ST_AsWKB(coalesce(c.geom, ST_Point(r.lon, r.lat))) AS wkb
        FROM kept k
        LEFT JOIN controls c USING (node_id)
        LEFT JOIN road_nodes r USING (node_id)
        ORDER BY k.node_id
    """).to_arrow_table()
    return _to_geodataframe(table)


def _to_geodataframe(table) -> gpd.GeoDataFrame:
    wkb = table.column("wkb").to_pylist()
    frame = table.drop(["wkb"]).to_pandas()
    # object dtype: pandas 3 would otherwise store strings as Arrow
    # large_string, which kepler.gl rejects (same as extract_map_features).
    for col in frame.columns:
        if pd.api.types.is_string_dtype(frame[col]):
            frame[col] = frame[col].astype(object)
    return gpd.GeoDataFrame(frame, geometry=shapely.from_wkb(wkb), crs="EPSG:4326")


def _write(gdf: gpd.GeoDataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    gdf.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def extract_vru_layers(
    pbf_path: str | Path,
    out_dir: str | Path,
    region: str,
    *,
    keep_temp: bool = False,
) -> dict[str, Path]:
    """Write the five layers to {out_dir}/{region}/ and return their paths,
    keyed by the names in LAYER_FILES."""
    pbf_path = Path(pbf_path)
    if not pbf_path.is_file():
        raise FileNotFoundError(pbf_path)
    if not region or region != Path(region).name or region in (".", ".."):
        raise ValueError(f"region must be a plain folder name, got {region!r}.")
    _require_osmium()

    region_dir = Path(out_dir) / region
    region_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="osm_vru_layers_"))
    started = time.monotonic()
    try:
        layer_pbf = tmp / "layers.osm.pbf"
        road_pbf = tmp / "roads.osm.pbf"
        _tags_filter(pbf_path, LAYER_FILTER, layer_pbf)
        _tags_filter(pbf_path, ROAD_FILTER, road_pbf)
        log.info("tags-filter done (%.1f s)", time.monotonic() - started)
        ndjson = _export_features(layer_pbf, tmp)
        opl = _road_way_opl(road_pbf, tmp)
        crossing_opl = _crossing_way_opl(layer_pbf, tmp)
        log.info("export done (%.1f s)", time.monotonic() - started)

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute(f"SET temp_directory = '{tmp / 'duckdb'}'")
        _load_features(con, ndjson)
        _load_crossing_way_nodes(con, crossing_opl)

        paths: dict[str, Path] = {}
        for layer in LAYER_COLUMNS:
            gdf = _query_layer(con, layer)
            paths[layer] = region_dir / LAYER_FILES[layer]
            _write(gdf, paths[layer])
            log.info("%s: %d rows", LAYER_FILES[layer], len(gdf))

        gdf = _query_intersections_and_controls(con, opl)
        paths["intersections_and_controls"] = region_dir / LAYER_FILES["intersections_and_controls"]
        _write(gdf, paths["intersections_and_controls"])
        log.info("%s: %d rows", LAYER_FILES["intersections_and_controls"], len(gdf))
        con.close()
    finally:
        if keep_temp:
            log.info("temporary files kept in %s", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    log.info("finished in %.1f s", time.monotonic() - started)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pbf_path", type=Path, help="input .osm.pbf")
    p.add_argument("out_dir", type=Path, help="folder that receives the REGION folder")
    p.add_argument(
        "--region",
        required=True,
        help="region name for the output folder, e.g. thailand",
    )
    p.add_argument("--keep-temp", action="store_true",
                   help="keep the filtered pbf and export files for inspection")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    paths = extract_vru_layers(args.pbf_path, args.out_dir, args.region,
                               keep_temp=args.keep_temp)
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
