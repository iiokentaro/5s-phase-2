"""Fetch Overture transportation road segments for a region GeoJSON.

Default is S3 bbox extract, then keep original geometries that
intersect the region polygon (no clip). The file in data/raw is the
polygon-filtered result. Crossing the boundary does not cut the line.

The region file is an argument, not a constant. All segment columns except
`sources` are kept.

Usage
-----
    # repo root
    source .venv/bin/activate
    python src/fetch_overture_roads.py data/raw/INDIA-Maharashtra.geojson
    python src/fetch_overture_roads.py --polygon-filter \
      --roads data/raw/INDIA-Maharashtra_overture_roads.parquet \
      --boundary data/raw/INDIA-Maharashtra.geojson
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

DEFAULT_RELEASE = "2026-08-19.0"
S3_REGION = "us-west-2"
OVERTURE_SEGMENTS = (
    "s3://overturemaps-us-west-2/release/{release}/theme=transportation/type=segment/*"
)
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
# Current transportation.segment schema minus sources (large, unused here).
ROAD_COLUMNS = "* EXCLUDE (sources)"
CLIP_BATCH_SIZE = 20_000
MAX_PIECE_VERTICES = 256
MAX_SUBDIVIDE_DEPTH = 16


def _sql_str(value: str | Path) -> str:
    return str(value).replace("'", "''")


def connect() -> duckdb.DuckDBPyConnection:
    spill = INTERIM_DIR / "duckdb"
    spill.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial;")
    con.sql("INSTALL httpfs; LOAD httpfs;")
    con.sql(f"SET s3_region='{S3_REGION}';")
    con.sql("SET preserve_insertion_order=false;")
    con.sql(f"SET temp_directory='{_sql_str(spill.resolve())}';")
    return con


def region_bounds(
    con: duckdb.DuckDBPyConnection, boundary_path: Path
) -> dict[str, float]:
    """Load the GeoJSON, union features, store as temp table `region`."""
    path = _sql_str(boundary_path.resolve())
    con.sql(
        f"""
        CREATE OR REPLACE TEMP TABLE region AS
        SELECT ST_SetCRS(ST_Union_Agg(geom), 'OGC:CRS84') AS geom
        FROM ST_Read('{path}')
        """
    )
    row = con.sql(
        """
        SELECT
            ST_XMin(geom) AS min_lon,
            ST_YMin(geom) AS min_lat,
            ST_XMax(geom) AS max_lon,
            ST_YMax(geom) AS max_lat
        FROM region
        """
    ).fetchone()
    if row is None or any(v is None for v in row):
        raise ValueError(f"no geometry in {boundary_path}")
    return {
        "min_lon": float(row[0]),
        "min_lat": float(row[1]),
        "max_lon": float(row[2]),
        "max_lat": float(row[3]),
    }


def _flatten_polygons(geom):
    if geom is None or geom.is_empty:
        return
    t = geom.geom_type
    if t == "Polygon":
        yield geom
    elif t in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _flatten_polygons(part)


def subdivide_region(region, max_vertices: int = MAX_PIECE_VERTICES):
    """Split a high-vertex polygon into STRtree-friendly pieces."""
    pieces: list = []

    def rec(geom, depth: int) -> None:
        for poly in _flatten_polygons(geom):
            n = int(shapely.get_num_coordinates(poly))
            if n <= max_vertices or depth >= MAX_SUBDIVIDE_DEPTH:
                pieces.append(poly)
                continue
            minx, miny, maxx, maxy = poly.bounds
            if maxx - minx >= maxy - miny:
                mid = (minx + maxx) / 2.0
                left = shapely.box(minx, miny, mid, maxy)
                right = shapely.box(mid, miny, maxx, maxy)
                rec(poly.intersection(left), depth + 1)
                rec(poly.intersection(right), depth + 1)
            else:
                mid = (miny + maxy) / 2.0
                bottom = shapely.box(minx, miny, maxx, mid)
                top = shapely.box(minx, mid, maxx, maxy)
                rec(poly.intersection(bottom), depth + 1)
                rec(poly.intersection(top), depth + 1)

    rec(shapely.make_valid(region), 0)
    if not pieces:
        raise ValueError("region subdivided to empty")
    return pieces


def default_polygon_output_path(roads_path: Path) -> Path:
    return roads_path.with_name(f"{roads_path.stem}_polygon.parquet")


def filter_to_polygon(
    roads_path: Path,
    boundary_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Keep bbox-extract rows whose geometry intersects the region polygon.

    Geometries are not clipped. DuckDB is used only to concatenate chunks.
    """
    roads_path = Path(roads_path)
    boundary_path = Path(boundary_path)
    if not roads_path.is_file():
        raise FileNotFoundError(roads_path)
    if not boundary_path.is_file():
        raise FileNotFoundError(boundary_path)
    output_path = (
        Path(output_path) if output_path else default_polygon_output_path(roads_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    region = _load_region_polygon(boundary_path)
    pieces = subdivide_region(region)
    print(f"Subdivided region into {len(pieces)} pieces", flush=True)
    tree = shapely.STRtree(pieces)

    pf = pq.ParquetFile(roads_path)
    schema = pf.schema_arrow
    if output_path.exists():
        output_path.unlink()

    n_in = 0
    n_keep = 0
    writer = None
    try:
        for i, batch in enumerate(pf.iter_batches(batch_size=CLIP_BATCH_SIZE)):
            table = batch.to_pandas()
            n_in += len(table)
            if table.empty:
                continue
            geoms = shapely.from_wkb(table["geometry"].to_numpy())
            pairs = tree.query(geoms, predicate="intersects")
            if pairs.size == 0:
                continue
            hit_idx = np.unique(pairs[0])
            kept = table.iloc[hit_idx]
            arrow = pa.Table.from_pandas(kept, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, arrow.schema)
            writer.write_table(arrow)
            n_keep += len(kept)
            if (i + 1) % 10 == 0:
                print(f"  batch {i + 1}, kept {n_keep} / {n_in}", flush=True)
        if writer is None:
            raise RuntimeError(f"no roads intersected {boundary_path}")
        print(f"Saved {output_path} ({n_keep} / {n_in} roads)", flush=True)
        return output_path
    finally:
        if writer is not None:
            writer.close()


def default_output_path(boundary_path: Path) -> Path:
    return RAW_DIR / f"{boundary_path.stem}_overture_roads.parquet"


def _load_region_polygon(boundary_path: Path):
    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    region = gdf.geometry.union_all()
    if region is None or region.is_empty:
        raise ValueError(f"no geometry in {boundary_path}")
    return region


def refine_to_polygon(
    bbox_path: Path,
    output_path: Path,
    region,
    *,
    clip: bool,
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Intersect (and optionally clip) a bbox extract against the polygon."""
    shapely.prepare(region)
    chunk_dir = INTERIM_DIR / f"{output_path.stem}_chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(bbox_path)
    n_out = 0
    n_rows = 0
    try:
        for i, batch in enumerate(pf.iter_batches(batch_size=CLIP_BATCH_SIZE)):
            table = batch.to_pandas()
            geoms = shapely.from_wkb(table["geometry"].to_numpy())
            hit = np.asarray(shapely.intersects(geoms, region))
            if not hit.any():
                continue
            table = table.loc[hit].reset_index(drop=True)
            geoms = geoms[hit]
            if clip:
                inside = np.asarray(shapely.within(geoms, region))
                to_clip = ~inside
                if to_clip.any():
                    geoms = geoms.copy()
                    geoms[to_clip] = shapely.intersection(geoms[to_clip], region)
                empty = shapely.is_empty(geoms)
                line = np.isin(shapely.get_type_id(geoms), (1, 5))
                keep = (~empty) & line
                if not keep.all():
                    table = table.loc[keep].reset_index(drop=True)
                    geoms = geoms[keep]
                if len(table) == 0:
                    continue
            gdf = gpd.GeoDataFrame(
                table.drop(columns=["geometry"]), geometry=geoms, crs="EPSG:4326"
            )
            path = chunk_dir / f"{i:05d}.parquet"
            gdf.to_parquet(path, index=False)
            n_out += 1
            n_rows += len(gdf)
            if (i + 1) % 10 == 0:
                print(f"  batch {i + 1}, kept {n_rows} roads", flush=True)
        if n_out == 0:
            raise RuntimeError(f"no roads intersected the polygon for {output_path}")
        glob = _sql_str((chunk_dir / "*.parquet").as_posix())
        out = _sql_str(output_path.resolve())
        con.sql(
            f"COPY (SELECT * FROM read_parquet('{glob}')) TO '{out}' (FORMAT PARQUET)"
        )
        print(f"  wrote {n_rows} roads", flush=True)
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


def fetch_overture_roads(
    boundary_path: Path,
    output_path: Path | None = None,
    *,
    release: str = DEFAULT_RELEASE,
    clip: bool = False,
    con: duckdb.DuckDBPyConnection | None = None,
) -> Path:
    boundary_path = Path(boundary_path)
    if not boundary_path.is_file():
        raise FileNotFoundError(boundary_path)
    output_path = Path(output_path) if output_path else default_output_path(boundary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    bbox_path = INTERIM_DIR / f"{output_path.stem}.bbox.parquet"
    if bbox_path.exists():
        bbox_path.unlink()

    own_con = con is None
    if con is None:
        con = connect()
    try:
        bbox = region_bounds(con, boundary_path)
        s3_path = _sql_str(OVERTURE_SEGMENTS.format(release=release))
        bbox_sql = _sql_str(bbox_path.resolve())

        print(f"BBox filter {bbox} from S3 -> {bbox_path}", flush=True)
        con.sql(
            f"""
            COPY (
                SELECT {ROAD_COLUMNS}
                FROM read_parquet('{s3_path}', hive_partitioning=1)
                WHERE subtype = 'road'
                  AND bbox.xmin <= {bbox['max_lon']}
                  AND bbox.xmax >= {bbox['min_lon']}
                  AND bbox.ymin <= {bbox['max_lat']}
                  AND bbox.ymax >= {bbox['min_lat']}
            ) TO '{bbox_sql}' (FORMAT PARQUET)
            """
        )
    finally:
        if own_con:
            con.close()

    try:
        print(f"Polygon filter -> {output_path}", flush=True)
        filter_to_polygon(bbox_path, boundary_path, output_path)
        return output_path
    finally:
        bbox_path.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "boundaries",
        nargs="*",
        type=Path,
        help="Region GeoJSON path(s) for S3 bbox extract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GeoParquet. Allowed only when a single input is given.",
    )
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help=f"Overture release id (default: {DEFAULT_RELEASE})",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="After bbox extract, clip geometries to the polygon (slow).",
    )
    parser.add_argument(
        "--polygon-filter",
        action="store_true",
        help="Filter a local bbox parquet by polygon intersects (no S3, no clip).",
    )
    parser.add_argument(
        "--roads",
        type=Path,
        default=None,
        help="Local GeoParquet for --polygon-filter.",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=None,
        help="Region GeoJSON for --polygon-filter.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.polygon_filter:
        if args.roads is None or args.boundary is None:
            raise SystemExit("--polygon-filter requires --roads and --boundary")
        out = args.output
        print(f"Polygon-filter {args.roads} with {args.boundary}", flush=True)
        filter_to_polygon(args.roads, args.boundary, out)
        return
    if not args.boundaries:
        raise SystemExit("pass region GeoJSON path(s), or use --polygon-filter")
    if args.output is not None and len(args.boundaries) != 1:
        raise SystemExit("--output requires exactly one boundary GeoJSON")
    clip = args.clip
    for boundary in args.boundaries:
        out = args.output if args.output is not None else default_output_path(boundary)
        print(f"Downloading roads for {boundary} -> {out}", flush=True)
        fetch_overture_roads(
            boundary,
            out,
            release=args.release,
            clip=clip,
        )


if __name__ == "__main__":
    main()
