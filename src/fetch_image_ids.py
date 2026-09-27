"""Fetch Mapillary image IDs along a road corridor, then keep those inside it.

Pipeline (call in this order, or run the CLI):

1. fetch_image_ids()   every tile of data/mapillary/tile_cover/{region}_z14.parquet
                       (written by extract_map_features.py for the same region)
                       from the Mapillary "image" vector tile layer (mly1_public),
                       decoded and saved, one file per tile
2. filter_image_ids()  road lines -> corridor polygon (10 m by default; the
                       cover was built with 30 m, so it holds every tile needed); per tile,
                       keep the images inside the corridor and append them to
                       one merged table, written in parts of rows_per_file rows

Why z14 and not z13
-------------------
A probe on 2026-09-25 near Pune returned, for the same place:
    z13 5776/3667   HTTP 200, layers {'sequence': 159}           (no images)
    z14 11552/7334  HTTP 200, layers {'sequence': 72, 'image': 7284}
The image layer exists at z14 only (as the mapillary-python-sdk docstring of
VectorTiles.get_image_layer says, "zoom: 14"). A z13 request does not fail;
it silently returns no images. So the existing z14 tile cover is used and no
z13 cover is written. The same probe returned 7,284 images for one tile in a
single response: vector tiles have no 2,000-image cap and no paging.next.

The SDK's images_in_bbox() is not used: its image_type filter condition
(`is not None or != "all"`, mapillary/controller/image.py) is always true, so
it drops every panoramic image. The SDK parts it is built from are used
directly: VectorTiles.get_image_layer for the URL and
vt2geojson.tools.vt_bytes_to_geojson for decoding.

Requests are spaced as in extract_map_features.py: SigmoidIntervalPacer
learns the gap between request starts from successes and 429 responses and
keeps it at the shortest gap predicted to succeed 95% of the time;
BackoffPolicy spaces retries.

Outputs
-------
data/mapillary/vector_tiles/mly_image_ids/{region}/14/{x}/{y}_{fetched_at}.parquet
    The images of one tile, one row per image (IMAGE_SCHEMA).
data/mapillary/vector_tiles/mly_image_ids/{region}/14/{x}/{y}_{fetched_at}.empty
    Zero-byte marker for a tile without images.
data/mapillary/filtered_image_ids/{region}/{region}_image_ids_{created_at}_part{NNNN}.parquet
    The images inside the 10 m corridor as GeoParquet, one row per image:
    the OUTPUT_SCHEMA columns plus a WKB point `geometry` column (EPSG:4326),
    so QGIS and kepler.gl both open each file as points. Split
    into parts of at most rows_per_file rows so that every file stays under
    GitHub's 100 MB limit. Read the whole set with
    pd.read_parquet(folder) or DuckDB read_parquet('folder/*_{created_at}_part*.parquet').
    The tile files are never changed by filter_image_ids().

Re-running skips every tile that already has a saved file, so an interrupted
run resumes where it stopped.

Usage
-----
    # tile count and request count only (no API call)
    python src/fetch_image_ids.py Maharashtra --plan-only

    # fetch only
    python src/fetch_image_ids.py Maharashtra

    # fetch, then filter by the corridor around the given road files. Use the
    # road files that produced the region's z14 cover; a mismatch is refused.
    python src/fetch_image_ids.py Maharashtra \
        data/raw/ADB_Innovation_Maharashtra.geojson data/raw/ADBMaha_combined_20250808.geojson
    python src/fetch_image_ids.py Thailand \
        data/raw/ADB_Innovation_Thailand.geojson data/raw/combined_traffic_stats_Thailand.geojson

    # first 5 unsaved tiles only (no filtering unless every tile is saved)
    python src/fetch_image_ids.py Thailand --max-tiles 5

The access token is read from --access-token, else MAPILLARY_ACCESS_TOKEN
(the .env file in the repository root is loaded when python-dotenv is installed).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Sequence

import geopandas as gpd
import mercantile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mapillary.config.api.vector_tiles import VectorTiles
from tqdm import tqdm
from vt2geojson.tools import vt_bytes_to_geojson

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_map_features import (  # noqa: E402
    DEFAULT_TILE_SAVE_DIR,
    TILE_COVER_DIR_NAME,
    VECTOR_TILE_DAILY_LIMIT,
    BackoffPolicy,
    CorridorTileTest,
    LayerSpec,
    SigmoidIntervalPacer,
    TileFetcher,
    TileKey,
    TileStore,
    TqdmLoggingHandler,
    _run,
    _utc_date,
    fetch_in_rounds,
    generate_buffer_polygon,
    resolve_access_token,
    tiles_touching,
    utc_timestamp,
)
from kepler_parquet import write_kepler_parquet  # noqa: E402

log = logging.getLogger("fetch_image_ids")

LAYER_NAME = "mly_image_ids"
ZOOM = 14
# The merged output goes to {FILTERED_DIR_NAME}/{region}/, a sibling of
# tile_save_dir, like filtered_map_features/.
FILTERED_DIR_NAME = "filtered_image_ids"
# A GeoParquet part takes about 32 bytes per image (zstd, WKB points included),
# so 2 million rows make a file of about 65 MB, under GitHub's 100 MB limit.
ROWS_PER_FILE = 2_000_000
# The images are kept within 10 m of the roads. The tile cover was built with
# a 30 m corridor, so every tile the 10 m corridor touches has been fetched.
FILTER_BUFFER_M = 10.0


# ---------------------------------------------------------------------------
# Vector tile layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageLayerSpec(LayerSpec):
    """The Mapillary "image" layer; its URL comes from mapillary-python-sdk."""

    def url(self, z: int, x: int, y: int) -> str:
        return VectorTiles.get_image_layer(x=x, y=y, z=z)


IMAGE_LAYER = ImageLayerSpec(LAYER_NAME, "image", min_zoom=ZOOM, max_zoom=ZOOM)


def validate_region(region: str) -> str:
    # The region names the input cover file and an output folder.
    if not region or region != Path(region).name or region in (".", ".."):
        raise ValueError(f"region must be a plain folder name, got {region!r}.")
    return region


def load_tile_cover(region: str, cover_dir: str | Path) -> list[tuple[int, int, int]]:
    """(z, x, y) of every tile in {cover_dir}/{region}_z14.parquet."""
    path = Path(cover_dir) / f"{validate_region(region)}_z{ZOOM}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; write it with extract_map_features.py --region {region} --plan-only."
        )
    df = pd.read_parquet(path, columns=["z", "x", "y"])
    zooms = set(df["z"].unique().tolist())
    if zooms - {ZOOM}:
        raise ValueError(f"{path} holds zoom(s) {sorted(zooms)}; only z{ZOOM} is served.")
    return [(int(z), int(x), int(y)) for z, x, y in df.itertuples(index=False)]


# ---------------------------------------------------------------------------
# Decode and store
# ---------------------------------------------------------------------------


# Columns of one saved tile: every property the image layer carried in
# September 2026, plus the position and the tile. A missing integer is -1, a
# missing float is NaN, a missing bool or string is null.
IMAGE_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("longitude", pa.float64()),
        ("latitude", pa.float64()),
        ("captured_at", pa.int64()),
        ("compass_angle", pa.float64()),
        ("creator_id", pa.int64()),
        ("organization_id", pa.int64()),
        ("sequence_id", pa.string()),
        ("is_pano", pa.bool_()),
        ("foot", pa.bool_()),
        ("quality_score", pa.float64()),
        ("tile_z", pa.int32()),
        ("tile_x", pa.int32()),
        ("tile_y", pa.int32()),
    ]
)
INT_PROPERTIES = ("captured_at", "creator_id", "organization_id")
FLOAT_PROPERTIES = ("compass_angle", "quality_score")
OTHER_PROPERTIES = ("id", "sequence_id", "is_pano", "foot", "tile_z", "tile_x", "tile_y")
# Anything else raises, so a new Mapillary property is noticed and never
# dropped without a trace.
IMAGE_PROPERTIES = set(INT_PROPERTIES + FLOAT_PROPERTIES + OTHER_PROPERTIES)


def features_to_image_table(features: Sequence[dict]) -> pa.Table:
    """Decoded GeoJSON point features of one tile -> an IMAGE_SCHEMA table."""
    props = [f["properties"] for f in features]
    for f, p in zip(features, props):
        if f["geometry"]["type"] != "Point":
            raise ValueError(f"Image tiles should hold points only, found {f['geometry']['type']}.")
        unknown = p.keys() - IMAGE_PROPERTIES
        if unknown:
            raise ValueError(f"Unexpected image properties {sorted(unknown)}; extend IMAGE_SCHEMA.")
    coords = [f["geometry"]["coordinates"] for f in features]

    def column(name, missing):
        return [p.get(name) if p.get(name) is not None else missing for p in props]

    columns = {
        "id": [p["id"] for p in props],
        "longitude": [c[0] for c in coords],
        "latitude": [c[1] for c in coords],
    }
    columns.update({name: column(name, -1) for name in INT_PROPERTIES})
    columns.update({name: column(name, float("nan")) for name in FLOAT_PROPERTIES})
    columns.update({name: column(name, None) for name in ("sequence_id", "is_pano", "foot")})
    columns.update({name: [p[name] for p in props] for name in ("tile_z", "tile_x", "tile_y")})
    return pa.table({name: columns[name] for name in IMAGE_SCHEMA.names}, schema=IMAGE_SCHEMA)


class ImageTileDecoder:
    """Image vector tile bytes -> GeoJSON point features inside the tile.

    Tiles carry a buffer of points just outside their bounds, which the
    neighbouring tile also holds. Only points inside the tile are kept, with
    the east and north edges excluded, so a point on a shared edge belongs to
    exactly one tile.
    """

    def decode(self, content: bytes, key: TileKey, source_layer: str) -> list[dict]:
        if not content:
            return []
        geojson = vt_bytes_to_geojson(content, key.x, key.y, key.z, layer=source_layer)
        west, south, east, north = mercantile.bounds(key.x, key.y, key.z)
        out: list[dict] = []
        seen: set[int] = set()
        for feat in geojson["features"]:
            lon, lat = feat["geometry"]["coordinates"][:2]
            if not (west <= lon < east and south <= lat < north):
                continue
            props = dict(feat.get("properties") or {})
            fid = props.get("id")
            if fid is None:
                raise ValueError(f"Image without id in tile {key.z}/{key.x}/{key.y}.")
            if fid in seen:
                continue
            seen.add(fid)
            props.update(tile_z=key.z, tile_x=key.x, tile_y=key.y)
            out.append({"type": "Feature", "geometry": feat["geometry"], "properties": props})
        return out


class ImageTileStore(TileStore):
    """One file per tile under {root}/{layer}/{region}/{z}/{x}/:

    {y}_{fetched_at}.parquet  the tile's images (IMAGE_SCHEMA, zstd)
    {y}_{fetched_at}.empty    zero-byte marker for a tile without images
    """

    def __init__(self, root: Path, region: str):
        super().__init__(root)
        self.region = validate_region(region)

    def _dir(self, key: TileKey) -> Path:
        return self.root / key.layer / self.region / str(key.z) / str(key.x)

    def save(self, key: TileKey, features: Sequence[dict], fetched_at: str) -> Path:
        folder = self._dir(key)
        folder.mkdir(parents=True, exist_ok=True)
        if not features:
            path = folder / f"{key.y}_{fetched_at}.empty"
            path.touch()
            return path
        path = folder / f"{key.y}_{fetched_at}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(features_to_image_table(features), tmp, compression="zstd")
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        path = Path(path)
        table = IMAGE_SCHEMA.empty_table() if path.suffix == ".empty" else pq.read_table(path)
        return table.to_pandas()


# ---------------------------------------------------------------------------
# Step 1: fetch
# ---------------------------------------------------------------------------


def fetch_image_ids(
    region: str,
    *,
    tile_save_dir: str | Path = DEFAULT_TILE_SAVE_DIR,
    cover_dir: str | Path | None = None,
    access_token: str | None = None,
    max_tiles: int | None = None,
    max_concurrency: int = 20,
    max_retry_rounds: int = 3,
    plan_only: bool = False,
) -> tuple[list[tuple[TileKey, Path]], list[tuple[TileKey, str]]]:
    """Fetch and save the images of every tile in the region's z14 cover.

    cover_dir defaults to tile_cover/ next to tile_save_dir. max_tiles caps
    how many unsaved tiles this call fetches (retries of those come on top).
    Returns the saved (key, path) pairs, including tiles saved by earlier
    runs, and the (key, reason) pairs that still failed.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}.")
    tile_save_dir = Path(tile_save_dir)
    if cover_dir is None:
        cover_dir = tile_save_dir.parent / TILE_COVER_DIR_NAME
    tiles = load_tile_cover(region, cover_dir)
    print(f"対象タイル数: {len(tiles)} (z{ZOOM}, {region})")
    print(f"必要リクエスト数: {len(tiles)}")
    if plan_only:
        return [], []

    store = ImageTileStore(tile_save_dir, region)
    saved: list[tuple[TileKey, Path]] = []
    pending: list[TileKey] = []
    for z, x, y in tiles:
        key = TileKey(LAYER_NAME, z, x, y)
        path = store.existing(key)
        if path is not None:
            saved.append((key, path))
        else:
            pending.append(key)
    log.info("%d tile(s) already saved, %d to fetch", len(saved), len(pending))
    if len(pending) > VECTOR_TILE_DAILY_LIMIT:
        log.warning(
            "%d requests exceed the vector tile limit of %d per day; "
            "expect 429 responses. Re-run later to resume from the saved tiles.",
            len(pending),
            VECTOR_TILE_DAILY_LIMIT,
        )
    if max_tiles is not None and len(pending) > max_tiles:
        log.info("max_tiles=%d: fetching the first %d of them", max_tiles, max_tiles)
        pending = pending[:max_tiles]

    failed: list[tuple[TileKey, str]] = []
    if pending:
        token = resolve_access_token(access_token)

        async def rounds():
            # The pacer holds an asyncio.Lock, so it is created inside the
            # loop that uses it.
            fetcher = TileFetcher(
                access_token=token,
                store=store,
                decoder=ImageTileDecoder(),
                pacer=SigmoidIntervalPacer(),
                backoff=BackoffPolicy(),
                max_concurrency=max_concurrency,
                layers={LAYER_NAME: IMAGE_LAYER},
            )
            return await fetch_in_rounds(fetcher, pending, max_retry_rounds, desc="image tiles")

        saved_now, failed = _run(rounds())
        saved.extend(saved_now)

    for key, reason in failed:
        log.error("FAILED z%d/%d/%d: %s", key.z, key.x, key.y, reason)
    if failed:
        log.error("%d tile(s) failed after %d retry round(s).", len(failed), max_retry_rounds)
    print(f"保存済みタイル: {len(saved)} / {len(tiles)} (失敗 {len(failed)})")
    print(f"出力フォルダ: {tile_save_dir / LAYER_NAME / region / str(ZOOM)}")
    return saved, failed


# ---------------------------------------------------------------------------
# Step 2: corridor filter and merge
# ---------------------------------------------------------------------------


# Columns of the merged output: those of IMAGE_SCHEMA, with the id as a string
# (64-bit integers arrive in JavaScript, e.g. kepler.gl, as BigInt) and
# captured_at as a "YYYY-MM-DD" UTC date, as in filtered_map_features.
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field(f.name, pa.string()) if f.name in ("id", "captured_at") else f
        for f in IMAGE_SCHEMA
    ]
)


def to_output_table(table: pa.Table) -> pa.Table:
    """An IMAGE_SCHEMA table -> an OUTPUT_SCHEMA table."""
    ids = table.column("id").to_numpy()
    captured = table.column("captured_at").to_numpy()
    table = table.set_column(
        IMAGE_SCHEMA.get_field_index("id"),
        OUTPUT_SCHEMA.field("id"),
        pa.array(ids.astype(str).astype(object), type=pa.string()),
    )
    table = table.set_column(
        IMAGE_SCHEMA.get_field_index("captured_at"),
        OUTPUT_SCHEMA.field("captured_at"),
        _utc_date(captured),
    )
    return table.cast(OUTPUT_SCHEMA)


class RollingParquetWriter:
    """Streams tables into {out_dir}/{prefix}_part{NNNN}.parquet files.

    A new part starts every rows_per_file rows. Each part is a GeoParquet
    file written by write_kepler_parquet(): the table's columns plus a WKB
    point `geometry` column (EPSG:4326) built from longitude and latitude, so
    QGIS and kepler.gl both open it. The rows of the current part are held in
    memory until the part is full. Parts are written under .tmp names and
    renamed only when the `with` block ends without an exception; after an
    exception every part of this writer is deleted, so a failed run leaves no
    partial output. No input rows means no file.
    """

    def __init__(self, out_dir: str | Path, prefix: str, schema: pa.Schema, rows_per_file: int):
        if rows_per_file < 1:
            raise ValueError(f"rows_per_file must be >= 1, got {rows_per_file}.")
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.schema = schema
        self.rows_per_file = rows_per_file
        self.paths: list[Path] = []
        self._tmp_paths: list[Path] = []
        self._part: list[pa.Table] = []
        self._rows_in_part = 0

    def __enter__(self) -> "RollingParquetWriter":
        return self

    def write(self, table: pa.Table) -> None:
        if not table.schema.equals(self.schema):
            raise ValueError(f"Table schema differs from the writer's:\n{table.schema}")
        while table.num_rows:
            chunk = table.slice(0, self.rows_per_file - self._rows_in_part)
            self._part.append(chunk)
            self._rows_in_part += chunk.num_rows
            table = table.slice(chunk.num_rows)
            if self._rows_in_part == self.rows_per_file:
                self._write_part()

    def _write_part(self) -> None:
        if not self._rows_in_part:
            return
        df = pa.concat_tables(self._part).to_pandas()
        self._part, self._rows_in_part = [], 0
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.out_dir / f"{self.prefix}_part{len(self._tmp_paths) + 1:04d}.parquet.tmp"
        self._tmp_paths.append(tmp)
        write_kepler_parquet(gdf, tmp, compression="zstd")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        ok = exc_type is None
        try:
            if ok:
                self._write_part()
        except BaseException:
            ok = False
            raise
        finally:
            self._part, self._rows_in_part = [], 0
            if ok:
                for tmp in self._tmp_paths:
                    final = tmp.with_suffix("")
                    os.replace(tmp, final)
                    self.paths.append(final)
            else:
                for tmp in self._tmp_paths:
                    tmp.unlink(missing_ok=True)
            self._tmp_paths = []


@dataclass
class FilterResult:
    paths: list[Path] = field(default_factory=list)
    tiles_outside: int = 0
    tiles_empty: int = 0
    tiles_inside: int = 0
    tiles_edge: int = 0
    rows_read: int = 0
    rows_kept: int = 0


class ImageIdFilter:
    """Keeps the saved images that lie inside the corridor, tile by tile.

    A tile outside the corridor is never read, a tile inside it is kept
    whole, and only a tile crossing the corridor edge has its points tested.
    Kept rows go straight to the writer, which holds at most one output part
    (rows_per_file rows) in memory.
    """

    def __init__(self, corridor_test: CorridorTileTest, store: ImageTileStore):
        self.corridor_test = corridor_test
        self.store = store

    def saved_files(self, tiles: Sequence[tuple[int, int, int]]) -> list[tuple[TileKey, Path]]:
        """Newest saved file of every tile; raises if any tile has none."""
        found, missing = [], []
        for z, x, y in tiles:
            key = TileKey(LAYER_NAME, z, x, y)
            path = self.store.existing(key)
            if path is None:
                missing.append(key)
            else:
                found.append((key, path))
        if missing:
            sample = ", ".join(f"{k.z}/{k.x}/{k.y}" for k in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} tile(s) have no saved file (e.g. {sample}); "
                f"run fetch_image_ids() for region {self.store.region!r} first."
            )
        return found

    def run(self, tiles: Sequence[tuple[int, int, int]], writer: RollingParquetWriter) -> FilterResult:
        result = FilterResult()
        kept_ids: list[np.ndarray] = []
        for key, path in tqdm(self.saved_files(tiles), desc="corridor filter", unit="tile"):
            place = self.corridor_test.classify(key.z, key.x, key.y)
            if place == CorridorTileTest.OUTSIDE:
                result.tiles_outside += 1
                continue
            if path.suffix == ".empty":
                result.tiles_empty += 1
                continue
            table = pq.read_table(path)
            if not table.schema.equals(IMAGE_SCHEMA):
                raise ValueError(f"{path} does not have IMAGE_SCHEMA:\n{table.schema}")
            result.rows_read += table.num_rows
            if place == CorridorTileTest.EDGE:
                result.tiles_edge += 1
                inside = self.corridor_test.points_inside(
                    table.column("longitude").to_numpy(), table.column("latitude").to_numpy()
                )
                table = table.filter(pa.array(inside))
            else:
                result.tiles_inside += 1
            result.rows_kept += table.num_rows
            kept_ids.append(table.column("id").to_numpy())
            writer.write(to_output_table(table))
        ids = np.concatenate(kept_ids) if kept_ids else np.empty(0, dtype=np.int64)
        # Each tile keeps only points inside its own half-open bounds, so an
        # image appears in one tile only. Checked here, not assumed.
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate image ids across tiles; the output was not written.")
        return result


def filter_image_ids(
    region: str,
    geojson_paths: Sequence[str | Path],
    *,
    buffer_m: float = FILTER_BUFFER_M,
    tile_save_dir: str | Path = DEFAULT_TILE_SAVE_DIR,
    cover_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    rows_per_file: int = ROWS_PER_FILE,
) -> FilterResult:
    """Keep the saved images inside the corridor and write them in parts.

    The corridor is built from geojson_paths with generate_buffer_polygon()
    (buffer_m metres, 10 by default). Every z14 tile it touches must be in the
    region's cover, which holds the tiles that were fetched; otherwise the
    road files or the buffer do not match the cover and the call raises.
    Only the tiles the corridor touches are read.
    output_dir defaults to filtered_image_ids/{region}/ next to tile_save_dir.
    """
    if not geojson_paths:
        raise ValueError("geojson_paths is empty; give the road file(s) of the region.")
    tile_save_dir = Path(tile_save_dir)
    if cover_dir is None:
        cover_dir = tile_save_dir.parent / TILE_COVER_DIR_NAME
    tiles = load_tile_cover(region, cover_dir)
    corridor = generate_buffer_polygon([Path(p) for p in dict.fromkeys(geojson_paths)], buffer_m)

    corridor_tiles = set(tiles_touching(corridor, ZOOM))
    outside_cover = corridor_tiles - set(tiles)
    if outside_cover:
        raise ValueError(
            f"The {buffer_m:g} m corridor of the given road file(s) touches "
            f"{len(outside_cover)} z{ZOOM} tile(s) outside the {region} cover, so their "
            f"images were never fetched. Give the road files that produced the cover, "
            f"with a buffer no wider than the cover's."
        )
    tiles = [t for t in tiles if t in corridor_tiles]
    log.info("%d of the cover's tiles touch the %g m corridor", len(tiles), buffer_m)

    if output_dir is None:
        output_dir = tile_save_dir.parent / FILTERED_DIR_NAME / region
    store = ImageTileStore(tile_save_dir, region)
    prefix = f"{region}_image_ids_{utc_timestamp()}"
    with RollingParquetWriter(output_dir, prefix, OUTPUT_SCHEMA, rows_per_file) as writer:
        result = ImageIdFilter(CorridorTileTest(corridor), store).run(tiles, writer)
    result.paths = writer.paths

    log.info(
        "tiles: %d outside, %d empty, %d inside, %d on the edge; rows: %d read, %d kept",
        result.tiles_outside,
        result.tiles_empty,
        result.tiles_inside,
        result.tiles_edge,
        result.rows_read,
        result.rows_kept,
    )
    print(f"コリドー内の画像: {result.rows_kept}件 (タイル内 {result.rows_read}件のうち)")
    print(f"出力ファイル数: {len(result.paths)} ({output_dir})")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("region", help="region name, e.g. Maharashtra; reads tile_cover/REGION_z14.parquet")
    p.add_argument(
        "geojson_paths",
        type=Path,
        nargs="*",
        help="road network GeoJSON file(s) that produced the cover; when given, "
        "the saved images are filtered by their corridor after the fetch",
    )
    p.add_argument(
        "--buffer-m",
        type=float,
        default=FILTER_BUFFER_M,
        help=f"corridor half-width for the filter, metres (default: {FILTER_BUFFER_M:g})",
    )
    p.add_argument(
        "--rows-per-file",
        type=int,
        default=ROWS_PER_FILE,
        help=f"rows per merged output file (default: {ROWS_PER_FILE})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"folder for the merged files (default: {FILTERED_DIR_NAME}/REGION)",
    )
    p.add_argument(
        "--tile-save-dir",
        type=Path,
        default=DEFAULT_TILE_SAVE_DIR,
        help=f"folder for the saved tiles (default: {DEFAULT_TILE_SAVE_DIR})",
    )
    p.add_argument(
        "--cover-dir",
        type=Path,
        default=None,
        help=f"folder holding REGION_z14.parquet (default: {TILE_COVER_DIR_NAME}/ next to --tile-save-dir)",
    )
    p.add_argument("--access-token", default=None, help="default: $MAPILLARY_ACCESS_TOKEN")
    p.add_argument("--max-concurrency", type=int, default=20)
    p.add_argument("--max-retry-rounds", type=int, default=3)
    p.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="fetch at most this many unsaved tiles in this run",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="print tile and request counts, call no API",
    )
    args = p.parse_args(argv)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    _, failed = fetch_image_ids(
        args.region,
        tile_save_dir=args.tile_save_dir,
        cover_dir=args.cover_dir,
        access_token=args.access_token,
        max_tiles=args.max_tiles,
        max_concurrency=args.max_concurrency,
        max_retry_rounds=args.max_retry_rounds,
        plan_only=args.plan_only,
    )
    if failed:
        return 1
    if args.plan_only or not args.geojson_paths:
        return 0
    if args.max_tiles is not None:
        # A capped run may leave tiles unsaved; filter_image_ids() then
        # refuses, naming the missing tiles.
        log.info("--max-tiles was given; filtering needs every tile saved.")
    filter_image_ids(
        args.region,
        args.geojson_paths,
        buffer_m=args.buffer_m,
        tile_save_dir=args.tile_save_dir,
        cover_dir=args.cover_dir,
        output_dir=args.output_dir,
        rows_per_file=args.rows_per_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
