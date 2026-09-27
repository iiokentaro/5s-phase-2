"""Extract Mapillary map features along a road network via Mapillary vector tiles.

Pipeline (call in this order, or run the CLI):

1. generate_buffer_polygon()      road lines -> corridor polygon (buffer_m metres)
2. compute_tiles()                corridor -> vector tiles that actually touch it
3. request_and_save_vecor_tiles() fetch each (layer, tile), decode to lon/lat,
                                  save one Parquet (or .empty marker) per tile
4. decode()                       read the saved tiles one at a time into one
                                  compact table; keep one row per id
5. filter()                       per tile, keep features whose object_value is
                                  requested and whose point lies inside the
                                  corridor; merge same-class features within
                                  1 m; write one Parquet per class

Outputs
-------
data/mapillary/vector_tiles/{layer}/{z}/{x}/{y}_{fetched_at}.parquet
    Decoded features of one tile, one row per feature (TILE_SCHEMA).
data/mapillary/vector_tiles/{layer}/{z}/{x}/{y}_{fetched_at}.empty
    Zero-byte marker for a tile with no features. An empty Parquet file
    would still take about 5 KB for its schema, and most tiles are empty.
data/mapillary/tile_cover/{region}_z{zoom}.parquet
    The tiles from compute_tiles() as square polygons (GeoParquet, EPSG:4326),
    for review in QGIS. Each run overwrites the region's file.
data/mapillary/filtered_map_features/{region}/{region}_{object_value}_{created_at}.parquet
    The filtered features, one file per class, one row per point, with
    latitude/longitude columns (loads in kepler.gl as a point layer). The
    region name is a required argument. The tile files are never changed by
    decode() or filter().

Re-running skips every (layer, tile) that already has a saved file, so an
interrupted run resumes where it stopped.

Usage
-----
    # tile count and request count only (no API call); points layer by default
    # (add --layers mly_map_feature_point mly_map_feature_traffic_sign for both)
    python src/extract_map_features.py data/raw/ADB_Innovation_Maharashtra.geojson \
        --region Maharashtra --plan-only

    # several road files: their lines are merged before buffering
    python src/extract_map_features.py data/raw/ADB_Innovation_Maharashtra.geojson \
        data/raw/ADBMaha_combined_20250808.geojson --region Maharashtra --plan-only

    # full run; one Parquet per class under filtered_map_features/Maharashtra/
    python src/extract_map_features.py data/raw/ADB_Innovation_Maharashtra.geojson \
        data/raw/ADBMaha_combined_20250808.geojson --region Maharashtra

    # zebra crossings only
    python src/extract_map_features.py data/raw/ADB_Innovation_Maharashtra.geojson \
        --region Maharashtra --object-values marking--discrete--crosswalk-zebra

The access token is read from --access-token, else MAPILLARY_ACCESS_TOKEN
(the .env file in the repository root is loaded when python-dotenv is installed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

import aiohttp
import geopandas as gpd
import mapbox_vector_tile
import mercantile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

log = logging.getLogger("extract_map_features")

DEFAULT_TILE_SAVE_DIR = Path("data/mapillary/vector_tiles")
# The filtered Parquet files go to {FILTERED_DIR_NAME}/{region}/, a sibling of
# tile_save_dir, so that data/mapillary/ itself only gains sub-folders.
FILTERED_DIR_NAME = "filtered_map_features"
TILE_COVER_DIR_NAME = "tile_cover"

# Mapillary API documentation: vector tiles are limited to 50,000 requests per day.
VECTOR_TILE_DAILY_LIMIT = 50_000

TOKEN_ENV_VAR = "MAPILLARY_ACCESS_TOKEN"

# Timestamps in file names: UTC, sortable as plain strings.
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


# ---------------------------------------------------------------------------
# Vector tile layers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerSpec:
    """One Mapillary vector tile layer and the zoom levels it is served at.

    source_layer is the layer name inside the tile. Tiles also carry other
    layers: a tile without map features holds a "water" layer with one
    polygon covering the whole tile, which must not be read as a feature.
    """

    name: str
    source_layer: str
    min_zoom: int
    max_zoom: int

    def url(self, z: int, x: int, y: int) -> str:
        # Same URL shape as mapillary-python-sdk
        # src/mapillary/config/api/vector_tiles.py (get_map_feature_point /
        # get_map_feature_traffic_sign). The token goes in the query string.
        return f"https://tiles.mapillary.com/maps/vtp/{self.name}/2/{z}/{x}/{y}/"


# Zoom ranges come from mapillary-python-sdk:
# - config/api/vector_tiles.py documents both map feature layers as "zoom: 14".
# - models/api/vector_tiles.py __zoom_range_check raises for any zoom other
#   than 14 on "map_feature" and "traffic_sign".
# So z14 is the only zoom that returns these layers at all. The source layer
# names ("point", "traffic_sign") are from the same docstrings and match the
# tiles the API returned in September 2026. To add a layer,
# add an entry here; the rest of the pipeline reads this registry.
LAYER_REGISTRY: dict[str, LayerSpec] = {
    spec.name: spec
    for spec in (
        LayerSpec("mly_map_feature_point", "point", min_zoom=14, max_zoom=14),
        LayerSpec("mly_map_feature_traffic_sign", "traffic_sign", min_zoom=14, max_zoom=14),
    )
}
# Default: points only. Zebra crossings and the other point classes live in
# this layer; adding the traffic sign layer doubles the request count
# (one request per layer per tile), which would push Maharashtra and
# Thailand over the 50,000 requests/day limit.
DEFAULT_LAYERS: tuple[str, ...] = ("mly_map_feature_point",)


def resolve_layers(names: Iterable[str]) -> list[LayerSpec]:
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("layers is empty; give at least one vector tile layer.")
    unknown = [n for n in names if n not in LAYER_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown layer(s) {unknown}. Known layers: {sorted(LAYER_REGISTRY)}"
        )
    return [LAYER_REGISTRY[n] for n in names]


def select_zoom(layers: Sequence[LayerSpec]) -> int:
    """Pick the zoom that fetches every feature with the fewest requests.

    Each step down in zoom divides the tile count by about four, so the
    lowest zoom that every selected layer serves needs the fewest requests.
    min_zoom in LAYER_REGISTRY is the lowest zoom at which a layer carries
    its complete feature set, so no feature is lost at that zoom.
    """
    lo = max(spec.min_zoom for spec in layers)
    hi = min(spec.max_zoom for spec in layers)
    if lo > hi:
        raise ValueError(
            "The selected layers share no zoom level: "
            + ", ".join(f"{s.name} z{s.min_zoom}-{s.max_zoom}" for s in layers)
        )
    return lo


class TileKey(NamedTuple):
    layer: str
    z: int
    x: int
    y: int


# ---------------------------------------------------------------------------
# Step 1: corridor polygon
# ---------------------------------------------------------------------------


LINE_TYPES = {"LineString", "MultiLineString"}


def load_line_geometries(geojson_path: Path) -> list[BaseGeometry]:
    with open(geojson_path, encoding="utf-8-sig") as f:
        data = json.load(f)
    features = data.get("features") or []
    if not features:
        raise ValueError(f"{geojson_path} has no features.")
    geoms = []
    for i, feat in enumerate(features):
        geom = feat.get("geometry")
        if geom is None:
            raise ValueError(f"{geojson_path}: feature #{i} has no geometry.")
        if geom.get("type") not in LINE_TYPES:
            raise ValueError(
                f"{geojson_path}: feature #{i} is a {geom.get('type')}; "
                f"only LineString and MultiLineString are accepted."
            )
        geoms.append(shape(geom))
    # GeoJSON (RFC 7946) coordinates are WGS84 lon/lat; the corridor code
    # relies on that, so reject files that are clearly in another system.
    xmin, ymin, xmax, ymax = shapely.total_bounds(np.asarray(geoms, dtype=object))
    if xmin < -180 or xmax > 180 or ymin < -90 or ymax > 90:
        raise ValueError(
            f"{geojson_path}: coordinates ({xmin}, {ymin}, {xmax}, {ymax}) are "
            f"not WGS84 longitude/latitude."
        )
    return geoms


def utm_epsg(lon: float, lat: float) -> int:
    zone = min(max(int((lon + 180) // 6) + 1, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _reproject(geoms: np.ndarray, transformer: Transformer) -> np.ndarray:
    def fn(coords: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    out = shapely.transform(geoms, fn)
    if not np.all(np.isfinite(shapely.get_coordinates(out))):
        raise ValueError(f"Reprojection {transformer.name} produced non-finite coordinates.")
    return out


class CorridorBuilder:
    """Merges road lines into one network, buffers it in metres, merges again.

    The lines (from one or more road files) are first merged with union_all,
    so a road present in several files is buffered once.

    Projection choice: each line of the merged network is buffered in the
    UTM zone of its own centroid. A single UTM zone for a whole state
    stretches distances away from its central meridian (Maharashtra spans
    zones 43 and 44), while a line's own zone keeps the scale error of a
    30 m buffer well under 0.1 m. The buffered pieces are projected back to
    WGS84 and merged there.
    """

    def __init__(self, buffer_m: float):
        if buffer_m <= 0:
            raise ValueError(f"buffer_m must be positive, got {buffer_m}.")
        self.buffer_m = buffer_m

    def build(self, lines: Sequence[BaseGeometry]) -> BaseGeometry:
        network = shapely.union_all(np.asarray(lines, dtype=object))
        geoms = shapely.get_parts(network)
        log.info("%d input line(s) merged into %d line(s)", len(lines), len(geoms))
        centroids = shapely.centroid(geoms)
        epsgs = np.array(
            [utm_epsg(x, y) for x, y in zip(shapely.get_x(centroids), shapely.get_y(centroids))]
        )
        pieces = []
        for epsg in np.unique(epsgs):
            group = geoms[epsgs == epsg]
            try:
                to_m = Transformer.from_crs(4326, int(epsg), always_xy=True)
                to_deg = Transformer.from_crs(int(epsg), 4326, always_xy=True)
                buffered = shapely.buffer(_reproject(group, to_m), self.buffer_m)
                pieces.append(_reproject(buffered, to_deg))
            except Exception as exc:
                raise ValueError(
                    f"Coordinate transformation to/from EPSG:{epsg} failed: {exc}"
                ) from exc
            log.info("EPSG:%d  %d line(s) buffered", epsg, len(group))
        return shapely.union_all(np.concatenate(pieces))


def generate_buffer_polygon(
    geojson_paths: Sequence[str | Path], buffer_m: float
) -> BaseGeometry:
    """Road GeoJSON file(s) -> one corridor polygon, buffer_m metres around the roads."""
    lines: list[BaseGeometry] = []
    for path in geojson_paths:
        part = load_line_geometries(Path(path))
        log.info("%d road line(s) read from %s", len(part), path)
        lines.extend(part)
    return CorridorBuilder(buffer_m).build(lines)


# ---------------------------------------------------------------------------
# Step 2: tile cover
# ---------------------------------------------------------------------------


def tiles_touching(polygon: BaseGeometry, zoom: int) -> list[tuple[int, int, int]]:
    """Tiles at `zoom` whose extent intersects the polygon's actual shape."""
    west, south, east, north = polygon.bounds
    candidates = list(mercantile.tiles(west, south, east, north, zooms=zoom))
    if not candidates:
        return []
    bounds = np.array([tuple(mercantile.bounds(t)) for t in candidates])
    boxes = shapely.box(bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3])
    shapely.prepare(polygon)
    hit = shapely.intersects(polygon, boxes)
    return [(t.z, t.x, t.y) for t, keep in zip(candidates, hit) if keep]


def write_tile_cover_parquet(tiles: Sequence[tuple[int, int, int]], path: Path) -> None:
    """Tiles as square polygons in GeoParquet (WKB geometry, EPSG:4326).

    QGIS opens the file as a polygon layer. Columns: z, x, y, tile ("z/x/y"),
    west, south, east, north, geometry.
    """
    zxy = np.array(tiles, dtype=np.int32).reshape(-1, 3)
    bounds = np.array(
        [tuple(mercantile.bounds(int(x), int(y), int(z))) for z, x, y in zxy],
        dtype=np.float64,
    ).reshape(-1, 4)
    gdf = gpd.GeoDataFrame(
        {
            "z": zxy[:, 0],
            "x": zxy[:, 1],
            "y": zxy[:, 2],
            # object dtype: pandas 3 would otherwise store strings as Arrow
            # large_string, which kepler.gl rejects ("arrow type not supported").
            "tile": pd.Series([f"{z}/{x}/{y}" for z, x, y in zxy], dtype=object),
            "west": bounds[:, 0],
            "south": bounds[:, 1],
            "east": bounds[:, 2],
            "north": bounds[:, 3],
        },
        geometry=shapely.box(bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]),
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    gdf.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Step 3: fetch, decode, store
# ---------------------------------------------------------------------------


# Columns of one saved tile. Missing ids and timestamps are -1. The tiles'
# "value" property is stored once, as object_value.
TILE_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("object_value", pa.string()),
        ("longitude", pa.float64()),
        ("latitude", pa.float64()),
        ("first_seen_at", pa.int64()),
        ("last_seen_at", pa.int64()),
        ("layer", pa.string()),
        ("tile_z", pa.int32()),
        ("tile_x", pa.int32()),
        ("tile_y", pa.int32()),
    ]
)
# Every property a decoded feature may carry. Anything else raises, so a new
# Mapillary property is noticed and never dropped without a trace.
TILE_PROPERTIES = {
    "id",
    "value",
    "object_value",
    "first_seen_at",
    "last_seen_at",
    "layer",
    "tile_z",
    "tile_x",
    "tile_y",
}


def features_to_tile_table(features: Sequence[dict]) -> pa.Table:
    """Decoded GeoJSON point features of one tile -> a TILE_SCHEMA table."""
    props = [f["properties"] for f in features]
    for f, p in zip(features, props):
        if f["geometry"]["type"] != "Point":
            raise ValueError(f"Map feature tiles should hold points only, found {f['geometry']['type']}.")
        unknown = p.keys() - TILE_PROPERTIES
        if unknown:
            raise ValueError(f"Unexpected feature properties {sorted(unknown)}; extend TILE_SCHEMA.")
        if "value" in p and p.get("object_value", p["value"]) != p["value"]:
            raise ValueError(f"value {p['value']!r} differs from object_value {p['object_value']!r}.")
    coords = [f["geometry"]["coordinates"] for f in features]

    def ints(name):
        return [p.get(name) if p.get(name) is not None else -1 for p in props]

    return pa.table(
        {
            "id": ints("id"),
            "object_value": [p.get("object_value", p.get("value")) for p in props],
            "longitude": [c[0] for c in coords],
            "latitude": [c[1] for c in coords],
            "first_seen_at": ints("first_seen_at"),
            "last_seen_at": ints("last_seen_at"),
            "layer": [p.get("layer") for p in props],
            "tile_z": [p["tile_z"] for p in props],
            "tile_x": [p["tile_x"] for p in props],
            "tile_y": [p["tile_y"] for p in props],
        },
        schema=TILE_SCHEMA,
    )


class TileStore:
    """One file per (layer, tile) under {root}/{layer}/{z}/{x}/:

    {y}_{fetched_at}.parquet  the tile's features (TILE_SCHEMA, zstd)
    {y}_{fetched_at}.empty    zero-byte marker for a tile without features
    """

    SUFFIXES = (".parquet", ".empty")

    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, key: TileKey) -> Path:
        return self.root / key.layer / str(key.z) / str(key.x)

    def existing(self, key: TileKey) -> Path | None:
        """Newest saved file for the tile, or None."""
        folder = self._dir(key)
        if not folder.is_dir():
            return None
        found = sorted(
            (f for f in folder.glob(f"{key.y}_*") if f.suffix in self.SUFFIXES),
            key=lambda f: f.stem,
        )
        return found[-1] if found else None

    def save(self, key: TileKey, features: Sequence[dict], fetched_at: str) -> Path:
        folder = self._dir(key)
        folder.mkdir(parents=True, exist_ok=True)
        if not features:
            path = folder / f"{key.y}_{fetched_at}.empty"
            path.touch()
            return path
        path = folder / f"{key.y}_{fetched_at}.parquet"
        # Write to a temporary name first so an interrupted run never leaves a
        # half-written file that existing() would treat as done.
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(features_to_tile_table(features), tmp, compression="zstd")
        os.replace(tmp, path)
        return path

    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        path = Path(path)
        table = TILE_SCHEMA.empty_table() if path.suffix == ".empty" else pq.read_table(path)
        return table.to_pandas()


class VectorTileDecoder:
    """Mapbox vector tile bytes -> GeoJSON point features in WGS84."""

    def decode(self, content: bytes, key: TileKey, source_layer: str) -> list[dict]:
        """Features of the tile's `source_layer`; other layers are ignored."""
        if not content:
            return []
        # y_coord_down=True keeps tile y growing downwards (row 0 at the top),
        # which is what the projection below assumes. Same option as
        # vt2geojson.tools.vt_bytes_to_geojson used by mapillary-python-sdk.
        layers = mapbox_vector_tile.decode(content, default_options={"y_coord_down": True})
        out = []
        layer_obj = layers.get(source_layer)
        if layer_obj is not None:
            extent = layer_obj.get("extent", 4096)
            for feat in layer_obj["features"]:
                props = dict(feat.get("properties") or {})
                # The map feature id is a property. The protobuf feature id is
                # only an index within the tile, so it cannot identify a
                # feature across tiles and is not used.
                fid = props.get("id")
                geometry = {
                    "type": feat["geometry"]["type"],
                    "coordinates": self._project(
                        feat["geometry"]["coordinates"], key, extent
                    ),
                }
                props.update(
                    id=fid,
                    # Mapillary names the class "value" inside the tiles.
                    object_value=props.get("value"),
                    layer=key.layer,
                    tile_z=key.z,
                    tile_x=key.x,
                    tile_y=key.y,
                )
                out.append(
                    {"type": "Feature", "id": fid, "geometry": geometry, "properties": props}
                )
        return out

    @classmethod
    def _project(cls, coords, key: TileKey, extent: int):
        """Tile-local pixel coordinates -> [lon, lat].

        Same arithmetic as vt2geojson.features.Feature.toGeoJSON: the tile is
        placed on a world grid of extent * 2**z pixels and the pixel position
        is inverted through the Web Mercator formula.
        """
        if coords and isinstance(coords[0], (int, float)):
            size = extent * 2**key.z
            px = coords[0] + extent * key.x
            py = coords[1] + extent * key.y
            lon = px * 360.0 / size - 180.0
            y2 = 180.0 - py * 360.0 / size
            lat = 360.0 / math.pi * math.atan(math.exp(y2 * math.pi / 180.0)) - 90.0
            return [lon, lat]
        return [cls._project(c, key, extent) for c in coords]


def _sigmoid(v: float) -> float:
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


class SigmoidIntervalPacer:
    """Spaces request starts, learning the spacing from 429 responses.

    Model: P(success | interval) = sigmoid(w0 + w1 * log(interval)), fitted
    online by one gradient step per observed request. w1 is kept positive
    (a longer gap never lowers the success rate). The next interval is the
    shortest one whose predicted success rate reaches `target`, clamped to
    [min_interval, max_interval]. Successes push the interval down, 429s
    push it up.
    """

    def __init__(
        self,
        target: float = 0.95,
        initial_interval: float = 0.05,
        min_interval: float = 0.005,
        max_interval: float = 5.0,
        learning_rate: float = 0.05,
        initial_slope: float = 1.0,
    ):
        self.target = target
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.lr = learning_rate
        self.w1 = initial_slope
        logit_target = math.log(target / (1 - target))
        self.w0 = logit_target - self.w1 * math.log(initial_interval)
        self._interval = initial_interval
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    @property
    def interval(self) -> float:
        return self._interval

    async def wait(self) -> float:
        """Sleep until this request may start; return the interval it used."""
        async with self._lock:
            interval = self._interval
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + interval
        if start > now:
            await asyncio.sleep(start - now)
        return interval

    def record(self, interval: float, success: bool) -> None:
        x = math.log(max(interval, self.min_interval))
        error = (1.0 if success else 0.0) - _sigmoid(self.w0 + self.w1 * x)
        self.w0 += self.lr * error
        self.w1 = max(0.1, self.w1 + self.lr * error * x)
        logit_target = math.log(self.target / (1 - self.target))
        best = math.exp((logit_target - self.w0) / self.w1)
        self._interval = min(max(best, self.min_interval), self.max_interval)


@dataclass(frozen=True)
class BackoffPolicy:
    """Exponential backoff with jitter for 429, 5xx and network errors."""

    base_s: float = 1.0
    factor: float = 2.0
    max_s: float = 60.0
    max_attempts: int = 4

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        d = min(self.base_s * self.factor**attempt, self.max_s)
        if retry_after is not None:
            d = max(d, retry_after)
        return d * random.uniform(1.0, 1.25)


class AuthenticationError(RuntimeError):
    """The API refused the token; retrying cannot help."""


class FetchError(RuntimeError):
    """One request key failed for good; args[0] is (key, reason)."""


class TileFetchError(FetchError):
    pass


class RequestFailed(RuntimeError):
    """A GET that retrying cannot fix, or that ran out of attempts.

    status and body are those of the last response (None and b"" after a
    network error), so a caller can tell one kind of failure from another.
    """

    def __init__(self, reason: str, status: int | None = None, body: bytes = b""):
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.body = body


class PacedGetter:
    """GET with the request gap learned by the pacer and retries by backoff.

    Success is HTTP 200/204. 401/403 raise AuthenticationError. 429 is
    recorded in the pacer and retried; 5xx and network errors are retried;
    any other status fails at once. is_final(status, body) can mark a
    response final that would otherwise be retried (e.g. a 500 that says the
    request asks for too much data).
    """

    def __init__(
        self,
        access_token: str,
        pacer: SigmoidIntervalPacer,
        backoff: BackoffPolicy,
        is_final: Callable[[int, bytes], bool] | None = None,
    ):
        self.token = access_token
        self.pacer = pacer
        self.backoff = backoff
        self.is_final = is_final or (lambda status, body: False)
        # Seconds from sending a request to reading its whole body, per attempt.
        self.latencies: list[float] = []

    async def get(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        url: str,
        params: Mapping[str, str | int] | None = None,
    ) -> bytes:
        query = {**(params or {}), "access_token": self.token}
        reason = "no attempt made"
        for attempt in range(self.backoff.max_attempts):
            retry_after = None
            async with sem:
                interval = await self.pacer.wait()
                sent = time.monotonic()
                try:
                    async with session.get(url, params=query) as resp:
                        status = resp.status
                        body = await resp.read()
                        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                    status, body = None, b""
                    reason = f"{type(exc).__name__}: {exc}"
                self.latencies.append(time.monotonic() - sent)

            if status in (200, 204):
                self.pacer.record(interval, success=True)
                return body
            if status in (401, 403):
                raise AuthenticationError(f"HTTP {status}: the access token was refused.")
            if status == 429:
                self.pacer.record(interval, success=False)
                reason = "HTTP 429"
            elif status is not None and self.is_final(status, body):
                raise RequestFailed(
                    f"HTTP {status}: {body[:200].decode(errors='replace')}", status, body
                )
            elif status is not None and status >= 500:
                reason = f"HTTP {status}"
            elif status is not None:
                raise RequestFailed(f"HTTP {status}", status, body)
            await asyncio.sleep(self.backoff.delay(attempt, retry_after))
        raise RequestFailed(reason)


class PacedBatchFetcher:
    """Runs _fetch_one for many keys concurrently through one PacedGetter.

    Subclasses implement _fetch_one(session, sem, key) -> (key, result) and
    raise FetchError((key, reason)) for a key that failed; AuthenticationError
    stops the batch. fetch_all returns the (key, result) pairs and the
    (key, reason) failures, with a progress bar showing the learned gap.
    """

    def __init__(self, getter: PacedGetter, max_concurrency: int, timeout_s: float = 30.0):
        self.getter = getter
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s

    @property
    def pacer(self) -> SigmoidIntervalPacer:
        return self.getter.pacer

    @property
    def latencies(self) -> list[float]:
        return self.getter.latencies

    async def fetch_all(self, keys: Sequence, desc: str) -> tuple[list[tuple], list[tuple]]:
        done: list[tuple] = []
        failed: list[tuple] = []
        if not keys:
            return done, failed
        sem = asyncio.Semaphore(self.max_concurrency)
        connector = aiohttp.TCPConnector(limit=self.max_concurrency)
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = {
                asyncio.ensure_future(self._fetch_one(session, sem, key)): key for key in keys
            }
            with tqdm(total=len(keys), desc=desc, unit="req", dynamic_ncols=True) as bar:
                try:
                    for fut in asyncio.as_completed(list(tasks)):
                        try:
                            done.append(await fut)
                        except AuthenticationError:
                            raise
                        except FetchError as exc:
                            failed.append(exc.args[0])
                        bar.update(1)
                        bar.set_postfix(
                            ok=len(done),
                            failed=len(failed),
                            gap=f"{self.pacer.interval * 1000:.0f}ms",
                        )
                finally:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        return done, failed

    async def _fetch_one(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, key):
        raise NotImplementedError


class TileFetcher(PacedBatchFetcher):
    """Fetches (layer, tile) pairs concurrently and saves each decoded tile.

    `layers` maps each TileKey.layer to the LayerSpec that gives its URL and
    source layer. Another vector tile layer (e.g. Mapillary images) is served
    by passing its own layers, store and decoder.
    """

    def __init__(
        self,
        access_token: str,
        store: TileStore,
        decoder: VectorTileDecoder,
        pacer: SigmoidIntervalPacer,
        backoff: BackoffPolicy,
        max_concurrency: int,
        timeout_s: float = 30.0,
        layers: Mapping[str, LayerSpec] = LAYER_REGISTRY,
    ):
        super().__init__(PacedGetter(access_token, pacer, backoff), max_concurrency, timeout_s)
        self.store = store
        self.decoder = decoder
        self.layers = layers

    async def _fetch_one(
        self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, key: TileKey
    ) -> tuple[TileKey, Path]:
        url = self.layers[key.layer].url(key.z, key.x, key.y)
        try:
            body = await self.getter.get(session, sem, url)
        except RequestFailed as exc:
            raise TileFetchError((key, exc.reason)) from exc
        try:
            features = self.decoder.decode(body, key, self.layers[key.layer].source_layer)
            path = self.store.save(key, features, utc_timestamp())
        except Exception as exc:
            raise TileFetchError((key, f"decode/save failed: {exc}")) from exc
        return key, path


async def fetch_in_rounds(
    fetcher: PacedBatchFetcher,
    pending: Sequence,
    max_retry_rounds: int,
    desc: str = "vector tiles",
) -> tuple[list[tuple], list[tuple]]:
    """Fetch `pending`, then retry the failed keys up to max_retry_rounds times.

    Every round uses the same fetcher, so the request gap learned by its
    pacer carries over from one round to the next.
    """
    started = time.monotonic()
    saved, failed = await fetcher.fetch_all(pending, desc=desc)
    for round_no in range(1, max_retry_rounds + 1):
        if not failed:
            break
        log.info("Retry round %d: %d request(s)", round_no, len(failed))
        more, failed = await fetcher.fetch_all(
            [key for key, _ in failed], desc=f"retry {round_no}"
        )
        saved.extend(more)
    elapsed = time.monotonic() - started
    if fetcher.latencies:
        lat = np.array(fetcher.latencies)
        log.info(
            "%d attempt(s) in %.1f s (%.1f req/s); latency median %.2f s, "
            "p90 %.2f s, max %.2f s; request gap now %.1f ms",
            len(lat),
            elapsed,
            len(lat) / elapsed,
            np.median(lat),
            np.percentile(lat, 90),
            lat.max(),
            fetcher.pacer.interval * 1000,
        )
    return saved, failed


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _run(coro):
    """asyncio.run, also from inside a running loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict = {}

    def target():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


# ---------------------------------------------------------------------------
# Steps 4-5: merge and filter
# ---------------------------------------------------------------------------


class FeatureTableBuilder:
    """Collects tile tables (TILE_SCHEMA) into one table of compact columns.

    Thailand's tiles hold about 17 million features. Class and layer names
    are kept once each and referenced by integer codes, so a feature costs
    about 50 bytes in memory. A missing id or timestamp is -1.
    """

    NUMERIC = {
        "id": np.int64,
        "longitude": np.float64,
        "latitude": np.float64,
        "first_seen_at": np.int64,
        "last_seen_at": np.int64,
        "tile_z": np.int8,
        "tile_x": np.int32,
        "tile_y": np.int32,
    }

    def __init__(self) -> None:
        self._chunks: dict[str, list[np.ndarray]] = {}
        self._classes: dict[str, int] = {}
        self._layers: dict[str, int] = {}

    @staticmethod
    def _codes(vocab: dict[str, int], values: pd.Series) -> np.ndarray:
        """Strings -> codes shared across every tile; missing values -> -1."""
        cat = pd.Categorical(values)
        lookup = np.array([vocab.setdefault(c, len(vocab)) for c in cat.categories], dtype=np.int32)
        codes = cat.codes.astype(np.int64)
        return np.where(codes >= 0, lookup[np.maximum(codes, 0)] if len(lookup) else -1, -1)

    def add_frame(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        columns = {name: df[name].to_numpy(dtype=dtype) for name, dtype in self.NUMERIC.items()}
        columns["class_code"] = self._codes(self._classes, df["object_value"]).astype(np.int32)
        columns["layer_code"] = self._codes(self._layers, df["layer"]).astype(np.int16)
        for name, values in columns.items():
            self._chunks.setdefault(name, []).append(values)

    def to_frame(self) -> pd.DataFrame:
        if not self._chunks:
            df = pd.DataFrame({k: np.array([], dtype=t) for k, t in self.NUMERIC.items()})
            df["object_value"] = pd.Categorical([])
            df["layer"] = pd.Categorical([])
            return df
        df = pd.DataFrame({k: np.concatenate(v) for k, v in self._chunks.items()})
        df["object_value"] = pd.Categorical.from_codes(
            df.pop("class_code"), categories=list(self._classes)
        )
        df["layer"] = pd.Categorical.from_codes(df.pop("layer_code"), categories=list(self._layers))
        return df


class IdDeduplicator:
    """Keeps one row per Mapillary map feature id.

    Mapillary gives one id to one map feature, built from detections in
    several images. The same id shows up twice when a feature sits exactly
    on a tile edge (3 of 17.9 million features in Maharashtra and Thailand).
    Among rows sharing an id, the row kept is the one nearest to the point
    (median longitude, median latitude) of those rows; distance uses the
    flat-earth approximation, with longitude differences scaled by
    cos(latitude), which is exact enough at these sub-metre spreads. Ties
    keep the row read first. Rows without an id (-1) are all kept.
    """

    def dedupe(self, df: pd.DataFrame) -> pd.DataFrame:
        ids = df["id"].to_numpy()
        dup = (ids >= 0) & df["id"].duplicated(keep=False).to_numpy()
        if not dup.any():
            return df
        rows = df.loc[dup, ["id", "longitude", "latitude"]]
        med = rows.groupby("id")[["longitude", "latitude"]].transform("median")
        scale = np.cos(np.radians(rows["latitude"].to_numpy()))
        dist2 = ((rows["longitude"] - med["longitude"]) * scale) ** 2 + (
            rows["latitude"] - med["latitude"]
        ) ** 2
        keep_labels = dist2.groupby(rows["id"]).idxmin()
        drop = rows.index.difference(pd.Index(keep_labels.to_numpy()))
        return df.drop(index=drop).reset_index(drop=True)


class FeatureFilter:
    """Keeps features of the requested classes that lie inside the corridor.

    object_values are Mapillary class names, e.g. the zebra crossing class
    "marking--discrete--crosswalk-zebra" in the points taxonomy. Check exact
    strings against the official list before use:
    https://www.mapillary.com/developer/api-documentation/points
    https://www.mapillary.com/developer/api-documentation/traffic-signs
    None disables the class condition (every class inside the corridor).

    The corridor test runs tile by tile: a tile lying wholly inside the
    corridor keeps all its points and a tile outside it drops them, so only
    tiles crossing the corridor edge test each point.
    """

    def __init__(self, corridor: BaseGeometry, object_values: Iterable[str] | None):
        self.corridor = corridor
        self.object_values = set(object_values) if object_values is not None else None

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        keep = np.ones(len(df), dtype=bool)
        if self.object_values is not None:
            keep &= df["object_value"].isin(self.object_values).to_numpy()
        rows = np.flatnonzero(keep)
        if len(rows) == 0:
            return keep
        test = CorridorTileTest(self.corridor)
        z = df["tile_z"].to_numpy()[rows].astype(np.int64)
        x = df["tile_x"].to_numpy()[rows].astype(np.int64)
        y = df["tile_y"].to_numpy()[rows].astype(np.int64)
        tile_key = (z << 58) | (x << 29) | y
        order = np.argsort(tile_key, kind="stable")
        _, starts = np.unique(tile_key[order], return_index=True)
        lon = df["longitude"].to_numpy()
        lat = df["latitude"].to_numpy()
        for group in tqdm(np.split(order, starts[1:]), desc="filter", unit="tile"):
            idx = rows[group]
            i = group[0]
            place = test.classify(int(z[i]), int(x[i]), int(y[i]))
            if place == CorridorTileTest.INSIDE:
                continue
            if place == CorridorTileTest.OUTSIDE:
                keep[idx] = False
                continue
            keep[idx] = test.points_inside(lon[idx], lat[idx])
        return keep


class CorridorTileTest:
    """Where a tile lies relative to the corridor, and which points are inside.

    classify() lets callers keep a whole tile inside the corridor and skip a
    whole tile outside it; only tiles crossing the corridor edge need the
    per-point test of points_inside().
    """

    INSIDE = "inside"
    OUTSIDE = "outside"
    EDGE = "edge"

    def __init__(self, corridor: BaseGeometry):
        self.corridor = corridor
        shapely.prepare(self.corridor)

    def classify(self, z: int, x: int, y: int) -> str:
        tile_box = shapely.box(*mercantile.bounds(x, y, z))
        if self.corridor.contains(tile_box):
            return self.INSIDE
        if not self.corridor.intersects(tile_box):
            return self.OUTSIDE
        return self.EDGE

    def points_inside(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return shapely.contains_xy(self.corridor, lon, lat)


EARTH_RADIUS_M = 6_371_008.8


class NearbyFeatureMerger:
    """Merges features of the same class that lie within radius_m of each other.

    Mapillary often stores one physical object as many map features (one
    Sisaket tile holds up to 45 utility poles on the same 1 m spot).

    Grouping, per object_value: features are visited from the earliest
    first_seen_at; each feature not yet in a group starts a new group and
    takes every ungrouped feature within radius_m of it. So every member
    lies within radius_m of its group's first feature, and a group is at
    most 2 * radius_m across. Linking every pair within radius_m
    transitively is avoided on purpose: in dense data (Sisaket, Thailand)
    such chains joined up to 789,360 poles spread over several kilometres.

    Each group becomes one row: id of its first feature, mean position,
    earliest first_seen_at, latest last_seen_at, and merged_count members.

    Distances are straight lines between points on a sphere in 3D
    Earth-centred coordinates, so one KD-tree covers any area without
    choosing a projection. At 1 m they match surface distances to well
    under a millimetre.
    """

    def __init__(self, radius_m: float = 1.0):
        self.radius_m = radius_m

    def merge(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.radius_m <= 0 or df.empty:
            return df.assign(merged_count=np.ones(len(df), dtype=np.int32))
        lon = np.radians(df["longitude"].to_numpy())
        lat = np.radians(df["latitude"].to_numpy())
        xyz = EARTH_RADIUS_M * np.column_stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        # Missing first_seen_at (-1) sorts after every real timestamp.
        first_seen = df["first_seen_at"].to_numpy()
        seen_key = np.where(first_seen < 0, np.iinfo(np.int64).max, first_seen)
        codes = df["object_value"].cat.codes.to_numpy()
        group = np.empty(len(df), dtype=np.int64)
        for code in tqdm(np.unique(codes), desc="merge", unit="class"):
            idx = np.flatnonzero(codes == code)
            pairs = cKDTree(xyz[idx]).query_pairs(self.radius_m, output_type="ndarray")
            group[idx] = idx[self._leader_groups(len(idx), pairs, seen_key[idx])]

        ordered = df.assign(_group=group, _seen_key=seen_key).sort_values(
            ["_group", "_seen_key"], kind="stable"
        )
        grouped = ordered.groupby("_group", sort=False, observed=True)
        out = grouped.agg(
            id=("id", "first"),
            object_value=("object_value", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            first_seen_at=("_seen_key", "min"),
            last_seen_at=("last_seen_at", "max"),
            layer=("layer", "first"),
            tile_z=("tile_z", "first"),
            tile_x=("tile_x", "first"),
            tile_y=("tile_y", "first"),
            merged_count=("id", "size"),
        ).reset_index(drop=True)
        out["first_seen_at"] = out["first_seen_at"].where(
            out["first_seen_at"] != np.iinfo(np.int64).max, -1
        )
        return out

    @staticmethod
    def _leader_groups(n: int, pairs: np.ndarray, seen_key: np.ndarray) -> np.ndarray:
        """Local index of each point's group leader (see the class docstring)."""
        leader = np.arange(n)
        if len(pairs) == 0:
            return leader
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adjacency = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n)).tocsr()
        indptr, indices = adjacency.indptr, adjacency.indices
        assigned = np.zeros(n, dtype=bool)
        # Points without a neighbour within the radius stay alone; only the
        # rest need the ordered pass.
        has_neighbour = np.diff(indptr) > 0
        candidates = np.flatnonzero(has_neighbour)
        for i in candidates[np.argsort(seen_key[candidates], kind="stable")]:
            if assigned[i]:
                continue
            assigned[i] = True
            near = indices[indptr[i] : indptr[i + 1]]
            near = near[~assigned[near]]
            assigned[near] = True
            leader[near] = i
        return leader


def _utc_date(ms: np.ndarray) -> pa.Array:
    """Epoch milliseconds -> "YYYY-MM-DD" UTC strings; -1 becomes null."""
    text = np.datetime_as_string(ms.astype("datetime64[ms]"), unit="D").astype(object)
    return pa.array(text, type=pa.string(), mask=ms < 0)


def write_features_parquet(df: pd.DataFrame, path: Path) -> None:
    """Plain Parquet (no geometry column) that kepler.gl loads as points.

    kepler.gl builds a point layer from a latitude/longitude column pair;
    "latitude"/"longitude" is one of the name pairs it looks for. Column
    types are kept to plain Arrow types that the browser reads without
    conversion: strings for class names (pandas categoricals would become
    Arrow dictionaries) and strings for Mapillary ids, because 64-bit
    integers arrive in JavaScript as BigInt.

    Size choices, measured on Thailand's 10 million rows: timestamps are
    "YYYY-MM-DD" UTC dates (a few thousand distinct values compress to
    6 MB; to the second they take 31-50 MB, as strings or as integers),
    and zstd compression (179 MB against 276 MB with snappy); kepler.gl
    reads zstd.
    """
    ids = df["id"].to_numpy()
    table = pa.table(
        {
            "id": pa.array(ids.astype(str).astype(object), type=pa.string(), mask=ids < 0),
            "object_value": pa.array(df["object_value"].astype(object), type=pa.string()),
            "latitude": pa.array(df["latitude"].to_numpy(), type=pa.float64()),
            "longitude": pa.array(df["longitude"].to_numpy(), type=pa.float64()),
            "merged_count": pa.array(df["merged_count"].to_numpy(), type=pa.int32()),
            "first_seen_at": _utc_date(df["first_seen_at"].to_numpy()),
            "last_seen_at": _utc_date(df["last_seen_at"].to_numpy()),
            "layer": pa.array(df["layer"].astype(object), type=pa.string()),
            "tile_z": pa.array(df["tile_z"].to_numpy(), type=pa.int32()),
            "tile_x": pa.array(df["tile_x"].to_numpy(), type=pa.int32()),
            "tile_y": pa.array(df["tile_y"].to_numpy(), type=pa.int32()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def class_file_token(object_value) -> str:
    """Class name as a file name part: "unknown" when missing; characters
    outside letters, digits, "-", "_" and "." become "_" (Mapillary class
    names such as "marking--discrete--crosswalk-zebra" need no change)."""
    if object_value is None or (isinstance(object_value, float) and math.isnan(object_value)):
        return "unknown"
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(object_value)) or "unknown"


def write_features_by_class(
    df: pd.DataFrame, out_dir: Path, region: str, stamp: str
) -> dict[str, Path]:
    """One Parquet per object_value: {out_dir}/{region}_{class}_{stamp}.parquet.

    kepler.gl then loads only the classes needed; in Thailand utility poles
    alone are 4.6 million of 10 million rows. Returns {class token: path}.
    """
    tokens = df["object_value"].astype(object).map(class_file_token)
    paths: dict[str, Path] = {}
    for token, part in df.groupby(tokens.to_numpy(), sort=True):
        path = Path(out_dir) / f"{region}_{token}_{stamp}.parquet"
        write_features_parquet(part.reset_index(drop=True), path)
        paths[token] = path
    return paths


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def resolve_access_token(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        from dotenv import load_dotenv

        # utf-8-sig: the repository's .env starts with a byte order mark,
        # which plain utf-8 would glue onto the first variable name.
        load_dotenv(encoding="utf-8-sig")
    except ImportError:
        pass
    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        raise RuntimeError(
            f"No Mapillary access token: pass access_token or set {TOKEN_ENV_VAR}."
        )
    return token


class MapFeatureExtractor:
    """Road GeoJSON -> Mapillary map features inside a buffer around the roads.

    The collaborators (store, decoder, pacer, backoff) can be injected to swap
    behaviour, for example in tests; the defaults cover normal use.
    """

    def __init__(
        self,
        geojson_paths: str | Path | Sequence[str | Path],
        region: str,
        tile_save_dir: str | Path = DEFAULT_TILE_SAVE_DIR,
        buffer_m: float = 30.0,
        access_token: str | None = None,
        max_concurrency: int = 20,
        object_values: Sequence[str] | None = None,
        layers: Iterable[str] = DEFAULT_LAYERS,
        max_retry_rounds: int = 3,
        merge_radius_m: float = 1.0,
        store: TileStore | None = None,
        decoder: VectorTileDecoder | None = None,
        pacer: SigmoidIntervalPacer | None = None,
        backoff: BackoffPolicy | None = None,
    ):
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}.")
        if isinstance(geojson_paths, (str, Path)):
            geojson_paths = [geojson_paths]
        self.geojson_paths = [Path(p) for p in dict.fromkeys(geojson_paths)]
        if not self.geojson_paths:
            raise ValueError("geojson_paths is empty; give at least one road GeoJSON.")
        # The region names the output folder and prefixes each output file,
        # e.g. "Maharashtra" -> filtered_map_features/Maharashtra/Maharashtra_...
        if not region or region != Path(region).name or region in (".", ".."):
            raise ValueError(f"region must be a plain folder name, got {region!r}.")
        self.region = region
        self.tile_save_dir = Path(tile_save_dir)
        self.buffer_m = buffer_m
        self.access_token = resolve_access_token(access_token)
        self.max_concurrency = max_concurrency
        self.object_values = list(object_values) if object_values is not None else None
        self.layers = resolve_layers(layers)
        self.zoom = select_zoom(self.layers)
        self.max_retry_rounds = max_retry_rounds
        self.merge_radius_m = merge_radius_m

        self.store = store or TileStore(self.tile_save_dir)
        self.decoder = decoder or VectorTileDecoder()
        self.pacer = pacer
        self.backoff = backoff or BackoffPolicy()

        self.corridor_polygon: BaseGeometry | None = None
        self.tiles: list[tuple[int, int, int]] | None = None
        self.saved_tiles: list[tuple[str, int, int, int, Path]] | None = None
        self.failed_tiles: list[tuple[TileKey, str]] = []
        self.decoded_features: pd.DataFrame | None = None
        self.filtered_features: pd.DataFrame | None = None
        self.output_dir: Path | None = None
        self.output_paths: dict[str, Path] = {}

    # 1 ---------------------------------------------------------------
    def generate_buffer_polygon(self) -> BaseGeometry:
        self.corridor_polygon = generate_buffer_polygon(self.geojson_paths, self.buffer_m)
        return self.corridor_polygon

    # 2 ---------------------------------------------------------------
    def compute_tiles(self) -> list[tuple[int, int, int]]:
        if self.corridor_polygon is None:
            raise RuntimeError("Run generate_buffer_polygon() before compute_tiles().")
        self.tiles = tiles_touching(self.corridor_polygon, self.zoom)
        n_requests = len(self.tiles) * len(self.layers)
        print(f"対象タイル数: {len(self.tiles)} (z{self.zoom})")
        print(
            f"必要リクエスト数: {n_requests} "
            f"({len(self.tiles)} タイル × {len(self.layers)} レイヤー)"
        )
        return self.tiles

    def save_tiles_parquet(self, output_path: str | Path | None = None) -> Path:
        """Write the tiles from compute_tiles() as square polygons, for review in QGIS.

        The default name {region}_z{zoom}.parquet carries no timestamp, so
        each run overwrites the region's previous file.
        """
        if self.tiles is None:
            raise RuntimeError("Run compute_tiles() before save_tiles_parquet().")
        if output_path is None:
            output_path = (
                self.tile_save_dir.parent / TILE_COVER_DIR_NAME / f"{self.region}_z{self.zoom}.parquet"
            )
        output_path = Path(output_path)
        write_tile_cover_parquet(self.tiles, output_path)
        print(f"タイル範囲のParquet: {output_path}")
        return output_path

    # 3 ---------------------------------------------------------------
    def request_and_save_vecor_tiles(
        self, max_requests: int | None = None
    ) -> list[tuple[str, int, int, int, Path]]:
        """Fetch, decode and save every (layer, tile) not saved yet.

        max_requests caps how many unsaved (layer, tile) pairs this call
        fetches (retries of those pairs come on top); None fetches all.
        """
        if self.tiles is None:
            raise RuntimeError("Run compute_tiles() before request_and_save_vecor_tiles().")
        # Tile-major order, so a capped run gets every layer of the same tiles.
        keys = [TileKey(s.name, z, x, y) for (z, x, y) in self.tiles for s in self.layers]

        saved: list[tuple[TileKey, Path]] = []
        pending: list[TileKey] = []
        for key in keys:
            path = self.store.existing(key)
            if path is not None:
                saved.append((key, path))
            else:
                pending.append(key)
        log.info("%d request(s) already saved, %d to fetch", len(saved), len(pending))
        if len(pending) > VECTOR_TILE_DAILY_LIMIT:
            log.warning(
                "%d requests exceed the vector tile limit of %d per day; "
                "expect 429 responses. Re-run later to resume from the saved tiles.",
                len(pending),
                VECTOR_TILE_DAILY_LIMIT,
            )
        if max_requests is not None and len(pending) > max_requests:
            log.info("max_requests=%d: fetching the first %d of them", max_requests, max_requests)
            pending = pending[:max_requests]

        failed: list[tuple[TileKey, str]] = []
        if pending:
            saved_now, failed = _run(self._fetch_rounds(pending))
            saved.extend(saved_now)

        self.failed_tiles = failed
        for key, reason in failed:
            log.error("FAILED %s z%d/%d/%d: %s", key.layer, key.z, key.x, key.y, reason)
        if failed:
            log.error("%d request(s) failed after %d retry round(s).", len(failed), self.max_retry_rounds)

        self.saved_tiles = [(k.layer, k.z, k.x, k.y, p) for k, p in saved]
        return self.saved_tiles

    async def _fetch_rounds(self, pending: list[TileKey]):
        # The pacer holds an asyncio.Lock, so it is created inside the loop
        # that uses it.
        fetcher = TileFetcher(
            access_token=self.access_token,
            store=self.store,
            decoder=self.decoder,
            pacer=self.pacer or SigmoidIntervalPacer(),
            backoff=self.backoff,
            max_concurrency=self.max_concurrency,
        )
        return await fetch_in_rounds(fetcher, pending, self.max_retry_rounds)

    # 4 ---------------------------------------------------------------
    def decode(self) -> pd.DataFrame:
        """Read every saved tile, one at a time, into one table; one row per id."""
        if self.saved_tiles is None:
            raise RuntimeError("Run request_and_save_vecor_tiles() before decode().")
        builder = FeatureTableBuilder()
        for *_, path in tqdm(self.saved_tiles, desc="decode", unit="tile"):
            builder.add_frame(self.store.load(path))
        table = builder.to_frame()
        self.decoded_features = IdDeduplicator().dedupe(table)
        log.info(
            "%d feature(s) read from %d tile file(s); %d left after keeping one row per id",
            len(table),
            len(self.saved_tiles),
            len(self.decoded_features),
        )
        return self.decoded_features

    # 5 ---------------------------------------------------------------
    def filter(self, output_dir: str | Path | None = None) -> pd.DataFrame:
        """Keep corridor features of the requested classes, merge them within
        merge_radius_m, and write one Parquet per class to output_dir
        (default: filtered_map_features/{region}/ next to tile_save_dir)."""
        if self.decoded_features is None:
            raise RuntimeError("Run decode() before filter().")
        if self.corridor_polygon is None:
            raise RuntimeError("Run generate_buffer_polygon() before filter().")
        df = self.decoded_features
        keep = FeatureFilter(self.corridor_polygon, self.object_values).mask(df)
        kept = df[keep].reset_index(drop=True)
        log.info("%d of %d feature(s) inside the corridor", len(kept), len(df))
        self.filtered_features = NearbyFeatureMerger(self.merge_radius_m).merge(kept)
        log.info(
            "%d feature(s) after merging same-class features within %.1f m",
            len(self.filtered_features),
            self.merge_radius_m,
        )

        if output_dir is None:
            output_dir = self.tile_save_dir.parent / FILTERED_DIR_NAME / self.region
        self.output_dir = Path(output_dir)
        self.output_paths = write_features_by_class(
            self.filtered_features, self.output_dir, self.region, utc_timestamp()
        )
        print(f"コリドー内のMap Feature: {len(self.filtered_features)}件")
        print(f"出力ファイル数: {len(self.output_paths)} (種類ごと)")
        return self.filtered_features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TqdmLoggingHandler(logging.Handler):
    """Writes log lines through tqdm so they do not break the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "geojson_paths",
        type=Path,
        nargs="+",
        help="road network GeoJSON file(s) (LineString/MultiLineString); several files are merged",
    )
    p.add_argument(
        "--tile-save-dir",
        type=Path,
        default=DEFAULT_TILE_SAVE_DIR,
        help=f"folder for decoded tiles (default: {DEFAULT_TILE_SAVE_DIR})",
    )
    p.add_argument(
        "--region",
        required=True,
        help="region name for the output folder and file names, e.g. Maharashtra",
    )
    p.add_argument("--buffer-m", type=float, default=30.0)
    p.add_argument("--access-token", default=None, help=f"default: ${TOKEN_ENV_VAR}")
    p.add_argument("--max-concurrency", type=int, default=20)
    p.add_argument("--object-values", nargs="+", default=None, metavar="VALUE")
    p.add_argument(
        "--layers",
        nargs="+",
        default=list(DEFAULT_LAYERS),
        choices=list(LAYER_REGISTRY),
        help=f"vector tile layers to fetch (default: {' '.join(DEFAULT_LAYERS)})",
    )
    p.add_argument("--max-retry-rounds", type=int, default=3)
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="fetch at most this many unsaved (layer, tile) pairs in this run",
    )
    p.add_argument(
        "--merge-radius-m",
        type=float,
        default=1.0,
        help="merge same-class features within this distance (0 disables; default: 1.0)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"folder for the per-class Parquet files (default: {FILTERED_DIR_NAME}/REGION)",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="stop after compute_tiles(): print tile and request counts, call no API",
    )
    args = p.parse_args(argv)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    extractor = MapFeatureExtractor(
        geojson_paths=args.geojson_paths,
        region=args.region,
        tile_save_dir=args.tile_save_dir,
        buffer_m=args.buffer_m,
        access_token=args.access_token,
        max_concurrency=args.max_concurrency,
        object_values=args.object_values,
        layers=args.layers,
        max_retry_rounds=args.max_retry_rounds,
        merge_radius_m=args.merge_radius_m,
    )
    extractor.generate_buffer_polygon()
    extractor.compute_tiles()
    extractor.save_tiles_parquet()
    if args.plan_only:
        return 0
    extractor.request_and_save_vecor_tiles(max_requests=args.max_requests)
    extractor.decode()
    extractor.filter(args.output_dir)
    print(f"該当件数: {len(extractor.filtered_features)}")
    print(f"出力フォルダ: {extractor.output_dir} ({len(extractor.output_paths)} ファイル)")
    return 1 if extractor.failed_tiles else 0


if __name__ == "__main__":
    sys.exit(main())
