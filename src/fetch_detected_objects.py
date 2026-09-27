"""Record which objects Mapillary detected in the images at each segment's central image.

Downstream, V_safe may be raised on a segment when the roadside has the right
objects. For every non-POI segment with a central image
(data/mapillary/central_image_ids/{region}_central_image_ids.csv, written by
central_image_ids.py) this module:

  ① takes the box whose opposite corners are the central image's original
     position (lat_original, lng_original) and its position matched to the
     segment (lat_mapmatched, lng_mapmatched);
  ② widens it by pad_m metres on every side, converting metres to degrees at
     that latitude (metres_to_degrees), and rounds the edges outwards to 6
     decimals;
  ③ asks the Graph API for the images in the box (/images?bbox=…, at most
     `limit` images) with their positions and detections;
  ④ saves one GeoJSON file per segment: the request, and for each image its
     position and the object classes detected in it.

Choices measured on the live API in September 2026
--------------------------------------------------
- The bbox search takes `limit` candidates first and filters them by the box
  afterwards. A box of a few metres returned no image with limit=20; on 20
  Thailand segments limit=100 missed the central image of 2 to 9 of them
  (the result varies between calls), while 500 and 2000 found all 20, and
  2000 sometimes returned more images than 500. Responses took about 0.4 s
  whatever the limit. Default limit: 2000, the API maximum. Where the API
  answers HTTP 500 "reduce the amount of data" (a few very dense places),
  the search is repeated at limit 500, then 100, then 10; request.limit records the
  limit that succeeded.
- lat/lng_original come from z14 vector tiles, quantised to about 0.56 m, and
  differ from the API position by up to 0.76 m. With 0.1 m of padding the
  central image lay inside the box on 26 of 80 segments, with 1.0 m on all
  80. Default pad_m: 1.0.
- A GeoJSON file with 1 to 8 images takes 0.6 to 2.8 KB; a GeoParquet file
  holding the same takes about 3.4 KB, most of it fixed schema and geo
  metadata. Hence one GeoJSON per segment. kepler.gl and QGIS open it.

Output
------
data/mapillary/detected_objects/{region}/{segment_id}.geojson
    {"type": "FeatureCollection",
     "request": {method, segment_id, central_image_id, center [lat, lng],
                 bbox "w,s,e,n", limit, pad_m, fields, fetched_at},
     "features": [Point features; properties image_id, captured_at
                  (YYYY-MM-DD, UTC), lat, lng, position, objects]}
    lat/lng (and the Point) come from computed_geometry when the image has
    one, else from geometry ("position" says which), rounded to 6 decimals.
    "objects" lists the distinct detected classes (detections.value), sorted
    and comma-separated: whether a class appears, not how often. A box with
    no image is saved with an empty feature list, so the segment counts as
    done.
    method is "bbox" when the images come from the bbox search (limit is the
    limit that succeeded). Where the search fails at every limit (HTTP 500
    "reduce the amount of data", which even limit=1 gets at 2 Thailand spots),
    method is "central_image": the central image alone, looked up by id,
    saved in the same format (limit is the last limit tried).

Requests are spaced as in the other Mapillary modules: SigmoidIntervalPacer
learns the gap between request starts from successes and 429 responses, and
BackoffPolicy spaces retries. Re-running skips every segment that already has
a file, so an interrupted run resumes where it stopped.

Usage
-----
    python src/fetch_detected_objects.py Maharashtra
    python src/fetch_detected_objects.py Thailand --max-segments 20
    python src/fetch_detected_objects.py Thailand --limit 500 --pad-m 1.0

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
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_map_features import (  # noqa: E402
    BackoffPolicy,
    FetchError,
    PacedBatchFetcher,
    PacedGetter,
    RequestFailed,
    SigmoidIntervalPacer,
    TqdmLoggingHandler,
    _run,
    fetch_in_rounds,
    resolve_access_token,
    utc_timestamp,
)

log = logging.getLogger("fetch_detected_objects")

REGIONS = ("Maharashtra", "Thailand")
CENTRAL_DIR = Path("data/mapillary/central_image_ids")
OUT_DIR = Path("data/mapillary/detected_objects")
CATALOG_PATH = Path("data/mapillary/mapillary_detectable_objects.csv")
GRAPH_URL = "https://graph.mapillary.com"
FIELDS = "id,captured_at,geometry,computed_geometry,detections.value"
DEFAULT_LIMIT = 2000
# Tried in turn when a search at a larger limit answers "reduce the amount of
# data". In a few very dense places (5 Thailand segments in September 2026)
# limit=2000 does, although the box is a few metres wide; for 2 of them
# limit=100 did as well.
FALLBACK_LIMITS = (500, 100, 10)
DEFAULT_PAD_M = 1.0
DECIMALS = 6
CENTRAL_COLUMNS = (
    "segment_id",
    "central_mapillary_image_id",
    "lat_original",
    "lng_original",
    "lat_mapmatched",
    "lng_mapmatched",
)

# WGS84 ellipsoid.
_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.694_379_990_14e-3


def validate_region(region: str) -> str:
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}.")
    return region


# ---------------------------------------------------------------------------
# ① ② the box
# ---------------------------------------------------------------------------


def metres_to_degrees(metres: float, lat: float) -> tuple[float, float]:
    """(degrees of latitude, degrees of longitude) spanning `metres` at `lat`.

    Uses the WGS84 radii of curvature at that latitude: the meridional radius
    M for north-south and the prime-vertical radius N (times cos lat) for
    east-west.
    """
    if not -90 < lat < 90:
        raise ValueError(f"lat must be strictly between -90 and 90, got {lat}.")
    phi = math.radians(lat)
    w = 1 - _WGS84_E2 * math.sin(phi) ** 2
    m_radius = _WGS84_A * (1 - _WGS84_E2) / w**1.5
    n_radius = _WGS84_A / math.sqrt(w)
    return math.degrees(metres / m_radius), math.degrees(metres / (n_radius * math.cos(phi)))


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    def param(self) -> str:
        """The Graph API bbox parameter, "west,south,east,north"."""
        return ",".join(f"{v:.{DECIMALS}f}" for v in (self.west, self.south, self.east, self.north))

    @property
    def center(self) -> list[float]:
        """[lat, lng] of the box's middle, as in the request record."""
        return [(self.south + self.north) / 2, (self.west + self.east) / 2]


def segment_bbox(
    lat_a: float, lng_a: float, lat_b: float, lng_b: float, pad_m: float = DEFAULT_PAD_M
) -> BBox:
    """The box with corners a and b, widened by pad_m on every side.

    The edges are rounded outwards to DECIMALS decimals (west and south
    down, east and north up), so the rounding never takes back the padding.
    """
    if pad_m < 0:
        raise ValueError(f"pad_m must be >= 0, got {pad_m}.")
    south, north = sorted((lat_a, lat_b))
    west, east = sorted((lng_a, lng_b))
    dlat, dlng = metres_to_degrees(pad_m, (south + north) / 2)
    scale = 10**DECIMALS
    # round() first removes float noise such as 100.4544550000001 * 1e6,
    # which would otherwise floor or ceil a whole step too far.
    return BBox(
        west=math.floor(round((west - dlng) * scale, 3)) / scale,
        south=math.floor(round((south - dlat) * scale, 3)) / scale,
        east=math.ceil(round((east + dlng) * scale, 3)) / scale,
        north=math.ceil(round((north + dlat) * scale, 3)) / scale,
    )


def load_central_images(region: str, central_dir: str | Path = CENTRAL_DIR) -> pd.DataFrame:
    """The central image rows of the region, one per segment."""
    path = Path(central_dir) / f"{validate_region(region)}_central_image_ids.csv"
    df = pd.read_csv(
        path, dtype={"segment_id": str, "central_mapillary_image_id": str}
    )
    missing = [c for c in CENTRAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks column(s) {missing}.")
    dup = df["segment_id"].duplicated()
    if dup.any():
        raise ValueError(f"{path} repeats segment_id(s) {df.loc[dup, 'segment_id'].head().tolist()}.")
    if df[list(CENTRAL_COLUMNS)].isna().any().any():
        raise ValueError(f"{path} has empty cells in {list(CENTRAL_COLUMNS)}.")
    return df[list(CENTRAL_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ④ parse and store
# ---------------------------------------------------------------------------


class ObjectCatalog:
    """The object classes Mapillary can detect (mapillary_detectable_objects.csv).

    A detected class missing from the list is still recorded; it is counted
    and logged once, so a new Mapillary class is noticed and never dropped.
    """

    def __init__(self, values: Iterable[str]):
        self.values = frozenset(values)
        self.unknown: Counter[str] = Counter()

    @classmethod
    def from_csv(cls, path: str | Path = CATALOG_PATH) -> "ObjectCatalog":
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        if "value" not in df.columns:
            raise ValueError(f"{path} has no 'value' column.")
        return cls(df["value"].dropna())

    def note(self, values: Iterable[str]) -> None:
        for v in values:
            if v not in self.values:
                if v not in self.unknown:
                    log.warning("Detected class %r is not in the catalog; recorded anyway.", v)
                self.unknown[v] += 1


def _utc_date(ms) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def image_feature(image: dict, catalog: ObjectCatalog) -> dict:
    """One Graph API image record -> one GeoJSON Point feature."""
    computed = image.get("computed_geometry")
    if computed and computed.get("coordinates"):
        lng, lat = computed["coordinates"][:2]
        position = "computed"
    elif image.get("geometry") and image["geometry"].get("coordinates"):
        lng, lat = image["geometry"]["coordinates"][:2]
        position = "original"
    else:
        raise ValueError(f"Image {image.get('id')} has no geometry.")
    lat, lng = round(lat, DECIMALS), round(lng, DECIMALS)
    values = sorted({d["value"] for d in (image.get("detections") or {}).get("data", []) if d.get("value")})
    catalog.note(values)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lng, lat]},
        "properties": {
            "image_id": str(image["id"]),
            "captured_at": _utc_date(image.get("captured_at")),
            "lat": lat,
            "lng": lng,
            "position": position,
            "objects": ",".join(values),
        },
    }


class DetectionStore:
    """One GeoJSON file per segment under {root}/{region}/."""

    def __init__(self, root: str | Path, region: str):
        self.dir = Path(root) / validate_region(region)

    def path(self, segment_id: str) -> Path:
        name = f"{segment_id}.geojson"
        if Path(name).name != name or segment_id in ("", ".", ".."):
            raise ValueError(f"segment_id {segment_id!r} cannot be a file name.")
        return self.dir / name

    def exists(self, segment_id: str) -> bool:
        return self.path(segment_id).is_file()

    def save(self, request: dict, features: Sequence[dict]) -> Path:
        path = self.path(request["segment_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"type": "FeatureCollection", "request": request, "features": list(features)}
        tmp = path.with_suffix(".geojson.tmp")
        tmp.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return path


# ---------------------------------------------------------------------------
# ③ fetch
# ---------------------------------------------------------------------------


def too_much_data(status: int | None, body: bytes) -> bool:
    """HTTP 500 "Please reduce the amount of data you're asking for": the same
    request fails again, so it is not retried; a smaller limit is tried instead."""
    return status == 500 and b"reduce the amount of data" in body


def fallback_limits(limit: int) -> list[int]:
    """limit, then each FALLBACK_LIMITS value below it, largest first."""
    return [limit] + [v for v in FALLBACK_LIMITS if v < limit]


@dataclass(frozen=True)
class SegmentRequest:
    segment_id: str
    central_image_id: str
    bbox: BBox
    limit: int
    pad_m: float

    def record(self, method: str) -> dict:
        return {
            "method": method,
            "segment_id": self.segment_id,
            "central_image_id": self.central_image_id,
            "center": [round(v, DECIMALS + 1) for v in self.bbox.center],
            "bbox": self.bbox.param(),
            "limit": self.limit,
            "pad_m": self.pad_m,
            "fields": FIELDS,
        }


# request.method: how the images of a file were found.
METHOD_BBOX = "bbox"  # the bbox search
METHOD_CENTRAL = "central_image"  # the central image alone, looked up by id


class DetectionFetcher(PacedBatchFetcher):
    """Fetches the images of each segment's box and saves them.

    Where the bbox search answers "reduce the amount of data" at every limit
    of fallback_limits() (a few extremely dense spots, where even limit=1
    fails), the central image alone is looked up by id and saved in the
    same format, with request.method "central_image".
    """

    def __init__(
        self,
        getter: PacedGetter,
        store: DetectionStore,
        catalog: ObjectCatalog,
        max_concurrency: int,
        graph_url: str = GRAPH_URL,
        timeout_s: float = 60.0,
    ):
        super().__init__(getter, max_concurrency, timeout_s)
        self.store = store
        self.catalog = catalog
        self.graph_url = graph_url.rstrip("/")

    async def _fetch_one(
        self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, req: SegmentRequest
    ) -> tuple[SegmentRequest, Path]:
        try:
            method, used, payload = await self._search(session, sem, req)
            images = payload.get("data")
            if not isinstance(images, list):
                raise ValueError(f"no 'data' list in the response: {str(payload)[:200]}")
            for image in images:
                await self._complete_detections(session, sem, image)
            features = [image_feature(image, self.catalog) for image in images]
            record = {**replace(req, limit=used).record(method), "fetched_at": utc_timestamp()}
            return req, self.store.save(record, features)
        except RequestFailed as exc:
            raise FetchError((req, exc.reason)) from exc
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise FetchError((req, f"{type(exc).__name__}: {exc}")) from exc

    async def _search(
        self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, req: SegmentRequest
    ) -> tuple[str, int, dict]:
        """The bbox search at req.limit, then at each smaller FALLBACK_LIMITS
        value while the API answers "reduce the amount of data"; after the
        last one, the central image alone by id. Returns the method, the
        limit (the one that succeeded, or the last one tried before the
        central image lookup) and a response of the form {"data": [...]}."""
        limits = fallback_limits(req.limit)
        for i, limit in enumerate(limits):
            params = {"bbox": req.bbox.param(), "limit": limit, "fields": FIELDS}
            try:
                body = await self.getter.get(session, sem, f"{self.graph_url}/images", params)
                return METHOD_BBOX, limit, self._json(body)
            except RequestFailed as exc:
                if not too_much_data(exc.status, exc.body):
                    raise
                if i + 1 < len(limits):
                    log.info(
                        "%s: too much data at limit=%d; retrying at limit=%d",
                        req.segment_id, limit, limits[i + 1],
                    )
        log.info(
            "%s: too much data at every limit; looking up the central image %s alone",
            req.segment_id, req.central_image_id,
        )
        url = f"{self.graph_url}/{req.central_image_id}"
        image = self._json(await self.getter.get(session, sem, url, {"fields": FIELDS}))
        if str(image.get("id")) != req.central_image_id:
            raise ValueError(f"lookup of {req.central_image_id} returned {str(image)[:200]}")
        return METHOD_CENTRAL, limits[-1], {"data": [image]}

    async def _complete_detections(
        self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, image: dict
    ) -> None:
        """Follow detections.paging.next, so no detected class is missed."""
        detections = image.get("detections")
        while detections and (detections.get("paging") or {}).get("next"):
            page = self._json(await self.getter.get(session, sem, detections["paging"]["next"]))
            detections["data"] = detections.get("data", []) + page.get("data", [])
            detections["paging"] = page.get("paging")

    @staticmethod
    def _json(body: bytes) -> dict:
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"response is not JSON: {body[:200]!r}") from exc


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_requests(central: pd.DataFrame, limit: int, pad_m: float) -> list[SegmentRequest]:
    return [
        SegmentRequest(
            segment_id=row.segment_id,
            central_image_id=row.central_mapillary_image_id,
            bbox=segment_bbox(
                row.lat_original, row.lng_original, row.lat_mapmatched, row.lng_mapmatched, pad_m
            ),
            limit=limit,
            pad_m=pad_m,
        )
        for row in central.itertuples(index=False)
    ]


def fetch_detected_objects(
    region: str,
    *,
    limit: int = DEFAULT_LIMIT,
    pad_m: float = DEFAULT_PAD_M,
    central_dir: str | Path = CENTRAL_DIR,
    out_dir: str | Path = OUT_DIR,
    catalog_path: str | Path = CATALOG_PATH,
    access_token: str | None = None,
    max_segments: int | None = None,
    max_concurrency: int = 20,
    max_retry_rounds: int = 3,
    graph_url: str = GRAPH_URL,
) -> tuple[list[tuple[SegmentRequest, Path]], list[tuple[SegmentRequest, str]]]:
    """Fetch and save the detections of every segment not saved yet.

    max_segments caps how many unsaved segments this call fetches (retries of
    those come on top). Returns the (request, path) pairs saved in this call
    and the (request, reason) pairs that still failed.
    """
    if not 1 <= limit <= 2000:
        raise ValueError(f"limit must be between 1 and 2000, got {limit}.")
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}.")
    region = validate_region(region)
    requests = build_requests(load_central_images(region, central_dir), limit, pad_m)
    store = DetectionStore(out_dir, region)
    catalog = ObjectCatalog.from_csv(catalog_path)

    pending = [r for r in requests if not store.exists(r.segment_id)]
    print(f"{region}: {len(requests):,} segment(s), {len(requests) - len(pending):,} already saved")
    if max_segments is not None and len(pending) > max_segments:
        pending = pending[:max_segments]
    print(f"リクエスト数: {len(pending):,} (limit={limit}, pad_m={pad_m:g})")
    if not pending:
        return [], []

    token = resolve_access_token(access_token)

    async def rounds():
        # The pacer holds an asyncio.Lock, so it is created inside the loop
        # that uses it.
        getter = PacedGetter(token, SigmoidIntervalPacer(), BackoffPolicy(), is_final=too_much_data)
        fetcher = DetectionFetcher(getter, store, catalog, max_concurrency, graph_url=graph_url)
        return await fetch_in_rounds(fetcher, pending, max_retry_rounds, desc="detections")

    saved, failed = _run(rounds())
    for req, reason in failed:
        log.error("FAILED %s: %s", req.segment_id, reason)
    if catalog.unknown:
        log.warning("Classes not in the catalog: %s", dict(catalog.unknown))

    n_images = n_with_objects = n_with_central = 0
    for req, path in saved:
        doc = json.loads(path.read_text(encoding="utf-8"))
        ids = [f["properties"]["image_id"] for f in doc["features"]]
        n_images += len(ids)
        n_with_objects += sum(1 for f in doc["features"] if f["properties"]["objects"])
        n_with_central += req.central_image_id in ids
    print(f"保存: {len(saved):,} セグメント (失敗 {len(failed):,})")
    print(
        f"画像: {n_images:,} 枚 (物体あり {n_with_objects:,} 枚); "
        f"central画像を含むセグメント {n_with_central:,} / {len(saved):,}"
    )
    print(f"出力フォルダ: {store.dir}")
    return saved, failed


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("region", choices=REGIONS)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"images per request (default: {DEFAULT_LIMIT})")
    p.add_argument(
        "--pad-m", type=float, default=DEFAULT_PAD_M, help=f"box padding per side, metres (default: {DEFAULT_PAD_M:g})"
    )
    p.add_argument("--central-dir", type=Path, default=CENTRAL_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    p.add_argument("--access-token", default=None, help="default: $MAPILLARY_ACCESS_TOKEN")
    p.add_argument("--max-segments", type=int, default=None, help="fetch at most this many unsaved segments")
    p.add_argument("--max-concurrency", type=int, default=20)
    p.add_argument("--max-retry-rounds", type=int, default=3)
    args = p.parse_args(argv)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    _, failed = fetch_detected_objects(
        args.region,
        limit=args.limit,
        pad_m=args.pad_m,
        central_dir=args.central_dir,
        out_dir=args.out_dir,
        catalog_path=args.catalog,
        access_token=args.access_token,
        max_segments=args.max_segments,
        max_concurrency=args.max_concurrency,
        max_retry_rounds=args.max_retry_rounds,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
