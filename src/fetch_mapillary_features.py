"""Fetch Mapillary /map_features for every road segment in both GeoJSON files.

For each feature the StreetImageLink property contains a bbox as
"lon1,lat1,lon2,lat2".  We query the Mapillary Graph API's /map_features
endpoint for the 26 VRU-related object_values listed in OBJECT_VALUES.

bboxes that are >= 0.01 deg² (the API hard limit) are split into an N×N grid
of sub-tiles; results are deduplicated by map-feature id before saving.

Outputs
-------
data/mapillary/map_features_maharashtra.json
data/mapillary/map_features_thailand.json

Each file is a JSON object keyed by OBJECTID (str) whose value is the list of
map_feature dicts (id, object_value, geometry) found within that segment's bbox.
An empty list means zero matching features – not a missing result.

Checkpointing
-------------
Completed results are appended line-by-line to a .jsonl sidecar file.
Re-running the script resumes from where it left off.

Usage
-----
    # full run (both countries)
    python src/fetch_mapillary_features.py

    # smoke-test: first N features of Maharashtra only
    python src/fetch_mapillary_features.py --limit 10

    # dry run: print tasks without hitting the API
    python src/fetch_mapillary_features.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
TOKEN: str = os.environ.get("MAPILLARY_TOKEN", "")

BASE_URL = "https://graph.mapillary.com/map_features"
MAX_BBOX_AREA = 0.0099  # deg² – stay just under the 0.01 deg² API limit
MAX_RETRIES = 3
TIMEOUT_SEC = 30
CONCURRENCY = 15

OBJECT_VALUES: list[str] = [
    "construction--flat--crosswalk-plain",
    "marking--discrete--crosswalk-zebra",
    "marking--discrete--symbol--bicycle",
    "object--bike-rack",
    "regulatory--in-street-pedestrian-crossing--g1",
    "regulatory--pedestrians-bicycles-permitted--g1",
    "regulatory--pedestrians-priority-zone--g1",
    "regulatory--pedestrians-push-button--g1",
    "regulatory--pedestrians-push-button--g2",
    "regulatory--shared-path-bicycles-and-pedestrians--g1",
    "regulatory--shared-path-pedestrians-and-bicycles--g1",
    "regulatory--yield-or-stop-for-pedestrians--g1",
    "warning--dual-path-cyclists-and-pedestrians--g1",
    "warning--pedestrians-crossing--g10",
    "warning--pedestrians-crossing--g11",
    "warning--pedestrians-crossing--g12",
    "warning--pedestrians-crossing--g1",
    "warning--pedestrians-crossing--g4",
    "warning--pedestrians-crossing--g5",
    "warning--pedestrians-crossing--g6",
    "warning--pedestrians-crossing--g7",
    "warning--pedestrians-crossing--g8",
    "warning--pedestrians-crossing--g9",
    "warning--school-zone--g2",
    "regulatory--end-of-school-zone--g1",
    "information--hospital--g1",
]
OBJ_VALUES_STR = ",".join(OBJECT_VALUES)

DATASETS = [
    {
        "geojson": "data/raw/ADB_Innovation_Maharashtra.geojson",
        "out_json": "data/mapillary/map_features_maharashtra.json",
        "ckpt": "data/mapillary/.map_features_maharashtra.jsonl",
        "country": "maharashtra",
    },
    {
        "geojson": "data/raw/ADB_Innovation_Thailand.geojson",
        "out_json": "data/mapillary/map_features_thailand.json",
        "ckpt": "data/mapillary/.map_features_thailand.jsonl",
        "country": "thailand",
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# bbox helpers
# ---------------------------------------------------------------------------


def _parse_link(link: str) -> tuple[float, float, float, float]:
    """Parse 'lon1,lat1,lon2,lat2' → (lon_min, lat_min, lon_max, lat_max)."""
    parts = link.split(",")
    lon1, lat1, lon2, lat2 = map(float, parts)
    return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)


def split_bbox(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> list[tuple[float, float, float, float]]:
    """Return a list of sub-tile bboxes each < MAX_BBOX_AREA deg².

    If the input bbox is already within the limit, returns it unchanged as a
    single-element list.
    """
    area = (lon_max - lon_min) * (lat_max - lat_min)
    if area < MAX_BBOX_AREA:
        return [(lon_min, lat_min, lon_max, lat_max)]

    n = math.ceil(math.sqrt(area / MAX_BBOX_AREA))
    lon_step = (lon_max - lon_min) / n
    lat_step = (lat_max - lat_min) / n

    tiles: list[tuple[float, float, float, float]] = []
    for i in range(n):
        for j in range(n):
            tlon_min = lon_min + i * lon_step
            tlon_max = tlon_min + lon_step
            tlat_min = lat_min + j * lat_step
            tlat_max = tlat_min + lat_step
            tiles.append((tlon_min, tlat_min, tlon_max, tlat_max))
    return tiles


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------


async def fetch_subtile(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    bbox: tuple[float, float, float, float],
    token: str,
) -> list[dict]:
    """Fetch map_features for a single sub-tile bbox with retry logic."""
    lon_min, lat_min, lon_max, lat_max = bbox
    bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    params = {
        "access_token": token,
        "bbox": bbox_str,
        "fields": "id,object_value,geometry",
        "object_values": OBJ_VALUES_STR,
        "limit": 2000,
    }

    wait = 0.0
    for attempt in range(1, MAX_RETRIES + 1):
        if wait:
            await asyncio.sleep(wait)  # sleep outside the semaphore
        async with sem:
            try:
                async with session.get(BASE_URL, params=params) as resp:
                    if resp.status >= 500:
                        log.debug(
                            "HTTP %s for bbox %s – skipping", resp.status, bbox_str
                        )
                        return []  # 5xx means no photos -- skip immediately
                    if resp.status == 429:
                        wait = 2**attempt
                        log.warning(
                            "HTTP 429 for bbox %s, retry %d/%d in %.0fs",
                            bbox_str,
                            attempt,
                            MAX_RETRIES,
                            wait,
                        )
                        continue  # release the semaphore before the next loop
                    resp.raise_for_status()
                    return (await resp.json()).get("data", [])
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.debug("Connection error (%s) for bbox %s – skipping", exc, bbox_str)
                return []  # a connection error is treated the same as no photos -- skip immediately

    log.error("All retries exhausted for bbox %s – returning empty list", bbox_str)
    return []


async def fetch_feature(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    objectid: str,
    link: str,
    token: str,
) -> tuple[str, list[dict]]:
    """Fetch all map_features for one road-segment bbox.

    Splits oversized bboxes into sub-tiles, fetches all, deduplicates by id.
    Returns (objectid, deduplicated_features).
    """
    lon_min, lat_min, lon_max, lat_max = _parse_link(link)
    tiles = split_bbox(lon_min, lat_min, lon_max, lat_max)

    tasks = [fetch_subtile(session, sem, tile, token) for tile in tiles]
    results_per_tile = await asyncio.gather(*tasks)

    seen: set[str] = set()
    merged: list[dict] = []
    for features in results_per_tile:
        for feat in features:
            fid = feat.get("id")
            if fid and fid not in seen:
                seen.add(fid)
                merged.append(feat)

    return objectid, merged


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint(path: str | Path) -> dict[str, list]:
    """Load already-completed results from a JSONL checkpoint file."""
    done: dict[str, list] = {}
    p = Path(path)
    if not p.exists():
        return done
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done[str(record["objectid"])] = record["features"]
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def append_checkpoint(path: str | Path, objectid: str, features: list[dict]) -> None:
    """Append one completed record to the JSONL checkpoint."""
    with open(path, "a") as f:
        f.write(json.dumps({"objectid": objectid, "features": features}) + "\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run(
    dataset: dict,
    token: str,
    concurrency: int = CONCURRENCY,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    geojson_path = Path(dataset["geojson"])
    out_json = Path(dataset["out_json"])
    ckpt_path = Path(dataset["ckpt"])
    country = dataset["country"]

    out_json.parent.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Loading %s", country, geojson_path)
    with geojson_path.open() as f:
        gj = json.load(f)
    features = gj["features"]
    if limit is not None:
        features = features[:limit]

    # Build full result set from checkpoint
    results: dict[str, list] = load_checkpoint(ckpt_path)
    log.info("[%s] Checkpoint: %d already done", country, len(results))

    # Build pending task list
    pending = [
        (
            str(feat["properties"]["OBJECTID"]),
            feat["properties"].get("StreetImageLink", ""),
        )
        for feat in features
        if feat["properties"].get("StreetImageLink")
        and str(feat["properties"]["OBJECTID"]) not in results
    ]
    log.info("[%s] Pending: %d features", country, len(pending))

    if dry_run:
        log.info("[%s] --dry-run: printing first 5 tasks", country)
        for oid, link in pending[:5]:
            lon_min, lat_min, lon_max, lat_max = _parse_link(link)
            tiles = split_bbox(lon_min, lat_min, lon_max, lat_max)
            print(f"  OBJECTID={oid}  link={link}  subtiles={len(tiles)}")
        return

    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SEC)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_feature(session, sem, oid, link, token) for oid, link in pending]
        total = len(tasks)
        done_count = 0
        pbar = tqdm(total=total, desc=country, unit="seg")
        for coro in asyncio.as_completed(tasks):
            objectid, feats = await coro
            results[objectid] = feats
            append_checkpoint(ckpt_path, objectid, feats)
            done_count += 1
            pbar.update(1)
            pbar.set_postfix(features=sum(len(v) for v in results.values()))
            if done_count % 500 == 0 or done_count == total:
                log.info(
                    "[%s] %d / %d done  (found %d features so far)",
                    country,
                    done_count,
                    total,
                    sum(len(v) for v in results.values()),
                )
        pbar.close()

    # Ensure all features from original file are present (even those without link)
    for feat in features:
        oid = str(feat["properties"]["OBJECTID"])
        if oid not in results:
            results[oid] = []

    log.info("[%s] Writing %s (%d entries)", country, out_json, len(results))
    with out_json.open("w") as f:
        json.dump(results, f, ensure_ascii=False)
    log.info("[%s] Done.", country)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Mapillary map_features for ADB segments"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N features of each dataset (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tasks without making API requests",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help=f"Number of concurrent requests (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--country",
        choices=["maharashtra", "thailand"],
        default=None,
        help="Process only one country (default: both)",
    )
    args = parser.parse_args()

    if not args.dry_run and not TOKEN:
        log.error("MAPILLARY_TOKEN not set. Add it to .env or export it.")
        sys.exit(1)

    datasets = DATASETS
    if args.country:
        datasets = [d for d in datasets if d["country"] == args.country]

    for dataset in datasets:
        asyncio.run(
            run(
                dataset,
                token=TOKEN,
                concurrency=args.concurrency,
                limit=args.limit,
                dry_run=args.dry_run,
            )
        )


if __name__ == "__main__":
    main()
