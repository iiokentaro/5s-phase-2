"""Convert saved vector tile JSON files to the Parquet / .empty tile format.

extract_map_features.py first saved each decoded tile as
{y}_{fetched_at}.json; it now saves {y}_{fetched_at}.parquet, or a zero-byte
{y}_{fetched_at}.empty for a tile without features. This script converts
every JSON file under the tile folder with TileStore.save, keeping the
fetched_at of each file, then checks a random sample of the conversions
feature by feature. With --delete-json, the JSON files are deleted only when
every sampled file matches.

Usage
-----
    python src/migrate_tile_json_to_parquet.py                # convert and check
    python src/migrate_tile_json_to_parquet.py --delete-json  # ... then delete JSON
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from tqdm import tqdm

from extract_map_features import DEFAULT_TILE_SAVE_DIR, TileKey, TileStore


def tile_key(payload: dict) -> TileKey:
    t = payload["tile"]
    return TileKey(payload["layer"], t["z"], t["x"], t["y"])


def convert(json_path: Path, store: TileStore) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return store.save(tile_key(payload), payload["features"], payload["fetched_at"])


def expected_path(json_path: Path, has_features: bool) -> Path:
    return json_path.with_suffix(".parquet" if has_features else ".empty")


def compare(json_path: Path, store: TileStore) -> list[str]:
    """Differences between a JSON tile and its converted file (empty = identical)."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    features = payload["features"]
    target = expected_path(json_path, bool(features))
    if not target.exists():
        return [f"{target.name} is missing"]
    df = store.load(target)
    if len(df) != len(features):
        return [f"{len(features)} features in JSON, {len(df)} rows in {target.name}"]
    problems = []
    for i, (feat, row) in enumerate(zip(features, df.itertuples(index=False))):
        p = feat["properties"]
        want = {
            "id": p["id"],
            "object_value": p["object_value"],
            "longitude": feat["geometry"]["coordinates"][0],
            "latitude": feat["geometry"]["coordinates"][1],
            "first_seen_at": p["first_seen_at"],
            "last_seen_at": p["last_seen_at"],
            "layer": p["layer"],
            "tile_z": p["tile_z"],
            "tile_x": p["tile_x"],
            "tile_y": p["tile_y"],
        }
        extra = set(p) - set(want) - {"value"}
        if extra:
            problems.append(f"row {i}: JSON properties not stored: {sorted(extra)}")
        if p.get("value") != p["object_value"]:
            problems.append(f"row {i}: value {p.get('value')!r} != object_value {p['object_value']!r}")
        if feat["geometry"]["type"] != "Point":
            problems.append(f"row {i}: geometry {feat['geometry']['type']}")
        for name, value in want.items():
            got = getattr(row, name)
            if got != value:
                problems.append(f"row {i}: {name} JSON {value!r} != Parquet {got!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tile-save-dir", type=Path, default=DEFAULT_TILE_SAVE_DIR)
    ap.add_argument("--sample-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--delete-json",
        action="store_true",
        help="delete the JSON files when every sampled conversion matches",
    )
    args = ap.parse_args(argv)

    store = TileStore(args.tile_save_dir)
    json_files = sorted(args.tile_save_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON tile files under {args.tile_save_dir}.")
        return 0
    json_bytes = sum(f.stat().st_size for f in json_files)
    print(f"JSON: {len(json_files)} ファイル, {json_bytes / 1e9:.2f} GB")

    written = [convert(f, store) for f in tqdm(json_files, desc="convert", unit="file")]
    n_parquet = sum(p.suffix == ".parquet" for p in written)
    out_bytes = sum(p.stat().st_size for p in written)
    print(
        f"変換後: .parquet {n_parquet} ファイル, .empty {len(written) - n_parquet} ファイル, "
        f"{out_bytes / 1e9:.3f} GB"
    )

    k = math.ceil(len(json_files) * args.sample_fraction)
    sample = random.Random(args.seed).sample(json_files, k)
    failures = {}
    for f in tqdm(sample, desc="verify", unit="file"):
        problems = compare(f, store)
        if problems:
            failures[f] = problems
    n_features = sum(len(json.loads(f.read_text(encoding="utf-8"))["features"]) for f in sample)
    print(f"検証: {k} ファイル (地物 {n_features} 件), 不一致 {len(failures)} ファイル")
    for f, problems in list(failures.items())[:20]:
        print(f"  {f}: {problems[:5]}", file=sys.stderr)
    if failures:
        print("不一致があるため JSON は削除しません。", file=sys.stderr)
        return 1

    if args.delete_json:
        for f in json_files:
            f.unlink()
        print(f"JSON {len(json_files)} ファイルを削除しました ({json_bytes / 1e9:.2f} GB)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
