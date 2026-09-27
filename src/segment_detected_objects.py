"""Record, per non-POI segment, which scored Mapillary objects were detected on it.

Downstream, V_safe may be raised on a segment when the roadside has the right
objects. The images of data/mapillary/detected_objects/{region}/ (written by
fetch_detected_objects.py) carry the object classes detected in them; this
module places them on the segments and adds up the weights of
data/mapillary/mapillary_detectable_objects.csv:

  ① load_scoring_catalog   the weight columns are column E and every column
                            right of it, taken by position (their names are
                            never written here); the scored objects are the
                            rows with a non-zero weight.
  ② write_object_points    one GeoParquet of points per scored object: every
                            image (deduplicated by image_id) in whose objects
                            it appears, at the position where it was taken.
  ③ match_points           each image to its nearest OSM way within
                            MAX_WAY_M metres (osm_ways_{country}.parquet, built
                            from the region's .osm.pbf, every highway class),
                            then from that position to its nearest segment of
                            the region, POI or not, within MAX_SEGMENT_M
                            metres; images matched to a POI segment are
                            dropped, as in central_image_ids.py, and so are
                            images matched to a segment without its own
                            GeoJSON file (no images were fetched for it, so
                            its V_safe does not use them). The object
                            files of ② are overwritten with the matched rows.
                            Some images are dropped at each stage; that is
                            intended.
  ④ ⑤ segment_table        one row per non-POI segment: a Bool column per
                            scored object (True however many of its images are
                            matched there), then, per weight column, the sum of
                            the weights of the objects present. The sum can
                            exceed 1.

apply_object_separation, the build step "object_separation" of build_v_safe.py
(after the sandwich step, before the scores), reads the segment tables into
the build: a non-POI segment with MEDIAN_COL >= 1 becomes is_divided, and one
with every PROTECTION_COLS column >= 1 gets has_vru_barrier, which
safe_speed.add_v_safe turns into 50 km/h (side_impact:vru_barrier) where
there is no access control, and one with MOTORCYCLE_SEPARATION_COL >= 1 gets
has_motorcycle_separation, which lets an access-controlled segment that
motorcyclists may use take the four-wheeled speed. V_safe is only ever raised
there.

The segment table is its own file keyed by segment_id, joined to
segments_v_safe.parquet when needed: the pipeline rewrites that file, and a
change of the weights then rebuilds this small file alone.

Matching always starts from lng_original and lat_original, so running ③ again
never moves a point twice. Distances are measured in the UTM zone of each
image (geometry.utm_epsg_for_lonlat), since both regions straddle two zones.

Outputs
-------
data/mapillary/detected_object_points/{region}/{stem}.parquet
    One file per scored object; stem is the value with "--" replaced by "-"
    (construction--barrier--guard-rail -> construction-barrier-guard-rail).
    Columns: value, image_id, captured_at, position, source_segment_id (the
    segment whose request returned the image), lng_original, lat_original,
    osm_way_id, osm_match_dist_m, segment_id, segment_match_dist_m and a Point
    geometry (EPSG:4326) at the matched position on the segment. An object
    without any matched image gets a file with no rows. A .parquet file there
    that belongs to no scored object is removed.
data/mapillary/segment_detected_objects/{region}_segment_detected_objects.parquet
    segment_id, has_detection_file (the segment has its own GeoJSON file;
    False means no images were fetched, and the row is then all False and 0,
    while an all-False row with True means images without scored objects),
    n_matched_images, one Bool column
    per scored object (named by its value) and one float column per weight
    column (named as in the CSV).

Usage
-----
    python src/segment_detected_objects.py Maharashtra
    python src/segment_detected_objects.py Thailand
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from central_image_ids import load_segments, select_non_poi_segments  # noqa: E402
from fetch_detected_objects import REGIONS, metres_to_degrees, validate_region  # noqa: E402
from junction_speed_cap import JUNCTION_V_SAFE_CAP  # noqa: E402
from osm_ways import ways_path  # noqa: E402
from safe_speed import add_v_safe  # noqa: E402

CATALOG_PATH = Path("data/mapillary/mapillary_detectable_objects.csv")
DETECTED_DIR = Path("data/mapillary/detected_objects")
POINTS_DIR = Path("data/mapillary/detected_object_points")
OUT_DIR = Path("data/mapillary/segment_detected_objects")
SEGMENTS_PATH = Path("data/processed/segments_v_safe.parquet")
# Column E: the first weight column of the catalog.
FIRST_WEIGHT_COLUMN = 4
# An image farther than this from every OSM way is dropped.
MAX_WAY_M = 20.0
# A matched image farther than this from every segment is dropped.
MAX_SEGMENT_M = 8.0
# The degree margin of the candidate search is this many times the degrees
# spanned by the metric limit, so no line within the limit is missed.
_DEG_MARGIN_FACTOR = 1.5

IMAGE_COLUMNS = [
    "image_id", "captured_at", "position", "source_segment_id", "lng_original", "lat_original",
]
MATCH_COLUMNS = ["osm_way_id", "osm_match_dist_m", "segment_id", "segment_match_dist_m"]
POINT_COLUMNS = ["value", *IMAGE_COLUMNS, *MATCH_COLUMNS]


def default_out_path(region: str, out_dir: Path = OUT_DIR) -> Path:
    return Path(out_dir) / f"{region}_segment_detected_objects.parquet"


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# ①  the scored objects
# --------------------------------------------------------------------------- #


def load_scoring_catalog(path: str | Path = CATALOG_PATH,
                         first_weight_column: int = FIRST_WEIGHT_COLUMN) -> pd.DataFrame:
    """The weights of the scored objects: index value, one column per weight
    column of the CSV (column first_weight_column and every one right of it),
    only the rows with a non-zero weight."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"value": str})
    if "value" not in df.columns:
        raise ValueError(f"{path} has no 'value' column.")
    weight_cols = list(df.columns[first_weight_column:])
    if not weight_cols:
        raise ValueError(f"{path} has no column at position {first_weight_column} or right of it.")
    if df["value"].isna().any():
        raise ValueError(f"{path} has rows without a value.")
    dup = df["value"].duplicated()
    if dup.any():
        raise ValueError(f"{path} repeats value(s) {df.loc[dup, 'value'].tolist()}.")
    weights = df.set_index("value")[weight_cols].apply(pd.to_numeric, errors="coerce")
    bad = weights.isna() | (weights < 0) | (weights > 1)
    if bad.any().any():
        where = [(v, c) for v, c in zip(*np.nonzero(bad.to_numpy()))][:5]
        cells = [(weights.index[i], weight_cols[j]) for i, j in where]
        raise ValueError(f"{path}: weights must be numbers from 0 to 1; not so at {cells}.")
    return weights.loc[(weights != 0).any(axis=1)].astype(float)


def value_stem(value: str) -> str:
    """The file name stem of an object: "--" replaced by "-"."""
    stem = value.replace("--", "-")
    if not stem or Path(stem).name != stem or stem in (".", ".."):
        raise ValueError(f"value {value!r} cannot be a file name.")
    return stem


def value_stems(values: Sequence[str]) -> dict[str, str]:
    """value -> stem; two values with the same stem are refused."""
    stems = {v: value_stem(v) for v in values}
    seen: dict[str, str] = {}
    for v, s in stems.items():
        if s in seen:
            raise ValueError(f"values {seen[s]!r} and {v!r} share the file name {s!r}.")
        seen[s] = v
    return stems


# --------------------------------------------------------------------------- #
# ②  object points
# --------------------------------------------------------------------------- #


def read_detections(detected_dir: str | Path, region: str) -> tuple[pd.DataFrame, set[str]]:
    """The images of every GeoJSON file of the region, one row per image_id
    (IMAGE_COLUMNS plus objects), and the segment ids that have a file.

    An image in several files (boxes of neighbouring segments overlap) is kept
    once, from the first file in name order."""
    files = sorted((Path(detected_dir) / validate_region(region)).glob("*.geojson"))
    if not files:
        raise FileNotFoundError(f"no GeoJSON files in {Path(detected_dir) / region}")
    rows, segment_ids = [], set()
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        seg = str(doc["request"]["segment_id"])
        segment_ids.add(seg)
        for feat in doc["features"]:
            p = feat["properties"]
            rows.append((str(p["image_id"]), p.get("captured_at"), p.get("position"), seg,
                         float(p["lng"]), float(p["lat"]), p.get("objects") or ""))
    images = pd.DataFrame(rows, columns=[*IMAGE_COLUMNS, "objects"])
    return images.drop_duplicates("image_id").reset_index(drop=True), segment_ids


def object_rows(images: pd.DataFrame, values: Sequence[str]) -> pd.DataFrame:
    """value, image_id: one row per scored object detected in an image."""
    pairs = images[["image_id", "objects"]].assign(value=images["objects"].str.split(","))
    pairs = pairs.explode("value")
    pairs = pairs.loc[pairs["value"].isin(set(values)), ["value", "image_id"]]
    return pairs.drop_duplicates().reset_index(drop=True)


def _points(df: pd.DataFrame, lng: str, lat: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lng], df[lat]), crs="EPSG:4326")


def write_object_points(images: pd.DataFrame, pairs: pd.DataFrame, stems: dict[str, str],
                        points_dir: Path) -> dict[str, Path]:
    """② one GeoParquet per scored object at the original positions. Files of
    the directory that belong to no scored object are removed."""
    points_dir.mkdir(parents=True, exist_ok=True)
    rows = pairs.merge(images[IMAGE_COLUMNS], on="image_id", how="left", validate="many_to_one")
    paths = {}
    for value, stem in stems.items():
        part = rows.loc[rows["value"] == value].sort_values("image_id").reset_index(drop=True)
        paths[value] = points_dir / f"{stem}.parquet"
        _write_atomic(_points(part, "lng_original", "lat_original"), paths[value])
    keep = set(paths.values())
    for stale in sorted(points_dir.glob("*.parquet")):
        if stale not in keep:
            print(f"  removed {stale} (no longer a scored object)")
            stale.unlink()
    return paths


# --------------------------------------------------------------------------- #
# ③  matching
# --------------------------------------------------------------------------- #


def utm_epsg(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """geometry.utm_epsg_for_lonlat for arrays."""
    zone = np.floor((np.asarray(lon) + 180) / 6).astype(int) + 1
    return np.where(np.asarray(lat) >= 0, 32600, 32700) + zone


def degree_margin(metres: float, lat: np.ndarray) -> float:
    """Degrees that surely span `metres` at every latitude of lat."""
    dlat, dlng = metres_to_degrees(metres, float(np.max(np.abs(lat))))
    return _DEG_MARGIN_FACTOR * max(dlat, dlng)


def candidate_lines(points: np.ndarray, lines: gpd.GeoDataFrame, degrees: float) -> gpd.GeoDataFrame:
    """The lines (EPSG:4326) within `degrees` of some point (shapely Points)."""
    tree = shapely.STRtree(lines.geometry.values)
    _, idx = tree.query(points, predicate="dwithin", distance=degrees)
    return lines.iloc[np.unique(idx)]


def nearest_line_match(points: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, key_col: str,
                       max_distance: float) -> pd.DataFrame:
    """Each point (image_id, geometry) to its nearest line (key_col, geometry)
    within max_distance, both in the same metric CRS. Returns image_id,
    key_col, dist_m and the nearest position on the line (x, y). A point
    equally near two lines keeps the smaller key."""
    cols = ["image_id", key_col, "dist_m", "x", "y"]
    if points.empty or lines.empty:
        return pd.DataFrame(columns=cols)
    joined = gpd.sjoin_nearest(
        points[["image_id", "geometry"]], lines[[key_col, "geometry"]].reset_index(drop=True),
        how="inner", max_distance=max_distance, distance_col="dist_m",
    )
    joined = joined.sort_values(["image_id", "dist_m", key_col]).drop_duplicates("image_id")
    geoms = lines.geometry.values[joined["index_right"].to_numpy()]
    snapped = shapely.line_interpolate_point(geoms, shapely.line_locate_point(geoms, joined.geometry.values))
    return pd.DataFrame({
        "image_id": joined["image_id"].to_numpy(),
        key_col: joined[key_col].to_numpy(),
        "dist_m": joined["dist_m"].to_numpy(),
        "x": shapely.get_x(snapped),
        "y": shapely.get_y(snapped),
    })


@dataclass
class MatchStats:
    images: int = 0
    no_way: int = 0
    no_segment: int = 0
    on_poi_segment: int = 0
    on_segment_without_file: int = 0

    @property
    def kept(self) -> int:
        return (self.images - self.no_way - self.no_segment - self.on_poi_segment
                - self.on_segment_without_file)


def match_points(images: pd.DataFrame, ways: gpd.GeoDataFrame, segs: gpd.GeoDataFrame,
                 non_poi_ids: set[str], file_segment_ids: set[str],
                 max_way_m: float = MAX_WAY_M,
                 max_segment_m: float = MAX_SEGMENT_M) -> tuple[pd.DataFrame, MatchStats]:
    """③ for images (image_id, lng_original, lat_original), ways (osm_way_id,
    geometry) and segments (segment_id, geometry), both EPSG:4326.

    Images matched to a POI segment, or to a segment outside
    file_segment_ids (the segments with their own GeoJSON file), are dropped.
    Returns image_id, MATCH_COLUMNS and the matched position on the segment
    (lng, lat), for the images kept, and the drop counts."""
    stats = MatchStats(images=len(images))
    out_cols = ["image_id", *MATCH_COLUMNS, "lng", "lat"]
    if images.empty:
        return pd.DataFrame(columns=out_cols), stats
    lon = images["lng_original"].to_numpy(float)
    lat = images["lat_original"].to_numpy(float)
    pts = shapely.points(lon, lat)
    ways = candidate_lines(pts, ways[["osm_way_id", "geometry"]], degree_margin(max_way_m, lat))
    segs = candidate_lines(pts, segs[["segment_id", "geometry"]],
                           degree_margin(max_way_m + max_segment_m, lat))

    parts = []
    epsgs = utm_epsg(lon, lat)
    for epsg in np.unique(epsgs):
        sel = epsgs == epsg
        zone_pts = gpd.GeoDataFrame({"image_id": images["image_id"].to_numpy()[sel]},
                                    geometry=pts[sel], crs="EPSG:4326").to_crs(epsg=int(epsg))
        on_way = nearest_line_match(zone_pts, ways.to_crs(epsg=int(epsg)), "osm_way_id", max_way_m)
        on_way = on_way.rename(columns={"dist_m": "osm_match_dist_m"})
        way_pts = gpd.GeoDataFrame(on_way[["image_id"]], crs=f"EPSG:{int(epsg)}",
                                   geometry=gpd.points_from_xy(on_way["x"], on_way["y"]))
        on_seg = nearest_line_match(way_pts, segs.to_crs(epsg=int(epsg)), "segment_id", max_segment_m)
        on_seg = on_seg.rename(columns={"dist_m": "segment_match_dist_m"})
        stats.no_way += int(sel.sum()) - len(on_way)
        stats.no_segment += len(on_way) - len(on_seg)
        if on_seg.empty:
            continue
        to_wgs84 = Transformer.from_crs(int(epsg), 4326, always_xy=True)
        x, y = to_wgs84.transform(on_seg["x"].to_numpy(), on_seg["y"].to_numpy())
        parts.append(on_way[["image_id", "osm_way_id", "osm_match_dist_m"]].merge(
            on_seg[["image_id", "segment_id", "segment_match_dist_m"]].assign(lng=x, lat=y),
            on="image_id", validate="one_to_one"))
    if not parts:
        return pd.DataFrame(columns=out_cols), stats
    matched = pd.concat(parts, ignore_index=True)
    on_poi = ~matched["segment_id"].isin(non_poi_ids).to_numpy()
    no_file = ~on_poi & ~matched["segment_id"].isin(file_segment_ids).to_numpy()
    stats.on_poi_segment = int(on_poi.sum())
    stats.on_segment_without_file = int(no_file.sum())
    return matched.loc[~(on_poi | no_file), out_cols].reset_index(drop=True), stats


def overwrite_matched(paths: dict[str, Path], matches: pd.DataFrame) -> pd.DataFrame:
    """③ each object file of ② keeps its matched rows, placed on the segment.
    Returns value, image_id, segment_id of every kept row."""
    kept = []
    for value, path in paths.items():
        pts = pd.DataFrame(gpd.read_parquet(path).drop(columns="geometry"))
        pts = pts.drop(columns=[c for c in MATCH_COLUMNS if c in pts.columns])
        rows = pts.merge(matches, on="image_id", how="inner", validate="one_to_one")
        _write_atomic(_points(rows, "lng", "lat")[[*POINT_COLUMNS, "geometry"]], path)
        kept.append(rows[["value", "image_id", "segment_id"]])
    return pd.concat(kept, ignore_index=True)


# --------------------------------------------------------------------------- #
# ④ ⑤  the segment table
# --------------------------------------------------------------------------- #


def segment_table(non_poi_ids: Sequence[str], file_segment_ids: set[str], kept: pd.DataFrame,
                  weights: pd.DataFrame) -> pd.DataFrame:
    """One row per non-POI segment: segment_id, has_detection_file,
    n_matched_images, a Bool column per scored object (weights.index) and the
    sum of the weights of the objects present per weight column."""
    values = list(weights.index)
    clash = set(values) & set(weights.columns) | set(values + list(weights.columns)) & {
        "segment_id", "has_detection_file", "n_matched_images"}
    if clash:
        raise ValueError(f"column name(s) {sorted(clash)} would appear twice.")
    ids = pd.Index(sorted(set(map(str, non_poi_ids))), name="segment_id")
    stray = set(kept["segment_id"]) - (set(ids) & set(file_segment_ids))
    if stray:
        raise ValueError(f"{len(stray)} matched segment(s) are not non-POI segments with a file.")
    present = (pd.crosstab(kept["segment_id"], kept["value"]) > 0) if not kept.empty else pd.DataFrame()
    present = present.reindex(index=ids, columns=values, fill_value=False).astype(bool)
    n_images = kept.groupby("segment_id")["image_id"].nunique().reindex(ids, fill_value=0)
    scores = present.astype(float).to_numpy() @ weights.loc[values].to_numpy()
    out = pd.DataFrame({
        "has_detection_file": ids.isin(list(file_segment_ids)),
        "n_matched_images": n_images.astype("int64").to_numpy(),
    }, index=ids)
    out = pd.concat([out, present, pd.DataFrame(scores, index=ids, columns=weights.columns)], axis=1)
    return out.reset_index()


# --------------------------------------------------------------------------- #
# the build step: is_divided and has_vru_barrier from the segment tables
# --------------------------------------------------------------------------- #

# The weight columns the two rules read. They are named because the rules are
# about these columns; a table without one of them is refused.
MEDIAN_COL = "physical_median_likelihood"
PROTECTION_COLS = (
    "pedestrian_protection_likelihood",
    "cyclist_protection_likelihood",
    "motorcyclist_protection_likelihood",
)
# Riders are separated from four-wheeled traffic where this column reaches
# the threshold.
MOTORCYCLE_SEPARATION_COL = "motorcyclist_protection_likelihood"
# A rule holds where the column's weight sum reaches this.
RULE_THRESHOLD = 1.0


def load_segment_tables(table_dir: str | Path = OUT_DIR) -> pd.DataFrame:
    """segment_id, mapillary_divided, has_vru_barrier and
    has_motorcycle_separation of every region with a segment table; a region
    without one is skipped with a message."""
    parts = []
    for region in REGIONS:
        path = default_out_path(region, table_dir)
        if not path.is_file():
            print(f"[object_separation] {region}: no {path}; left unchanged")
            continue
        table = pd.read_parquet(path)
        missing = [c for c in ("segment_id", MEDIAN_COL, *PROTECTION_COLS) if c not in table.columns]
        if missing:
            raise ValueError(f"{path} lacks column(s) {missing}.")
        parts.append(pd.DataFrame({
            "segment_id": table["segment_id"].astype(str),
            "mapillary_divided": table[MEDIAN_COL] >= RULE_THRESHOLD,
            "has_vru_barrier": (table[list(PROTECTION_COLS)] >= RULE_THRESHOLD).all(axis=1),
            "has_motorcycle_separation": table[MOTORCYCLE_SEPARATION_COL] >= RULE_THRESHOLD,
        }))
    if not parts:
        return pd.DataFrame(columns=["segment_id", "mapillary_divided", "has_vru_barrier",
                                     "has_motorcycle_separation"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("segment_id")


def apply_object_separation(target: pd.DataFrame, table_dir: str | Path = OUT_DIR) -> pd.DataFrame:
    """Set is_divided, has_vru_barrier and has_motorcycle_separation on the non-POI segments of target
    from the segment tables, and raise V_safe where the rules allow it.

    Only segments that are non-POI in this build are touched. On them V_safe
    is recomputed with safe_speed.add_v_safe (the is_vru cap included), then
    capped at JUNCTION_V_SAFE_CAP near a junction, and taken only where it
    exceeds the current value: v_safe, v_safe_basis and collision_type change
    on those rows alone, so a value set by a POI zone or the sandwich rule is
    never lowered. Adds mapillary_divided, has_vru_barrier and
    has_motorcycle_separation (False on every other row)."""
    target = target.copy()
    all_rules = load_segment_tables(table_dir)
    non_poi = set(select_non_poi_segments(target)["segment_id"].astype(str))
    rules = all_rules.loc[all_rules["segment_id"].isin(non_poi)].set_index("segment_id")
    seg_ids = target["segment_id"].astype(str)
    on = seg_ids.isin(rules.index)
    for col in ("mapillary_divided", "has_vru_barrier", "has_motorcycle_separation"):
        target[col] = False
        target.loc[on, col] = seg_ids[on].map(rules[col]).astype(bool).to_numpy()
    divided_before = target["is_divided"].fillna(False).astype(bool)
    target["is_divided"] = divided_before | target["mapillary_divided"]

    touched = (target["mapillary_divided"] | target["has_vru_barrier"]
               | target["has_motorcycle_separation"])
    raised = pd.Series(False, index=target.index)
    if touched.any():
        new = add_v_safe(target.loc[touched])
        if "near_junction" in new.columns:
            capped = new["near_junction"].fillna(False).astype(bool) & (new["v_safe"] > JUNCTION_V_SAFE_CAP)
            new.loc[capped, "v_safe"] = JUNCTION_V_SAFE_CAP
            new.loc[capped, "collision_type"] = "side_impact"
            new.loc[capped, "v_safe_basis"] = "side_impact:junction_buffer"
        up = new["v_safe"] > target.loc[touched, "v_safe"]
        rows = up[up].index
        target.loc[rows, ["v_safe", "v_safe_basis", "collision_type"]] = (
            new.loc[rows, ["v_safe", "v_safe_basis", "collision_type"]])
        raised.loc[rows] = True

    for country, rows in target.groupby("country").groups.items():
        print(f"[object_separation] {country}: "
              f"{int(on.loc[rows].sum()):,} segment(s) with a table row, "
              f"{int((target.loc[rows, 'mapillary_divided'] & ~divided_before.loc[rows]).sum()):,} newly divided, "
              f"{int(target.loc[rows, 'has_vru_barrier'].sum()):,} with has_vru_barrier, "
              f"{int(target.loc[rows, 'has_motorcycle_separation'].sum()):,} with has_motorcycle_separation, "
              f"{int(raised.loc[rows].sum()):,} V_safe raised")
    # Table rows of segments that are gone (the build cut the segments
    # differently) or became POI segments in this build; they raise nothing.
    print(f"[object_separation] {len(all_rules) - len(rules):,} table row(s) match no non-POI "
          f"segment of this build")
    return target


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def run(region: str, *, catalog_path: Path = CATALOG_PATH, detected_dir: Path = DETECTED_DIR,
        points_dir: Path = POINTS_DIR, out_dir: Path = OUT_DIR,
        segments_path: Path = SEGMENTS_PATH, ways_file: Path | None = None,
        max_way_m: float = MAX_WAY_M, max_segment_m: float = MAX_SEGMENT_M) -> pd.DataFrame:
    region = validate_region(region)
    ways_file = Path(ways_file) if ways_file else ways_path(region.lower())

    weights = load_scoring_catalog(catalog_path)
    stems = value_stems(list(weights.index))
    print(f"{region}: ① {len(weights)} scored object(s), weight columns {list(weights.columns)}")

    images, file_segment_ids = read_detections(detected_dir, region)
    pairs = object_rows(images, weights.index)
    paths = write_object_points(images, pairs, stems, Path(points_dir) / region)
    scored_images = images.loc[images["image_id"].isin(set(pairs["image_id"]))]
    print(f"  ② {len(images):,} image(s) in {len(file_segment_ids):,} file(s); "
          f"{len(scored_images):,} with a scored object; wrote {len(paths)} file(s) "
          f"to {Path(points_dir) / region}")

    segs = load_segments(segments_path, region)
    non_poi_ids = set(select_non_poi_segments(segs)["segment_id"].astype(str))
    ways = gpd.read_parquet(ways_file, columns=["osm_way_id", "geometry"])
    matches, stats = match_points(scored_images, ways, segs, non_poi_ids, file_segment_ids,
                                  max_way_m, max_segment_m)
    kept = overwrite_matched(paths, matches)
    print(f"  ③ dropped {stats.no_way:,} farther than {max_way_m:g} m from every OSM way, "
          f"{stats.no_segment:,} farther than {max_segment_m:g} m from every segment, "
          f"{stats.on_poi_segment:,} on POI segments, {stats.on_segment_without_file:,} on "
          f"segments without a file; kept {stats.kept:,} of {stats.images:,}")

    table = segment_table(sorted(non_poi_ids), file_segment_ids, kept, weights)
    out_path = default_out_path(region, out_dir)
    _write_atomic(table, out_path)
    print(f"  ④ ⑤ {len(table):,} non-POI segment(s), "
          f"{int((table['n_matched_images'] > 0).sum()):,} with a scored object; wrote {out_path}")
    return table


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("region", choices=REGIONS)
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    p.add_argument("--detected-dir", type=Path, default=DETECTED_DIR)
    p.add_argument("--points-dir", type=Path, default=POINTS_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--segments", type=Path, default=SEGMENTS_PATH)
    p.add_argument("--ways", type=Path, default=None,
                   help="default: data/processed/osm_ways_{country}.parquet")
    p.add_argument("--max-way-m", type=float, default=MAX_WAY_M)
    p.add_argument("--max-segment-m", type=float, default=MAX_SEGMENT_M)
    args = p.parse_args(argv)
    run(args.region, catalog_path=args.catalog, detected_dir=args.detected_dir,
        points_dir=args.points_dir, out_dir=args.out_dir, segments_path=args.segments,
        ways_file=args.ways, max_way_m=args.max_way_m, max_segment_m=args.max_segment_m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
