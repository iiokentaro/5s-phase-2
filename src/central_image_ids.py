"""Pick one Mapillary image per non-POI segment, for the Detection API requests.

The segments whose V_safe no POI zone set are the ones the Detection API is
asked about. One image per segment keeps the request count down:

  ① select_non_poi_segments   the segments of the region whose POI reason
                               columns (poi_speed_zones.REASON_COLS) are all
                               False.
  ② match_images              every image of filtered_image_ids/{region}/ goes
                               to its nearest segment of the region, POI or
                               not, within MAX_MATCH_M metres; its matched
                               position is the nearest point on that segment.
                               Images matched to a segment outside ① are
                               dropped, so an image nearer to a parallel POI
                               segment never stands for a segment of ①.
  ③                           segments of ① without a matched image are
                               dropped; the count and total length of the rest
                               are printed.
  ④ central_image_per_segment per segment, among the images matched to it, the
                               one whose matched position is nearest to the
                               point halfway along the segment is its
                               central_mapillary_image_id. The image nearest
                               to the halfway point may be matched to another
                               segment and is then not a candidate.
  ⑤                           segment_id, central_mapillary_image_id, where
                               the image was taken (lat_original, lng_original)
                               and its matched position (lat_mapmatched,
                               lng_mapmatched), all EPSG:4326, are written to
                               data/mapillary/central_image_ids/{region}_central_image_ids.csv.

Distances are measured in the UTM zone of each segment (geometry.py), since
both regions straddle two zones.

Usage
-----
    python src/central_image_ids.py Maharashtra
    python src/central_image_ids.py Thailand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry import add_representative_point, add_utm_epsg, to_utm_by_zone  # noqa: E402
from poi_speed_zones import REASON_COLS  # noqa: E402

REGIONS = ("Maharashtra", "Thailand")
SEGMENTS_PATH = Path("data/processed/segments_v_safe.parquet")
IMAGE_DIR = Path("data/mapillary/filtered_image_ids")
OUT_DIR = Path("data/mapillary/central_image_ids")
# An image farther than this from every segment is not matched.
MAX_MATCH_M = 8.0
# Margin (degrees) around a zone's segments when images are pre-filtered in
# EPSG:4326; 0.001 degree is over 100 m at these latitudes, well above MAX_MATCH_M.
_BBOX_MARGIN_DEG = 0.001


def validate_region(region: str) -> str:
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}.")
    return region


def default_out_path(region: str) -> Path:
    return OUT_DIR / f"{region}_central_image_ids.csv"


# --------------------------------------------------------------------------- #
# ①  non-POI segments
# --------------------------------------------------------------------------- #


def select_non_poi_segments(segs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rows whose POI reason columns are all False (a missing column counts as False)."""
    cols = [c for c in REASON_COLS.values() if c in segs.columns]
    poi = segs[cols].fillna(False).astype(bool).any(axis=1)
    return segs.loc[~poi]


def load_segments(path: Path, region: str) -> gpd.GeoDataFrame:
    cols = ["segment_id", "country", *REASON_COLS.values(), "geometry"]
    segs = gpd.read_parquet(path, columns=cols)
    return segs.loc[segs["country"] == region.lower()].reset_index(drop=True)


def load_images(image_dir: Path, region: str) -> pd.DataFrame:
    """id, longitude, latitude of every part file of the region."""
    files = sorted((image_dir / region).glob(f"{region}_image_ids_*_part*.parquet"))
    if not files:
        raise FileNotFoundError(f"no image parts in {image_dir / region}")
    tables = [pq.read_table(f, columns=["id", "longitude", "latitude"]) for f in files]
    return pd.concat([t.to_pandas() for t in tables], ignore_index=True)


# --------------------------------------------------------------------------- #
# ② ④  matching and the central image
# --------------------------------------------------------------------------- #


def match_images(images: gpd.GeoDataFrame, segs: gpd.GeoDataFrame,
                 max_distance: float = MAX_MATCH_M) -> pd.DataFrame:
    """Each image to its nearest segment within max_distance (both in the same
    metric CRS). Returns id, segment_id, match_dist_m, the matched position
    (snapped_x, snapped_y, in the CRS of segs) and mid_dist_m, the distance
    from the matched position to the segment's halfway point."""
    joined = gpd.sjoin_nearest(
        images[["id", "geometry"]], segs[["segment_id", "geometry"]],
        how="inner", max_distance=max_distance, distance_col="match_dist_m",
    )
    # An image equally near two segments is joined to both; keep one.
    joined = joined.sort_values(["id", "match_dist_m", "segment_id"])
    joined = joined.drop_duplicates("id")
    lines = segs.geometry.values[joined["index_right"].to_numpy()]
    pts = joined.geometry.values
    matched = shapely.line_interpolate_point(lines, shapely.line_locate_point(lines, pts))
    mid = shapely.line_interpolate_point(lines, 0.5, normalized=True)
    return pd.DataFrame({
        "id": joined["id"].to_numpy(),
        "segment_id": joined["segment_id"].to_numpy(),
        "match_dist_m": joined["match_dist_m"].to_numpy(),
        "snapped_x": shapely.get_x(matched),
        "snapped_y": shapely.get_y(matched),
        "mid_dist_m": shapely.distance(matched, mid),
    })


def central_image_per_segment(matches: pd.DataFrame) -> pd.DataFrame:
    """segment_id, central_mapillary_image_id and its matched position
    (lat_mapmatched, lng_mapmatched in EPSG:4326), one row per matched segment. matches needs a utm_epsg
    column naming the CRS of snapped_x and snapped_y."""
    best = matches.sort_values(["segment_id", "mid_dist_m", "id"]).drop_duplicates("segment_id")
    best = best.reset_index(drop=True)
    lat = pd.Series(float("nan"), index=best.index)
    lng = pd.Series(float("nan"), index=best.index)
    for epsg, rows in best.groupby("utm_epsg"):
        to_wgs84 = Transformer.from_crs(int(epsg), 4326, always_xy=True)
        x, y = to_wgs84.transform(rows["snapped_x"].to_numpy(), rows["snapped_y"].to_numpy())
        lng.loc[rows.index], lat.loc[rows.index] = x, y
    return pd.DataFrame({
        "segment_id": best["segment_id"].astype(str).to_numpy(),
        "central_mapillary_image_id": best["id"].astype(str).to_numpy(),
        "lat_mapmatched": lat.to_numpy(),
        "lng_mapmatched": lng.to_numpy(),
    })


def add_original_position(central: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Insert lat_original and lng_original, where each central image was
    taken (the latitude and longitude of load_images), before the matched
    position."""
    original = (images[["id", "latitude", "longitude"]].drop_duplicates("id")
                .rename(columns={"id": "central_mapillary_image_id",
                                 "latitude": "lat_original", "longitude": "lng_original"}))
    out = central.merge(original, on="central_mapillary_image_id", how="left", validate="one_to_one")
    return out[["segment_id", "central_mapillary_image_id", "lat_original", "lng_original",
                "lat_mapmatched", "lng_mapmatched"]]


def match_by_zone(segs: gpd.GeoDataFrame, images: pd.DataFrame,
                  max_distance: float = MAX_MATCH_M) -> tuple[pd.DataFrame, pd.Series]:
    """match_images in each UTM zone of segs (EPSG:4326). Returns the matches
    and the length (m) of every segment."""
    zones = to_utm_by_zone(add_utm_epsg(add_representative_point(segs)))
    lon = images["longitude"].to_numpy()
    lat = images["latitude"].to_numpy()
    parts, lengths = [], []
    for epsg, zone_segs in zones.items():
        zone_segs = zone_segs[["segment_id", "geometry"]].reset_index(drop=True)
        lengths.append(pd.Series(zone_segs.length.to_numpy(), index=zone_segs["segment_id"]))
        x0, y0, x1, y1 = segs.loc[segs["segment_id"].isin(zone_segs["segment_id"])].total_bounds
        m = _BBOX_MARGIN_DEG
        near = (lon >= x0 - m) & (lon <= x1 + m) & (lat >= y0 - m) & (lat <= y1 + m)
        zone_imgs = gpd.GeoDataFrame(
            {"id": images["id"].to_numpy()[near]},
            geometry=gpd.points_from_xy(lon[near], lat[near]), crs="EPSG:4326",
        ).to_crs(epsg=epsg)
        parts.append(match_images(zone_imgs, zone_segs, max_distance).assign(utm_epsg=epsg))
    matches = pd.concat(parts, ignore_index=True)
    # An image near a zone boundary can be matched in both zones; keep the nearer.
    matches = matches.sort_values(["id", "match_dist_m", "segment_id"]).drop_duplicates("id")
    return matches, pd.concat(lengths)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def central_images(segs: gpd.GeoDataFrame, images: pd.DataFrame,
                   max_distance: float = MAX_MATCH_M) -> tuple[pd.DataFrame, dict]:
    """② ③ ④ for the segments of one region (EPSG:4326, with the POI reason
    columns) and its images (id, longitude, latitude).

    Every image is matched against all segments, POI or not, so it goes to
    its truly nearest segment; only then are the matches to POI segments
    dropped. Returns the central images (add_original_position columns) and
    the counts printed by run()."""
    non_poi_ids = set(select_non_poi_segments(segs)["segment_id"])
    matches, lengths = match_by_zone(segs, images, max_distance)
    on_non_poi = matches["segment_id"].isin(non_poi_ids).to_numpy()
    central = add_original_position(central_image_per_segment(matches.loc[on_non_poi]), images)
    non_poi_lengths = lengths.loc[lengths.index.isin(non_poi_ids)]
    stats = {
        "non_poi_segments": len(non_poi_ids),
        "non_poi_km": non_poi_lengths.sum() / 1000,
        "images_matched": len(matches),
        "images_on_poi_segments": int((~on_non_poi).sum()),
        "central_km": lengths.loc[central["segment_id"]].sum() / 1000,
    }
    return central, stats


def run(region: str, segments_path: Path = SEGMENTS_PATH, image_dir: Path = IMAGE_DIR,
        out_path: Path | None = None, max_distance: float = MAX_MATCH_M) -> pd.DataFrame:
    region = validate_region(region)
    out_path = Path(out_path) if out_path else default_out_path(region)

    segs = load_segments(segments_path, region)
    images = load_images(image_dir, region)
    central, stats = central_images(segs, images, max_distance)
    print(f"{region}: {len(segs):,} segments, {stats['non_poi_segments']:,} without a POI zone; "
          f"{len(images):,} images")
    print(f"  ① non-POI segments:           {stats['non_poi_segments']:,}, "
          f"{stats['non_poi_km']:,.1f} km")
    print(f"  ② images matched within {max_distance:g} m: {stats['images_matched']:,} "
          f"({stats['images_on_poi_segments']:,} to POI segments, dropped)")
    print(f"  ③ segments with an image:     {len(central):,}, {stats['central_km']:,.1f} km")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    central.to_csv(out_path, index=False)
    print(f"  ⑤ wrote {out_path}")
    return central


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("region", choices=REGIONS)
    p.add_argument("--segments", type=Path, default=SEGMENTS_PATH)
    p.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="default: data/mapillary/central_image_ids/{region}_central_image_ids.csv")
    p.add_argument("--max-distance", type=float, default=MAX_MATCH_M)
    args = p.parse_args(argv)
    run(args.region, args.segments, args.image_dir, args.out, args.max_distance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
