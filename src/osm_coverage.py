"""Pedestrian-infrastructure coverage in sample urban/rural areas, plus a
feasibility test of buffering OSM crossings against road segments as a
VRU-exposure signal.

A zero or near-zero count in a rural sample can mean the sidewalks are
unmapped. That is why rural exposure is built from population and POI
density.
"""

import sys
import warnings

import geopandas as gpd
import osmnx as ox

sys.path.insert(0, "src")
from schema import load_thailand

warnings.filterwarnings("ignore", category=UserWarning)

PEDESTRIAN_TAGS = {
    "highway": ["footway", "crossing", "pedestrian", "path", "steps"],
    "footway": "crossing",
    "sidewalk": True,
}

# One urban + one rural sample per country (bbox = left, bottom, right, top).
AREAS = {
    "thailand_urban_bangkok": (100.4, 13.6, 100.7, 13.9),
    "thailand_rural": (101.6, 14.9, 101.8, 15.1),
    "maharashtra_urban_pune": (73.7, 18.4, 73.95, 18.65),
    "maharashtra_rural": (76.9, 18.2, 77.1, 18.4),
}


def pedestrian_feature_counts() -> dict[str, int]:
    counts = {}
    for name, bbox in AREAS.items():
        try:
            gdf = ox.features_from_bbox(bbox, PEDESTRIAN_TAGS)
            counts[name] = len(gdf)
            print(f"{name}: {len(gdf)} pedestrian-tagged features")
            if "highway" in gdf.columns:
                print(gdf["highway"].value_counts(dropna=False).to_dict())
        except Exception as e:  # osmnx raises InsufficientResponseError on zero matches
            counts[name] = 0
            print(f"{name}: 0 features ({type(e).__name__})")
    return counts


def crossing_buffer_overlap_bangkok(buffer_m: float = 25) -> None:
    """For Bangkok, what fraction of road segments have an OSM crossing
    within `buffer_m` metres? Tests feasibility of a spatial-join exposure
    signal where OSM pedestrian tagging is dense enough to use."""
    bbox = AREAS["thailand_urban_bangkok"]
    crossings = ox.features_from_bbox(bbox, {"highway": "crossing"})
    crossings = crossings[crossings.geometry.type == "Point"]

    th = load_thailand()
    segments = th.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    segments_utm = segments.set_geometry("geometry").to_crs(epsg=32647)
    crossings_utm = crossings.set_geometry("geometry").to_crs(epsg=32647)

    buffered = segments_utm.copy()
    buffered["geometry"] = buffered.geometry.buffer(buffer_m)

    joined = gpd.sjoin(crossings_utm, buffered[["segment_id", "geometry"]], predicate="within")
    n_segments_matched = joined["segment_id"].nunique()
    print(
        f"\nBangkok: {n_segments_matched} / {len(segments)} road segments "
        f"({n_segments_matched / len(segments):.1%}) have an OSM crossing within {buffer_m}m"
    )
    print(f"{joined.index.nunique()} / {len(crossings_utm)} crossings matched to >=1 segment")


if __name__ == "__main__":
    pedestrian_feature_counts()
    crossing_buffer_overlap_bangkok()
