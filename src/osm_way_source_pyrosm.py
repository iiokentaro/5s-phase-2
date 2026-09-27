"""A second WaySource, kept to be compared with the first.

Its only caller is tests/test_osm_way_source_equivalence.py. Two independent
readers disagreeing is what exposed the behaviour of pyrosm's
get_data_by_custom_criteria (only two-node ways are returned), and the test
keeps running that comparison.

`get_network` is a different pyrosm code path from that one and measures
correct: on one Bangkok extract it returns all 62,246 highway ways with total
length identical to osmium to 0.000 m. Its geometries arrive as MultiLineStrings
of node-pair pieces, so they need line_merge to become the way again. 0.41% stay
MultiLineString after merging -- ways that self-intersect or double back, which
cannot be expressed as one LineString. Those rows are dropped here and the count
is reported, because silently emitting a MultiLineString would break the
WAY_COLUMNS contract that every consumer relies on.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from pyrosm import OSM

from osm_way_source import PROMOTED_TAGS

warnings.filterwarnings("ignore", category=UserWarning)


class PyrosmWaySource:
    name = "pyrosm-get_network"

    def extract(self, pbf_path: str | Path) -> gpd.GeoDataFrame:
        raw = OSM(str(pbf_path)).get_network(network_type="all")
        merged = shapely.line_merge(raw.geometry.values)

        keep = shapely.get_type_id(merged) == 1  # LineString
        self.dropped_multilinestrings = int((~keep).sum())
        raw = raw.loc[keep].reset_index(drop=True)
        merged = merged[keep]

        frame = pd.DataFrame({"osm_way_id": raw["id"].astype("int64")})
        for tag in PROMOTED_TAGS:
            # get_network promotes a different set of keys than we ask for, so
            # anything it did not promote is recovered from its catch-all.
            frame[tag] = (raw[tag].astype("object") if tag in raw.columns
                          else _from_tags(raw, tag))
        frame["tags"] = raw["tags"].astype("object") if "tags" in raw.columns else None
        return gpd.GeoDataFrame(frame, geometry=merged, crs="EPSG:4326")


def _from_tags(raw: gpd.GeoDataFrame, key: str) -> pd.Series:
    if "tags" not in raw.columns:
        return pd.Series([None] * len(raw), index=raw.index, dtype="object")

    def pick(blob):
        if isinstance(blob, str):
            try:
                blob = json.loads(blob)
            except ValueError:
                return None
        return blob.get(key) if isinstance(blob, dict) else None

    return raw["tags"].map(pick)
