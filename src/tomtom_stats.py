"""Ingest and normalise the TomTom Route Analysis / Traffic Stats layer.

ADB provided this for two regions:

    thailand     combined_traffic_stats_Thailand.geojson   21,369 features
    maharashtra  ADBMaha_combined_20250808.geojson         27,559 features

It is an OPTIONAL enrichment layer: nothing in the pipeline requires it, and
with the file absent every consumer degrades to the Overture-only behaviour.
See tomtom_enrichment.py for the pipeline step.

★ What this module does ★
It reads the raw GeoJSON, flattens the single `segmentTimeResults` element,
pairs the two directions of one link into one row, drops values that cannot be
a posted speed limit, and writes a cache. Spatial matching against the ADB
segments is tomtom_data_integration.py's job.

★ Directions are paired, never mixed ★
Features sharing abs(segmentId) are one carriageway traversed both ways
(Thailand 4,740 pairs, Maharashtra 8,616; every pair Hausdorff distance 0.0 m).
The measurements of the two directions are different populations of trips, so
they are kept side by side in `*_along` / `*_against` columns and never
averaged or summed. `along` means travel in the direction the stored geometry
is drawn; `against` the opposite.

Which record travels which way was measured, not assumed:
  * In every pair the negative record's vertices are exactly the positive
    record's in reverse (Maharashtra 8,616 of 8,616, Thailand 4,740 of 4,740):
    each record is drawn in its own direction of travel.
  * Unpaired records on OSM oneway ways run along the way's direction 91-93%
    of the time whatever their sign (Maharashtra, 6,863 records), so an
    unpaired record is likewise taken to travel along its own geometry.
  * A pair with identical vertex order would carry no geometric evidence; it
    would be labelled "same" and TomTom's sign convention followed (negative =
    against). None occurs in either export.
The stored geometry is the positive record's when there is one.

★ Speed limits that are not a multiple of 5 ★
A posted limit is a multiple of 5 km/h (Maharashtra's export is dominated by
55 and 45). Anything else (18, 21, 36, 44, 51, 67, 74,
91 ...) looks like a distance-weighted average over the constituent links or a
map default, and is dropped to NA: a backfill that installs 67 km/h as a posted
limit would be worse than leaving the limit missing.

★ Fields deliberately dropped ★
- `travelTimeRatio` -- 1.0 on 99.9% of rows. It is measured against the Base
  Set (the first TimeSet), which was not written to this export, so every
  segment is being compared against itself.
- `timeSet` / `dateRange` -- constant 2 / 1 across all rows, and the export
  carries no definition of the period or time-of-day they name. The constants
  are recorded once in the meta JSON.
- `sourceAccount` -- an artifact of ADB concatenating many jobs.
- `sourceJob` (Thailand) -- reduced to `tomtom_single_direction_job`, the only
  part that carries analytical meaning. Maharashtra's `job_name` carries no
  -FT-/-TF- marker, so the flag is False there.
"""

import json
import sys
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

RAW_PATHS = {
    "thailand": "data/raw/combined_traffic_stats_Thailand.geojson",
    "maharashtra": "data/raw/ADBMaha_combined_20250808.geojson",
}
PROCESSED_DIR = "data/processed"
COUNTRY = "thailand"
DIRECTIONS = ("along", "against")


def cache_path(country: str) -> str:
    return f"{PROCESSED_DIR}/tomtom_segments_{country}.parquet"


def meta_path(country: str) -> str:
    return f"{PROCESSED_DIR}/tomtom_segments_{country}_meta.json"


# A posted speed limit is a multiple of 5 km/h. Anything else is an aggregate
# or a map default, not a sign that exists on the road.
ROUND_SPEED_LIMITS = frozenset(range(5, 130, 5))

# TomTom's speedPercentiles: 19 integers, ascending, 5th through 95th.
PERCENTILE_LEVELS = tuple(range(5, 100, 5))
P50_INDEX = PERCENTILE_LEVELS.index(50)  # 9
P85_INDEX = PERCENTILE_LEVELS.index(85)  # 16

# Every per-direction measurement. Each becomes `{name}_along` and
# `{name}_against` in the cache.
DIRECTIONAL_COLUMNS = [
    "tomtom_signed_segment_id",
    "tomtom_speed_limit",
    "tomtom_median_speed",
    "tomtom_mean_speed",
    "tomtom_harmonic_speed",
    "tomtom_sd_speed",
    "tomtom_travel_time_sd",
    "tomtom_mean_travel_time",
    "tomtom_median_travel_time",
    "tomtom_p85_speed",
    "tomtom_sample_size",
    "tomtom_speed_percentiles",
]

CACHE_COLUMNS = [
    "tomtom_segment_id",
    "tomtom_frc",
    "tomtom_street_name",
    "tomtom_length_m",
    *[f"{c}_{d}" for d in DIRECTIONS for c in DIRECTIONAL_COLUMNS],
    "tomtom_n_directions",
    "tomtom_pair_vertex_order",
    "tomtom_single_direction_job",
    "country",
    "geometry",
]


def is_available(path: str = RAW_PATHS[COUNTRY]) -> bool:
    """Whether the optional raw layer is on disk. Never raises."""
    import os

    return os.path.exists(path)


def normalise_speed_limit(s: pd.Series) -> pd.Series:
    """Multiples of 5 km/h pass through; everything else becomes NA.

    0 and negative values are not posted limits either, and NaN stays NaN --
    `isin` on a frozenset of ints handles all three without a special case.
    """
    numeric = pd.to_numeric(s, errors="coerce")
    keep = numeric.isin(list(ROUND_SPEED_LIMITS))
    return numeric.where(keep).astype("Int16")


def _unpack_time_results(raw: pd.Series) -> pd.DataFrame:
    """Flatten segmentTimeResults[0]. Every feature has exactly one element."""
    records = []
    for value in raw:
        if isinstance(value, str):
            value = json.loads(value)
        first = value[0] if value else {}
        percentiles = first.get("speedPercentiles")
        records.append(
            {
                "tomtom_median_speed": first.get("medianSpeed"),
                "tomtom_mean_speed": first.get("averageSpeed"),
                "tomtom_harmonic_speed": first.get("harmonicAverageSpeed"),
                "tomtom_sd_speed": first.get("standardDeviationSpeed"),
                "tomtom_travel_time_sd": first.get("travelTimeStandardDeviation"),
                "tomtom_mean_travel_time": first.get("averageTravelTime"),
                "tomtom_median_travel_time": first.get("medianTravelTime"),
                "tomtom_sample_size": first.get("sampleSize"),
                "tomtom_speed_percentiles": (
                    [float(p) for p in percentiles]
                    if isinstance(percentiles, (list, tuple))
                    and len(percentiles) == len(PERCENTILE_LEVELS)
                    else None
                ),
            }
        )
    return pd.DataFrame.from_records(records, index=raw.index)


def load_raw(path: str = RAW_PATHS[COUNTRY], country: str = COUNTRY) -> gpd.GeoDataFrame:
    """Read the raw GeoJSON into the canonical tomtom_* column names.

    One row per raw feature -- directions are NOT yet paired.
    """
    gdf = gpd.read_file(path)
    unpacked = _unpack_time_results(gdf["segmentTimeResults"])
    if "sourceJob" in gdf.columns:
        single_direction = (gdf["sourceJob"].astype("string")
                            .str.contains("-FT-|-TF-", regex=True, na=False))
    else:
        single_direction = pd.Series(False, index=gdf.index)

    out = gpd.GeoDataFrame(
        {
            # The sign encodes direction; the magnitude identifies the link.
            "tomtom_segment_id": gdf["segmentId"].astype("int64").abs(),
            "tomtom_signed_segment_id": gdf["segmentId"].astype("int64"),
            "tomtom_speed_limit": normalise_speed_limit(gdf["speedLimit"]),
            "tomtom_frc": pd.to_numeric(gdf["frc"], errors="coerce").astype("Int8"),
            "tomtom_street_name": gdf["streetName"].astype("string"),
            "tomtom_length_m": pd.to_numeric(gdf["distance"], errors="coerce"),
            "tomtom_single_direction_job": single_direction.to_numpy(dtype=bool),
            "country": country,
        },
        geometry=gdf.geometry,
        crs=gdf.crs,
    )
    for column in unpacked.columns:
        out[column] = unpacked[column]
    out["tomtom_sample_size"] = pd.to_numeric(
        out["tomtom_sample_size"], errors="coerce"
    ).astype("Int64")
    out["tomtom_p85_speed"] = [
        p[P85_INDEX] if isinstance(p, list) else np.nan
        for p in out["tomtom_speed_percentiles"]
    ]
    # Recorded here so build_cache can report it without re-reading the
    # GeoJSON purely to count what normalise_speed_limit threw away.
    out.attrs["n_non_round_speed_limits_dropped"] = int(
        pd.to_numeric(gdf["speedLimit"], errors="coerce").notna().sum()
        - out["tomtom_speed_limit"].notna().sum()
    )
    return out


def _same_vertices(a, b) -> bool:
    ca, cb = np.asarray(a.coords), np.asarray(b.coords)
    return ca.shape == cb.shape and np.allclose(ca, cb, rtol=0.0, atol=1e-9)


def _pair_group(group: pd.DataFrame) -> dict:
    """One output row from the 1 or 2 records sharing an abs(segmentId)."""
    positive = group[group["tomtom_signed_segment_id"] > 0]
    stored = positive.iloc[0] if len(positive) else group.iloc[0]
    row = {
        "tomtom_segment_id": int(stored["tomtom_segment_id"]),
        "tomtom_frc": stored["tomtom_frc"],
        "tomtom_street_name": stored["tomtom_street_name"],
        "tomtom_length_m": stored["tomtom_length_m"],
        "geometry": stored["geometry"],
        "country": stored["country"],
        "tomtom_n_directions": len(group),
        # A pair where either job was direction-restricted is still a pair;
        # `any` keeps the flag meaning "at least one side was requested alone".
        "tomtom_single_direction_job": bool(group["tomtom_single_direction_job"].any()),
    }
    for d in DIRECTIONS:
        for c in DIRECTIONAL_COLUMNS:
            row[f"{c}_{d}"] = None

    if len(group) == 1:
        row["tomtom_pair_vertex_order"] = "unpaired"
        assigned = {"along": stored}
    else:
        other = group[group.index != stored.name].iloc[0]
        if _same_vertices(stored["geometry"], other["geometry"]):
            # No geometric evidence; TomTom's sign convention decides.
            row["tomtom_pair_vertex_order"] = "same"
        else:
            row["tomtom_pair_vertex_order"] = "reversed"
        assigned = {"along": stored, "against": other}

    for d, record in assigned.items():
        for c in DIRECTIONAL_COLUMNS:
            row[f"{c}_{d}"] = record[c]
    return row


def pair_directions(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per abs(segmentId), each direction in its own columns."""
    rows = [_pair_group(group) for _, group in gdf.groupby("tomtom_segment_id", sort=True)]
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
    for d in DIRECTIONS:
        out[f"tomtom_signed_segment_id_{d}"] = out[f"tomtom_signed_segment_id_{d}"].astype("Int64")
        out[f"tomtom_speed_limit_{d}"] = out[f"tomtom_speed_limit_{d}"].astype("Int16")
        out[f"tomtom_sample_size_{d}"] = out[f"tomtom_sample_size_{d}"].astype("Int64")
        for c in DIRECTIONAL_COLUMNS:
            if c in ("tomtom_signed_segment_id", "tomtom_speed_limit",
                     "tomtom_sample_size", "tomtom_speed_percentiles"):
                continue
            out[f"{c}_{d}"] = pd.to_numeric(out[f"{c}_{d}"], errors="coerce").astype(float)
    out["tomtom_n_directions"] = out["tomtom_n_directions"].astype("Int8")
    out["tomtom_street_name"] = out["tomtom_street_name"].astype("string")
    return out[CACHE_COLUMNS]


def build_cache(country: str = COUNTRY) -> gpd.GeoDataFrame:
    """Read the raw layer, normalise it, and write the cache plus its meta JSON."""
    path = RAW_PATHS[country]
    raw = load_raw(path, country)
    paired = pair_directions(raw)
    paired.to_parquet(cache_path(country))

    order = paired["tomtom_pair_vertex_order"].value_counts().to_dict()
    meta = {
        "country": country,
        "source": "TomTom Route Analysis / Traffic Stats (provided by ADB)",
        "raw_path": path,
        # Constant across every row; the export carries no definition of what
        # period or time-of-day these name. Recorded here, not per row.
        "time_set": 2,
        "date_range": 1,
        "time_set_definition": "not provided in the export",
        "n_raw_features": len(raw),
        "n_segments": len(paired),
        "n_bidirectional_pairs": int((paired["tomtom_n_directions"] == 2).sum()),
        "pair_vertex_order": {k: int(v) for k, v in order.items()},
        "n_non_round_speed_limits_dropped": raw.attrs["n_non_round_speed_limits_dropped"],
        "total_length_km": round(float(paired["tomtom_length_m"].sum()) / 1000, 1),
        "percentile_levels": list(PERCENTILE_LEVELS),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path(country), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return paired


def load_tomtom(country: str = COUNTRY) -> gpd.GeoDataFrame | None:
    """The normalised layer, or None when it is unavailable.

    Cache first (when it carries the current columns), then the raw file, then
    None. None is what makes the whole layer optional: every caller treats it
    as "run the Overture-only path".
    """
    import os

    if country not in RAW_PATHS:
        return None
    if os.path.exists(cache_path(country)):
        cached = gpd.read_parquet(cache_path(country))
        if list(cached.columns) == CACHE_COLUMNS:
            return cached
    if is_available(RAW_PATHS[country]):
        return pair_directions(load_raw(RAW_PATHS[country], country))
    return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the TomTom cache for a country.")
    ap.add_argument("--country", choices=[*RAW_PATHS, "both"], default="both")
    args = ap.parse_args()
    for country in ([*RAW_PATHS] if args.country == "both" else [args.country]):
        paired = build_cache(country)
        print(f"tomtom_stats [{country}]: wrote {len(paired)} segments -> {cache_path(country)}")
        print(f"  bidirectional pairs: {(paired['tomtom_n_directions'] == 2).sum()}")
        print(f"  pair vertex order: {paired['tomtom_pair_vertex_order'].value_counts().to_dict()}")
        print(f"  total length: {paired['tomtom_length_m'].sum() / 1000:.1f} km")
