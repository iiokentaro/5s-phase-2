"""Common schema for Thailand and Maharashtra road-segment data.

The GeoJSON files are the canonical source: they carry the full LineString
geometry per segment, where the CSVs carry only a textual StreetImageLink
start/end-point string. Row order and OBJECTID are identical between the
GeoJSON and CSV exports for both countries.

`SpeedLimit` and `LandUse` are estimates, not ground truth.
"""

import geopandas as gpd
import pandas as pd

RAW_DIR = "data/raw"

# Renames to a shared column set. Only columns that exist for both countries
# (or are country-specific but useful, e.g. urban_pc) are kept.
THAILAND_RENAME = {
    "OBJECTID": "segment_id",
    "RoadClass": "road_class",
    "LandUse": "land_use",
    "SpeedLimit": "speed_limit",
    "MedianSpeed": "median_speed",
    "F85thPercentileSpeed": "f85_speed",
    "SampleSize_avg": "sample_size_avg",
    "SampleSizeTotal": "sample_size_total",
    "Shape_Length": "shape_length",
    "StreetImageLink": "street_image_link",
    "AnalysisStatus": "analysis_status",
}

MAHARASHTRA_RENAME = {
    "OBJECTID": "segment_id",
    "RoadClass": "road_class",
    "LandUse": "land_use",
    "UrbanPC": "urban_pc",
    "SpeedLimit": "speed_limit",
    "MedianSpeed": "median_speed",
    "F85thPercentileSpeed": "f85_speed",
    "SampleSize_avg": "sample_size_avg",
    "Sample_Size_Total": "sample_size_total",
    "Shape_Length": "shape_length",
    "StreetImageLink": "street_image_link",
    "AnalysisStatus": "analysis_status",
    "ExcludeFromSpeedSPI": "exclude_from_speedspi",
}

COMMON_COLUMNS = [
    "segment_id",
    "road_class",
    "land_use",
    "urban_pc",
    "speed_limit",
    "median_speed",
    "f85_speed",
    "sample_size_avg",
    "sample_size_total",
    "shape_length",
    "street_image_link",
    "analysis_status",
    "exclude_from_speedspi",
    "country",
    "geometry",
]


def load_thailand(path: str = f"{RAW_DIR}/ADB_Innovation_Thailand.geojson") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf.rename(columns=THAILAND_RENAME)
    gdf["urban_pc"] = pd.NA
    gdf["country"] = "thailand"
    gdf["speed_limit"] = pd.to_numeric(gdf["speed_limit"])
    # Thailand's export has no ExcludeFromSpeedSPI column; none of its Valid
    # rows are missing speed_limit, so 0 is a safe default.
    gdf["exclude_from_speedspi"] = 0
    return gdf[COMMON_COLUMNS]


def load_maharashtra(path: str = f"{RAW_DIR}/ADB_Innovation_Maharashtra.geojson") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf.rename(columns=MAHARASHTRA_RENAME)
    gdf["country"] = "maharashtra"
    # SpeedLimit is exported as a mix of JSON strings ("55") and null in this
    # GeoJSON (every other numeric column is a proper float) -- coerce it.
    gdf["speed_limit"] = pd.to_numeric(gdf["speed_limit"])
    return gdf[COMMON_COLUMNS]


def load_combined() -> gpd.GeoDataFrame:
    th = load_thailand()
    ind = load_maharashtra()
    combined = gpd.GeoDataFrame(pd.concat([th, ind], ignore_index=True), crs=th.crs)
    return combined


def valid_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rows with usable probe-derived speed data (per AnalysisStatus == 'Valid')."""
    return gdf[gdf["analysis_status"] == "Valid"]


def has_invalid_zero_speeds(gdf: gpd.GeoDataFrame) -> pd.Series:
    """speed_limit, median_speed, and f85_speed all exactly 0 together is a
    placeholder/export artifact, not a real "posted limit of zero" road:
    a road with hundreds of thousands of probe samples (sample_size_total)
    and a genuinely-zero observed speed doesn't exist. Found in 410/15,121
    Thailand-only segments, all AnalysisStatus=='Valid' and
    exclude_from_speedspi==0 -- i.e. they'd otherwise silently pass through
    into the V_safe-SpeedLimit misalignment pipeline, where V_safe - 0 = V_safe
    would read as a maximal false-positive gap."""
    return (gdf["speed_limit"] == 0) & (gdf["median_speed"] == 0) & (gdf["f85_speed"] == 0)


def load_target() -> gpd.GeoDataFrame:
    """The analysis population: 15,121 segments (Thailand 11,544 +
    Maharashtra 3,577) with AnalysisStatus=='Valid' and a real SpeedLimit.

    Valid alone is 15,554; Maharashtra's 433 Valid rows with
    ExcludeFromSpeedSPI==1 have no SpeedLimit at all, making the
    V_safe-SpeedLimit misalignment impossible to compute -- these are
    excluded here, at the schema level, since there is no value to retain.

    Thailand's 410 rows with speed_limit/median_speed/f85_speed all exactly
    0 (see has_invalid_zero_speeds) are a different kind of problem: the row
    itself is fine, only the speed fields are placeholder/export artifacts.
    They are kept and marked data_quality_flag='invalid_speed' so the
    deliverable can report their count; the misalignment and score steps
    filter on this flag."""
    gdf = valid_only(load_combined())
    gdf = gdf[gdf["exclude_from_speedspi"] != 1].copy()
    gdf["data_quality_flag"] = pd.NA
    gdf.loc[has_invalid_zero_speeds(gdf), "data_quality_flag"] = "invalid_speed"
    return gdf


if __name__ == "__main__":
    combined = load_combined()
    print(combined.head())
    print("\nshape:", combined.shape)
    print("\nby country:\n", combined["country"].value_counts())
    print("\nvalid rows by country:\n", valid_only(combined)["country"].value_counts())
    print("\ntarget rows by country:\n", load_target()["country"].value_counts())
