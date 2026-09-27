"""Split the priority list into two tracks by `speedlimit_plausibility`.

★ SpeedLimit is not ground truth (FAQ) ★
A segment can land in the priority list for two structurally different reasons:
- the posted SpeedLimit really is too high for who's exposed here
  (`speedlimit_plausibility == "high"`) -- a genuine policy review case.
- the *recorded* SpeedLimit itself looks unreliable
  (`speedlimit_plausibility == "low"`, an IQR outlier within its
  (country, road_class, land_use) peer group or >30km/h off observed
  speed -- see speedlimit_plausibility.py) -- the apparent misalignment may
  just be a bad record, not a bad speed limit.

These must not be merged into one list: a true-misalignment recommendation
diluted with sign-recording errors would undermine the credibility of the
whole priority list. `review_track` keeps them as two disjoint categories:
  - "Review Needed" (review the speed limit) -- plausible record, large gap.
  - "Field Verification Needed" (verify on the ground first) -- record itself is suspect.
Only rows in the priority list (Top Priority, Priority or Watch) get a
track; every other row, the `data_quality_flag` rows included, is `pd.NA`.

`street_image_link` (despite its name, a "lon1,lat1,lon2,lat2" endpoint
string, not an image URL -- see schema.py) is carried through unchanged for
the "Field Verification Needed" track so each segment can be located for field/imagery
verification. Caveat that must travel with it: street-level imagery can
usually confirm LandUse/environment, but there's no guarantee the posted
speed-limit sign itself is visible or legible in frame -- it corroborates
context, not the SpeedLimit value.
"""

import sys

import pandas as pd

sys.path.insert(0, "src")

import elvik_2019  # noqa: E402

REVIEW_NEEDED = "Review Needed"
FIELD_CHECK_NEEDED = "Field Verification Needed"

LIST_COLUMNS = [
    "segment_id", "overture_segment_id", "country", "road_class", "land_use",
    "speed_limit", "speed_data_source",
    "median_speed", "v_safe", "misalignment", "f85_speed", "operating_gap",
    "exposure_level", "confidence_level", "speedlimit_plausibility",
    "road_environment", elvik_2019.REPORTED_COLUMN,
    "safety_score", "priority_class", "score_explanation", "street_image_link",
]


def add_review_track(gdf):
    gdf = gdf.copy()
    on_priority_list = gdf["priority_class"].isin(["Top Priority", "Priority", "Watch"])

    gdf["review_track"] = pd.NA
    gdf.loc[on_priority_list & (gdf["speedlimit_plausibility"] == "high"), "review_track"] = REVIEW_NEEDED
    gdf.loc[on_priority_list & (gdf["speedlimit_plausibility"] == "low"), "review_track"] = FIELD_CHECK_NEEDED

    return gdf


def write_lists(gdf, out_dir="outputs"):
    # The quick path enters at a stored parquet that may predate the reported
    # Elvik (2019) column.
    gdf = elvik_2019.ensure_exponential_columns(gdf)
    review = gdf[gdf["review_track"] == REVIEW_NEEDED].sort_values("safety_score", ascending=False)
    field_check = gdf[gdf["review_track"] == FIELD_CHECK_NEEDED].sort_values("safety_score", ascending=False)

    review_path = f"{out_dir}/priority_review_needed.csv"
    field_check_path = f"{out_dir}/priority_field_check.csv"
    review[LIST_COLUMNS].to_csv(review_path, index=False)
    field_check[LIST_COLUMNS].to_csv(field_check_path, index=False)
    return review_path, field_check_path


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, _ = build()  # build() already runs add_review_track as part of the pipeline

    review = target[target["review_track"] == REVIEW_NEEDED]
    field_check = target[target["review_track"] == FIELD_CHECK_NEEDED]

    print(f"Review Needed (plausibility=high, on priority list): {len(review)}")
    print(f"Field Verification Needed (plausibility=low,  on priority list): {len(field_check)}")
    print(f"ratio: {len(review)} : {len(field_check)} "
          f"({len(review) / max(1, len(field_check)):.1f} : 1)")
    print()
    print("by priority_class:")
    print(target[target["review_track"].notna()].groupby("priority_class")["review_track"].value_counts())

    review_path, field_check_path = write_lists(target)
    print(f"\nsaved {review_path} ({len(review)} rows)")
    print(f"saved {field_check_path} ({len(field_check)} rows)")

    print("\n=== spot check: top 5 of Review Needed ===")
    top5 = review.nlargest(5, "safety_score")
    print(top5[["segment_id", "country", "road_class", "land_use", "speed_limit", "v_safe",
                "misalignment", "exposure_level"]].to_string(index=False))
