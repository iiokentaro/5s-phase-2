"""Split the priority rows into an urban and a rural list, ranked within each.

★ Why two lists ★
exposure_level.py builds urban and rural exposure on separate scales (a
rural segment's "high" exposure is measured differently from an urban one's).
A single ranked list would compare the two on one axis, and whether rural
segments then sit lower because they are safer or because the scales differ
cannot be tested without ground truth. So the two never share a list.
`priority_class`, and the score behind it, is the same for both lists.

★ Why the split is `road_environment`, and why a motorway is rural ★
`road_environment` is `rural` for land_use=='RURAL' and for every
road_class=='motorway', and `urban` for the rest. A motorway is a high-speed
through road without pedestrian frontage wherever it runs, so it goes on the
rural list, with the kind of intervention recommended there (signage and
self-enforcing road design, where the urban list calls for crosswalks and
traffic calming).

★ Why `rank_within_environment` is computed per country ★
Within each file, ranking Thailand and Maharashtra as one pool would let
Thailand's larger misalignment crowd Maharashtra out of the low rank numbers,
the "who gets acted on first" question this rank answers. The rank is
computed within each (country, road_environment) cell. Both countries share
a CSV, sorted by `rank_within_environment`, so their rank-1 rows interleave
at the top.

★ Why `rank_within_environment` leaves out the confidence axis ★
`safety_score.py`'s 0.50/0.35/0.15 weighting folds data confidence into the
score that decides `priority_class`, the gate for being on a list at all.
Ranking *within* a list by that score would move a rural safety-margin
segment (exposure raised because the signal is too thin to read directly,
exposure_level.apply_rural_safety_margin) a few places down for being less
measured, where it is no less dangerous. `rank_within_environment` uses the
misalignment and exposure terms alone, renormalised to sum to 1
(0.50/0.35 -> 0.5882/0.4118). Confidence is kept as a plain-language flag,
`confidence_note`.

★ What `exp_delta_fatal_abs` is, and what it is not ★
Neither country's dataset carries a crash or fatality count, so there is no
observed baseline for "N fewer deaths". `exp_delta_fatal_abs =
(exp_delta_fatal_percent_uniform / 100) * sample_size_avg` is a
traffic-volume-weighted index: it lets a small percentage reduction on a
heavily-travelled urban road and a large one on a lightly-travelled rural
road be compared on one scale. It is NOT a predicted body count, and must
never be presented as one -- `sample_size_avg` is a probe-sample count, a
traffic-volume proxy with no crash history in it. It is the mean over the
TomTom samples taken every 10 km along the segment and in each direction;
`sample_size_total`, their sum, also grows with the segment's length and the
number of directions sampled, and would weight a long segment for its length.
Segments already at/below v_safe carry a non-positive value and are not
reduction candidates. elvik_2019.py writes it, beside
exp_delta_fatal_percent_uniform.

★ The tie-break ★
MISALIGNMENT_CAP_KMH/EXPOSURE_POINTS saturate at the top of the rank_key,
so many segments legitimately tie at rank_within_environment==1. The lists
order tied rows by `exp_delta_fatal_percent_uniform` (Elvik 2019), largest
first. That orders rows within a rank; the stored rank_within_environment
value never depends on it.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

import elvik_2019  # noqa: E402
from safety_score import EXPOSURE_POINTS, MISALIGNMENT_CAP_KMH, WEIGHT_EXPOSURE, WEIGHT_MISALIGNMENT  # noqa: E402

ENVIRONMENTS = ["urban", "rural"]
ON_LIST_CLASSES = ["Top Priority", "Priority", "Watch"]

LIST_COLUMNS = [
    "segment_id", "overture_segment_id", "country", "road_class", "land_use",
    "road_environment",
    "rank_within_environment", "speed_limit", "speed_data_source",
    "median_speed", "v_safe", "misalignment",
    "exposure_level", "confidence_level", "confidence_note", "speedlimit_plausibility",
    elvik_2019.REPORTED_COLUMN, elvik_2019.ABS_COLUMN, "sample_size_avg",
    "priority_class", "review_track", "score_explanation", "street_image_link",
]


def road_environment_for(gdf: pd.DataFrame) -> pd.Series:
    """rural for RURAL land_use OR any motorway (regardless of land_use);
    urban for everything else (i.e. URBAN, non-motorway)."""
    is_rural = (gdf["road_class"] == "motorway") | (gdf["land_use"] == "RURAL")
    return pd.Series(np.where(is_rural, "rural", "urban"), index=gdf.index)


def add_priority_environment_rank(gdf: pd.DataFrame) -> pd.DataFrame:
    gdf = gdf.copy()
    has_flag_col = "data_quality_flag" in gdf.columns
    valid_mask = gdf["data_quality_flag"].isna() if has_flag_col else pd.Series(True, index=gdf.index)
    on_list = valid_mask & gdf["priority_class"].isin(ON_LIST_CLASSES)

    gdf["road_environment"] = pd.NA
    gdf.loc[valid_mask, "road_environment"] = road_environment_for(gdf.loc[valid_mask]).values

    misalignment_norm = gdf["misalignment_magnitude"].clip(upper=MISALIGNMENT_CAP_KMH) / MISALIGNMENT_CAP_KMH
    exposure_norm = gdf["exposure_level"].astype(str).map(EXPOSURE_POINTS).astype(float)
    weight_total = WEIGHT_MISALIGNMENT + WEIGHT_EXPOSURE
    rank_key = (WEIGHT_MISALIGNMENT * misalignment_norm + WEIGHT_EXPOSURE * exposure_norm) / weight_total

    gdf["rank_within_environment"] = pd.NA
    for env in ENVIRONMENTS:
        for country in gdf["country"].unique():
            mask = on_list & (gdf["road_environment"] == env) & (gdf["country"] == country)
            if not mask.any():
                continue
            ranks = rank_key.loc[mask].rank(ascending=False, method="min")
            gdf.loc[mask, "rank_within_environment"] = ranks.values

    gdf["confidence_note"] = pd.NA
    gdf.loc[on_list & (gdf["confidence_level"] == "low"), "confidence_note"] = "Exposure uncertain -- field verification recommended"

    gdf["rank_within_environment"] = pd.to_numeric(gdf["rank_within_environment"])
    return gdf


def write_priority_environment_lists(gdf: pd.DataFrame, out_dir: str = "outputs") -> tuple[str, str]:
    # Ties within a rank are ordered by the Elvik (2019) figure (see the module
    # docstring). The quick path enters at a stored parquet that may predate
    # those columns, hence the ensure.
    gdf = elvik_2019.ensure_exponential_columns(gdf)
    sort_cols = ["rank_within_environment", elvik_2019.REPORTED_COLUMN]
    sort_order = [True, False]
    on_list = gdf["rank_within_environment"].notna()
    urban = gdf[on_list & (gdf["road_environment"] == "urban")].sort_values(
        sort_cols, ascending=sort_order
    )
    rural = gdf[on_list & (gdf["road_environment"] == "rural")].sort_values(
        sort_cols, ascending=sort_order
    )

    urban_path = f"{out_dir}/priority_urban.csv"
    rural_path = f"{out_dir}/priority_rural.csv"
    urban[LIST_COLUMNS].to_csv(urban_path, index=False)
    rural[LIST_COLUMNS].to_csv(rural_path, index=False)
    return urban_path, rural_path


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, _ = build()  # build() already runs add_priority_environment_rank as part of the pipeline
    valid = target[target["data_quality_flag"].isna()]
    on_list = valid["rank_within_environment"].notna()
    pri = valid[on_list]

    print(f"priority list (both environments combined): n={len(pri)}")
    print(pri.groupby("road_environment").size())

    print("\n=== confirm: confidence does not move rank (low-confidence segments present at rank 1) ===")
    for env in ENVIRONMENTS:
        sub = pri[pri["road_environment"] == env]
        rank1 = sub[sub["rank_within_environment"] == 1]
        low_conf_at_rank1 = (rank1["confidence_level"] == "low").sum()
        print(f"{env}: {low_conf_at_rank1}/{len(rank1)} segments tied at rank 1 are confidence_level=='low' "
              f"(non-zero is expected -- confidence must not auto-demote rank)")

    print("\n=== exp_delta_fatal_abs: cross-environment comparability check ===")
    print(pri.groupby("road_environment")[elvik_2019.ABS_COLUMN].describe())

    urban_path, rural_path = write_priority_environment_lists(target)

    print(f"\n=== top 5 of each environment list (as written to CSV, ties broken by {elvik_2019.REPORTED_COLUMN}) ===")
    cols = ["segment_id", "country", "road_class", "land_use", "rank_within_environment",
            elvik_2019.REPORTED_COLUMN, elvik_2019.ABS_COLUMN, "confidence_note"]
    for path in [urban_path, rural_path]:
        print(f"\n-- {path} --")
        print(pd.read_csv(path)[cols].head(5).to_string(index=False))
    print(f"\nsaved {urban_path}")
    print(f"saved {rural_path}")
