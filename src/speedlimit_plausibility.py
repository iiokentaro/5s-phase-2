"""speedlimit_plausibility -- how much to trust the recorded SpeedLimit value
itself, NOT whether it's the right speed to post.

There's no external ground-truth speed-limit dataset to check SpeedLimit
against (FAQ: impossible), so this only uses internal consistency:

1. Road-class/land-use consistency: is this SpeedLimit a statistical
   outlier (IQR rule) within its own (country, road_class, land_use) peer
   group? An urban-primary segment posted at 90 when its peers cluster
   around 50 looks like a recording error, not a real policy choice.
2. Extreme divergence from observed speed: |F85 - SpeedLimit| > 30 km/h.
   This uses observed speed ONLY to flag the *SpeedLimit's* credibility --
   it must never feed back into V_safe (see safe_speed.py).
"""

import sys
import warnings

import pandas as pd

sys.path.insert(0, "src")

warnings.filterwarnings("ignore", category=UserWarning)

DIVERGENCE_THRESHOLD_KMH = 30  # ~top decile of |F85 - SpeedLimit|


def _iqr_outlier_mask(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=s.index)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < lo) | (s > hi)


def add_speedlimit_plausibility(gdf):
    """Rows flagged data_quality_flag=='invalid_speed' (speed_limit/median/f85
    all exactly 0, see schema.has_invalid_zero_speeds) are excluded from the
    peer-group IQR computation -- a handful of 0s would otherwise drag a
    (country, road_class, land_use) group's quartiles down and contaminate
    the plausibility verdict for the *other*, real segments in that group --
    and get no plausibility verdict of their own (pd.NA): speed_limit=0 here
    isn't a real measurement, so "high"/"low" trust in it is meaningless.
    """
    gdf = gdf.copy()
    gdf["speedlimit_roadclass_outlier"] = False
    has_flag_col = "data_quality_flag" in gdf.columns
    valid_mask = gdf["data_quality_flag"].isna() if has_flag_col else pd.Series(True, index=gdf.index)

    for _, group in gdf[valid_mask].groupby(["country", "road_class", "land_use"]):
        mask = _iqr_outlier_mask(group["speed_limit"])
        gdf.loc[group.index[mask], "speedlimit_roadclass_outlier"] = True

    gdf["speedlimit_f85_divergence"] = (gdf["f85_speed"] - gdf["speed_limit"]).abs()
    gdf["speedlimit_extreme_divergence"] = gdf["speedlimit_f85_divergence"] > DIVERGENCE_THRESHOLD_KMH

    gdf["speedlimit_plausibility"] = "high"
    low_mask = gdf["speedlimit_roadclass_outlier"] | gdf["speedlimit_extreme_divergence"]
    gdf.loc[low_mask, "speedlimit_plausibility"] = "low"

    invalid_mask = ~valid_mask
    gdf["speedlimit_roadclass_outlier"] = gdf["speedlimit_roadclass_outlier"].astype(object)
    gdf["speedlimit_extreme_divergence"] = gdf["speedlimit_extreme_divergence"].astype(object)
    gdf.loc[invalid_mask, "speedlimit_roadclass_outlier"] = pd.NA
    gdf.loc[invalid_mask, "speedlimit_extreme_divergence"] = pd.NA
    gdf.loc[invalid_mask, "speedlimit_plausibility"] = pd.NA

    return gdf


if __name__ == "__main__":
    from schema import load_target

    target = load_target()
    target = add_speedlimit_plausibility(target)

    print("speedlimit_plausibility distribution:")
    print(target["speedlimit_plausibility"].value_counts())
    print()
    print("by check:")
    print("  road-class outlier:", target["speedlimit_roadclass_outlier"].sum())
    print("  extreme divergence:", target["speedlimit_extreme_divergence"].sum())
    print("  both:", (target["speedlimit_roadclass_outlier"] & target["speedlimit_extreme_divergence"]).sum())
    print()
    print("by country:")
    print(target.groupby("country")["speedlimit_plausibility"].value_counts())
