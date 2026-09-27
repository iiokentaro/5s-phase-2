"""Misalignment (policy axis) vs operating_gap (diagnostic axis).

★ The two axes must never be conflated ★
- `misalignment = speed_limit - v_safe` is the main axis: how far the
  *posted limit* sits above/below the safe speed. Positive means the limit
  is too high for who's exposed to this road -- the priority signal for
  "review this speed limit."
- `operating_gap = f85_speed - v_safe` is diagnostic only: whether traffic
  is *actually* travelling above the safe speed, i.e. whether enforcement
  or physical engineering (beyond a sign change) is needed. It must never
  feed into the priority score.

`misalignment_magnitude = max(0, misalignment)` isolates the "too high"
direction that drives priority (VRU protection is the point of this
challenge). The opposite direction (limit too low / possible
over-regulation) is kept as its own column, `excess_caution_magnitude`, and
is out of the priority axis.

Rows flagged `data_quality_flag=='invalid_speed'` (schema.has_invalid_zero_speeds)
have placeholder zeros for speed_limit and f85_speed, so both axes are NA on
them.
"""

import sys

import pandas as pd

sys.path.insert(0, "src")


def add_misalignment(gdf):
    gdf = gdf.copy()
    has_flag_col = "data_quality_flag" in gdf.columns
    valid_mask = gdf["data_quality_flag"].isna() if has_flag_col else pd.Series(True, index=gdf.index)

    gdf["misalignment"] = pd.NA
    gdf["misalignment_magnitude"] = pd.NA
    gdf["excess_caution_magnitude"] = pd.NA
    gdf["operating_gap"] = pd.NA

    misalignment = gdf.loc[valid_mask, "speed_limit"] - gdf.loc[valid_mask, "v_safe"]
    gdf.loc[valid_mask, "misalignment"] = misalignment
    gdf.loc[valid_mask, "misalignment_magnitude"] = misalignment.clip(lower=0)
    gdf.loc[valid_mask, "excess_caution_magnitude"] = (-misalignment).clip(lower=0)
    gdf.loc[valid_mask, "operating_gap"] = gdf.loc[valid_mask, "f85_speed"] - gdf.loc[valid_mask, "v_safe"]

    for col in ["misalignment", "misalignment_magnitude", "excess_caution_magnitude", "operating_gap"]:
        gdf[col] = pd.to_numeric(gdf[col])

    # The same policy axis against the pre-split recommended speed, present only
    # when build() ran with the legacy comparison path on. safety_score reads
    # `misalignment_magnitude` alone.
    if "v_safe_legacy" in gdf.columns:
        gdf["misalignment_legacy"] = pd.NA
        gdf["misalignment_magnitude_legacy"] = pd.NA
        misalignment_legacy = gdf.loc[valid_mask, "speed_limit"] - gdf.loc[valid_mask, "v_safe_legacy"]
        gdf.loc[valid_mask, "misalignment_legacy"] = misalignment_legacy
        gdf.loc[valid_mask, "misalignment_magnitude_legacy"] = misalignment_legacy.clip(lower=0)
        for col in ["misalignment_legacy", "misalignment_magnitude_legacy"]:
            gdf[col] = pd.to_numeric(gdf[col])

    return gdf


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, _ = build()  # build() already runs add_misalignment as part of the pipeline
    valid = target[target["data_quality_flag"].isna()]

    print(f"valid (scoreable) segments: {len(valid)} / {len(target)}\n")

    print("=== misalignment (main axis, SpeedLimit - V_safe) ===")
    too_high = valid["misalignment"] > 0
    print(f"too-high direction (review priority): {too_high.sum()} / {len(valid)} ({too_high.mean():.1%})")
    print("\nby road_class:")
    print(valid.groupby("road_class")["misalignment"].agg(["count", lambda s: (s > 0).mean()]).rename(
        columns={"<lambda_0>": "pct_too_high"}))
    print("\nby land_use:")
    print(valid.groupby("land_use")["misalignment"].agg(["count", lambda s: (s > 0).mean()]).rename(
        columns={"<lambda_0>": "pct_too_high"}))

    print("\n=== operating_gap (diagnostic axis, F85 - V_safe) ===")
    over = valid["operating_gap"] > 0
    print(f"observed speed exceeds V_safe: {over.sum()} / {len(valid)} ({over.mean():.1%})")

    print("\n=== agreement between axes (top decile each) ===")
    top_n = max(1, len(valid) // 10)
    top_misalignment = set(valid.nlargest(top_n, "misalignment_magnitude").index)
    top_operating_gap = set(valid.nlargest(top_n, "operating_gap").index)
    overlap = top_misalignment & top_operating_gap
    print(f"top decile by misalignment_magnitude: {len(top_misalignment)} segments")
    print(f"top decile by operating_gap: {len(top_operating_gap)} segments")
    print(f"overlap (limit too high AND actually speeding -- highest-confidence priority): "
          f"{len(overlap)} ({len(overlap) / top_n:.1%} of each decile)")
