"""Sensitivity analysis, answering the organizers' Methodology Warning directly.

Three independent questions, the first two phrased as "if we'd made a
different defensible choice, would the Top Priority list have come out
basically the same list of segments?", the third phrased as "how far does
the reported benefit estimate move under the alternatives it was chosen over?":

1. **Sample-size robustness.** `sample_size_avg` (probes per sampling point,
   see aadt_estimation.py) is right-skewed, and its bottom quartile is the
   candidate "low confidence" threshold. If the Top Priority list is largely
   segments whose extreme misalignment/exposure reading is an artifact of a
   thin sample, restricting to the well-sampled segments and recomputing
   should shrink or reshuffle the list a lot. It shouldn't if the conclusion
   is real.
2. **Weight robustness.** `safety_score.py`'s 0.50/0.35/0.15 weighting is a
   stated assumption. Re-running the score under a handful of other
   defensible weightings and checking whether the Top Priority list stays
   close to the baseline tests that the conclusion does not hinge on the
   weight choice.
3. **The Elvik (2019) benefit estimate.** `exp_delta_fatal_percent_uniform`
   is the reported fatal-crash reduction (elvik_2019.py). This aggregates it
   for the segments the benefit headline is actually about (`review_track ==
   "Review Needed"`, restricted to a positive estimate -- segments already at
   or below v_safe have no reduction to claim), next to two things that
   bound how far to trust it: the `tailcap` scenario (the same model, with
   the measure biting only on the fastest drivers), and on TomTom-covered
   rows the same estimate recomputed over the measured 19-point distribution
   in place of the normal assumption.

Tests 1-2 measure the overlap of two Top Priority lists by length (the WGS84
geodesic length safety_score cuts the classes by): recall is the share of the
baseline list's length that the other list keeps, and Jaccard is the shared
length over the length of either list. The segment counts are kept beside it.

All of them reuse the pipeline's own code: 1-2 call `add_safety_score`
directly, and 3 reads `add_exponential_reduction`'s output columns.
"""

import argparse
import sys
import warnings

import pandas as pd

sys.path.insert(0, "src")

import elvik_2019  # noqa: E402
from geometry import geodesic_length_m  # noqa: E402
from review_track import REVIEW_NEEDED  # noqa: E402
from safety_score import SCORE_OUTPUT_COLUMNS, add_safety_score  # noqa: E402

SAMPLE_SIZE_LOW_QUANTILE = 0.25  # bottom-quartile "low confidence" candidate threshold

WEIGHT_SCENARIOS = {
    "baseline (0.50/0.35/0.15)": (0.50, 0.35, 0.15),
    "equal weights (0.33/0.33/0.34)": (1 / 3, 1 / 3, 1 / 3),
    "misalignment-heavy (0.70/0.20/0.10)": (0.70, 0.20, 0.10),
    "exposure-heavy (0.30/0.55/0.15)": (0.30, 0.55, 0.15),
    "confidence-heavy (0.40/0.30/0.30)": (0.40, 0.30, 0.30),
}


def _top_segments(gdf) -> dict:
    """segment_id -> length in metres, for the Top Priority rows."""
    top = gdf[gdf["priority_class"] == "Top Priority"]
    return dict(zip(top["segment_id"], geodesic_length_m(top)))


def _overlap_stats(baseline_top: dict, other_top: dict) -> dict:
    """Overlap of two {segment_id: length_m} lists, by length and by count."""
    inter = baseline_top.keys() & other_top.keys()
    union = baseline_top.keys() | other_top.keys()
    length = {**baseline_top, **other_top}
    km = lambda ids: sum(length[i] for i in ids) / 1000  # noqa: E731
    base_km, inter_km, union_km = km(baseline_top), km(inter), km(union)
    return {
        "n_baseline": len(baseline_top),
        "n_other": len(other_top),
        "n_overlap": len(inter),
        "km_baseline": base_km,
        "km_other": km(other_top),
        "km_overlap": inter_km,
        "recall_of_baseline": inter_km / base_km if base_km > 0 else float("nan"),
        "jaccard": inter_km / union_km if union_km > 0 else float("nan"),
    }


def sample_size_robustness(valid: pd.DataFrame) -> dict:
    baseline_top = _top_segments(valid)

    threshold = valid["sample_size_avg"].quantile(SAMPLE_SIZE_LOW_QUANTILE)
    high_sample = valid[valid["sample_size_avg"] >= threshold].copy()
    # Drop the columns add_safety_score recreates, so re-running it on the
    # filtered population can't accidentally read stale values.
    high_sample = high_sample.drop(columns=SCORE_OUTPUT_COLUMNS, errors="ignore")
    recomputed, _ = add_safety_score(high_sample)
    recomputed_top = _top_segments(recomputed)

    n_excluded_from_baseline_top = len(baseline_top.keys() - set(high_sample["segment_id"]))

    stats = _overlap_stats(baseline_top, recomputed_top)
    stats["sample_size_threshold"] = threshold
    stats["n_excluded_low_sample_segments"] = len(valid) - len(high_sample)
    stats["n_baseline_top_excluded_as_low_sample"] = n_excluded_from_baseline_top
    return stats


def weight_robustness(valid: pd.DataFrame) -> pd.DataFrame:
    base = valid.drop(columns=SCORE_OUTPUT_COLUMNS, errors="ignore")
    baseline_top = None
    rows = []
    for label, (w_m, w_e, w_c) in WEIGHT_SCENARIOS.items():
        recomputed, _ = add_safety_score(base, weight_misalignment=w_m, weight_exposure=w_e, weight_confidence=w_c)
        top = _top_segments(recomputed)
        if baseline_top is None:
            baseline_top = top  # first entry in dict is the baseline scenario
        stats = _overlap_stats(baseline_top, top)
        stats["scenario"] = label
        rows.append(stats)
    return pd.DataFrame(rows).set_index("scenario")


def exponential_model_comparison(valid: pd.DataFrame, severity: str = "fatal") -> pd.DataFrame:
    """The reported Elvik (2019) estimate beside its two alternatives.

    Population: `review_track == REVIEW_NEEDED` with a positive reported
    estimate (elvik_2019.REPORTED_COLUMN > 0, i.e. median_speed > v_safe).

    `severity` is a key of elvik_2019.json's `severity_sets`: "fatal" (k=0.08
    only, the reported one) or "all3" (0.08/0.06/0.04, indexed by
    (road_environment, severity)). The exponential columns are computed
    on the fly when the frame predates them, so switching severity never needs
    a rebuild.

    Columns:
      uniform_mean / tailcap_mean   -- normal assumption, every row.
      tailcap_minus_uniform_mean    -- what the distribution adds when the
                                       measure bites on the fastest drivers
                                       (Elvik 2019 Fig. 3).
      n_tomtom and the *_tomtom_* pairs -- on TomTom-covered rows only, the
                                       normal estimate and the same estimate
                                       over the measured 19-point distribution,
                                       over identical rows.

    Grouped by `road_environment`, the urban/rural split the lists use.
    """
    severities = elvik_2019.severity_sets()[severity]
    valid = elvik_2019.ensure_exponential_columns(valid)
    candidates = valid[
        (valid["review_track"] == REVIEW_NEEDED) & (valid[elvik_2019.REPORTED_COLUMN] > 0)
    ]

    frames = []
    for name in severities:
        uniform, tailcap = f"exp_delta_{name}_percent_uniform", f"exp_delta_{name}_percent_tailcap"
        uniform_emp = uniform + elvik_2019.EMPIRICAL_SUFFIX
        tailcap_emp = tailcap + elvik_2019.EMPIRICAL_SUFFIX
        table = candidates.groupby("road_environment").agg(
            n=(uniform, "count"),
            uniform_mean=(uniform, "mean"),
            tailcap_mean=(tailcap, "mean"),
            # sigma <= 0 leaves no distribution to specify; count the rows the
            # estimate is missing on, so they stay visible.
            n_nan=(uniform, lambda col: int(col.isna().sum())),
        )
        table["tailcap_minus_uniform_mean"] = table["tailcap_mean"] - table["uniform_mean"]

        # Normal vs measured, over exactly the rows that carry both.
        both = candidates[candidates[uniform_emp].notna()]
        check = both.groupby("road_environment").agg(
            n_tomtom=(uniform_emp, "count"),
            uniform_tomtom_normal=(uniform, "mean"),
            uniform_tomtom_empirical=(uniform_emp, "mean"),
            tailcap_tomtom_normal=(tailcap, "mean"),
            tailcap_tomtom_empirical=(tailcap_emp, "mean"),
        )
        table = table.join(check)
        table["n_tomtom"] = table["n_tomtom"].fillna(0).astype(int)
        table["severity"] = name
        frames.append(table.set_index("severity", append=True))

    combined = pd.concat(frames)
    return combined.droplevel("severity") if len(severities) == 1 else combined


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--severity", default="fatal", choices=sorted(elvik_2019.severity_sets()),
                        help="which Elvik (2019) severity coefficients analysis 3 reports")
    args = parser.parse_args()

    import geopandas as gpd

    df = gpd.read_parquet("data/processed/segments_v_safe.parquet")
    valid = df[df["data_quality_flag"].isna()].copy()

    print("=== 1. sample-size robustness ===")
    s = sample_size_robustness(valid)
    print(f"sample_size_avg threshold (25th pct): {s['sample_size_threshold']:.0f}")
    print(f"segments excluded as low-sample: {s['n_excluded_low_sample_segments']} / {len(valid)}")
    print(f"baseline Top Priority (n={s['n_baseline']}) segments excluded outright as low-sample: "
          f"{s['n_baseline_top_excluded_as_low_sample']}")
    print(f"recomputed-on-high-sample-only Top Priority: n={s['n_other']} ({s['km_other']:.1f} km)")
    print(f"overlap: {s['n_overlap']} segments, {s['km_overlap']:.1f} of {s['km_baseline']:.1f} km "
          f"(recall of baseline={s['recall_of_baseline']:.1%}, jaccard={s['jaccard']:.1%}, by length)")

    print("\n=== 2. weight robustness ===")
    w = weight_robustness(valid)
    print(w[["n_baseline", "n_other", "n_overlap", "km_baseline", "km_other", "km_overlap",
             "recall_of_baseline", "jaccard"]].to_string(
        formatters={"recall_of_baseline": "{:.1%}".format, "jaccard": "{:.1%}".format,
                    "km_baseline": "{:.1f}".format, "km_other": "{:.1f}".format,
                    "km_overlap": "{:.1f}".format}
    ))

    print("\n=== 3. Elvik (2019) benefit estimate ===")
    coefficients = elvik_2019.load_coefficients()
    used = {k: v for k, v in coefficients.items() if k in elvik_2019.severity_sets()[args.severity]}
    print(f"k = {used}; normal assumption on every row; reported = {elvik_2019.REPORTED_COLUMN}")
    print("never enters safety_score / priority_class / rank_within_environment")
    print(exponential_model_comparison(valid, severity=args.severity).round(1).to_string())
