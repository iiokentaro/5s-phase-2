"""Elvik (2019)'s exponential model, specified over the whole speed distribution.

The REPORTED benefit estimator. `exp_delta_fatal_percent_uniform` is the
fatal-crash reduction figure the README, the CSV lists and the map popups
carry. It never enters the Speed Safety Score; it orders segments that share a
score when safety_score cuts priority_class, and segments that share a rank
when the lists are written.

★ Why this model ★
Elvik (2019) AAP 125:63-69 specifies the entire speed distribution, so the
estimate reflects that the fastest drivers carry the highest fatality rates.
Its two inputs are the mean speed and the standard deviation of speed.

    relative injury rate = exp(k * (v - v_ref))        k: 0.08 / 0.06 / 0.04
    12 intervals, mean +/- 3 SD, each half a SD wide

★ Where the mean and SD come from (distribution_inputs) ★
  tomtom -- TomTom's measured `tomtom_mean_speed` and `tomtom_sd_speed`, on
            every row where both are present and positive.
  adb    -- elsewhere, the ADB `median_speed` as the mean (the two are the same
            number under the paper's normal assumption) and
            sd = (f85_speed - median_speed) / 1.04, the paper's own proxy.
`exp_mean_speed`, `exp_sigma_speed` and `exp_speed_source` record, per row,
the values the estimate used and where they came from.

Elvik (2019) chooses the exponential form because it fits individual-driver
data and high-speed data points well.

★ Two scenarios, because only one of them needs the distribution ★
For a UNIFORM shift the per-interval reduction is identical in every interval
and the 12-interval sum collapses exactly to `(1 - exp(-k*delta)) * 100` --
the distribution machinery buys nothing. It buys something only under upper-
tail compression (the paper's Fig. 3: 16.8% predicted from the mean shift
alone vs 29% once the intervals are respected, because the highest speeds
carry the highest fatality rates). Both are therefore computed:

  uniform  -- the whole distribution slides so its mean lands on v_safe.
              The reported scenario: Elvik (2019) section 4 names lowering the
              speed limit as the measure whose effect resembles this shift,
              and a new limit at v_safe is what this project proposes.
  tailcap  -- every interval speed is capped at v_safe, lower intervals
              untouched; the post-measure 85th percentile becomes v_safe.
              This repo's own stylisation of the paper's Fig. 3 (the paper's
              Table 3 lowers each interval by an arbitrary amount, it does not
              cap), closer to what speed cameras do. Sensitivity only.

`uniform` is deliberately NOT short-circuited to the closed form. Computing
it through the 12 intervals is what lets tests/test_elvik_2019.py use that
algebraic identity as proof the interval machinery is correct.

★ Sign conventions differ between the two scenarios, on purpose ★
`uniform` goes negative when median_speed < v_safe: there is no benefit to
claim, and the sign says so. `tailcap` is
non-negative by construction: capping a distribution that is already entirely
below v_safe changes nothing, so it reads exactly 0. Neither absolute value should be read as a magnitude when the
segment is already at or below v_safe.

★ No confidence-interval columns ★
k does not vary by road environment (the paper reports no environment
moderation) and the paper gives no CI for k. A
`exp_k_used` column would be a constant and `exp_*_ci_low/high` would be
fabricated, so neither is written. The one genuinely per-row derived quantity,
`exp_sigma_speed`, is written -- it is what a reviewer needs to audit a
surprising tailcap value.

★ One distribution shape on every row: the normal ★
The reported columns use the paper's normal assumption on EVERY row, with the
mean and SD above. Where TomTom's 19 measured percentile points exist they are
used a second time, into separate `*_empirical` columns, and only as a check.
They are kept out of the reported figure because that figure is also a sort
key across rows: the measured grid stops at p95 while the normal grid reaches
+2.75 SD, so on the same distribution the measured tailcap reads lower, and a
sort mixing the two shapes would order tied rows by which data happened to
exist.

★ A tie-break, never a score input ★
The pipeline step is registered in pipeline_order.POST_REFINE straight after
misalignment and before safety_score, which reads the reported column to
order segments with the same score before cutting the top 3% / 10% / 20% of
each cell into priority_class. It changes which of those tied segments fill a
class, and never a segment's score. tests/test_pipeline_steps.py pins that
ordering. The CSV writers use the same column as a tie-break when they sort,
which changes the order rows are listed in within a rank and never the stored
rank itself.
"""

import json
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

K_TABLE_PATH = "src/elvik_2019.json"

# The paper's own divisor: the 85th percentile is taken to sit 1.04 SD above
# the mean (the exact value is 1.0364; 1.04 is what Elvik (2019) section 2
# uses, and matching the paper matters more here than the third decimal).
P85_Z = 1.04

N_INTERVALS = 12
SPAN_SD = 3.0
INTERVAL_SD = 0.5
SCENARIOS = ("uniform", "tailcap")

SIGMA_COLUMN = "exp_sigma_speed"
MEAN_COLUMN = "exp_mean_speed"
SOURCE_COLUMN = "exp_speed_source"
TOMTOM_MEAN, TOMTOM_SD = "tomtom_mean_speed", "tomtom_sd_speed"

# The one figure reported as "the" fatal-crash reduction, and the tie-break
# safety_score orders equal scores by and priority_lists sorts by within a rank.
REPORTED_COLUMN = "exp_delta_fatal_percent_uniform"
# REPORTED_COLUMN / 100 * sample_size_avg: the traffic-volume-weighted index
# priority_lists.py's docstring describes. Written by this step, beside the
# column it is derived from.
ABS_COLUMN = "exp_delta_fatal_abs"
EMPIRICAL_SUFFIX = "_empirical"

# --- The empirical check (optional TomTom layer) ------------------------------
# TomTom reports the whole speed distribution as 19 percentile points. Where
# they exist the same two scenarios are also run over the 19 measured bands,
# written to *_empirical columns beside the reported ones, and never mixed
# into them (see the module docstring).
PERCENTILE_LEVELS = tuple(range(5, 100, 5))          # 5, 10, ... 95
N_PERCENTILES = len(PERCENTILE_LEVELS)
PERCENTILES_COLUMN = "tomtom_speed_percentiles"


def _column(severity: str, scenario: str) -> str:
    return f"exp_delta_{severity}_percent_{scenario}"


def _empirical_column(severity: str, scenario: str) -> str:
    return _column(severity, scenario) + EMPIRICAL_SUFFIX


def load_coefficients(path: str = K_TABLE_PATH) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)["coefficients"]


def severity_sets(path: str = K_TABLE_PATH) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)["severity_sets"]


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def interval_grid() -> tuple[np.ndarray, np.ndarray]:
    """(z_mid, share), each shape (12,).

    z_mid runs -2.75 .. +2.75 in steps of 0.5 -- the midpoints of the paper's
    twelve half-SD intervals spanning mean +/- 3 SD.

    The mass beyond +/-3 SD is absorbed into the outermost interval. That is
    what Elvik (2019) Table 1 does:
    its top share is 0.6% = 1 - Phi(2.5), not the 0.49% a truncated normal
    would give. Shares therefore sum to exactly 1 without renormalisation.
    """
    edges = [-math.inf] + [
        -SPAN_SD + INTERVAL_SD * i for i in range(1, N_INTERVALS)
    ] + [math.inf]
    share = np.array([_phi(hi) - _phi(lo) for lo, hi in zip(edges[:-1], edges[1:])])
    z_mid = np.array([
        -SPAN_SD + INTERVAL_SD * (i + 0.5) for i in range(N_INTERVALS)
    ])
    return z_mid, share


def speed_sd(median_speed, f85_speed):
    """(p85 - mean) / 1.04, with a non-positive spread mapped to NaN.

    A segment whose observed 85th percentile is at or below its median has no
    usable distribution, so its estimates are NaN. Substituting a different
    estimator for those rows would make the column mean two different things
    without saying so -- 8 of the 101,251 valid rows are affected.
    """
    sigma = (np.asarray(f85_speed, dtype=float) - np.asarray(median_speed, dtype=float)) / P85_Z
    return np.where(sigma > 0, sigma, np.nan)


def distribution_inputs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, sigma, source) per row: TomTom's measured mean and SD where both
    are present and positive, else the ADB median and (f85 - median) / 1.04.

    Both values come from the same source on a row, never one from each: a
    TomTom mean with an ADB spread would describe no distribution anyone
    measured.
    """
    adb_mean = frame["median_speed"].to_numpy(dtype=float)
    adb_sigma = speed_sd(frame["median_speed"], frame["f85_speed"])
    if TOMTOM_MEAN in frame.columns and TOMTOM_SD in frame.columns:
        tt_mean = pd.to_numeric(frame[TOMTOM_MEAN], errors="coerce").to_numpy(dtype=float)
        tt_sd = pd.to_numeric(frame[TOMTOM_SD], errors="coerce").to_numpy(dtype=float)
        use_tomtom = (tt_mean > 0) & (tt_sd > 0)
    else:
        tt_mean = tt_sd = np.full(len(frame), np.nan)
        use_tomtom = np.zeros(len(frame), dtype=bool)
    mean = np.where(use_tomtom, tt_mean, adb_mean)
    sigma = np.where(use_tomtom, tt_sd, adb_sigma)
    source = np.where(use_tomtom, "tomtom", "adb")
    return mean, sigma, source


def risk_ratio(v_before, v_after, share, k: float) -> np.ndarray:
    """Share-weighted injury rate after / before, per row.

    Public because tests/test_elvik_2019.py feeds it the
    paper's Table 3 interval speeds directly -- that table's post-measure
    state is an arbitrary per-interval speed vector, not any parameterised
    scenario, so reproducing the paper means calling this primitive.

    Accepts (n, 12) arrays or a single (12,) pair. The reference speed cancels
    in the ratio; callers should still pass speeds centred near zero so the
    exponent argument stays small (see the module's vectorisation note).
    """
    v_before = np.atleast_2d(np.asarray(v_before, dtype=float))
    v_after = np.atleast_2d(np.asarray(v_after, dtype=float))
    share = np.asarray(share, dtype=float)
    before = (share * np.exp(k * v_before)).sum(axis=1)
    after = (share * np.exp(k * v_after)).sum(axis=1)
    return after / before


def _centred_speeds(sigma) -> np.ndarray:
    """(n, 12) interval speeds expressed as offsets from each row's own median.

    Centring is not cosmetic: sigma reaches 49 km/h and speeds reach ~150, so
    the uncentred form would evaluate exp(0.08 * 197) and throw away precision
    for a quantity that cancels anyway. Centred, the argument never leaves
    +/- 3*k*sigma (about +/-11.8 at k=0.08).
    """
    z_mid, _ = interval_grid()
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    return sigma[:, None] * z_mid[None, :]


def delta_percent_uniform(median_speed, sigma, v_safe, k: float) -> np.ndarray:
    """Percent reduction if the whole distribution slides until its mean is v_safe.

    `median_speed` is the distribution's mean (see distribution_inputs).
    Algebraically identical to (1 - exp(-k*(median_speed - v_safe))) * 100 for
    any sigma; computed through the intervals regardless, as the docstring at
    the top of this module explains.
    """
    _, share = interval_grid()
    centred = _centred_speeds(sigma)
    shift = np.atleast_1d(np.asarray(v_safe, dtype=float) - np.asarray(median_speed, dtype=float))
    ratio = risk_ratio(centred, centred + shift[:, None], share, k)
    return (1.0 - ratio) * 100.0


def delta_percent_tailcap(median_speed, sigma, v_safe, k: float) -> np.ndarray:
    """Percent reduction if every interval speed above v_safe is capped at it.

    Elvik (2019) Fig. 3's shape: the upper tail is compressed and the lower
    part of the distribution is left alone. Non-negative by construction.
    """
    _, share = interval_grid()
    median_speed = np.atleast_1d(np.asarray(median_speed, dtype=float))
    v_safe = np.atleast_1d(np.asarray(v_safe, dtype=float))
    centred = _centred_speeds(sigma)
    capped = np.minimum(centred + median_speed[:, None], v_safe[:, None]) - median_speed[:, None]
    # capped <= centred elementwise, so the ratio cannot exceed 1 -- but summing
    # twelve exponentials leaves roundoff of order 1e-16, which surfaced as
    # delta = -4.4e-14 on 300 of the stored segments. Clamping the ratio keeps
    # "tailcap is never negative" literally true.
    ratio = np.minimum(risk_ratio(centred, capped, share, k), 1.0)
    return (1.0 - ratio) * 100.0


def empirical_shares() -> np.ndarray:
    """(19,) band widths around each reported percentile point.

    Each point represents the band running to the midpoint of its neighbours:
    p5 stands for [0, 7.5), p10 for [7.5, 12.5), ... p95 for [92.5, 100]. The
    two outer bands absorb the tails -- the same convention interval_grid()
    uses for the normal case, and for the same reason: the shares then sum to
    exactly 1 with no renormalisation.
    """
    levels = np.array(PERCENTILE_LEVELS, dtype=float)
    lo = np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2.0])
    hi = np.concatenate([(levels[:-1] + levels[1:]) / 2.0, [100.0]])
    return (hi - lo) / 100.0


def empirical_mean(speeds) -> np.ndarray:
    """Share-weighted mean of the 19 reported speeds, per row.

    This -- not TomTom's own averageSpeed -- is what the two empirical
    scenarios shift and cap around. averageSpeed is a different estimator over
    a different weighting, and substituting it would break the algebraic
    identity that makes delta_percent_uniform_empirical testable.
    """
    speeds = np.atleast_2d(np.asarray(speeds, dtype=float))
    return (empirical_shares() * speeds).sum(axis=1)


def delta_percent_uniform_empirical(speeds, v_safe, k: float) -> np.ndarray:
    """Percent reduction if all 19 speeds slide until their mean is v_safe.

    The mirror of delta_percent_uniform on a measured distribution. A uniform
    shift of delta gives ratio = exp(k*delta) whatever the shape, so this
    satisfies the same closed form with the share-weighted mean in place of
    median_speed -- which is what tests/test_elvik_2019.py anchors it on.
    """
    share = empirical_shares()
    speeds = np.atleast_2d(np.asarray(speeds, dtype=float))
    mean = empirical_mean(speeds)
    centred = speeds - mean[:, None]
    shift = np.atleast_1d(np.asarray(v_safe, dtype=float)) - mean
    ratio = risk_ratio(centred, centred + shift[:, None], share, k)
    return (1.0 - ratio) * 100.0


def delta_percent_tailcap_empirical(speeds, v_safe, k: float) -> np.ndarray:
    """Percent reduction if every reported speed above v_safe is capped at it.

    The measured counterpart of delta_percent_tailcap, and the one scenario
    where the measured distribution changes the answer itself (for the uniform
    shift it changes only the derivation). Same roundoff clamp, same reason.
    """
    share = empirical_shares()
    speeds = np.atleast_2d(np.asarray(speeds, dtype=float))
    v_safe = np.atleast_1d(np.asarray(v_safe, dtype=float))
    mean = empirical_mean(speeds)
    centred = speeds - mean[:, None]
    capped = np.minimum(speeds, v_safe[:, None]) - mean[:, None]
    ratio = np.minimum(risk_ratio(centred, capped, share, k), 1.0)
    return (1.0 - ratio) * 100.0


def _empirical_speeds(values) -> tuple[np.ndarray, np.ndarray]:
    """(mask, speeds) for rows carrying a usable 19-point distribution.

    A row qualifies only with all 19 points present and a non-degenerate
    spread; everything else -- including the whole frame when the column is
    absent -- gets NaN in the *_empirical columns.
    """
    usable = np.zeros(len(values), dtype=bool)
    speeds = np.zeros((len(values), N_PERCENTILES), dtype=float)
    for i, value in enumerate(values):
        if value is None or isinstance(value, float):
            continue
        arr = np.asarray(value, dtype=float)
        if arr.shape != (N_PERCENTILES,) or not np.isfinite(arr).all():
            continue
        if arr[-1] <= arr[0]:
            continue
        usable[i] = True
        speeds[i] = arr
    return usable, speeds


def exponential_columns(coefficients: dict | None = None) -> list[str]:
    """The numeric inputs used plus the normal-path estimates -- the columns
    every row carries (SOURCE_COLUMN, the one text column, is in written_columns)."""
    coefficients = coefficients or load_coefficients()
    return [MEAN_COLUMN, SIGMA_COLUMN] + [
        _column(sev, scenario) for sev in coefficients for scenario in SCENARIOS
    ]


def empirical_columns(coefficients: dict | None = None) -> list[str]:
    """The check columns, filled on TomTom-covered rows only."""
    coefficients = coefficients or load_coefficients()
    return [
        _empirical_column(sev, scenario) for sev in coefficients for scenario in SCENARIOS
    ]


def written_columns(coefficients: dict | None = None) -> list[str]:
    """Every column add_exponential_reduction writes."""
    return (exponential_columns(coefficients) + empirical_columns(coefficients)
            + [ABS_COLUMN, SOURCE_COLUMN])


def add_exponential_reduction(gdf, coefficients: dict | None = None) -> pd.DataFrame:
    """Write sigma, the normal-path estimates, their empirical checks and the abs index.

    All three severities are computed unconditionally. The frontend's severity
    selector chooses what gets REPORTED, never what gets computed: making the
    committed parquet's contents depend on an invisible run flag would be a
    reproducibility trap, and the extra cost is two numpy passes.
    """
    gdf = gdf.copy()
    coefficients = coefficients or load_coefficients()

    for col in written_columns(coefficients):
        gdf[col] = np.nan
    gdf[SOURCE_COLUMN] = pd.Series(None, index=gdf.index, dtype="object")

    has_flag_col = "data_quality_flag" in gdf.columns
    valid_mask = gdf["data_quality_flag"].isna() if has_flag_col else pd.Series(True, index=gdf.index)
    valid = gdf.loc[valid_mask]

    median_speed, sigma, source = distribution_inputs(valid)
    v_safe = valid["v_safe"].to_numpy(dtype=float)

    if PERCENTILES_COLUMN in valid.columns:
        empirical, speeds = _empirical_speeds(valid[PERCENTILES_COLUMN].to_numpy())
    else:
        empirical = np.zeros(len(valid), dtype=bool)
        speeds = np.zeros((len(valid), N_PERCENTILES), dtype=float)

    # The mean and SD the reported columns were computed with, and their source.
    gdf.loc[valid_mask, MEAN_COLUMN] = median_speed
    gdf.loc[valid_mask, SIGMA_COLUMN] = sigma
    gdf.loc[valid_mask, SOURCE_COLUMN] = source

    for severity, k in coefficients.items():
        for scenario, normal_fn, empirical_fn in (
            ("uniform", delta_percent_uniform, delta_percent_uniform_empirical),
            ("tailcap", delta_percent_tailcap, delta_percent_tailcap_empirical),
        ):
            gdf.loc[valid_mask, _column(severity, scenario)] = normal_fn(median_speed, sigma, v_safe, k)
            if empirical.any():
                gdf.loc[valid_mask, _empirical_column(severity, scenario)] = np.where(
                    empirical, empirical_fn(speeds, v_safe, k), np.nan
                )

    if "sample_size_avg" in gdf.columns:
        gdf.loc[valid_mask, ABS_COLUMN] = (
            gdf.loc[valid_mask, REPORTED_COLUMN] / 100
            * pd.to_numeric(gdf.loc[valid_mask, "sample_size_avg"], errors="coerce")
        )

    print(
        f"elvik_2019: TomTom mean/SD on {int((source == 'tomtom').sum())} / {len(valid)} valid rows; "
        f"sigma <= 0 on {int(np.isnan(sigma).sum())} -> NaN (no distribution to specify); empirical check on "
        f"{int(empirical.sum())} TomTom rows"
    )
    return gdf


def ensure_exponential_columns(df, coefficients: dict | None = None) -> pd.DataFrame:
    """Return df unchanged if the columns are already there, else compute them.

    The `quick` preset enters at the stored parquet and never runs POST_REFINE,
    so without this the sensitivity step and the CSV writers would KeyError on
    any parquet built before these columns existed. The check is over every
    written column, so a parquet from before the *_empirical split -- whose
    exp_delta_* columns mixed the two methods -- is recomputed.
    Recomputing costs well under a second.
    """
    coefficients = coefficients or load_coefficients()
    if all(col in df.columns for col in written_columns(coefficients)):
        return df
    return add_exponential_reduction(df, coefficients)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    coefficients = load_coefficients()
    z_mid, share = interval_grid()
    print(f"coefficients: {coefficients}")
    print(f"interval shares (sum={share.sum():.12f}): {np.round(share, 4)}")
    print(f"interval midpoints (SD): {z_mid}")

    # Elvik (2019) Table 2: a uniform 6 km/h reduction is a 38.1% fatality drop.
    print("\nTable 2 check (uniform -6 km/h, k=0.08):")
    for test_sigma in (4.0, 8.0, 20.0):
        got = delta_percent_uniform([60.0], [test_sigma], [54.0], 0.08)[0]
        print(f"  sigma={test_sigma:5.1f} -> {got:.12f}%  (closed form {(1 - math.exp(-0.48)) * 100:.12f}%)")

    print("\ntailcap vs uniform (mean=60, sigma=8, v_safe=54, k=0.08):")
    print(f"  uniform {delta_percent_uniform([60.0], [8.0], [54.0], 0.08)[0]:.2f}%")
    print(f"  tailcap {delta_percent_tailcap([60.0], [8.0], [54.0], 0.08)[0]:.2f}%")

    df = pd.read_parquet("data/processed/segments_v_safe.parquet")
    out = add_exponential_reduction(df)
    valid = out[out["data_quality_flag"].isna()]
    print("\n=== on the stored parquet ===")
    print(valid[exponential_columns()].describe().round(2).to_string())
