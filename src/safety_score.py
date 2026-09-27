"""Speed Safety Score -- a transparent, explainable one-dimensional priority
score over three axes: misalignment, exposure, and confidence.

★ Three axes, weighted sum, no black box ★
(a) misalignment  -- `misalignment_magnitude` (main "too high" direction
    only), the policy gap this whole deliverable is about.
(b) exposure       -- `exposure_level` (high/medium/low). Same gap, more
    people exposed, higher priority -- this is what lets two segments with
    an identical km/h gap rank differently.
(c) confidence     -- `exposure_confidence`, `road_structure_confidence`, and
    `speedlimit_plausibility` integrated into one `confidence_level`
    (high/medium/low) by counting how many of the three read "low". This is
    the first point `speedlimit_plausibility` feeds the score itself
    (review_track.py then uses it again, on its own, to split the list).

Weights (WEIGHT_MISALIGNMENT=0.50, WEIGHT_EXPOSURE=0.35,
WEIGHT_CONFIDENCE=0.15): misalignment gets the largest share because it's
the deliverable's main axis; exposure is weighted
second so it can meaningfully separate segments that tie on misalignment,
without ever letting a low-confidence reading outweigh the two substantive
axes. These are stated assumptions, not fitted values -- sensitivity_analysis.py
tests them under alternative weightings.

★ Why the classes are shares of the network ★
"High misalignment AND high exposure" alone covers about a fifth of the
segments: the problem this dataset surfaces is widespread, and an agency
cannot act on thousands of segments at once. So each class is the most
severe slice by share of length (PRIORITY_CLASS_SHARES: top 3% / 10% / 20%),
and the score ranks every segment the same transparent way.
sensitivity_analysis.py re-runs this under other weights to check the list
does not hinge on the weight choice.

★ Segments that share a score ★
The score takes few distinct values (the misalignment axis saturates at
MISALIGNMENT_CAP_KMH, the other two axes are three-step ladders), so many
segments tie. A cut at a score value would put every tied segment in the
higher class: in Thailand rural 13% of segments score 100, which would fill
Top Priority and leave Priority empty. So each cell is sorted by score, then by
the Elvik (2019) fatal-crash reduction (elvik_2019.REPORTED_COLUMN, largest
first, missing last), then by segment_id so the order is the same on every
run.

★ Cut by length ★
The classes take the top 3% / 10% / 20% of each cell's total length, walking
down that order: a segment joins a class when the length before it is below
the class's share of the cell (so the segment that crosses a boundary is
included, and a class overshoots its share by at most that one segment).
The build splits segments around POI zones and influence zones, so a count
would give a heavily split stretch more of the list than an unsplit stretch
of the same length; and an authority budgets road works by length. Length
is the WGS84 geodesic length of the geometry (geometry.geodesic_length_m),
the same value the GIS exports write as shape_length.
The reduction never changes a segment's score; it only decides which of the
tied segments fill a class first.

★ Why the cutoff is computed per (country, land_use) ★
Thailand and Maharashtra are two separately funded programmes, each needing
its own "top N segments to act on". Maharashtra's score distribution sits
lower across the board, so a single pooled percentile put 0 Maharashtra
segments in Top Priority / Priority and only 69 of 3,577 in Watch. The same
holds between urban and rural within a country: exposure_level.py builds
their exposure from different signal sets, so their scores are not on one
scale. So the cutoff is taken within each of the four (country, land_use)
cells. `safety_score` itself is computed the same way everywhere; only
`priority_class` is cut per cell.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from elvik_2019 import REPORTED_COLUMN as TIE_BREAK_COLUMN  # noqa: E402
from geometry import geodesic_length_m  # noqa: E402

MISALIGNMENT_CAP_KMH = 60  # >=60 km/h over V_safe is already "as severe as it gets" for ranking purposes
WEIGHT_MISALIGNMENT = 0.50
WEIGHT_EXPOSURE = 0.35
WEIGHT_CONFIDENCE = 0.15

EXPOSURE_POINTS = {"low": 0.0, "medium": 0.5, "high": 1.0}
CONFIDENCE_COLUMNS = ["exposure_confidence", "road_structure_confidence", "speedlimit_plausibility"]

# Corroboration by the optional TomTom layer, worth exactly one rung of the
# three-point confidence ladder below. It is added AFTER the ladder, outside
# CONFIDENCE_COLUMNS, for two reasons: _confidence_level's cut points
# (0 low -> high, 1 -> medium) are calibrated for exactly three columns, and a
# "low" in a fourth column would encode the ABSENCE of TomTom as evidence of
# poor data, when it is evidence of nothing at all -- the same reason TomTom
# stays out of speedlimit_plausibility. Rows without the layer keep their
# ladder value.
TOMTOM_CONFIDENCE_BONUS = 1 / 3

# Cumulative top-down shares of the *valid* rows' total length, taken
# separately within each (country, land_use) cell (see module docstring).
# top: most severe top 3% -- the literal "act on this first" list.
# priority: next 7% (top 10% cumulative) -- the follow-up pipeline.
# watch: next 10% (top 20% cumulative) -- monitor, no action yet.
PRIORITY_CLASS_SHARES = {
    "Top Priority": 0.03,
    "Priority": 0.10,
    "Watch": 0.20,
}
PRIORITY_CLASSES_ORDERED = ["Top Priority", "Priority", "Watch", "Low Priority"]
DATA_QUALITY_CLASS = "Data Quality Issue (Excluded)"

CONFIDENCE_LEVELS = ["high", "medium", "low"]

# Every column add_safety_score writes. sensitivity_analysis re-runs the score
# and has to drop these first; naming them once here is what stops that list
# and this function from drifting apart (tests/test_safety_score.py pins it).
SCORE_OUTPUT_COLUMNS = [
    "confidence_level",
    "misalignment_norm",
    "exposure_norm",
    "confidence_norm",
    "safety_score",
    "priority_class",
    "score_explanation",
    "exposure_level_code",
    "confidence_level_code",
    "speed_data_source_code",
]

# Integer codes for the three categorical values folded into score_explanation's
# text. build_tiles.py puts these *_code columns in the PMTiles and leaves the
# string columns in the parquet -- a repeated small int costs far
# less per feature than a repeated word, and the score_explanation sentence
# already spells the value out in English for anyone reading the tile.
EXPOSURE_LEVEL_CODES = {"low": 0, "medium": 1, "high": 2}
CONFIDENCE_LEVEL_CODES = {"low": 0, "medium": 1, "high": 2}
SPEED_DATA_SOURCE_CODES = {"overture": 0, "tomtom": 1}

# Stated on every row, in every country, whether or not a TomTom layer exists.
# A sentence that appears or disappears depending on an invisible file on disk
# would be a worse reproducibility trap than one that is always present.
SPEED_SOURCE_LABEL_EN = {
    "overture": "Overture/ADB probe data",
    "tomtom": "Overture/ADB, corroborated by TomTom Traffic Stats",
}
DEFAULT_SPEED_SOURCE = "overture"


def _misalignment_norm(df: pd.DataFrame) -> pd.Series:
    return df["misalignment_magnitude"].clip(upper=MISALIGNMENT_CAP_KMH) / MISALIGNMENT_CAP_KMH


def _exposure_norm(df: pd.DataFrame) -> pd.Series:
    return df["exposure_level"].astype(str).map(EXPOSURE_POINTS).astype(float)


def ensure_norm_columns(gdf: pd.DataFrame) -> pd.DataFrame:
    """Add misalignment_norm and exposure_norm to a frame scored before
    add_safety_score wrote them (such as an older committed
    segments_v_safe.parquet), from the same inputs and on the same rows it
    scores. A frame that already has both is returned unchanged."""
    if {"misalignment_norm", "exposure_norm"} <= set(gdf.columns):
        return gdf
    gdf = gdf.copy()
    valid_mask = gdf["safety_score"].notna()
    valid = gdf.loc[valid_mask]
    for col, values in (("misalignment_norm", _misalignment_norm(valid)),
                        ("exposure_norm", _exposure_norm(valid))):
        gdf[col] = pd.NA
        gdf.loc[valid_mask, col] = values.values
        gdf[col] = pd.to_numeric(gdf[col])
    return gdf


def _confidence_level(low_count: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([low_count == 0, low_count == 1], ["high", "medium"], default="low"),
        index=low_count.index,
    )


def _confidence_label_en(level: str) -> str:
    return {"high": "High", "medium": "Medium", "low": "Low"}[level]


def _exposure_label_en(level: str) -> str:
    return {"high": "High", "medium": "Medium", "low": "Low"}[level]


def _speed_source_label_en(source) -> str:
    return SPEED_SOURCE_LABEL_EN.get(str(source), SPEED_SOURCE_LABEL_EN[DEFAULT_SPEED_SOURCE])


def _source_sentence(speed_data_source) -> str:
    return f" Speed data source: {_speed_source_label_en(speed_data_source)}."


def _explain(misalignment_magnitude, exposure_level, confidence_level,
             speed_data_source=DEFAULT_SPEED_SOURCE) -> str:
    if misalignment_magnitude <= 0:
        gap = "Posted speed limit does not exceed the safe speed (V_safe)"
    else:
        gap = f"Posted speed limit is {misalignment_magnitude:.0f} km/h above the safe speed (V_safe)"
    return (
        f"{gap}. VRU exposure: {_exposure_label_en(exposure_level)}. "
        f"Data confidence: {_confidence_label_en(confidence_level)}."
        f"{_source_sentence(speed_data_source)}"
    )


def _speed_data_source(gdf) -> pd.Series:
    """The provenance column, or the Overture default for a frame that predates
    tomtom_enrichment."""
    if "speed_data_source" in gdf.columns:
        return gdf["speed_data_source"].fillna(DEFAULT_SPEED_SOURCE).astype(str)
    return pd.Series(DEFAULT_SPEED_SOURCE, index=gdf.index)


def add_safety_score(
    gdf,
    weight_misalignment=WEIGHT_MISALIGNMENT,
    weight_exposure=WEIGHT_EXPOSURE,
    weight_confidence=WEIGHT_CONFIDENCE,
):
    """Weight overrides exist solely for sensitivity_analysis.py's weight-
    sensitivity test -- the pipeline itself always calls this with the
    documented defaults above."""
    gdf = gdf.copy()
    has_flag_col = "data_quality_flag" in gdf.columns
    valid_mask = gdf["data_quality_flag"].isna() if has_flag_col else pd.Series(True, index=gdf.index)
    valid = gdf.loc[valid_mask]

    low_count = (valid[CONFIDENCE_COLUMNS] == "low").sum(axis=1)
    confidence_level = _confidence_level(low_count)

    misalignment_norm = _misalignment_norm(valid)
    exposure_norm = _exposure_norm(valid)
    source = _speed_data_source(valid)
    bonus = (source == "tomtom").astype(float) * TOMTOM_CONFIDENCE_BONUS
    # Clipped so confidence_norm stays in [1/3, 1] and the score in [0, 100];
    # the quantile machinery below is therefore untouched by the bonus.
    confidence_norm = (
        confidence_level.map({"high": 1.0, "medium": 2 / 3, "low": 1 / 3}) + bonus
    ).clip(upper=1.0)

    score = 100 * (
        weight_misalignment * misalignment_norm
        + weight_exposure * exposure_norm
        + weight_confidence * confidence_norm
    )

    priority_class = pd.Series(DATA_QUALITY_CLASS, index=gdf.index, dtype=object)
    valid_class = pd.Series("Low Priority", index=valid.index, dtype=object)
    order = pd.DataFrame({
        "country": valid["country"],
        "land_use": valid["land_use"],
        "score": score,
        "tie_break": pd.to_numeric(valid[TIE_BREAK_COLUMN], errors="coerce"),
        "segment_id": valid["segment_id"].astype(str),
        "length_m": geodesic_length_m(valid),
    }, index=valid.index).sort_values(["score", "tie_break", "segment_id"], ascending=[False, False, True],
                   na_position="last", kind="mergesort")
    # thresholds holds, per cell and class, the lowest score that made the class
    # (NaN when no segment made it).
    thresholds = {}
    for (country, land_use), cell in order.groupby(["country", "land_use"], sort=False):
        total = cell["length_m"].sum()
        # Share of the cell's length before each segment in the order; a cell of
        # zero length puts nobody in a class.
        before = ((cell["length_m"].cumsum() - cell["length_m"]) / total).to_numpy() if total > 0 \
            else np.full(len(cell), np.inf)
        cell_thresholds = {}
        # From the widest share down, so a stricter class overwrites a wider one.
        for label in ["Watch", "Priority", "Top Priority"]:
            # The tolerance keeps a segment that starts exactly at the share
            # out, however the float sum of the lengths before it rounds.
            members = cell.index[before < PRIORITY_CLASS_SHARES[label] - 1e-9]
            valid_class.loc[members] = label
            cell_thresholds[label] = score.loc[members].min() if len(members) else np.nan
        thresholds[(country, land_use)] = {
            label: cell_thresholds[label] for label in PRIORITY_CLASS_SHARES
        }

    all_source = _speed_data_source(gdf)
    explanation = pd.Series(
        [
            "Speed data (SpeedLimit/MedianSpeed/F85) are all zero; excluded from scoring (data quality issue)."
            + _source_sentence(src)
            for src in all_source
        ],
        index=gdf.index,
        dtype=object,
    )
    valid_explanation = [
        _explain(mag, level, conf, src)
        for mag, level, conf, src in zip(
            valid["misalignment_magnitude"], valid["exposure_level"], confidence_level, source
        )
    ]

    gdf["confidence_level"] = pd.NA
    gdf.loc[valid_mask, "confidence_level"] = confidence_level.values
    # The three terms of the score, each in [0, 1], so a reader can see how much
    # of safety_score each axis contributed (weight x term x 100).
    # confidence_norm is also needed because confidence_level alone no longer
    # explains the confidence term once the TomTom bonus can move it.
    for col, values in (("misalignment_norm", misalignment_norm),
                        ("exposure_norm", exposure_norm),
                        ("confidence_norm", confidence_norm)):
        gdf[col] = pd.NA
        gdf.loc[valid_mask, col] = values.values
        gdf[col] = pd.to_numeric(gdf[col])
    gdf["safety_score"] = pd.NA
    gdf.loc[valid_mask, "safety_score"] = score.values
    gdf["safety_score"] = pd.to_numeric(gdf["safety_score"])
    priority_class.loc[valid_mask] = valid_class.values
    gdf["priority_class"] = pd.Categorical(
        priority_class, categories=PRIORITY_CLASSES_ORDERED + [DATA_QUALITY_CLASS], ordered=True
    )
    explanation.loc[valid_mask] = valid_explanation
    gdf["score_explanation"] = explanation

    gdf["exposure_level_code"] = gdf["exposure_level"].astype(str).map(EXPOSURE_LEVEL_CODES)
    gdf["confidence_level_code"] = pd.NA
    gdf.loc[valid_mask, "confidence_level_code"] = confidence_level.map(CONFIDENCE_LEVEL_CODES).values
    gdf["confidence_level_code"] = pd.to_numeric(gdf["confidence_level_code"])
    gdf["speed_data_source_code"] = all_source.map(SPEED_DATA_SOURCE_CODES)

    return gdf, thresholds


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from build_v_safe import build

    target, _, thresholds = build()  # build() already runs add_safety_score as part of the pipeline
    valid = target[target["data_quality_flag"].isna()]

    print("priority class score thresholds (per country, land_use):")
    for cell, cell_thresholds in thresholds.items():
        print(f"  {cell}: {{{', '.join(f'{k}: {round(v, 1)}' for k, v in cell_thresholds.items())}}}")
    print()
    print("priority_class distribution by country, land_use:")
    print(valid.groupby(["country", "land_use"])["priority_class"].value_counts().unstack().reindex(columns=PRIORITY_CLASSES_ORDERED))
    print()
    print("=== sanity: top class composition ===")
    top = valid[valid["priority_class"] == "Top Priority"]
    print(f"n={len(top)}")
    print("by country:\n", top["country"].value_counts())
    print("exposure_level in top class:\n", top["exposure_level"].value_counts())
    print("motorway segments in top class:", (top["road_class"] == "motorway").sum())
    print("low-exposure segments in top class:", (top["exposure_level"] == "low").sum())
    print()
    print("=== sample explanations ===")
    for s in top["score_explanation"].head(3):
        print(" -", s)
