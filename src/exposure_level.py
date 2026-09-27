"""Discretize VRU exposure into high/medium/low, separately for urban and
rural systems, then apply the rural safety margin.

★ `exposure_level` is used only for prioritisation ★
It feeds safety_score.py's exposure axis (weight 0.35) and priority_lists.py's
rank_within_environment, and never sets V_safe. V_safe (safe_speed.py) reads
`is_vru` and the road-structure flags (`is_access_controlled`, `is_divided`),
and poi_speed_zones.py applies the POI zones. `is_mapillary_vru` is an URBAN
signal here as well as the main part of `is_vru`, and a school is counted in
`osm_poi_category_count` as well as capping V_safe through its zone, so both
contribute to both axes (the double count is intended, see README.md).

★ Why percentile-rank-then-average, not a raw sum ★
pop_density and poi_count are on different scales and both heavily
right-skewed; summing raw values lets whichever signal has the larger
numbers dominate. Each signal is converted to its percentile rank *within
its own land_use system* first (so urban and rural never share a scale),
then averaged. Every signal then raises the composite monotonically (more
population/POIs/crossings never lowers exposure), and urban's rich signal set
is kept off the axis of rural's sparse one.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

URBAN_SIGNALS = [
    "pop_density",
    "poi_count",
    "osm_poi_category_count",
    "is_mapillary_vru",
    "crossing_count",
]
RURAL_SIGNALS = ["pop_density", "poi_count", "osm_poi_category_count"]

LEVELS = ["low", "medium", "high"]


def _percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def _composite_and_level(gdf, mask, signals):
    sub = gdf.loc[mask]
    ranks = pd.concat([_percentile_rank(sub[s]) for s in signals], axis=1)
    composite = ranks.mean(axis=1)
    # qcut with duplicate edges dropped can collapse to <3 bins on heavily
    # tied data (e.g. many rural segments at poi_count==0); fall back to
    # rank-based tertiles which always produce 3 groups.
    try:
        level = pd.qcut(composite, q=3, labels=LEVELS)
    except ValueError:
        level = pd.qcut(composite.rank(method="first"), q=3, labels=LEVELS)
    return composite, level


def add_exposure_level(gdf):
    """Percentile rank + tertile split is computed within each of the four
    (country, land_use) cells. Thailand and Maharashtra's pop_density /
    poi_count distributions differ enough (see apply_rural_safety_margin's
    per-country threshold below) that a tertile pooled across countries would
    let one country's larger raw numbers dominate the other's percentile
    ranks."""
    gdf = gdf.copy()
    gdf["exposure_composite"] = np.nan
    gdf["exposure_level"] = pd.Categorical([None] * len(gdf), categories=LEVELS, ordered=True)

    for land_use, signals in [("URBAN", URBAN_SIGNALS), ("RURAL", RURAL_SIGNALS)]:
        for country in gdf["country"].unique():
            mask = (gdf["land_use"] == land_use) & (gdf["country"] == country)
            if not mask.any():
                continue
            composite, level = _composite_and_level(gdf, mask, signals)
            gdf.loc[mask, "exposure_composite"] = composite
            gdf.loc[mask, "exposure_level"] = level.values

    return gdf


def apply_rural_safety_margin(gdf, quantile=0.75):
    """Rural segments with substantial roadside population but no detected
    crossing structure (true for nearly all rural segments) are
    "exposure unknown, potentially high risk" -- not "low exposure". Raise
    them to at least medium, mark confidence low.

    Threshold is the 75th percentile of *that country's rural* pop_density.
    Thailand and Maharashtra rural pop_density distributions differ enough
    (medians 4.9 vs 13.8) that a threshold shared across countries uplifted
    51% of Maharashtra's rural segments but only 11% of Thailand's. Returns
    thresholds as a dict keyed by country for inspection.
    """
    gdf = gdf.copy()
    gdf["exposure_confidence"] = "high"

    rural_mask = gdf["land_use"] == "RURAL"
    thresholds = {}

    for country in gdf.loc[rural_mask, "country"].unique():
        country_rural_mask = rural_mask & (gdf["country"] == country)
        threshold = gdf.loc[country_rural_mask, "pop_density"].quantile(quantile)
        thresholds[country] = threshold

        uncertain_mask = country_rural_mask & (gdf["pop_density"] >= threshold) & (~gdf["has_crossing"])
        gdf.loc[uncertain_mask, "exposure_confidence"] = "low"
        levels_as_int = gdf["exposure_level"].cat.codes
        medium_code = LEVELS.index("medium")
        needs_raise = uncertain_mask & (levels_as_int < medium_code)
        gdf.loc[needs_raise, "exposure_level"] = "medium"

    return gdf, thresholds


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from exposure_signals import add_crossing_signal, add_poi_proximity
    from pop_density import add_pop_density
    from road_separation import add_road_structure
    from schema import load_target

    target = load_target()
    target = add_road_structure(target)  # add_poi_proximity needs is_access_controlled to mask is_vru/is_mapillary_vru
    target = add_pop_density(target)
    target = add_poi_proximity(target)
    target = add_crossing_signal(target)
    target = add_exposure_level(target)
    target, threshold = apply_rural_safety_margin(target)

    print("rural pop_density safety-margin threshold:", threshold)
    print()
    print(target.groupby(["country", "land_use"])["exposure_level"].value_counts())
    print()
    print("exposure_confidence by land_use:")
    print(target.groupby("land_use")["exposure_confidence"].value_counts())
