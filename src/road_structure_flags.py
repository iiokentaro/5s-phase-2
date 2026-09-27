"""Turn a segment-to-way attribution into the road-structure flags.

segment_way_match matches segments to ways, this module turns the match into
flags, and road_separation runs both.

segment_way_match attributes metres, so each way carries the share of the
segment it explains, and every flag is a length-weighted threshold on that
evidence. The safety-side rule is:

    raising V_safe above 30 km/h requires evidence covering the segment;
    leaving it at 30 km/h requires none.

So `is_access_controlled` and `is_divided` are True only when a high share of
the matched length says so, and unmatched length never counts as agreement.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from segment_way_match import load_rules, required

# Way-level predicates stay in road_separation: they are statements about OSM
# tags, unchanged by how segments are matched, and road_separation's tests
# already pin them.
FLAG_COLUMNS = [
    "is_access_controlled",
    "is_divided",
    "is_grade_separated",
    "access_control_basis",
    "road_structure_confidence",
]

QUALITY_COLUMNS = [
    "osm_match_exact_frac",
    "osm_match_tol_frac",
    "osm_match_unmatched_frac",
    "osm_way_count",
    "osm_divided_frac",
    "osm_access_controlled_frac",
    "osm_grade_separated_frac",
]


def empty_flags(index, include_legacy: bool = False) -> pd.DataFrame:
    """Every flag defaults to the reading that cannot raise V_safe. A segment
    with no OSM evidence is 'structure not confirmed', never 'confirmed
    absent' -- the two are indistinguishable in the data and only one of them
    is safe to act on."""
    out = pd.DataFrame(index=index)
    out["is_access_controlled"] = False
    out["is_divided"] = False
    out["is_grade_separated"] = False
    out["access_control_basis"] = pd.Series(pd.NA, index=index, dtype="object")
    out["road_structure_confidence"] = "low"
    for col in QUALITY_COLUMNS:
        out[col] = 0 if col == "osm_way_count" else 0.0
    # Nothing was explained, so ALL of the row is unexplained. Defaulting this
    # to 0.0 alongside the others reads as "fully accounted for", which is the
    # opposite of what an unmatched row means.
    out["osm_match_unmatched_frac"] = 1.0
    if include_legacy:
        out["is_separated_legacy"] = False
    return out


def derive_flags(row_matches: pd.DataFrame, ways: pd.DataFrame, row_len_m: np.ndarray,
                 index, *, include_legacy: bool = False,
                 rules: dict | None = None) -> pd.DataFrame:
    """Aggregate one country's attribution into per-row flags.

    `row_matches` carries `row_index` positional into `index`.
    `row_len_m` is each row's own length, so the share of a row left
    unexplained is measured directly.
    """
    rules = rules or load_rules()
    min_divided = required(rules, "divided_min_length_frac")
    min_high = required(rules, "confidence_high_min_matched_frac")

    out = empty_flags(index, include_legacy)
    if not len(row_matches):
        return out

    way_flags = ways.set_index("osm_way_id")
    joined = row_matches.join(
        way_flags[["is_access_controlled_way", "is_divided_way", "is_grade_separated_way",
                   "access_control_basis_way", "is_separated_way_legacy"]],
        on="osm_way_id", how="left")

    length = joined["matched_len_m"].to_numpy()
    rows = joined["row_index"].to_numpy()
    n = len(index)

    matched = np.bincount(rows, weights=length, minlength=n)
    exact = np.bincount(rows, weights=length * (joined["match_kind"] == "exact"), minlength=n)
    tol = matched - exact
    total = np.asarray(row_len_m, dtype=float)

    def share(mask) -> np.ndarray:
        """Fraction OF THE MATCHED LENGTH that satisfies a way-level flag.
        Denominator is matched, not total: an unexplained stretch is neither
        agreement nor disagreement, and `matched_frac` reports it separately."""
        hit = np.bincount(rows, weights=length * np.asarray(mask, dtype=float), minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(matched > 0, hit / matched, 0.0)

    access_frac = share(joined["is_access_controlled_way"].fillna(False))
    divided_frac = share(joined["is_divided_way"].fillna(False))
    grade_frac = share(joined["is_grade_separated_way"].fillna(False))

    with np.errstate(invalid="ignore", divide="ignore"):
        matched_frac = np.where(total > 0, matched / total, 0.0)
        out["osm_match_exact_frac"] = np.where(total > 0, exact / total, 0.0)
        out["osm_match_tol_frac"] = np.where(total > 0, tol / total, 0.0)
    out["osm_match_unmatched_frac"] = np.clip(1.0 - matched_frac, 0.0, 1.0)
    out["osm_way_count"] = (joined.groupby("row_index")["osm_way_id"].nunique()
                            .reindex(range(n), fill_value=0).to_numpy())
    out["osm_access_controlled_frac"] = access_frac
    out["osm_divided_frac"] = divided_frac
    out["osm_grade_separated_frac"] = grade_frac

    # Confidence is how much of the row OSM accounts for. It stays two-valued
    # because safety_score._confidence_level's cut points are calibrated for
    # exactly three two-valued columns.
    out["road_structure_confidence"] = np.where(matched_frac >= min_high, "high", "low")

    # Access control is all-or-nothing: a stretch pedestrians may enter makes
    # the whole row enterable. Unanimity is over matched LENGTH, so a 12 m stub
    # cannot outvote 3 km of motorway.
    out["is_access_controlled"] = access_frac >= 1.0 - 1e-9
    out["is_divided"] = divided_frac >= min_divided
    # `.any()` on purpose. is_grade_separated EXEMPTS a row from
    # junction_speed_cap's 50 km/h cap, so loosening it raises V_safe; README
    # records that it over-matches (about a quarter of rows in each country).
    # osm_grade_separated_frac is the evidence to tighten it with, and doing so
    # is a V_safe decision.
    out["is_grade_separated"] = grade_frac > 0.0

    basis = _access_basis(joined, out["is_access_controlled"].to_numpy(), n, index)
    out["access_control_basis"] = basis

    if include_legacy:
        out["is_separated_legacy"] = share(joined["is_separated_way_legacy"].fillna(False)) >= min_divided
    return out


def _access_basis(joined: pd.DataFrame, is_access: np.ndarray, n: int, index) -> pd.Series:
    """`osm_motorroad` wins a mixed row, matching road_separation's rule: only
    a full motorway is exempt from the junction cap, so the weaker claim has
    to be the one recorded."""
    out = pd.Series(pd.NA, index=index, dtype="object")
    rows = np.flatnonzero(is_access)
    if not len(rows):
        return out
    sub = joined[joined["row_index"].isin(rows) & joined["access_control_basis_way"].notna()]
    if not len(sub):
        return out
    agg = sub.groupby("row_index")["access_control_basis_way"].agg(
        lambda s: "osm_motorroad" if "osm_motorroad" in set(s) else "osm_motorway")
    out.iloc[agg.index.to_numpy()] = agg.to_numpy()
    return out
