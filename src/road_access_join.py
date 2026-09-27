"""Carry per-mode legal access from OSM ways onto the ADB segments.

`road_access.py` answers "may a motor vehicle / motorcycle / bicycle /
pedestrian legally use this OSM WAY". This module answers it for a ROW of the
deliverable, by length-weighting over the ways segment_way_match attributed to
that row.

★ `legal_foot == 'yes'` does NOT feed V_safe -- `legal_foot == 'no'` does ★
  Measured on the way layer:

      legal_foot == 'yes'     Thailand 96.83% of ways, Maharashtra 92.83%
      of which tag-evidenced  Thailand  3.30%,          Maharashtra  7.56%

  Pedestrians are *permitted* on essentially every road, and 97% of that
  verdict comes from src/road_access_rules.json's defaults. ORing it into
  `is_vru`, which caps V_safe at 30 km/h, would set the flag almost everywhere
  on the strength of a rule we wrote and make the result look like evidence.
  `is_vru` stays with the Mapillary detections.

  `has_sidewalk` is real structural evidence, but covers 0.17% of Thai ways
  and 0.13% of km, so on its own it moves almost nothing.

  `legal_foot == 'no'` is the informative signal, and `add_road_access` uses it
  to raise `is_access_controlled`. The gate has two parts, and the second one
  is what makes the first safe.

  ★ 1. The prohibition has to be evidenced ★
    `legal_foot_confidence` must be medium or high, i.e. a real tag on the way
    decided it (a `default:*` rule out of road_access_rules.json gives low).
    This is the same bar `legal_foot == 'yes'` fails: 96.83% of Thai ways
    permit pedestrians and only 3.30% of that is tag-evidenced.

  ★ 2. A cyclist is a vulnerable road user too ★
    `is_access_controlled` both raises V_safe and, through
    `exposure_signals.vru_mask`, forces `is_vru` False. Neither is defensible
    on a road where cyclists are still permitted, whatever the pedestrian tag
    says, so the condition is `legal_foot == 'no'` AND `legal_bicycle == 'no'`,
    both resolved safety-side (see RESOLUTION): any length of the row where a
    mode is permitted makes the row permitted.

    The bicycle verdict is taken at whatever confidence it carries, tag or
    rule. It is read as "is there any reason to think a cyclist may be here",
    where a permissive default is the cautious answer and a prohibition is the
    claim needing evidence -- and road_access_rules.json's `_base` permits
    bicycles on every class but motorway, so a `no` verdict on anything else
    can only have come from a tag.

    A pedestrian-only condition would raise 1,136 rows; 324 of them still
    permit a cyclist somewhere along their length and keep 30 km/h, leaving 812
    (809 Thailand, 3 Maharashtra).

  ★ Motorcycles are not part of this condition ★
    A motorcycle is a motor vehicle: whether a rider may be present says
    nothing about whether pedestrians and cyclists are separated from the
    carriageway, which is the only thing this flag claims. With motorcycles in
    the condition, 80 rows with tag-evidenced `foot=no` and `bicycle=no` were
    reported as not access-controlled.

    Whether a rider may be present is a speed question, and it is answered in
    safe_speed.classify_collision_type, which holds such rows at 30 km/h unless
    a barrier separates riders from four-wheeled traffic. `RESOLUTION["motorcycle"]` is `vru_safe_side` for that
    consumer: any stretch a rider may legally use makes the row one a rider may
    legally use.

  The basis recorded is `osm_vru_prohibited`, and safe_speed appends it to
  `v_safe_basis` (`separated:trunk:osm_vru_prohibited`), so these rows can be
  counted and reviewed apart from those raised by `highway=motorway`.

  It is NOT in junction_speed_cap.MOTORWAY_BASES: prohibiting pedestrians and
  cyclists says nothing about grade separation, so these rows stay subject to
  the 50 km/h junction cap. `is_divided` keeps its OSM value -- a prohibition
  says nothing about the carriageway -- so this basis alone yields 70 km/h,
  and reaches the 80-100 table only where the carriageway evidence
  independently says divided AND motorcycles are prohibited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ACCESS_DIR = Path("data/interim")
ACCESS_PARQUET = "osm_access_{country}.parquet"

MODES = ("motor_vehicle", "motorcycle", "bicycle", "foot")
VALUES = ("yes", "no", "restricted", "unknown")

# How a row's single value is picked from the mix along it.
#
#   vru_safe_side : any length where the mode is permitted makes the row
#                   permitted. Asking "might a pedestrian legally be here",
#                   a 200 m stretch of yes inside 2 km of no still means yes.
#   dominant      : the value covering the most length. `motor_vehicle` is
#                   diagnostic only, so the row reports what it mostly is.
#
# `motorcycle` is `vru_safe_side` because it decides the motorcycle rule on
# V_safe. Under a majority vote, 2 rows (overture_segment_id 15206 and 15207,
# Bangkok, highway=primary near 13.8243 N 100.4585 E) were raised to 80 km/h
# although part of each row -- 25.9 m of 160.1 m, and 71.9 m of 150.0 m --
# lies on a way where a motorcycle may be ridden.
RESOLUTION = {
    "foot": "vru_safe_side",
    "bicycle": "vru_safe_side",
    "motor_vehicle": "dominant",
    "motorcycle": "vru_safe_side",
}

CONFIDENCE_ORDER = ("low", "medium", "high")

# Per road_access_rules.json's confidence_rule: 'low' means the value came
# from a default:* rule. Medium/high mean a real tag on the way, which is the
# bar for feeding a V_safe input.
TAG_EVIDENCED = ("medium", "high")

# Every mode that must be prohibited before a row counts as access-controlled.
# `foot` additionally has to be tag-evidenced; see the module docstring.
#
# `is_access_controlled` means pedestrians and cyclists cannot legally enter
# (exposure_signals.vru_mask reads it for exactly that). Motorcycle access
# decides in safe_speed.classify_collision_type whether such a row can rise
# above 30 km/h.
VRU_MODES = ("foot", "bicycle")
VRU_PROHIBITED_BASIS = "osm_vru_prohibited"


def access_path(country: str) -> Path:
    return ACCESS_DIR / ACCESS_PARQUET.format(country=country)


def load_access(country: str) -> pd.DataFrame:
    path = access_path(country)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it with:\n"
            f"    python src/road_access.py --country {country}")
    return pd.read_parquet(path)


def access_columns() -> list[str]:
    cols = []
    for mode in MODES:
        cols += [f"legal_{mode}", f"legal_{mode}_confidence"]
        cols += [f"legal_{mode}_{v}_frac" for v in VALUES]
    return cols + ["has_sidewalk", "has_cycleway_infra", "access_matched_frac"]


def empty_access(index) -> pd.DataFrame:
    """A row with no attributed way knows nothing about access, which is
    'unknown' -- distinct from 'no', and distinct from the permissive default
    a missing value would otherwise inherit downstream."""
    out = pd.DataFrame(index=index)
    for mode in MODES:
        out[f"legal_{mode}"] = "unknown"
        out[f"legal_{mode}_confidence"] = "low"
        for v in VALUES:
            out[f"legal_{mode}_{v}_frac"] = 0.0
    out["has_sidewalk"] = False
    out["has_cycleway_infra"] = False
    out["access_matched_frac"] = 0.0
    return out


def derive_access(row_matches: pd.DataFrame, access: pd.DataFrame,
                  row_len_m: np.ndarray, index) -> pd.DataFrame:
    """Length-weighted per-mode access for one country's rows."""
    out = empty_access(index)
    if not len(row_matches):
        return out

    joined = row_matches.join(access.set_index("osm_way_id"), on="osm_way_id", how="left")
    length = joined["matched_len_m"].to_numpy(dtype=float)
    rows = joined["row_index"].to_numpy()
    n = len(index)

    matched = np.bincount(rows, weights=length, minlength=n)
    total = np.asarray(row_len_m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["access_matched_frac"] = np.where(total > 0, matched / total, 0.0)

    def frac(mask: np.ndarray) -> np.ndarray:
        hit = np.bincount(rows, weights=length * mask.astype(float), minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(matched > 0, hit / matched, 0.0)

    for mode in MODES:
        col = joined[f"legal_{mode}"]
        fracs = {v: frac((col == v).to_numpy()) for v in VALUES}
        for v in VALUES:
            out[f"legal_{mode}_{v}_frac"] = fracs[v]

        if RESOLUTION[mode] == "vru_safe_side":
            value = np.where(fracs["yes"] > 0.0, "yes",
                     np.where(fracs["restricted"] > 0.0, "restricted",
                      np.where(fracs["no"] > 0.0, "no", "unknown")))
        else:
            stacked = np.vstack([fracs[v] for v in VALUES])
            value = np.array(VALUES)[stacked.argmax(axis=0)]
            value = np.where(matched > 0, value, "unknown")
        out[f"legal_{mode}"] = np.where(matched > 0, value, "unknown")

        # The worst rung any constituent way carries. A row is not better
        # evidenced than its weakest part.
        rung = joined[f"legal_{mode}_confidence"].map(
            {c: i for i, c in enumerate(CONFIDENCE_ORDER)}).fillna(0).to_numpy()
        worst = pd.Series(rung, index=rows).groupby(level=0).min()
        worst = worst.reindex(range(n)).fillna(0).astype(int).to_numpy()
        out[f"legal_{mode}_confidence"] = np.array(CONFIDENCE_ORDER)[worst]

    for col in ("has_sidewalk", "has_cycleway_infra"):
        out[col] = frac(joined[col].fillna(False).to_numpy()) > 0.0
    return out


def add_road_access(gdf):
    """The `road_access_join` pipeline step.

    Reuses road_separation's memoised parent match, so the attribution runs
    once per country per build.
    """
    import road_separation

    gdf = gdf.copy()
    base = empty_access(gdf.index)
    for col in base.columns:
        gdf[col] = base[col]

    if "is_access_controlled" not in gdf.columns:
        gdf["is_access_controlled"] = False
    if "access_control_basis" not in gdf.columns:
        gdf["access_control_basis"] = pd.Series(pd.NA, index=gdf.index, dtype="object")

    for country in gdf["country"].unique():
        mask = (gdf["country"] == country).to_numpy()
        rows = gdf.loc[mask]
        row_matches, _ways, row_len = road_separation._country_matches(country, rows)
        derived = derive_access(row_matches, load_access(country), row_len, rows.index)
        for col in derived.columns:
            gdf.loc[mask, col] = derived[col].to_numpy()

        # Access control means neither a pedestrian nor a cyclist may be here,
        # so both have to be prohibited, not the pedestrian alone. Motorcycles
        # are handled in safe_speed; see VRU_MODES above and the docstring for
        # why, and for why is_divided is left untouched.
        prohibited = np.ones(len(derived), dtype=bool)
        for mode in VRU_MODES:
            prohibited &= derived[f"legal_{mode}"].to_numpy() == "no"
        # The pedestrian clause additionally has to rest on a tag.
        prohibited &= derived["legal_foot_confidence"].isin(TAG_EVIDENCED).to_numpy()

        not_yet_controlled = ~gdf.loc[mask, "is_access_controlled"].fillna(False).to_numpy()
        newly = prohibited & not_yet_controlled
        if newly.any():
            idx = rows.index[newly]
            gdf.loc[idx, "is_access_controlled"] = True
            gdf.loc[idx, "access_control_basis"] = VRU_PROHIBITED_BASIS
    return gdf
