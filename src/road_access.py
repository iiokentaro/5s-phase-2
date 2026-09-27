"""Per-mode legal access for every OSM way, from the .osm.pbf alone.

What this answers
-----------------
For each `highway=*` way in a country extract, may a motor vehicle / motorcycle /
bicycle / pedestrian *legally* use it, as far as OSM's tags say. Four modes, four
values each (`yes` / `no` / `restricted` / `unknown`), plus the basis and a
confidence for every one.

Motorcycle is its own mode, not folded into motor_vehicle: two-wheelers are about
half of road deaths in both study countries and behave differently from cars --
banned on Indian Expressways and Thai motorways, ridden on `path` and `track`.
OSRM's stock car/bicycle/foot profiles cannot express that at all; Valhalla can,
via its own `kMotorcycleAccess` bit, which is the precedent followed here.

★ legal_* is NOT an exposure statement ★
  `legal_foot == 'no'` means pedestrians are prohibited, never that they are
  absent. Pedestrians are present at toll plazas, interchanges and severed
  settlements on access-controlled roads in both countries. Presence is the
  exposure signals' question (exposure_signals.py); presence coefficients
  estimated from these tags would bury an unvalidated guess inside a
  deliverable.

★ The basis distribution is the point ★
  OSM's mode-specific access tags are sparse in both countries, so most of this
  output is decided by src/road_access_rules.json, with no tag on the way
  speaking to it. `basis_report` measures exactly that: per country, per highway class, per
  mode, how much network km was decided by an explicit tag versus by a rule we
  wrote. A low `tag_evidenced_pct_of_km` is the finding, not a defect -- it says
  where the estimate is load-bearing, and it is the sampling frame for validating
  the weakest rules next.

★ Confidence means evidence, not plausibility ★
  `low` means no tag on the way spoke to this mode. `residential -> foot=yes` is
  `low` and almost certainly correct. The two must not be conflated.

Out of scope, stated so the gaps are visible:
  * `barrier=*` -- overwhelmingly a node tag, and the way extract carries no node
    data. An always-empty barrier column would imply coverage that does not exist.
  * `sidewalk=*` / `cycleway=*` -- structural evidence about separation, not legal
    permission. Emitted as `has_sidewalk` / `has_cycleway_infra`; folding them
    into legal_* would let `sidewalk=both` on a motorway flip pedestrian
    legality.
  * `*:conditional` -- surfaced as `has_conditional` / `conditional_keys`, never
    folded into the value. OSRM ignoring conditional restrictions by default is
    right for routing and wrong for analysis.

Output is keyed on `osm_way_id`; road_access_join.py carries it onto the ADB
segments.

Usage
-----
    python src/road_access.py --country maharashtra --json outputs/road_access_basis_maharashtra.json
    python src/road_access.py --country both --json outputs/road_access_basis.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS, PBF_PATHS

import osm_ways

warnings.filterwarnings("ignore", category=UserWarning)

RULES_PATH = "src/road_access_rules.json"
INTERIM_DIR = Path("data/interim")
OUT_PARQUET = "osm_access_{country}.parquet"

# pyrosm promotes these out of the catch-all `tags` dict into their own columns,
# so they must be merged back before any tag lookup. See _merged_tags.
PROMOTED_KEYS = ("highway", "oneway")

SIDEWALK_KEYS = ("sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both")
CYCLEWAY_KEYS = ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #

def load_rules(path: str = RULES_PATH) -> dict:
    """The rule table. utf-8-sig like elvik_2019.load_coefficients."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _public(d: dict) -> dict:
    """A rules sub-dict without its `_note` / `_why` prose keys."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# The resolver -- pure, pandas-free, one mode at a time
# --------------------------------------------------------------------------- #

def _norm(value):
    """A tag value from a parsed dict. NaN / None / '' all mean absent."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    value = str(value).strip()
    return value or None


def _default_for(highway, country: str, mode: str, rules: dict) -> tuple[str, str]:
    """(value, basis) from the highway table alone: `_base` overlaid by the
    country diff. The diff is per-mode, so two modes on the same way can carry
    different bases."""
    base = rules["defaults"]["_base"].get(highway)
    override = _public(rules["defaults"].get(country, {}).get(highway, {}))
    if mode in override:
        return override[mode], f"default:{country}"
    if base is None:
        return "unknown", "default:unknown_highway"
    if mode in base:
        return base[mode], "default:base"
    return "unknown", "default:incomplete_base"


def resolve_mode(tags: dict, country: str, mode: str, rules: dict,
                 unmapped: dict | None = None) -> tuple[str, str]:
    """(value, basis) for one mode on one way.

    Order is OSRM's and Valhalla's: the highway-class default first, then the
    access hierarchy walked general -> specific with every present key
    overwriting, so the most specific key ends up in control. The walk is never
    short-circuited, so `basis` names the key that genuinely decided it.

    `tags` must already have the promoted columns merged back in (_merged_tags).
    Unrecognised tag values resolve to 'unknown' and are counted into `unmapped`
    so a value worth adding to value_map surfaces in the report.
    """
    value, basis = _default_for(_norm(tags.get("highway")), country, mode, rules)
    value_map = rules["value_map"]
    for key in rules["access_hierarchy"][mode]:
        raw = _norm(tags.get(key))
        if raw is None:
            continue
        mapped = value_map.get(raw.lower())
        if mapped is None:
            if unmapped is not None:
                label = f"{key}={raw}"
                unmapped[label] = unmapped.get(label, 0) + 1
            mapped = "unknown"
        value, basis = mapped, f"tag:{key}"
    return value, basis


def confidence_for(value: str, basis: str, rules: dict) -> str:
    """Strength of the evidence, not plausibility of the answer.

    `low` means nothing on the way spoke to this mode, so the rules table decided
    it. `residential -> foot=yes` is `low` and almost certainly correct.
    """
    if value == "unknown" or basis.startswith("default:"):
        return "low"
    key = basis.split(":", 1)[1]
    return "high" if key in rules["mode_specific_keys"] else "medium"


def _merged_tags(tags_raw, highway, oneway) -> dict:
    """The way's tags as one dict, with pyrosm's promoted columns merged back.

    pyrosm MOVES a tag out of the catch-all `tags` dict into its own column once
    that key is named in `tags_as_columns` -- the same hazard
    road_separation._is_grade_separated_way documents. Reading only `tags` here
    would leave `highway` absent on every row, which would send every way down
    the `default:unknown_highway` branch: silently, and in the direction that
    makes the whole output worthless.
    """
    if isinstance(tags_raw, str):
        try:
            tags = json.loads(tags_raw)
        except ValueError:
            tags = {}
    elif isinstance(tags_raw, dict):
        tags = dict(tags_raw)
    else:
        tags = {}
    tags["highway"] = highway
    tags["oneway"] = oneway
    return tags


def _has_sidewalk(tags: dict, rules: dict) -> bool:
    present = rules["structural_keys"]["sidewalk_present_values"]
    return any((_norm(tags.get(k)) or "").lower() in present for k in SIDEWALK_KEYS)


def _has_cycleway_infra(tags: dict, rules: dict) -> bool:
    prefixes = tuple(rules["structural_keys"]["cycleway_present_prefixes"])
    for k in CYCLEWAY_KEYS:
        v = (_norm(tags.get(k)) or "").lower()
        if v and v.startswith(prefixes):
            return True
    return False


def _conditional_keys(tags: dict, access_keys: frozenset) -> str:
    """`;`-joined FULL tag names of the conditional variants present, e.g.
    'motor_vehicle:conditional'. The full name is kept so the value is directly
    greppable against OSM. Only conditionals on keys
    this module evaluates are reported; maxspeed:conditional and friends are not
    access statements."""
    hits = sorted(k for k in tags
                  if k.endswith(":conditional") and k[:-len(":conditional")] in access_keys)
    return ";".join(hits)


# --------------------------------------------------------------------------- #
# Loading the ways
# --------------------------------------------------------------------------- #

def _way_ids(ways: gpd.GeoDataFrame) -> np.ndarray:
    """The OSM way id, from whichever column the layer names it.

    Never falls back to a row counter: invented ids would join to nothing and
    look like real ones.
    """
    for col in ("osm_way_id", "id"):
        if col in ways.columns:
            return ways[col].to_numpy()
    raise KeyError(
        "the way layer carries neither `osm_way_id` nor `id`; refusing to "
        "number the rows, because the output is keyed on the OSM way id and a "
        "row counter would be indistinguishable from one")


def load_ways(country: str) -> gpd.GeoDataFrame:
    """Every `highway=*` way with full geometry and tags, via osm_ways.

    osm_ways.load_ways is checked against the .pbf's own declared counts on
    every build (osm_way_source.assert_conservation).

    Coverage note: the layer is clipped to the country's administrative
    boundary grown by osm_way_source.CLIP_MARGIN_M, so "every way" means every
    way in that area. This matters for Maharashtra, whose pbf is the wider
    western zone.
    """
    ways = osm_ways.load_ways(country)
    print(f"[load] {country}: {len(ways):,} ways from {osm_ways.ways_path(country)}")
    return ways


# --------------------------------------------------------------------------- #
# Annotation
# --------------------------------------------------------------------------- #

def annotate_access(ways: gpd.GeoDataFrame, country: str, rules: dict) -> pd.DataFrame:
    """One row per way: the four modes' value/basis/confidence, the structural
    and conditional flags, and the way's length.

    Single pass over numpy arrays: four modes over 2.9 million Thai ways, with
    the `tags` JSON parsed exactly once per way.
    """
    modes = rules["modes"]
    access_keys = frozenset(k for m in modes for k in rules["access_hierarchy"][m])

    t0 = time.time()
    length_m = ways.to_crs(BUFFER_CRS[country]).geometry.length.to_numpy()
    tags_raw = ways["tags"].to_numpy() if "tags" in ways.columns else np.full(len(ways), None)
    hw_col = ways["highway"].to_numpy() if "highway" in ways.columns else np.full(len(ways), None)
    ow_col = ways["oneway"].to_numpy() if "oneway" in ways.columns else np.full(len(ways), None)
    ids = _way_ids(ways)

    out: dict[str, list] = {"osm_way_id": [], "highway": []}
    for mode in modes:
        out[f"legal_{mode}"] = []
        out[f"legal_{mode}_basis"] = []
        out[f"legal_{mode}_confidence"] = []
    for col in ("has_sidewalk", "has_cycleway_infra", "has_conditional", "conditional_keys"):
        out[col] = []

    unmapped: dict[str, int] = {}
    for way_id, hw, ow, traw in zip(ids, hw_col, ow_col, tags_raw):
        tags = _merged_tags(traw, hw, ow)
        out["osm_way_id"].append(way_id)
        out["highway"].append(_norm(hw))
        for mode in modes:
            value, basis = resolve_mode(tags, country, mode, rules, unmapped)
            out[f"legal_{mode}"].append(value)
            out[f"legal_{mode}_basis"].append(basis)
            out[f"legal_{mode}_confidence"].append(confidence_for(value, basis, rules))
        out["has_sidewalk"].append(_has_sidewalk(tags, rules))
        out["has_cycleway_infra"].append(_has_cycleway_infra(tags, rules))
        cond = _conditional_keys(tags, access_keys)
        out["has_conditional"].append(bool(cond))
        out["conditional_keys"].append(cond)

    df = pd.DataFrame(out)
    df["length_m"] = length_m
    df.attrs["unmapped_tag_values"] = unmapped
    print(f"[resolve] {country}: {len(df):,} ways x {len(modes)} modes in "
          f"{time.time() - t0:.0f}s")
    return df


# --------------------------------------------------------------------------- #
# Step 1 report: the basis distribution
# --------------------------------------------------------------------------- #

def _km(series_len_m) -> float:
    return float(np.nansum(series_len_m)) / 1000.0


def _km_split(df: pd.DataFrame, col: str, total_km: float) -> dict:
    """km and % of km per distinct value of `col`, largest first."""
    grouped = df.groupby(col, dropna=False)["length_m"].sum() / 1000.0
    grouped = grouped.sort_values(ascending=False)
    return {str(k): {"km": round(float(v), 1),
                     "pct_of_km": round(100 * float(v) / total_km, 2) if total_km else 0.0}
            for k, v in grouped.items()}


def basis_report(df: pd.DataFrame, country: str, rules: dict) -> dict:
    """Everything km-weighted. `_km` is a local copy on purpose: importing
    road_class_coverage for it would drag in tomtom_enrichment and safe_speed."""
    modes = rules["modes"]
    total_km = _km(df["length_m"])
    report: dict = {
        "country": country,
        "n_ways": int(len(df)),
        "total_km": round(total_km, 1),
        "rules_schema_version": rules["schema_version"],
    }

    by_mode = {}
    for mode in modes:
        basis_col = f"legal_{mode}_basis"
        tag_km = _km(df.loc[df[basis_col].str.startswith("tag:"), "length_m"])
        by_mode[mode] = {
            # The headline number: how much of this mode's answer any tag on the
            # way actually evidenced. Expected to be low -- that IS the finding.
            "tag_evidenced_pct_of_km": round(100 * tag_km / total_km, 2) if total_km else 0.0,
            "value_km": _km_split(df, f"legal_{mode}", total_km),
            "basis_km": _km_split(df, basis_col, total_km),
            "confidence_km": _km_split(df, f"legal_{mode}_confidence", total_km),
        }
    report["by_mode"] = by_mode

    # The sampling frame for the next step: which (class, mode) cells rest on a
    # rule we wrote, ranked by km at stake.
    by_highway: dict = {}
    for hw, grp in df.groupby(df["highway"].astype(str)):
        hw_km = _km(grp["length_m"])
        entry = {"km": round(hw_km, 1), "ways": int(len(grp)),
                 "pct_of_km": round(100 * hw_km / total_km, 2) if total_km else 0.0,
                 "modes": {}}
        for mode in modes:
            entry["modes"][mode] = {
                "value_km": _km_split(grp, f"legal_{mode}", hw_km),
                "confidence_km": _km_split(grp, f"legal_{mode}_confidence", hw_km),
            }
        by_highway[hw] = entry
    report["by_highway"] = dict(sorted(by_highway.items(),
                                       key=lambda kv: -kv[1]["km"]))

    # Rows the rules table has no entry for: complete it from data, not guesswork.
    known = set(rules["defaults"]["_base"])
    unknown = df[~df["highway"].astype(str).isin(known)]
    unknown_km = _km(unknown["length_m"])
    report["unmapped_highway_values"] = {
        "km": round(unknown_km, 1),
        "pct_of_km": round(100 * unknown_km / total_km, 2) if total_km else 0.0,
        "by_value": _km_split(unknown, "highway", total_km) if len(unknown) else {},
    }

    report["unmapped_tag_values"] = dict(
        sorted(df.attrs.get("unmapped_tag_values", {}).items(), key=lambda kv: -kv[1])
    )

    # Each country diff's blast radius, so a hand-written rule is auditable.
    effect = {}
    for hw, override in rules["defaults"].get(country, {}).items():
        if hw.startswith("_"):
            continue
        base = rules["defaults"]["_base"].get(hw, {})
        for mode, value in _public(override).items():
            sel = ((df["highway"].astype(str) == hw)
                   & (df[f"legal_{mode}_basis"] == f"default:{country}"))
            effect[f"{hw}.{mode}"] = {
                "base_value": base.get(mode),
                "override_value": value,
                "changes_value": base.get(mode) != value,
                "km_decided_by_this_rule": round(_km(df.loc[sel, "length_m"]), 1),
            }
    report["country_override_effect"] = effect

    report["structural_flags_km"] = {
        "has_sidewalk": round(_km(df.loc[df["has_sidewalk"], "length_m"]), 1),
        "has_cycleway_infra": round(_km(df.loc[df["has_cycleway_infra"], "length_m"]), 1),
        "has_conditional": round(_km(df.loc[df["has_conditional"], "length_m"]), 1),
    }
    return report


def print_summary(report: dict, rules: dict) -> None:
    print(f"\n=== {report['country']}: {report['n_ways']:,} ways, "
          f"{report['total_km']:,.1f} km ===")
    for mode in rules["modes"]:
        m = report["by_mode"][mode]
        values = ", ".join(f"{k} {v['pct_of_km']}%" for k, v in m["value_km"].items())
        conf = ", ".join(f"{k} {v['pct_of_km']}%" for k, v in m["confidence_km"].items())
        print(f"  {mode:<14} tag-evidenced {m['tag_evidenced_pct_of_km']:>5.2f}% of km")
        print(f"  {'':<14}   value:      {values}")
        print(f"  {'':<14}   confidence: {conf}")
    um = report["unmapped_highway_values"]
    print(f"  unmapped highway values: {um['km']:,.1f} km ({um['pct_of_km']}%)")
    if report["unmapped_tag_values"]:
        top = list(report["unmapped_tag_values"].items())[:8]
        print(f"  unmapped tag values (top): {top}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def run_country(country: str, rules: dict, parquet_out: str | None) -> dict:
    df = annotate_access(load_ways(country), country, rules)
    if parquet_out:
        out_dir = Path(parquet_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / OUT_PARQUET.format(country=country)
        df.to_parquet(path, index=False)
        print(f"[write] {country}: {path}")
    report = basis_report(df, country, rules)
    print_summary(report, rules)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--country", choices=["thailand", "maharashtra", "both"], default="both")
    ap.add_argument("--rules", default=RULES_PATH)
    ap.add_argument("--parquet-out", default=str(INTERIM_DIR),
                    help="directory for the per-way parquet; '' to skip writing it")
    ap.add_argument("--json", default=None, help="also write the basis report here")
    args = ap.parse_args()

    rules = load_rules(args.rules)
    countries = ["thailand", "maharashtra"] if args.country == "both" else [args.country]
    reports = [run_country(c, rules, args.parquet_out or None) for c in countries]

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(reports, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
