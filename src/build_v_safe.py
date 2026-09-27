"""End-to-end pipeline: connect exposure signals and V_safe, write
data/processed/segments_v_safe.parquet, and run sanity checks that do not
use speed_limit to validate V_safe itself.
"""

import argparse
import json
import os
import sys
import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "src")
from aadt_estimation import add_aadt_column
from exposure_level import add_exposure_level, apply_rural_safety_margin
from exposure_signals import add_crossing_signal, add_poi_proximity
from elvik_2019 import add_exponential_reduction
from junction_speed_cap import add_junction_speed_cap
from misalignment import add_misalignment
from pipeline_order import (GEOMETRY_SIGNALS, LOAD_TARGET, OBJECT_SEPARATION, POI_ZONES, POST_REFINE,
                            REFINE, SANDWICH, TOMTOM)
from poi_speed_zones import REASON_COLS, apply_poi_speed_zones
from poi_params import DEFAULT as DEFAULT_POI_PARAMS
from poi_params import PoiParams
from pop_density import add_pop_density
from priority_lists import add_priority_environment_rank, write_priority_environment_lists
from priority_map import build_priority_map, plot_static_summary, write_gdb_zip, write_geo_outputs, write_gpkg
from review_track import add_review_track, write_lists
from road_access_join import add_road_access
from road_separation import add_road_structure
from safe_speed import _bool_col, add_v_safe
from segment_detected_objects import apply_object_separation
from safety_score import add_safety_score
from sandwich_segments import smooth_sandwich_segments
from schema import load_target
from segment_localization import refine_influenced_segments
from speedlimit_plausibility import add_speedlimit_plausibility
from tomtom_enrichment import add_tomtom_enrichment

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_PATH = "data/processed/segments_v_safe.parquet"
# The POI parameters the committed parquet was built with, beside it.
POI_PARAMS_PATH = "data/processed/segments_v_safe_poi_params.json"


def _legacy(state) -> bool:
    """Whether this build also computes the pre-split comparison columns.

    Read from the state dict build() already threads through every geometry
    step, so the switch reaches the four functions that need it without a
    module-level global -- and so the legacy columns are recomputed on the
    Stage 2 child geometries by the very same call that computes the current
    ones (segment_localization._make_child copies the parent's values, and only
    a re-run overwrites them).
    """
    return bool(state.get("include_legacy", False))


def _poi_params(state) -> PoiParams:
    return state.get("poi_params", DEFAULT_POI_PARAMS)


def _rural_margin(target, state):
    target, state["rural_thresholds"] = apply_rural_safety_margin(target)
    return target


def _safety_score(target, state):
    target, state["score_thresholds"] = add_safety_score(target)
    return target


# Execution order and labels live in pipeline_order.py (no imports, so the web
# server can read them without pulling geopandas into its process); this module
# owns the mapping from those ids to the actual functions. The asserts below fail
# at import if the two ever disagree, so the GUI's progress chain always shows
# what build() really runs.
_GEOMETRY_FNS = {
    "road_structure": lambda t, s: add_road_structure(t, include_legacy=_legacy(s)),
    "road_access_join": lambda t, s: add_road_access(t),
    "pop_density": lambda t, s: add_pop_density(t),
    "poi_proximity": lambda t, s: add_poi_proximity(t, include_legacy=_legacy(s), params=_poi_params(s)),
    "crossing_signal": lambda t, s: add_crossing_signal(t),
    "exposure_level": lambda t, s: add_exposure_level(t),
    "rural_margin": _rural_margin,
    "v_safe": lambda t, s: add_v_safe(t, include_legacy=_legacy(s)),
    "junction_cap": lambda t, s: add_junction_speed_cap(t, include_legacy=_legacy(s)),
}

_POST_REFINE_FNS = {
    "speedlimit_plausibility": lambda t, s: add_speedlimit_plausibility(t),
    "misalignment": lambda t, s: add_misalignment(t),
    "safety_score": _safety_score,
    "review_track": lambda t, s: add_review_track(t),
    "priority_rank": lambda t, s: add_priority_environment_rank(t),
    "exp_fatal_reduction": lambda t, s: add_exponential_reduction(t),
}

assert {sid for sid, _, _ in GEOMETRY_SIGNALS} == set(_GEOMETRY_FNS)
assert {sid for sid, _, _ in POST_REFINE} == set(_POST_REFINE_FNS)

GEOMETRY_SIGNAL_STEPS = [(sid, en, ja, _GEOMETRY_FNS[sid]) for sid, en, ja in GEOMETRY_SIGNALS]
POST_REFINE_STEPS = [(sid, en, ja, _POST_REFINE_FNS[sid]) for sid, en, ja in POST_REFINE]


@contextmanager
def _span(progress, step_id, label_en, label_ja, group=None):
    """No-op when progress is None, so the CLI path is unchanged."""
    if progress is None:
        yield
        return
    with progress(step_id, label_en, label_ja, group):
        yield


def _apply_geometry_signals(target, state, progress=None, group=None):
    """Run all geometry-driven signal functions in order.

    Called twice in build(): once for Stage 1 (whole segments) and once for
    Stage 2 (after refine_influenced_segments splits influenced segments). Running
    the same functions on child geometries ensures that influenced children get the
    lower V_safe and non-influenced children retain the original higher V_safe.
    Non-split rows produce identical values on the second pass (same geometry).
    """
    for step_id, label_en, label_ja, fn in GEOMETRY_SIGNAL_STEPS:
        with _span(progress, step_id, label_en, label_ja, group):
            target = fn(target, state)
    return target


def build(progress=None, include_legacy=None,
          poi_params: PoiParams | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """`include_legacy=True` adds the pre-split rule's `*_legacy` columns
    alongside the current ones (see safe_speed.classify_collision_type_legacy).
    Defaults to the V_SAFE_INCLUDE_LEGACY=1 environment variable, which is how
    the browser GUI reaches it -- pipeline_runner has no flag of its own for it.

    `poi_params` (poi_params.PoiParams) sets the Overture confidence rule and
    which POI types cap V_safe at what speed; None means the defaults."""
    if include_legacy is None:
        include_legacy = os.environ.get("V_SAFE_INCLUDE_LEGACY", "") == "1"
    state = {"include_legacy": bool(include_legacy), "poi_params": poi_params or DEFAULT_POI_PARAMS}

    with _span(progress, *LOAD_TARGET):
        target = load_target()

    # Optional TomTom layer: cut into uninterrupted segments where TomTom
    # matches, attach its values and backfill unusable speed fields, BEFORE
    # stage 1, so the population and the geometry granularity are both settled
    # before the first signal runs. A no-op for a country without the layer.
    with _span(progress, *TOMTOM):
        target = add_tomtom_enrichment(target)

    # Stage 1: whole-segment signal computation (is_vru, near_junction, v_safe).
    # road_structure runs ahead of add_poi_proximity: it doesn't depend on
    # pop_density/POI/exposure columns, and add_poi_proximity needs its
    # is_access_controlled to mask is_vru (and is_mapillary_vru).
    target = _apply_geometry_signals(target, state, progress, group="stage1")

    # Stage 2: clip influenced segments at their influence-zone boundary, then re-run
    # the same geometry-driven functions so each child segment gets its own V_safe.
    with _span(progress, *REFINE):
        target = refine_influenced_segments(target, _poi_params(state))
    target = _apply_geometry_signals(target, state, progress, group="stage2")

    # POI zones and sandwich segments act on the final geometry and write
    # V_safe directly. The geometry signals are not re-run on their pieces:
    # a re-run would recompute V_safe without the POI zones.
    with _span(progress, *POI_ZONES):
        target = apply_poi_speed_zones(target, _poi_params(state))
    with _span(progress, *SANDWICH):
        max_length_m = _poi_params(state).sandwich_max_length_m
        target, passes = smooth_sandwich_segments(
            target, max_length_m=max_length_m, parent_id_col="poi_parent_id")
        state["sandwich_passes"] = passes
        print(f"[sandwich] converged after {passes} iterations (max length {max_length_m:g} m)")
    with _span(progress, *OBJECT_SEPARATION):
        target = apply_object_separation(target)

    for step_id, label_en, label_ja, fn in POST_REFINE_STEPS:
        with _span(progress, step_id, label_en, label_ja):
            target = fn(target, state)

    return target, state["rural_thresholds"], state["score_thresholds"]


def write_parquet(target: pd.DataFrame, poi_params: PoiParams) -> None:
    target.to_parquet(OUTPUT_PATH)
    with open(POI_PARAMS_PATH, "w") as f:
        f.write(json.dumps(poi_params.to_dict(), indent=2) + "\n")


def sanity_checks(gdf: pd.DataFrame) -> None:
    print("=== v_safe by road_class x land_use (no speed_limit involved) ===")
    print(gdf.groupby(["road_class", "land_use"])["v_safe"].median().unstack())

    print("\n=== road structure: is_access_controlled x is_divided (independent flags) ===")
    print(gdf.groupby(["country", "is_access_controlled", "is_divided"]).size())
    print(gdf.groupby("country")["access_control_basis"].value_counts(dropna=False))
    print("\n=== v_safe_basis ===")
    print(gdf["v_safe_basis"].value_counts())

    print("\n=== data_quality_flag: retained but excluded from misalignment scoring ===")
    invalid = gdf["data_quality_flag"] == "invalid_speed"
    print(f"{invalid.sum()} / {len(gdf)} segments flagged invalid_speed "
          f"(speed_limit/median_speed/f85_speed all exactly 0)")
    print(gdf.loc[invalid, "road_class"].value_counts())

    print("\n=== diagnostic only: v_safe vs f85_speed (not used to change v_safe; excludes invalid_speed) ===")
    usable = gdf[~invalid]
    over = (usable["f85_speed"] > usable["v_safe"]).sum()
    print(f"segments where observed F85 > v_safe (operating speed exceeds safe speed): "
          f"{over} / {len(usable)} ({over / len(usable):.1%})")

    print("\n=== speed data provenance (optional TomTom layer) ===")
    source = gdf["speed_data_source"] if "speed_data_source" in gdf.columns else None
    if source is None:
        print("speed_data_source absent -- this parquet predates the TomTom layer")
    else:
        print(source.value_counts().to_dict())
        for country in sorted(gdf["country"].unique()):
            rows = gdf["country"] == country
            print(f"{country} rows corroborated by TomTom: "
                  f"{(source[rows] == 'tomtom').sum()} / {rows.sum()} "
                  f"({(source[rows] == 'tomtom').mean():.1%})")
        if "speed_limit_source" in gdf.columns:
            print(f"speed limits taken from TomTom: {int((gdf['speed_limit_source'] == 'tomtom').sum())}")
        for col, label in (("speed_limit_backfilled", "speed limits"),
                           ("observed_speed_backfilled", "observed speeds")):
            if col in gdf.columns:
                print(f"{label} backfilled from TomTom (ADB had none): {int(gdf[col].sum())}")
        if "exp_speed_source" in gdf.columns:
            print(f"Elvik (2019) mean/SD from TomTom: {int((gdf['exp_speed_source'] == 'tomtom').sum())}")

    print("\n=== basis / confidence flags present ===")
    for col in ["v_safe_basis", "exposure_confidence", "speedlimit_plausibility", "road_structure_confidence"]:
        print(f"{col}: {gdf[col].notna().mean():.1%} non-null")

    print("\n=== misalignment (main axis, SpeedLimit - V_safe) ===")
    too_high = usable["misalignment"] > 0
    print(f"too-high direction (SpeedLimit > V_safe, review priority): {too_high.sum()} / {len(usable)} "
          f"({too_high.mean():.1%})")
    print("by road_class:")
    print(usable.groupby("road_class")["misalignment"].apply(lambda s: (s > 0).mean()))
    print("by land_use:")
    print(usable.groupby("land_use")["misalignment"].apply(lambda s: (s > 0).mean()))

    print("\n=== axis agreement: top decile by misalignment_magnitude vs by operating_gap ===")
    top_n = max(1, len(usable) // 10)
    top_misalignment = set(usable.nlargest(top_n, "misalignment_magnitude").index)
    top_operating_gap = set(usable.nlargest(top_n, "operating_gap").index)
    overlap = top_misalignment & top_operating_gap
    print(f"overlap: {len(overlap)} / {top_n} ({len(overlap) / top_n:.1%}) -- "
          f"limit too high AND actually speeding, the highest-confidence priority segments")

    print("\n=== Speed Safety Score / priority_class ===")
    from safety_score import PRIORITY_CLASSES_ORDERED
    print(usable["priority_class"].value_counts().reindex(PRIORITY_CLASSES_ORDERED))
    top = usable[usable["priority_class"] == "Top Priority"]
    print(f"\nTop Priority (n={len(top)}): exposure_level=low count={int((top['exposure_level'] == 'low').sum())}, "
          f"motorway count={int((top['road_class'] == 'motorway').sum())} "
          f"(should both be 0/near-0 -- low-exposure rural motorway must not dominate the top class)")

    print("\n=== review_track: plausibility-based split of the priority list ===")
    from review_track import FIELD_CHECK_NEEDED, REVIEW_NEEDED
    review = usable[usable["review_track"] == REVIEW_NEEDED]
    field_check = usable[usable["review_track"] == FIELD_CHECK_NEEDED]
    print(f"Review Needed (plausibility=high): {len(review)}")
    print(f"Field Verification Needed (plausibility=low):  {len(field_check)}")
    print(f"ratio: {len(review)} : {len(field_check)} ({len(review) / max(1, len(field_check)):.1f} : 1)")
    print(usable[usable["review_track"].notna()].groupby("priority_class")["review_track"].value_counts())

    print("\n=== motorway segments with no OSM match (Overture fallback, F85 >= 50) ===")
    moto_uncertain = gdf[(gdf["road_class"] == "motorway") & (gdf["road_structure_confidence"] == "low")]
    print(f"{len(moto_uncertain)} segments; access_control_basis="
          f"{moto_uncertain['access_control_basis'].value_counts(dropna=False).to_dict()}; "
          f"v_safe={moto_uncertain['v_safe'].value_counts().to_dict()}")

    print("\n=== V_safe invariants (each must be 0) ===")
    access = gdf["is_access_controlled"]
    # `!= "no"`: unknown and restricted both mean a rider may be there, as in
    # safe_speed._motorcycle_present_col.
    moto = gdf["legal_motorcycle"] != "no"
    invariants = {
        "is_vru on an access-controlled segment": gdf["is_vru"] & access,
        "v_safe > 70 on an access-controlled undivided segment": access & ~gdf["is_divided"] & (gdf["v_safe"] > 70),
        "v_safe above its VRU speed cap": gdf["is_vru"] & (gdf["v_safe"] > gdf["vru_speed_cap"]),
        "v_safe > 50 near a capped junction": gdf["near_junction"] & (gdf["v_safe"] > 50),
        # A rider sharing lanes with four-wheeled traffic stays at 30 unless a
        # barrier separates them (has_motorcycle_separation or has_vru_barrier).
        "v_safe > 30 where a motorcycle may be present with no barrier": (
            moto & (gdf["v_safe"] > 30)
            & ~_bool_col(gdf, "has_motorcycle_separation")
            & ~_bool_col(gdf, "has_vru_barrier")),
    }
    for label, bad in invariants.items():
        print(f"{label}: {int(bad.sum())}")
    print(f"is_vru True overall: {gdf['is_vru'].sum()} / {len(gdf)} "
          f"(of which is_mapillary_vru: {gdf['is_mapillary_vru'].sum()}, POI-added (OSM ∪ Overture): "
          f"{(gdf['is_vru'] & ~gdf['is_mapillary_vru']).sum()})")
    print(f"VRU cap set by: {gdf['vru_cap_source'].value_counts().to_dict()}")

    print("\n=== POI V_safe zones (poi_speed_zones.py) and sandwich segments ===")
    for col in REASON_COLS.values():
        if col in gdf.columns:
            print(f"{col}: {int(gdf[col].sum())} rows")
    if "poi_parent_id" in gdf.columns:
        pieces = gdf["segment_id"].astype(str) != gdf["poi_parent_id"].astype(str)
        print(f"rows that are POI-zone pieces: {int(pieces.sum())} "
              f"(from {gdf.loc[pieces, 'poi_parent_id'].nunique()} segments)")
    if "is_sandwich" in gdf.columns:
        print(f"is_sandwich: {int(gdf['is_sandwich'].sum())} rows")
    print(f"duplicate segment_id: {int(gdf['segment_id'].duplicated().sum())} (must be 0)")

    print("\n=== motorcycle rule: legal_motorcycle x v_safe ===")
    print(pd.crosstab(gdf["legal_motorcycle"], gdf["v_safe"]))

    print("\n=== junction speed cap (50km/h within 300m of highway=traffic_signals / junction=yes, "
          "excluding motorway and grade-separated segments) ===")
    capped = gdf["v_safe_basis"] == "side_impact:junction_buffer"
    print(f"segments capped to 50km/h: {capped.sum()} / {len(gdf)} ({capped.mean():.1%})")
    print(f"near a junction (motorway/grade-separated already excluded from near_junction) but "
          f"already <=50 before the cap (left untouched): "
          f"{(gdf['near_junction'] & ~capped).sum()}")

    if "v_safe_legacy" in gdf.columns:
        print("\n=== legacy comparison (pre-split is_separated rule, same rows) ===")
        print("NOTE: the old build split rows on its own is_vru/near_junction, so these "
              "columns are the old rule read on the CURRENT row set -- per-row values are "
              "the old rule's, aggregate counts are not the old build's.")
        print(pd.crosstab(gdf["v_safe_legacy"], gdf["v_safe"],
                          rownames=["v_safe_legacy"], colnames=["v_safe"]))
        agree = gdf["v_safe"] == gdf["v_safe_legacy"]
        print(f"\nrows where both rules agree: {int(agree.sum())} / {len(gdf)} ({agree.mean():.1%})")
        print(f"legacy flags: is_separated_legacy={int(gdf['is_separated_legacy'].sum())}, "
              f"is_vru_legacy={int(gdf['is_vru_legacy'].sum())} (current is_vru: {int(gdf['is_vru'].sum())}), "
              f"near_junction_legacy={int(gdf['near_junction_legacy'].sum())} "
              f"(current near_junction: {int(gdf['near_junction'].sum())})")
        print("v_safe_basis_legacy:")
        print(gdf["v_safe_basis_legacy"].value_counts())


def plot_map(gdf, out_path="outputs/v_safe_map.png"):
    # exposure_confidence=='low' is drawn as a grey halo underneath the
    # v_safe-coloured line (wider, plotted first), so every line's own colour
    # comes from the RdYlGn scale of the legend.
    cmap, vmin, vmax = "RdYlGn", 30, 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, country in zip(axes, ["thailand", "maharashtra"]):
        sub = gdf[gdf["country"] == country]

        low_conf = sub[sub["exposure_confidence"] == "low"]
        low_conf.plot(ax=ax, color="grey", linewidth=2.0,
                      label="rural safety-margin applied (confidence=low)")

        sub.plot(column="v_safe", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, linewidth=0.6)

        ax.set_title(f"{country} V_safe (n={len(sub)})")
        ax.set_aspect("equal")
        ax.legend(loc="lower left", fontsize=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=axes, label="V_safe (km/h)", shrink=0.6)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument(
        "--legacy",
        action="store_true",
        help="also compute the pre-split rule's V_safe into *_legacy columns "
             "(parquet only; the GeoParquet/geodatabase/PMTiles/CSV deliverables drop them)",
    )
    _parser.add_argument(
        "--poi-params",
        default=None,
        help="JSON object of POI parameters (see src/poi_params.py); omitted keys keep their defaults",
    )
    _args = _parser.parse_args()
    _poi = PoiParams.from_json(_args.poi_params)

    target, rural_thresholds, score_thresholds = build(include_legacy=_args.legacy, poi_params=_poi)
    print("rural safety-margin thresholds:", rural_thresholds)
    print("safety_score priority-class thresholds (per country, land_use):")
    for cell, cell_thresholds in score_thresholds.items():
        print(f"  {cell}: {{{', '.join(f'{k}: {round(v, 1)}' for k, v in cell_thresholds.items())}}}")
    print()

    write_parquet(target, _poi)
    print(f"saved {OUTPUT_PATH} ({len(target)} rows)\n")

    sanity_checks(target)
    plot_map(target)

    review_path, field_check_path = write_lists(target)
    print(f"\nsaved {review_path}")
    print(f"saved {field_check_path}")

    urban_path, rural_path = write_priority_environment_lists(target)
    print(f"saved {urban_path}")
    print(f"saved {rural_path}")

    priority_map = build_priority_map(target)
    priority_map.save("outputs/priority_map.html")
    print("saved outputs/priority_map.html")

    # The deliverables carry AADT; the exports read it off the same frame.
    target = add_aadt_column(target)

    parquet_path = write_geo_outputs(target)
    print(f"saved {parquet_path}")

    gpkg_path = write_gpkg(target)
    print(f"saved {gpkg_path}")

    png_path = plot_static_summary(target)
    print(f"saved {png_path}")

    gdb_path = write_gdb_zip(target)
    print(f"saved {gdb_path}")
