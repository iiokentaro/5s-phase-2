"""Ordered metadata for the V_safe build, importable without geopandas.

The browser GUI's step chain and `build_v_safe.build()` must agree on what runs
and in what order. `build_v_safe` cannot be imported to find that out -- it pulls
in geopandas, rasterio, pyrosm and matplotlib, several seconds and hundreds of MB
that the web server must never pay -- so the *order and labels* live here, in a
module with no imports at all, and `build_v_safe` pairs them with the actual
functions at import time, asserting that the two sets match.

Each entry is (step_id, label_en, label_ja).
"""

# Run twice by build(): once on whole segments, once after the influence-zone
# split. See _apply_geometry_signals' docstring for why.
GEOMETRY_SIGNALS = [
    ("road_structure", "Access control / divided (OSM)", "歩車分離・中央分離 (OSM)"),
    # Straight after road_structure: it reuses that step's memoised
    # segment-to-way attribution.
    ("road_access_join", "Per-mode legal access (OSM)", "モード別通行可否 (OSM)"),
    ("pop_density", "Population density", "人口密度"),
    ("poi_proximity", "VRU POI proximity", "VRU POI 近接"),
    ("crossing_signal", "At-grade crossings", "平面横断箇所"),
    ("exposure_level", "Exposure level", "暴露レベル"),
    ("rural_margin", "Rural safety margin", "郊外安全マージン"),
    ("v_safe", "V_safe", "V_safe"),
    ("junction_cap", "Junction 50 km/h cap", "交差点 50km/h 上限"),
]

POST_REFINE = [
    ("speedlimit_plausibility", "Speed-limit plausibility", "制限速度の妥当性"),
    ("misalignment", "Misalignment", "乖離 (misalignment)"),
    # Before safety_score. The Elvik (2019) estimate is the reported fatal-crash
    # reduction. It never enters the score itself; it orders segments that share
    # a score when safety_score cuts priority_class, and segments that share a
    # rank when the lists are written. tests/test_pipeline_steps.py pins the
    # ordering.
    ("exp_fatal_reduction", "Fatal-crash reduction (Elvik 2019)", "死亡事故削減効果 (Elvik 2019)"),
    ("safety_score", "Speed Safety Score", "Speed Safety Score"),
    ("review_track", "Review track split", "レビュートラック分岐"),
    ("priority_rank", "Priority environment rank", "環境別優先順位"),
]

LOAD_TARGET = ("load_target", "Load ADB GeoJSON", "ADB GeoJSON 読み込み")

# Optional, between load_target and the first geometry stage: it fixes the
# population and the geometry granularity before any signal runs, so the nine
# signals still run exactly twice. With no TomTom layer on disk it only sets
# speed_data_source.
TOMTOM = ("tomtom", "TomTom enrichment (optional)", "TomTom による補強 (任意)")

REFINE = ("refine", "Split influenced segments", "影響受けセグメントの分割")

# After stage 2: the POI zones set V_safe on the final geometry (splitting a
# segment where a zone covers less than 80% of it), then short segments take
# their neighbours' V_safe.
POI_ZONES = ("poi_zones", "POI V_safe zones", "POI による V_safe")
SANDWICH = ("sandwich", "Short sandwich segments", "短いセグメント (sandwich)")
# After the sandwich step, before the scores: the Mapillary object tables
# (segment_detected_objects.py) set is_divided, has_vru_barrier and
# has_motorcycle_separation on non-POI segments and raise V_safe where that
# allows it, so the scores see the result.
OBJECT_SEPARATION = ("object_separation", "Mapillary object separation", "Mapillary 物体による分離")

STAGE_LABELS = {
    "stage1": ("Stage 1 — whole segments", "ステージ1 — セグメント全体"),
    "stage2": ("Stage 2 — after influence-zone split", "ステージ2 — 影響域分割後"),
}


def build_sequence() -> list[tuple[str, str, str, str | None]]:
    """The full ordered (step_id, label_en, label_ja, group) sequence of build()."""
    seq: list[tuple[str, str, str, str | None]] = [(*LOAD_TARGET, None), (*TOMTOM, None)]
    for group in ("stage1", "stage2"):
        if group == "stage2":
            seq.append((*REFINE, None))
        seq += [(f"{group}.{sid}", en, ja, group) for sid, en, ja in GEOMETRY_SIGNALS]
    seq += [(*POI_ZONES, None), (*SANDWICH, None), (*OBJECT_SEPARATION, None)]
    seq += [(sid, en, ja, None) for sid, en, ja in POST_REFINE]
    return seq
