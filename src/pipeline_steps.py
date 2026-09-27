"""Declarative registry of pipeline steps for the browser GUI.

Three presets, each runnable with one click:

  quick     read the committed parquet, rewrite every deliverable  (~10 s)
  full      re-derive the parquet from the raw GeoJSON via build()  (minutes)
  complete  full, preceded by the external acquisition steps        (hours)

Import cost matters: `serve_map.py` imports this module to answer
`/api/pipeline/steps`, so nothing here may import geopandas, rasterio, pyrosm or
matplotlib at module scope. Every `run` does its own lazy import; every `check`
uses only the standard library.

A missing prerequisite is never an error. `check()` returns a human-readable
reason, the step is reported as skipped, and everything that `needs` it is
skipped too -- which is how `complete` degrades to `full` on a machine with no
osmium.

The build reads the Mapillary files already on disk: the scripts that call
the Mapillary API (fetch_mapillary_features.py, fetch_image_ids.py and the
rest of PIPELINE.md section 5) are run by hand, and tests/test_pipeline_steps.py
pins that no step calls the API.
"""

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_order

REPO_ROOT = Path(__file__).resolve().parent.parent
COUNTRIES = ("thailand", "maharashtra")
PRESETS = ("quick", "full", "complete")
# A fourth preset of three trivial standard-library steps lets
# tests/test_pipeline_api.py exercise the run/stream/cancel machinery in about
# a second. It is outside PRESETS, so the sidebar leaves it out.
SELFTEST_PRESET = "selftest"
ALL_PRESETS = (*PRESETS, SELFTEST_PRESET)

QUICK = frozenset({"quick"})
FULL = frozenset({"full", "complete"})
COMPLETE = frozenset({"complete"})
ALL = frozenset(PRESETS)
SELFTEST = frozenset({SELFTEST_PRESET})

PARQUET_PATH = "data/processed/segments_v_safe.parquet"

# Source .osm.pbf files, mirroring exposure_signals.PBF_PATHS (duplicated here
# because importing that module costs pyrosm + geopandas).
PBF_PATHS = {
    "thailand": "data/external/thailand-260621.osm.pbf",
    "maharashtra": "data/external/western-zone-260621.osm.pbf",
}

# Which Elvik (2019) severity coefficients the benefit-estimate sensitivity
# step reports. Mirrors elvik_2019.json's severity_sets, duplicated here for
# the same reason as PBF_PATHS above (importing that module costs
# pandas, which the web server must never pay); tests/test_elvik_2019.py asserts
# the two stay in step. This selects what is REPORTED -- all three severities
# are always computed, so changing it never requires a rebuild.
SEVERITY_CHOICES = frozenset({"fatal", "all3"})
DEFAULT_SEVERITY = "fatal"


class Ctx:
    """Carries the frame and the thresholds between steps of one run."""

    def __init__(self, log: Callable[[str], None]):
        self.gdf = None
        self.state: dict = {}
        self.log = log
        self.artifacts: list[dict] = []

    def artifact(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            self.artifacts.append({"path": str(p), "bytes": p.stat().st_size})


@dataclass(frozen=True)
class Step:
    id: str
    label_en: str
    label_ja: str
    presets: frozenset
    run: Callable[[Ctx], None]
    check: Callable[[], str | None] = lambda: None
    needs: tuple = ()
    optional: bool = False
    timeout_s: int = 1800
    group: str | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# prerequisite probes (standard library only)
# --------------------------------------------------------------------------

def _exists(*rel_paths: str) -> bool:
    return all((REPO_ROOT / p).exists() for p in rel_paths)


def _need_file(rel_path: str, hint: str = "") -> Callable[[], str | None]:
    def check() -> str | None:
        if _exists(rel_path):
            return None
        return f"{rel_path} missing{f' -- {hint}' if hint else ''}"
    return check


def _check_osmium() -> str | None:
    if shutil.which("osmium") is None:
        return "osmium-tool not installed (`brew install osmium-tool`)"
    missing = [p for p in PBF_PATHS.values() if not _exists(p)]
    if missing:
        return f"source .osm.pbf missing: {', '.join(missing)}"
    return None


def _check_pbf() -> str | None:
    missing = [p for p in PBF_PATHS.values() if not _exists(p)]
    if missing:
        return f"source .osm.pbf missing: {', '.join(missing)} (see PIPELINE.md)"
    return None


def _check_junction_exports() -> str | None:
    missing = [f"data/external/osm_junctions_{c}.geojson" for c in COUNTRIES
               if not _exists(f"data/external/osm_junctions_{c}.geojson")]
    if missing:
        return "manual Overpass export missing (see PIPELINE.md, Phase 1)"
    return None


def _check_osm_pois() -> str | None:
    missing = [f"data/processed/osm_pois_{c}.parquet" for c in COUNTRIES
               if not _exists(f"data/processed/osm_pois_{c}.parquet")]
    if missing:
        return f"{', '.join(missing)} missing -- run exposure_extract first"
    return None


def _check_rasters() -> str | None:
    return None  # fetch_worldpop is idempotent: it skips files already on disk


def _check_tiles() -> str | None:
    from build_tiles import check as tiles_check
    return tiles_check()


# --------------------------------------------------------------------------
# step bodies (heavy imports happen here, inside the runner process)
# --------------------------------------------------------------------------

def _run_worldpop(ctx: Ctx) -> None:
    from fetch_worldpop import fetch_maharashtra, fetch_thailand
    ctx.log(f"thailand raster: {fetch_thailand()}")
    ctx.log(f"maharashtra raster: {fetch_maharashtra()}")


def _run_prefilter(ctx: Ctx) -> None:
    from prefilter_pbf import ensure_filtered_pbf
    for country, pbf in PBF_PATHS.items():
        ctx.log(f"[{country}] {ensure_filtered_pbf(pbf)}")


def _run_exposure_extract(ctx: Ctx) -> None:
    from exposure_signals import extract_and_cache
    for country in COUNTRIES:
        signals = extract_and_cache(country)
        ctx.log(f"[{country}] " + ", ".join(f"{k}={len(v)}" for k, v in signals.items()))


def _run_road_cache(ctx: Ctx) -> None:
    from road_separation import cache_road_network
    for country in COUNTRIES:
        ctx.log(f"[{country}] {len(cache_road_network(country))} road ways cached")


def _run_junction_cache(ctx: Ctx) -> None:
    from junction_speed_cap import cache_junctions
    for country in COUNTRIES:
        ctx.log(f"[{country}] {len(cache_junctions(country))} junction nodes cached")


def _run_overture_pois(ctx: Ctx) -> None:
    import poi_sources
    from fetch_overture_pois import fetch_overture_pois, write
    for country in COUNTRIES:
        gdf, meta = fetch_overture_pois(country)
        write(country, gdf, meta)
        ctx.log(f"[{country}] {len(poi_sources.build_and_cache(country))} POIs in the OSM ∪ Overture union")


def _run_isochrones(ctx: Ctx) -> None:
    """Make sure every POI type that caps V_safe has its isochrones at the
    minutes this run asks for. What is missing is built with Valhalla, one
    country at a time (valhalla_service starts and stops its container)."""
    from poi_isochrone import build_and_cache, cache_path, is_current
    from poi_params import DEFAULT
    from poi_sources import poi_candidates
    from valhalla_service import serving

    caps = ctx.state.get("poi_params", DEFAULT).enabled_caps()
    if not caps:
        ctx.log("no POI type caps V_safe; no isochrones needed")
        return
    for country in COUNTRIES:
        missing = []
        for cap in caps:
            path = cache_path(country, cap.poi_type, cap.iso_min_urban, cap.iso_min_rural)
            if is_current(country, cap.poi_type, cap.iso_min_urban, cap.iso_min_rural):
                ctx.log(f"[{country}] {cap.poi_type}: cached ({path})")
            else:
                missing.append(cap)
        if not missing:
            continue
        # Any candidate of a missing type sits in the country's network, which
        # is what tells the two countries' Valhalla containers apart.
        points = poi_candidates(country, missing[0].poi_type).geometry
        if points.empty:
            points = poi_candidates(country, "school").geometry
        probe = points.iloc[0].representative_point()
        with serving(country, (probe.x, probe.y), log=ctx.log):
            for cap in missing:
                iso = build_and_cache(country, cap.poi_type, cap.iso_min_urban, cap.iso_min_rural,
                                      force=True, log_fn=ctx.log)
                ctx.log(f"[{country}] {cap.poi_type}: built {len(iso)} isochrones")
                ctx.artifact(cache_path(country, cap.poi_type, cap.iso_min_urban, cap.iso_min_rural))


def _run_load_parquet(ctx: Ctx) -> None:
    import geopandas as gpd
    from tomtom_enrichment import ensure_tomtom_columns
    # See quick_reproduce: a parquet predating the TomTom layer must still work.
    ctx.gdf = ensure_tomtom_columns(gpd.read_parquet(PARQUET_PATH))
    ctx.log(f"loaded {PARQUET_PATH} ({len(ctx.gdf)} rows) -- "
            "no raw GeoJSON / rasters / OSM extraction needed")


def _run_write_parquet(ctx: Ctx) -> None:
    from build_v_safe import OUTPUT_PATH, POI_PARAMS_PATH, write_parquet
    from poi_params import DEFAULT
    write_parquet(ctx.gdf, ctx.state.get("poi_params", DEFAULT))
    ctx.artifact(POI_PARAMS_PATH)
    ctx.log(f"saved {OUTPUT_PATH} ({len(ctx.gdf)} rows)")
    ctx.artifact(OUTPUT_PATH)


def _run_sanity_checks(ctx: Ctx) -> None:
    from build_v_safe import sanity_checks
    ctx.log(f"rural safety-margin thresholds: {ctx.state.get('rural_thresholds')}")
    ctx.log(f"score thresholds: {ctx.state.get('score_thresholds')}")
    sanity_checks(ctx.gdf)


def _run_plot_map(ctx: Ctx) -> None:
    from build_v_safe import plot_map
    plot_map(ctx.gdf)
    ctx.artifact("outputs/v_safe_map.png")


def _run_write_lists(ctx: Ctx) -> None:
    from review_track import write_lists
    for path in write_lists(ctx.gdf):
        ctx.log(f"saved {path}")
        ctx.artifact(path)


def _run_environment_lists(ctx: Ctx) -> None:
    from priority_lists import write_priority_environment_lists
    for path in write_priority_environment_lists(ctx.gdf):
        ctx.log(f"saved {path}")
        ctx.artifact(path)


def _run_priority_map(ctx: Ctx) -> None:
    from priority_map import build_priority_map
    build_priority_map(ctx.gdf).save("outputs/priority_map.html")
    ctx.log("saved outputs/priority_map.html")
    ctx.artifact("outputs/priority_map.html")


def _run_aadt(ctx: Ctx) -> None:
    from aadt_estimation import DEFAULT as DEFAULT_AADT_PARAMS
    from aadt_estimation import AADT_MAX, AADT_MIN, add_aadt_column
    params = ctx.state.get("aadt_params", DEFAULT_AADT_PARAMS)
    ctx.gdf = add_aadt_column(ctx.gdf, params)
    aadt = ctx.gdf["AADT"]
    ctx.log(f"AADT: median {int(aadt.median())} vehicles/day, "
            f"{int((aadt == AADT_MAX).sum())} rows at the {AADT_MAX} ceiling, "
            f"{int((aadt == AADT_MIN).sum())} at the {AADT_MIN} floor")


def _run_geo_outputs(ctx: Ctx) -> None:
    from priority_map import write_geo_outputs
    path = write_geo_outputs(ctx.gdf)
    ctx.log(f"saved {path}")
    ctx.artifact(path)


def _run_gdb_output(ctx: Ctx) -> None:
    from priority_map import write_gdb_zip
    path = write_gdb_zip(ctx.gdf)
    ctx.log(f"saved {path}")
    ctx.artifact(path)


def _run_gpkg_output(ctx: Ctx) -> None:
    from priority_map import write_gpkg
    path = write_gpkg(ctx.gdf)
    ctx.log(f"saved {path}")
    ctx.artifact(path)


def _run_static_summary(ctx: Ctx) -> None:
    from priority_map import plot_static_summary
    path = plot_static_summary(ctx.gdf)
    ctx.log(f"saved {path}")
    ctx.artifact(path)


def _valid(ctx: Ctx):
    return ctx.gdf[ctx.gdf["data_quality_flag"].isna()]


def _run_sens_sample(ctx: Ctx) -> None:
    from sensitivity_analysis import sample_size_robustness
    s = sample_size_robustness(_valid(ctx))
    ctx.log(f"sample-size robustness: recall={s['recall_of_baseline']:.1%}, jaccard={s['jaccard']:.1%}")


def _run_sens_weight(ctx: Ctx) -> None:
    from sensitivity_analysis import weight_robustness
    ctx.log(str(weight_robustness(_valid(ctx))[["n_other", "km_other", "recall_of_baseline", "jaccard"]]))


def _run_sens_exp(ctx: Ctx) -> None:
    from sensitivity_analysis import exponential_model_comparison
    severity = ctx.state.get("severity", DEFAULT_SEVERITY)
    ctx.log(f"severity set: {severity}")
    # to_string(): this table is ten columns wide, and str() would elide the
    # middle of it in the GUI log.
    ctx.log(exponential_model_comparison(_valid(ctx), severity=severity).round(1).to_string())


def _run_build_tiles(ctx: Ctx) -> None:
    from build_tiles import PMTILES_PATH, build_tiles
    build_tiles(log=ctx.log)
    ctx.artifact(PMTILES_PATH)


# --------------------------------------------------------------------------
# the single ordered registry; the three presets are views onto it
# --------------------------------------------------------------------------

# Probes for build sub-steps that have an optional input. These only grey the
# step out in describe(): pipeline_runner short-circuits BUILD_STEP_IDS into a
# single build_v_safe.build() call before it ever consults check(), so a
# missing optional input cannot skip any real work.
# Mirrors tomtom_stats.RAW_PATHS (duplicated for the same reason as PBF_PATHS).
TOMTOM_RAW_PATHS = {
    "thailand": "data/raw/combined_traffic_stats_Thailand.geojson",
    "maharashtra": "data/raw/ADBMaha_combined_20250808.geojson",
}


def _check_tomtom() -> str | None:
    if any(_exists(p) for p in TOMTOM_RAW_PATHS.values()):
        return None
    return (f"{', '.join(TOMTOM_RAW_PATHS.values())} missing -- "
            "optional layer (Thailand, Maharashtra); the build runs without it")


_BUILD_CHECKS = {"tomtom": _check_tomtom}


def _build_steps() -> list[Step]:
    """build()'s own sub-steps, generated from pipeline_order so the chain the
    GUI draws is by construction the sequence build() executes."""
    return [
        Step(id=sid, label_en=en, label_ja=ja, presets=FULL, group=group,
             run=_unreachable, needs=("load_target",) if sid != "load_target" else (),
             check=_BUILD_CHECKS.get(sid, lambda: None))
        for sid, en, ja, group in pipeline_order.build_sequence()
    ]


def _selftest_echo(ctx: Ctx) -> None:
    print("selftest: echo on stdout")
    ctx.log("selftest: echo via ctx.log")


def _selftest_sleep(ctx: Ctx) -> None:
    import time
    time.sleep(float(os.environ.get("PIPELINE_SELFTEST_SLEEP", "0.3")))


def _selftest_fail(ctx: Ctx) -> None:
    raise RuntimeError("selftest failure (expected)")


def _unreachable(ctx: Ctx) -> None:
    # build()'s sub-steps are driven by build_v_safe.build(progress=...) inside
    # pipeline_runner, not called one by one -- the frame has to stay in build()'s
    # local scope. These entries exist so the chain can be rendered and timed.
    raise AssertionError("build sub-steps are executed by build_v_safe.build()")


STEPS: list[Step] = [
    # ---- acquisition (complete only) ----
    Step("worldpop", "Fetch WorldPop rasters", "WorldPop ラスタ取得", COMPLETE,
         _run_worldpop, _check_rasters, timeout_s=7200,
         meta={"downloads_gb": 2.1}),
    Step("prefilter_pbf", "Prefilter OSM extracts (osmium)", "OSM 抽出の事前フィルタ (osmium)",
         COMPLETE, _run_prefilter, _check_osmium, timeout_s=1800),
    Step("exposure_extract", "Extract exposure signals (pyrosm)", "暴露シグナル抽出 (pyrosm)",
         COMPLETE, _run_exposure_extract, _check_pbf, timeout_s=3600),
    Step("road_network_cache", "Cache OSM road network", "OSM 道路網キャッシュ",
         COMPLETE, _run_road_cache, _check_pbf, timeout_s=3600),
    Step("junctions_cache", "Cache OSM junctions", "OSM 交差点キャッシュ",
         COMPLETE, _run_junction_cache, _check_junction_exports, timeout_s=900),
    Step("overture_pois", "Fetch Overture POIs, build OSM ∪ Overture union",
         "Overture POI 取得・OSM との和集合作成",
         COMPLETE, _run_overture_pois, _check_osm_pois, timeout_s=3600),
    # ---- full + complete: the isochrones this run's POI parameters need ----
    # Not optional: a V_safe for a zone other than the one asked for is wrong.
    # Eight hours covers a first-time routing-tile build for both countries.
    Step("poi_isochrones", "POI isochrones (Valhalla, started as needed)",
         "POI アイソクロン (必要なら Valhalla を起動)",
         FULL, _run_isochrones, timeout_s=8 * 3600),

    # ---- the V_safe build (full + complete), generated from pipeline_order ----
    *_build_steps(),
    Step("write_parquet", "Write segments_v_safe.parquet", "segments_v_safe.parquet 書き出し",
         FULL, _run_write_parquet, needs=("load_target",)),
    Step("sanity_checks", "Sanity checks", "サニティチェック",
         FULL, _run_sanity_checks, needs=("write_parquet",), optional=True),
    Step("plot_map", "V_safe overview PNG", "V_safe 概観 PNG",
         FULL, _run_plot_map, needs=("write_parquet",), optional=True),

    # ---- quick's entry point: reads what full has just written ----
    Step("load_parquet", "Load segments_v_safe.parquet", "segments_v_safe.parquet 読み込み",
         QUICK, _run_load_parquet, _need_file(PARQUET_PATH, "run the full preset first"),
         timeout_s=600),

    # ---- deliverables, shared by all three presets ----
    Step("write_lists", "Review / field-check lists", "レビュー・現地確認リスト",
         ALL, _run_write_lists, timeout_s=600),
    Step("environment_lists", "Urban / rural priority lists", "都市・郊外 優先リスト",
         ALL, _run_environment_lists, timeout_s=600),
    Step("priority_map", "Interactive folium map", "folium インタラクティブ地図",
         ALL, _run_priority_map, timeout_s=900),
    # Before the three exports, which share the same frame: all carry AADT.
    Step("aadt", "AADT estimation", "AADT 推定",
         ALL, _run_aadt, timeout_s=600),
    Step("geo_outputs", "GeoParquet export", "GeoParquet 出力",
         ALL, _run_geo_outputs, needs=("aadt",), timeout_s=900),
    Step("gpkg_output", "GeoPackage export", "GeoPackage 出力",
         ALL, _run_gpkg_output, needs=("aadt",), timeout_s=900),
    Step("static_summary", "Static summary PNG", "静的サマリ PNG",
         ALL, _run_static_summary, optional=True, timeout_s=600),
    Step("sens_sample_size", "Sensitivity: sample size", "感度分析: サンプルサイズ",
         ALL, _run_sens_sample, optional=True, timeout_s=600),
    Step("sens_weight", "Sensitivity: weights", "感度分析: 重み",
         ALL, _run_sens_weight, optional=True, timeout_s=600),
    Step("sens_exp_model", "Sensitivity: Elvik (2019) estimate", "感度分析: Elvik (2019) 推計",
         ALL, _run_sens_exp, optional=True, timeout_s=600),
    Step("build_tiles", "Rebuild map tiles (tippecanoe)", "地図タイル再生成 (tippecanoe)",
         ALL, _run_build_tiles, _check_tiles, needs=("geo_outputs",),
         optional=True, timeout_s=900),
    # Last: the zipped File Geodatabase uploaded to ArcGIS Online.
    Step("gdb_output", "File Geodatabase for ArcGIS Online", "ArcGIS Online 用ファイルジオデータベース",
         ALL, _run_gdb_output, needs=("aadt",), timeout_s=900),

    # ---- selftest only ----
    Step("selftest_echo", "Selftest: echo", "セルフテスト: echo", SELFTEST,
         _selftest_echo, timeout_s=60),
    Step("selftest_sleep", "Selftest: sleep", "セルフテスト: sleep", SELFTEST,
         _selftest_sleep, timeout_s=120),
    Step("selftest_fail", "Selftest: fail", "セルフテスト: fail", SELFTEST,
         _selftest_fail, timeout_s=60),
    Step("selftest_after", "Selftest: unreached", "セルフテスト: 未到達", SELFTEST,
         _selftest_echo, timeout_s=60),
]

_BY_ID = {s.id: s for s in STEPS}
assert len(_BY_ID) == len(STEPS), "duplicate step id"
for _s in STEPS:
    for _n in _s.needs:
        assert _n in _BY_ID, f"{_s.id} needs unknown step {_n}"

BUILD_STEP_IDS = frozenset(sid for sid, _, _, _ in pipeline_order.build_sequence())


def steps_for(preset: str) -> list[Step]:
    if preset not in ALL_PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    return [s for s in STEPS if preset in s.presets]


def describe(preset: str) -> list[dict]:
    """Step list plus a freshly evaluated skip reason for each, so the GUI can
    grey out unavailable steps before the user presses Run."""
    out = []
    skipped: set[str] = set()
    for step in steps_for(preset):
        reason = next((f"depends on {n}" for n in step.needs if n in skipped), None)
        if reason is None:
            try:
                reason = step.check()
            except Exception as exc:  # a probe must never break the page
                reason = f"check failed: {exc}"
        if reason:
            skipped.add(step.id)
        out.append({
            "id": step.id,
            "label_en": step.label_en,
            "label_ja": step.label_ja,
            "group": step.group,
            "optional": step.optional,
            "skip_reason": reason,
        })
    return out


PRESET_META = {
    "quick": {
        "label_en": "Quick", "label_ja": "Quick",
        "desc_en": "Rebuild every deliverable from the committed parquet.",
        "desc_ja": "コミット済み parquet から全成果物を再生成。",
        "eta": "~10 s", "confirm": False,
    },
    "full": {
        "label_en": "Full", "label_ja": "Full",
        "desc_en": "Re-derive V_safe from the raw ADB GeoJSON, then all deliverables. "
                   "Missing POI isochrones are built with Valhalla (Docker).",
        "desc_ja": "生の ADB GeoJSON から V_safe を再計算し、全成果物まで。"
                   "足りない POI アイソクロンは Valhalla (Docker) で作成。",
        "eta": "minutes", "confirm": False,
    },
    "complete": {
        "label_en": "Complete", "label_ja": "Complete",
        "desc_en": "Everything, including WorldPop, OSM extraction, Overture and Valhalla. "
                   "Mapillary data is read from disk; no Mapillary API request is made.",
        "desc_ja": "WorldPop・OSM 抽出・Overture・Valhalla を含む全工程。"
                   "Mapillary は保存済みデータを使い、API へのリクエストはしない。",
        "eta": "hours", "confirm": True,
        "confirm_en": "Downloads about 2.1 GB of WorldPop rasters, re-parses the OSM extracts, "
                      "re-fetches Overture POIs and rebuilds their isochrones with Valhalla (Docker).",
        "confirm_ja": "WorldPop ラスタ約 2.1GB をダウンロードし、OSM 抽出を再解析し、"
                      "Overture POI を取り直して Valhalla (Docker) でアイソクロンを作り直します。",
    },
}
