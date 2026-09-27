"""Rebuild every deliverable from the committed segments_v_safe.parquet.

`build_v_safe.py` is the full pipeline. It starts from `schema.load_target()`,
which needs the raw GeoJSON (`data/raw/*.geojson`, gitignored) and the WorldPop
rasters (`data/external/*.tif`, about 2 GB, fetched by `fetch_worldpop.py`).

`data/processed/segments_v_safe.parquet` is committed and holds every column
the build computes (V_safe, exposure, misalignment, safety_score,
priority_class, review_track). The map, the GeoParquet / GeoPackage /
geodatabase, the static PNG, the priority CSV lists and the sensitivity tables
are all derived from its columns, so this script reproduces them from a clean
clone in under a minute.

To re-derive the parquet itself, follow README.md's "Full rebuild": copy
`QGIS/ADB_Innovation_*.geojson` (byte-identical to the raw files) into
`data/raw/`, run `src/fetch_worldpop.py`, then `python src/build_v_safe.py`.
"""

import sys
import warnings

sys.path.insert(0, "src")

warnings.filterwarnings("ignore", category=UserWarning)

INPUT_PATH = "data/processed/segments_v_safe.parquet"


def main():
    import geopandas as gpd

    from aadt_estimation import add_aadt_column
    from priority_lists import write_priority_environment_lists
    from priority_map import build_priority_map, plot_static_summary, write_gdb_zip, write_geo_outputs, write_gpkg
    from review_track import FIELD_CHECK_NEEDED, REVIEW_NEEDED, write_lists
    from sensitivity_analysis import (exponential_model_comparison, sample_size_robustness,
                                      weight_robustness)

    from elvik_2019 import ensure_exponential_columns
    from safety_score import ensure_norm_columns
    from tomtom_enrichment import ensure_tomtom_columns

    gdf = gpd.read_parquet(INPUT_PATH)
    # A parquet built before the TomTom layer existed has none of its columns;
    # filling them Overture-only here keeps the whole quick path working on it,
    # exactly as ensure_exponential_columns does for the Elvik columns.
    gdf = ensure_tomtom_columns(gdf)
    # Likewise for the Elvik (2019) columns: once here, so the lists, the map,
    # the GeoParquet/GeoPackage/geodatabase and the sensitivity tables all read the same values.
    gdf = ensure_exponential_columns(gdf)
    # And for misalignment_norm / exposure_norm, the score terms add_safety_score
    # writes beside confidence_norm.
    gdf = ensure_norm_columns(gdf)
    print(f"loaded {INPUT_PATH} ({len(gdf)} rows) -- no raw GeoJSON / rasters / OSM extraction needed")

    valid = gdf[gdf["data_quality_flag"].isna()]
    print(f"\npriority_class by country:")
    print(valid.groupby("country")["priority_class"].value_counts().unstack())

    review_path, field_check_path = write_lists(gdf)
    print(f"\nsaved {review_path} ({(valid['review_track'] == REVIEW_NEEDED).sum()} rows)")
    print(f"saved {field_check_path} ({(valid['review_track'] == FIELD_CHECK_NEEDED).sum()} rows)")

    urban_path, rural_path = write_priority_environment_lists(gdf)
    on_list = valid["rank_within_environment"].notna()
    print(f"saved {urban_path} ({(on_list & (valid['road_environment'] == 'urban')).sum()} rows)")
    print(f"saved {rural_path} ({(on_list & (valid['road_environment'] == 'rural')).sum()} rows)")

    fmap = build_priority_map(gdf)
    fmap.save("outputs/priority_map.html")
    print("saved outputs/priority_map.html")

    # The deliverables carry AADT; the exports read it off the same frame.
    gdf = add_aadt_column(gdf)

    parquet_path = write_geo_outputs(gdf)
    print(f"saved {parquet_path}")

    gpkg_path = write_gpkg(gdf)
    print(f"saved {gpkg_path}")

    png_path = plot_static_summary(gdf)
    print(f"saved {png_path}")

    print("\n=== sensitivity analysis ===")
    s = sample_size_robustness(valid)
    print(f"sample-size robustness: recall={s['recall_of_baseline']:.1%}, jaccard={s['jaccard']:.1%}")
    w = weight_robustness(valid)
    print(w[["n_other", "km_other", "recall_of_baseline", "jaccard"]])
    # The same table the GUI's sens_exp_model step reports; kept here so the
    # non-GUI fast path stays equivalent to the full one.
    print(exponential_model_comparison(valid).round(1).to_string())

    # Last, as in the pipeline: the zipped File Geodatabase uploaded to ArcGIS Online.
    gdb_path = write_gdb_zip(gdf)
    print(f"\nsaved {gdb_path}")


if __name__ == "__main__":
    main()
