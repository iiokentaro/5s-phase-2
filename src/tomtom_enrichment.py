"""The optional TomTom enrichment step: backfill, provenance, reverse mapping.

Registered as `tomtom` in pipeline_order, immediately after `load_target` and
before the first geometry stage. That position is deliberate:

* The population is fixed before any geometry signal runs, which is the
  invariant the two-stage (stage1 / refine / stage2) design rests on.
* Cutting into uninterrupted segments here means segment_localization's
  influence split later cuts those pieces further, and its _make_child copies
  the whole parent row dict -- so every column written here reaches the
  grandchildren with no change to that module.
* The nine geometry signals therefore still run exactly twice, over geometry
  that is TomTom-granular from the outset.

The cutting and matching are tomtom_data_integration's; this module decides
what the pipeline does with the result.

★ Optional means optional ★
With no TomTom layer for a country this step only writes `speed_data_source =
'overture'` plus all-NA TomTom columns for its rows.

★ One direction, chosen, never blended ★
TomTom values arrive per direction of travel (*_fwd / *_bwd). The single-valued
columns the rest of the pipeline reads (tomtom_median_speed, tomtom_p85_speed,
tomtom_sd_speed, tomtom_speed_percentiles, ...) are copied from ONE direction:
the one with the larger sample, recorded in `tomtom_selected_direction`. The
posted limit follows the same choice, falling back to the other direction when
the selected one has none. The two directions may legitimately carry different
limits; `tomtom_speed_limit_directions_differ` records where they do.

★ TomTom's posted limit wins ★
Wherever TomTom carries a posted limit -- non-zero and a multiple of 5 km/h,
which tomtom_stats.normalise_speed_limit guarantees for every non-null
`tomtom_speed_limit` -- it replaces the ADB `speed_limit`, whatever the ADB
value was. TomTom's limit is the more reliable record. The ADB value is kept
in `speed_limit_adb` and `speed_limit_source` says which one `speed_limit`
holds, so every assessment stays traceable to its record.
`speed_limit_backfilled` marks the subset where the ADB value was unusable
(null, or a row schema.has_invalid_zero_speeds identified: speed_limit /
median_speed / f85_speed all exactly 0 -- an export artifact).

Observed speed is different: `median_speed` / `f85_speed` are filled from
TomTom on the 'invalid_speed' rows only. TomTom's measured mean and SD reach
the Elvik (2019) estimate directly (elvik_2019.distribution_inputs).

★ Why clearing data_quality_flag needs all three fields ★
'invalid_speed' is not a statement about the limit alone; it means all three
speed fields are placeholder zeros. Clearing it with median_speed still 0 would
hand elvik_2019 a median of 0, and its speed_sd would read (0 - 0) / 1.04. So the flag is
cleared only once TomTom has supplied a usable limit AND a usable observed
speed; a row TomTom can only half-repair keeps the flag and stays out of
scoring.
"""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

import tomtom_data_integration as tdi  # noqa: E402
import tomtom_stats  # noqa: E402

SOURCE_OVERTURE = "overture"
SOURCE_TOMTOM = "tomtom"

# `tomtom=None` has to mean "there is no layer" -- it is what a caller gets
# back from tomtom_stats.load_tomtom() and what the degraded path is tested
# with. So "read it from disk" has a sentinel of its own.
AUTO_LOAD = object()

# The single-valued columns, copied from the selected direction.
SELECTED_COLUMNS = {
    "tomtom_segment_id": "tomtom_segment_id",
    "tomtom_mean_speed": "tomtom_mean_speed",
    "tomtom_median_speed": "tomtom_median_speed",
    "tomtom_p85_speed": "tomtom_p85_speed",
    "tomtom_sd_speed": "tomtom_sd_speed",
    "tomtom_sample_size": "tomtom_sample_size",
    "tomtom_speed_percentiles": "tomtom_speed_percentiles",
    "tomtom_street_name": "tomtom_street_name",
}

SIDE_COLUMNS = [
    f"{c}_{side}"
    for side in tdi.SIDES
    for c in ["tomtom_segment_id", *tdi.SIDE_VALUE_COLUMNS,
              "tomtom_street_name", "tomtom_covered_len_m"]
]

# Every column this module guarantees to exist. Same role as
# elvik_2019.exponential_columns(): a frame built before this module can be
# brought up to the current schema without re-running the build.
TOMTOM_COLUMNS = [
    "speed_data_source",
    "overture_segment_id",
    "uninterrupted_segment_id",
    "tomtom_selected_direction",
    "tomtom_segment_id",
    "tomtom_coverage_fraction",
    "tomtom_speed_limit",
    "tomtom_speed_limit_directions_differ",
    "tomtom_mean_speed",
    "tomtom_median_speed",
    "tomtom_p85_speed",
    "tomtom_sd_speed",
    "tomtom_sample_size",
    "tomtom_speed_percentiles",
    "tomtom_street_name",
    *SIDE_COLUMNS,
    "speed_limit_adb",
    "speed_limit_source",
    "speed_limit_backfilled",
    "observed_speed_backfilled",
]

_NA_DEFAULTS = {
    "uninterrupted_segment_id": None,
    "tomtom_selected_direction": None,
    "tomtom_segment_id": pd.NA,
    "tomtom_coverage_fraction": 0.0,
    "tomtom_speed_limit": pd.NA,
    "tomtom_speed_limit_directions_differ": False,
    "tomtom_mean_speed": np.nan,
    "tomtom_median_speed": np.nan,
    "tomtom_p85_speed": np.nan,
    "tomtom_sd_speed": np.nan,
    "tomtom_sample_size": pd.NA,
    "tomtom_speed_percentiles": None,
    "tomtom_street_name": pd.NA,
    **{c: (0.0 if c.startswith("tomtom_covered_len_m") else
           None if c.startswith("tomtom_speed_percentiles") else pd.NA)
       for c in SIDE_COLUMNS},
    "speed_limit_source": "adb",
    "speed_limit_backfilled": False,
    "observed_speed_backfilled": False,
}


def ensure_tomtom_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a frame up to the enrichment schema, Overture-only.

    Returned by identity when nothing is missing, so a frame that already went
    through the enriched path is never disturbed. Mirrors
    elvik_2019.ensure_exponential_columns, and exists for the same reason: the
    `quick` preset reads a committed parquet that may predate this module, and
    must not KeyError on it.
    """
    if all(column in df.columns for column in TOMTOM_COLUMNS):
        return df

    df = df.copy()
    if "speed_data_source" not in df.columns:
        df["speed_data_source"] = SOURCE_OVERTURE
    if "overture_segment_id" not in df.columns:
        df["overture_segment_id"] = (
            df["segment_id"].astype(str) if "segment_id" in df.columns else pd.NA
        )
    for column, default in _NA_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default
    if "speed_limit_adb" not in df.columns:
        df["speed_limit_adb"] = df["speed_limit"] if "speed_limit" in df.columns else np.nan
    df["tomtom_segment_id"] = df["tomtom_segment_id"].astype("Int64")
    return df


def apply_tomtom_speed_limit(gdf: pd.DataFrame) -> pd.DataFrame:
    """Take `speed_limit` from TomTom wherever TomTom has a posted limit.

    A non-null `tomtom_speed_limit` is already a non-zero multiple of 5 km/h
    (tomtom_stats.normalise_speed_limit), so it replaces the ADB value on
    every row that carries one -- see the module docstring. The ADB value is
    kept in `speed_limit_adb`; `speed_limit_source` is 'tomtom' or 'adb'.
    `speed_limit_backfilled` marks the replaced rows whose ADB value was
    unusable: null, or flagged 'invalid_speed' (all three speed fields 0).
    """
    gdf = gdf.copy()
    flagged = (
        gdf["data_quality_flag"].eq("invalid_speed")
        if "data_quality_flag" in gdf.columns
        else pd.Series(False, index=gdf.index)
    )
    gdf["speed_limit_adb"] = gdf["speed_limit"]
    from_tomtom = gdf["tomtom_speed_limit"].notna()
    gdf.loc[from_tomtom, "speed_limit"] = gdf.loc[from_tomtom, "tomtom_speed_limit"].astype(float)
    gdf["speed_limit_source"] = np.where(from_tomtom, "tomtom", "adb")
    gdf["speed_limit_backfilled"] = from_tomtom & (gdf["speed_limit_adb"].isna() | flagged)
    return gdf


def backfill_observed_speed(gdf: pd.DataFrame) -> pd.DataFrame:
    """Fill `median_speed` / `f85_speed` on 'invalid_speed' rows only.

    Observed speed reaches V_safe only through the Overture motorway fallback
    (road_separation, F85 >= 50), so replacing a placeholder zero with a real
    measurement changes that fallback and the diagnostics it feeds:
    operating_gap, speedlimit_plausibility and the Elvik (2019) estimate. Rows
    that are not flagged keep the ADB measurement.
    """
    gdf = gdf.copy()
    flagged = (
        gdf["data_quality_flag"].eq("invalid_speed")
        if "data_quality_flag" in gdf.columns
        else pd.Series(False, index=gdf.index)
    )
    fillable = (
        flagged
        & gdf["tomtom_median_speed"].notna()
        & gdf["tomtom_p85_speed"].notna()
        & (gdf["tomtom_median_speed"] > 0)
    )
    gdf.loc[fillable, "median_speed"] = gdf.loc[fillable, "tomtom_median_speed"]
    gdf.loc[fillable, "f85_speed"] = gdf.loc[fillable, "tomtom_p85_speed"]
    gdf["observed_speed_backfilled"] = fillable
    return gdf


def clear_repaired_quality_flags(gdf: pd.DataFrame) -> pd.DataFrame:
    """Return a fully repaired row to the scored population.

    Fully repaired means every field the 'invalid_speed' flag was raised over
    now carries a real value. A half-repaired row keeps the flag: readmitting
    it would put a median_speed of 0 into elvik_2019's estimate.
    """
    if "data_quality_flag" not in gdf.columns:
        return gdf
    gdf = gdf.copy()
    repaired = (
        gdf["data_quality_flag"].eq("invalid_speed")
        & gdf["speed_limit"].gt(0)
        & gdf["median_speed"].gt(0)
        & gdf["f85_speed"].gt(0)
    )
    gdf.loc[repaired, "data_quality_flag"] = pd.NA
    return gdf


def select_direction(gdf: pd.DataFrame) -> pd.DataFrame:
    """Fill the single-valued TomTom columns from ONE direction per row.

    The direction with the larger sample wins (fwd on a tie); values are
    copied, never averaged. The posted limit comes from the same direction,
    or from the other one when the selected direction reports none.
    """
    gdf = gdf.copy()
    n_fwd = pd.to_numeric(gdf["tomtom_sample_size_fwd"], errors="coerce").fillna(-1)
    n_bwd = pd.to_numeric(gdf["tomtom_sample_size_bwd"], errors="coerce").fillna(-1)
    has_fwd = gdf["tomtom_segment_id_fwd"].notna()
    has_bwd = gdf["tomtom_segment_id_bwd"].notna()
    use_fwd = has_fwd & (~has_bwd | (n_fwd >= n_bwd))
    use_bwd = has_bwd & ~use_fwd
    gdf["tomtom_selected_direction"] = np.where(use_fwd, "fwd", np.where(use_bwd, "bwd", None))

    for target, source in SELECTED_COLUMNS.items():
        fwd, bwd = gdf[f"{source}_fwd"], gdf[f"{source}_bwd"]
        if isinstance(fwd.dtype, pd.api.extensions.ExtensionDtype) or fwd.dtype == object:
            values = fwd.where(use_fwd, bwd.where(use_bwd))
        else:
            values = np.where(use_fwd, fwd, np.where(use_bwd, bwd, np.nan))
        gdf[target] = values
    gdf["tomtom_segment_id"] = gdf["tomtom_segment_id"].astype("Int64")
    gdf["tomtom_sample_size"] = gdf["tomtom_sample_size"].astype("Int64")

    lim_f = gdf["tomtom_speed_limit_fwd"].astype("Int16")
    lim_b = gdf["tomtom_speed_limit_bwd"].astype("Int16")
    first = lim_f.where(use_fwd, lim_b.where(use_bwd))
    other = lim_b.where(use_fwd, lim_f.where(use_bwd))
    gdf["tomtom_speed_limit"] = first.where(first.notna(), other)
    gdf["tomtom_speed_limit_directions_differ"] = (
        (lim_f.notna() & lim_b.notna() & (lim_f != lim_b)).fillna(False).astype(bool).to_numpy())

    covered = np.where(use_fwd, gdf["tomtom_covered_len_m_fwd"],
                       np.where(use_bwd, gdf["tomtom_covered_len_m_bwd"], 0.0))
    length = pd.to_numeric(gdf.get("shape_length"), errors="coerce") if "shape_length" in gdf else None
    if length is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            gdf["tomtom_coverage_fraction"] = np.where(length > 0, covered / length, 0.0)
    return gdf


def _country_network(country: str, target: gpd.GeoDataFrame, network):
    """(network frame, class column, id column) for one country.

    AUTO_LOAD reads the raw ADB layer: junctions and matches are decided
    against every ADB road, not just the analysed subset, and the raw file
    carries the class column the subset renamed or lacks. A frame passed in
    (tests) is used as-is with the pipeline's own column names.
    """
    if network is AUTO_LOAD:
        net = gpd.read_file(tdi.RAW_ADB_PATHS[country])
        return net, tdi.CLASS_COLUMN[country], tdi.ID_COLUMN
    if network is None:
        return target, "road_class", "segment_id"
    net = network[network["country"] == country] if "country" in network.columns else network
    return net, "road_class", "segment_id"


def add_tomtom_enrichment(gdf: gpd.GeoDataFrame,
                          tomtom=AUTO_LOAD,
                          network=AUTO_LOAD,
                          ways=AUTO_LOAD) -> gpd.GeoDataFrame:
    """The `tomtom` pipeline step. Degrades to a no-op when the layer is absent.

    Each argument defaults to AUTO_LOAD, which reads it from disk per country.
    `tomtom=None` forces the Overture-only path; `network=None` uses `gdf`
    itself as the junction network; `ways=None` skips OSM junctions.
    """
    gdf = gdf.copy()
    # Set before the split, so every piece of a parent shares the parent's id
    # and the reverse mapping has a join key on every row of every country.
    gdf["overture_segment_id"] = gdf["segment_id"].astype(str)
    if tomtom is None:
        print("[tomtom_enrichment] no TomTom layer; Overture-only path")
        return ensure_tomtom_columns(gdf)

    parts = []
    for country, rows in gdf.groupby("country", sort=False):
        rows = rows.reset_index(drop=True)
        if tomtom is AUTO_LOAD:
            tt = tomtom_stats.load_tomtom(country)
        elif tomtom is None:
            tt = None
        else:
            tt = tomtom[tomtom["country"] == country] if "country" in tomtom.columns else tomtom
        if tt is None or len(tt) == 0:
            print(f"[tomtom_enrichment] {country}: no TomTom layer; Overture-only path")
            parts.append(ensure_tomtom_columns(rows))
            continue

        net, class_col, id_col = _country_network(country, rows, network)
        net = net.reset_index(drop=True)
        if ways is AUTO_LOAD:
            import osm_ways
            try:
                way_layer = osm_ways.load_ways(country)
            except FileNotFoundError as exc:
                print(f"[tomtom_enrichment] {country}: {exc}; OSM junctions skipped")
                way_layer = None
        else:
            way_layer = ways
        table, meta, _ = tdi.integrate(net, tt, way_layer, country, class_col=class_col)

        position = {str(v): i for i, v in enumerate(net[id_col])}
        frame_line = rows["segment_id"].astype(str).map(position)
        if frame_line.isna().any():
            missing = rows.loc[frame_line.isna(), "segment_id"].head().tolist()
            raise KeyError(f"{country}: target segments missing from the ADB network: {missing}")
        out = tdi.apply_cuts(rows, table, tt, id_col="segment_id", country=country,
                             frame_line=frame_line.to_numpy(dtype=int),
                             length_cols={"shape_length": 1.0})
        # sample_size_avg and sample_size_total are not apportioned: they count
        # the probes that passed the road, and every piece of the road was passed
        # by them.
        out = out.drop(columns=["adb_parent_id"])
        out = select_direction(out)
        print(f"[tomtom_enrichment] {country}: {meta['n_tomtom_matched']} / "
              f"{meta['n_tomtom_segments']} TomTom segments matched; "
              f"{len(rows)} -> {len(out)} rows")
        parts.append(out)

    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=gdf.crs)
    out = ensure_tomtom_columns(out)
    out = apply_tomtom_speed_limit(out)
    out = backfill_observed_speed(out)
    out = clear_repaired_quality_flags(out)
    out["speed_data_source"] = np.where(
        out["tomtom_segment_id"].notna(), SOURCE_TOMTOM, SOURCE_OVERTURE
    )
    out["tomtom_segment_id"] = out["tomtom_segment_id"].astype("Int64")
    out["segment_id"] = out["segment_id"].astype(str)

    print(
        f"[tomtom_enrichment] {int((out['speed_data_source'] == SOURCE_TOMTOM).sum())}"
        f" / {len(out)} rows corroborated by TomTom; "
        f"{int((out['speed_limit_source'] == 'tomtom').sum())} speed limits taken from TomTom "
        f"({int(out['speed_limit_backfilled'].sum())} of them where ADB had none) and "
        f"{int(out['observed_speed_backfilled'].sum())} observed speeds backfilled"
    )
    return out


def aggregate_to_overture(gdf: pd.DataFrame, columns: list[str],
                          weight_col: str = "shape_length") -> pd.DataFrame:
    """Reverse mapping: collapse pieces back to one row per ADB segment.

    Length-weighted mean for numerics, longest piece wins for categoricals,
    and the worst class wins for an ordered categorical. No pipeline step
    calls it: every output is at sub-segment granularity, and this is the
    documented way back to ADB granularity.
    """
    if "overture_segment_id" not in gdf.columns:
        raise KeyError("aggregate_to_overture needs overture_segment_id; "
                       "run add_tomtom_enrichment first")
    weights = gdf[weight_col].astype(float)
    out = {}
    for key, group in gdf.groupby("overture_segment_id", sort=True):
        w = weights.loc[group.index]
        total = w.sum()
        row = {}
        for column in columns:
            values = group[column]
            if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.ordered:
                row[column] = values.max()
            elif pd.api.types.is_numeric_dtype(values):
                usable = values.notna()
                row[column] = (
                    float((values[usable] * w[usable]).sum() / w[usable].sum())
                    if usable.any() and w[usable].sum() > 0
                    else np.nan
                )
            else:
                row[column] = values.loc[w.idxmax()] if total > 0 else values.iloc[0]
        out[key] = row
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("overture_segment_id")


if __name__ == "__main__":
    from schema import load_target

    enriched = add_tomtom_enrichment(load_target())
    print(enriched["speed_data_source"].value_counts().to_dict())
