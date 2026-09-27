"""Verify that the osmium-prefiltered pbf yields output identical to the
committed data/processed/osm_*.parquet baselines.

★ WHAT THIS SCRIPT CAN AND CANNOT DETECT ★
  It answers exactly one question: does `prefilter_pbf.py`'s osmium tags-filter
  drop anything the reader would otherwise have seen? For that question,
  comparing the same reader on the filtered pbf against the same reader on the
  full pbf is the right test, and the committed parquets are a valid oracle:
  they came off the FULL pbf through the very code paths under test.

  It CANNOT detect what the reader itself leaves out, because both sides of
  every comparison use the same reader. On `osm_roads_*.parquet` it returns
  PASS although pyrosm 0.9.1's `get_data_by_custom_criteria` returns only ways
  whose node count is exactly 2 -- 12.5% of the network by length -- because
  both sides miss the same ways.

  The reader question is answered by comparisons against something else:
    * osm_way_source.assert_conservation  -- checks the extract against the
      .pbf's OWN declared counts, on every extraction.
    * tests/test_osm_way_source_equivalence.py -- runs two independent readers
      over one fixture and diffs them.

  `roads` is out of scope: the way layer the pipeline reads is built by
  src/osm_ways.py from its own highway-filtered pbf.

★ Regenerated in memory ★
  Everything is regenerated in memory and compared, because
  `python src/exposure_signals.py` would overwrite the very baselines under
  check. --save-dir (off by default) dumps the regenerated frames elsewhere
  for manual inspection.

Recovery if a baseline is ever clobbered:
    git checkout -- data/processed/osm_*.parquet

Usage (from the repo root):
    python src/verify_prefilter_equivalence.py --country maharashtra
    python src/verify_prefilter_equivalence.py --country thailand
Exit code 0 = PASS, 1 = MISMATCH.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, "src")
from exposure_signals import (  # noqa: E402
    BUFFER_CRS,
    PBF_PATHS,
    PROCESSED_DIR,
    _is_grade_separated,
    extract_raw,
    load_cached_signals,
    split_signals,
)
from prefilter_pbf import ensure_filtered_pbf  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# `roads` deliberately absent -- see the docstring. The three that remain are
# produced by exposure_signals.extract_raw, which does still read the
# prefiltered pbf, so the prefilter question is live for them.
OUTPUTS = ("pois", "pedestrian_ways", "crossings_at_grade")

# Row counts of the committed baselines, restated so a silently-regenerated
# baseline cannot make the comparison vacuously pass.
EXPECTED_ROWS = {
    "thailand": {"pois": 71041, "pedestrian_ways": 44798,
                 "crossings_at_grade": 22604},
    "maharashtra": {"pois": 26306, "pedestrian_ways": 6641,
                    "crossings_at_grade": 1603},
}
# Grade-separated crossing nodes excluded / total crossing nodes, as printed
# by exposure_signals.split_signals.
EXPECTED_EXCLUSIONS = {"thailand": (57, 22661), "maharashtra": (4, 1607)}
# One pyrosm call-order artefact, NOT a prefilter effect.
#
# Node 6788786522 (a shop=computer POI in Powai) carries an OSM tag whose key is
# literally `geometry`, colliding with the GeoDataFrame's reserved column name.
# It is the only such object in either country's pbf (Thailand has none).
# Whether pyrosm 0.9.1 keeps that key in the catch-all `tags` depends on the
# call order within the process, not on which pbf it reads (measured 2026-09-13):
#
#   both countries, thailand first (what exposure_signals.__main__ does)
#       -> key absent, byte-identical to the committed baseline
#   maharashtra alone in a fresh process, PREFILTERED pbf   -> key present
#   maharashtra alone in a fresh process, FULL pbf (control, 1122 s) -> key present
#
# The last two agree, and `osmium getid -f opl` shows the raw tags are identical
# in the full and filtered pbfs -- so the input file is not the variable and the
# prefilter is not implicated. The committed baseline was produced by the
# both-countries __main__ loop, which is why a `--country maharashtra` run is the
# only invocation that surfaces this.
#
# Harmless downstream regardless: `tags` is only ever read for lanes:divided /
# dual_carriageway / bridge / tunnel / layer (road_separation), and this is a POI.
# Scoped to the exact country/output/key/id so any other tag difference still fails.
KNOWN_BASELINE_TAG_DIFFS = {
    ("maharashtra", "pois"): {"node:6788786522": {"geometry"}},
}

# osm_type histograms -- the dedicated regression detector for the nwr/ prefix.
EXPECTED_OSM_TYPES = {
    "thailand": {
        "pois": {"node": 49546, "way": 20632, "relation": 863},
        "pedestrian_ways": {"way": 44797, "relation": 1},
        "crossings_at_grade": {"node": 22604},
    },
    "maharashtra": {
        "pois": {"node": 23103, "way": 3177, "relation": 26},
        "pedestrian_ways": {"way": 6640, "relation": 1},
        "crossings_at_grade": {"node": 1603},
    },
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _as_tag_dict(value) -> dict:
    """Normalise pyrosm's catch-all `tags` cell to a plain dict.

    Compared as dicts so key ordering inside the JSON string can never cause a
    false failure.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {"__unparsable__": value}
        return parsed if isinstance(parsed, dict) else {"__nondict__": value}
    return {"__unexpected_type__": repr(value)}


def _keyed(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Index by (osm_type, id) and sort, so both sides align row-for-row."""
    out = df.copy()
    # A string key, not a tuple: a tuple index makes `.loc[key]` look like a
    # multi-axis indexer to pandas ("Too many indexers") in the diff reporting.
    out["__key"] = out["osm_type"].astype(str) + ":" + out["id"].astype("int64").astype(str)
    return out.set_index("__key").sort_index()


def _fmt(counter: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))


# --------------------------------------------------------------------------- #
# per-output comparison
# --------------------------------------------------------------------------- #

def compare(country: str, name: str, base: gpd.GeoDataFrame, new: gpd.GeoDataFrame,
            failures: list[str]) -> dict:
    """Run every check for one output. Appends human-readable failures in place."""
    tag = f"{country}/{name}"
    result = {"country": country, "output": name,
              "rows": f"{len(base)}/{len(new)}", "keys": "-", "cols": "-",
              "tags": "-", "derived": "-", "geom": "-"}

    # 1. row count (weak; the expected value guards a regenerated baseline)
    expected = EXPECTED_ROWS[country][name]
    if len(base) != expected:
        failures.append(f"{tag}: baseline has {len(base)} rows, expected {expected} "
                        "-- the committed parquet is not the reference data")

    # 2. key set -- the primary check
    kb, kn = set(base.index), set(new.index)
    missing, extra = sorted(kb - kn), sorted(kn - kb)
    result["keys"] = f"-{len(missing)}/+{len(extra)}"
    if missing:
        failures.append(f"{tag}: {len(missing)} keys MISSING from the prefiltered run "
                        f"(subset bug -- widen FILTER_EXPRESSIONS): {missing[:10]}")
    if extra:
        failures.append(f"{tag}: {len(extra)} UNEXPECTED extra keys -- pyrosm re-filters, "
                        f"so this should be impossible; re-examine the superset "
                        f"premise before shipping: {extra[:10]}")

    # 3. osm_type histogram (nwr/ prefix regression detector)
    hb = base["osm_type"].value_counts().to_dict()
    hn = new["osm_type"].value_counts().to_dict()
    exp_hist = EXPECTED_OSM_TYPES[country][name]
    if hb != hn:
        failures.append(f"{tag}: osm_type histogram differs -- base [{_fmt(hb)}] vs new [{_fmt(hn)}]")
    if hn != exp_hist:
        failures.append(f"{tag}: osm_type histogram [{_fmt(hn)}] != expected [{_fmt(exp_hist)}]")
    log.info("  %-18s osm_type base[%s] new[%s]", name, _fmt(hb), _fmt(hn))

    if missing or extra:
        # Row-aligned checks below are meaningless without matching key sets.
        result.update(cols="skip", tags="skip", derived="skip", geom="skip")
        return result

    base, new = base.loc[sorted(kb)], new.loc[sorted(kb)]

    # 4. shared attribute columns (NaN-aware). Column sets are intersected per
    #    country: maharashtra's roads baseline has no `changeset`, and the roads
    #    baseline predates is_grade_separated_way/bridge/tunnel/layer.
    only_base = sorted(set(base.columns) - set(new.columns) - {"__key"})
    only_new = sorted(set(new.columns) - set(base.columns) - {"__key"})
    if only_base or only_new:
        log.info("  %-18s columns only in base=%s only in new=%s", name, only_base, only_new)
    shared = sorted((set(base.columns) & set(new.columns)) - {"geometry", "tags", "__key"})
    col_bad = []
    for col in shared:
        a, b = base[col], new[col]
        same = (a.isna() & b.isna()) | (a == b)
        n_bad = int((~same).sum())
        if n_bad:
            col_bad.append(f"{col}({n_bad})")
            ex = base.index[~same][:3]
            failures.append(f"{tag}: column '{col}' differs on {n_bad} rows, e.g. "
                            f"{[(k, a.loc[k], b.loc[k]) for k in ex]}")
    result["cols"] = "OK" if not col_bad else ",".join(col_bad)

    # 5. `tags` catch-all -- the -t/--remove-tags guard
    tb = base["tags"].map(_as_tag_dict)
    tn = new["tags"].map(_as_tag_dict)
    allowed = KNOWN_BASELINE_TAG_DIFFS.get((country, name), {})

    def _minus(d: dict, drop) -> dict:
        return {k: v for k, v in d.items() if k not in drop} if drop else d

    # Never assign a dict back into a Series (pandas expands it into a Series);
    # compare row by row instead, forgiving only the allow-listed keys on the
    # allow-listed objects.
    differing, known_hits = [], 0
    for key, db, dn in zip(tb.index, tb, tn):
        if db == dn:
            continue
        drop = allowed.get(key)
        if drop and _minus(db, drop) == _minus(dn, drop):
            known_hits += 1
        else:
            differing.append(key)
    if known_hits:
        log.info("  %-18s %d known pre-existing baseline tag diff(s) allowed "
                 "(see KNOWN_BASELINE_TAG_DIFFS)", name, known_hits)
    n_tag_bad = len(differing)
    result["tags"] = ("OK" if not known_hits else f"OK ({known_hits} known)") \
        if n_tag_bad == 0 else f"{n_tag_bad} differ"
    if n_tag_bad:
        failures.append(f"{tag}: `tags` differs on {n_tag_bad} rows (did -t/--remove-tags "
                        f"slip in, or was reference completion suppressed?), e.g. "
                        f"{[(k, tb.loc[k], tn.loc[k]) for k in differing[:3]]}")
    forgiven = set().union(*allowed.values()) if allowed else set()
    keys_b = (set().union(*tb) if len(tb) else set()) - forgiven
    keys_n = (set().union(*tn) if len(tn) else set()) - forgiven
    if keys_b != keys_n:
        failures.append(f"{tag}: tag-key universe differs -- only base "
                        f"{sorted(keys_b - keys_n)[:10]}, only new {sorted(keys_n - keys_b)[:10]}")
    # 6. derived booleans, recomputed on BOTH sides (the baseline predates some
    #    of these columns, so reading them would not exercise the tags path)
    derived: dict[str, tuple] = {}
    if name == "pedestrian_ways":
        derived["grade_separated"] = (_is_grade_separated,)
    d_bad = []
    for col, (fn,) in derived.items():
        a = base.apply(fn, axis=1)
        b = new.apply(fn, axis=1)
        n = int((a != b).sum())
        log.info("  %-18s %s n_true base=%d new=%d", name, col, int(a.sum()), int(b.sum()))
        if n:
            d_bad.append(f"{col}({n})")
            failures.append(f"{tag}: derived '{col}' differs on {n} rows "
                            f"(base n_true={int(a.sum())}, new n_true={int(b.sum())})")
    result["derived"] = "n/a" if not derived else ("OK" if not d_bad else ",".join(d_bad))

    # 7. geometry -- exact equality is the expectation: osmium copies node
    #    coordinates verbatim and nothing in this path reprojects or rounds.
    eq = base.geometry.geom_equals_exact(new.geometry, tolerance=0)
    n_geom = int((~eq).sum())
    if n_geom == 0:
        result["geom"] = "exact"
    else:
        loose = int((~base.geometry.geom_equals_exact(new.geometry, tolerance=1e-9)).sum())
        maxd = float(base.geometry.distance(new.geometry).max())
        result["geom"] = f"{n_geom} differ"
        failures.append(f"{tag}: {n_geom} geometries differ at tolerance=0 "
                        f"({loose} at 1e-9, max distance {maxd:g}). Exact equality is "
                        "expected here -- treat any deviation as a real finding "
                        "(-R slipped in, or nodes are missing).")
    return result


# --------------------------------------------------------------------------- #
# per-country driver
# --------------------------------------------------------------------------- #

def verify_country(country: str, force: bool, save_dir: Path | None) -> tuple[list[dict], list[str]]:
    failures: list[str] = []
    log.info("=== %s ===", country)

    source = PBF_PATHS[country]
    t0 = time.monotonic()
    filtered = ensure_filtered_pbf(source, force=force)
    log.info("prefilter resolved in %.1fs -> %s (%.1f MB)", time.monotonic() - t0,
             Path(filtered).name, Path(filtered).stat().st_size / 1e6)
    if filtered == str(source):
        raise RuntimeError("the prefilter did not run -- there is nothing to verify")

    # regenerate from the filtered pbf (in memory only)
    t0 = time.monotonic()
    raw = extract_raw(country, pbf_path=filtered)
    new = split_signals(raw, BUFFER_CRS[country])
    log.info("%s: regenerated from the filtered pbf in %.1fs", country, time.monotonic() - t0)

    base = load_cached_signals(country)

    # 8. documented oracle: split_signals' grade-separation exclusion counts
    n_at_grade = len(new["crossings_at_grade"])
    exp_excluded, exp_total = EXPECTED_EXCLUSIONS[country]
    if n_at_grade != exp_total - exp_excluded:
        failures.append(f"{country}: at-grade crossings {n_at_grade} != "
                        f"{exp_total} - {exp_excluded} = {exp_total - exp_excluded} "
                        "(README.md L367 oracle)")

    rows = []
    for name in OUTPUTS:
        b, n = _keyed(base[name]), _keyed(new[name])
        before = len(failures)
        r = compare(country, name, b, n, failures)
        r["verdict"] = "PASS" if len(failures) == before else "FAIL"
        rows.append(r)
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            new[name].to_parquet(save_dir / f"osm_{name}_{country}.parquet")
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify prefiltered-pbf extraction against the committed parquets"
    )
    parser.add_argument("--country", choices=list(PBF_PATHS), default=None,
                        help="target country (both if omitted)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the filtered pbf even if the cache is fresh")
    parser.add_argument("--save-dir", default=None,
                        help="optional directory to dump the regenerated frames into "
                             "(NEVER data/processed -- that holds the baselines)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None and save_dir.resolve() == Path(PROCESSED_DIR).resolve():
        parser.error("--save-dir must not be data/processed (it holds the baselines)")

    countries = [args.country] if args.country else list(PBF_PATHS)
    started = time.monotonic()
    all_rows, all_failures = [], []
    for country in countries:
        rows, failures = verify_country(country, args.force, save_dir)
        all_rows += rows
        all_failures += failures

    hdr = f"{'country':<12} {'output':<19} {'rows base/new':<16} {'keys':<10} " \
          f"{'cols':<10} {'tags':<10} {'derived':<28} {'geom':<10} verdict"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(f"{r['country']:<12} {r['output']:<19} {r['rows']:<16} {r['keys']:<10} "
              f"{r['cols']:<10} {r['tags']:<10} {r['derived']:<28} {r['geom']:<10} {r['verdict']}")
    print(f"\ntotal verify time: {time.monotonic() - started:.1f}s")

    if all_failures:
        print(f"\nMISMATCH -- {len(all_failures)} finding(s):")
        for f in all_failures:
            print(f"  - {f}")
        print("\nTriage: missing keys -> the filter is a subset; inspect with "
              "`osmium getid -r <filtered.pbf> n<id>`, widen FILTER_EXPRESSIONS and "
              "bump FILTER_VERSION. tags differ -> -t slipped in. geometry differs -> "
              "-R slipped in or nodes are missing.")
        sys.exit(1)

    known = sum("known" in r["tags"] for r in all_rows)
    print("\nPASS -- the prefiltered pbf reproduces every committed baseline"
          + ("." if not known else
             f", aside from {known} allow-listed pre-existing baseline tag diff(s) "
             "that a full-pbf control run reproduces too (see KNOWN_BASELINE_TAG_DIFFS)."))
    sys.exit(0)


if __name__ == "__main__":
    main()
