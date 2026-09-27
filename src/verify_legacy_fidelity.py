"""Verify that the restored legacy rule reproduces the build it came from.

`build_v_safe.py --legacy` recomputes the pre-split V_safe rule into `*_legacy`
columns so the old and new recommended speeds can be read off one table. That
is only worth having if the restored code really is the old rule, so this script
checks it against the output the old rule actually produced: the committed
`data/processed/segments_v_safe.parquet` at commit LEGACY_BASELINE_COMMIT, the
last build before the flag split.

★ Why the current build cannot be compared directly ★
  segment_localization splits rows on `is_vru | near_junction`, and the old VRU
  mask excluded trunk outright -- so the old rule produced a different row set
  (68,852 rows against today's 102,508). The `*_legacy` columns are therefore the
  old rule read on the CURRENT rows, and their aggregate counts are not the old
  build's. What can be checked exactly is the rule itself, on the old rows, which
  is what this script does.

The baseline is read from git history into memory.

Usage (from the repo root):
    python src/verify_legacy_fidelity.py
Exit code 0 = PASS, 1 = MISMATCH.
"""

import argparse
import io
import logging
import subprocess
import sys
import warnings

import geopandas as gpd
import pandas as pd

sys.path.insert(0, "src")
from exposure_signals import legacy_vru_mask  # noqa: E402
from junction_speed_cap import JUNCTION_V_SAFE_CAP, LEGACY_EXCLUDED_ROAD_CLASS  # noqa: E402
from road_separation import add_road_structure  # noqa: E402
from safe_speed import add_v_safe  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# The last build before is_separated was split into is_access_controlled /
# is_divided. Its parquet is the oracle for everything below.
LEGACY_BASELINE_COMMIT = "7cd7d15"
BASELINE_PATH = "data/processed/segments_v_safe.parquet"
EXPECTED_BASELINE_ROWS = 68852


def load_baseline() -> gpd.GeoDataFrame:
    """The committed pre-split build, read from git history into memory."""
    result = subprocess.run(
        ["git", "show", f"{LEGACY_BASELINE_COMMIT}:{BASELINE_PATH}"],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not read {LEGACY_BASELINE_COMMIT}:{BASELINE_PATH} from git "
            f"(exit {result.returncode}): {result.stderr.decode(errors='replace').strip()}\n"
            "The baseline is the whole oracle here, so there is no fallback: fix git "
            "(a shallow clone will not have the commit) and re-run."
        )
    blob = result.stdout
    gdf = gpd.read_parquet(io.BytesIO(blob))
    log.info("baseline %s:%s -> %d rows", LEGACY_BASELINE_COMMIT, BASELINE_PATH, len(gdf))
    return gdf


def _report(name: str, mismatches: int, total: int, failures: list) -> None:
    verdict = "OK" if mismatches == 0 else f"{mismatches} differ"
    log.info("  %-34s %6d / %6d  %s", name, total - mismatches, total, verdict)
    if mismatches:
        failures.append(f"{name}: {mismatches} / {total} rows differ")


def check_rule(baseline: gpd.GeoDataFrame, failures: list) -> None:
    """Level A -- the decision itself.

    Feeds the baseline's own inputs (road_class, is_separated, is_vru, f85_speed)
    through the restored legacy chain and demands the baseline's own verdict back.
    The junction cap is replayed from the baseline's `near_junction`, which is
    already post-exclusion, so this isolates the classification from the spatial
    work.
    """
    log.info("Level A -- collision type / V_safe from the baseline's own inputs")
    frame = pd.DataFrame({
        "road_class": baseline["road_class"],
        "f85_speed": baseline["f85_speed"],
        "is_separated_legacy": baseline["is_separated"].fillna(False).astype(bool),
        "is_vru_legacy": baseline["is_vru"].fillna(False).astype(bool),
        # add_v_safe computes the current rule too; these keep it fed with the
        # safe defaults, and none of its output is read here.
        "is_access_controlled": False,
        "is_divided": False,
        "is_vru": False,
    })
    out = add_v_safe(frame, include_legacy=True)

    capped = baseline["near_junction"].fillna(False).astype(bool) & (out["v_safe_legacy"] > JUNCTION_V_SAFE_CAP)
    out.loc[capped, "v_safe_legacy"] = JUNCTION_V_SAFE_CAP
    out.loc[capped, "collision_type_legacy"] = "side_impact"
    out.loc[capped, "v_safe_basis_legacy"] = "side_impact:junction_buffer"

    total = len(baseline)
    _report("motorway_tag_suspect", int((out["motorway_tag_suspect_legacy"]
                                         != baseline["motorway_tag_suspect"]).sum()), total, failures)
    for col in ("collision_type", "v_safe", "v_safe_basis"):
        _report(col, int((out[f"{col}_legacy"] != baseline[col]).sum()), total, failures)


def check_mask(baseline: gpd.GeoDataFrame, failures: list) -> None:
    """Level B -- the VRU mask, checked as an invariant of the baseline itself.

    The baseline stores is_mapillary_vru already masked, so the mask cannot be
    replayed from it; what it can prove is that `legacy_vru_mask` describes the
    same rows the old build actually masked.
    """
    log.info("Level B -- VRU mask invariants on the baseline")
    frame = pd.DataFrame({
        "road_class": baseline["road_class"],
        "is_separated_legacy": baseline["is_separated"].fillna(False).astype(bool),
    })
    masked = legacy_vru_mask(frame)
    is_vru = baseline["is_vru"].fillna(False).astype(bool)
    unmasked_expected = (baseline["is_mapillary_vru"].fillna(False).astype(bool)
                         | baseline["is_school"].fillna(False).astype(bool))

    _report("is_vru False on every masked row", int((masked & is_vru).sum()), len(baseline), failures)
    _report("is_vru == mapillary|school elsewhere",
            int((~masked & (is_vru != unmasked_expected)).sum()), len(baseline), failures)


def check_junction_exclusion(baseline: gpd.GeoDataFrame, failures: list) -> None:
    """Level C -- the junction exclusion, as an invariant of the baseline.

    `near_junction` is stored post-exclusion, so no excluded row may carry it.
    """
    log.info("Level C -- junction exclusion invariants on the baseline")
    excluded = ((baseline["road_class"] == LEGACY_EXCLUDED_ROAD_CLASS)
                | baseline["is_grade_separated"].fillna(False).astype(bool))
    near = baseline["near_junction"].fillna(False).astype(bool)
    _report("near_junction False on excluded rows", int((excluded & near).sum()), len(baseline), failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    baseline = load_baseline()
    failures: list = []
    if len(baseline) != EXPECTED_BASELINE_ROWS:
        failures.append(f"baseline has {len(baseline)} rows, expected {EXPECTED_BASELINE_ROWS} "
                        "-- is LEGACY_BASELINE_COMMIT still the pre-split build?")

    check_rule(baseline, failures)
    check_mask(baseline, failures)
    check_junction_exclusion(baseline, failures)

    print()
    if failures:
        print("MISMATCH -- the restored legacy rule does not reproduce "
              f"{LEGACY_BASELINE_COMMIT}:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"PASS -- the restored legacy rule reproduces {LEGACY_BASELINE_COMMIT} "
          f"on all {len(baseline)} rows of the pre-split build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
