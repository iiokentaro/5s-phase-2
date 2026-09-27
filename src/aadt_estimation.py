"""Annual average daily traffic (AADT) for every segment of the deliverable.

The probe data behind the analysis population reports, per segment,
`sample_size_avg`: how many probe vehicles passed one sampling point of it over
the collection period, averaged over the points. The TomTom data was sampled
every 10 km along a section and in each direction, and `sample_size_total` is
the sum over those samples, so it grows with the section's length and with how
many directions were sampled. The average is the count proportional to traffic
volume. It is not a volume, so it is converted with one scale factor per region,
calibrated against the few segments whose traffic was actually counted.

The calibration lives in `docs/AADT Scaling.xlsx` (seven segments: four in
Maharashtra, three in Thailand). For each region the workbook

  1. carries the counted volume from its count year to the year the probes
     were collected, by the product of the annual GDP growth factors of that
     country over the intervening years (World Bank NY.GDP.MKTP.KD.ZG);
  2. divides the sum of those carried volumes by the sum of the matching
     `sample_size_avg` values, giving one scale factor for the region
     (`WeighedScale`, column K of Sheet1);
  3. carries the result to 2025 where the probes are older than that
     (`Scaling_to_2025`, column L).

This module applies steps 2 and 3 to every segment:

    AADT = clip(round(sample_size_avg * weighted_scale * scaling_to_2025),
                AADT_MIN, AADT_MAX)

`sample_size_avg` is a property of the ADB segment the pipeline started from:
when a segment is split, every piece keeps the parent's count, because the same
probes passed every piece of it (segment_localization.py). AADT is a flow along
the road and behaves the same way, so the estimate is made once per parent
segment and given to all of its pieces.

Standard library only at import time: serve_map.py validates a request body
with AadtParams, and it must not pay for pandas to do so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# The regions of the analysis population. Each carries its own scale factor:
# the probe fleet and the collection period differ between them.
COUNTRIES = ("maharashtra", "thailand")

# docs/AADT Scaling.xlsx, Sheet1 column K (WeighedScale), one value per region:
# SUM(AADT_probe_yr) / SUM(SampleSize_avg) over that region's calibration
# segments. Summing before dividing weights each calibration segment by its
# traffic, so the busy segments carry the factor.
WEIGHTED_SCALE = {
    "maharashtra": 1.0953284597075048,
    "thailand": 0.045002453618601286,
}

# docs/AADT Scaling.xlsx, Sheet1 column L (Scaling_to_2025). Maharashtra's
# probes were collected in 2025, so its volumes are already 2025 volumes.
# Thailand's were collected in 2024, so they are carried one year by
# Thailand's 2025 GDP growth factor.
SCALING_TO_2025 = {
    "maharashtra": 1.0,
    "thailand": 1.0244247444990198,
}

# The range a scale factor may be set to from the browser or the command line.
# A factor is vehicles per day per probe: the calibrated values are around 0.05
# to 1.1, and the bound is wide enough to explore well past them.
SCALE_MIN, SCALE_MAX = 0.0, 100.0

# `sample_size_avg` spans seven orders of magnitude, and the calibration has
# four points in Maharashtra and three in Thailand, so a linear factor puts the
# tail of the distribution outside anything a road carries. The estimate is held
# to a plausible range: 200,000 vehicles/day is about what the busiest sections
# of the Thai national highway network and the Indian national highways carry,
# and 10 keeps a segment a probe actually passed from reading as empty.
AADT_MIN, AADT_MAX = 10, 200_000

# Where the calibration was worked out. Committed, so scale_from_workbook() can
# recompute the constants above and the tests can check they still agree.
WORKBOOK_PATH = "docs/AADT Scaling.xlsx"


def _default_scales() -> tuple:
    return tuple((c, WEIGHTED_SCALE[c]) for c in COUNTRIES)


@dataclass(frozen=True)
class AadtParams:
    """The tunable part of the estimate: one scale factor per region.

    Scaling_to_2025 and the clip bounds are fixed. They follow from the
    collection years and from what a road can carry, so they are not choices a
    run makes."""

    weighted_scale: tuple = field(default_factory=_default_scales)

    def scale(self, country: str) -> float:
        """Vehicles per day per probe for `country`, already carried to 2025."""
        factor = next(v for c, v in self.weighted_scale if c == country)
        return factor * SCALING_TO_2025[country]

    def to_dict(self) -> dict:
        return {"weighted_scale": {c: v for c, v in self.weighted_scale}}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict | None) -> "AadtParams":
        """Validate and build; missing keys take their defaults. Raises ValueError."""
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise ValueError("aadt_params must be an object")
        unknown = set(d) - {"weighted_scale"}
        if unknown:
            raise ValueError(f"unknown aadt_params keys: {sorted(unknown)}")
        given = d.get("weighted_scale")
        if given is None:
            given = {}
        if not isinstance(given, dict):
            raise ValueError("weighted_scale must be an object keyed by country")
        bad = set(given) - set(COUNTRIES)
        if bad:
            raise ValueError(f"unknown countries in weighted_scale: {sorted(bad)}")
        scales = tuple(
            (c, float(_number(given.get(c, WEIGHTED_SCALE[c]),
                              f"weighted_scale.{c}", SCALE_MIN, SCALE_MAX)))
            for c in COUNTRIES
        )
        return cls(scales)

    @classmethod
    def from_json(cls, s: str | None) -> "AadtParams":
        return cls.from_dict(json.loads(s) if s else {})


def _number(value, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


DEFAULT = AadtParams()


# --- the calibration, recomputed from the workbook ---------------------------

def scale_from_workbook(path: str = WORKBOOK_PATH) -> AadtParams:
    """Read the calibration sheet and return the scale factors it implies.

    Sheet1 holds one row per calibration segment: `Segment ID` names the region
    in its prefix, `AADT_probe_yr` is the counted volume carried to the probe
    year, and `SampleSize_avg` is that segment's probe count. The factor for a
    region is the sum of the first over the sum of the second, which is what the
    workbook's `WeighedScale` column computes."""
    import openpyxl

    sheet = openpyxl.load_workbook(path, data_only=True)["Sheet1"]
    rows = sheet.iter_rows(values_only=True)
    header = [str(h) if h is not None else "" for h in next(rows)]
    idx = {name: header.index(name) for name in
           ("Segment ID", "AADT_probe_yr", "SampleSize_avg")}
    totals = {c: [0.0, 0.0] for c in COUNTRIES}
    for row in rows:
        segment = row[idx["Segment ID"]]
        if segment is None:
            continue
        country = str(segment).split("_")[0]
        if country not in totals:
            raise ValueError(f"{path}: segment {segment!r} is in no known region")
        totals[country][0] += float(row[idx["AADT_probe_yr"]])
        totals[country][1] += float(row[idx["SampleSize_avg"]])
    for country, (volume, probes) in totals.items():
        if probes <= 0:
            raise ValueError(f"{path}: no calibration segments for {country}")
    return AadtParams(tuple((c, totals[c][0] / totals[c][1]) for c in COUNTRIES))


# --- the estimate ------------------------------------------------------------

def parent_ids(gdf):
    """The ADB segment each row came from, e.g. `maharashtra_2225`.

    Every split appends to `segment_id` (`#j` for the TomTom cut, `-k` for the
    influence zone, `-pNN-k` for the POI zone), so the parent is its first
    field. `overture_segment_id` holds that same number untouched, and is used
    where it is present."""
    import pandas as pd

    if "overture_segment_id" in gdf.columns:
        return gdf["country"].astype(str) + "_" + gdf["overture_segment_id"].astype(str)
    parent = gdf["segment_id"].astype(str).str.extract(r"^([a-z]+_\d+)", expand=False)
    if parent.isna().any():
        bad = sorted(gdf.loc[parent.isna(), "segment_id"].astype(str).unique())[:3]
        raise ValueError(f"segment_id without a parent: {bad}")
    return pd.Series(parent, index=gdf.index)


def estimate(gdf, params: AadtParams = DEFAULT):
    """AADT (vehicles/day) for every row, as int64.

    One estimate per parent segment, given to all of its pieces."""
    import numpy as np
    import pandas as pd

    missing = {"country", "sample_size_avg"} - set(gdf.columns)
    if missing:
        raise ValueError(f"cannot estimate AADT without {sorted(missing)}")
    unknown = sorted(set(gdf["country"].dropna().unique()) - set(COUNTRIES))
    if unknown:
        raise ValueError(f"no scale factor for {unknown}")
    if gdf["sample_size_avg"].isna().any():
        n = int(gdf["sample_size_avg"].isna().sum())
        raise ValueError(f"{n} rows have no sample_size_avg")

    parents = parent_ids(gdf)
    frame = pd.DataFrame({"parent": parents.to_numpy(),
                          "country": gdf["country"].to_numpy(),
                          "probes": gdf["sample_size_avg"].to_numpy(dtype="float64")})
    spread = frame.groupby("parent")["probes"].nunique()
    if (spread > 1).any():
        bad = sorted(spread[spread > 1].index)[:3]
        raise ValueError(f"sample_size_avg differs within a parent segment: {bad}")

    per_parent = frame.drop_duplicates("parent").set_index("parent")
    scale = per_parent["country"].map(lambda c: params.scale(c))
    aadt = np.rint(per_parent["probes"] * scale).clip(AADT_MIN, AADT_MAX).astype("int64")
    return pd.Series(parents.map(aadt).to_numpy(dtype="int64"), index=gdf.index, name="AADT")


def add_aadt_column(gdf, params: AadtParams = DEFAULT):
    """A copy of `gdf` carrying the estimate as its `AADT` column."""
    out = gdf.copy()
    out["AADT"] = estimate(gdf, params)
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="outputs/segments_priority.parquet",
                        help="GeoParquet to read")
    parser.add_argument("--output", default=None,
                        help="where to write the frame with AADT; omit to only report")
    parser.add_argument("--weighted-scale", default=None,
                        help='JSON object of scale factors per country, e.g. \'{"thailand": 0.04}\'')
    parser.add_argument("--xlsx", nargs="?", const=WORKBOOK_PATH, default=None,
                        help=f"recompute the scale factors from a calibration workbook "
                             f"(default {WORKBOOK_PATH})")
    args = parser.parse_args()

    import geopandas as gpd

    if args.xlsx and args.weighted_scale:
        parser.error("--xlsx and --weighted-scale both set the scale factors")
    if args.xlsx:
        params = scale_from_workbook(args.xlsx)
    elif args.weighted_scale:
        params = AadtParams.from_dict({"weighted_scale": json.loads(args.weighted_scale)})
    else:
        params = DEFAULT
    print(f"scale factors: {params.to_json()}")

    gdf = gpd.read_parquet(args.input)
    aadt = estimate(gdf, params)
    print(f"{len(aadt)} rows, {parent_ids(gdf).nunique()} parent segments")
    print(f"AADT min {aadt.min()}, median {int(aadt.median())}, max {aadt.max()}, "
          f"{int((aadt == AADT_MAX).sum())} at the ceiling, {int((aadt == AADT_MIN).sum())} at the floor")
    if args.output:
        gdf.assign(AADT=aadt).to_parquet(args.output)
        print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
