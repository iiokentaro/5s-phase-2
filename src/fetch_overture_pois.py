"""Fetch Overture Places POIs of the same kinds as data/processed/osm_pois_*.

OSM kind -> Overture `taxonomy.hierarchy`:
  amenity=school      -> education > place_of_learning > school (incl. preschool)
  amenity=hospital    -> health_care > hospital
  amenity=marketplace -> shopping > market
  shop=*              -> shopping (outside the market subtree)
highway=bus_stop has no Overture Places counterpart and is not fetched.

The taxonomy alone lets through tutoring centres, yoga studios, clinics, pet
hospitals and rooms inside hospitals, at any confidence. Schools and
hospitals therefore also need a name that hits the INCLUDE list and misses the
EXCLUDE list. Sub-district health promotion hospitals (รพ.สต.) are primary care
units, a different kind from OSM amenity=hospital, and are excluded.

Places sharing a category and a normalised name within DEDUPE_RADIUS_M are
one place registered several times; the highest-confidence one is kept.
`confidence` is kept on every row so callers can threshold it.

The S3 read is cached to data/interim/overture_places_{country}_{release}.parquet;
pass --refresh to re-read S3.

Usage (repo root)
-----
    .venv/bin/python src/fetch_overture_pois.py thailand maharashtra [--refresh]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_overture_roads import DEFAULT_RELEASE, _sql_str, connect, region_bounds  # noqa: E402
from poi_sources import BUFFER_CRS  # noqa: E402

PROCESSED_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim")
BOUNDARIES = {
    "thailand": Path("data/raw/THAILAND.geojson"),
    "maharashtra": Path("data/raw/INDIA-Maharashtra.geojson"),
}
OVERTURE_PLACES = "s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
CLOSED_STATUSES = ("permanently_closed", "temporarily_closed")
FLAG_COLS = ["is_school", "is_hospital", "is_marketplace", "is_shop"]
DEDUPE_RADIUS_M = 100


def _words(*terms: str) -> str:
    return r"\b(?:" + "|".join(terms) + r")\b"


SCHOOL_NAME_INCLUDE = re.compile(
    "|".join([
        _words(r"\w*schools?", r"vidyalay\w*", r"vidyalai", r"vidya ?mandir", r"vidya ?niketan",
               r"\w*shala", r"madhyamik", r"prathamik", r"kindergarten", r"pre-?school",
               r"play ?school", r"nursery", r"montessori", r"balwadi", r"kidzee", r"eurokids",
               r"madra?sa", r"madarsa", r"maktab", r"junior college", r"convent"),
        r"โรงเรียน", r"(?:^|\s)ร\.?\s?ร\.?(?=\s|[ก-๙])", r"อนุบาล", r"ศูนย์(?:พัฒนา)?เด็กเล็ก",
        r"ศูนย์อบรมเด็กก่อนเกณฑ์",
        r"विद्यालय", r"शाळा", r"प्रशाला", r"बालवाड", r"पाठशाला",
    ]),
    re.IGNORECASE,
)
SCHOOL_NAME_EXCLUDE = re.compile(
    "|".join([
        _words(r"tutor\w*", r"tuitions?", r"coaching", r"classes", r"kumon", r"abacus",
               r"computers?", r"driving", r"dance", r"music", r"yoga", r"beauty", r"salon",
               r"cooking", r"languages?", r"spoken english", r"ielts", r"robot\w*", r"coding",
               r"swimming", r"martial", r"karate", r"taekwondo", r"studio", r"training"),
        r"กวดวิชา", r"ติวเตอร์", r"สอนดนตรี", r"สอนภาษา", r"โรงเรียนภาษา", r"คุมอง", r"จินตคณิต",
        r"ว่ายน้ำ", r"ศูนย์ฝึก", r"สอนขับรถ",
        r"โรงอาหาร", r"ห้อง", r"อาคาร", r"ชั้น\s?\d",
    ]),
    re.IGNORECASE,
)
HOSPITAL_NAME_INCLUDE = re.compile(
    "|".join([_words(r"hospitals?"), r"โรงพยาบาล", r"(?:^|\s)รพ\.?", r"रुग्णालय", r"इस्पितळ", r"हॉस्पिटल"]),
    re.IGNORECASE,
)
HEALTH_PROMOTION_HOSPITAL = re.compile(r"รพ\.?\s*สต|(?:โรงพยาบาล|รพ\.?\s*)ส่งเสริมสุขภาพ")
HOSPITAL_NAME_EXCLUDE = re.compile(
    "|".join([
        _words(r"animal", r"pets?", r"veterinar\w*", r"vet", r"wards?", r"department",
               r"dept", r"unit"),
        r"สัตว์", r"รพส\.", r"ห้อง", r"ตึก", r"อาคาร", r"ชั้น", r"แผนก", r"หอผู้ป่วย", r"วอร์ด", r"แพทย์แผนไทย",
        r"คล[ิี]นิ[กค]",
        HEALTH_PROMOTION_HOSPITAL.pattern,
    ]),
    re.IGNORECASE,
)


def normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", name).lower())


def _taxonomy_flags(hierarchy) -> tuple[bool, bool, bool, bool]:
    h = list(hierarchy) if hierarchy is not None else []
    school = h[:3] == ["education", "place_of_learning", "school"]
    hospital = h[:2] == ["health_care", "hospital"]
    marketplace = h[:2] == ["shopping", "market"]
    shop = h[:1] == ["shopping"] and not marketplace
    return school, hospital, marketplace, shop


def taxonomy_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = [_taxonomy_flags(h) for h in df["taxonomy_hierarchy"]]
    return pd.DataFrame(flags, columns=FLAG_COLS, index=df.index)


def _name_ok(names: pd.Series, include: re.Pattern, exclude: re.Pattern) -> pd.Series:
    names = names.fillna("")
    return names.map(lambda n: bool(include.search(n)) and not exclude.search(n))


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four is_* flags: taxonomy, plus the name lists for school/hospital."""
    df = df.copy()
    flags = taxonomy_flags(df)
    flags["is_school"] &= _name_ok(df["name"], SCHOOL_NAME_INCLUDE, SCHOOL_NAME_EXCLUDE)
    flags["is_hospital"] &= _name_ok(df["name"], HOSPITAL_NAME_INCLUDE, HOSPITAL_NAME_EXCLUDE)
    for col in FLAG_COLS:
        df[col] = flags[col].astype(bool)
    return df


def dedupe(gdf: gpd.GeoDataFrame, metric_crs: str) -> gpd.GeoDataFrame:
    """Collapse same-category, same-normalised-name places within DEDUPE_RADIUS_M."""
    gdf = gdf.reset_index(drop=True)
    key = gdf[FLAG_COLS].idxmax(axis=1) + "|" + gdf["name"].map(normalize_name)
    named = gdf["name"].map(normalize_name) != ""
    pts = gdf.geometry.to_crs(metric_crs).representative_point()
    left = gpd.GeoDataFrame({"key": key[named]}, geometry=pts[named])
    pairs = gpd.sjoin(left, left, predicate="dwithin", distance=DEDUPE_RADIUS_M)
    pairs = pairs[(pairs["key_left"] == pairs["key_right"]) & (pairs.index != pairs["index_right"])]
    n = len(gdf)
    graph = coo_matrix((np.ones(len(pairs)), (pairs.index.to_numpy(), pairs["index_right"].to_numpy())), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    order = gdf.assign(_c=component, _conf=gdf["confidence"].fillna(-1)).sort_values("_conf", ascending=False)
    keep = order.drop_duplicates("_c").index
    return gdf.loc[sorted(keep)].reset_index(drop=True)


def _query_s3(country: str, release: str) -> gpd.GeoDataFrame:
    con = connect()
    try:
        bbox = region_bounds(con, BOUNDARIES[country])
        s3_path = _sql_str(OVERTURE_PLACES.format(release=release))
        closed = ", ".join(f"'{s}'" for s in CLOSED_STATUSES)
        print(f"[{country}] bbox {bbox} -> S3 places {release}", flush=True)
        df = con.sql(
            f"""
            SELECT
                p.id,
                p.names.primary AS name,
                p.basic_category,
                p.taxonomy.primary AS taxonomy_primary,
                p.taxonomy.hierarchy AS taxonomy_hierarchy,
                p.confidence,
                p.operating_status,
                p.sources[1].dataset AS source_dataset,
                ST_AsWKB(p.geometry) AS geometry_wkb
            FROM read_parquet('{s3_path}', hive_partitioning=1) p, region r
            WHERE p.bbox.xmin <= {bbox['max_lon']}
              AND p.bbox.xmax >= {bbox['min_lon']}
              AND p.bbox.ymin <= {bbox['max_lat']}
              AND p.bbox.ymax >= {bbox['min_lat']}
              AND p.taxonomy.hierarchy[1] IN ('education', 'health_care', 'shopping')
              AND coalesce(p.operating_status, '') NOT IN ({closed})
              AND ST_Intersects(p.geometry, r.geom)
            """
        ).df()
    finally:
        con.close()
    geom = shapely.from_wkb(df.pop("geometry_wkb").map(bytes).to_numpy())
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def load_raw(country: str, release: str, refresh: bool) -> gpd.GeoDataFrame:
    cache = INTERIM_DIR / f"overture_places_{country}_{release}.parquet"
    if cache.exists() and not refresh:
        print(f"[{country}] using cached {cache}", flush=True)
        return gpd.read_parquet(cache)
    raw = _query_s3(country, release)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache)
    return raw


def fetch_overture_pois(
    country: str, release: str = DEFAULT_RELEASE, refresh: bool = False
) -> tuple[gpd.GeoDataFrame, dict]:
    raw = load_raw(country, release, refresh)
    tax = taxonomy_flags(raw)
    classified = classify(raw)
    kept = classified[classified[FLAG_COLS].any(axis=1)]
    gdf = dedupe(kept, BUFFER_CRS[country])
    pts = gdf.geometry.representative_point()
    gdf.insert(1, "lon", pts.x)
    gdf.insert(2, "lat", pts.y)

    health_promotion = tax["is_hospital"] & raw["name"].fillna("").str.contains(HEALTH_PROMOTION_HOSPITAL)
    meta = {
        "release": release,
        "boundary": str(BOUNDARIES[country]),
        "n_rows": int(len(gdf)),
        "n_by_flag": {
            c: {
                "taxonomy_only": int(tax[c].sum()),
                "after_name_lists": int(kept[c].sum()),
                "after_dedupe": int(gdf[c].sum()),
            }
            for c in FLAG_COLS
        },
        "n_health_promotion_hospital_excluded": int(health_promotion.sum()),
        "n_by_source_dataset": {k: int(v) for k, v in gdf["source_dataset"].value_counts().items()},
    }
    return gdf, meta


def write(country: str, gdf: gpd.GeoDataFrame, meta: dict) -> Path:
    out = PROCESSED_DIR / f"overture_pois_{country}.parquet"
    gdf.to_parquet(out)
    (PROCESSED_DIR / f"overture_pois_{country}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[{country}] wrote {out}: {json.dumps(meta['n_by_flag'])}", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("countries", nargs="+", choices=sorted(BOUNDARIES))
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--refresh", action="store_true", help="re-read S3 even if the interim cache exists")
    args = parser.parse_args(argv)
    for country in args.countries:
        gdf, meta = fetch_overture_pois(country, args.release, args.refresh)
        write(country, gdf, meta)


if __name__ == "__main__":
    main()
