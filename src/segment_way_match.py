"""Attribute each ADB segment to the OSM ways it is made of, by shape.

The premise, and why it holds
-----------------------------
The ADB segments are an Overture Maps export, and Overture's road geometry IS
OSM's road geometry. So a segment is not merely *near* some OSM ways -- it is
built from their vertices, and the correspondence can be established by
coordinate identity. Measured on 203 segments lying
wholly inside a Bangkok bbox: 87.56% of ADB vertices are an OSM node to within
6 decimal degrees, per-segment median 0.948. The residual is the ~1.5 year lag
between the Overture snapshot (Dec 2024) and the OSM pbf (Jun 2026).

Two consequences:

  * A ramp lying beside a mainline is a different way with different vertices,
    so it cannot be attributed to the mainline segment. (A 15 m corridor with a
    0.5 overlap rule attributes it: measured, trunk_link would veto
    `is_divided` on 3,160 Thai segments, primary_link on 1,659, secondary_link
    on 799.)
  * Each way carries the metres of the segment it explains, so the flags can
    be length-weighted on measured evidence.

Why the primary rule is vertex-pair and not edge equality
---------------------------------------------------------
Requiring an ADB edge to equal an OSM edge scored 79.10% against 87.31% for
bare vertex agreement, and 21.5% of the edges it rejected had BOTH endpoints
sitting on real OSM nodes -- the path exists in OSM, with a different number of
intermediate vertices, because one side dropped or added an interior node. So
the rule here asks whether ONE way carries both endpoints of the segment edge,
which is insensitive to what either side did in between.

Match the PARENT, then apportion to the rows
--------------------------------------------
The correspondence with OSM exists only for the ADB segment as Overture
exported it. By the time build() reaches a geometry signal the rows have
already been cut -- by the TomTom layer, and again by the influence-zone
refine -- at positions chosen from other data, which are not OSM nodes. Both
endpoints of such a child are interpolated points that no OSM node can equal.

Measured, matching the children directly:

    Thailand child vertices   2      3      4-5    6-10   >100
    share with no match      98.5%  74.5%   9.3%   5.9%   0.6%

    unsplit Thailand segments   92.7% by length, median 1.000
    split Thailand segments     80.3% by length, median 0.737
    Maharashtra (then without TomTom, so never split)  96.3% by length, median 1.000

That gradient is the splitting, not OSM coverage. So `match_parents` runs once
against the original ADB geometry and records, for every parent, which way
covers which ARC-LENGTH RANGE of it; `apportion_to_rows` then intersects those
ranges with the arc-length range each row occupies in its parent. One
mechanism covers both the TomTom split and the refine split, and a row shorter
than the OSM node spacing still inherits the right way.

Scope
-----
It produces the attribution and measures its own quality. Turning that into
`is_access_controlled` / `is_divided` / `is_grade_separated` belongs to
road_structure_flags, and joining per-mode access belongs to road_access_join.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS
# The bearing test this pipeline already matches linework with.
from tomtom_data_integration import _bearing_diff, _bearings

RULES_PATH = Path(__file__).with_name("match_rules.json")

# Cell packing. lon*1e6 + 180e6 needs 29 bits, lat*1e6 + 90e6 needs 28, so one
# int64 holds a cell id with room to spare and the whole index sorts with
# np.sort, which is cheaper than hashing 28 million tuples.
_LON_OFFSET = 180_000_000
_LAT_OFFSET = 90_000_000
_LAT_BITS = 28


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def required(rules: dict, key: str):
    """A null in match_rules.json means the value has not been decided by
    measurement yet, so raise."""
    value = rules.get(key)
    if value is None:
        raise ValueError(
            f"match_rules.json has {key}=null: it is still undecided. "
            f"Measure it (python src/segment_way_match.py --country both) and "
            f"record the value with its provenance before relying on it."
        )
    return value


# --------------------------------------------------------------------------- #
# The cell index
# --------------------------------------------------------------------------- #

def cell_ids(coords: np.ndarray, decimals: int) -> np.ndarray:
    """Quantize lon/lat to a grid and pack the cell into one int64."""
    scale = 10 ** decimals
    lon = np.rint(coords[:, 0] * scale).astype(np.int64) + _LON_OFFSET
    lat = np.rint(coords[:, 1] * scale).astype(np.int64) + _LAT_OFFSET
    return (lon << _LAT_BITS) | lat


def neighbour_offsets(radius: int) -> np.ndarray:
    """The packed deltas for a (2r+1)^2 cell block, centre first so the exact
    cell is tried before its neighbours."""
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            offsets.append((abs(dx) + abs(dy), (dx << _LAT_BITS) + dy))
    offsets.sort()
    return np.array([o for _, o in offsets], dtype=np.int64)


class WayVertexIndex:
    """Every OSM way vertex, as (cell id -> way row), sorted for searchsorted.

    The query side is expanded to the cell neighbourhood, never this side:
    expanding here would multiply a 28-million-entry index by nine.
    """

    def __init__(self, ways: gpd.GeoDataFrame, decimals: int):
        coords = shapely.get_coordinates(ways.geometry.values)
        counts = shapely.get_num_coordinates(ways.geometry.values)
        way_row = np.repeat(np.arange(len(ways), dtype=np.int32), counts)
        keys = cell_ids(coords, decimals)
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.way_rows = way_row[order]
        self.way_ids = ways["osm_way_id"].to_numpy()

    def lookup(self, query_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(query position, way row) for every hit, one row per pair."""
        lo = np.searchsorted(self.keys, query_keys, side="left")
        hi = np.searchsorted(self.keys, query_keys, side="right")
        n = hi - lo
        hit = n > 0
        if not hit.any():
            return np.empty(0, np.int64), np.empty(0, np.int32)
        starts, lengths = lo[hit], n[hit]
        total = int(lengths.sum())
        # Expand every [start, start+len) range at once. group_start is where
        # each range begins in the OUTPUT, so subtracting it from a running
        # arange leaves the offset within the range.
        group_start = lengths.cumsum() - lengths
        within = np.arange(total) - np.repeat(group_start, lengths)
        idx = np.repeat(starts, lengths) + within
        return np.repeat(np.flatnonzero(hit), lengths), self.way_rows[idx]


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def _segment_edges(segments: gpd.GeoDataFrame, crs: str):
    """Edges in metric CRS for lengths, and the same edges' endpoint vertex
    positions so they can be looked up in the (geographic) cell index."""
    metric = segments.to_crs(crs).geometry.values
    counts = shapely.get_num_coordinates(metric)
    seg_of_vertex = np.repeat(np.arange(len(segments)), counts)
    mcoords = shapely.get_coordinates(metric)

    # an edge is (v, v+1) whenever both vertices belong to the same segment
    v = np.arange(len(mcoords) - 1)
    same = seg_of_vertex[:-1] == seg_of_vertex[1:]
    v = v[same]
    lengths = np.hypot(mcoords[v + 1, 0] - mcoords[v, 0],
                       mcoords[v + 1, 1] - mcoords[v, 1])
    # Arc position of each edge measured from the start of ITS segment, so an
    # edge can later be intersected with the range a child row occupies.
    seg_of_edge = seg_of_vertex[v]
    cum = np.cumsum(lengths)
    first = np.zeros(len(segments))
    starts = cum - lengths
    offset = np.zeros(len(segments))
    boundary = np.flatnonzero(np.r_[True, seg_of_edge[1:] != seg_of_edge[:-1]])
    offset[seg_of_edge[boundary]] = starts[boundary]
    edge_start = starts - offset[seg_of_edge]
    edge_end = edge_start + lengths
    del first
    return v, v + 1, lengths, seg_of_vertex, edge_start, edge_end



def match_parents(parents: gpd.GeoDataFrame, ways: gpd.GeoDataFrame,
                  country: str, rules: dict | None = None):
    """Attribute the ORIGINAL ADB segments to OSM ways.

    Returns (edge_matches, quality).

    edge_matches : one row per (parent, matched edge, way) carrying the arc
                   range [start_m, end_m) of the parent that the way explains.
                   Ranges, not totals, because apportion_to_rows has to know
                   WHERE along the parent each way sits.
    quality      : per parent -- segment_len_m, matched_len_m, unmatched_len_m,
                   matched_frac, osm_way_count.
    """
    rules = rules or load_rules()
    decimals = rules["vertex_cell_decimals"]
    radius = rules["vertex_cell_neighbourhood"]

    index = WayVertexIndex(ways, decimals)
    crs = BUFFER_CRS[country]
    a_idx, b_idx, edge_len, seg_of_vertex, edge_start, edge_end = _segment_edges(parents, crs)

    base = cell_ids(shapely.get_coordinates(parents.geometry.values), decimals)
    offsets = neighbour_offsets(radius)

    # (vertex position, way row) over the cell neighbourhood. A vertex can hit
    # the same way from several cells, so pack the pair into one int64 and
    # deduplicate in 1-D -- np.unique(..., axis=1) would lexsort a 2 x N array
    # of several million columns instead.
    stride = np.int64(len(ways)) + 1
    packed = []
    for off in offsets:
        q, w = index.lookup(base + off)
        if len(q):
            packed.append(q * stride + w.astype(np.int64))

    # Carry on when nothing matches by vertex: the tolerance stage below is
    # what handles that case.
    if packed:
        present = np.unique(np.concatenate(packed))
        vertex_pos, way_row = present // stride, present % stride

        # An edge matches a way when THAT way carries both of its endpoints,
        # which is insensitive to how many intermediate vertices either side
        # has. The second endpoint is a membership test against `present`;
        # a join would explode on a way sharing many vertices with a segment.
        a_pos = np.searchsorted(a_idx, vertex_pos)
        on_edge = ((a_pos < len(a_idx))
                   & (a_idx[np.clip(a_pos, 0, len(a_idx) - 1)] == vertex_pos))
        cand_edge, cand_way = a_pos[on_edge], way_row[on_edge]

        partner = b_idx[cand_edge] * stride + cand_way
        found = np.searchsorted(present, partner)
        keep = ((found < len(present))
                & (present[np.clip(found, 0, len(present) - 1)] == partner))
        edge_hit, way_hit = cand_edge[keep], cand_way[keep]
    else:
        edge_hit = np.empty(0, dtype=np.int64)
        way_hit = np.empty(0, dtype=np.int64)

    exact = pd.DataFrame({
        "seg_index": seg_of_vertex[a_idx[edge_hit]],
        "osm_way_id": index.way_ids[way_hit],
        "start_m": edge_start[edge_hit],
        "end_m": edge_end[edge_hit],
        "match_kind": "exact",
    })

    ways_per_edge = np.bincount(edge_hit, minlength=len(a_idx))
    tol = _tolerance_stage(parents, ways, crs, a_idx, b_idx, edge_start, edge_end,
                           seg_of_vertex, ways_per_edge == 0, rules)
    edge_matches = pd.concat([exact, tol], ignore_index=True) if len(tol) else exact

    covered = ways_per_edge > 0
    if len(tol):
        covered = covered.copy()
        covered[tol["_edge"].to_numpy()] = True
        edge_matches = edge_matches.drop(columns=["_edge"], errors="ignore")
    quality = _quality(parents, edge_len, seg_of_vertex, a_idx, covered, edge_matches)
    return edge_matches, quality


def _tolerance_stage(parents, ways, crs, a_idx, b_idx, edge_start, edge_end,
                     seg_of_vertex, unmatched, rules) -> pd.DataFrame:
    """Recover the edges the vertex rule missed, where OSM plainly still has
    the road.

    Measured on Thailand: for an unmatched parent edge the nearest OSM way is
    a median 1.57 m away, and 100% are within 50 m -- so these are node-level
    edits between the Overture snapshot and the pbf, not missing geometry.
    Rows produced here are labelled `tolerance`, never `exact`, so a consumer
    can weight or exclude them.
    """
    if not unmatched.any():
        return pd.DataFrame()
    max_dist = required(rules, "tolerance_distance_m")
    max_bearing = required(rules, "bearing_tolerance_deg")

    edges = np.flatnonzero(unmatched)
    mcoords = shapely.get_coordinates(parents.to_crs(crs).geometry.values)
    A, B = mcoords[a_idx[edges]], mcoords[b_idx[edges]]
    mid = shapely.points((A + B) / 2)
    # atan2(dy, dx) % 180, matching tomtom_data_integration._bearings' convention exactly
    edge_bearing = np.degrees(np.arctan2(B[:, 1] - A[:, 1], B[:, 0] - A[:, 0])) % 180.0

    way_geom = ways.to_crs(crs).geometry.values
    # EVERY way within the radius, not just the nearest one. Taking only the
    # nearest lets a wrong candidate block a right one: a ramp crossing the
    # midpoint is nearer than the mainline 2 m away, fails the bearing test,
    # and the edge would be abandoned with the correct way never considered.
    q_edge, q_way = shapely.STRtree(way_geom).query(
        mid, predicate="dwithin", distance=max_dist)
    if not len(q_edge):
        return pd.DataFrame()

    cand = way_geom[q_way]
    cand_mid = mid[q_edge]
    dist = shapely.distance(cand_mid, cand)
    diff = _bearing_diff(_bearings(cand, shapely.line_locate_point(cand, cand_mid)),
                         edge_bearing[q_edge])

    ok = diff <= max_bearing
    if not ok.any():
        return pd.DataFrame()
    q_edge, q_way, dist = q_edge[ok], q_way[ok], dist[ok]

    # Closest surviving candidate per edge.
    order = np.lexsort((dist, q_edge))
    q_edge, q_way = q_edge[order], q_way[order]
    first = np.r_[True, q_edge[1:] != q_edge[:-1]]
    q_edge, q_way = q_edge[first], q_way[first]

    hit = edges[q_edge]
    return pd.DataFrame({
        "seg_index": seg_of_vertex[a_idx[hit]],
        "osm_way_id": ways["osm_way_id"].to_numpy()[q_way],
        "start_m": edge_start[hit],
        "end_m": edge_end[hit],
        "match_kind": "tolerance",
        "_edge": hit,
    })


def apportion_to_rows(edge_matches: pd.DataFrame, parents: gpd.GeoDataFrame,
                      rows: gpd.GeoDataFrame, country: str, *,
                      parent_key: str = "overture_segment_id") -> pd.DataFrame:
    """Carry a parent's way attribution down to the rows cut from it.

    A row occupies an arc range of its parent; a way explains an arc range of
    that same parent; the metres a way explains of the row is the overlap. A
    row whose own endpoints are interpolated -- which is every TomTom or refine
    child -- is handled by this and cannot be handled by matching it directly.
    """
    crs = BUFFER_CRS[country]
    parents = parents.reset_index(drop=True)
    for frame, label in ((parents, "parents"), (rows, "rows")):
        if parent_key not in frame.columns:
            raise KeyError(
                f"apportion_to_rows needs `{parent_key}` on {label} to know which "
                f"parent each row was cut from; got {sorted(frame.columns)[:12]}. "
                f"A row without it cannot be attributed, and guessing by position "
                f"would silently pair rows with the wrong parent.")
    pgeom = parents.to_crs(crs).geometry.values
    pos_of_parent = {k: i for i, k in enumerate(parents[parent_key].astype(str))}

    row_parent = rows[parent_key].astype(str).map(pos_of_parent)
    known = row_parent.notna().to_numpy()
    if not known.any():
        return _empty_row_matches()
    rp = row_parent[known].astype(int).to_numpy()
    rgeom = rows.loc[known].to_crs(crs).geometry.values

    # Where the row starts and ends along its parent. line_locate_point on the
    # row's own endpoints keeps this exact for an uncut row and correct for a
    # cut one, without assuming anything about where the cut fell.
    ends = shapely.get_point(rgeom, -1)
    s0 = shapely.line_locate_point(pgeom[rp], shapely.get_point(rgeom, 0))
    s1 = shapely.line_locate_point(pgeom[rp], ends)
    lo, hi = np.minimum(s0, s1), np.maximum(s0, s1)

    frame = pd.DataFrame({"row_index": np.flatnonzero(known), "seg_index": rp,
                          "row_lo": lo, "row_hi": hi})
    joined = frame.merge(edge_matches, on="seg_index", how="inner")
    overlap = (np.minimum(joined["end_m"], joined["row_hi"])
               - np.maximum(joined["start_m"], joined["row_lo"])).clip(lower=0)
    joined["matched_len_m"] = overlap
    joined = joined[joined["matched_len_m"] > 0]
    return (joined.groupby(["row_index", "osm_way_id"], as_index=False)
                  .agg(matched_len_m=("matched_len_m", "sum"),
                       match_kind=("match_kind", "first")))


def _empty_edge_matches() -> pd.DataFrame:
    return pd.DataFrame({"seg_index": pd.Series(dtype="int64"),
                         "osm_way_id": pd.Series(dtype="int64"),
                         "start_m": pd.Series(dtype="float64"),
                         "end_m": pd.Series(dtype="float64"),
                         "match_kind": pd.Series(dtype="object")})


def _empty_row_matches() -> pd.DataFrame:
    return pd.DataFrame({"row_index": pd.Series(dtype="int64"),
                         "osm_way_id": pd.Series(dtype="int64"),
                         "matched_len_m": pd.Series(dtype="float64"),
                         "match_kind": pd.Series(dtype="object")})


def _quality(segments, edge_len, seg_of_vertex, a_idx, matched_edge,
             edge_matches) -> pd.DataFrame:
    n = len(segments)
    seg_of_edge = seg_of_vertex[a_idx]
    total = np.bincount(seg_of_edge, weights=edge_len, minlength=n)
    got = (np.bincount(seg_of_edge[matched_edge], weights=edge_len[matched_edge],
                       minlength=n) if matched_edge.any() else np.zeros(n))
    counts = (edge_matches.groupby("seg_index")["osm_way_id"].nunique()
              .reindex(range(n), fill_value=0).to_numpy()
              if edge_matches is not None and len(edge_matches) else np.zeros(n, dtype=int))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, got / total, 0.0)
    return pd.DataFrame({
        "seg_index": np.arange(n),
        "segment_len_m": total,
        "matched_len_m": got,
        "unmatched_len_m": total - got,
        "matched_frac": frac,
        "osm_way_count": counts,
    })



def _report(country: str, quality: pd.DataFrame, edge_matches: pd.DataFrame) -> dict:
    f = quality["matched_frac"].to_numpy()
    by_len = (quality["matched_len_m"].sum() / max(quality["segment_len_m"].sum(), 1e-9))
    kinds = (edge_matches["match_kind"].value_counts().to_dict()
             if len(edge_matches) else {})
    rep = {
        "country": country,
        "edge_match_kinds": kinds,
        "segments": int(len(quality)),
        "matched_frac_by_length": round(float(by_len), 4),
        "matched_frac_quantiles": {q: round(float(np.quantile(f, q)), 4)
                                   for q in (0.05, 0.25, 0.50, 0.75, 0.95)},
        "segments_fully_matched": int((f >= 0.999).sum()),
        "segments_unmatched": int((f <= 0.001).sum()),
        "ways_per_segment_median": float(np.median(quality["osm_way_count"])),
        "parent_edge_match_rows": int(len(edge_matches)),
    }
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        rep[f"segments_frac_ge_{thr}"] = int((f >= thr).sum())
    return rep


if __name__ == "__main__":
    import argparse
    import time
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    import osm_ways
    from exposure_signals import PBF_PATHS
    from schema import load_target
    from tomtom_enrichment import add_tomtom_enrichment

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--country", choices=[*PBF_PATHS, "both"], default="both")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    # Parents are the ADB geometry as exported: before the TomTom split and
    # before the refine split, which is the only state that corresponds to OSM.
    parents_all = load_target()
    parents_all["overture_segment_id"] = parents_all["segment_id"].astype(str)
    rows_all = add_tomtom_enrichment(load_target())

    reports = []
    for country in ([*PBF_PATHS] if args.country == "both" else [args.country]):
        parents = parents_all[parents_all["country"] == country].reset_index(drop=True)
        rows = rows_all[rows_all["country"] == country].reset_index(drop=True)
        ways = osm_ways.load_ways(country)

        t0 = time.time()
        edge_matches, quality = match_parents(parents, ways, country)
        t_parent = time.time() - t0
        t0 = time.time()
        row_matches = apportion_to_rows(edge_matches, parents, rows, country)
        t_rows = time.time() - t0

        rep = _report(country, quality, edge_matches)
        rep["parent_match_s"] = round(t_parent, 1)
        rep["apportion_s"] = round(t_rows, 1)
        rep["rows"] = int(len(rows))
        rep["rows_with_a_way"] = int(row_matches["row_index"].nunique())
        rep["rows_with_no_way"] = int(len(rows) - row_matches["row_index"].nunique())
        rep["row_match_rows"] = int(len(row_matches))
        reports.append(rep)

        print(f"\n=== {country}: {len(parents):,} parents, {len(rows):,} rows, "
              f"{len(ways):,} ways ===")
        print(json.dumps(rep, indent=2))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json}")
