"""Cut ADB segments into uninterrupted segments and attach TomTom Traffic Stats.

What an uninterrupted segment is
--------------------------------
A stretch of ADB road between two at-grade junctions, over which the road
class does not change. Nothing on such a stretch gives traffic a reason to
change character, so a TomTom measurement taken anywhere on it describes all
of it -- including the parts of it the TomTom line does not run beside.

Junctions come from two sources only:
  * ADB with ADB -- a point where three or more ADB arms meet (shared vertex,
    or a line end touching another line's interior: a T junction);
  * ADB with OSM -- an OSM node shared by three or more arms of public,
    motor-accessible OSM road, lying on an OSM way the ADB line is built from.
TomTom lines are NOT a junction source: a TomTom line routinely crosses the
very ADB line it describes, a few metres off at a shallow angle.
Grade separation needs no special case: a bridge does not share a node with
the road below it, in OSM or in ADB, so it never produces three arms.

How a TomTom segment is matched
-------------------------------
From the TomTom side, by shape. Each TomTom line is sampled (every vertex plus
evenly spaced points, never fewer than min_samples_per_tomtom) and each sample
goes to the nearest ADB piece within max_distance_m whose local bearing agrees.
The samples are grouped by uninterrupted segment; each group is a PART. A part
that is both under min_part_frac of the TomTom length and no longer than
max_distance_m is an overhang past a junction and is dropped. A part
matches when the discrete Frechet distance between it and the stretch of the
uninterrupted segment it projects onto is within max_distance_m.

Frechet compares shape and order at once, and every threshold is either a
distance or a share of the TomTom segment's own length -- there is no absolute
length floor, so a 3 m TomTom segment is judged exactly like a 3 km one:
  * a TomTom line crossing at a junction projects onto a near-point, and its
    far end is then far from that point -> rejected (a short one fails the
    bearing test instead);
  * a TomTom line weaving across its ADB line stays close, in order -> matched;
  * a road ADB does not carry, passing close for a while, leaves most of its
    length unattributed -> rejected by min_tomtom_frac.

No TomTom value is ever mixed with another
------------------------------------------
Not across TomTom segments, and not across the two directions of one. Where
two TomTom segments serve one uninterrupted segment, the ADB line is cut
between their ranges; each direction of travel is partitioned on its own, and
the values land in *_fwd / *_bwd columns (fwd = travel in the direction the
ADB line is drawn).

Output identity
---------------
An ADB segment no TomTom segment reaches is returned untouched, id included.
A cut one yields "{id}#{j}", j counted along the line from 0 -- the convention
segment_localization already normalises. Runs of unmatched pieces within one
parent are joined back together, so only matched uninterrupted segments cause
cuts.

Output files
------------
data/processed/adb_segments_tomtom_{country}.parquet is the full result. Next to
it, adb_segments_tomtom_{country}_kepler.parquet holds the same rows for
kepler.gl: text columns as plain strings and the list columns (the TomTom
speed percentiles) dropped.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.ops import substring

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS  # noqa: E402
from kepler_parquet import write_kepler_parquet  # noqa: E402

RULES_PATH = Path(__file__).with_name("tomtom_match_rules.json")

RAW_ADB_PATHS = {
    "thailand": "data/raw/ADB_Innovation_Thailand.geojson",
    "maharashtra": "data/raw/ADB_Innovation_Maharashtra.geojson",
}
# Maharashtra's RoadClass is filled on 4,010 of 14,082 rows; `class` on all of
# them. Thailand's export has RoadClass only.
CLASS_COLUMN = {"thailand": "RoadClass", "maharashtra": "class"}
ID_COLUMN = "OBJECTID"
OUTPUT_DIR = Path("data/processed")

# --- which OSM ways can form a junction ----------------------------------- #
# Public roads a car can drive on. Excluded:
#   service  -- driveways, parking aisles, private access
#   track    -- agricultural / forestry tracks
#   footway, path, pedestrian, steps, cycleway, bridleway, corridor, platform
#            -- not for motor vehicles (crossings are footway=crossing)
#   busway   -- buses only
#   construction, proposed -- not open
CAR_HIGHWAYS = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    "unclassified", "residential", "living_street", "road",
})
# Access values that make a way non-public for this purpose.
PRIVATE_ACCESS = frozenset({
    "private", "no", "customers", "delivery", "permit", "agricultural", "forestry",
})
PRIVATE_MOTOR = frozenset({"private", "no"})

DIRECTIONS = ("along", "against")
SIDES = ("fwd", "bwd")
# Per-direction TomTom values carried onto each ADB piece.
SIDE_VALUE_COLUMNS = [
    "tomtom_signed_segment_id",
    "tomtom_speed_limit",
    "tomtom_mean_speed",
    "tomtom_sd_speed",
    "tomtom_sample_size",
    "tomtom_median_speed",
    "tomtom_harmonic_speed",
    "tomtom_p85_speed",
    "tomtom_speed_percentiles",
]


def load_rules(path: Path = RULES_PATH) -> dict:
    rules = json.loads(Path(path).read_text())
    for key, value in rules.items():
        if not key.startswith("_") and value is None:
            raise ValueError(
                f"{path.name} has {key}=null: it is still undecided. Measure it "
                f"and record the value with its provenance before relying on it.")
    return rules


# --------------------------------------------------------------------------- #
# Bearings (segment_way_match imports them)
# --------------------------------------------------------------------------- #

def _bearings(lines, positions_m, window_m: float = 20.0) -> np.ndarray:
    """Local bearing (degrees, 0-180) of each line at its given measure."""
    lengths = shapely.length(lines)
    lo = np.clip(positions_m - window_m, 0.0, lengths)
    hi = np.clip(positions_m + window_m, 0.0, lengths)
    # A window that collapsed to a point carries no direction; widen it to the
    # whole line (a point would give a meaningless 0 degrees).
    degenerate = (hi - lo) < 1e-9
    lo = np.where(degenerate, 0.0, lo)
    hi = np.where(degenerate, lengths, hi)
    a = shapely.get_coordinates(shapely.line_interpolate_point(lines, lo))
    b = shapely.get_coordinates(shapely.line_interpolate_point(lines, hi))
    return np.degrees(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])) % 180.0


def _bearing_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest angle between two 0-180 bearings."""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

class _DSU:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _vertex_table(geoms) -> pd.DataFrame:
    """Every vertex of every line, with its measure along the line.

    Consecutive repeated coordinates are dropped first: a zero-length edge
    would otherwise make one visit look like two and inflate an arm count.
    """
    coords, index = shapely.get_coordinates(geoms, return_index=True)
    keep = np.ones(len(coords), dtype=bool)
    same_line = index[1:] == index[:-1]
    repeated = same_line & np.all(coords[1:] == coords[:-1], axis=1)
    keep[1:] = ~repeated
    coords, index = coords[keep], index[keep]
    step = np.zeros(len(coords))
    same_line = index[1:] == index[:-1]
    step[1:] = np.where(same_line, np.hypot(*(coords[1:] - coords[:-1]).T), 0.0)
    cum = np.cumsum(step)
    first = np.r_[True, index[1:] != index[:-1]]
    last = np.r_[index[1:] != index[:-1], True]
    start_of_line = np.maximum.accumulate(np.where(first, np.arange(len(coords)), 0))
    measure = cum - cum[start_of_line]
    return pd.DataFrame({"line": index, "x": coords[:, 0], "y": coords[:, 1],
                         "m": measure, "first": first, "last": last})


def _cluster(points: np.ndarray, radius: float) -> np.ndarray:
    """Label points so that any two within `radius` share a label."""
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points)))
    return connected_components(graph, directed=False)[1]


def _visits(verts: pd.DataFrame, cluster: np.ndarray) -> pd.DataFrame:
    """Collapse consecutive vertices of one line in one cluster into a visit,
    and give each visit its arm count: 2 through the middle of a line, 1 at a
    line end. A closed line's two ends are two visits, so its closing point
    counts 1 + 1 = 2 like any other through-point."""
    line = verts["line"].to_numpy()
    new = np.r_[True, (line[1:] != line[:-1]) | (cluster[1:] != cluster[:-1])]
    visit = np.cumsum(new) - 1
    grouped = pd.DataFrame({"visit": visit, "line": line, "cluster": cluster,
                            "m": verts["m"].to_numpy(),
                            "first": verts["first"].to_numpy(),
                            "last": verts["last"].to_numpy()})
    out = grouped.groupby("visit").agg(line=("line", "first"), cluster=("cluster", "first"),
                                       m=("m", "mean"), has_first=("first", "any"),
                                       has_last=("last", "any"))
    out["arms"] = (~out["has_first"]).astype(int) + (~out["has_last"]).astype(int)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. ADB-with-ADB junctions and end topology
# --------------------------------------------------------------------------- #

def adb_topology(geoms_m, rules: dict):
    """Junction cut positions from the ADB network itself, and what happens at
    every line end.

    Returns (cuts, ends):
      cuts : DataFrame(line, m) -- junction positions, line ends included
      ends : DataFrame indexed by (line, end) with `node` (an id shared by
             line ends at the same point), `node_arms`, `junction` (bool).
    """
    verts = _vertex_table(geoms_m)
    xy = verts[["x", "y"]].to_numpy()
    cluster = _cluster(xy, rules["adb_vertex_snap_m"])
    visits = _visits(verts, cluster)
    arms = np.bincount(visits["cluster"], weights=visits["arms"],
                       minlength=int(cluster.max()) + 1 if len(cluster) else 0)
    junction_cluster = arms >= 3

    # T junctions: a dangling line end within tolerance of another line's
    # interior. Its partner gets a cut there; the end becomes a junction.
    ends_v = visits[visits["has_first"] | visits["has_last"]].copy()
    dangling = ends_v[arms[ends_v["cluster"].to_numpy()] == 1]
    t_cuts, t_clusters = [], set()
    if len(dangling):
        first_xy = verts.groupby("line")[["x", "y"]].first()
        last_xy = verts.groupby("line")[["x", "y"]].last()
        pts = []
        for _, v in dangling.iterrows():
            src = first_xy if v["has_first"] else last_xy
            pts.append(shapely.Point(*src.loc[int(v["line"])].to_numpy()))
        pts = np.asarray(pts, dtype=object)
        tol = rules["t_junction_tolerance_m"]
        same = rules["same_point_tolerance_m"]
        qi, li = shapely.STRtree(geoms_m).query(pts, predicate="dwithin", distance=tol)
        for q, other in zip(qi, li):
            v = dangling.iloc[q]
            m = float(shapely.line_locate_point(geoms_m[other], pts[q]))
            length = float(shapely.length(geoms_m[other]))
            if other == v["line"] and abs(m - v["m"]) <= max(tol, same) * 2:
                continue  # the end itself
            if m <= same or m >= length - same:
                continue  # end-to-end gap, handled by the snap, not a T
            t_cuts.append((int(other), m))
            t_clusters.add(int(v["cluster"]))

    junction_cluster = junction_cluster.copy()
    for c in t_clusters:
        junction_cluster[c] = True

    jv = visits[junction_cluster[visits["cluster"].to_numpy()]]
    cuts = pd.concat([jv[["line", "m"]],
                      pd.DataFrame(t_cuts, columns=["line", "m"])], ignore_index=True)

    rows = []
    for _, v in ends_v.iterrows():
        c = int(v["cluster"])
        for end, flag in ((0, v["has_first"]), (1, v["has_last"])):
            if flag:
                rows.append((int(v["line"]), end, c, float(arms[c]), bool(junction_cluster[c])))
    ends = pd.DataFrame(rows, columns=["line", "end", "node", "node_arms", "junction"])
    ends = ends.drop_duplicates(["line", "end"]).set_index(["line", "end"]).sort_index()
    return cuts, ends


# --------------------------------------------------------------------------- #
# 2. ADB-with-OSM junctions
# --------------------------------------------------------------------------- #

def car_ways(ways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Public roads a motor vehicle may use."""
    keep = ways["highway"].isin(CAR_HIGHWAYS)
    if "access" in ways.columns:
        keep &= ~ways["access"].isin(PRIVATE_ACCESS)
    if "tags" in ways.columns:
        def motor_blocked(tags) -> bool:
            if tags is None or (isinstance(tags, float) and np.isnan(tags)):
                return False
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except ValueError:
                    return False
            if not isinstance(tags, dict):
                return False
            return any(tags.get(k) in PRIVATE_MOTOR for k in ("motor_vehicle", "motorcar"))
        keep &= ~ways["tags"].map(motor_blocked)
    return ways.loc[keep].reset_index(drop=True)


def osm_junction_vertices(ways: gpd.GeoDataFrame) -> pd.DataFrame:
    """(way row, lon, lat) for every vertex where 3+ arms of `ways` meet.

    OSM ways that share a node carry bit-identical coordinates, so identity is
    exact here -- no tolerance is needed or wanted.
    """
    verts = _vertex_table(ways.geometry.values)
    key = pd.MultiIndex.from_arrays([verts["x"].to_numpy(), verts["y"].to_numpy()])
    codes, _ = pd.factorize(key)
    visits = _visits(verts, codes)
    arms = np.bincount(visits["cluster"], weights=visits["arms"])
    hit = arms[codes] >= 3
    return pd.DataFrame({"way_row": verts["line"].to_numpy()[hit],
                         "lon": verts["x"].to_numpy()[hit],
                         "lat": verts["y"].to_numpy()[hit]}).drop_duplicates()


def osm_junction_cuts(network: gpd.GeoDataFrame, geoms_m, ways: gpd.GeoDataFrame,
                      country: str) -> pd.DataFrame:
    """Junction positions on each ADB line from the OSM ways it is built from."""
    from segment_way_match import load_rules as load_way_rules
    from segment_way_match import match_parents

    ways = car_ways(ways)
    if not len(ways):
        return pd.DataFrame(columns=["line", "m"])
    way_rules = load_way_rules()
    tol = way_rules["tolerance_distance_m"]
    edge_matches, _ = match_parents(network.reset_index(drop=True), ways, country, way_rules)
    if not len(edge_matches):
        return pd.DataFrame(columns=["line", "m"])

    # Which arc range of each ADB line each way explains, merged per (line, way).
    em = edge_matches.sort_values(["seg_index", "osm_way_id", "start_m"])
    spans = (em.groupby(["seg_index", "osm_way_id"])
               .agg(s=("start_m", "min"), e=("end_m", "max")).reset_index())

    jv = osm_junction_vertices(ways)
    jv["osm_way_id"] = ways["osm_way_id"].to_numpy()[jv["way_row"].to_numpy()]
    rows = spans.merge(jv, on="osm_way_id", how="inner")
    if not len(rows):
        return pd.DataFrame(columns=["line", "m"])

    crs = BUFFER_CRS[country]
    pts = gpd.GeoSeries(gpd.points_from_xy(rows["lon"], rows["lat"]), crs="EPSG:4326").to_crs(crs).values
    lines = np.asarray(geoms_m, dtype=object)[rows["seg_index"].to_numpy()]
    m = shapely.line_locate_point(lines, pts)
    d = shapely.distance(lines, pts)
    s, e = rows["s"].to_numpy(), rows["e"].to_numpy()
    inside = (m >= s - tol) & (m <= e + tol) & (d <= tol)

    # A line that passes the same place twice: the global projection chose the
    # other pass. Project within the way's own range instead.
    redo = np.flatnonzero(~inside & (d <= tol))
    for i in redo:
        part = substring(lines[i], max(s[i] - tol, 0.0), e[i] + tol)
        if shapely.distance(part, pts[i]) <= tol:
            m[i] = max(s[i] - tol, 0.0) + float(shapely.line_locate_point(part, pts[i]))
            inside[i] = True
    out = pd.DataFrame({"line": rows["seg_index"].to_numpy()[inside], "m": m[inside]})
    return out.drop_duplicates()


# --------------------------------------------------------------------------- #
# 3. Pieces and uninterrupted segments
# --------------------------------------------------------------------------- #

def build_pieces(geoms_m, classes: np.ndarray, adb_cuts: pd.DataFrame,
                 osm_cuts: pd.DataFrame, ends: pd.DataFrame, rules: dict):
    """Cut every line at its junctions and chain the pieces into
    uninterrupted segments.

    Returns a DataFrame with one row per piece: line, lo, hi, group, forward
    (drawn in chain direction), offset (chain measure of its chain-start end),
    plus a dict of group -> (length, is_cycle).
    """
    same = rules["same_point_tolerance_m"]
    lengths = shapely.length(geoms_m)
    n = len(lengths)
    cuts = pd.concat([adb_cuts.assign(src="adb"), osm_cuts.assign(src="osm")],
                     ignore_index=True)

    # A cut within tolerance of a line end marks that end as a junction.
    end_junction = np.zeros((n, 2), dtype=bool)
    if len(ends):
        idx = ends.index.to_frame(index=False)
        end_junction[idx["line"].to_numpy(), idx["end"].to_numpy()] = ends["junction"].to_numpy()
    interior: dict[int, list[float]] = {}
    for line, m in zip(cuts["line"].to_numpy(dtype=int), cuts["m"].to_numpy(dtype=float)):
        if m <= same:
            end_junction[line, 0] = True
        elif m >= lengths[line] - same:
            end_junction[line, 1] = True
        else:
            interior.setdefault(line, []).append(m)

    p_line, p_lo, p_hi = [], [], []
    first_piece = np.zeros(n, dtype=np.int64)
    last_piece = np.zeros(n, dtype=np.int64)
    for line in range(n):
        ms = np.sort(np.asarray(interior.get(line, []), dtype=float))
        # Two sources seeing one junction a few decimetres apart are one cut.
        kept = []
        for m in ms:
            if not kept or m - kept[-1] > same:
                kept.append(m)
        bounds = [0.0, *kept, float(lengths[line])]
        first_piece[line] = len(p_line)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            p_line.append(line)
            p_lo.append(lo)
            p_hi.append(hi)
        last_piece[line] = len(p_line) - 1
    pieces = pd.DataFrame({"line": p_line, "lo": p_lo, "hi": p_hi})
    n_pieces = len(pieces)

    # Links between piece ends: only at a line end that is a plain
    # continuation -- exactly two ADB arms, no OSM junction, same class.
    link: dict[tuple[int, int], tuple[int, int]] = {}
    dsu = _DSU(n_pieces)
    if len(ends):
        e = ends.reset_index()
        e = e[(e["node_arms"] == 2)]
        for node, grp in e.groupby("node"):
            if len(grp) != 2:
                continue
            (l1, e1), (l2, e2) = grp[["line", "end"]].to_numpy()
            if end_junction[l1, e1] or end_junction[l2, e2]:
                continue
            c1, c2 = classes[l1], classes[l2]
            if not (c1 == c2 or (pd.isna(c1) and pd.isna(c2))):
                continue
            p1 = first_piece[l1] if e1 == 0 else last_piece[l1]
            p2 = first_piece[l2] if e2 == 0 else last_piece[l2]
            # A piece end is (piece, 0=start, 1=end).
            pe1 = (int(p1), 0 if e1 == 0 else 1)
            pe2 = (int(p2), 0 if e2 == 0 else 1)
            if pe1 == pe2:
                continue
            link[pe1] = pe2
            link[pe2] = pe1
            dsu.union(pe1[0], pe2[0])

    roots = np.array([dsu.find(i) for i in range(n_pieces)])
    # Walk every group from a free end (or anywhere, for a cycle).
    group = np.full(n_pieces, -1, dtype=np.int64)
    forward = np.ones(n_pieces, dtype=bool)
    offset = np.zeros(n_pieces)
    plen = (pieces["hi"] - pieces["lo"]).to_numpy()
    members: dict[int, list[int]] = {}
    for i, r in enumerate(roots):
        members.setdefault(int(r), []).append(i)
    groups_info = {}
    gid = 0
    for r in sorted(members):
        ids = members[r]
        start = None
        for p in ids:
            for end in (0, 1):
                if (p, end) not in link:
                    start = (p, end)
                    break
            if start is not None:
                break
        is_cycle = start is None
        if is_cycle:
            start = (min(ids), 0)
        p, enter = start
        pos = 0.0
        seen = set()
        while p not in seen:
            seen.add(p)
            group[p] = gid
            forward[p] = enter == 0
            offset[p] = pos
            pos += plen[p]
            nxt = link.get((p, 1 - enter))
            if nxt is None:
                break
            p, enter = nxt
        groups_info[gid] = (pos, is_cycle)
        gid += 1
    pieces["group"] = group
    pieces["forward"] = forward
    pieces["offset"] = offset
    return pieces, groups_info


def _piece_geoms(geoms_m, pieces: pd.DataFrame) -> np.ndarray:
    out = []
    for line, lo, hi in pieces[["line", "lo", "hi"]].itertuples(index=False):
        g = geoms_m[int(line)]
        if lo <= 0.0 and hi >= g.length:
            out.append(g)
        else:
            out.append(substring(g, lo, hi))
    return np.asarray(out, dtype=object)


def _chain_geometry(pgeoms, pieces: pd.DataFrame, gid: int, doubled: bool):
    """The uninterrupted segment as one line, in chain order."""
    members = pieces[pieces["group"] == gid].sort_values("offset")
    coords = []
    for idx, fwd in zip(members.index, members["forward"]):
        c = shapely.get_coordinates(pgeoms[idx])
        if not fwd:
            c = c[::-1]
        if coords and len(c) and np.allclose(coords[-1][-1], c[0]):
            c = c[1:]
        if len(c):
            coords.append(c)
    xy = np.vstack(coords) if coords else np.empty((0, 2))
    if doubled and len(xy):
        xy = np.vstack([xy, xy[1:]])
    if len(xy) < 2:
        return shapely.Point(*xy[0]) if len(xy) else None
    return shapely.LineString(xy)


# --------------------------------------------------------------------------- #
# 4. Matching TomTom to uninterrupted segments
# --------------------------------------------------------------------------- #

def _tomtom_samples(tt_geoms, rules: dict):
    """(tt index, position, weight) for every sample of every TomTom line.

    Weight is the length each sample stands for (half the gap to each
    neighbour), so shares are shares of LENGTH whatever the vertex spacing.
    """
    step, n_min = rules["sample_step_m"], rules["min_samples_per_tomtom"]
    verts = _vertex_table(tt_geoms)
    lengths = shapely.length(tt_geoms)
    tt, pos, w = [], [], []
    vm = verts.groupby("line")["m"].apply(np.asarray)
    for i, length in enumerate(lengths):
        spacing = min(step, length / n_min) if length > 0 else 1.0
        grid = np.arange(0.0, length, spacing) if length > 0 else np.array([0.0])
        p = np.unique(np.r_[grid, vm.get(i, np.array([])), length])
        gaps = np.diff(p)
        weight = np.r_[gaps, 0.0] / 2 + np.r_[0.0, gaps] / 2
        tt.append(np.full(len(p), i))
        pos.append(p)
        w.append(weight)
    return np.concatenate(tt), np.concatenate(pos), np.concatenate(w)


def _unwrap_cycle(cm: np.ndarray, length: float) -> np.ndarray:
    """Chain measures on a ring, shifted so they read as one contiguous run."""
    if len(cm) < 2 or length <= 0:
        return cm
    s = np.sort(cm % length)
    gaps = np.diff(np.r_[s, s[0] + length])
    cut = s[(int(np.argmax(gaps)) + 1) % len(s)]
    return np.where(cm % length < cut, cm % length + length, cm % length)


def match_tomtom(tomtom_m: gpd.GeoDataFrame, pieces: pd.DataFrame, pgeoms,
                 groups_info: dict, rules: dict, batch: int = 2000):
    """One row per matched (TomTom segment, uninterrupted segment) part.

    Columns: tt, group, lo, hi (chain measure), along_chain (the TomTom
    geometry is drawn in increasing chain measure), mean_d, frechet, part_frac.
    Also returns per-TomTom diagnostics for calibration.
    """
    d_max = rules["max_distance_m"]
    max_bearing = rules["bearing_tolerance_deg"]
    window = rules["bearing_window_m"]
    seg_len = rules["frechet_segmentize_m"]
    tt_geoms = np.asarray(tomtom_m.geometry.values, dtype=object)
    tt_len = shapely.length(tt_geoms)
    tree = shapely.STRtree(pgeoms)
    plen = (pieces["hi"] - pieces["lo"]).to_numpy()
    p_group = pieces["group"].to_numpy()
    p_fwd = pieces["forward"].to_numpy()
    p_off = pieces["offset"].to_numpy()

    chain_cache: dict[tuple[int, bool], object] = {}
    matches, diag = [], []

    for start in range(0, len(tt_geoms), batch):
        sub = np.arange(start, min(start + batch, len(tt_geoms)))
        s_tt, s_pos, s_w = _tomtom_samples(tt_geoms[sub], rules)
        s_tt = sub[s_tt]
        lines = tt_geoms[s_tt]
        pts = shapely.line_interpolate_point(lines, s_pos)
        tt_bearing = _bearings(lines, s_pos, window)

        qi, pi = tree.query(pts, predicate="dwithin", distance=d_max)
        best_piece = np.full(len(pts), -1, dtype=np.int64)
        best_loc = np.zeros(len(pts))
        best_d = np.full(len(pts), np.inf)
        if len(qi):
            cand = pgeoms[pi]
            dist = shapely.distance(pts[qi], cand)
            loc = shapely.line_locate_point(cand, pts[qi])
            ok = _bearing_diff(_bearings(cand, loc, window), tt_bearing[qi]) <= max_bearing
            qi, pi, dist, loc = qi[ok], pi[ok], dist[ok], loc[ok]
            order = np.lexsort((dist, qi))
            qi, pi, dist, loc = qi[order], pi[order], dist[order], loc[order]
            first = np.r_[True, qi[1:] != qi[:-1]] if len(qi) else np.zeros(0, bool)
            best_piece[qi[first]] = pi[first]
            best_loc[qi[first]] = loc[first]
            best_d[qi[first]] = dist[first]

        assigned = best_piece >= 0
        g = np.where(assigned, p_group[np.clip(best_piece, 0, None)], -1)
        cm = np.where(p_fwd[np.clip(best_piece, 0, None)], best_loc,
                      plen[np.clip(best_piece, 0, None)] - best_loc)
        cm = cm + p_off[np.clip(best_piece, 0, None)]

        frame = pd.DataFrame({"tt": s_tt, "pos": s_pos, "w": s_w, "g": g,
                              "cm": cm, "d": best_d})
        for t, samples in frame.groupby("tt", sort=True):
            length = float(tt_len[t])
            total_w = samples["w"].sum()
            got = samples[samples["g"] >= 0]
            frac = float(got["w"].sum() / total_w) if total_w > 0 else float(len(got) > 0)
            rec = {"tt": int(t), "length_m": length, "attributed_frac": frac,
                   "n_parts": 0, "status": "matched"}
            if frac < rules["min_tomtom_frac"]:
                rec["status"] = "too_little_beside_adb"
                diag.append(rec)
                continue
            n_ok = n_small = n_shape = 0
            worst_fd = 0.0
            for gid, part in got.groupby("g"):
                part_frac = float(part["w"].sum() / total_w) if total_w > 0 else 1.0
                # An overhang past a junction is small in BOTH senses: a small
                # share of the TomTom segment, and no longer than the matching
                # tolerance that let it reach across. A long stretch that is a
                # small share of a very long TomTom segment is a real part.
                if (part_frac < rules["min_part_frac"]
                        and part["w"].sum() <= rules["max_distance_m"]):
                    n_small += 1
                    continue
                t_lo, t_hi = float(part["pos"].min()), float(part["pos"].max())
                if t_hi - t_lo < 1e-6 and length > 1e-6:
                    continue
                glen, is_cycle = groups_info[int(gid)]
                c = part["cm"].to_numpy()
                if is_cycle:
                    c = _unwrap_cycle(c, glen)
                c_lo, c_hi = float(c.min()), float(c.max())
                key = (int(gid), bool(is_cycle))
                if key not in chain_cache:
                    chain_cache[key] = _chain_geometry(pgeoms, pieces, int(gid), is_cycle)
                chain = chain_cache[key]
                p = part["pos"].to_numpy()
                along = float(np.sum((p - p.mean()) * (c - c.mean()))) >= 0
                tt_part = substring(tt_geoms[t], t_lo, t_hi)
                adb_part = substring(chain, c_lo, c_hi)
                # Frechet walks both lines from their starts, so they must be
                # compared in the same direction of travel.
                if not along:
                    adb_part = shapely.reverse(adb_part)
                fd = float(shapely.frechet_distance(
                    shapely.segmentize(tt_part, seg_len), shapely.segmentize(adb_part, seg_len)))
                if fd > d_max:
                    n_shape += 1
                    worst_fd = max(worst_fd, fd)
                    continue
                matches.append({"tt": int(t), "group": int(gid), "lo": c_lo, "hi": c_hi,
                                "along_chain": along, "mean_d": float(part["d"].mean()),
                                "frechet": fd, "part_frac": part_frac})
                n_ok += 1
            rec.update(n_parts=n_ok, n_parts_too_small=n_small,
                       n_parts_shape_mismatch=n_shape, worst_rejected_frechet=worst_fd)
            if n_ok == 0:
                rec["status"] = "shape_mismatch" if n_shape else "only_small_parts"
            diag.append(rec)

    cols = ["tt", "group", "lo", "hi", "along_chain", "mean_d", "frechet", "part_frac"]
    return pd.DataFrame(matches, columns=cols), pd.DataFrame(diag)


# --------------------------------------------------------------------------- #
# 5. Partition each uninterrupted segment among its TomTom segments
# --------------------------------------------------------------------------- #

def partition(matches: pd.DataFrame, has_direction: pd.DataFrame, groups_info: dict,
              rules: dict):
    """For every (group, chain travel direction): the TomTom segment serving
    each stretch of the chain.

    has_direction: DataFrame indexed by tt with boolean columns along/against
    -- whether the TomTom segment carries data for that direction of travel.

    Returns DataFrame(group, dir (+1/-1), start, end, tt, tt_dir, cov_lo,
    cov_hi) and the number of TomTom ranges dropped in heavy overlaps.
    """
    rows, dropped = [], 0
    if not len(matches):
        return pd.DataFrame(columns=["group", "dir", "start", "end", "tt", "tt_dir",
                                     "cov_lo", "cov_hi"]), 0
    ivals = []
    for m in matches.itertuples(index=False):
        for direction in (+1, -1):
            # Travel in +chain is travel along the TomTom geometry when the
            # geometry itself runs in +chain.
            tt_dir = "along" if (m.along_chain == (direction == +1)) else "against"
            if not has_direction.at[m.tt, tt_dir]:
                continue
            glen, is_cycle = groups_info[m.group]
            spans = [(m.lo, m.hi)]
            if is_cycle and m.hi > glen:
                spans = [(m.lo, glen), (0.0, m.hi - glen)] if m.lo < glen else [(m.lo - glen, m.hi - glen)]
            for lo, hi in spans:
                ivals.append((m.group, direction, lo, hi, m.tt, tt_dir, m.mean_d))
    iv = pd.DataFrame(ivals, columns=["group", "dir", "lo", "hi", "tt", "tt_dir", "mean_d"])
    thr = rules["heavy_overlap_frac"]
    for (gid, direction), grp in iv.groupby(["group", "dir"], sort=True):
        glen = groups_info[gid][0]
        # Merge the spans of one TomTom segment (a ring can split one in two).
        grp = (grp.groupby(["tt", "tt_dir"], as_index=False)
                  .agg(lo=("lo", "min"), hi=("hi", "max"), mean_d=("mean_d", "mean"))
                  .sort_values(["lo", "hi"]).reset_index(drop=True))
        kept = []
        for r in grp.itertuples(index=False):
            if kept:
                prev = kept[-1]
                overlap = min(prev.hi, r.hi) - max(prev.lo, r.lo)
                shorter = max(min(prev.hi - prev.lo, r.hi - r.lo), 1e-9)
                if overlap >= thr * shorter:
                    dropped += 1
                    if r.mean_d < prev.mean_d:
                        kept[-1] = r
                    continue
            kept.append(r)
        bounds = [0.0]
        for a, b in zip(kept[:-1], kept[1:]):
            bounds.append(max(bounds[-1], min((a.hi + b.lo) / 2, glen)))
        bounds.append(glen)
        for k, r in enumerate(kept):
            rows.append((gid, direction, bounds[k], bounds[k + 1], r.tt, r.tt_dir, r.lo, r.hi))
    out = pd.DataFrame(rows, columns=["group", "dir", "start", "end", "tt", "tt_dir",
                                      "cov_lo", "cov_hi"])
    return out, dropped


# --------------------------------------------------------------------------- #
# 6. Back onto the ADB lines
# --------------------------------------------------------------------------- #

def cut_table(pieces: pd.DataFrame, parts: pd.DataFrame, n_lines: int) -> pd.DataFrame:
    """One row per output piece of each ADB line.

    Columns: line, j, lo, hi, n (pieces of that line), group, and for each
    side in (fwd, bwd): tt_{side}, tt_dir_{side}, covered_{side}.
    """
    by_gd: dict[tuple[int, int], pd.DataFrame] = {
        k: v.sort_values("start") for k, v in parts.groupby(["group", "dir"])
    } if len(parts) else {}
    records = []
    for p in pieces.itertuples():
        a, b = p.offset, p.offset + (p.hi - p.lo)
        breaks = {a, b}
        per_dir = {}
        for direction in (+1, -1):
            d = by_gd.get((p.group, direction))
            if d is not None:
                d = d[(d["end"] > a) & (d["start"] < b)]
                breaks.update(v for v in d["start"] if a < v < b)
                per_dir[direction] = d
        cuts = sorted(breaks)
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            if c1 - c0 < 1e-6:
                continue
            mid = (c0 + c1) / 2
            rec = {"line": p.line, "group": p.group}
            if p.forward:
                rec["lo"], rec["hi"] = p.lo + (c0 - a), p.lo + (c1 - a)
            else:
                rec["lo"], rec["hi"] = p.hi - (c1 - a), p.hi - (c0 - a)
            for side in SIDES:
                # Travelling in the drawn direction of the ADB line is +chain
                # on a forward piece and -chain on a reversed one.
                direction = (+1 if p.forward else -1) * (+1 if side == "fwd" else -1)
                d = per_dir.get(direction)
                hit = d[(d["start"] <= mid) & (d["end"] >= mid)] if d is not None else None
                if hit is not None and len(hit):
                    h = hit.iloc[0]
                    rec[f"tt_{side}"] = int(h["tt"])
                    rec[f"tt_dir_{side}"] = h["tt_dir"]
                    rec[f"covered_{side}"] = max(0.0, min(c1, h["cov_hi"]) - max(c0, h["cov_lo"]))
                else:
                    rec[f"tt_{side}"] = -1
                    rec[f"tt_dir_{side}"] = None
                    rec[f"covered_{side}"] = 0.0
            records.append(rec)
    sub = pd.DataFrame(records).sort_values(["line", "lo"]).reset_index(drop=True)

    # Join runs of unmatched sub-pieces back together within each line.
    unmatched = (sub["tt_fwd"] < 0) & (sub["tt_bwd"] < 0)
    line = sub["line"].to_numpy()
    start_run = np.r_[True, (line[1:] != line[:-1]) | ~unmatched.to_numpy()[1:]
                      | ~unmatched.to_numpy()[:-1]]
    run = np.cumsum(start_run) - 1
    agg = {c: "first" for c in sub.columns if c not in ("lo", "hi", "covered_fwd", "covered_bwd")}
    agg.update(lo="min", hi="max", covered_fwd="sum", covered_bwd="sum")
    merged = sub.groupby(run, sort=True).agg(agg)
    # A joined run spans several uninterrupted segments; it belongs to none.
    multi = sub.groupby(run)["group"].nunique() > 1
    merged.loc[multi.to_numpy(), "group"] = -1
    merged = merged.sort_values(["line", "lo"]).reset_index(drop=True)
    merged["j"] = merged.groupby("line").cumcount()
    merged["n"] = merged.groupby("line")["line"].transform("size")
    missing = set(range(n_lines)) - set(merged["line"])
    assert not missing, f"lines lost in cut_table: {sorted(missing)[:5]}"
    return merged


# --------------------------------------------------------------------------- #
# The whole run
# --------------------------------------------------------------------------- #

def integrate(network: gpd.GeoDataFrame, tomtom: gpd.GeoDataFrame | None,
              ways: gpd.GeoDataFrame | None, country: str, *, class_col: str,
              rules: dict | None = None):
    """Cut table for every row of `network` (positional), plus a meta dict.

    `network` is the whole ADB layer of one country in EPSG:4326: junctions
    and matches are decided against all of it even when only part of it is
    later kept, so a TomTom line beside a road outside the analysed subset
    is not handed to the nearest analysed one.
    """
    rules = rules or load_rules()
    crs = BUFFER_CRS[country]
    network = network.reset_index(drop=True)
    geoms_m = np.asarray(network.to_crs(crs).geometry.values, dtype=object)
    classes = network[class_col].to_numpy(dtype=object)

    adb_cuts, ends = adb_topology(geoms_m, rules)
    if ways is not None and len(ways):
        osm_cuts = osm_junction_cuts(network, geoms_m, ways, country)
    else:
        osm_cuts = pd.DataFrame(columns=["line", "m"])
    pieces, groups_info = build_pieces(geoms_m, classes, adb_cuts, osm_cuts, ends, rules)
    pgeoms = _piece_geoms(geoms_m, pieces)

    meta = {
        "country": country,
        "n_adb_lines": int(len(network)),
        "n_adb_junction_cuts": int(len(adb_cuts)),
        "n_osm_junction_cuts": int(len(osm_cuts)),
        "n_pieces_before_matching": int(len(pieces)),
        "n_uninterrupted_segments": int(len(groups_info)),
        "n_uninterrupted_cycles": int(sum(1 for v in groups_info.values() if v[1])),
    }

    if tomtom is None or not len(tomtom):
        matches = pd.DataFrame(columns=["tt", "group", "lo", "hi", "along_chain",
                                        "mean_d", "frechet", "part_frac"])
        diag = pd.DataFrame(columns=["tt", "length_m", "attributed_frac", "n_parts", "status"])
        parts, dropped = partition(matches, pd.DataFrame(), groups_info, rules)
    else:
        tomtom = tomtom.reset_index(drop=True)
        tomtom_m = tomtom.to_crs(crs)
        matches, diag = match_tomtom(tomtom_m, pieces, pgeoms, groups_info, rules)
        has_direction = pd.DataFrame({
            d: tomtom[f"tomtom_signed_segment_id_{d}"].notna().to_numpy() for d in DIRECTIONS
        })
        parts, dropped = partition(matches, has_direction, groups_info, rules)

    table = cut_table(pieces, parts, len(network))
    table["uninterrupted_segment_id"] = np.where(
        table["group"] >= 0, [f"{country}_u{g}" for g in table["group"]], None)

    meta.update({
        "n_tomtom_segments": int(0 if tomtom is None else len(tomtom)),
        "n_tomtom_matched": int(diag["status"].eq("matched").sum()) if len(diag) else 0,
        "tomtom_status": diag["status"].value_counts().to_dict() if len(diag) else {},
        "tomtom_km_matched": round(float(diag.loc[diag["status"] == "matched", "length_m"].sum()) / 1000, 1)
        if len(diag) else 0.0,
        "tomtom_km_unmatched": round(float(diag.loc[diag["status"] != "matched", "length_m"].sum()) / 1000, 1)
        if len(diag) else 0.0,
        "n_match_parts": int(len(matches)),
        "n_heavy_overlaps_dropped": int(dropped),
        "n_lines_split": int((table.groupby("line")["n"].first() > 1).sum()),
        "n_output_pieces": int(len(table)),
        "n_pieces_with_tomtom": int(((table["tt_fwd"] >= 0) | (table["tt_bwd"] >= 0)).sum()),
    })
    if len(diag):
        meta["calibration"] = {
            "attributed_frac_quantiles": _quantiles(diag["attributed_frac"]),
            "frechet_m_quantiles": _quantiles(matches["frechet"]) if len(matches) else {},
            "mean_distance_m_quantiles": _quantiles(matches["mean_d"]) if len(matches) else {},
            "part_frac_quantiles": _quantiles(matches["part_frac"]) if len(matches) else {},
        }
    return table, meta, diag


def _quantiles(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s):
        return {}
    return {str(q): round(float(s.quantile(q)), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}


# --------------------------------------------------------------------------- #
# Applying the cut table to a frame
# --------------------------------------------------------------------------- #

def _cut_line(geom_4326, geom_m, lo: float, hi: float):
    """The [lo, hi] metre range of a line, keeping its original vertices
    bit-for-bit and interpolating only the two new end points."""
    c4 = shapely.get_coordinates(geom_4326)
    cm = shapely.get_coordinates(geom_m)
    step = np.r_[0.0, np.hypot(*np.diff(cm, axis=0).T)]
    meas = np.cumsum(step)

    def at(m):
        i = int(np.clip(np.searchsorted(meas, m, side="right") - 1, 0, len(meas) - 2))
        span = meas[i + 1] - meas[i]
        t = 0.0 if span <= 0 else (m - meas[i]) / span
        return c4[i] + t * (c4[i + 1] - c4[i])

    inner = c4[(meas > lo) & (meas < hi)]
    xy = np.vstack([at(lo), inner, at(hi)])
    return shapely.LineString(xy)


def side_columns(tomtom: gpd.GeoDataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """The TomTom values for each output piece, by side (fwd / bwd)."""
    out = {}
    for side in SIDES:
        tt = table[f"tt_{side}"].to_numpy()
        tdir = table[f"tt_dir_{side}"].to_numpy()
        has = tt >= 0
        safe = np.clip(tt, 0, None)
        out[f"tomtom_segment_id_{side}"] = pd.array(
            np.where(has, tomtom["tomtom_segment_id"].to_numpy()[safe], 0), dtype="Int64")
        out[f"tomtom_segment_id_{side}"][~has] = pd.NA
        for c in SIDE_VALUE_COLUMNS:
            values = []
            for i, (h, t, d) in enumerate(zip(has, safe, tdir)):
                values.append(tomtom[f"{c}_{d}"].iat[t] if h else None)
            s = pd.Series(values, dtype=object)
            if c in ("tomtom_signed_segment_id", "tomtom_sample_size"):
                s = pd.array(pd.to_numeric(s, errors="coerce"), dtype="Int64")
            elif c == "tomtom_speed_limit":
                s = pd.array(pd.to_numeric(s, errors="coerce"), dtype="Int16")
            elif c != "tomtom_speed_percentiles":
                s = pd.to_numeric(s, errors="coerce").astype(float)
            out[f"{c}_{side}"] = s
        out[f"tomtom_street_name_{side}"] = pd.array(
            [tomtom["tomtom_street_name"].iat[t] if h else pd.NA for h, t in zip(has, safe)],
            dtype="string")
        out[f"tomtom_covered_len_m_{side}"] = table[f"covered_{side}"].to_numpy(dtype=float)
    return pd.DataFrame(out, index=table.index)


def apply_cuts(frame: gpd.GeoDataFrame, table: pd.DataFrame, tomtom, *, id_col: str,
               country: str, frame_line: np.ndarray, length_cols: dict[str, float] | None = None,
               extensive_cols: tuple[str, ...] = ()) -> gpd.GeoDataFrame:
    """Rows of `frame` cut per `table`.

    frame_line   : for each row of `frame`, its positional index in the
                   network the table was built on.
    length_cols  : column -> unit scale; rescaled by the piece's share of its
                   parent's length.
    extensive_cols : counts apportioned by length share.
    An uncut row keeps its geometry and id untouched.
    """
    crs = BUFFER_CRS[country]
    frame = frame.reset_index(drop=True)
    geoms_m = np.asarray(frame.to_crs(crs).geometry.values, dtype=object)
    by_line = {k: v for k, v in table.groupby("line")}
    values = side_columns(tomtom, table) if tomtom is not None and len(tomtom) else None

    records, geoms, idx = [], [], []
    for r, line in enumerate(frame_line):
        pieces = by_line[int(line)]
        parent = frame.iloc[r]
        total = float(geoms_m[r].length)
        for _, p in pieces.iterrows():
            row = parent.to_dict()
            if len(pieces) == 1:
                geom = parent.geometry
            else:
                geom = _cut_line(parent.geometry, geoms_m[r], p["lo"], p["hi"])
                row[id_col] = f"{parent[id_col]}#{int(p['j'])}"
                share = (p["hi"] - p["lo"]) / total if total > 0 else 0.0
                for col in (length_cols or {}):
                    if col in row and pd.notna(row[col]):
                        row[col] = row[col] * share
                for col in extensive_cols:
                    if col in row and pd.notna(row[col]):
                        row[col] = row[col] * share
            row["adb_parent_id"] = str(parent[id_col])
            row["uninterrupted_segment_id"] = p["uninterrupted_segment_id"]
            row.pop(frame.geometry.name, None)
            records.append(row)
            geoms.append(geom)
            idx.append(p.name)
    out = gpd.GeoDataFrame(records, geometry=geoms, crs=frame.crs)
    out[id_col] = out[id_col].astype(str)
    if values is not None:
        v = values.loc[idx].reset_index(drop=True)
        for c in v.columns:
            out[c] = v[c].to_numpy() if not hasattr(v[c], "array") else v[c].array
    assert out[id_col].is_unique, "output ids are not unique"
    return out


# --------------------------------------------------------------------------- #
# Stand-alone run: the raw ADB layer in, a parquet out
# --------------------------------------------------------------------------- #

def output_path(country: str) -> Path:
    return OUTPUT_DIR / f"adb_segments_tomtom_{country}.parquet"


def kepler_output_path(country: str) -> Path:
    return output_path(country).with_name(f"adb_segments_tomtom_{country}_kepler.parquet")


def run(country: str) -> gpd.GeoDataFrame:
    import time
    import warnings

    import osm_ways
    import tomtom_stats

    warnings.filterwarnings("ignore", category=UserWarning)
    t0 = time.time()
    network = gpd.read_file(RAW_ADB_PATHS[country])
    tomtom = tomtom_stats.load_tomtom(country)
    ways = osm_ways.load_ways(country)
    table, meta, diag = integrate(network, tomtom, ways, country,
                                  class_col=CLASS_COLUMN[country])
    out = apply_cuts(network, table, tomtom, id_col=ID_COLUMN, country=country,
                     frame_line=np.arange(len(network)),
                     length_cols={"Shape_Length": 1.0, "RoadLength": 1.0})
    out = out.rename(columns={ID_COLUMN: "segment_id"})
    out.insert(0, ID_COLUMN, network[ID_COLUMN].to_numpy()[
        out["adb_parent_id"].map({str(v): i for i, v in enumerate(network[ID_COLUMN])}).to_numpy()])
    out.to_parquet(output_path(country))
    write_kepler_parquet(out, kepler_output_path(country), drop_lists=True)

    meta.update({
        "input_adb": RAW_ADB_PATHS[country],
        "input_tomtom": tomtom_stats.RAW_PATHS.get(country),
        "class_column": CLASS_COLUMN[country],
        "rules": {k: v for k, v in load_rules().items() if not k.startswith("_")},
        "elapsed_s": round(time.time() - t0, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    meta_file = output_path(country).with_name(output_path(country).stem + "_meta.json")
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    diag.to_parquet(output_path(country).with_name(output_path(country).stem + "_tomtom_diag.parquet"))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--country", choices=[*RAW_ADB_PATHS, "both"], default="maharashtra")
    args = ap.parse_args()
    for c in ([*RAW_ADB_PATHS] if args.country == "both" else [args.country]):
        run(c)
