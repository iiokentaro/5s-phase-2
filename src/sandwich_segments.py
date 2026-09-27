"""Short "sandwich" segments: a short segment whose speed is higher than
everything around it takes its along-road neighbours' speed.

Rule (step 1, repeated until nothing changes). Segment Y is lowered when all of
these hold:
  - Y is not excluded (by default: not is_access_controlled);
  - Y's line length (along the line, not end to end) <= max_length_m;
  - each end of Y has an along-road neighbour (A at one end, B at the other):
    among the segments with an end within touch_tol_m of Y's end, the one whose
    direction turns least from Y's;
  - chi < speed(Y), where chi is the highest speed of every segment within
    touch_tol_m of either end of Y (along-road or not).
Y's new speed is max(speed(A), speed(B)) and Y is flagged is_sandwich. Its id
does not change. All rows are judged on the same speeds in each pass, so the
result does not depend on row order; the passes stop when one changes nothing.

Step 2 (once, after step 1 converges). A sandwich row is joined to its
along-road neighbours that have the same speed and the same parent id (they are
pieces of one segment). The joined row takes the non-sandwich member's
attributes, is_sandwich False, and its parent id as id when no other row still
carries that parent id.

Everything the rule reads is a parameter, so it can run on any speed column.
"""

from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import linemerge, unary_union

sys.path.insert(0, "src")
from exposure_signals import BUFFER_CRS

# How far along the line (m) the direction at an end is measured, capped at
# half the line so a short segment still has one.
_DIRECTION_SAMPLE_M = 10.0


# --------------------------------------------------------------------------- #
# geometry: ends, directions and who touches whom
# --------------------------------------------------------------------------- #

def _end_points_and_directions(lines: np.ndarray):
    """Per line: its two end points (start, end) and, at each end, the unit
    vector pointing from the end into the line."""
    length = shapely.length(lines)
    d = np.minimum(_DIRECTION_SAMPLE_M, length / 2)
    start = shapely.line_interpolate_point(lines, 0.0)
    end = shapely.line_interpolate_point(lines, length)
    start_in = shapely.line_interpolate_point(lines, d)
    end_in = shapely.line_interpolate_point(lines, length - d)
    ends = np.stack([start, end], axis=1)

    def unit(a, b):
        v = shapely.get_coordinates(b) - shapely.get_coordinates(a)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return np.divide(v, n, out=np.zeros_like(v), where=n > 0)

    inward = np.stack([unit(start, start_in), unit(end, end_in)], axis=1)
    return ends, inward, length


def build_endpoint_adjacency(lines: np.ndarray, touch_tol_m: float):
    """For each line i and end e (0 = start, 1 = end):
      touching[i][e]  every other line within touch_tol_m of that end;
      continuing[i][e] (j, f, turn) for every other line j whose end f is
                       within touch_tol_m of it, with the turn angle (radians,
                       0 = straight on) from line i into line j.
    Returns (touching, continuing, length)."""
    ends, inward, length = _end_points_and_directions(lines)
    n = len(lines)
    flat_ends = ends.reshape(-1)  # index = 2 * line + end

    line_tree = shapely.STRtree(lines)
    q_end, q_line = line_tree.query(flat_ends, predicate="dwithin", distance=touch_tol_m)
    touching = [[set(), set()] for _ in range(n)]
    for qe, j in zip(q_end, q_line):
        i, e = divmod(int(qe), 2)
        if j != i:
            touching[i][e].add(int(j))

    end_tree = shapely.STRtree(flat_ends)
    q_end, q_other = end_tree.query(flat_ends, predicate="dwithin", distance=touch_tol_m)
    continuing = [[[], []] for _ in range(n)]
    flat_inward = inward.reshape(-1, 2)
    for qe, qo in zip(q_end, q_other):
        i, e = divmod(int(qe), 2)
        j, f = divmod(int(qo), 2)
        if j == i:
            continue
        # Travelling along i out of its end e is -inward[i, e]; carrying on
        # into j from its end f is inward[j, f]. Straight on = angle 0.
        cos = float(np.clip(np.dot(-flat_inward[qe], flat_inward[qo]), -1.0, 1.0))
        continuing[i][e].append((j, f, float(np.arccos(cos))))
    return touching, continuing, length


def pick_along_road_neighbor(candidates: list) -> int | None:
    """The line that turns least from this end, or None when nothing
    continues from it."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: (c[2], c[0]))[0]


# --------------------------------------------------------------------------- #
# step 1: lower the sandwich segments
# --------------------------------------------------------------------------- #

def lower_sandwiches(speed: np.ndarray, length: np.ndarray, excluded: np.ndarray,
                     touching: list, along: list, max_length_m: float):
    """Repeat the step-1 rule until a pass changes nothing.

    Returns (new speed, is_sandwich, source neighbour per row (-1 if none),
    number of passes that changed something)."""
    speed = speed.astype(float).copy()
    is_sandwich = np.zeros(len(speed), dtype=bool)
    source = np.full(len(speed), -1, dtype=int)
    eligible = [i for i in range(len(speed))
                if not excluded[i] and length[i] <= max_length_m and along[i] is not None]
    passes = 0
    while True:
        updates = {}
        for i in eligible:
            around = touching[i][0] | touching[i][1]
            if not around:
                continue
            chi = max(speed[j] for j in around)
            if chi < speed[i]:
                a, b = along[i]
                updates[i] = a if speed[a] >= speed[b] else b
        if not updates:
            return speed, is_sandwich, source, passes
        passes += 1
        new_speed = speed.copy()
        for i, src in updates.items():
            new_speed[i] = speed[src]
            is_sandwich[i] = True
            source[i] = src
        speed = new_speed


# --------------------------------------------------------------------------- #
# step 2: join a sandwich back to its own pieces
# --------------------------------------------------------------------------- #

def _groups_to_absorb(is_sandwich, speed, parent, along) -> list[list[int]]:
    """Union-find over (sandwich row, along-road neighbour) pairs that share
    speed and parent id. Returns the groups with more than one row."""
    root = list(range(len(speed)))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for i in np.flatnonzero(is_sandwich):
        if along[i] is None:
            continue
        for n in along[i]:
            if speed[n] == speed[i] and parent[n] == parent[i]:
                root[find(i)] = find(n)
    groups: dict[int, list[int]] = {}
    for i in range(len(speed)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def absorb_sandwiches(gdf: gpd.GeoDataFrame, groups: list[list[int]], length: np.ndarray, *,
                      id_col: str, parent_id_col: str, length_col: str | None) -> gpd.GeoDataFrame:
    """Join each group (positions into `gdf`) into one row."""
    if not groups:
        return gdf
    is_sw = gdf["is_sandwich"].to_numpy()
    merged_rows, drop = [], set()
    for g in groups:
        plain = [i for i in g if not is_sw[i]] or g
        base = max(plain, key=lambda i: length[i])
        row = gdf.iloc[base].to_dict()
        geom = linemerge(unary_union([gdf.geometry.iloc[i] for i in g]))
        row[gdf.geometry.name] = geom
        if length_col is not None and length_col in gdf.columns:
            row[length_col] = float(sum(length[i] for i in g))
        row["is_sandwich"] = False
        merged_rows.append((g, row))
        drop.update(g)

    kept = gdf.drop(index=gdf.index[sorted(drop)])
    remaining_parents = pd.Series(kept[parent_id_col].astype(str).to_numpy())
    for g, row in merged_rows:
        others = remaining_parents.eq(str(row[parent_id_col])).any() or any(
            str(r[parent_id_col]) == str(row[parent_id_col]) for h, r in merged_rows if h is not g)
        if not others:
            row[id_col] = row[parent_id_col]
    joined = gpd.GeoDataFrame([r for _, r in merged_rows], geometry=gdf.geometry.name, crs=gdf.crs)
    out = pd.concat([kept, joined], ignore_index=True)
    return gpd.GeoDataFrame(out, geometry=gdf.geometry.name, crs=gdf.crs)


# --------------------------------------------------------------------------- #
# public
# --------------------------------------------------------------------------- #

def smooth_sandwich_segments(
    gdf: gpd.GeoDataFrame,
    *,
    max_length_m: float,
    parent_id_col: str,
    speed_col: str = "v_safe",
    id_col: str = "segment_id",
    excluded_col: str | None = "is_access_controlled",
    basis_col: str | None = "v_safe_basis",
    copy_cols: tuple = ("collision_type",),
    length_col: str | None = "shape_length",
    touch_tol_m: float = 1.0,
    country_col: str = "country",
    crs_by_country: dict = BUFFER_CRS,
) -> tuple[gpd.GeoDataFrame, int]:
    """Apply steps 1 and 2 (module docstring) per country.

    A lowered row's `basis_col` becomes "sandwich:" + the source neighbour's
    basis, and `copy_cols` are copied from that neighbour. `length_col` is
    rewritten on joined rows. Returns the new frame and the largest number of
    step-1 passes that changed something in any country."""
    gdf = gdf.copy()
    gdf["is_sandwich"] = False
    out, max_passes = [], 0
    for country in gdf[country_col].unique():
        sub = gdf[gdf[country_col] == country].reset_index(drop=True)
        lines = sub.geometry.to_crs(crs_by_country[country]).to_numpy()
        touching, continuing, length = build_endpoint_adjacency(lines, touch_tol_m)
        along = []
        for i in range(len(sub)):
            a = pick_along_road_neighbor(continuing[i][0])
            b = pick_along_road_neighbor(continuing[i][1])
            along.append((a, b) if a is not None and b is not None else None)

        excluded = (sub[excluded_col].fillna(False).astype(bool).to_numpy()
                    if excluded_col else np.zeros(len(sub), dtype=bool))
        old = sub[speed_col].to_numpy()
        speed, is_sw, source, passes = lower_sandwiches(
            old, length, excluded, touching, along, max_length_m)
        max_passes = max(max_passes, passes)

        changed = np.flatnonzero(is_sw)
        src = source[changed]
        sub[speed_col] = speed.astype(sub[speed_col].dtype)
        sub["is_sandwich"] = is_sw
        if basis_col is not None and basis_col in sub.columns:
            basis = sub[basis_col].to_numpy(dtype=object).copy()
            # The source's own basis as it stood before this step, so a chain
            # of sandwiches does not stack "sandwich:sandwich:...".
            orig = sub[basis_col].to_numpy(dtype=object)
            for i, s in zip(changed, src):
                b = str(orig[s])
                basis[i] = b if b.startswith("sandwich:") else f"sandwich:{b}"
            sub[basis_col] = basis
        for col in copy_cols:
            if col in sub.columns:
                values = sub[col].to_numpy(dtype=object).copy()
                values[changed] = sub[col].to_numpy(dtype=object)[src]
                sub[col] = values

        groups = _groups_to_absorb(is_sw, speed, sub[parent_id_col].astype(str).to_numpy(), along)
        n_rows = len(sub)
        sub = absorb_sandwiches(sub, groups, length, id_col=id_col,
                                parent_id_col=parent_id_col, length_col=length_col)
        print(f"[sandwich] {country}: {len(changed)} segments lowered in {passes} passes; "
              f"{len(groups)} joined back to their own pieces ({n_rows} -> {len(sub)} rows)")
        out.append(sub)

    result = pd.concat(out, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry=gdf.geometry.name, crs=gdf.crs), max_passes
