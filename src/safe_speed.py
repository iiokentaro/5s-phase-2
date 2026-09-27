"""The V_safe decision: start at 30 km/h and raise only on evidence.

★ Structural guarantee ★
`classify_collision_type` and `compute_v_safe` take no `speed_limit` or
observed-speed argument anywhere in their signatures. V_safe cannot be a
function of the current posted limit even by accident -- it isn't in scope
to use. (Observed F85 enters upstream in exactly one place: the Overture
motorway fallback in road_separation.add_road_structure.)

Fixed speed thresholds are human injury-tolerance values, sourced per value
in README.md "Safe System target speeds by road segment condition". They are
assumptions of the Safe System approach, never fitted to this dataset.

★ Decision order ★
1. Initial value 30 km/h (`pedestrian:default_vru_possible`). Unless there is
   a clear reason pedestrians and cyclists cannot be on the road, assume they
   can be.
2. The raises, from road_separation's two independent flags and from the
   Mapillary objects:
   - is_access_controlled AND is_divided     -> road_class table (80-100):
     no VRU and no head-on conflict.
   - is_access_controlled AND NOT is_divided -> 70 (head_on): no VRU, but
     opposing traffic can still meet.
   - NOT is_access_controlled AND has_vru_barrier -> 50. The column is set
     by segment_detected_objects.apply_object_separation (the build step
     "object_separation", after the sandwich step) on the non-POI segments
     whose Mapillary objects physically separate pedestrians, cyclists and
     motorcyclists; that step recomputes V_safe there and keeps it only
     where it rises.
3. The motorcycle rule, applied to the two access-controlled raises. Both of
   them price a crash between vehicle occupants; a motorcyclist has no
   vehicle around them. Where one may legally be on the road and no physical
   barrier separates them from four-wheeled traffic, V_safe stays at 30
   (`motorcycle`). A median does not count: it separates the two directions,
   and riders still share lanes with cars. `has_motorcycle_separation` is
   that barrier, set by apply_object_separation from the Mapillary objects;
   where it holds, the road carries four-wheeled traffic alone and the two
   raises apply unchanged.
4. Caps, applied after whichever branch fired, so no raise can bypass them:
   - is_vru -> min(V_safe, vru_speed_cap). A Mapillary VRU detection caps at
     30, and a Mapillary hospital sign at the hospital speed when poi_params
     enables hospitals. A binding cap of 30 reads `pedestrian:vru_detected`;
     any other speed reads `pedestrian:vru_cap_<speed>kmh:<source>`.
   - near an at-grade junction -> min(V_safe, 50), in junction_speed_cap.py,
     which runs after this module.
   - inside the walking isochrone of an enabled POI type -> min(V_safe, the
     type's speed), in poi_speed_zones.py, after stage 2.

So a road without access control is 30 km/h unless has_vru_barrier raises
it, with or without VRU evidence. `is_vru` is itself forced False on
access-controlled roads (exposure_signals.add_poi_proximity), so the VRU cap
changes a value only where has_vru_barrier raised one: a segment with
separating barriers where a crosswalk, school-zone sign or bicycle marking
was also detected stays at 30.

★ Why the motorcycle rule lives here ★
`is_access_controlled` means pedestrians and cyclists cannot legally enter --
that is what exposure_signals.vru_mask reads it for. A motorcycle is a motor
vehicle, so whether one may be present says nothing about that flag. The flag
answers one question and this module answers the other.

★ `v_safe_basis` names the evidence, not only the rule ★
On the collision types that only access control can produce (`separated`,
`head_on` and `motorcycle`), the basis string carries a third
component: the value of
`access_control_basis`, i.e. which tag proved access control. So
`separated:trunk:osm_motorway` and `separated:trunk:osm_vru_prohibited` are
the same speed reached from different evidence, and can be counted, filtered
and reviewed apart. Rows that were never raised keep the two-component form.

★ `exposure_level` does not drive this decision ★
The exposure composite (population/POI/crossing density tertile) is reserved
for prioritization only (safety_score.py, priority_lists.py).
"""

import sys

import pandas as pd

sys.path.insert(0, "src")

# Safe System injury-tolerance thresholds. Per-value sources: README.md
# "Safe System target speeds by road segment condition".
# 30 pedestrian (Hussain et al. 2019), 50 side impact and 70 head-on
# (Doecke et al. 2020, Turner et al. 2016), 80-100 separated (ITF 2016,
# Woolley et al. 2018).
V_SAFE_TABLE = {
    "pedestrian": 30,
    "side_impact": 50,
    "head_on": 70,
    # Access-controlled roads a motorcyclist may legally use, with no barrier
    # between riders and four-wheeled traffic. `head_on` (70) and the
    # `separated` table (80-100) both price a crash between people inside
    # vehicles; a rider is outside one, so the unprotected threshold applies.
    "motorcycle": 30,
}
MOTORCYCLE_COLLISION_TYPES = ("motorcycle",)
# "separated" (access-controlled and divided) gets a road_class-dependent value.
SEPARATED_V_SAFE_BY_ROAD_CLASS = {
    "motorway": 100,
    "trunk": 90,
    "primary": 80,
    "secondary": 80,
}
SEPARATED_V_SAFE_DEFAULT = 80

# --- legacy comparison path (see classify_collision_type_legacy) ---
# The pre-split rule trusted road_class=='motorway' as proof of access control
# unless the observed speed contradicted it. Same number as
# road_separation.MOTORWAY_FALLBACK_MIN_F85 and the same reasoning behind it,
# but a different rule: there it lets an unmatched motorway raise V_safe, here
# it took the motorway override away.
MOTORWAY_TAG_SUSPECT_F85_THRESHOLD = 50


def classify_collision_type(
    is_access_controlled: bool,
    is_divided: bool,
    is_vru: bool,
    has_vru_barrier: bool = False,
    motorcycle_may_be_present: bool = True,
    has_motorcycle_separation: bool = False,
) -> tuple[str, str]:
    """Returns (collision_type, basis_suffix): the worst survivable conflict
    that can still occur on the segment, and which rule decided it.

    Which evidence established `is_access_controlled` is not an input here --
    the collision type is the same whichever tag proved it. That evidence is
    recorded by compute_v_safe, which appends `access_control_basis` to the
    basis string so a raise above 30 km/h can be traced to its source.

    `motorcycle_may_be_present` defaults to True: a caller that cannot say is
    treated as one where a rider may be, which is the direction that lowers
    V_safe. `has_motorcycle_separation` is a physical barrier between riders
    and four-wheeled traffic, read from the Mapillary objects; a median is
    not one, since riders still share lanes with cars.
    """
    collision_type, basis = "pedestrian", "default_vru_possible"
    if is_access_controlled:
        if is_divided:
            collision_type, basis = "separated", "access_controlled_divided"
        else:
            collision_type, basis = "head_on", "access_controlled_undivided"
    elif has_vru_barrier:
        collision_type, basis = "side_impact", "vru_barrier"

    # Both raises above assume everyone in the crash is inside a vehicle.
    if collision_type in ("separated", "head_on") and motorcycle_may_be_present:
        if not has_motorcycle_separation:
            collision_type, basis = "motorcycle", "motorcycle_unseparated"

    if is_vru:
        collision_type, basis = "pedestrian", "vru_detected"
    return collision_type, basis


# The collision types that only `is_access_controlled` can produce, the
# motorcycle one included since they are reached through the same branch. On
# these, and only these, the basis string carries a third component naming the
# evidence that established access control (road_separation and
# road_access_join write it into `access_control_basis`): `osm_motorway`,
# `osm_motorroad`, `osm_vru_prohibited`, `overture_motorway_fallback`. Without
# it, a segment raised because foot and bicycle are both tagged `no` is
# indistinguishable in the output from one raised by `highway=motorway`,
# although the two rest on different evidence and are reviewed differently.
ACCESS_CONTROLLED_COLLISION_TYPES = ("separated", "head_on",
                                    *MOTORCYCLE_COLLISION_TYPES)


def compute_v_safe(collision_type: str, basis: str, road_class,
                   access_control_basis=None) -> tuple[int, str]:
    """Returns (v_safe, v_safe_basis). v_safe_basis records exactly which rule
    fired, for segment-level explainability.

    `access_control_basis` is appended as a third colon-separated component on
    the access-controlled collision types, e.g. `separated:trunk:osm_motorway`
    or `head_on:access_controlled_undivided:osm_vru_prohibited`. A missing or
    null value leaves the string at two components (the form of every row
    that was not raised above 30 km/h).
    """
    if collision_type == "separated":
        v_safe = SEPARATED_V_SAFE_BY_ROAD_CLASS.get(road_class, SEPARATED_V_SAFE_DEFAULT)
        out = f"separated:{road_class}"
    else:
        v_safe = V_SAFE_TABLE[collision_type]
        out = f"{collision_type}:{basis}"
    if collision_type in ACCESS_CONTROLLED_COLLISION_TYPES and _basis_text(access_control_basis):
        out = f"{out}:{_basis_text(access_control_basis)}"
    return v_safe, out


def _basis_text(value) -> str:
    """`access_control_basis` is an object column holding pd.NA for rows that
    were never raised; float NaN also reaches here from parquet round-trips."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def classify_collision_type_legacy(road_class, is_separated: bool, is_vru: bool,
                                   motorway_tag_suspect: bool = False) -> str:
    """The pre-split rule, restored verbatim for the comparison path.

    Its default was `head_on` (70), not `pedestrian` (30): a road was assumed
    free of VRU conflict unless one was detected. `motorway` took the separated
    branch regardless of `is_vru` unless its tag looked suspect -- trunk was
    excluded from that override, though the mask upstream had already forced
    `is_vru` False there.
    """
    if road_class == "motorway" and not motorway_tag_suspect:
        return "separated" if is_separated else "head_on"
    if is_vru:
        return "pedestrian"
    return "separated" if is_separated else "head_on"


def compute_v_safe_legacy(collision_type: str, road_class) -> tuple[int, str]:
    """Legacy (v_safe, v_safe_basis), from the same speed tables as the current
    rule: the change was in how the collision type is reached, never in what a
    survivable impact is. Only the basis strings differ."""
    if collision_type == "separated":
        v_safe = SEPARATED_V_SAFE_BY_ROAD_CLASS.get(road_class, SEPARATED_V_SAFE_DEFAULT)
        return v_safe, f"separated:{road_class}"
    return V_SAFE_TABLE[collision_type], collision_type


def _motorcycle_present_col(gdf):
    """Rows where a motorcyclist may legally be, i.e. `legal_motorcycle` is
    anything but `no`. `unknown` and `restricted` count as present: not knowing
    is not the same as knowing they are absent, and the consequence of being
    wrong is a recommended speed a rider cannot survive.

    A missing column means road_access_join did not run, and the whole frame is
    treated as motorcycle-carrying -- the direction that lowers V_safe."""
    if "legal_motorcycle" not in gdf.columns:
        return pd.Series(True, index=gdf.index)
    return gdf["legal_motorcycle"].ne("no").fillna(True).astype(bool)


def _bool_col(gdf, col):
    """A missing column means its step didn't run: fall back to False, the safe
    default for every flag here (no raise, no barrier)."""
    if col not in gdf.columns:
        return pd.Series(False, index=gdf.index)
    return gdf[col].fillna(False).astype(bool)


def _vru_cap_cols(gdf) -> tuple[pd.Series, pd.Series]:
    """`vru_speed_cap` / `vru_cap_source` from exposure_signals.add_poi_proximity.
    A frame that has only `is_vru` (an older caller) caps at the pedestrian
    threshold, which is what `is_vru` has always meant."""
    if "vru_speed_cap" in gdf.columns:
        source = (gdf["vru_cap_source"] if "vru_cap_source" in gdf.columns
                  else pd.Series(None, index=gdf.index, dtype="object"))
        return gdf["vru_speed_cap"].astype(float), source
    cap = pd.Series(float("nan"), index=gdf.index)
    cap[_bool_col(gdf, "is_vru")] = V_SAFE_TABLE["pedestrian"]
    return cap, pd.Series(None, index=gdf.index, dtype="object")


def add_v_safe(gdf, include_legacy: bool = False):
    """`include_legacy` additionally writes the pre-split rule's verdict into
    `collision_type_legacy` / `v_safe_legacy` / `v_safe_basis_legacy` (plus the
    `motorway_tag_suspect_legacy` input it turns on), reading
    `is_separated_legacy` / `is_vru_legacy`."""
    gdf = gdf.copy()
    classified = [
        classify_collision_type(ac, div, False, barrier, moto, moto_sep)
        for ac, div, barrier, moto, moto_sep in zip(
            _bool_col(gdf, "is_access_controlled"),
            _bool_col(gdf, "is_divided"),
            _bool_col(gdf, "has_vru_barrier"),
            _motorcycle_present_col(gdf),
            _bool_col(gdf, "has_motorcycle_separation"),
        )
    ]
    acb = (gdf["access_control_basis"] if "access_control_basis" in gdf.columns
           else pd.Series(None, index=gdf.index, dtype="object"))
    results = [
        compute_v_safe(ct, basis, rc, ac)
        for (ct, basis), rc, ac in zip(classified, gdf["road_class"], acb)
    ]
    collision = pd.Series([ct for ct, _ in classified], index=gdf.index, dtype="object")
    v_safe = pd.Series([r[0] for r in results], index=gdf.index)
    basis = pd.Series([r[1] for r in results], index=gdf.index, dtype="object")

    cap, source = _vru_cap_cols(gdf)
    binds = cap.notna() & (cap <= v_safe)
    at_threshold = binds & (cap == V_SAFE_TABLE["pedestrian"])
    collision[binds] = "pedestrian"
    v_safe[binds] = cap[binds].astype(int)
    basis[at_threshold] = "pedestrian:vru_detected"
    other = binds & ~at_threshold
    basis[other] = ("pedestrian:vru_cap_" + cap[other].astype(int).astype(str) + "kmh:"
                    + source[other].fillna("vru").astype(str))
    gdf["collision_type"] = collision
    gdf["v_safe"] = v_safe.astype(int)
    gdf["v_safe_basis"] = basis

    if include_legacy:
        gdf["motorway_tag_suspect_legacy"] = (gdf["road_class"] == "motorway") & (
            gdf["f85_speed"] < MOTORWAY_TAG_SUSPECT_F85_THRESHOLD
        )
        legacy_types = [
            classify_collision_type_legacy(rc, sep, vru, suspect)
            for rc, sep, vru, suspect in zip(
                gdf["road_class"],
                _bool_col(gdf, "is_separated_legacy"),
                _bool_col(gdf, "is_vru_legacy"),
                gdf["motorway_tag_suspect_legacy"],
            )
        ]
        gdf["collision_type_legacy"] = legacy_types
        legacy_results = [
            compute_v_safe_legacy(ct, rc) for ct, rc in zip(legacy_types, gdf["road_class"])
        ]
        gdf["v_safe_legacy"] = [r[0] for r in legacy_results]
        gdf["v_safe_basis_legacy"] = [r[1] for r in legacy_results]

    return gdf


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from exposure_level import add_exposure_level, apply_rural_safety_margin
    from exposure_signals import add_crossing_signal, add_poi_proximity
    from pop_density import add_pop_density
    from road_separation import add_road_structure
    from schema import load_target

    target = load_target()
    target = add_road_structure(target)  # add_poi_proximity needs is_access_controlled to mask is_vru/is_mapillary_vru
    target = add_pop_density(target)
    target = add_poi_proximity(target)
    target = add_crossing_signal(target)
    target = add_exposure_level(target)  # kept for prioritization only, not read by add_v_safe
    target, thresholds = apply_rural_safety_margin(target)
    target = add_v_safe(target)
    print("rural safety-margin thresholds by country:", thresholds)
    print()

    print(target.groupby(["road_class", "land_use"])["v_safe"].describe())
    print()
    print("v_safe_basis distribution:")
    print(target["v_safe_basis"].value_counts())
