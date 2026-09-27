"""Priority-segment deliverable map.

Three outputs, all built from the same `priority_class` / `review_track`
columns so they can never drift from each other or from the review lists in
review_track.py:
- an interactive Folium/Leaflet HTML map (primary deliverable, clickable
  popups explain each segment in plain language)
- GeoParquet, a GeoPackage and a zipped File Geodatabase (the geodatabase is
  the file uploaded to ArcGIS Online)
- a static PNG (fallback for a reviewer who can't open the HTML)

★ Display category: map_class (marks Aligned within "Low Priority") ★
The data column priority_class is never modified. For display only,
derive_map_class() gives the "Low Priority" rows whose posted limit is not above
V_safe (misalignment <= 0) their own map_class value, Aligned: nothing to lower
there. The other Low Priority rows (a gap exists, but the composite score is
below the top 20%) keep Low Priority (see README). Aligned/Low Priority are rendered without popups (the bulk of the
network) to keep the HTML light; Top Priority/Priority/Watch get full popups.

② All five valid categories (Top Priority/Priority/Watch/Low Priority/Aligned) are shown by default
at load. Data Quality Issue stays toggle-on.

③ V_safe-driving point sources are plotted as clustered markers
(folium.plugins.MarkerCluster): Mapillary VRU detections, school POIs (OSM + Overture), and
junction nodes -- the features that actually localize each segment's V_safe.

④ Two continuous-color speed layers (off by default): current posted
`speed_limit` vs recommended `v_safe`, one FeatureGroup per country per field.
Both use the same RdYlGn/30-100km/h scale as v_safe_map.png so switching
between them (or eyeballing both at once) reads as "did the color change".
These are popup-free -- they're a pure choropleth, and each already duplicates
the full network once per field, so no popup fields are serialized at all.
Unlike the priority-class layers, both speed layers include
data_quality_flag=='invalid_speed' rows too (excluded elsewhere on the map):
in the current-SpeedLimit layer they're drawn black (speed_limit==0 there is
a data artifact, not a real posted limit -- see SPEED_LAYER_FIELDS), while in
the V_safe layer they're colored normally, since V_safe is computed
independently of speed_limit and is real for every row.

⑤ 300m junction buffer (off by default): one true circle
(folium.Circle/Leaflet L.circle, not a shapely.buffer() polygon) per cached
junction node, radius=JUNCTION_BUFFER_M -- the exact same zone
junction_speed_cap.py's dwithin check uses to cap V_safe to 50km/h. A real
circle needs only center+radius, so this stays light even at ~10,700 points
combined, unlike a buffered polygon which would need dozens of vertices per
point to look round.

★ review_track is a line style ★
review_track is rendered as the solid/dashed line style of the country x
map_class layers. A separate Review Needed / Field Verification Needed layer
would embed every one of those rows a second time.

Geometry for the interactive map is rendered at full resolution (no Douglas-Peucker
simplification). Coordinates are rounded to COORD_ROUND_DECIMALS (~1.1 m) only to
cut HTML weight; the GeoParquet/geodatabase exports are also unaffected.
Popup field values are rounded (see _sanitize_for_geojson) before being turned
into strings, since a float's full repr adds bytes with no reading value.

★ Visual encoding ★
Color = map_class (Top Priority=red, Priority=orange, Watch=yellow, Low Priority=yellow-green,
Aligned=cyan, Data Quality Issue=dark grey). Line style = review_track
(Review Needed=solid, Field Verification Needed=dashed). Points colored by source type.

The layer tree (folium.plugins.TreeLayerControl) groups overlays by country, by
map_class within country, a speed-layer group, and a separate POI/junction
point group.
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from folium.plugins import FastMarkerCluster, TreeLayerControl

sys.path.insert(0, "src")
import elvik_2019
from exposure_signals import load_mapillary_pois
from poi_sources import load_pois
from junction_speed_cap import JUNCTION_BUFFER_M, load_cached_junctions
from geometry import geodesic_length_m
from kepler_parquet import write_kepler_parquet

# Display-only split of the "Low Priority" tier (the underlying priority_class column
# is never changed): Aligned = the posted limit is not above V_safe (misalignment <= 0,
# nothing to lower); the rest keep Low Priority (a gap exists, or is unknown, but the
# composite score is below the top 20%) -- see README.
MAP_ALIGNED = "Aligned"
MAP_LOW_PRIORITY = "Low Priority"

PRIORITY_COLORS = {
    "Top Priority": "#d73027",
    "Priority": "#fc8d59",
    "Watch": "#fee08b",
    MAP_LOW_PRIORITY: "#a6d96a",  # yellow-green: gap exists but low priority
    MAP_ALIGNED: "#4dd0e1",       # light blue: limit not above V_safe
    "Data Quality Issue (Excluded)": "#636363",
}
# All line categories are rendered at the same weight; color alone distinguishes them.
LINE_WEIGHT = 2
PRIORITY_WEIGHT = {cls: LINE_WEIGHT for cls in [
    "Top Priority", "Priority", "Watch", MAP_LOW_PRIORITY, MAP_ALIGNED,
    "Data Quality Issue (Excluded)",
]}
# The five valid priority tiers shown by default (② all categories on at load).
# Aligned/Low Priority are rendered without popups to keep the HTML light (they
# are the bulk of the network, ~"no action needed").
MAP_CLASSES_VALID = ["Top Priority", "Priority", "Watch", MAP_LOW_PRIORITY, MAP_ALIGNED]
POPUP_CLASSES = {"Top Priority", "Priority", "Watch"}
LIGHT_CLASSES = {MAP_ALIGNED, MAP_LOW_PRIORITY}

# V_safe-driving point sources plotted as clustered markers (③).
POINT_STYLES = {
    "mapillary_vru": {"color": "#762a83", "label": "Mapillary VRU Detection (signs/markings)"},
    "school": {"color": "#1b7837", "label": "School (OSM + Overture)"},
    "junction": {"color": "#2166ac", "label": "Junction Node"},
}
_MAPILLARY_VRU_FLAGS = ["map_is_pedestrian", "map_is_bicycle", "map_is_school"]

COORD_ROUND_DECIMALS = 5  # ~1.1m; rounds rendered line coords to cut HTML weight (display only)

# ④ Continuous speed layers (current SpeedLimit vs recommended V_safe). Same
# vmin/vmax as build_v_safe.py:plot_map's RdYlGn scale for v_safe_map.png,
# so the two interactive layers and the static V_safe map all read the same way.
SPEED_COLOR_VMIN, SPEED_COLOR_VMAX = 30, 100
# label -> (field, flag_missing). flag_missing=True draws data_quality_flag==
# 'invalid_speed' rows black -- only meaningful for the
# current-SpeedLimit layer (those rows' speed_limit==0 is a data artifact, not
# a real posted limit); v_safe is computed independently of speed_limit and is
# always real, so those same rows get colored normally in that layer.
SPEED_LAYER_FIELDS = {
    "Current Speed Limit (SpeedLimit)": ("speed_limit", True),
    "Recommended Safe Speed (V_safe)": ("v_safe", False),
}

# Popup-enabled layers only. road_environment is left out: it is
# implied by road_class/land_use already shown. The fatal-crash reduction is
# the Elvik (2019) figure (elvik_2019.REPORTED_COLUMN); it has no CI to show.
POPUP_FIELDS = [
    "score_explanation", "priority_class", "review_track",
    "speed_limit", "speed_data_source", "v_safe", "misalignment", "exposure_level",
    "confidence_level", "speedlimit_plausibility", "road_class", "land_use",
    "exp_delta_fatal_percent_uniform",
    "country", "street_image_link",
]
POPUP_ALIASES = [
    "Explanation", "Priority Class", "Review Track",
    "Speed Limit (km/h)", "Speed Data Source", "Safe Speed V_safe (km/h)", "Gap (km/h)", "VRU Exposure Level",
    "Data Confidence", "Speed Limit Plausibility", "Road Class", "Land Use",
    "Estimated Fatal Crash Reduction (Elvik 2019, %)",
    "Country", "Field-check coordinates (lon1,lat1,lon2,lat2)",
]

# Numeric fields get rounded before being turned into popup strings (see
# _sanitize_for_geojson) -- a raw float repr (e.g. "91.744665...") is bytes
# of precision nobody reads in a popup.
POPUP_ROUND_DECIMALS = {
    "speed_limit": 0, "v_safe": 0, "misalignment": 1, "exp_delta_fatal_percent_uniform": 1,
}


def derive_map_class(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Display-only category: marks Aligned within the valid "Low Priority" tier.

    priority_class is left untouched in the data; this only drives map color/layers.
    - Top Priority / Priority / Watch / Data Quality Issue: unchanged.
    - Low Priority & misalignment <= 0  -> Aligned      (posted limit not above V_safe).
    - Low Priority & misalignment >  0  -> Low Priority (gap exists but low composite score).
    - Low Priority & misalignment is NA -> Low Priority (conservative: don't label as Aligned).
    """
    pc = gdf["priority_class"].astype(str)
    mis = pd.to_numeric(gdf["misalignment"], errors="coerce")
    return pc.mask((pc == MAP_LOW_PRIORITY) & (mis <= 0), MAP_ALIGNED)


def _sanitize_for_geojson(gdf: gpd.GeoDataFrame, fields: list[str]) -> gpd.GeoDataFrame:
    """folium/Leaflet renders fields as raw JS; pd.NA / NaN in object columns
    breaks the popup template, so every field actually shown is forced to a
    plain string with an explicit placeholder for missing values.

    Fields listed in POPUP_ROUND_DECIMALS are rounded to that precision first,
    on the numeric value, so "91.744665..." becomes "91.7", shaving bytes off
    every popup-enabled feature.
    """
    sub = gdf[fields + ["geometry"]].copy()
    for col in fields:
        if col in POPUP_ROUND_DECIMALS:
            decimals = POPUP_ROUND_DECIMALS[col]
            rounded = pd.to_numeric(sub[col], errors="coerce").round(decimals)
            sub[col] = rounded.astype("Int64") if decimals == 0 else rounded
        sub[col] = sub[col].astype(object)
        sub[col] = sub[col].where(sub[col].notna(), "(unknown)").astype(str)
    return sub


def _style_function(feature):
    props = feature["properties"]
    cls = props.get("map_class", props.get("priority_class"))
    style = {
        "color": PRIORITY_COLORS.get(cls, "#999999"),
        "weight": PRIORITY_WEIGHT.get(cls, LINE_WEIGHT),
        "opacity": 0.5 if cls in LIGHT_CLASSES else 0.85,
    }
    if props.get("review_track") == "Field Verification Needed":
        style["dashArray"] = "6,6"
    return style


def _layer_for(gdf, label, with_popup, show=False):
    fg = folium.FeatureGroup(name=label, show=show)
    if len(gdf) == 0:
        return fg
    fields = list(POPUP_FIELDS) if with_popup else ["priority_class", "review_track"]
    if "map_class" not in fields:
        fields = fields + ["map_class"]
    sanitized = _sanitize_for_geojson(gdf, fields)
    gj = folium.GeoJson(
        sanitized,
        style_function=_style_function,
        popup=folium.GeoJsonPopup(fields=POPUP_FIELDS, aliases=POPUP_ALIASES, max_width=320) if with_popup else None,
    )
    gj.add_to(fg)
    return fg


SPEED_MISSING_COLOR = "#000000"  # black: no real current speed-limit value to color by


def _speed_style_function(field, colormap, flag_missing: bool):
    """Style-function factory for a continuous speed field (speed_limit / v_safe).

    Values are clipped to [SPEED_COLOR_VMIN, SPEED_COLOR_VMAX] before coloring --
    same clipping the static v_safe_map.png colorbar implicitly applies via
    vmin/vmax -- so an occasional out-of-range value doesn't blow out the scale.

    flag_missing=True (current-SpeedLimit layer only): rows with
    data_quality_flag=='invalid_speed' have speed_limit==0 as a data artifact
    (schema.has_invalid_zero_speeds -- speed_limit/median_speed/f85_speed all
    exactly 0), not a real posted limit, so they're drawn black.
    """
    def _style(feature):
        props = feature["properties"]
        if flag_missing and props.get("data_quality_flag") == "invalid_speed":
            return {"color": SPEED_MISSING_COLOR, "weight": LINE_WEIGHT, "opacity": 0.85}
        try:
            val = float(props.get(field))
        except (TypeError, ValueError):
            color = "#999999"
        else:
            val = min(max(val, SPEED_COLOR_VMIN), SPEED_COLOR_VMAX)
            color = colormap(val)
        return {"color": color, "weight": LINE_WEIGHT, "opacity": 0.85}
    return _style


def _speed_layer_for(gdf, label, field, colormap, flag_missing: bool, show=False) -> folium.FeatureGroup:
    """Popup-free choropleth for one speed field. Only the fields the style
    function actually reads are kept in the serialized properties (`field`,
    plus `data_quality_flag` when flag_missing) -- no popup means nothing else
    needs to travel with the feature."""
    fg = folium.FeatureGroup(name=label, show=show)
    if len(gdf) == 0:
        return fg
    fields = [field, "data_quality_flag"] if flag_missing else [field]
    sanitized = _sanitize_for_geojson(gdf, fields)
    gj = folium.GeoJson(
        sanitized,
        style_function=_speed_style_function(field, colormap, flag_missing),
    )
    gj.add_to(fg)
    return fg


def _vru_points(country: str) -> gpd.GeoDataFrame:
    map_pois = load_mapillary_pois(country)
    if len(map_pois) == 0:
        return map_pois
    mask = pd.Series(False, index=map_pois.index)
    for flag in _MAPILLARY_VRU_FLAGS:
        if flag in map_pois.columns:
            mask = mask | (map_pois[flag] == True)  # noqa: E712
    return map_pois[mask]


def _school_points(country: str) -> gpd.GeoDataFrame:
    pois = load_pois(country)
    return pois[pois["is_school"]]


def _point_layer(points_gdf, label, color, show) -> folium.FeatureGroup:
    """Clustered point layer using FastMarkerCluster.

    FastMarkerCluster serializes only a [[lat, lon], ...] array plus one JS
    callback, which is far lighter than one folium marker per point (tens of
    thousands of VRU/junction points would otherwise bloat the HTML)."""
    fg = folium.FeatureGroup(name=label, show=show)
    if points_gdf is None or len(points_gdf) == 0:
        return fg
    pts = points_gdf
    if pts.crs is not None and str(pts.crs).upper() != "EPSG:4326":
        pts = pts.to_crs("EPSG:4326")
    coords = []
    for geom in pts.geometry:
        if geom is None or geom.is_empty:
            continue
        # representative_point() for any non-Point geometry (some OSM POIs are ways).
        pt = geom if geom.geom_type == "Point" else geom.representative_point()
        coords.append([round(pt.y, COORD_ROUND_DECIMALS), round(pt.x, COORD_ROUND_DECIMALS)])
    if not coords:
        return fg
    callback = (
        "function (row) {"
        f"  return L.circleMarker(new L.LatLng(row[0], row[1]), "
        f"{{radius: 3, color: '{color}', fillColor: '{color}', fillOpacity: 0.7, weight: 1}});"
        "}"
    )
    FastMarkerCluster(data=coords, callback=callback).add_to(fg)
    return fg


def _junction_buffer_layer(points_gdf, label, color, radius_m, show=False) -> folium.FeatureGroup:
    """True-circle junction buffer (Leaflet L.circle: center + radius in
    meters), one folium.Circle per junction node -- the exact JUNCTION_BUFFER_M
    zone junction_speed_cap.py's dwithin check caps V_safe within. A circle
    needs only a centre and a radius; a shapely.buffer() polygon would need
    dozens of vertices per point to look round, for ~10,700 points combined
    across both countries."""
    fg = folium.FeatureGroup(name=label, show=show)
    if points_gdf is None or len(points_gdf) == 0:
        return fg
    pts = points_gdf
    if pts.crs is not None and str(pts.crs).upper() != "EPSG:4326":
        pts = pts.to_crs("EPSG:4326")
    for geom in pts.geometry:
        if geom is None or geom.is_empty:
            continue
        pt = geom if geom.geom_type == "Point" else geom.representative_point()
        folium.Circle(
            location=[round(pt.y, COORD_ROUND_DECIMALS), round(pt.x, COORD_ROUND_DECIMALS)],
            radius=radius_m,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
        ).add_to(fg)
    return fg


def _round_geometry(geom):
    """Round line coordinates to COORD_ROUND_DECIMALS (~1m) to cut HTML weight.
    Display-only; never applied to the GeoParquet/geodatabase exports."""
    if geom is None or geom.is_empty:
        return geom
    return shapely.transform(geom, lambda a: np.round(a, COORD_ROUND_DECIMALS))


def build_priority_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    # The popup's fatal-crash reduction column; a stored parquet from before it
    # existed gets it computed here.
    gdf = elvik_2019.ensure_exponential_columns(gdf)
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_round_geometry)
    gdf["map_class"] = derive_map_class(gdf)

    valid = gdf[gdf["data_quality_flag"].isna()]
    invalid = gdf[gdf["data_quality_flag"].notna()]

    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=4, tiles="cartodbpositron")

    country_tree = []
    for country in ["thailand", "maharashtra"]:
        country_valid = valid[valid["country"] == country]
        children = []
        for cls in MAP_CLASSES_VALID:
            sub = country_valid[country_valid["map_class"] == cls]
            # ② all five categories shown by default; Aligned/Low Priority stay popup-less to
            # keep the HTML light (they are the bulk of the network).
            fg = _layer_for(sub, f"{country}: {cls} (n={len(sub)})",
                            with_popup=(cls in POPUP_CLASSES), show=True)
            fg.add_to(fmap)
            children.append({"label": f"{cls} (n={len(sub)})", "layer": fg})
        country_tree.append({"label": f"{country} (n={len(country_valid)})", "children": children, "collapsed": True})

    # review_track is shown by the dashArray styling above (Review Needed=solid /
    # Field Verification Needed=dashed); see the module docstring.

    # ③ V_safe-driving point sources (Mapillary VRU / school POIs / junctions),
    # clustered. Shown by default so reviewers can see what localized each V_safe.
    point_children = []
    buffer_children = []
    for country in ["thailand", "maharashtra"]:
        vru = _vru_points(country)
        school = _school_points(country)
        junc = load_cached_junctions(country)
        ckids = []
        for key, pts in (("mapillary_vru", vru), ("school", school), ("junction", junc)):
            style = POINT_STYLES[key]
            n = 0 if pts is None else len(pts)
            fg = _point_layer(pts, f"{country}: {style['label']} (n={n})", style["color"], show=True)
            fg.add_to(fmap)
            ckids.append({"label": f"{style['label']} (n={n})", "layer": fg})
        point_children.append({"label": f"{country} points", "children": ckids, "collapsed": True})

        # ⑤ 300m junction buffer (true circle, not a buffered polygon) -- the
        # exact JUNCTION_BUFFER_M zone junction_speed_cap.py caps V_safe within.
        # Off by default: ~10,700 circles combined would otherwise dominate the
        # initial view if shown alongside the priority-class layers.
        n_junc = 0 if junc is None else len(junc)
        buf_fg = _junction_buffer_layer(junc, f"{country} (n={n_junc})",
                                         POINT_STYLES["junction"]["color"], JUNCTION_BUFFER_M, show=False)
        buf_fg.add_to(fmap)
        buffer_children.append({"label": f"{country} (n={n_junc})", "layer": buf_fg})
    country_tree.append({"label": "POI/Junctions (V_safe-driving features, points)", "children": point_children, "collapsed": True})
    country_tree.append({
        "label": f"Junction buffer (radius {JUNCTION_BUFFER_M}m circle, same zone as the V_safe cap)",
        "children": buffer_children,
        "collapsed": True,
    })

    # ④ Continuous speed layers: current SpeedLimit vs recommended V_safe, same
    # RdYlGn/30-100km/h scale so toggling one then the other reads as "did the
    # color change on this segment". Off by default (each covers the full
    # network -- including data_quality_flag rows, see SPEED_LAYER_FIELDS --
    # on top of the priority-class layers already shown at load).
    speed_colormap = cm.linear.RdYlGn_11.scale(SPEED_COLOR_VMIN, SPEED_COLOR_VMAX)
    speed_colormap.caption = "Speed (km/h, red=low / green=high, black=no current speed-limit data)"
    speed_children = []
    for label, (field, flag_missing) in SPEED_LAYER_FIELDS.items():
        field_children = []
        for country in ["thailand", "maharashtra"]:
            sub = gdf[gdf["country"] == country]
            fg = _speed_layer_for(sub, f"{country} (n={len(sub)})", field, speed_colormap,
                                   flag_missing, show=False)
            fg.add_to(fmap)
            field_children.append({"label": f"{country} (n={len(sub)})", "layer": fg})
        speed_children.append({"label": f"{label} (n={len(gdf)})", "children": field_children, "collapsed": True})
    country_tree.append({
        "label": "Speed layers (current speed limit vs recommended V_safe, color=km/h)",
        "children": speed_children,
        "collapsed": True,
    })
    speed_colormap.add_to(fmap)

    dq_label = f"Data Quality Issue (Excluded, n={len(invalid)})"
    dq_fg = _layer_for(invalid, dq_label, with_popup=False, show=False)
    dq_fg.add_to(fmap)
    country_tree.append({"label": dq_label, "layer": dq_fg})

    TreeLayerControl(overlay_tree=country_tree).add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #999;
                border-radius: 4px; font-size: 13px; line-height: 1.5;">
      <b>Priority class</b><br>
      <span style="color:#d73027;">━━</span> Top Priority&nbsp;&nbsp;
      <span style="color:#fc8d59;">━━</span> Priority&nbsp;&nbsp;
      <span style="color:#fee08b;">━━</span> Watch<br>
      <span style="color:#a6d96a;">━━</span> Low Priority (gap exists, low score)&nbsp;&nbsp;
      <span style="color:#4dd0e1;">━━</span> Aligned (speed limit &le; V_safe)<br>
      <b>V_safe-driving features (points)</b><br>
      <span style="color:#762a83;">●</span> Mapillary VRU&nbsp;&nbsp;
      <span style="color:#1b7837;">●</span> School (OSM + Overture)&nbsp;&nbsp;
      <span style="color:#2166ac;">●</span> Junction<br>
      <b>Review track</b><br>
      Solid = Review Needed (SpeedLimit record is plausible)&nbsp;&nbsp;
      Dashed = Field Verification Needed (SpeedLimit record looks unreliable)<br>
      <b>Speed layers</b> (hidden by default, toggle via the layer tree): Current Speed Limit (SpeedLimit) / Recommended Safe Speed (V_safe).
      Color follows the colorbar (km/h) at bottom right.<br>
      <span style="color:#000000;">━━</span> Current-speed-limit layer only: no current speed-limit data (`data_quality_flag='invalid_speed'`)<br>
      <b>Junction buffer</b> (hidden by default, toggle via the layer tree): 300m-radius circle centered on each junction node.
      <span style="color:#2166ac;">○</span> Same zone used to cap V_safe at 50km/h.
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap


# TomTom's 19-point speed distribution, one list per row: the selected
# direction's column and the _fwd / _bwd ones. A list column cannot be styled or
# filtered in a GIS viewer, kepler.gl refuses the Arrow list type outright, and
# OGR (build_tiles.py converts this export to GeoJSON for tippecanoe) has no
# field type for one. The distribution is carried as its mean and standard
# deviation, tomtom_mean_speed / tomtom_sd_speed (and their _fwd / _bwd), which
# TomTom reports directly and which are all Elvik (2019)'s normal assumption
# needs. The 19 points stay in data/processed/segments_v_safe.parquet, which is
# where anything that recomputes reads them from.
GEO_EXPORT_DROP_PREFIX = "tomtom_speed_percentiles"

# The pre-split rule's comparison columns (build_v_safe.py --legacy). Same
# reasoning as above: this is a deliverable, and a second recommended speed
# sitting next to the real one is exactly the kind of thing a GIS reader would
# mistake for it. They stay in data/processed/segments_v_safe.parquet, which is
# where a comparison belongs.
GEO_EXPORT_DROP_SUFFIX = "_legacy"


# Annual average daily traffic (vehicles/day, a non-negative integer), the last
# column of the deliverable. aadt_estimation.py puts it on the frame before the
# export, scaling each segment's probe count (sample_size_avg) by a factor
# calibrated against counted traffic. A frame that reaches the export without
# it, such as an older committed parquet or a test fixture, falls back to this
# placeholder so the column is always present and always an integer.
AADT_PLACEHOLDER = 1000

# Empty fields placed after AADT, to be filled in later in the GIS: whether a
# segment gets a median or a sidewalk, and its benefit-cost ratio. They are
# created on every export with no values, as a nullable boolean or float64,
# so each output file has the field with its type from the start.
EMPTY_EXPORT_FIELDS = {
    "apply_median": "boolean",
    "apply_sidewalk": "boolean",
    "bcr": "float64",
}


def _geo_export_frame(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The deliverable's columns: every column but the dropped ones above,
    followed by AADT and the EMPTY_EXPORT_FIELDS. shape_length is replaced by the WGS84 geodesic length
    (metres, float) of the final geometry."""
    export = gdf.drop(columns=[
        c for c in gdf.columns
        if c.startswith(GEO_EXPORT_DROP_PREFIX) or c.endswith(GEO_EXPORT_DROP_SUFFIX)
    ])
    export["shape_length"] = geodesic_length_m(export)
    aadt = export.pop("AADT") if "AADT" in export.columns else None
    export["AADT"] = (np.full(len(export), AADT_PLACEHOLDER, dtype="int64")
                      if aadt is None else aadt.astype("int64"))
    export = export.drop(columns=[c for c in EMPTY_EXPORT_FIELDS if c in export.columns])
    for name, dtype in EMPTY_EXPORT_FIELDS.items():
        export[name] = pd.Series(None, index=export.index, dtype=dtype)
    return export


def write_geo_outputs(gdf: gpd.GeoDataFrame, out_dir: str = "outputs") -> str:
    return write_kepler_parquet(_geo_export_frame(gdf), f"{out_dir}/segments_priority.parquet")


GDB_LAYER = "segments_priority"
GPKG_LAYER = "segments_priority"


def _ogr_export_frame(gdf: gpd.GeoDataFrame, narrow_int64: bool) -> gpd.GeoDataFrame:
    """_geo_export_frame, with each column given a type an OGR field holds.

    An object column of True/False/None becomes a nullable boolean, one of
    numbers becomes float64, and any other object column holding something
    but strings is refused, since a field holds one scalar. With narrow_int64, a 64-bit integer column whose
    values fit in 32 bits (v_safe, the counts and codes) is written as a
    32-bit one."""
    export = _geo_export_frame(gdf)
    for col in export.columns:
        if col == export.geometry.name:
            continue
        s = export[col]
        if str(s.dtype) in ("int64", "Int64"):
            lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
            if narrow_int64 and s.dropna().between(lo, hi).all():
                export[col] = s.astype("int32" if s.dtype == "int64" else "Int32")
            continue
        if s.dtype != object:
            continue
        kinds = set(s.dropna().map(type))
        if kinds <= {bool, np.bool_}:
            export[col] = s.astype("boolean")
        elif all(issubclass(k, (int, float, np.number)) for k in kinds):
            export[col] = pd.to_numeric(s).astype("float64")
        elif not kinds <= {str}:
            raise ValueError(f"column {col!r} holds {sorted(k.__name__ for k in kinds)}; "
                             "a GIS field holds one scalar")
    return export


def write_gdb_zip(gdf: gpd.GeoDataFrame, out_dir: str = "outputs") -> str:
    """outputs/segments_priority.gdb.zip for ArcGIS Online: a zipped File
    Geodatabase (segments_priority.gdb at the root of the archive, as ArcGIS
    Online expects) holding the columns of segments_priority.parquet in one
    feature class, GDB_LAYER. Written with GDAL's OpenFileGDB driver.

    Column types follow _ogr_export_frame. 64-bit integers that fit are
    narrowed to 32 bits: the driver would otherwise store them as floats for
    the ArcGIS versions without 64-bit integers. The geodatabase is built in a
    temporary directory and the archive moved into place, so a failed write
    leaves the previous file intact."""
    export = _ogr_export_frame(gdf, narrow_int64=True)
    path = Path(out_dir) / "segments_priority.gdb.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        gdb = Path(tmp) / "segments_priority.gdb"
        export.to_file(gdb, layer=GDB_LAYER, driver="OpenFileGDB")
        archive = Path(tmp) / path.name
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(gdb.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(tmp).as_posix())
        os.replace(archive, path)
    return str(path)


def write_gpkg(gdf: gpd.GeoDataFrame, out_dir: str = "outputs") -> str:
    """outputs/segments_priority.gpkg: the columns of segments_priority.parquet
    in one GeoPackage layer, GPKG_LAYER. Column types follow _ogr_export_frame;
    64-bit integers stay 64-bit, which GeoPackage holds. Written to a temporary
    file and moved into place, so a failed write leaves the previous file
    intact."""
    export = _ogr_export_frame(gdf, narrow_int64=False)
    path = Path(out_dir) / "segments_priority.gpkg"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / path.name
        export.to_file(staged, layer=GPKG_LAYER, driver="GPKG")
        os.replace(staged, path)
    return str(path)


def plot_static_summary(gdf: gpd.GeoDataFrame, out_path: str = "outputs/priority_map_static.png") -> str:
    labels = ["Top Priority", "Priority", "Watch", MAP_LOW_PRIORITY, MAP_ALIGNED, "Data Quality Issue (Excluded)"]
    cmap = mcolors.ListedColormap([PRIORITY_COLORS[label] for label in labels])
    gdf = gdf.copy()
    gdf["map_class"] = derive_map_class(gdf).astype(str)
    gdf["map_class"] = gdf["map_class"].where(gdf["map_class"].isin(labels), labels[-1])
    code = gdf["map_class"].map({label: i for i, label in enumerate(labels)})

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, country in zip(axes, ["thailand", "maharashtra"]):
        sub_mask = gdf["country"] == country
        gdf[sub_mask].plot(ax=ax, color=cmap(code[sub_mask] / (len(labels) - 1)), linewidth=0.7)
        ax.set_title(f"{country} map_class (n={sub_mask.sum()})")
        ax.set_aspect("equal")

    handles = [plt.Line2D([0], [0], color=PRIORITY_COLORS[label], lw=3, label=label) for label in labels]
    fig.legend(handles=handles, loc="lower center", ncol=len(labels), fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    import geopandas as gpd

    gdf = gpd.read_parquet("data/processed/segments_v_safe.parquet")

    fmap = build_priority_map(gdf)
    html_path = "outputs/priority_map.html"
    fmap.save(html_path)
    print(f"saved {html_path}")

    parquet_path = write_geo_outputs(gdf)
    print(f"saved {parquet_path}")

    png_path = plot_static_summary(gdf)
    print(f"saved {png_path}")

    print("\n=== weight check ===")
    import os
    print(f"HTML size: {os.path.getsize(html_path) / 1e6:.1f} MB")

    print("\n=== geographic sanity check: Top Priority by road_class / land_use ===")
    valid = gdf[gdf["data_quality_flag"].isna()]
    top = valid[valid["priority_class"] == "Top Priority"]
    print(top.groupby(["road_class", "land_use"]).size())
