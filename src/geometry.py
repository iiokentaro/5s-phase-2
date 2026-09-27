"""Representative points and UTM projection helpers.

Geometry comes from the GeoJSON LineStrings (see schema.py), in EPSG:4326
with [lon, lat] ordering.
"""

import geopandas as gpd
import numpy as np
from pyproj import Geod

_WGS84 = Geod(ellps="WGS84")


def geodesic_length_m(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """WGS84 geodesic length of each geometry, in metres (float64)."""
    geom = gdf.geometry.to_crs("EPSG:4326")
    return np.array([_WGS84.geometry_length(g) for g in geom], dtype="float64")


def add_representative_point(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The point halfway along each LineString."""
    gdf = gdf.copy()
    gdf["rep_point"] = gdf.geometry.interpolate(0.5, normalized=True)
    return gdf


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def add_utm_epsg(gdf: gpd.GeoDataFrame, point_col: str = "rep_point") -> gpd.GeoDataFrame:
    """Per-row UTM EPSG code, since both Thailand (47N/48N) and Maharashtra
    (43N/44N) straddle two zones -- a single fixed zone would distort
    distances for the rows nearest the zone boundary."""
    gdf = gdf.copy()
    gdf["utm_epsg"] = [
        utm_epsg_for_lonlat(pt.x, pt.y) for pt in gdf[point_col]
    ]
    return gdf


def to_utm_by_zone(gdf: gpd.GeoDataFrame, geom_col: str = "geometry") -> dict[int, gpd.GeoDataFrame]:
    """Split by utm_epsg and reproject each subset to its own zone, since
    geopandas .to_crs() needs one CRS per call."""
    if "utm_epsg" not in gdf.columns:
        raise ValueError("call add_utm_epsg first")
    out = {}
    for epsg, subset in gdf.groupby("utm_epsg"):
        out[epsg] = subset.set_geometry(geom_col).to_crs(epsg=epsg)
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "src")
    from schema import load_combined

    combined = add_utm_epsg(add_representative_point(load_combined()))
    print(combined[["country", "utm_epsg"]].value_counts())
