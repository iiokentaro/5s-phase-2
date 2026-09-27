"""Sample WorldPop population density along each segment.

Samples at the start, midpoint and end of each LineString and takes the max:
a midpoint sample alone can miss a roadside settlement at one end of a long
rural segment, the case the 100 m raster resolution was chosen to catch.
"""

import sys
import warnings

import numpy as np
import rasterio

sys.path.insert(0, "src")
from schema import load_target

warnings.filterwarnings("ignore", category=UserWarning)

RASTER_PATHS = {
    "thailand": "data/external/tha_ppp_2020.tif",
    "maharashtra": "data/external/maharashtra_ppp_2020.tif",
}


def _sample_points(geom):
    return [geom.interpolate(f, normalized=True) for f in (0.0, 0.5, 1.0)]


def add_pop_density(gdf):
    gdf = gdf.copy()
    gdf["pop_density"] = np.nan

    for country, raster_path in RASTER_PATHS.items():
        mask = gdf["country"] == country
        if not mask.any():
            continue
        with rasterio.open(raster_path) as src:
            nodata = src.nodata
            coords = []
            counts = []
            for geom in gdf.loc[mask, "geometry"]:
                pts = _sample_points(geom)
                coords.extend((p.x, p.y) for p in pts)
                counts.append(len(pts))

            values = np.array([v[0] for v in src.sample(coords)], dtype=float)
            values[values == nodata] = 0.0  # outside-raster/nodata treated as unpopulated
            values = np.maximum(values, 0.0)  # WorldPop has no negative populations

            # un-flatten: 3 values per segment -> max per segment
            maxima = []
            i = 0
            for n in counts:
                maxima.append(values[i : i + n].max())
                i += n
            gdf.loc[mask, "pop_density"] = maxima

    return gdf


if __name__ == "__main__":
    target = load_target()
    target = add_pop_density(target)
    print(target.groupby("country")["pop_density"].describe())
    print("\nmissing pop_density:", target["pop_density"].isna().sum())
