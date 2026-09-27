"""Check that the auxiliary data sources (WorldPop population density, OSM
POIs) respond at their fixed URLs, as a test that they are free and
reproducible.
"""

import warnings

import osmnx as ox
import requests

warnings.filterwarnings("ignore", category=UserWarning)

# WorldPop chosen over GHSL: GHSL's bulk/global download was "coming soon" and
# its UI requires clicking individual map tiles, which isn't scriptable.
# WorldPop serves a direct, stable per-country GeoTIFF over plain HTTP.
WORLDPOP_URL_PATTERN = "https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{iso3}/{iso3_lower}_ppp_{year}.tif"

COUNTRIES = {"thailand": "THA", "maharashtra": "IND"}  # Maharashtra has no own ISO3; IND covers the whole country


def check_worldpop_url(iso3: str, year: int = 2020) -> None:
    url = WORLDPOP_URL_PATTERN.format(year=year, iso3=iso3, iso3_lower=iso3.lower())
    resp = requests.head(url, timeout=30)
    size_mb = int(resp.headers.get("Content-Length", 0)) / 1e6
    print(f"{iso3}: {resp.status_code} {url} ({size_mb:.0f} MB)")


def check_osm_poi(bbox: tuple[float, float, float, float]) -> None:
    tags = {"amenity": ["school", "marketplace"], "highway": "bus_stop", "shop": True}
    gdf = ox.features_from_bbox(bbox, tags)
    print(f"OSM POIs in sample bbox: {len(gdf)} total")
    print(f"  schools: {(gdf.get('amenity') == 'school').sum()}")
    print(f"  marketplaces: {(gdf.get('amenity') == 'marketplace').sum()}")
    print(f"  bus stops: {(gdf.get('highway') == 'bus_stop').sum()}")
    print(f"  shop=*: {gdf.get('shop').notna().sum()}")


if __name__ == "__main__":
    print("--- WorldPop population density (CC-BY 4.0) ---")
    for country, iso3 in COUNTRIES.items():
        check_worldpop_url(iso3)

    print("\n--- OSM POI (ODbL), Bangkok sample ---")
    check_osm_poi((100.4, 13.6, 100.7, 13.9))
