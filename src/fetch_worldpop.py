"""Fetch WorldPop 100m population rasters into data/external/.

WorldPop only distributes population grids at the whole-country level (no
state-level India product), and its HTTP server advertises
`Accept-Ranges: bytes` but doesn't actually honor Range requests (confirmed:
GDAL's /vsicurl/ refuses it, and a manual Range request still returns the
full body with HTTP 200). So a windowed remote read isn't possible -- this
downloads India's full 1.8GB raster once, crops it to the Maharashtra target
segments' bounding box (+0.05 deg margin), and deletes the full-country file.

Thailand's own raster (296MB) is small enough to keep uncropped.
"""

import os
import sys

import rasterio
import requests
from rasterio.windows import from_bounds

sys.path.insert(0, "src")
import warnings

from schema import load_target

warnings.filterwarnings("ignore", category=UserWarning)

EXTERNAL_DIR = "data/external"
URL_PATTERN = "https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{iso3}/{iso3_lower}_ppp_{year}.tif"


def download(url: str, dest: str, timeout: int = 5400) -> None:
    print(f"downloading {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def crop_to_bbox(src_path: str, dst_path: str, bbox: tuple[float, float, float, float]) -> None:
    with rasterio.open(src_path) as src:
        window = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1], transform=src.window_transform(window))
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data, 1)


def fetch_thailand() -> str:
    dest = f"{EXTERNAL_DIR}/tha_ppp_2020.tif"
    if not os.path.exists(dest):
        download(URL_PATTERN.format(year=2020, iso3="THA", iso3_lower="tha"), dest)
    return dest


def fetch_maharashtra() -> str:
    dest = f"{EXTERNAL_DIR}/maharashtra_ppp_2020.tif"
    if os.path.exists(dest):
        return dest

    full_path = f"{EXTERNAL_DIR}/ind_ppp_2020.tif"
    download(URL_PATTERN.format(year=2020, iso3="IND", iso3_lower="ind"), full_path)

    target = load_target()
    bounds = target[target["country"] == "maharashtra"].total_bounds
    margin = 0.05
    bbox = (bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin)
    crop_to_bbox(full_path, dest, bbox)

    os.remove(full_path)  # 1.8GB whole-country file; only the crop is kept
    return dest


if __name__ == "__main__":
    th_path = fetch_thailand()
    maha_path = fetch_maharashtra()
    print(f"Thailand raster: {th_path}")
    print(f"Maharashtra raster (cropped): {maha_path}")
