"""Visual sanity check that representative points land on the expected country
shapes, and a check for points falling outside the expected lon/lat ranges."""

import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from geometry import add_representative_point
from schema import load_combined

EXPECTED_BOUNDS = {
    "thailand": {"lon": (97, 106), "lat": (6, 21)},
    "maharashtra": {"lon": (72, 81), "lat": (15, 23)},
}


def check_outliers(gdf):
    for country, bounds in EXPECTED_BOUNDS.items():
        sub = gdf[gdf["country"] == country]
        lon, lat = sub["rep_point"].x, sub["rep_point"].y
        out_of_bounds = sub[
            (lon < bounds["lon"][0]) | (lon > bounds["lon"][1])
            | (lat < bounds["lat"][0]) | (lat > bounds["lat"][1])
        ]
        print(f"{country}: {len(out_of_bounds)} / {len(sub)} representative points outside expected bounds")
        if len(out_of_bounds):
            print(out_of_bounds[["segment_id", "analysis_status"]].head(10))


def plot(gdf, out_path="outputs/representative_points.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, country in zip(axes, ["thailand", "maharashtra"]):
        sub = gdf[gdf["country"] == country]
        ax.scatter(sub["rep_point"].x, sub["rep_point"].y, s=0.5, alpha=0.4)
        ax.set_title(f"{country} (n={len(sub)})")
        ax.set_aspect("equal")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    combined = add_representative_point(load_combined())
    check_outliers(combined)
    plot(combined)
