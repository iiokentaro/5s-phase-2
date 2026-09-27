"""Inventory the Safe System inputs in the provided data: what is available
as a proxy for infrastructure protection and VRU exposure, and what is
missing. SpeedLimit and F85thPercentileSpeed are plotted against each other
as a diagnostic of limit-vs-operating-speed divergence.
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "src")
from schema import load_combined, valid_only

HIGH_CLASS_ROADS = {"motorway", "trunk", "primary"}


def crosstab_road_land_use(df: pd.DataFrame) -> None:
    print("\n--- road_class x land_use, by country ---")
    for country in df["country"].unique():
        sub = df[df["country"] == country]
        ct = pd.crosstab(sub["road_class"], sub["land_use"], margins=True)
        print(f"\n{country}:\n{ct}")

    print("\n--- road_class x land_use, combined ---")
    print(pd.crosstab(df["road_class"], df["land_use"], margins=True))


def urban_high_class_share(df: pd.DataFrame) -> None:
    print("\n--- urban x high-class-road share (VRU risk hotspot proxy) ---")
    for country in df["country"].unique():
        sub = df[df["country"] == country]
        urban_high = sub[(sub["land_use"] == "URBAN") & (sub["road_class"].isin(HIGH_CLASS_ROADS))]
        print(f"{country}: {len(urban_high)} / {len(sub)} = {len(urban_high) / len(sub):.1%}")
    urban_high = df[(df["land_use"] == "URBAN") & (df["road_class"].isin(HIGH_CLASS_ROADS))]
    print(f"combined: {len(urban_high)} / {len(df)} = {len(urban_high) / len(df):.1%}")


def boxplot_f85_by_category(df: pd.DataFrame, out_dir: str = "outputs") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    order = sorted(df["road_class"].dropna().unique())
    data = [df.loc[df["road_class"] == rc, "f85_speed"].dropna() for rc in order]
    axes[0].boxplot(data, tick_labels=order)
    axes[0].set_title("F85thPercentileSpeed by road_class (diagnostic only)")
    axes[0].set_ylabel("km/h")

    order2 = sorted(df["land_use"].dropna().unique())
    data2 = [df.loc[df["land_use"] == lu, "f85_speed"].dropna() for lu in order2]
    axes[1].boxplot(data2, tick_labels=order2)
    axes[1].set_title("F85thPercentileSpeed by land_use (diagnostic only)")
    axes[1].set_ylabel("km/h")

    fig.tight_layout()
    path = f"{out_dir}/f85_by_category.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def sample_size_thresholds(df: pd.DataFrame) -> None:
    print("\n--- sample_size_total distribution (confidence-threshold candidates) ---")
    q = df["sample_size_total"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(q)
    low_conf_cutoff = df["sample_size_total"].quantile(0.25)
    print(f"\ncandidate 'low confidence' cutoff (bottom 25%): sample_size_total < {low_conf_cutoff:.0f}")


def percent_over_limit_extremity(combined: pd.DataFrame) -> None:
    # PercentOverLimit / NumberOverLimit aren't in the common schema (schema.py
    # only carries the analysis-ready columns); pull them from the raw frames.
    sys.path.insert(0, "src")
    import geopandas as gpd

    th = gpd.read_file("data/raw/ADB_Innovation_Thailand.geojson")
    ind = gpd.read_file("data/raw/ADB_Innovation_Maharashtra.geojson")
    th["country"] = "thailand"
    ind["country"] = "maharashtra"
    th = th.rename(columns={"SampleSizeTotal": "sample_size_total"})
    ind = ind.rename(columns={"Sample_Size_Total": "sample_size_total"})
    raw = pd.concat(
        [
            th[["AnalysisStatus", "sample_size_total", "PercentOverLimit", "NumberOverLimit", "country"]],
            ind[["AnalysisStatus", "sample_size_total", "PercentOverLimit", "NumberOverLimit", "country"]],
        ],
        ignore_index=True,
    )
    raw = raw[raw["AnalysisStatus"] == "Valid"]

    low_conf_cutoff = raw["sample_size_total"].quantile(0.25)
    low = raw[raw["sample_size_total"] < low_conf_cutoff]
    high = raw[raw["sample_size_total"] >= low_conf_cutoff]

    print("\n--- PercentOverLimit: low-sample vs high-sample segments ---")
    print(pd.DataFrame({"low_sample (n={})".format(len(low)): low["PercentOverLimit"].describe(),
                          "high_sample (n={})".format(len(high)): high["PercentOverLimit"].describe()}))
    print(f"\nlow-sample segments at PercentOverLimit==0 or 1 (extreme): "
          f"{(low['PercentOverLimit'].isin([0, 1])).sum()} / {len(low)} "
          f"({(low['PercentOverLimit'].isin([0, 1])).mean():.1%})")
    print(f"high-sample segments at PercentOverLimit==0 or 1 (extreme): "
          f"{(high['PercentOverLimit'].isin([0, 1])).sum()} / {len(high)} "
          f"({(high['PercentOverLimit'].isin([0, 1])).mean():.1%})")


def speed_limit_vs_f85_scatter(df: pd.DataFrame, out_dir: str = "outputs") -> None:
    # Diagnostic of limit-vs-operating-speed divergence only.
    fig, ax = plt.subplots(figsize=(6, 6))
    for country, marker in [("thailand", "o"), ("maharashtra", "^")]:
        sub = df[df["country"] == country]
        ax.scatter(sub["speed_limit"], sub["f85_speed"], s=2, alpha=0.3, label=country, marker=marker)
    lim = [0, max(df["speed_limit"].max(), df["f85_speed"].max())]
    ax.plot(lim, lim, "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("SpeedLimit (posted)")
    ax.set_ylabel("F85thPercentileSpeed (observed, diagnostic only)")
    ax.set_title("Limit vs operating-speed divergence (NOT used to set target speed)")
    ax.legend()
    fig.tight_layout()
    path = f"{out_dir}/speedlimit_vs_f85_diagnostic.png"
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    combined = valid_only(load_combined())

    crosstab_road_land_use(combined)
    urban_high_class_share(combined)
    boxplot_f85_by_category(combined)
    sample_size_thresholds(combined)
    percent_over_limit_extremity(combined)
    speed_limit_vs_f85_scatter(combined)
