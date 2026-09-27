"""Inspect the raw Thailand / Maharashtra CSV inputs (structure, dtypes, missingness)."""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

RAW = {
    "thailand": "data/raw/ADB_Innovation_Thailand.csv",
    "maharashtra": "data/raw/ADB_Innovation_Maharashtra.csv",
}


def inspect(name: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n{'=' * 80}\n{name.upper()}  shape={df.shape}\n{'=' * 80}")
    print("\n--- dtypes ---")
    print(df.dtypes)
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print("\n--- missing values (count, %) ---")
    print(pd.DataFrame({"missing": missing, "pct": missing_pct})[missing > 0])
    print("\n--- describe (numeric) ---")
    print(df.describe())
    return df


if __name__ == "__main__":
    frames = {name: inspect(name, path) for name, path in RAW.items()}

    th, ind = frames["thailand"], frames["maharashtra"]

    print(f"\n{'=' * 80}\nKEY COLUMN CHECKS\n{'=' * 80}")

    print("\n--- Thailand SpeedLimit value counts ---")
    print(th["SpeedLimit"].value_counts(dropna=False))

    print("\n--- Maharashtra SpeedLimit value counts ---")
    print(ind["SpeedLimit"].value_counts(dropna=False))

    print("\n--- Thailand LandUse value counts ---")
    print(th["LandUse"].value_counts(dropna=False))

    print("\n--- Maharashtra LandUse value counts ---")
    print(ind["LandUse"].value_counts(dropna=False))

    print("\n--- Maharashtra UrbanPC describe ---")
    print(ind["UrbanPC"].describe())

    print("\n--- Thailand RoadClass value counts ---")
    print(th["RoadClass"].value_counts(dropna=False))

    print("\n--- Maharashtra RoadClass value counts ---")
    print(ind["RoadClass"].value_counts(dropna=False))

    print("\n--- 85th percentile >= median speed check ---")
    th_bad = th[th["F85thPercentileSpeed"] < th["MedianSpeed"]]
    ind_bad = ind[ind["F85thPercentileSpeed"] < ind["MedianSpeed"]]
    print(f"Thailand rows where F85 < Median: {len(th_bad)} / {len(th)}")
    print(f"Maharashtra rows where F85 < Median: {len(ind_bad)} / {len(ind)}")

    print("\n--- StreetImageLink sample ---")
    print(th["StreetImageLink"].dropna().iloc[0])
    print(ind["StreetImageLink"].dropna().iloc[0])

    print("\n--- SampleSizeTotal / Sample_Size_Total describe ---")
    print(th["SampleSizeTotal"].describe())
    print(ind["Sample_Size_Total"].describe())
