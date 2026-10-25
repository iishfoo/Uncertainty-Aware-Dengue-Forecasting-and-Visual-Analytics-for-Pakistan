"""
Build the modelling table: weekly panel + weekly climate + data-quality flags.

Run AFTER parse_bulletins.py and download_climate.py:
    python build_dataset.py

Input : data/weekly_panel.csv, data/climate_daily.csv
Output: data/model_table.csv  - complete region x week grid (2023-W22 -> last bulletin week);
        weeks with no bulletin stay in the table, flagged, so gaps are visible and never filled in.
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data")
REGIONS = ["AJK", "Balochistan", "GB", "ICT", "KP", "Punjab", "Sindh"]


def iso_label(ts):
    iso = ts.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def runs(flags, weeks):
    """consecutive True stretches -> list of (first_week, last_week, length)"""
    out, start, prev, n = [], None, None, 0
    for f, w in zip(flags, weeks):
        if f:
            if start is None:
                start, n = w, 0
            prev, n = w, n + 1
        elif start is not None:
            out.append((start, prev, n))
            start = None
    if start is not None:
        out.append((start, prev, n))
    return out


def main():
    panel = pd.read_csv(D / "weekly_panel.csv", parse_dates=["week_start"])
    clim = pd.read_csv(D / "climate_daily.csv", parse_dates=["date"])

    # 1) complete region x week calendar
    weeks = pd.date_range(panel.week_start.min(), panel.week_start.max(), freq="7D")
    grid = pd.MultiIndex.from_product([weeks, REGIONS], names=["week_start", "region"]).to_frame(index=False)
    iso = grid.week_start.dt.isocalendar()
    grid["year"], grid["week"] = iso["year"].astype(int), iso["week"].astype(int)
    have = set(panel.week_start.unique())
    grid["bulletin_missing"] = ~grid.week_start.isin(have)
    keep = ["week_start", "region", "cases", "is_nr", "expected", "received", "compliance", "tests", "positives",
            "lab_format", "national_total"]
    df = grid.merge(panel[keep], on=["week_start", "region"], how="left")

    # 2) weekly climate (ISO weeks, Monday-Sunday); incomplete weeks -> NaN
    clim["week_start"] = clim["date"] - pd.to_timedelta(clim["date"].dt.weekday, unit="D")
    wk = (clim.groupby(["region", "week_start"])
              .agg(t2m=("T2M", "mean"), rh2m=("RH2M", "mean"), rain_mm=("PRECTOTCORR", "sum"), days=("T2M", "count"))
              .reset_index())
    wk.loc[wk.days < 7, ["t2m", "rh2m", "rain_mm"]] = np.nan
    df = df.merge(wk.drop(columns="days"), on=["region", "week_start"], how="left")

    # 3) flags and simple features
    df["is_nr"] = df["is_nr"].fillna(False).astype(bool)
    df["status"] = np.where(df.bulletin_missing, "bulletin_missing", np.where(df.is_nr, "NR", "observed"))
    df["test_positivity"] = np.where((df.lab_format == "tests_pos") & (df.tests > 0), df.positives / df.tests, np.nan)
    df["low_compliance"] = df.compliance < 0.5
    df["sin_week"] = np.sin(2 * np.pi * df.week / 52.18)
    df["cos_week"] = np.cos(2 * np.pi * df.week / 52.18)
    df = df.sort_values(["region", "week_start"]).reset_index(drop=True)
    df.to_csv(D / "model_table.csv", index=False)

    # 4) coverage summary (numbers for the paper)
    print(f"Saved data/model_table.csv | {df.week_start.nunique()} weeks x {df.region.nunique()} regions "
          f"({iso_label(weeks[0])} -> {iso_label(weeks[-1])})")
    print(f"Weeks with no bulletin at all: {int(df[df.region == 'KP'].bulletin_missing.sum())}")
    rows = []
    for r, g in df.groupby("region"):
        g = g[~g.bulletin_missing]
        rows.append({"region": r, "weeks_with_bulletin": len(g), "observed_%": round(100 * (g.status == 'observed').mean(), 1),
                     "NR_%": round(100 * (g.status == 'NR').mean(), 1),
                     "median_compliance_%": round(100 * g.compliance.median(), 1) if g.compliance.notna().any() else None,
                     "compliance<50%_weeks": int(g.low_compliance.sum()),
                     "weeks_with_lab_tests": int(g.tests.notna().sum()),
                     "median_tests": g.tests.median() if g.tests.notna().any() else None})
    print("\n", pd.DataFrame(rows).to_string(index=False))

    pj = df[(df.region == "Punjab") & (~df.bulletin_missing)]
    print("\nPunjab reported (not NR) in these stretches:")
    for a, b, n in runs((pj.status == "observed").tolist(), pj.week_start.tolist()):
        print(f"  {iso_label(a)} -> {iso_label(b)}  ({n} weeks)")
    print("\nClimate columns missing:", int(df.t2m.isna().sum()), "region-weeks (expected: weeks after the last full climate week)")


if __name__ == "__main__":
    main()
