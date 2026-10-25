"""
Download daily climate (temperature, humidity, rainfall) from NASA POWER for one representative
city per region. Saved as DAILY data; we aggregate to the bulletin's epi-weeks later.

Run:  python download_climate.py
Output: data/climate_daily.csv
Columns: date, region, city, T2M (deg C), RH2M (%), PRECTOTCORR (mm/day)
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M,RH2M,PRECTOTCORR"
START = "20230101"
END = (date.today() - timedelta(days=5)).strftime(
    "%Y%m%d"
)  # POWER lags a few days behind

# One representative point per region (a simplification - mention it in the paper's limitations)
POINTS = [
    ("Sindh", "Karachi", 24.86, 67.01),
    ("KP", "Peshawar", 34.01, 71.58),
    ("Balochistan", "Quetta", 30.18, 66.99),
    ("ICT", "Islamabad", 33.68, 73.05),
    ("Punjab", "Lahore", 31.55, 74.34),
    ("AJK", "Muzaffarabad", 34.37, 73.47),
    ("GB", "Gilgit", 35.92, 74.31),
]
OUT = Path("data/climate_daily.csv")


def to_frame(payload, region, city):
    """NASA POWER JSON -> tidy DataFrame (-999 means missing)."""
    block = payload["properties"]["parameter"]
    df = pd.DataFrame(block)  # index = YYYYMMDD strings, columns = parameters
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.where(df > -900)  # -999 -> NaN
    df.insert(0, "city", city)
    df.insert(0, "region", region)
    df.index.name = "date"
    return df.reset_index()


def fetch(region, city, lat, lon, retries=3):
    q = {
        "parameters": PARAMS,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": START,
        "end": END,
        "format": "JSON",
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(API, params=q, timeout=120)
            r.raise_for_status()
            return to_frame(r.json(), region, city)
        except Exception as e:
            print(f"   attempt {attempt}/{retries} failed: {e}")
            time.sleep(3 * attempt)
    return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for region, city, lat, lon in POINTS:
        print(f"{region} ({city}) ...")
        df = fetch(region, city, lat, lon)
        if df is None:
            print(f"   FAILED: {region}")
            continue
        frames.append(df)
        time.sleep(1)
    if not frames:
        sys.exit("No climate data downloaded. Copy the error above and send it to me.")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(
        f"\nSaved {OUT} | rows: {len(out)} | regions: {out['region'].nunique()} "
        f"| {out['date'].min().date()} -> {out['date'].max().date()}"
    )
    print(
        out.groupby("region")[["T2M", "RH2M", "PRECTOTCORR"]]
        .apply(lambda g: g.isna().mean())
        .round(3)
    )


if __name__ == "__main__":
    main()
