"""
Rolling-origin forecasting experiment (classic ML part): baselines vs LightGBM, with an ablation
that switches CLIMATE and REPORTING features (compliance, lab tests) on and off.

Run after build_dataset.py:
    python run_models.py
    python run_models.py --regions KP Sindh Balochistan --test-start 2025-05-12 --refit-every 4

Design (so the numbers can be defended in the paper)
  * target: suspected dengue cases 1-4 weeks ahead, per region (direct forecasting, one model per horizon)
  * expanding window; a model is refitted every --refit-every origins using ONLY targets already observed
  * weeks with NR / no bulletin have no target and are never evaluated (and never filled in)
  * LightGBM is trained on log1p(cases) with quantile loss (10%, 50%, 90%) -> forecast + 80% interval
Output: results/pred_classic.csv, results/forecast_future.csv (next 4 weeks after the last bulletin)
"""
import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HORIZONS = [1, 2, 3, 4]
QUANTILES = [0.1, 0.5, 0.9]
CLIM = ["t2m", "rh2m", "rain_mm"]

BASE = ["c_l0", "c_l1", "c_l2", "c_l3", "c_m4", "c_m8", "c_l52", "c_l52_s", "sin_t", "cos_t", "region_code"]
CLIMF = [f"{v}_l{k}" for v in CLIM for k in (0, 2, 4, 6, 8)]
REPF = ["comp_l0", "comp_m4", "comp_d4", "logtests_l0", "pos_l0"]
VARIANTS = {"lgbm_base": BASE, "lgbm_climate": BASE + CLIMF, "lgbm_reporting": BASE + REPF,
            "lgbm_full": BASE + CLIMF + REPF}


def load_table(path="data/model_table.csv", regions=None):
    df = pd.read_csv(path, parse_dates=["week_start"]).sort_values(["region", "week_start"])
    if regions:
        df = df[df.region.isin(regions)]
    return df.reset_index(drop=True)


def region_features(g, h):
    """Features known at each origin week + the target h weeks later, for ONE region (complete weekly grid)."""
    g = g.reset_index(drop=True)
    c = np.log1p(g["cases"])
    f = pd.DataFrame({"origin": g.week_start, "region": g.region, "h": h})
    f["target_week_start"] = g.week_start + pd.to_timedelta(7 * h, unit="D")
    f["y"] = g["cases"].shift(-h)
    for k in range(4):
        f[f"c_l{k}"] = c.shift(k)
    f["c_last"] = c.ffill()
    f["m4_raw"] = g["cases"].rolling(4, min_periods=2).mean()
    f["c_m4"] = c.rolling(4, min_periods=2).mean()
    f["c_m8"] = c.rolling(8, min_periods=3).mean()
    f["c_l52"] = c.shift(52 - h)                                   # same ISO week last year
    f["c_l52_s"] = pd.concat([c.shift(51 - h), c.shift(52 - h), c.shift(53 - h)], axis=1).mean(axis=1)
    tw = f.target_week_start.dt.isocalendar().week.astype(int)
    f["sin_t"], f["cos_t"] = np.sin(2 * np.pi * tw / 52.18), np.cos(2 * np.pi * tw / 52.18)
    for v in CLIM:
        for k in (0, 2, 4, 6, 8):
            f[f"{v}_l{k}"] = g[v].shift(k)
    comp = g["compliance"]
    f["comp_l0"] = comp
    f["comp_m4"] = comp.rolling(4, min_periods=2).mean()
    f["comp_d4"] = f["comp_l0"] - f["comp_m4"]
    f["logtests_l0"] = np.log1p(g["tests"])
    f["pos_l0"] = g["test_positivity"]
    return f


def fit_predict(train, test, cols, q, seed=0):
    m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=200, learning_rate=0.05, num_leaves=6,
                          min_child_samples=8, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=1.0, random_state=seed, n_jobs=1, verbose=-1)
    m.fit(train[cols], np.log1p(train["y"]))
    return np.clip(np.expm1(m.predict(test[cols])), 0, None)


def baselines(rows):
    last = np.expm1(rows["c_last"])
    out = {"naive_last": last,
           "moving_avg4": rows["m4_raw"].fillna(last),
           "seasonal_naive": np.expm1(rows["c_l52"]).fillna(last),
           "seasonal_naive3": np.expm1(rows["c_l52_s"]).fillna(last)}
    return pd.DataFrame(out, index=rows.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/model_table.csv")
    ap.add_argument("--regions", nargs="+", default=["KP", "Sindh", "Balochistan"])
    ap.add_argument("--test-start", default="2025-05-12", help="first forecast origin (a Monday)")
    ap.add_argument("--refit-every", type=int, default=4)
    ap.add_argument("--min-train", type=int, default=40)
    a = ap.parse_args()
    Path("results").mkdir(exist_ok=True)

    df = load_table(a.table, a.regions)
    regs = sorted(df.region.unique())
    rows = pd.concat([region_features(g, h) for h in HORIZONS for _, g in df.groupby("region")], ignore_index=True)
    rows["region_code"] = rows.region.map({r: i for i, r in enumerate(regs)})
    weeks = sorted(df.week_start.unique())
    last = weeks[-1]
    origins = [w for w in weeks if pd.Timestamp(a.test_start) <= w < last]
    refit_pts = origins[:: a.refit_every]
    print(f"regions {regs} | {len(origins)} forecast origins from {origins[0].date()} to {origins[-1].date()} "
          f"| refit every {a.refit_every} origins ({len(refit_pts)} refits) | variants: {list(VARIANTS)}")

    preds = []

    def record(model, sub, q10, q50, q90, future=False):
        preds.append(pd.DataFrame({"model": model, "origin": sub.origin.values, "region": sub.region.values,
                                   "h": sub.h.values, "target_week_start": sub.target_week_start.values,
                                   "y": sub.y.values, "q10": q10, "q50": q50, "q90": q90, "is_future": future}))

    # baselines (no training needed) on every evaluable row
    ev = rows[rows.origin.isin(origins) & rows.y.notna()]
    for name, p in baselines(ev).items():
        record(name, ev, np.nan, p.values, np.nan)

    # LightGBM variants with expanding window + periodic refit
    for i, tr_end in enumerate(refit_pts):
        nxt = refit_pts[i + 1] if i + 1 < len(refit_pts) else last
        for h in HORIZONS:
            rh = rows[rows.h == h]
            train = rh[(rh.target_week_start <= tr_end) & rh.y.notna()]
            test = rh[(rh.origin >= tr_end) & (rh.origin < nxt) & rh.y.notna()]
            if len(train) < a.min_train or test.empty:
                continue
            for name, cols in VARIANTS.items():
                q = {qq: fit_predict(train, test, cols, qq) for qq in QUANTILES}
                lo, md, hi = np.sort(np.vstack([q[0.1], q[0.5], q[0.9]]), axis=0)
                record(name, test, lo, md, hi)
        print(f"  refit {i + 1}/{len(refit_pts)} (data up to {pd.Timestamp(tr_end).date()})", end="\r")
    print()
    out = pd.concat(preds, ignore_index=True)
    out.to_csv("results/pred_classic.csv", index=False)

    # genuine forecast for the weeks after the last bulletin (for the dashboard)
    fut = []
    for h in HORIZONS:
        rh = rows[rows.h == h]
        train = rh[rh.y.notna()]
        test = rh[rh.origin == last]
        for name, cols in VARIANTS.items():
            q = {qq: fit_predict(train, test, cols, qq) for qq in QUANTILES}
            lo, md, hi = np.sort(np.vstack([q[0.1], q[0.5], q[0.9]]), axis=0)
            fut.append(pd.DataFrame({"model": name, "origin": test.origin.values, "region": test.region.values,
                                     "h": h, "target_week_start": test.target_week_start.values,
                                     "q10": lo, "q50": md, "q90": hi}))
    pd.concat(fut).to_csv("results/forecast_future.csv", index=False)
    print(f"Saved results/pred_classic.csv ({len(out)} rows) and results/forecast_future.csv")
    print("Next: python summarize_results.py")


if __name__ == "__main__":
    main()
