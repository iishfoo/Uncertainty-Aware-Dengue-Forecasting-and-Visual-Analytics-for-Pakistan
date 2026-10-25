"""
Compare all models on the SAME forecasts (common support), test the ablations, and check interval calibration.

Run after run_models.py (and run_gru.py):
    python summarize_results.py
Reads results/pred_*.csv  ->  prints tables and saves results/metrics_*.csv
"""

import glob

import numpy as np
import pandas as pd

ALPHA = 0.2  # 80% interval (10% - 90%)
KEY = ["origin", "region", "h"]
BLOCK = 8  # weeks per block in the moving-block bootstrap (forecast errors are autocorrelated)
MIN_CAL = 30  # past forecasts needed before an interval is conformally adjusted


def wis(y, lo, md, hi):
    """Weighted interval score with one interval (K=1, alpha=0.2)."""
    pen = (2 / ALPHA) * (lo - y).clip(lower=0) + (2 / ALPHA) * (y - hi).clip(lower=0)
    return (0.5 * (y - md).abs() + (ALPHA / 2) * ((hi - lo) + pen)) / 1.5


def boot_diff(a, b, origins, n=3000, seed=0):
    """moving-block bootstrap (blocks of consecutive forecast origins) of mean(a) - mean(b)"""
    rng = np.random.default_rng(seed)
    d = (
        pd.DataFrame({"d": (a - b).values, "o": origins.values})
        .groupby("o")
        .d.agg(["sum", "count"])
        .sort_index()
    )
    s, c, k = d["sum"].values, d["count"].values, len(d)
    bl = min(BLOCK, k)
    nb = int(np.ceil(k / bl))
    starts = rng.integers(0, k - bl + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(bl)).reshape(n, -1)
    m = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def conformal_adjust(df):
    """
    Conformalized quantile regression, online: for every forecast, the interval is widened (on the log1p scale)
    by the 80% quantile of the miss-scores of the SAME model and horizon whose targets were already known
    at the forecast origin. Uses no future information.
    """
    out = []
    for (model, h), g in df[df.q10.notna()].groupby(["model", "h"]):
        g = g.sort_values("origin").copy()
        ly, lo, hi = np.log1p(g.y), np.log1p(g.q10), np.log1p(g.q90)
        g["score"] = np.maximum(lo - ly, ly - hi)
        adj = []
        for _, r in g.iterrows():
            past = g.score[g.target_week_start <= r.origin]
            adj.append(np.quantile(past, 1 - ALPHA) if len(past) >= MIN_CAL else np.nan)
        g["qhat"] = adj
        g = g[g.qhat.notna()].copy()
        g["q10"] = np.expm1(np.log1p(g.q10) - g.qhat).clip(lower=0)
        g["q90"] = np.expm1(np.log1p(g.q90) + g.qhat)
        g["model"] = model + "+cqr"
        out.append(g.drop(columns=["score", "qhat"]))
    return pd.concat(out) if out else pd.DataFrame()


def score_table(df):
    df = df.copy()
    df["ae"], df["se"] = (df.y - df.q50).abs(), (df.y - df.q50) ** 2
    df["lae"] = (np.log1p(df.y) - np.log1p(df.q50)).abs()
    pi = df.q10.notna()
    df.loc[pi, "wis"] = wis(df.y[pi], df.q10[pi], df.q50[pi], df.q90[pi])
    df.loc[pi, "covered"] = ((df.y >= df.q10) & (df.y <= df.q90))[pi].astype(float)
    df.loc[pi, "width"] = (df.q90 - df.q10)[pi]
    return df


def main():
    files = [
        f
        for f in glob.glob("results/pred_*.csv")
        if "conformal" not in f and "calibrated" not in f
    ]
    df = pd.concat(
        [pd.read_csv(f, parse_dates=["origin", "target_week_start"]) for f in files],
        ignore_index=True,
    )
    df = df[~df.is_future.astype(bool)]
    keep = df.groupby(KEY).model.nunique()
    df = df.merge(
        keep[keep == df.model.nunique()].rename("n").reset_index()[KEY], on=KEY
    )  # common support
    sc = score_table(df)
    print(
        f"{df.model.nunique()} models, {df[KEY].drop_duplicates().shape[0]} forecasts "
        f"(origin x region x horizon) with all models present; origins {df.origin.min().date()} -> {df.origin.max().date()}\n"
    )

    tab = sc.groupby("model").agg(
        MAE=("ae", "mean"),
        RMSE=("se", lambda s: np.sqrt(s.mean())),
        logMAE=("lae", "mean"),
        WIS=("wis", "mean"),
        coverage80=("covered", "mean"),
        width=("width", "mean"),
    )
    tab["MAE_vs_naive_last"] = tab.MAE / tab.loc["naive_last", "MAE"]
    tab["MAE_vs_seasonal_naive3"] = tab.MAE / tab.loc["seasonal_naive3", "MAE"]
    tab = tab.sort_values("MAE").round(3)
    print(
        "== Overall (lower is better; coverage80 should be near 0.80; ratios < 1 beat that baseline)\n",
        tab.to_string(),
        "\n",
    )
    tab.to_csv("results/metrics_overall.csv")

    for col, name in (("h", "horizon"), ("region", "region")):
        t = sc.pivot_table(
            index="model", columns=col, values="ae", aggfunc="mean"
        ).round(2)
        print(f"== MAE by {name}\n", t.loc[tab.index].to_string(), "\n")
        t.to_csv(f"results/metrics_by_{name}.csv")
    rel = sc.pivot_table(index="model", columns="region", values="ae", aggfunc="mean")
    rel = (rel / rel.loc["naive_last"]).round(2)
    print(
        "== MAE relative to naive_last, by region (so big regions do not dominate; < 1 is better than persistence)\n",
        rel.loc[tab.index].to_string(),
        "\n",
    )

    print(
        f"== Pairwise tests (moving-block bootstrap, blocks of {BLOCK} origins; negative = first model better)"
    )
    pairs = [
        ("gru_noreport", "naive_last", "GRU vs persistence"),
        ("gru_noreport", "lgbm_base", "GRU vs LightGBM"),
        (
            "gru_noreport",
            "naive_last",
            "GRU vs persistence, horizons 1-2 only (exploratory)",
            [1, 2],
        ),
        (
            "gru_noreport",
            "naive_last",
            "GRU vs persistence, horizons 3-4 only (exploratory)",
            [3, 4],
        ),
        ("lgbm_climate", "lgbm_base", "LightGBM: adding CLIMATE"),
        ("lgbm_reporting", "lgbm_base", "LightGBM: adding REPORTING"),
        ("lgbm_full", "lgbm_climate", "LightGBM: adding REPORTING on top of climate"),
        ("gru_report", "gru_noreport", "GRU: adding REPORTING"),
    ]
    piv_ae = sc.pivot_table(index=KEY, columns="model", values="ae")
    piv_wis = sc.pivot_table(index=KEY, columns="model", values="wis")
    rows = []
    for item in pairs:
        a, b, label = item[:3]
        hs = item[3] if len(item) > 3 else None
        if a not in piv_ae or b not in piv_ae:
            continue
        for metric, piv in (("abs error", piv_ae), ("WIS", piv_wis)):
            if hs is not None:
                piv = piv[piv.index.get_level_values("h").isin(hs)]
            if (
                a not in piv
                or b not in piv
                or piv[a].isna().all()
                or piv[b].isna().all()
            ):
                continue  # e.g. baselines have no interval, so no WIS
            m, lo, hi = boot_diff(
                piv[a],
                piv[b],
                pd.Series(piv.index.get_level_values("origin"), index=piv.index),
            )
            rows.append(
                {
                    "comparison": f"{label} ({a} vs {b})",
                    "metric": metric,
                    "mean_diff": round(m, 2),
                    "ci95_low": round(lo, 2),
                    "ci95_high": round(hi, 2),
                    "clear_difference": not (lo <= 0 <= hi),
                }
            )
    ab = pd.DataFrame(rows)
    print(ab.to_string(index=False), "\n")
    ab.to_csv("results/metrics_ablation.csv", index=False)

    cq = conformal_adjust(df)
    if len(cq):
        both = pd.concat([df[df.q10.notna()], cq], ignore_index=True)
        ok = cq[KEY + ["model"]].assign(
            base=lambda d: d.model.str.replace("+cqr", "", regex=False)
        )
        sub = df.merge(
            ok[KEY + ["base"]].drop_duplicates().rename(columns={"base": "model"}),
            on=KEY + ["model"],
        )
        t = (
            score_table(pd.concat([sub, cq], ignore_index=True))
            .groupby("model")
            .agg(
                n=("ae", "size"),
                coverage80=("covered", "mean"),
                WIS=("wis", "mean"),
                width=("width", "mean"),
            )
            .round(3)
        )
        print(
            "== Interval calibration: raw vs conformally adjusted (same forecasts; nominal coverage 0.80)\n",
            t.to_string(),
            "\n",
        )
        t.to_csv("results/metrics_calibration.csv")
        cq.to_csv("results/calibrated_predictions.csv", index=False)
    print("Saved results/metrics_*.csv")


if __name__ == "__main__":
    main()
