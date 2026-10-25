"""
Deep-learning part: GRU quantile forecaster (one global model for all regions), evaluated with exactly the
same rolling-origin protocol as run_models.py, so summarize_results.py can compare everything.

Run after run_models.py:
    python run_gru.py
    python run_gru.py --regions KP Sindh Balochistan --epochs 150 --seeds 3 --refit-every 8
Output: results/pred_gru.csv (+ results/forecast_future_gru.csv)

Model: GRU over the last 16 weeks -> 4 horizons x 3 quantiles (10/50/90) with pinball loss, on standardised
log1p(cases). Two variants for the ablation: gru_noreport (cases + climate + season) and
gru_report (+ reporting compliance and lab-test volume). Missing weeks are masked, never filled with data.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_models import HORIZONS, QUANTILES, load_table

warnings.filterwarnings("ignore")
L = 16                                     # input window (weeks)
QS = torch.tensor(QUANTILES, dtype=torch.float32)
H = len(HORIZONS)


def prepare(df, regs):
    weeks = sorted(df.week_start.unique())
    tw = pd.Series(weeks).dt.isocalendar().week.astype(int).values
    arr = {}
    for r in regs:
        g = df[df.region == r].set_index("week_start").reindex(weeks)
        arr[r] = pd.DataFrame({"lc": np.log1p(g.cases.values.astype(float)), "t2m": g.t2m.values, "rh2m": g.rh2m.values,
                               "rain_mm": g.rain_mm.values, "comp": g.compliance.values,
                               "ltests": np.log1p(g.tests.values.astype(float)),
                               "sin_w": np.sin(2 * np.pi * tw / 52.18), "cos_w": np.cos(2 * np.pi * tw / 52.18)})
    return weeks, arr


def build_inputs(arr, regs, variant, te):
    """standardise with statistics from weeks <= te only (no leakage); masks for the channels that can be missing"""
    cols = ["lc", "t2m", "rh2m", "rain_mm"] + (["comp", "ltests"] if variant == "gru_report" else [])
    X, stats = {}, {}
    for r in regs:
        d, parts = arr[r], []
        for c in cols:
            v = d[c].values.astype(float)
            mu, sd = np.nanmean(v[: te + 1]), np.nanstd(v[: te + 1])
            mu, sd = (0.0 if np.isnan(mu) else mu), (1.0 if (np.isnan(sd) or sd < 1e-6) else sd)
            parts.append(np.nan_to_num((v - mu) / sd))
            if c in ("lc", "comp", "ltests"):
                parts.append((~np.isnan(v)).astype(float))
            if c == "lc":
                stats[r] = (mu, sd)
        parts += [d["sin_w"].values, d["cos_w"].values]
        X[r] = np.stack(parts, 1).astype(np.float32)
    return X, stats


def samples(X, arr, stats, regs, origins_idx, te, with_targets=True):
    xs, rid, ys, ms = [], [], [], []
    for ri, r in enumerate(regs):
        lc = arr[r]["lc"].values.astype(float)
        mu, sd = stats[r]
        for i in origins_idx:
            xs.append(X[r][i - L + 1: i + 1])
            rid.append(ri)
            y, m = np.zeros(H, np.float32), np.zeros(H, np.float32)
            for k, h in enumerate(HORIZONS):
                j = i + h
                if with_targets and j <= te and j < len(lc) and not np.isnan(lc[j]):
                    y[k], m[k] = (lc[j] - mu) / sd, 1.0
            ys.append(y)
            ms.append(m)
    t = lambda a, dt=torch.float32: torch.tensor(np.array(a), dtype=dt)
    return t(xs), t(rid, torch.long), t(ys), t(ms)


class QGRU(nn.Module):
    def __init__(self, f_in, n_reg, hidden=32, emb=4, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_reg, emb)
        self.gru = nn.GRU(f_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden + emb, 32), nn.ReLU(), nn.Linear(32, H * len(QUANTILES)))

    def forward(self, x, r):
        _, h = self.gru(x)
        return self.head(torch.cat([h[-1], self.emb(r)], 1)).view(-1, H, len(QUANTILES))


def pinball(pred, y, m):
    e = y.unsqueeze(-1) - pred
    loss = torch.maximum(QS * e, (QS - 1) * e) * m.unsqueeze(-1)
    return loss.sum() / (m.sum() * len(QUANTILES) + 1e-8)


def train(Xs, R, Y, M, n_reg, epochs, seed, lr=0.01):
    torch.manual_seed(seed)
    model = QGRU(Xs.shape[2], n_reg)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = pinball(model(Xs, R), Y, M)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


@torch.no_grad()
def predict(models, Xs, R):
    return torch.stack([m(Xs, R) for m in models]).mean(0).numpy()      # (B, H, Q) standardised


def to_cases(out, stats, regs, rid):
    res = np.empty_like(out)
    for b, ri in enumerate(rid):
        mu, sd = stats[regs[ri]]
        res[b] = out[b] * sd + mu
    res = np.sort(res, axis=2)
    return np.clip(np.expm1(res), 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/model_table.csv")
    ap.add_argument("--regions", nargs="+", default=["KP", "Sindh", "Balochistan"])
    ap.add_argument("--test-start", default="2025-05-12")
    ap.add_argument("--refit-every", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=["gru_noreport", "gru_report"])
    ap.add_argument("--min-train", type=int, default=60)
    a = ap.parse_args()
    Path("results").mkdir(exist_ok=True)
    torch.set_num_threads(2)

    df = load_table(a.table, a.regions)
    regs = sorted(df.region.unique())
    weeks, arr = prepare(df, regs)
    T = len(weeks)
    cases = {r: df[df.region == r].set_index("week_start").reindex(weeks)["cases"].values for r in regs}
    origins = [i for i, w in enumerate(weeks) if pd.Timestamp(a.test_start) <= w < weeks[-1] and i >= L - 1]
    refits = origins[:: a.refit_every]
    print(f"GRU | regions {regs} | {len(origins)} origins | {len(refits)} refits x {len(a.variants)} variants x {a.seeds} seeds")

    rows = []
    for n, te in enumerate(refits):
        nxt = refits[n + 1] if n + 1 < len(refits) else T - 1
        for variant in a.variants:
            X, stats = build_inputs(arr, regs, variant, te)
            Xs, R, Y, M = samples(X, arr, stats, regs, range(L - 1, te), te)
            if (M.sum(1) > 0).sum() < a.min_train:
                continue
            models = [train(Xs, R, Y, M, len(regs), a.epochs, s) for s in range(a.seeds)]
            idx = [i for i in origins if te <= i < nxt]
            Xt, Rt, _, _ = samples(X, arr, stats, regs, idx, te, with_targets=False)
            q = to_cases(predict(models, Xt, Rt), stats, regs, Rt.numpy())
            b = 0
            for r in regs:
                for i in idx:
                    for k, h in enumerate(HORIZONS):
                        j = i + h
                        if j < T and not np.isnan(cases[r][j]):
                            rows.append({"model": variant, "origin": weeks[i], "region": r, "h": h,
                                         "target_week_start": weeks[j], "y": cases[r][j], "q10": q[b, k, 0],
                                         "q50": q[b, k, 1], "q90": q[b, k, 2], "is_future": False})
                    b += 1
        print(f"  refit {n + 1}/{len(refits)} done (data up to {pd.Timestamp(weeks[te]).date()})", end="\r")
    print()
    pd.DataFrame(rows).to_csv("results/pred_gru.csv", index=False)

    fut = []                                                   # genuine forecast after the last bulletin
    for variant in a.variants:
        X, stats = build_inputs(arr, regs, variant, T - 1)
        Xs, R, Y, M = samples(X, arr, stats, regs, range(L - 1, T - 1), T - 1)
        models = [train(Xs, R, Y, M, len(regs), a.epochs, s) for s in range(a.seeds)]
        Xt, Rt, _, _ = samples(X, arr, stats, regs, [T - 1], T - 1, with_targets=False)
        q = to_cases(predict(models, Xt, Rt), stats, regs, Rt.numpy())
        for b, r in enumerate(regs):
            for k, h in enumerate(HORIZONS):
                fut.append({"model": variant, "origin": weeks[-1], "region": r, "h": h,
                            "target_week_start": pd.Timestamp(weeks[-1]) + pd.Timedelta(days=7 * h),
                            "q10": q[b, k, 0], "q50": q[b, k, 1], "q90": q[b, k, 2]})
    pd.DataFrame(fut).to_csv("results/forecast_future_gru.csv", index=False)
    print(f"Saved results/pred_gru.csv ({len(rows)} rows) and results/forecast_future_gru.csv")
    print("Next: python summarize_results.py")


if __name__ == "__main__":
    main()
