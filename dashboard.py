"""
Interactive dashboard: Uncertainty-aware dengue surveillance for Pakistan (NIH IDSR weekly bulletins).

Run after the whole pipeline (parse_bulletins -> build_dataset -> run_models -> run_gru -> summarize_results):
    python dashboard.py
then open http://127.0.0.1:8050 in your browser.

Tabs: 1 Data reliability | 2 Cases & reporting (raw / reliability-encoded / scenario) | 3 Forecast (raw vs calibrated
intervals) | 4 Model comparison | 5 About & limits
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from plotly.colors import qualitative
from plotly.subplots import make_subplots

D, R = Path("data"), Path("results")
ORDER = ["Punjab", "Sindh", "KP", "Balochistan", "AJK", "GB", "ICT"]
ALPHA = 0.2
MIN_CAL = 30


# ------------------------------------------------------------------------------------------- data
def read(path, **kw):
    return pd.read_csv(path, **kw) if Path(path).exists() else pd.DataFrame()


MT = read(D / "model_table.csv", parse_dates=["week_start"])
PRED = pd.concat(
    [
        read(f, parse_dates=["origin", "target_week_start"])
        for f in sorted(R.glob("pred_*.csv"))
        if "conformal" not in f.name and "calibrated" not in f.name
    ]
    or [pd.DataFrame()],
    ignore_index=True,
)
FUT = pd.concat(
    [
        read(f, parse_dates=["origin", "target_week_start"])
        for f in sorted(R.glob("forecast_future*.csv"))
    ]
    or [pd.DataFrame()],
    ignore_index=True,
)
M_OVERALL, M_HOR = read(R / "metrics_overall.csv"), read(R / "metrics_by_horizon.csv")
M_CAL, M_ABL = read(R / "metrics_calibration.csv"), read(R / "metrics_ablation.csv")


def cqr_qhat(model, h):
    """conformal widening (log1p scale): 80% quantile of past miss-scores of this model at this horizon"""
    if PRED.empty or "q10" not in PRED:
        return np.nan
    g = PRED[
        (PRED.model == model)
        & (PRED.h == h)
        & PRED.q10.notna()
        & ~PRED.is_future.astype(bool)
    ]
    if len(g) < MIN_CAL:
        return np.nan
    ly, lo, hi = np.log1p(g.y), np.log1p(g.q10), np.log1p(g.q90)
    return float(np.quantile(np.maximum(lo - ly, ly - hi), 1 - ALPHA))


def calibrate(f):
    f = f.copy()
    f["qhat"] = [cqr_qhat(m, h) for m, h in zip(f.model, f.h)]
    f["c10"] = np.expm1(np.log1p(f.q10) - f.qhat).clip(lower=0)
    f["c90"] = np.expm1(np.log1p(f.q90) + f.qhat)
    return f


# ---------------------------------------------------------------------------------------- figures
def stretches(mask, weeks):
    out, start, prev = [], None, None
    for m, w in zip(mask, weeks):
        if m and start is None:
            start = w
        if m:
            prev = w
        if not m and start is not None:
            out.append((start, prev + pd.Timedelta(days=7)))
            start = None
    if start is not None:
        out.append((start, prev + pd.Timedelta(days=7)))
    return out


def fig_availability():
    code = {"observed": 0, "NR": 1, "bulletin_missing": 2}
    z = (
        MT.assign(z=MT.status.map(code))
        .pivot(index="region", columns="week_start", values="z")
        .reindex(ORDER)
    )
    f = go.Figure(
        go.Heatmap(
            z=z.values,
            x=z.columns,
            y=z.index,
            zmin=-0.5,
            zmax=2.5,
            showscale=False,
            colorscale=[
                [0, "#2a9d8f"],
                [1 / 3, "#2a9d8f"],
                [1 / 3, "#e9c46a"],
                [2 / 3, "#e9c46a"],
                [2 / 3, "#e76f51"],
                [1, "#e76f51"],
            ],
            hovertemplate="%{y} | %{x|%Y-%m-%d}<extra></extra>",
        )
    )
    for name, col in (
        ("reported", "#2a9d8f"),
        ("NR (region did not report)", "#e9c46a"),
        ("no bulletin / not readable", "#e76f51"),
    ):
        f.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=col),
                name=name,
            )
        )
    f.update_layout(
        title="What do we actually observe? (region x week)",
        height=360,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=50),
    )
    return f


def fig_compliance():
    c = (
        MT[MT.region != "Punjab"]
        .pivot(index="region", columns="week_start", values="compliance")
        .reindex(ORDER[1:])
    )
    f = go.Figure(
        go.Heatmap(
            z=c.values,
            x=c.columns,
            y=c.index,
            zmin=0,
            zmax=1,
            colorscale="RdYlGn",
            colorbar=dict(title="received / expected", tickformat=".0%"),
            hovertemplate="%{y} | %{x|%Y-%m-%d} | %{z:.0%}<extra></extra>",
        )
    )
    f.update_layout(
        title="Reporting compliance: share of expected IDSR reports received (blank = not available)",
        height=330,
        margin=dict(t=50),
    )
    return f


REGION_COLOR = dict(zip(ORDER, qualitative.D3))


def fig_lab():
    lab = MT[MT.lab_format == "tests_pos"]
    f = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Dengue lab tests per week (log scale; weeks with 0 tests not drawn)",
            "Test positivity",
        ),
    )
    for reg in ORDER:
        g = lab[lab.region == reg].sort_values("week_start")
        if g.tests.notna().sum() == 0:
            continue
        col = REGION_COLOR[reg]
        f.add_trace(
            go.Scatter(
                x=g.week_start,
                y=g.tests.where(g.tests > 0),
                mode="lines+markers",
                name=reg,
                legendgroup=reg,
                marker=dict(size=4, color=col),
                line=dict(color=col),
            ),
            row=1,
            col=1,
        )
        f.add_trace(
            go.Scatter(
                x=g.week_start,
                y=g.test_positivity,
                mode="lines+markers",
                name=reg,
                legendgroup=reg,
                showlegend=False,
                marker=dict(size=4, color=col),
                line=dict(color=col),
            ),
            row=2,
            col=1,
        )
    f.update_yaxes(type="log", row=1, col=1)
    f.update_yaxes(tickformat=".0%", row=2, col=1)
    f.update_layout(
        height=520,
        margin=dict(t=60),
        title="Where is dengue testing actually done? (2024 onward)",
    )
    return f


def fig_cases(region, view):
    g = MT[MT.region == region].sort_values("week_start")
    f = make_subplots(specs=[[{"secondary_y": True}]])
    if view == "reliability":
        f.add_trace(
            go.Scatter(
                x=g.week_start,
                y=g.cases,
                mode="lines",
                line=dict(color="#999", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        f.add_trace(
            go.Scatter(
                x=g.week_start,
                y=g.cases,
                mode="markers",
                name="reported cases",
                marker=dict(
                    size=9,
                    color=g.compliance,
                    colorscale="RdYlGn",
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="compliance", tickformat=".0%", x=1.08),
                    line=dict(width=0.5, color="#555"),
                ),
                customdata=g.compliance.fillna(-1),
                hovertemplate="%{x|%Y-%m-%d} | %{y} cases | compliance %{customdata:.0%}<extra></extra>",
            )
        )
    else:
        f.add_trace(
            go.Scatter(
                x=g.week_start,
                y=g.cases,
                mode="lines+markers",
                name="reported suspected cases",
                line=dict(color="#264653"),
                marker=dict(size=4),
            )
        )
    if view == "scenario":
        adj = g.cases / g.compliance.where(g.compliance >= 0.2)
        f.add_trace(
            go.Scatter(
                x=g.week_start,
                y=adj,
                mode="lines",
                name="scenario: cases / compliance",
                line=dict(color="#e76f51", dash="dash"),
            )
        )
    if region != "Punjab":
        f.add_trace(
            go.Scatter(
                x=g.week_start,
                y=g.compliance,
                mode="lines",
                name="compliance (right axis)",
                line=dict(color="#2a9d8f", width=1),
            ),
            secondary_y=True,
        )
        f.update_yaxes(
            range=[0, 1.05],
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            tickformat=".0%",
            secondary_y=True,
        )
    for a, b in stretches((g.status == "NR").tolist(), g.week_start.tolist()):
        f.add_vrect(x0=a, x1=b, fillcolor="#e9c46a", opacity=0.35, line_width=0)
    for a, b in stretches(
        (g.status == "bulletin_missing").tolist(), g.week_start.tolist()
    ):
        f.add_vrect(x0=a, x1=b, fillcolor="#e76f51", opacity=0.3, line_width=0)
    f.update_layout(
        height=470,
        title=f"{region}: weekly suspected dengue cases (yellow = region did not report, red = no bulletin)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60),
    )
    return f


def fig_forecast(region, model, mode):
    g = MT[MT.region == region].sort_values("week_start").tail(30)
    f = go.Figure(
        go.Scatter(
            x=g.week_start,
            y=g.cases,
            mode="lines+markers",
            name="reported cases",
            line=dict(color="#264653"),
            marker=dict(size=4),
        )
    )
    fu = (
        FUT[(FUT.region == region) & (FUT.model == model)].sort_values("h")
        if not FUT.empty
        else pd.DataFrame()
    )
    if fu.empty:
        f.update_layout(
            title="No forecast file found for this selection (run run_models.py / run_gru.py)",
            height=450,
        )
        return f, pd.DataFrame()
    fu = calibrate(fu)
    x = [fu.origin.iloc[0]] + list(fu.target_week_start)
    last = g.dropna(subset=["cases"]).iloc[-1]

    def band(lo, hi, name, color):
        f.add_trace(
            go.Scatter(
                x=x[1:] + x[1:][::-1],
                y=list(hi) + list(lo)[::-1],
                fill="toself",
                name=name,
                mode="lines",
                fillcolor=color,
                line=dict(width=0),
                hoverinfo="skip",
            )
        )

    if mode in ("raw", "both"):
        band(fu.q10, fu.q90, "80% interval, raw model", "rgba(231,111,81,0.30)")
    if mode in ("cal", "both") and fu.c10.notna().all():
        band(
            fu.c10,
            fu.c90,
            "80% interval, calibrated (conformal)",
            "rgba(42,157,143,0.25)",
        )
    f.add_trace(
        go.Scatter(
            x=[last.week_start] + list(fu.target_week_start),
            y=[last.cases] * (len(fu) + 1),
            mode="lines",
            name="persistence baseline (last value)",
            line=dict(color="#888", dash="dot", width=1.5),
        )
    )
    f.add_trace(
        go.Scatter(
            x=[last.week_start] + list(fu.target_week_start),
            y=[last.cases] + list(fu.q50),
            mode="lines+markers",
            name="forecast (median)",
            line=dict(color="#e76f51", dash="dash"),
        )
    )
    f.update_layout(
        height=470,
        title=f"{region} | {model}: forecast for the next 4 weeks after the last bulletin "
        f"({pd.Timestamp(fu.origin.iloc[0]).date()})",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60),
        yaxis_title="suspected cases per week",
    )
    tab = pd.DataFrame(
        {
            "week starting": fu.target_week_start.dt.date,
            "horizon (wk)": fu.h,
            "median": fu.q50.round(0),
            "raw 80% interval": [f"{a:.0f} - {b:.0f}" for a, b in zip(fu.q10, fu.q90)],
            "calibrated 80% interval": [
                f"{a:.0f} - {b:.0f}" if pd.notna(a) else "n/a"
                for a, b in zip(fu.c10, fu.c90)
            ],
        }
    )
    return f, tab


def fig_leaderboard():
    if M_OVERALL.empty:
        return go.Figure().update_layout(
            title="metrics_overall.csv not found (run summarize_results.py)"
        )
    t = M_OVERALL.sort_values("MAE", ascending=False)
    col = [
        (
            "#264653"
            if m.startswith("gru")
            else "#2a9d8f" if m.startswith("lgbm") else "#aaa"
        )
        for m in t.model
    ]
    f = go.Figure(
        go.Bar(
            x=t.MAE,
            y=t.model,
            orientation="h",
            marker_color=col,
            hovertemplate="%{y}: MAE %{x:.1f}<extra></extra>",
        )
    )
    f.update_layout(
        title="Mean absolute error, 1-4 weeks ahead (grey = baselines, green = LightGBM, dark = GRU)",
        height=400,
        margin=dict(t=50),
        xaxis_title="MAE (cases per week)",
    )
    return f


def fig_horizon():
    if M_HOR.empty:
        return go.Figure()
    f = go.Figure()
    for _, r in M_HOR.iterrows():
        f.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4],
                y=[r["1"], r["2"], r["3"], r["4"]],
                mode="lines+markers",
                name=r["model"],
            )
        )
    f.update_layout(
        title="Error grows with the forecast horizon",
        height=400,
        margin=dict(t=50),
        xaxis=dict(title="weeks ahead", dtick=1),
        yaxis_title="MAE",
    )
    return f


def fig_calibration():
    if M_CAL.empty:
        return go.Figure().update_layout(title="metrics_calibration.csv not found")
    t = M_CAL.copy()
    t["base"] = t.model.str.replace("+cqr", "", regex=False)
    t["kind"] = np.where(
        t.model.str.endswith("+cqr"), "calibrated (conformal)", "raw model interval"
    )
    f = go.Figure()
    for k, c in (
        ("raw model interval", "#e76f51"),
        ("calibrated (conformal)", "#2a9d8f"),
    ):
        s = t[t.kind == k]
        f.add_trace(go.Bar(x=s.base, y=s.coverage80, name=k, marker_color=c))
    f.add_hline(y=0.8, line_dash="dot", annotation_text="nominal 80%")
    f.update_layout(
        title="How often does the 80% interval contain the truth?",
        barmode="group",
        height=400,
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(t=50),
    )
    return f


# -------------------------------------------------------------------------------------------- app
ABOUT = """
**Data.** NIH Pakistan IDSR weekly Public Health Bulletins (2023-W22 onward), parsed from the PDFs. All cases are *suspected*
cases reported by participating sites - not confirmed cases and not incidence. Weeks are ISO weeks (Monday-Sunday).

**Reporting gaps are part of the data.** Punjab appears as *NR* (not reported) in most weeks; other regions' compliance varies a lot,
and dengue laboratory testing is concentrated in a few regions. A low number can therefore mean "few cases" or "few reports".
Weeks without a readable bulletin are shown as gaps and are never filled in.

**Forecasts** (1-4 weeks ahead) exist for KP, Sindh and Balochistan only, where reporting is continuous. Models were evaluated with a
rolling-origin protocol (no future information). Raw model intervals were too narrow, so a conformal adjustment based only on past
forecast errors is offered as the calibrated interval.

**Scenario view** (tab 2) divides reported cases by compliance. It assumes that sites which did not report resemble sites which did,
which is unverified; treat it as a sensitivity illustration, not an estimate.

**Limitations.** Three seasons of data; one test period; models were not extensively tuned; no population denominators yet.
"""

app = Dash(__name__, title="Dengue surveillance - Pakistan")
server = app.server
regions_fc = (
    sorted(FUT.region.unique()) if not FUT.empty else ["KP", "Sindh", "Balochistan"]
)
models_fc = [
    m
    for m in (FUT.model.unique() if not FUT.empty else [])
    if m.startswith(("gru", "lgbm"))
]
best = (
    M_OVERALL[M_OVERALL.model.isin(models_fc)].sort_values("MAE").model.iloc[0]
    if not M_OVERALL.empty and models_fc
    else (models_fc[0] if models_fc else None)
)

abl_cols = [{"name": c, "id": c} for c in M_ABL.columns] if not M_ABL.empty else []
app.layout = html.Div(
    style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "auto"},
    children=[
        html.H2("Uncertainty-aware dengue surveillance dashboard - Pakistan"),
        html.P(
            "Source: NIH IDSR weekly bulletins. Suspected cases as reported; read together with reporting compliance."
        ),
        dcc.Tabs(
            children=[
                dcc.Tab(
                    label="1. Data reliability",
                    children=[
                        dcc.Graph(figure=fig_availability()),
                        dcc.Graph(figure=fig_compliance()),
                        dcc.Graph(figure=fig_lab()),
                    ],
                ),
                dcc.Tab(
                    label="2. Cases & reporting",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "30px", "padding": "10px"},
                            children=[
                                dcc.Dropdown(
                                    id="reg1",
                                    options=ORDER,
                                    value="KP",
                                    clearable=False,
                                    style={"width": "200px"},
                                ),
                                dcc.RadioItems(
                                    id="view1",
                                    value="raw",
                                    inline=True,
                                    options=[
                                        {"label": " raw counts", "value": "raw"},
                                        {
                                            "label": " reliability-encoded (colour = compliance)",
                                            "value": "reliability",
                                        },
                                        {
                                            "label": " scenario: adjusted for compliance",
                                            "value": "scenario",
                                        },
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="cases"),
                    ],
                ),
                dcc.Tab(
                    label="3. Forecast",
                    children=[
                        html.Div(
                            style={"display": "flex", "gap": "30px", "padding": "10px"},
                            children=[
                                dcc.Dropdown(
                                    id="reg2",
                                    options=regions_fc,
                                    value=regions_fc[0],
                                    clearable=False,
                                    style={"width": "180px"},
                                ),
                                dcc.Dropdown(
                                    id="mod2",
                                    options=models_fc,
                                    value=best,
                                    clearable=False,
                                    style={"width": "220px"},
                                ),
                                dcc.RadioItems(
                                    id="mode2",
                                    value="both",
                                    inline=True,
                                    options=[
                                        {"label": " raw interval", "value": "raw"},
                                        {"label": " calibrated", "value": "cal"},
                                        {"label": " both", "value": "both"},
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="fc"),
                        html.Div(id="fctab", style={"padding": "10px"}),
                    ],
                ),
                dcc.Tab(
                    label="4. Model comparison",
                    children=[
                        dcc.Graph(figure=fig_leaderboard()),
                        dcc.Graph(figure=fig_horizon()),
                        dcc.Graph(figure=fig_calibration()),
                        html.H4(
                            "Do reporting / climate features help? (moving-block bootstrap, negative = first model better)"
                        ),
                        dash_table.DataTable(
                            data=M_ABL.to_dict("records"),
                            columns=abl_cols,
                            page_size=15,
                            style_cell={
                                "fontSize": 12,
                                "textAlign": "left",
                                "whiteSpace": "normal",
                            },
                        ),
                    ],
                ),
                dcc.Tab(
                    label="5. About & limits",
                    children=[dcc.Markdown(ABOUT, style={"padding": "15px"})],
                ),
            ]
        ),
    ],
)


@app.callback(
    Output("cases", "figure"), Input("reg1", "value"), Input("view1", "value")
)
def _cases(region, view):
    return fig_cases(region, view)


@app.callback(
    Output("fc", "figure"),
    Output("fctab", "children"),
    Input("reg2", "value"),
    Input("mod2", "value"),
    Input("mode2", "value"),
)
def _fc(region, model, mode):
    f, tab = fig_forecast(region, model, mode)
    if tab.empty:
        return f, ""
    return f, dash_table.DataTable(
        data=tab.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tab.columns],
        style_cell={"fontSize": 13, "textAlign": "left"},
    )


if __name__ == "__main__":
    if MT.empty:
        raise SystemExit("data/model_table.csv not found - run build_dataset.py first")
    print("Open http://127.0.0.1:8050 in your browser (Ctrl+C to stop)")
    app.run(debug=False)
