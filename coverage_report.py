"""
Coverage figures for the paper / dashboard prototype (interactive HTML, no extra libraries needed).

Run AFTER build_dataset.py:
    python coverage_report.py
Output: data/coverage_report.html  - open in any browser
  1. Data availability map   (observed / NR / bulletin missing, per region and week)
  2. Reporting compliance heatmap
  3. Suspected dengue cases per region, with NR and missing-bulletin weeks shaded
  4. Dengue lab testing: tests per week and test positivity
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

D = Path("data")
ORDER = ["Punjab", "Sindh", "KP", "Balochistan", "AJK", "GB", "ICT"]
COLOR = dict(zip(ORDER, qualitative.D3))


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


def main():
    df = pd.read_csv(D / "model_table.csv", parse_dates=["week_start"])
    weeks = sorted(df.week_start.unique())
    figs = []

    # 1) availability map
    code = {"observed": 0, "NR": 1, "bulletin_missing": 2}
    z = (
        df.assign(z=df.status.map(code))
        .pivot(index="region", columns="week_start", values="z")
        .reindex(ORDER)
    )
    f1 = go.Figure(
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
        ("observed", "#2a9d8f"),
        ("NR (no report)", "#e9c46a"),
        ("no bulletin", "#e76f51"),
    ):
        f1.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=col),
                name=name,
            )
        )
    f1.update_layout(
        title="1. What do we actually see? Data availability by region and week",
        height=380,
        legend=dict(orientation="h", y=-0.15),
    )
    figs.append(f1)

    # 2) compliance
    c = (
        df[df.region != "Punjab"]
        .pivot(index="region", columns="week_start", values="compliance")
        .reindex(ORDER[1:])
    )
    f2 = go.Figure(
        go.Heatmap(
            z=c.values,
            x=c.columns,
            y=c.index,
            zmin=0,
            zmax=1,
            colorscale="RdYlGn",
            colorbar=dict(title="received / expected"),
            hovertemplate="%{y} | %{x|%Y-%m-%d} | %{z:.0%}<extra></extra>",
        )
    )
    f2.update_layout(
        title="2. Reporting compliance (received / expected IDSR reports); blank = not available",
        height=340,
    )
    figs.append(f2)

    # 3) cases with shaded gaps
    f3 = make_subplots(
        rows=len(ORDER),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=ORDER,
    )
    for i, reg in enumerate(ORDER, 1):
        g = df[df.region == reg].sort_values("week_start")
        f3.add_trace(
            go.Scatter(
                x=g.week_start,
                y=g.cases,
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(width=1.3, color="#264653"),
                showlegend=False,
                hovertemplate=reg + " | %{x|%Y-%m-%d} | %{y} cases<extra></extra>",
            ),
            row=i,
            col=1,
        )
        for a, b in stretches((g.status == "NR").tolist(), g.week_start.tolist()):
            f3.add_vrect(
                x0=a,
                x1=b,
                fillcolor="#e9c46a",
                opacity=0.35,
                line_width=0,
                row=i,
                col=1,
            )
        for a, b in stretches(
            (g.status == "bulletin_missing").tolist(), g.week_start.tolist()
        ):
            f3.add_vrect(
                x0=a,
                x1=b,
                fillcolor="#e76f51",
                opacity=0.30,
                line_width=0,
                row=i,
                col=1,
            )
    f3.update_layout(
        title="3. Weekly suspected dengue cases (yellow = NR, red = no bulletin)",
        height=170 * len(ORDER),
    )
    figs.append(f3)

    # 4) lab testing
    lab = df[df.lab_format == "tests_pos"]
    f4 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Dengue tests per week (log scale; weeks with 0 tests not drawn)",
            "Test positivity",
        ),
    )
    for reg in ORDER:
        g = lab[lab.region == reg].sort_values("week_start")
        if g.tests.notna().sum() == 0:
            continue
        col = COLOR[reg]
        f4.add_trace(
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
        f4.add_trace(
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
    f4.update_yaxes(type="log", row=1, col=1)
    f4.update_yaxes(tickformat=".0%", row=2, col=1)
    f4.update_layout(
        title="4. Dengue lab testing (2024 onward: tests and positives are both reported)",
        height=620,
    )
    figs.append(f4)

    html = [
        "<html><head><meta charset='utf-8'><title>IDSR dengue coverage</title></head><body style='font-family:sans-serif'>"
    ]
    for k, f in enumerate(figs):
        html.append(
            f.to_html(full_html=False, include_plotlyjs=True if k == 0 else False)
        )
    html.append("</body></html>")
    out = D / "coverage_report.html"
    out.write_text("\n".join(html), encoding="utf-8")
    print(f"Saved {out} - open it in your browser")


if __name__ == "__main__":
    main()
