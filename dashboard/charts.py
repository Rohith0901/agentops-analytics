"""dashboard/charts.py — plotly figure builders.

Color theme follows a validated categorical palette (fixed hue order, never
cycled — see the project's dataviz guidance): blue, orange, aqua, yellow,
magenta, green, violet, red. A single sequential (blue) ramp is used for
magnitude/heatmap encodings. Figures are built to read reasonably in both
light and dark Streamlit themes (transparent paper/plot background, and
`template=None` so Streamlit's own theme fills in text/gridline colors).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SEQUENTIAL = "Blues"
DIVERGING = "RdBu"

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(size=13),
)


def _style(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(**_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=15)))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    return fig


def mrr_revenue_cost_trend(df: pd.DataFrame) -> go.Figure:
    """MRR, revenue, and LLM cost over time (three lines, one axis: dollars)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=df["mrr"], name="MRR",
                              line=dict(color=CATEGORICAL[0], width=2)))
    fig.add_trace(go.Scatter(x=df["month"], y=df["revenue"], name="Revenue",
                              line=dict(color=CATEGORICAL[1], width=2)))
    fig.add_trace(go.Scatter(x=df["month"], y=df["llm_cost"], name="LLM cost",
                              line=dict(color=CATEGORICAL[7], width=2, dash="dot")))
    return _style(fig, "MRR, revenue & LLM cost by month")


def success_rate_trend(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["month"], y=df["success_rate"], name="Success rate",
                              line=dict(color=CATEGORICAL[2], width=2), fill="tozeroy",
                              fillcolor="rgba(27,175,122,0.15)"))
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, "Run success rate by month")


def cohort_retention_heatmap(df: pd.DataFrame) -> go.Figure:
    """Cohort x months-since-signup retention heatmap (sequential blue)."""
    piv = df.pivot_table(index="cohort_month", columns="months_since_signup", values="retention")
    piv = piv.sort_index()
    fig = go.Figure(
        data=go.Heatmap(
            z=piv.values,
            x=[str(c) for c in piv.columns],
            y=[d.strftime("%Y-%m") for d in piv.index],
            colorscale=SEQUENTIAL,
            zmin=0, zmax=1,
            colorbar=dict(title="Retained", tickformat=".0%"),
        )
    )
    fig.update_xaxes(title="Months since signup")
    fig.update_yaxes(title="Signup cohort", autorange="reversed")
    return _style(fig, "Logo retention by signup cohort")


def activation_vs_outcome(df: pd.DataFrame) -> go.Figure:
    """Bar: activated vs not-activated -> conversion rate & retention rate."""
    grp = df.groupby("activated").agg(
        conversion_rate=("self_serve_converted", "mean"),
        retention_rate=("retained", "mean"),
        n=("customer_id", "nunique"),
    ).reset_index()
    grp["activated"] = grp["activated"].map({True: "Activated", False: "Not activated"})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp["activated"], y=grp["conversion_rate"], name="Conversion rate",
                          marker_color=CATEGORICAL[0]))
    fig.add_trace(go.Bar(x=grp["activated"], y=grp["retention_rate"], name="Retention rate",
                          marker_color=CATEGORICAL[1]))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group")
    return _style(fig, "Activation vs. conversion & retention")


def channel_comparison(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["acquisition_channel"], y=df["conversion_rate"],
                          name="Self-serve conversion", marker_color=CATEGORICAL[0]))
    fig.add_trace(go.Bar(x=df["acquisition_channel"], y=df["retention_rate"],
                          name="Retention", marker_color=CATEGORICAL[2]))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group")
    return _style(fig, "Conversion & retention by acquisition channel")


def model_task_heatmap(df: pd.DataFrame) -> go.Figure:
    piv = df.pivot_table(index="model", columns="task_type", values="success_rate")
    fig = go.Figure(
        data=go.Heatmap(
            z=piv.values, x=list(piv.columns), y=list(piv.index),
            colorscale=SEQUENTIAL, zmin=0, zmax=1,
            colorbar=dict(title="Success", tickformat=".0%"),
        )
    )
    return _style(fig, "Success rate by model x task type")


def cost_success_frontier(df: pd.DataFrame) -> go.Figure:
    """Scatter: avg cost vs success rate per (model, task_type), colored by model."""
    fig = px.scatter(
        df, x="avg_cost", y="success_rate", color="model", size="n_runs",
        hover_data=["task_type"], color_discrete_sequence=CATEGORICAL,
    )
    fig.update_yaxes(tickformat=".0%", title="Success rate")
    fig.update_xaxes(title="Avg LLM cost per run ($)")
    return _style(fig, "Cost-quality frontier (by model x task)")


def cost_per_success_bar(df: pd.DataFrame) -> go.Figure:
    df = df.sort_values("cost_per_success")
    fig = go.Figure(go.Bar(x=df["model"], y=df["cost_per_success"], marker_color=CATEGORICAL[:len(df)]))
    fig.update_yaxes(title="$ per successful run")
    return _style(fig, "Cost per successful run, by model")


def margin_by_plan_bar(df: pd.DataFrame) -> go.Figure:
    df = df.sort_values("gross_margin")
    colors = [CATEGORICAL[7] if m < 0 else CATEGORICAL[0] for m in df["gross_margin"]]
    fig = go.Figure(go.Bar(x=df["plan"], y=df["gross_margin"], marker_color=colors))
    fig.update_yaxes(tickformat=".0%", title="Gross margin")
    return _style(fig, "Gross margin by plan")


def cost_vs_revenue_per_credit(df: pd.DataFrame, team_ref: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["task_type"], y=df["cost_per_credit"], name="Cost per credit",
                          marker_color=CATEGORICAL[7]))
    if team_ref is not None:
        fig.add_hline(y=team_ref, line_dash="dash", line_color=CATEGORICAL[0],
                       annotation_text="Team revenue/credit", annotation_position="top left")
    fig.update_yaxes(title="$ per credit")
    return _style(fig, "Cost per credit by task (vs. Team revenue/credit)")


def margin_decile_bar(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("decile")["gross_margin"].mean().reset_index()
    colors = [CATEGORICAL[7] if m < 0 else CATEGORICAL[0] for m in grp["gross_margin"]]
    fig = go.Figure(go.Bar(x=grp["decile"], y=grp["gross_margin"], marker_color=colors))
    fig.update_xaxes(title="Customer margin decile (1 = lowest)", dtick=1)
    fig.update_yaxes(tickformat=".0%", title="Avg gross margin")
    return _style(fig, "Customer gross margin by decile")


def experiment_conversion_bar(df: pd.DataFrame, cis: dict) -> go.Figure:
    """Conversion rate by variant with Wilson CI error bars."""
    df = df.copy()
    df["rate"] = df["conversions"] / df["n"]
    lo = [cis[v][0] for v in df["variant"]]
    hi = [cis[v][1] for v in df["variant"]]
    err_plus = [h - r for h, r in zip(hi, df["rate"])]
    err_minus = [r - l for l, r in zip(lo, df["rate"])]
    fig = go.Figure(go.Bar(
        x=df["variant"], y=df["rate"], marker_color=CATEGORICAL[:len(df)],
        error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
    ))
    fig.update_yaxes(tickformat=".1%", title="Conversion rate (95% Wilson CI)")
    return _style(fig, "Pricing experiment: conversion by variant")


def routing_savings_gauge(current_cost: float, projected_cost: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=projected_cost,
        delta={"reference": current_cost, "relative": False, "valueformat": "$,.0f",
               "decreasing": {"color": CATEGORICAL[2]}, "increasing": {"color": CATEGORICAL[7]}},
        number={"prefix": "$", "valueformat": ",.0f"},
        title={"text": "Projected monthly LLM cost after routing"},
    ))
    return _style(fig)
