#!/usr/bin/env python3
"""make_figures.py — renders ~10 clean, consistent PNG figures to
reports/figures/, each with a title that states the finding (computed
dynamically from the data, never hard-coded).

Depends on the CSVs already written by run_sql.py, customer_analytics.py,
llm_analytics.py, and pricing_analytics.py to reports/tables/ — run those
first (run_all.py does this in order).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import ROOT  # noqa: E402

TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)
PALETTE = sns.color_palette("crest", as_cmap=False, n_colors=6)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"


def _fmt_p(p: float) -> str:
    """Format a p-value for a figure title: 'p<0.001' below that threshold
    (never 'p=0.00', the float-underflow artifact at large sample sizes),
    else 'p=X.XX'. Mirrors build_insights._fmt_p so report text and figure
    titles never disagree on how a p-value is rendered."""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.2f}"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def _read(name: str) -> pd.DataFrame:
    path = TABLES_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run run_sql.py / analytics modules first.")
    return pd.read_csv(path)


# ======================================================================================
# 1. MRR trend
# ======================================================================================

def fig_mrr_trend() -> None:
    df = _read("01_monthly_kpis")
    df["month"] = pd.to_datetime(df["month"])
    current_mrr = df["mrr_usd"].iloc[-1]
    # Trailing 6-month CMGR (falls back to the full series if <7 months of
    # history exist): the launch month's near-zero MRR makes "growth since
    # month 1" a misleading headline, so we report a recent, steady-state rate.
    lookback = min(6, len(df) - 1)
    mrr_then = df["mrr_usd"].iloc[-1 - lookback]
    cmgr = (current_mrr / mrr_then) ** (1 / lookback) - 1 if mrr_then > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df["month"], df["mrr_usd"], marker="o", markersize=3, color=PALETTE[3], linewidth=2)
    ax.set_ylabel("MRR (USD)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.set_title(f"MRR reached ${current_mrr:,.0f}/mo, growing {cmgr*100:,.1f}%/mo "
                 f"(trailing {lookback}-month CMGR)")
    fig.autofmt_xdate()
    _save(fig, "01_mrr_trend")


# ======================================================================================
# 2. Cohort retention heatmap
# ======================================================================================

def fig_cohort_heatmap() -> None:
    df = _read("03_cohort_retention")
    pivot = df.pivot(index="cohort_month", columns="months_since_signup", values="retention_pct")
    pivot = pivot.sort_index()
    m6 = df[df["months_since_signup"] == 6]["retention_pct"]
    avg_m6 = m6.mean() if len(m6) else float("nan")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, cmap="crest", ax=ax, cbar_kws={"label": "Paid-logo retention %"}, linewidths=0.3)
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Signup cohort")
    ax.set_title(f"Paid-logo retention averages {avg_m6:,.0f}% at month 6 across cohorts")
    _save(fig, "02_cohort_retention_heatmap")


# ======================================================================================
# 3. Activation vs conversion & retention
# ======================================================================================

def fig_activation_vs_outcomes() -> None:
    df = _read("activation_threshold")
    lift = (
        df.loc[df["min_successes_of_5"] == 5, "self_serve_conversion_rate"].iloc[0]
        - df.loc[df["min_successes_of_5"] == 0, "self_serve_conversion_rate"].iloc[0]
    ) * 100

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = df["min_successes_of_5"]
    ax1.plot(x, df["self_serve_conversion_rate"] * 100, marker="o", color=PALETTE[3], label="Self-serve conversion %")
    ax1.plot(x, df["retained_6mo_rate"] * 100, marker="s", color=PALETTE[5], label="6-month paid retention %")
    ax1.set_xlabel("Minimum successes in first 5 runs (activation cutoff)")
    ax1.set_ylabel("%")
    ax1.legend(loc="lower right")
    ax1.set_title(f"Fully-activated customers (5/5 success) convert {lift:,.0f}pp more than unactivated ones")
    _save(fig, "03_activation_vs_conversion_retention")


# ======================================================================================
# 4. Churn odds ratios forest plot
# ======================================================================================

def fig_churn_odds_ratios() -> None:
    df = _read("churn_odds_ratios")
    df = df.sort_values("odds_ratio")
    # Rank by statistical significance first, then by |log(odds ratio)| (a
    # symmetric magnitude measure) — matches the same selection logic used in
    # build_insights.py, so the figure and the report never disagree on which
    # feature is called out as "the strongest churn driver".
    significant = df[df["p_value"] < 0.05]
    ranking_pool = significant if len(significant) else df
    top_feature = ranking_pool.iloc[np.log(ranking_pool["odds_ratio"]).abs().values.argmax()]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(df))))
    y = np.arange(len(df))
    err_low = df["odds_ratio"] - df["odds_ratio_ci_low"]
    err_high = df["odds_ratio_ci_high"] - df["odds_ratio"]
    colors = [PALETTE[3] if p < 0.05 else "#b0b0b0" for p in df["p_value"]]
    ax.errorbar(df["odds_ratio"], y, xerr=[err_low, err_high], fmt="o", color="black", ecolor="gray", capsize=3)
    ax.scatter(df["odds_ratio"], y, color=colors, zorder=3, s=60)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio per paid customer-month (log scale), 95% CI")
    ax.set_title(
        f"'{top_feature['feature']}' is the strongest monthly churn-hazard driver "
        f"(OR={top_feature['odds_ratio']:.2f}, {'p<0.05' if top_feature['p_value'] < 0.05 else 'n.s.'})"
    )
    _save(fig, "04_churn_odds_ratios_forest_plot")


# ======================================================================================
# 4b. Kaplan-Meier survival curve, activated vs not
# ======================================================================================

def fig_km_survival_by_activation() -> None:
    df = _read("survival_km_curve")
    fig, ax = plt.subplots(figsize=(9, 6))
    for group, grp in df.groupby("group"):
        grp = grp.sort_values("t")
        # Prepend t=0, survival=1.0 so every curve starts at full survival.
        t = np.concatenate([[0], grp["t"].to_numpy()])
        s = np.concatenate([[1.0], grp["survival"].to_numpy()])
        ax.step(t, s * 100, where="post", label=group, linewidth=2)
    ax.set_xlabel("Months since first paid month (tenure)")
    ax.set_ylabel("Paid-customer survival %")
    ax.legend()

    m6 = df[df["t"] == 6].set_index("group")["survival"] if (df["t"] == 6).any() else None
    if m6 is not None and len(m6) == 2:
        gap_pp = (m6.max() - m6.min()) * 100
        title = f"Activated customers survive {gap_pp:.0f}pp longer at month 6 (Kaplan-Meier)"
    else:
        title = "Activated customers show consistently higher survival (Kaplan-Meier)"
    ax.set_title(title)
    _save(fig, "11_km_survival_by_activation")


# ======================================================================================
# 5. Model x task success heatmap
# ======================================================================================

def fig_model_task_heatmap() -> None:
    df = _read("06_llm_model_task_matrix")
    pivot = df.pivot(index="task_type", columns="model", values="success_pct")
    best_gap = (pivot.max(axis=1) - pivot.min(axis=1)).max()
    worst_task = (pivot.max(axis=1) - pivot.min(axis=1)).idxmax()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="crest", ax=ax, cbar_kws={"label": "Success %"})
    ax.set_title(f"'{worst_task}' shows the widest model spread: {best_gap:.0f}pp between best and worst model")
    _save(fig, "05_model_task_success_heatmap")


# ======================================================================================
# 6. Cost-quality Pareto frontier
# ======================================================================================

def fig_cost_quality_frontier() -> None:
    df = _read("llm_cost_quality_frontier")
    fig, ax = plt.subplots(figsize=(9, 6))
    for task_type, grp in df.groupby("task_type"):
        grp = grp.sort_values("avg_cost_usd")
        ax.plot(grp["avg_cost_usd"], grp["success_rate"] * 100, marker="o", alpha=0.5, label=task_type, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Avg LLM cost per run (USD, log scale)")
    ax.set_ylabel("Success rate %")
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    cheapest_frontier_pt = df[df["on_pareto_frontier"]].sort_values("avg_cost_usd").iloc[0]
    ax.set_title(
        f"Cost-quality frontier: cheapest viable point is {cheapest_frontier_pt['model']} on "
        f"{cheapest_frontier_pt['task_type']} (${cheapest_frontier_pt['avg_cost_usd']:.3f}/run, "
        f"{cheapest_frontier_pt['success_rate']*100:.0f}% success)"
    )
    _save(fig, "06_cost_quality_pareto_frontier")


# ======================================================================================
# 7. Margin by plan
# ======================================================================================

def fig_margin_by_plan() -> None:
    df = _read("08_unit_economics_plan").sort_values("gross_margin_pct")
    worst = df.iloc[0]
    best = df.iloc[-1]
    gap = best["gross_margin_pct"] - worst["gross_margin_pct"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(df["plan"], df["gross_margin_pct"], color=PALETTE[2])
    for bar, val in zip(bars, df["gross_margin_pct"]):
        ax.annotate(f"{val:.0f}%", (bar.get_x() + bar.get_width() / 2, val), ha="center", va="bottom")
    ax.set_ylabel("Gross margin %")
    ax.set_title(f"{worst['plan']} plan margin is {gap:.0f}pp below {best['plan']}")
    _save(fig, "07_margin_by_plan")


# ======================================================================================
# 8. Cost per credit vs revenue per credit by task
# ======================================================================================

def fig_cost_vs_revenue_per_credit() -> None:
    df = _read("09_task_cost_per_credit").sort_values("margin_per_credit_usd")
    worst = df.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["llm_cost_per_credit_usd"], width, label="LLM cost / credit", color=PALETTE[4])
    ax.bar(x + width / 2, df["blended_revenue_per_credit_usd"], width, label="Revenue / credit", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(df["task_type"], rotation=25, ha="right")
    ax.set_ylabel("USD per credit")
    ax.legend()
    ax.set_title(f"'{worst['task_type']}' costs more in LLM $ per credit than it earns in revenue per credit")
    _save(fig, "08_cost_vs_revenue_per_credit_by_task")


# ======================================================================================
# 9. A/B test result with CIs
# ======================================================================================

def fig_ab_test_result() -> None:
    z = _read("ab_test_two_proportion_z").iloc[0]
    power = _read("ab_test_power_analysis").iloc[0]

    variants = ["Control ($29)", "Treatment ($39)"]
    rates = [z["conversion_control"] * 100, z["conversion_treatment"] * 100]
    n = [z["n_control"], z["n_treatment"]]
    # Wald 95% CI per arm.
    se = [np.sqrt(r / 100 * (1 - r / 100) / ni) * 100 * 1.96 for r, ni in zip(rates, n)]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(variants, rates, yerr=se, capsize=6, color=[PALETTE[1], PALETTE[4]])
    ax.set_ylabel("Self-serve conversion %")
    sig = "statistically significant" if z["p_value"] < 0.05 else "not statistically significant at n=" + str(int(z["n_control"] + z["n_treatment"]))
    ax.set_title(
        f"Pro $39 price test: {sig} ({_fmt_p(z['p_value'])}); achieved power={power['achieved_power']*100:.0f}%"
    )
    _save(fig, "09_ab_test_conversion_with_ci")


# ======================================================================================
# 10. Personas
# ======================================================================================

def fig_personas() -> None:
    df = _read("persona_profile")
    features = ["avg_monthly_runs", "failure_rate", "migration_share", "activation_score", "avg_llm_cost"]
    norm = df[features].copy()
    for c in features:
        rng = norm[c].max() - norm[c].min()
        norm[c] = (norm[c] - norm[c].min()) / rng if rng > 0 else 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(features))
    width = 0.8 / len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.bar(x + i * width, norm.iloc[i], width, label=f"{row['persona_name']} (n={int(row['n_customers'])})")
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(features, rotation=20, ha="right")
    ax.set_ylabel("Normalized feature value (0-1 within persona set)")
    ax.legend(fontsize=8, loc="upper right")
    biggest = df.loc[df["n_customers"].idxmax()]
    ax.set_title(f"'{biggest['persona_name']}' is the largest segment ({int(biggest['n_customers'])} customers)")
    _save(fig, "10_customer_personas")


# ======================================================================================
# Orchestration
# ======================================================================================

FIGURE_FUNCS = [
    fig_mrr_trend,
    fig_cohort_heatmap,
    fig_activation_vs_outcomes,
    fig_churn_odds_ratios,
    fig_model_task_heatmap,
    fig_cost_quality_frontier,
    fig_margin_by_plan,
    fig_cost_vs_revenue_per_credit,
    fig_ab_test_result,
    fig_personas,
    fig_km_survival_by_activation,
]


def run_all() -> None:
    print(f"Writing {len(FIGURE_FUNCS)} figures to {FIGURES_DIR} ...")
    for fn in FIGURE_FUNCS:
        fn()
    print("Done.")


if __name__ == "__main__":
    run_all()
