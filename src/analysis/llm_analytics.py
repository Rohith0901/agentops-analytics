#!/usr/bin/env python3
"""llm_analytics.py — model x task performance analysis: pooled vs stratified
comparison (Simpson's paradox), chi-square independence test, cost-quality
Pareto frontier, an optimal routing policy with bootstrapped savings, and
human_rating vs status correlation.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import ROOT, query  # noqa: E402

TABLES_DIR = ROOT / "reports" / "tables"
RNG_SEED = 42


# ======================================================================================
# 1. Pooled vs stratified model comparison (Simpson's paradox)
# ======================================================================================

def load_runs() -> pd.DataFrame:
    return query(
        """
        SELECT model, task_type, status, llm_cost_usd, human_rating, plan_at_run
        FROM agent_runs
        """
    )


def pooled_vs_stratified(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare models two ways:
      pooled     -> success rate per model across ALL runs (task mix included)
      stratified -> success rate per model WITHIN each task_type

    Because models are routed differently by plan, and plans differ in task
    mix (e.g. Enterprise does more hard migrations), the pooled ranking of
    models can invert relative to every single stratum — the textbook
    Simpson's paradox setup. We report both so the reversal (if any) is
    explicit rather than hidden in a single aggregate number.
    """
    pooled = (
        runs.groupby("model")["status"]
        .apply(lambda s: (s == "success").mean())
        .rename("pooled_success_rate")
        .reset_index()
        .sort_values("pooled_success_rate", ascending=False)
    )

    stratified = (
        runs.groupby(["task_type", "model"])["status"]
        .apply(lambda s: (s == "success").mean())
        .rename("success_rate")
        .reset_index()
    )
    stratified["n_runs"] = runs.groupby(["task_type", "model"]).size().values

    return pooled, stratified


def detect_simpsons_reversal(pooled: pd.DataFrame, stratified: pd.DataFrame) -> pd.DataFrame:
    """For every pair of models, check whether the pooled ranking (which is
    best) disagrees with the ranking within *every* task stratum — i.e. a
    genuine Simpson's-paradox-style reversal driven by task-mix confounding.
    """
    rows = []
    models = pooled["model"].tolist()
    pooled_rank = dict(zip(pooled["model"], pooled["pooled_success_rate"]))
    for m1, m2 in combinations(models, 2):
        pooled_better = m1 if pooled_rank[m1] > pooled_rank[m2] else m2
        per_task = stratified[stratified["model"].isin([m1, m2])].pivot(
            index="task_type", columns="model", values="success_rate"
        )
        if m1 not in per_task or m2 not in per_task:
            continue
        per_task = per_task.dropna()
        stratified_winners = np.where(per_task[m1] > per_task[m2], m1, m2)
        agrees_with_pooled = (stratified_winners == pooled_better).mean() if len(stratified_winners) else np.nan
        rows.append({
            "model_a": m1,
            "model_b": m2,
            "pooled_winner": pooled_better,
            "pct_task_strata_agreeing_with_pooled_winner": agrees_with_pooled,
            "reversal_flag": bool(len(stratified_winners) and agrees_with_pooled == 0.0),
        })
    return pd.DataFrame(rows)


# ======================================================================================
# 2. Chi-square test: model vs outcome
# ======================================================================================

def chi_square_model_outcome(runs: pd.DataFrame) -> dict:
    contingency = pd.crosstab(runs["model"], runs["status"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    # Cramer's V effect size (0 = independent, larger = stronger association).
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramers_v),
        "n_observations": int(n),
    }


# ======================================================================================
# 3. Cost-quality Pareto frontier
# ======================================================================================

def cost_quality_frontier(runs: pd.DataFrame) -> pd.DataFrame:
    """One point per (task_type, model): avg LLM cost vs success rate. Flags
    points on the Pareto frontier (no other model for that task is both
    cheaper AND more successful).
    """
    agg = (
        runs.groupby(["task_type", "model"])
        .agg(avg_cost_usd=("llm_cost_usd", "mean"), success_rate=("status", lambda s: (s == "success").mean()),
             n_runs=("status", "size"))
        .reset_index()
    )

    def flag_frontier(group: pd.DataFrame) -> pd.Series:
        is_frontier = []
        for _, row in group.iterrows():
            dominated = (
                (group["avg_cost_usd"] <= row["avg_cost_usd"])
                & (group["success_rate"] >= row["success_rate"])
                & ((group["avg_cost_usd"] < row["avg_cost_usd"]) | (group["success_rate"] > row["success_rate"]))
            ).any()
            is_frontier.append(not dominated)
        return pd.Series(is_frontier, index=group.index)

    agg["on_pareto_frontier"] = agg.groupby("task_type", group_keys=False).apply(flag_frontier)
    return agg.sort_values(["task_type", "avg_cost_usd"])


# ======================================================================================
# 4. Routing policy: cheapest model within 2pp of best success, per task
# ======================================================================================

def bootstrap_success_rate_ci(
    statuses: np.ndarray, n_boot: int = 2000, rng: np.random.Generator | None = None
) -> tuple[float, float, float]:
    """Bootstrap a 95% CI on a success rate from an array of 'success'/other labels."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    successes = (statuses == "success").astype(float)
    point = successes.mean()
    n = len(successes)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = rng.choice(successes, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def routing_policy(runs: pd.DataFrame, current_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each task_type, find the best (highest) success rate among models,
    then pick the cheapest model whose success rate is within 2 percentage
    points of that best — the routing recommendation. Reports a bootstrap CI
    on each candidate's success rate so "within 2pp" is judged against
    estimation uncertainty, not just point estimates.
    """
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for task_type, grp in runs.groupby("task_type"):
        per_model = grp.groupby("model")
        stats_rows = []
        for model, mgrp in per_model:
            point, lo, hi = bootstrap_success_rate_ci(mgrp["status"].to_numpy(), rng=rng)
            stats_rows.append({
                "task_type": task_type,
                "model": model,
                "success_rate": point,
                "success_rate_ci_low": lo,
                "success_rate_ci_high": hi,
                "avg_cost_usd": mgrp["llm_cost_usd"].mean(),
                "n_runs": len(mgrp),
            })
        stats_df = pd.DataFrame(stats_rows)
        best_success = stats_df["success_rate"].max()
        eligible = stats_df[stats_df["success_rate"] >= best_success - 0.02]
        chosen = eligible.sort_values("avg_cost_usd").iloc[0]
        stats_df["is_current_best_success"] = stats_df["model"] == stats_df.loc[stats_df["success_rate"].idxmax(), "model"]
        stats_df["is_recommended"] = stats_df["model"] == chosen["model"]
        rows.append(stats_df)
    return pd.concat(rows, ignore_index=True)


def projected_routing_savings(runs: pd.DataFrame, routing: pd.DataFrame) -> pd.DataFrame:
    """Compare the CURRENT model mix's cost & success per task against the
    recommended (cheapest-within-2pp) model applied to all volume for that
    task, and project the $ impact if adopted, scaled to a monthly/annual run.

    Two distinct outcomes show up, and we label them rather than netting them
    together into one number:
      - "route_to_cheaper": the recommended model is cheaper than the current
        blended mix for that task (a real cost saving) — this happens on
        simple tasks where a cheaper model already matches frontier success.
      - "invest_in_reliability": no cheaper model is within 2pp of the best
        success rate, so the recommendation is to concentrate volume on the
        (already-best, usually frontier) model — this COSTS more per run but
        buys a success-rate improvement, which is a deliberate reliability
        trade-off, not a savings opportunity.
    """
    observed_months = query("SELECT COUNT(DISTINCT substr(started_at,1,7)) AS n FROM agent_runs")["n"].iloc[0]

    current = (
        runs.groupby("task_type")
        .agg(current_avg_cost_usd=("llm_cost_usd", "mean"),
             current_success_rate=("status", lambda s: (s == "success").mean()),
             n_runs=("status", "size"))
        .reset_index()
    )
    recommended = routing[routing["is_recommended"]][["task_type", "model", "success_rate", "avg_cost_usd"]].rename(
        columns={"model": "recommended_model", "success_rate": "recommended_success_rate",
                 "avg_cost_usd": "recommended_avg_cost_usd"}
    )
    merged = current.merge(recommended, on="task_type")
    merged["cost_delta_per_run_usd"] = merged["current_avg_cost_usd"] - merged["recommended_avg_cost_usd"]
    merged["routing_action"] = np.where(
        merged["cost_delta_per_run_usd"] > 0, "route_to_cheaper_model", "invest_in_reliability"
    )
    merged["success_rate_delta_pp"] = (merged["recommended_success_rate"] - merged["current_success_rate"]) * 100
    merged["runs_per_month"] = merged["n_runs"] / observed_months
    merged["monthly_dollar_impact_usd"] = merged["cost_delta_per_run_usd"] * merged["runs_per_month"]
    merged["annual_dollar_impact_usd"] = merged["monthly_dollar_impact_usd"] * 12
    return merged


# ======================================================================================
# 5. Human rating vs status: Spearman correlation
# ======================================================================================

STATUS_ORDER = {"failed": 0, "partial": 1, "success": 2}


def rating_vs_status_correlation(runs: pd.DataFrame) -> dict:
    rated = runs.dropna(subset=["human_rating"]).copy()
    rated["status_ordinal"] = rated["status"].map(STATUS_ORDER)
    rho, p = stats.spearmanr(rated["status_ordinal"], rated["human_rating"])
    return {
        "n_rated_runs": int(len(rated)),
        "spearman_rho": float(rho),
        "p_value": float(p),
        "mean_rating_by_status": rated.groupby("status")["human_rating"].mean().to_dict(),
    }


# ======================================================================================
# Orchestration
# ======================================================================================

def run_all() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()

    print("Pooled vs stratified model comparison (Simpson's paradox check) ...")
    pooled, stratified = pooled_vs_stratified(runs)
    pooled.to_csv(TABLES_DIR / "llm_pooled_success.csv", index=False)
    stratified.to_csv(TABLES_DIR / "llm_stratified_success.csv", index=False)
    reversal = detect_simpsons_reversal(pooled, stratified)
    reversal.to_csv(TABLES_DIR / "llm_simpsons_reversal_check.csv", index=False)
    print(pooled.to_string(index=False))
    print(reversal.to_string(index=False))

    print("\nChi-square test: model vs outcome ...")
    chi2_result = chi_square_model_outcome(runs)
    pd.DataFrame([chi2_result]).to_csv(TABLES_DIR / "llm_chi_square_model_outcome.csv", index=False)
    print(chi2_result)

    print("\nCost-quality Pareto frontier ...")
    frontier = cost_quality_frontier(runs)
    frontier.to_csv(TABLES_DIR / "llm_cost_quality_frontier.csv", index=False)
    print(frontier[frontier["on_pareto_frontier"]].to_string(index=False))

    print("\nRouting policy (cheapest model within 2pp of best success, per task) ...")
    routing = routing_policy(runs)
    routing.to_csv(TABLES_DIR / "llm_routing_policy.csv", index=False)
    print(routing[routing["is_recommended"]].to_string(index=False))

    print("\nProjected $ impact of adopting the routing policy ...")
    savings = projected_routing_savings(runs, routing)
    savings.to_csv(TABLES_DIR / "llm_routing_savings.csv", index=False)
    print(savings.to_string(index=False))
    real_savings = savings.loc[savings["routing_action"] == "route_to_cheaper_model", "annual_dollar_impact_usd"].sum()
    reliability_cost = -savings.loc[
        savings["routing_action"] == "invest_in_reliability", "annual_dollar_impact_usd"
    ].sum()
    print(f"\nAnnual $ savings from routing simple tasks to a cheaper model: ${real_savings:,.0f}")
    print(f"Annual $ cost to concentrate complex tasks on the highest-success model: ${reliability_cost:,.0f}")

    print("\nHuman rating vs status (Spearman) ...")
    rating_corr = rating_vs_status_correlation(runs)
    pd.DataFrame([{
        "n_rated_runs": rating_corr["n_rated_runs"],
        "spearman_rho": rating_corr["spearman_rho"],
        "p_value": rating_corr["p_value"],
    }]).to_csv(TABLES_DIR / "llm_rating_vs_status_spearman.csv", index=False)
    print(rating_corr)


if __name__ == "__main__":
    run_all()
