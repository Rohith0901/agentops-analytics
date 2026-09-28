#!/usr/bin/env python3
"""pricing_analytics.py — unit economics by plan/task/decile, the price A/B
test (two-proportion z-test, bootstrap CI on revenue, power & MDE analysis),
an arc price-elasticity estimate, and forward-looking pricing scenarios.

All dollar figures are computed from data/agentops.db; pricing-scenario
projections state their assumptions explicitly in the returned tables (see
`pricing_scenarios`) rather than only in prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import ROOT, query  # noqa: E402

TABLES_DIR = ROOT / "reports" / "tables"
RNG_SEED = 42
ALPHA = 0.05
TARGET_POWER = 0.80


# ======================================================================================
# 1. Margin by plan / task / decile (reuses the SQL tables Builder B already
#    wrote — this module re-derives the plan-level numbers directly from the
#    DB too, so it can be tested/used independent of sql/*.sql having run).
# ======================================================================================

def margin_by_plan() -> pd.DataFrame:
    months = query(
        """
        SELECT customer_id, month, plan, revenue_usd, credits_used
        FROM customer_months
        WHERE plan <> 'Free'
        """
    )
    run_cost = query(
        """
        SELECT customer_id, substr(started_at, 1, 7) || '-01' AS month, SUM(llm_cost_usd) AS llm_cost_usd
        FROM agent_runs
        GROUP BY customer_id, substr(started_at, 1, 7)
        """
    )
    merged = months.merge(run_cost, on=["customer_id", "month"], how="left")
    merged["llm_cost_usd"] = merged["llm_cost_usd"].fillna(0.0)
    agg = merged.groupby("plan").agg(
        total_revenue_usd=("revenue_usd", "sum"),
        total_llm_cost_usd=("llm_cost_usd", "sum"),
    )
    agg["gross_margin_pct"] = (agg["total_revenue_usd"] - agg["total_llm_cost_usd"]) / agg["total_revenue_usd"] * 100
    return agg.reset_index().sort_values("gross_margin_pct")


def margin_by_task() -> pd.DataFrame:
    runs = query("SELECT task_type, status, llm_cost_usd, credits_charged FROM agent_runs")
    blended_rate = query(
        "SELECT SUM(revenue_usd) * 1.0 / SUM(credits_used) AS rate FROM customer_months WHERE plan <> 'Free'"
    )["rate"].iloc[0]
    agg = runs.groupby("task_type").agg(
        total_llm_cost_usd=("llm_cost_usd", "sum"),
        total_credits=("credits_charged", "sum"),
    )
    agg["implied_revenue_usd"] = agg["total_credits"] * blended_rate
    agg["margin_pct"] = (agg["implied_revenue_usd"] - agg["total_llm_cost_usd"]) / agg["implied_revenue_usd"] * 100
    return agg.reset_index().sort_values("margin_pct")


def margin_by_decile() -> pd.DataFrame:
    """Delegates to the same NTILE(10) logic as sql/10_customer_margin_deciles.sql,
    recomputed here in pandas so pricing_analytics has no runtime dependency
    on run_sql.py having executed first."""
    months = query("SELECT customer_id, revenue_usd FROM customer_months WHERE plan <> 'Free'")
    run_cost = query("SELECT customer_id, SUM(llm_cost_usd) AS llm_cost_usd FROM agent_runs GROUP BY customer_id")
    rev = months.groupby("customer_id")["revenue_usd"].sum().rename("revenue_usd")
    merged = rev.to_frame().join(run_cost.set_index("customer_id"), how="left")
    merged["llm_cost_usd"] = merged["llm_cost_usd"].fillna(0.0)
    merged = merged[merged["revenue_usd"] > 0]
    merged["margin_pct"] = (merged["revenue_usd"] - merged["llm_cost_usd"]) / merged["revenue_usd"]
    merged = merged.sort_values("margin_pct").reset_index()
    merged["decile"] = pd.qcut(merged["margin_pct"].rank(method="first"), 10, labels=False) + 1
    return merged.groupby("decile").agg(
        customers=("customer_id", "size"),
        avg_margin_pct=("margin_pct", lambda s: s.mean() * 100),
        total_revenue_usd=("revenue_usd", "sum"),
    ).reset_index()


# ======================================================================================
# 2. Price A/B test: two-proportion z-test, bootstrap CI, power / MDE / n
# ======================================================================================

def load_experiment_data() -> pd.DataFrame:
    """Per-assigned-customer experiment data: variant, whether they converted
    within 30 days, and their first-3-calendar-month revenue if converted."""
    assignments = query(
        """
        SELECT ea.customer_id, ea.variant, ea.assigned_at, c.first_paid_date
        FROM experiment_assignments ea
        JOIN customers c ON c.customer_id = ea.customer_id
        WHERE ea.experiment_name = 'pro_price_2026Q2'
        """
    )
    assignments["converted"] = (
        assignments["first_paid_date"].notna()
        & (
            (pd.to_datetime(assignments["first_paid_date"]) - pd.to_datetime(assignments["assigned_at"])).dt.days
            <= 30
        )
    ).astype(int)

    months = query("SELECT customer_id, month, plan, revenue_usd FROM customer_months WHERE plan <> 'Free'")
    months["month_ts"] = pd.to_datetime(months["month"])

    revenue_first_3mo = []
    converted = assignments[assignments["converted"] == 1].copy()
    converted["first_paid_month"] = pd.to_datetime(converted["first_paid_date"]).values.astype("datetime64[M]")
    for _, row in converted.iterrows():
        cust_months = months[months["customer_id"] == row["customer_id"]]
        window_end = row["first_paid_month"] + pd.DateOffset(months=3)
        in_window = cust_months[
            (cust_months["month_ts"] >= row["first_paid_month"]) & (cust_months["month_ts"] < window_end)
        ]
        revenue_first_3mo.append(in_window["revenue_usd"].sum())
    converted["revenue_first_3mo_usd"] = revenue_first_3mo

    assignments = assignments.merge(
        converted[["customer_id", "revenue_first_3mo_usd"]], on="customer_id", how="left"
    )
    assignments["revenue_first_3mo_usd"] = assignments["revenue_first_3mo_usd"].fillna(0.0)
    return assignments


def two_proportion_z_test(exp: pd.DataFrame) -> dict:
    control = exp[exp["variant"] == "control"]
    treatment = exp[exp["variant"] == "treatment"]
    n1, n2 = len(control), len(treatment)
    x1, x2 = control["converted"].sum(), treatment["converted"].sum()
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pool if se_pool > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% CI on the difference in proportions (unpooled SE, standard practice for a CI).
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "n_control": int(n1), "n_treatment": int(n2),
        "conversion_control": float(p1), "conversion_treatment": float(p2),
        "diff_control_minus_treatment": float(diff),
        "diff_ci_low": float(ci_low), "diff_ci_high": float(ci_high),
        "z_statistic": float(z), "p_value": float(p_value),
    }


def bootstrap_revenue_ci(exp: pd.DataFrame, n_boot: int = 5000, rng: np.random.Generator | None = None) -> dict:
    """Bootstrap 95% CIs on mean first-3-month revenue per ASSIGNED user
    (i.e. including non-converters as $0), for each variant and their diff."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    control = exp.loc[exp["variant"] == "control", "revenue_first_3mo_usd"].to_numpy()
    treatment = exp.loc[exp["variant"] == "treatment", "revenue_first_3mo_usd"].to_numpy()

    def boot_mean(arr: np.ndarray) -> np.ndarray:
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        return arr[idx].mean(axis=1)

    control_boot = boot_mean(control)
    treatment_boot = boot_mean(treatment)
    diff_boot = treatment_boot - control_boot

    def summarize(arr: np.ndarray, boot: np.ndarray) -> dict:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi)}

    return {
        "control_revenue_per_assigned": summarize(control, control_boot),
        "treatment_revenue_per_assigned": summarize(treatment, treatment_boot),
        "diff_treatment_minus_control": summarize(treatment.mean() - control.mean() + np.zeros(1), diff_boot),
    }


def power_and_mde(exp: pd.DataFrame) -> dict:
    """Achieved power at the observed effect and sample size, plus the
    minimum detectable effect (MDE) at 80% power with the observed n per arm,
    and the required n per arm to detect the OBSERVED effect at 80% power."""
    control = exp[exp["variant"] == "control"]
    treatment = exp[exp["variant"] == "treatment"]
    n1, n2 = len(control), len(treatment)
    p1, p2 = control["converted"].mean(), treatment["converted"].mean()

    analysis = NormalIndPower()
    observed_effect_size = proportion_effectsize(p1, p2)

    achieved_power = analysis.power(effect_size=observed_effect_size, nobs1=n1, ratio=n2 / n1, alpha=ALPHA)

    mde_effect_size = analysis.solve_power(nobs1=n1, ratio=n2 / n1, alpha=ALPHA, power=TARGET_POWER)
    # Convert effect size back to an approximate proportion difference around p1.
    mde_p2 = _effect_size_to_p2(mde_effect_size, p1)
    mde_pp = abs(mde_p2 - p1) * 100

    required_n = analysis.solve_power(effect_size=observed_effect_size, ratio=1.0, alpha=ALPHA, power=TARGET_POWER)

    return {
        "n_control": int(n1), "n_treatment": int(n2),
        "conversion_control": float(p1), "conversion_treatment": float(p2),
        "observed_effect_size_h": float(observed_effect_size),
        "achieved_power": float(achieved_power),
        "mde_at_80pct_power_pp": float(mde_pp),
        "required_n_per_arm_for_observed_effect_at_80pct_power": float(required_n),
    }


def _effect_size_to_p2(h: float, p1: float, tol: float = 1e-6) -> float:
    """Invert Cohen's h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1)) for p2, by
    direct algebra (arcsine transform is monotonic so this is well-defined)."""
    target = 2 * np.arcsin(np.sqrt(np.clip(p1, tol, 1 - tol))) + h
    p2 = np.sin(target / 2) ** 2
    return float(np.clip(p2, 0.0, 1.0))


# ======================================================================================
# 3. Arc price elasticity estimate
# ======================================================================================

def arc_elasticity(exp: pd.DataFrame) -> dict:
    """Arc price elasticity of conversion demand between the $29 control
    price and $39 treatment price:
        E = (%change in quantity) / (%change in price)
    using the arc (midpoint) formula so it is symmetric in direction.
    Quantity here is self-serve conversion rate (a demand proxy), since we
    do not observe a continuous quantity-purchased variable at the price
    point itself.
    """
    control = exp[exp["variant"] == "control"]
    treatment = exp[exp["variant"] == "treatment"]
    q1, q2 = control["converted"].mean(), treatment["converted"].mean()
    p1, p2 = 29.0, 39.0

    pct_change_q = (q2 - q1) / ((q1 + q2) / 2)
    pct_change_p = (p2 - p1) / ((p1 + p2) / 2)
    elasticity = pct_change_q / pct_change_p

    return {
        "price_control": p1, "price_treatment": p2,
        "conversion_control": float(q1), "conversion_treatment": float(q2),
        "arc_price_elasticity_of_conversion": float(elasticity),
    }


# ======================================================================================
# 4. Pricing scenarios
# ======================================================================================

def pricing_scenarios() -> pd.DataFrame:
    """Project revenue / LLM cost / margin under four scenarios, all applied
    to the OBSERVED historical run & billing data (i.e. "what would this
    year's economics have looked like under each pricing model"), stating
    assumptions explicitly per row.

    Scenarios:
      1. status_quo                    — actual observed revenue & cost.
      2. complexity_weighted_migration_credits
         — code_migration credits raised from 6 to 10 (assumption: credits
           should scale with the observed avg_steps ratio vs other complex
           tasks; 10 approximates that ratio), applied to migration-plan
           revenue at the current blended revenue-per-credit rate. Only
           incremental credit revenue is modeled; LLM cost is unchanged.
      3. route_simple_tasks_to_cheaper_model
         — simple tasks (test_generation, bug_fix, dependency_upgrade) fully
           routed to balanced-medium; LLM cost recomputed at that model's
           observed avg cost-per-run for each task; credits/revenue unchanged.
      4. pro_39
         — Pro plan's monthly_fee_usd raised from $29 to $39 for all Pro
           customer-months (assumption: no incremental churn beyond what the
           A/B test already measured — i.e. this is an upper-bound revenue
           estimate, see caveat in reports/insights.md).
    """
    months = query("SELECT customer_id, month, plan, revenue_usd, credits_used FROM customer_months WHERE plan <> 'Free'")
    runs = query("SELECT customer_id, started_at, task_type, model, llm_cost_usd, credits_charged FROM agent_runs")
    runs["month"] = pd.to_datetime(runs["started_at"]).dt.strftime("%Y-%m-01")

    total_revenue = months["revenue_usd"].sum()
    total_llm_cost = runs["llm_cost_usd"].sum()

    rows = [{
        "scenario": "status_quo",
        "assumption": "Actual observed billing and LLM cost, no changes.",
        "total_revenue_usd": total_revenue,
        "total_llm_cost_usd": total_llm_cost,
    }]

    # --- Scenario 2: complexity-weighted migration credits -------------------
    SIMPLE = {"test_generation", "bug_fix", "dependency_upgrade"}
    avg_steps_by_task = runs.groupby("task_type").apply(lambda g: g["credits_charged"].mean(), include_groups=False)
    migration_credits_current = runs.loc[runs["task_type"] == "code_migration", "credits_charged"].sum()
    NEW_MIGRATION_CREDITS_PER_TASK = 10  # was 6; ~1.7x, mirroring extra steps/cost migrations actually incur
    n_migration_runs = (runs["task_type"] == "code_migration").sum()
    blended_rate = total_revenue / months["credits_used"].sum()
    incremental_credits = n_migration_runs * (NEW_MIGRATION_CREDITS_PER_TASK - 6)
    incremental_revenue = incremental_credits * blended_rate
    rows.append({
        "scenario": "complexity_weighted_migration_credits",
        "assumption": f"code_migration credits raised 6->{NEW_MIGRATION_CREDITS_PER_TASK}/task; "
                       "incremental revenue priced at the current blended $/credit rate; LLM cost unchanged.",
        "total_revenue_usd": total_revenue + incremental_revenue,
        "total_llm_cost_usd": total_llm_cost,
    })

    # --- Scenario 3: route simple tasks to balanced-medium --------------------
    balanced_cost_by_task = (
        runs[runs["model"] == "balanced-medium"].groupby("task_type")["llm_cost_usd"].mean()
    )
    simple_runs = runs[runs["task_type"].isin(SIMPLE)]
    non_simple_cost = runs.loc[~runs["task_type"].isin(SIMPLE), "llm_cost_usd"].sum()
    rerouted_cost = sum(
        (simple_runs["task_type"] == t).sum() * balanced_cost_by_task.get(t, np.nan) for t in SIMPLE
    )
    rows.append({
        "scenario": "route_simple_tasks_to_balanced_medium",
        "assumption": "test_generation/bug_fix/dependency_upgrade fully routed to balanced-medium at its "
                       "observed avg cost-per-run for each task; revenue (credits) unchanged.",
        "total_revenue_usd": total_revenue,
        "total_llm_cost_usd": non_simple_cost + rerouted_cost,
    })

    # --- Scenario 4: Pro plan at $39 -------------------------------------------
    pro_months = months[months["plan"] == "Pro"]
    incremental_pro_revenue = len(pro_months) * 10  # +$10/mo per Pro customer-month
    rows.append({
        "scenario": "pro_39",
        "assumption": "Pro monthly_fee_usd raised $29->$39 for all historical Pro customer-months; "
                       "assumes no incremental churn beyond what the A/B test measured (upper-bound estimate).",
        "total_revenue_usd": total_revenue + incremental_pro_revenue,
        "total_llm_cost_usd": total_llm_cost,
    })

    df = pd.DataFrame(rows)
    df["gross_profit_usd"] = df["total_revenue_usd"] - df["total_llm_cost_usd"]
    df["gross_margin_pct"] = df["gross_profit_usd"] / df["total_revenue_usd"] * 100
    return df


# ======================================================================================
# Orchestration
# ======================================================================================

def run_all() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Margin by plan ...")
    mp = margin_by_plan()
    mp.to_csv(TABLES_DIR / "pricing_margin_by_plan.csv", index=False)
    print(mp.to_string(index=False))

    print("\nMargin by task ...")
    mt = margin_by_task()
    mt.to_csv(TABLES_DIR / "pricing_margin_by_task.csv", index=False)
    print(mt.to_string(index=False))

    print("\nMargin by customer decile ...")
    md = margin_by_decile()
    md.to_csv(TABLES_DIR / "pricing_margin_by_decile.csv", index=False)
    print(md.to_string(index=False))

    exp = load_experiment_data()

    print("\nTwo-proportion z-test (conversion, control vs treatment) ...")
    ztest = two_proportion_z_test(exp)
    pd.DataFrame([ztest]).to_csv(TABLES_DIR / "ab_test_two_proportion_z.csv", index=False)
    print(ztest)

    print("\nBootstrap CI on first-3-month revenue per assigned user ...")
    boot = bootstrap_revenue_ci(exp)
    pd.DataFrame([
        {"variant": "control", **boot["control_revenue_per_assigned"]},
        {"variant": "treatment", **boot["treatment_revenue_per_assigned"]},
        {"variant": "diff_treatment_minus_control", **boot["diff_treatment_minus_control"]},
    ]).to_csv(TABLES_DIR / "ab_test_bootstrap_revenue.csv", index=False)
    print(boot)

    print("\nPower analysis / MDE / required sample size ...")
    power = power_and_mde(exp)
    pd.DataFrame([power]).to_csv(TABLES_DIR / "ab_test_power_analysis.csv", index=False)
    print(power)

    print("\nArc price elasticity ...")
    elasticity = arc_elasticity(exp)
    pd.DataFrame([elasticity]).to_csv(TABLES_DIR / "pricing_arc_elasticity.csv", index=False)
    print(elasticity)

    print("\nPricing scenarios ...")
    scenarios = pricing_scenarios()
    scenarios.to_csv(TABLES_DIR / "pricing_scenarios.csv", index=False)
    print(scenarios.to_string(index=False))


if __name__ == "__main__":
    run_all()
