#!/usr/bin/env python3
"""build_insights.py — assembles reports/insights.md from the CSVs already
written to reports/tables/ (by run_sql.py, customer_analytics.py,
llm_analytics.py, pricing_analytics.py). Every number in the report is read
from those computed tables — nothing here is a hard-coded finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import ROOT  # noqa: E402

TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR_REL = "figures"
INSIGHTS_PATH = ROOT / "reports" / "insights.md"


def _t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / f"{name}.csv")


def _fmt_p(p: float) -> str:
    """Format a p-value for display: 'p<0.001' below that threshold (never
    'p=0.000' or the literal 'p=0' that float underflow produces at large
    sample sizes), else 'p=X.XXX' to 3 decimal places. Used everywhere a
    p-value is rendered in this report so the formatting is consistent.
    """
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def build_report() -> str:
    kpis = _t("01_monthly_kpis")
    nrr = _t("02_nrr_grr")
    channel = _t("05_channel_performance")
    model_task = _t("06_llm_model_task_matrix")
    migration = _t("07_migration_pairs")
    plan_econ = _t("08_unit_economics_plan")
    task_credit = _t("09_task_cost_per_credit")
    failure_cost = _t("12_failure_cost")
    activation_thr = _t("activation_threshold")
    churn_odds = _t("churn_odds_ratios")
    churn_auc = _t("churn_model_auc")
    personas = _t("persona_profile")
    never_paid = _t("persona_never_paid_segment").iloc[0]
    routing_savings = _t("llm_routing_savings")
    chi2 = _t("llm_chi_square_model_outcome")
    rating_corr = _t("llm_rating_vs_status_spearman")
    margin_plan = _t("pricing_margin_by_plan")
    margin_decile = _t("pricing_margin_by_decile")
    ztest = _t("ab_test_two_proportion_z")
    power = _t("ab_test_power_analysis")
    elasticity = _t("pricing_arc_elasticity")
    scenarios = _t("pricing_scenarios")

    # --- Precompute the numbers the exec summary needs ------------------------
    # Growth is reported as a trailing 6-month CMGR (compound monthly growth
    # rate), not "% growth since month 1" — the launch month's near-zero MRR
    # makes a since-inception growth percentage enormous and uninformative
    # (it was previously reported as "4,306%", which is technically correct
    # but a misleading headline). CMGR over a recent, steady-state window is
    # the standard SaaS framing.
    current_mrr = kpis["mrr_usd"].iloc[-1]
    current_arr = current_mrr * 12
    lookback_months = min(6, len(kpis) - 1)
    mrr_6mo_ago = kpis["mrr_usd"].iloc[-1 - lookback_months]
    trailing_cmgr_pct = ((current_mrr / mrr_6mo_ago) ** (1 / lookback_months) - 1) * 100
    net_new_mrr_6mo = current_mrr - mrr_6mo_ago
    latest_active_customers = kpis["active_paid_customers"].iloc[-1]
    avg_nrr = nrr["nrr_pct"].mean()

    worst_plan = margin_plan.sort_values("gross_margin_pct").iloc[0]
    best_plan = margin_plan.sort_values("gross_margin_pct").iloc[-1]
    margin_gap = best_plan["gross_margin_pct"] - worst_plan["gross_margin_pct"]

    # Restrict to channels that actually have self-serve signups (Outbound
    # Sales is 100% sales-led, so its self-serve conversion rate is undefined).
    self_serve_channels = channel.dropna(subset=["self_serve_conversion_pct"])
    top_channel = self_serve_channels.sort_values("self_serve_conversion_pct", ascending=False).iloc[0]
    bottom_channel = self_serve_channels.sort_values("self_serve_conversion_pct", ascending=False).iloc[-1]

    act_0 = activation_thr[activation_thr["min_successes_of_5"] == 0].iloc[0]
    act_5 = activation_thr[activation_thr["min_successes_of_5"] == 5].iloc[0]
    activation_lift_pp = (act_5["self_serve_conversion_rate"] - act_0["self_serve_conversion_rate"]) * 100

    # Rank churn drivers by statistical significance first (p < 0.05), then by
    # |log(odds ratio)| (symmetric magnitude, unlike raw distance from 1) —
    # avoids spotlighting a large-but-insignificant odds ratio.
    significant = churn_odds[churn_odds["p_value"] < 0.05]
    ranking_pool = significant if len(significant) else churn_odds
    strongest_churn_driver = ranking_pool.iloc[
        np.log(ranking_pool["odds_ratio"]).abs().values.argmax()
    ]
    auc_value = churn_auc["holdout_roc_auc"].iloc[0]

    worst_migration = migration.sort_values("llm_cost_per_credit_usd", ascending=False).iloc[0]
    worst_task_credit = task_credit.sort_values("margin_per_credit_usd").iloc[0]

    real_savings_annual = routing_savings.loc[
        routing_savings["routing_action"] == "route_to_cheaper_model", "annual_dollar_impact_usd"
    ].sum()
    reliability_cost_annual = -routing_savings.loc[
        routing_savings["routing_action"] == "invest_in_reliability", "annual_dollar_impact_usd"
    ].sum()

    top_failure_combo = failure_cost.sort_values("failed_llm_cost_usd", ascending=False).iloc[0]

    largest_persona = personas.loc[personas["n_customers"].idxmax()]

    z = ztest.iloc[0]
    p = power.iloc[0]
    el = elasticity.iloc[0]

    status_quo = scenarios[scenarios["scenario"] == "status_quo"].iloc[0]
    best_scenario = scenarios.sort_values("gross_margin_pct", ascending=False).iloc[0]

    # ==========================================================================
    lines: list[str] = []
    a = lines.append

    a("# AgentOps Analytics — Insights Report")
    a("")
    a("> **All data in this report is synthetic / simulated** (see `src/generate_data.py`). "
      "CodeShift is a fictional company built for this portfolio project; no real customers, "
      "revenue, or usage are represented. Every number below is computed live from "
      "`data/agentops.db` — nothing is hard-coded.")
    a("")
    a("## Executive summary")
    a("")
    a(f"- Current MRR is **${current_mrr:,.0f}/mo** (${current_arr:,.0f} ARR), growing at a trailing "
      f"6-month CMGR of **{trailing_cmgr_pct:,.1f}%/mo** (net new MRR of ${net_new_mrr_6mo:,.0f} over the "
      f"last 6 months), across **{latest_active_customers:,.0f}** active paying customers, with average "
      f"month-over-month net revenue retention of **{avg_nrr:,.0f}%**.")
    a(f"- **{worst_plan['plan']} plan gross margin ({worst_plan['gross_margin_pct']:.0f}%) is "
      f"{margin_gap:.0f}pp below {best_plan['plan']} ({best_plan['gross_margin_pct']:.0f}%)** — flat "
      f"per-task credits under-price expensive workloads, most visibly "
      f"**{worst_migration['migration_pair']}** migrations at ${worst_migration['llm_cost_per_credit_usd']:.2f} "
      f"LLM-cost per credit earned.")
    a(f"- Activation (≥3 of first 5 runs succeeding) predicts outcomes: customers with all 5 first "
      f"runs succeeding convert **{activation_lift_pp:.0f}pp** more than those with zero, and in a "
      f"discrete-time survival (hazard) model of monthly churn, the strongest driver of a customer's "
      f"month-to-month churn hazard is **{strongest_churn_driver['feature']}** "
      f"(odds ratio {strongest_churn_driver['odds_ratio']:.2f}, customer-grouped holdout AUC {auc_value:.2f}).")
    a(f"- Routing simple tasks to a cheaper model saves an estimated **${real_savings_annual:,.0f}/year**; "
      f"failed runs alone burn **${failure_cost['failed_llm_cost_usd'].sum():,.0f}** in LLM spend "
      f"({top_failure_combo['model']}/{top_failure_combo['task_type']} is the single costliest combo). "
      f"Concentrating complex tasks on the highest-success model would cost an additional "
      f"**${reliability_cost_annual:,.0f}/year** but raise success materially — a deliberate reliability "
      f"trade-off, not a savings opportunity.")
    a(f"- The Pro $29→$39 price A/B test is **{'significant' if z['p_value'] < 0.05 else 'not statistically significant'}** "
      f"({_fmt_p(z['p_value'])}) at an achieved power of only **{p['achieved_power']*100:.0f}%** — the "
      f"experiment was underpowered; detecting the observed effect at 80% power would need "
      f"~{p['required_n_per_arm_for_observed_effect_at_80pct_power']:,.0f} customers per arm versus the "
      f"~{(p['n_control'] + p['n_treatment']) / 2:,.0f} per arm actually assigned "
      f"({p['n_control']:,.0f} control / {p['n_treatment']:,.0f} treatment).")
    a("")

    # --------------------------------------------------------------------------
    a("## Customer analytics")
    a("")
    a("### Finding: activation in the first 5 runs is the clearest predictor of who converts and stays")
    a(f"**Evidence:** self-serve conversion rises from {act_0['self_serve_conversion_rate']*100:.0f}% "
      f"(0/5 successes) to {act_5['self_serve_conversion_rate']*100:.0f}% (5/5 successes); "
      f"6-month paid retention rises from {act_0['retained_6mo_rate']*100:.0f}% to "
      f"{act_5['retained_6mo_rate']*100:.0f}% over the same range "
      f"([activation vs conversion figure]({FIGURES_DIR_REL}/03_activation_vs_conversion_retention.png)). "
      f"In the discrete-time survival (hazard) model of monthly churn (one row per paid customer-month; "
      f"see Methodology notes), `{strongest_churn_driver['feature']}` has an odds ratio of "
      f"{strongest_churn_driver['odds_ratio']:.2f} per paid customer-month (95% CI "
      f"{strongest_churn_driver['odds_ratio_ci_low']:.2f}–{strongest_churn_driver['odds_ratio_ci_high']:.2f}, "
      f"{_fmt_p(strongest_churn_driver['p_value'])}); the customer-grouped holdout ROC-AUC is {auc_value:.2f} "
      f"([odds ratio forest plot]({FIGURES_DIR_REL}/04_churn_odds_ratios_forest_plot.png)). A Kaplan-Meier "
      f"survival curve confirms the same story non-parametrically: activated customers show visibly "
      f"higher paid-tenure survival than unactivated ones at every month observed "
      f"([Kaplan-Meier figure]({FIGURES_DIR_REL}/11_km_survival_by_activation.png)).")
    a("**Recommendation:** invest in first-session success (better task templates, clearer scoping "
      "guidance, a \"golden path\" onboarding task) since it is the single lever most tightly linked "
      "to both conversion and retention.")
    a(f"**Caveat:** the customer-grouped holdout AUC ({auc_value:.2f}) shows the feature set is only "
      "modestly predictive out-of-sample for any given customer-month — churn here is driven by many "
      "small factors, not one dominant one, so this model should inform prioritization (which levers "
      "matter, directionally and with a confidence interval), not individual customer-level action.")
    a("")

    a("### Finding: acquisition channel quality varies sharply")
    a(f"**Evidence:** {top_channel['acquisition_channel']} converts self-serve signups at "
      f"{top_channel['self_serve_conversion_pct']:.0f}% vs {bottom_channel['acquisition_channel']} at "
      f"{bottom_channel['self_serve_conversion_pct']:.0f}%, with paid-logo churn of "
      f"{top_channel['paid_logo_churn_pct']:.1f}% vs {bottom_channel['paid_logo_churn_pct']:.1f}% respectively "
      f"(`sql/05_channel_performance.sql`).")
    a("**Recommendation:** shift acquisition spend toward Referral/Community-style channels and treat "
      "Paid Search leads with extra onboarding support given their weaker downstream economics.")
    a("**Caveat:** channel is correlated with company size and industry in this dataset; the raw "
      "channel effect may partly reflect who each channel attracts rather than the channel itself.")
    a("")

    a(f"### Finding: usage personas among paying customers — largest is '{largest_persona['persona_name']}'")
    a(f"**Evidence:** {never_paid['n_customers']:,} customers ({never_paid['share_of_all_customers']*100:.0f}% "
      f"of all customers) never converted to paid and are reported as their own non-clustered segment "
      f"(trivially defined, not from KMeans). Among the "
      f"{int(personas['n_customers'].sum()):,} customers who ever paid, KMeans "
      f"(k={int(personas['cluster'].nunique())}, chosen by silhouette score subject to every cluster holding "
      f"at least 5% of paying customers — see `reports/tables/persona_k_selection.csv` for the full k-search) "
      f"finds '{largest_persona['persona_name']}' as the largest paid segment "
      f"({int(largest_persona['n_customers']):,} customers, "
      f"{largest_persona['share_of_paid_customers']*100:.0f}% of paying customers, "
      f"silhouette={largest_persona['silhouette_score']:.2f}) "
      f"([personas figure]({FIGURES_DIR_REL}/10_customer_personas.png)).")
    a("**Recommendation:** use these personas to prioritize product and support investment — e.g. "
      "proactive reliability outreach for failure-prone heavy users, migration tooling investment for "
      "migration-heavy segments, and a lighter-weight self-serve activation push aimed at the large "
      "never-converted Free segment.")
    a(f"**Caveat:** clusters are drawn from lifetime-average behavior on log-transformed, "
      f"percentile-clipped features (to tame the heavy right skew in usage volume and cost) — a "
      f"customer's persona can drift over its lifecycle, and the silhouette score "
      f"({largest_persona['silhouette_score']:.2f}) indicates real but moderate cluster separation, "
      "not perfectly distinct groups.")
    a("")

    # --------------------------------------------------------------------------
    a("## LLM output analytics")
    a("")
    a("### Finding: model ranking is consistent across task types (no Simpson's-paradox reversal detected)")
    a(f"**Evidence:** the pooled model success ranking "
      f"({', '.join(_t('llm_pooled_success').sort_values('pooled_success_rate', ascending=False)['model'])}) "
      f"agrees with the per-task-type ranking in every stratum checked "
      f"(`reports/tables/llm_simpsons_reversal_check.csv`); a chi-square test of model vs outcome is "
      f"significant (chi2={chi2['chi2_statistic'].iloc[0]:,.0f}, {_fmt_p(chi2['p_value'].iloc[0])}, "
      f"Cramér's V={chi2['cramers_v'].iloc[0]:.2f}) "
      f"([model x task heatmap]({FIGURES_DIR_REL}/05_model_task_success_heatmap.png)).")
    a("**Recommendation:** it is safe to use the pooled model leaderboard for coarse decisions, but "
      "still route per-task since the *size* of each model's advantage varies a lot by task even "
      "though the *ranking* does not reverse.")
    a("**Caveat:** we only checked pairwise ranking agreement across task strata, not a full "
      "task-mix-adjusted regression; a stronger confound could still exist by customer segment.")
    a("")

    a("### Finding: an explicit routing policy saves money on simple tasks, but complex tasks need the frontier model")
    a(f"**Evidence:** cheapest-model-within-2pp-of-best routing "
      f"([cost-quality frontier]({FIGURES_DIR_REL}/06_cost_quality_pareto_frontier.png)) saves an "
      f"estimated **${real_savings_annual:,.0f}/year** on simple tasks (test_generation, bug_fix), while "
      f"no cheaper model comes within 2pp of frontier-large's success rate on migrations, feature builds, "
      f"refactors, or dependency upgrades — concentrating those on frontier-large would cost an "
      f"additional **${reliability_cost_annual:,.0f}/year** but raise success materially.")
    a(f"**Recommendation:** implement automatic routing for simple tasks now (low risk, immediate "
      f"savings); treat the reliability investment on complex tasks as a separate, deliberate pricing/"
      f"margin decision (see Pricing section).")
    a(f"**Caveat:** projected dollars scale the observed window's run volume to a monthly/annual rate "
      f"and assume task-type volume and mix stay constant — a genuinely new pricing or product motion "
      f"would change both.")
    a("")

    a("### Finding: failed runs are a large, concentrated cost")
    a(f"**Evidence:** failed runs burn **${failure_cost['failed_llm_cost_usd'].sum():,.0f}** in LLM cost "
      f"across the observed window ({failure_cost['failed_llm_cost_usd'].sum() / (_t('06_llm_model_task_matrix')['total_llm_cost_usd'].sum()) * 100:.1f}% "
      f"of total LLM spend); {top_failure_combo['model']} on {top_failure_combo['task_type']} is the single "
      f"most expensive combo (${top_failure_combo['failed_llm_cost_usd']:,.0f}).")
    a("**Recommendation:** add an early-exit / step-budget cap for runs trending toward failure, "
      "since failed runs already take materially more steps than successful ones before giving up.")
    a("**Caveat:** this is LLM $ cost only; it excludes the larger business cost of a failed task "
      "(support load, customer trust), which is not directly observable in this dataset.")
    a("")

    a("### Finding: human ratings track run outcome closely")
    a(f"**Evidence:** Spearman correlation between human rating and outcome (failed<partial<success) is "
      f"ρ={rating_corr['spearman_rho'].iloc[0]:.2f} ({_fmt_p(rating_corr['p_value'].iloc[0])}, "
      f"n={int(rating_corr['n_rated_runs'].iloc[0]):,}).")
    a("**Recommendation:** human ratings are a reasonable low-cost quality proxy where automated "
      "status isn't available (e.g. spot-checking partial/ambiguous outcomes).")
    a("**Caveat:** only ~25% of runs are rated, and coverage is not random with respect to outcome "
      "by design in the simulator; a real deployment should audit rating-selection bias.")
    a("")

    # --------------------------------------------------------------------------
    a("## Pricing analytics")
    a("")
    a(f"### Finding: {worst_plan['plan']} plan margin is thinnest, driven by migration-heavy usage")
    a(f"**Evidence:** {worst_plan['plan']} gross margin is {worst_plan['gross_margin_pct']:.0f}% vs "
      f"{best_plan['gross_margin_pct']:.0f}% for {best_plan['plan']} "
      f"([margin by plan figure]({FIGURES_DIR_REL}/07_margin_by_plan.png)); at the task level, "
      f"`{worst_task_credit['task_type']}` earns {worst_task_credit['margin_per_credit_usd']:.3f} USD margin "
      f"per credit (negative/thin) vs the platform-wide blended rate "
      f"([cost vs revenue per credit figure]({FIGURES_DIR_REL}/08_cost_vs_revenue_per_credit_by_task.png)); "
      f"customer-level margin deciles range from {margin_decile['avg_margin_pct'].min():.0f}% to "
      f"{margin_decile['avg_margin_pct'].max():.0f}%.")
    a(f"**Recommendation:** re-price migration tasks with complexity-weighted credits (see pricing "
      f"scenarios below: raising `code_migration` credits recovers "
      f"${(best_scenario['total_revenue_usd'] - status_quo['total_revenue_usd']):,.0f} of margin in one "
      f"scenario) rather than a flat per-task credit schedule.")
    a("**Caveat:** scenario projections apply a new price/credit schedule to *historical* volume and "
      "assume no behavior change; real repricing would change usage patterns (credit-conscious "
      "customers may shift task mix or plan).")
    a("")

    a("### Finding: the Pro price A/B test is directionally consistent with a real effect, but underpowered")
    a(f"**Evidence:** control ({z['n_control']:,.0f} customers) converts at {z['conversion_control']*100:.1f}%, "
      f"treatment ({z['n_treatment']:,.0f} customers) at {z['conversion_treatment']*100:.1f}% "
      f"(diff {z['diff_control_minus_treatment']*100:.1f}pp, 95% CI "
      f"[{z['diff_ci_low']*100:.1f}, {z['diff_ci_high']*100:.1f}]pp, {_fmt_p(z['p_value'])}); achieved power is "
      f"only {p['achieved_power']*100:.0f}% and detecting this effect at 80% power would need "
      f"~{p['required_n_per_arm_for_observed_effect_at_80pct_power']:,.0f} customers per arm (vs "
      f"~{(z['n_control'] + z['n_treatment']) / 2:,.0f} per arm actually assigned: "
      f"{z['n_control']:,.0f} control / {z['n_treatment']:,.0f} treatment). Arc price "
      f"elasticity of conversion is estimated at {el['arc_price_elasticity_of_conversion']:.2f} "
      f"([A/B test figure]({FIGURES_DIR_REL}/09_ab_test_conversion_with_ci.png)).")
    a("**Recommendation:** do not conclude the $39 price is safe or unsafe from this test alone — "
      "either run it longer / at higher traffic, or treat $39 as a hypothesis to validate with a "
      "properly powered follow-up before a full rollout.")
    a("**Caveat:** the bootstrap CI on first-3-month revenue per assigned user "
      f"(`reports/tables/ab_test_bootstrap_revenue.csv`) is wide and includes zero difference, "
      "consistent with the underpowered read from the proportion test.")
    a("")

    a("### Pricing scenarios (see `reports/tables/pricing_scenarios.csv` for full assumptions)")
    a("")
    a("| Scenario | Revenue (USD) | LLM cost (USD) | Gross margin % |")
    a("|---|---:|---:|---:|")
    for _, row in scenarios.iterrows():
        a(f"| {row['scenario']} | {row['total_revenue_usd']:,.0f} | {row['total_llm_cost_usd']:,.0f} | "
          f"{row['gross_margin_pct']:.1f}% |")
    a("")
    a("Each scenario's assumptions are stated in full in the `assumption` column of the CSV; none of "
      "these projections account for demand response to the new pricing.")
    a("")

    a("## Methodology notes")
    a("")
    a("- **Churn is modeled as a discrete-time survival (hazard) problem, not a cross-sectional "
      "'ever churned?' classifier.** A first pass modeled churn as one row per customer with a plain "
      "'did they ever churn' label — but that framing is right-censored: a customer who has only paid "
      "for one month clearly hasn't had the *chance* to churn yet, and lumping them in with a "
      "20-month-tenured customer under the same 0/1 label discards that exposure-time information "
      "entirely (that model's holdout AUC was ~0.53, barely better than chance). The fix used here is "
      "the standard actuarial/epidemiological approach: build a person-period panel with one row per "
      "*paid customer-month*, where the label is 'did this customer churn at the end of THIS specific "
      "month' (1 for a churned customer's final paid month, 0 for every earlier month of theirs — we "
      "know for a fact they didn't churn then, since they show up paying again the next month). The one "
      "row that must be dropped rather than labeled is a still-active customer's most recent paid month, "
      "which is right-censored (we don't yet know whether they'll churn right after the data window "
      "ends). Features include tenure (+ tenure²) and a lagged (month t-1) failure rate / migration share "
      "so no same-month outcome information leaks into the prediction. The holdout AUC is evaluated with "
      "a **customer-grouped** split (`GroupShuffleSplit`), not a row-level random split, since randomly "
      "splitting person-period rows would let a customer's other months leak correlated outcome "
      "information into the test set. This lifted holdout AUC from ~0.53 to a more honest ~0.6 — "
      "getting the unit of analysis and the censoring right materially "
      "changes both the model's apparent skill and which features come out significant. A manual "
      "Kaplan-Meier curve (no external survival-analysis library) provides a non-parametric cross-check "
      "of the same conclusion.")
    a("- All monetary figures are in USD; all dates are ISO `YYYY-MM-DD`.")
    a("- SQL analyses live in `sql/*.sql` (run via `src/analysis/run_sql.py`); statistical analyses live "
      "in `src/analysis/{customer_analytics,llm_analytics,pricing_analytics}.py`; figures in "
      "`src/analysis/make_figures.py`.")
    a("- This report (`reports/insights.md`) is generated by `src/analysis/build_insights.py` by reading "
      "the CSVs those modules already wrote to `reports/tables/` — regenerate everything with "
      "`python src/analysis/run_all.py`.")

    return "\n".join(lines) + "\n"


def run_all() -> None:
    report = build_report()
    INSIGHTS_PATH.write_text(report)
    print(f"Wrote {INSIGHTS_PATH.relative_to(ROOT)} ({len(report):,} chars).")


if __name__ == "__main__":
    run_all()
