"""dashboard/app.py — AgentOps Analytics (CodeShift, synthetic data).

A read-only, filter-driven Streamlit dashboard over data/agentops.db.
See ARCHITECTURE.md "Testing & security requirements" for the security
contract this file (and dashboard/data.py) must honor: read-only SQLite,
parameterized queries only, allow-listed filters, no free-text SQL, no file
uploads, no unsafe_allow_html with data, no eval/exec/pickle.
"""

from __future__ import annotations

import subprocess  # nosec B404 - runs only our own data generator, fixed argv
import sys

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

import charts
import data as d

st.set_page_config(
    page_title="AgentOps Analytics — CodeShift",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("AgentOps Analytics — CodeShift (synthetic data)")
st.caption(
    "A fictional AI coding-agent SaaS. All figures are computed live from "
    "data/agentops.db (read-only) — nothing here is hard-coded."
)

# --------------------------------------------------------------------------
# Guard: missing database
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="First start: generating the synthetic dataset (about 10 seconds)...")
def _bootstrap_database() -> bool:
    """Build the database once if it is missing.

    The SQLite file is gitignored (it's ~70 MB and fully reproducible), so a
    fresh hosted deployment such as Streamlit Community Cloud starts without
    it. The generator is deterministic (seed 42), so the result is identical
    to a local `make data`. Fixed argv, no shell, no user input involved.
    """
    subprocess.run(  # nosec B603 - fixed command: our own script and seed
        [sys.executable, str(d.ROOT / "src" / "generate_data.py"), "--seed", "42"],
        check=True, capture_output=True, timeout=600,
    )
    return d.db_exists()


if not d.db_exists():
    try:
        _bootstrap_database()
    except (subprocess.SubprocessError, OSError):
        pass
if not d.db_exists():
    st.error(
        "The database `data/agentops.db` was not found and could not be generated. "
        "Run `make data` (or `python src/generate_data.py`) and reload."
    )
    st.stop()

options = d.get_filter_options()

# --------------------------------------------------------------------------
# Sidebar filters — every widget's choices come straight from the DB
# allow-lists above; selections are re-validated against those same
# allow-lists in dashboard/data.py before being used in any query.
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    months_sel = st.select_slider(
        "Month range",
        options=options["months"],
        value=(options["months"][0], options["months"][-1]),
    )
    all_months = options["months"]
    lo, hi = all_months.index(months_sel[0]), all_months.index(months_sel[1])
    month_range = all_months[lo: hi + 1]

    plans_sel = st.multiselect("Plan", options["plans"], default=[])
    regions_sel = st.multiselect("Region", options["regions"], default=[])
    channels_sel = st.multiselect("Acquisition channel", options["channels"], default=[])
    industries_sel = st.multiselect("Industry", options["industries"], default=[])

    st.caption(
        "Leave a filter empty to include all values. Filters apply across "
        "every tab where the metric is customer/plan/date scoped."
    )

f = d.Filters.build(
    months=month_range,
    plans=plans_sel,
    regions=regions_sel,
    channels=channels_sel,
    industries=industries_sel,
    options=options,
)

tabs = st.tabs([
    "Executive Overview",
    "Customers & Retention",
    "LLM Performance",
    "Pricing & Unit Economics",
    "Pricing Experiment",
    "Routing Simulator",
])

# --------------------------------------------------------------------------
# Tab 1 — Executive Overview
# --------------------------------------------------------------------------
with tabs[0]:
    fin = d.monthly_financials(f)
    succ = d.run_success_rate(f)
    nrr_val = d.nrr(f)

    if fin.empty:
        st.info("No data for the selected filters.")
    else:
        latest = fin.iloc[-1]
        prev = fin.iloc[-2] if len(fin) > 1 else None

        def _delta(cur, prev_val):
            if prev_val in (None, 0) or prev_val != prev_val:
                return None
            return f"{(cur - prev_val) / prev_val:+.1%}"

        def _delta_pp(cur, prev_val):
            """Change in a rate, in percentage points (a relative % of a % misleads)."""
            if prev_val is None or prev_val != prev_val:
                return None
            return f"{(cur - prev_val) * 100:+.1f} pp"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("MRR", f"${latest['mrr']:,.0f}",
                   _delta(latest["mrr"], prev["mrr"]) if prev is not None else None)
        c2.metric("Revenue", f"${latest['revenue']:,.0f}",
                   _delta(latest["revenue"], prev["revenue"]) if prev is not None else None)
        c3.metric("Gross margin", f"{latest['gross_margin']:.1%}",
                   _delta_pp(latest["gross_margin"], prev["gross_margin"]) if prev is not None else None)
        c4.metric("Active paid customers", f"{int(latest['active_paid_customers']):,}",
                   _delta(latest["active_paid_customers"], prev["active_paid_customers"]) if prev is not None else None)
        latest_succ = succ.iloc[-1]["success_rate"] if not succ.empty else None
        prev_succ = succ.iloc[-2]["success_rate"] if len(succ) > 1 else None
        c5.metric("Run success rate", f"{latest_succ:.1%}" if latest_succ is not None else "n/a",
                   _delta_pp(latest_succ, prev_succ) if latest_succ is not None and prev_succ is not None else None)
        c6.metric("NRR (avg, period)", f"{nrr_val:.1%}" if nrr_val is not None else "n/a")

        st.plotly_chart(charts.mrr_revenue_cost_trend(fin), width='stretch')
        best_margin_month = fin.loc[fin["gross_margin"].idxmax(), "month"]
        st.caption(
            f"Takeaway: gross margin peaked in {best_margin_month} at "
            f"{fin['gross_margin'].max():.1%}; latest month is {latest['gross_margin']:.1%}."
        )

        if not succ.empty:
            st.plotly_chart(charts.success_rate_trend(succ), width='stretch')
            st.caption(
                f"Takeaway: run success rate averages {succ['success_rate'].mean():.1%} "
                f"over the selected period."
            )

# --------------------------------------------------------------------------
# Tab 2 — Customers & Retention
# --------------------------------------------------------------------------
with tabs[1]:
    cohort = d.cohort_retention(f)
    act = d.activation_by_customer(f)
    chan = d.channel_summary(f)

    if cohort.empty:
        st.info("No data for the selected filters.")
    else:
        st.plotly_chart(charts.cohort_retention_heatmap(cohort), width='stretch')
        m3 = cohort[cohort["months_since_signup"] == 3]
        m3_avg = m3["retention"].mean() if not m3.empty else float("nan")
        st.caption(
            f"Takeaway: average logo retention at month 3 is "
            f"{m3_avg:.1%}" if m3_avg == m3_avg else "Takeaway: not enough history yet for month-3 retention."
        )

    if not act.empty:
        st.plotly_chart(charts.activation_vs_outcome(act), width='stretch')
        act_rate = act.groupby("activated")["retained"].mean()
        if True in act_rate.index and False in act_rate.index:
            lift = act_rate.get(True, 0) - act_rate.get(False, 0)
            st.caption(
                f"Takeaway: activated customers retain {lift:+.1%} points more than "
                f"non-activated customers."
            )

    if chan is not None and not chan.empty:
        st.plotly_chart(charts.channel_comparison(chan), width='stretch')
        best_channel = chan.loc[chan["conversion_rate"].idxmax(), "acquisition_channel"]
        st.caption(f"Takeaway: {best_channel} has the highest self-serve conversion rate.")

# --------------------------------------------------------------------------
# Tab 3 — LLM Performance
# --------------------------------------------------------------------------
with tabs[2]:
    mts = d.model_task_success(f)
    cps = d.cost_per_successful_run(f)
    mig = d.migration_pairs(f)
    fail = d.failure_cost(f)

    if mts.empty:
        st.info("No data for the selected filters.")
    else:
        st.plotly_chart(charts.model_task_heatmap(mts), width='stretch')
        best = mts.loc[mts["success_rate"].idxmax()]
        st.caption(
            f"Takeaway: {best['model']} on {best['task_type']} has the highest "
            f"observed success rate ({best['success_rate']:.1%})."
        )

        st.plotly_chart(charts.cost_success_frontier(mts), width='stretch')
        st.caption(
            "Takeaway: points near the top-left are the cost-quality frontier — "
            "high success at low cost per run."
        )

    if not cps.empty:
        st.plotly_chart(charts.cost_per_success_bar(cps), width='stretch')
        cheapest = cps.loc[cps["cost_per_success"].idxmin()]
        st.caption(
            f"Takeaway: {cheapest['model']} is cheapest per successful run "
            f"(${cheapest['cost_per_success']:.3f})."
        )

    if not mig.empty:
        st.subheader("Migration pairs")
        show = mig.copy()
        show["success_rate"] = show["success_rate"].map(lambda x: f"{x:.1%}")
        show["avg_cost"] = show["avg_cost"].map(lambda x: f"${x:.3f}")
        st.dataframe(show, width='stretch', hide_index=True)
        st.caption(
            f"Takeaway: {mig.iloc[0]['source_stack']} → {mig.iloc[0]['target_stack']} "
            f"is the highest-volume migration pair in the selected period."
        )

    if fail["n_runs"]:
        st.metric(
            "LLM $ spent on non-successful runs",
            f"${fail['failed_cost']:,.0f}",
            f"{fail['share']:.1%} of total LLM spend",
            delta_color="inverse",
        )
        st.caption(
            f"Takeaway: {fail['n_failed']:,} of {fail['n_runs']:,} runs did not succeed, "
            f"consuming {fail['share']:.1%} of all LLM spend without a completed task."
        )

# --------------------------------------------------------------------------
# Tab 4 — Pricing & Unit Economics
# --------------------------------------------------------------------------
with tabs[3]:
    mbp = d.margin_by_plan(f)
    crc = d.cost_revenue_per_credit_by_task(f)
    team_ref = d.team_revenue_per_credit(f)
    deciles = d.customer_margin_deciles(f)

    if not mbp.empty:
        st.plotly_chart(charts.margin_by_plan_bar(mbp), width='stretch')
        weakest = mbp.loc[mbp["gross_margin"].idxmin()]
        st.caption(
            f"Takeaway: {weakest['plan']} has the weakest gross margin "
            f"({weakest['gross_margin']:.1%})."
        )

    if not crc.empty:
        st.plotly_chart(charts.cost_vs_revenue_per_credit(crc, team_ref), width='stretch')
        if team_ref is not None:
            over = crc[crc["cost_per_credit"] > team_ref]
            if not over.empty:
                worst = over.loc[over["cost_per_credit"].idxmax()]
                st.caption(
                    f"Takeaway: {worst['task_type']} costs more per credit "
                    f"(${worst['cost_per_credit']:.3f}) than Team's revenue per credit "
                    f"(${team_ref:.3f}) — a margin drag on that plan."
                )
        else:
            st.caption("Takeaway: no Team-plan billing data in the selected filters.")

    if not deciles.empty:
        st.plotly_chart(charts.margin_decile_bar(deciles), width='stretch')
        neg_deciles = (deciles.groupby("decile")["gross_margin"].mean() < 0).sum()
        st.caption(
            f"Takeaway: {neg_deciles} of 10 customer margin deciles run negative "
            f"gross margin on average."
        )

# --------------------------------------------------------------------------
# Tab 5 — Pricing Experiment
# --------------------------------------------------------------------------
with tabs[4]:
    exp = d.experiment_conversion(f)

    if exp.empty or len(exp) < 2:
        st.info("No experiment data for the selected filters (experiment filters ignore date range).")
    else:
        exp = exp.set_index("variant")
        cis = {}
        for v in exp.index:
            n, k = int(exp.loc[v, "n"]), int(exp.loc[v, "conversions"])
            lo_ci, hi_ci = proportion_confint(k, n, alpha=0.05, method="wilson")
            cis[v] = (lo_ci, hi_ci)
        exp_reset = exp.reset_index()
        st.plotly_chart(charts.experiment_conversion_bar(exp_reset, cis), width='stretch')

        if "control" in exp.index and "treatment" in exp.index:
            n1, k1 = int(exp.loc["control", "n"]), int(exp.loc["control", "conversions"])
            n2, k2 = int(exp.loc["treatment", "n"]), int(exp.loc["treatment", "conversions"])
            p1, p2 = k1 / n1, k2 / n2
            p_pool = (k1 + k2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p1 - p2) / se if se > 0 else 0.0
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            st.subheader("Two-proportion z-test")
            colA, colB, colC = st.columns(3)
            colA.metric("Control conversion", f"{p1:.1%}", f"n={n1}")
            colB.metric("Treatment conversion", f"{p2:.1%}", f"n={n2}")
            colC.metric("p-value", f"{p_value:.3f}")
            st.caption(
                f"Takeaway: the observed conversion gap ({p1 - p2:+.1%}) is "
                + ("statistically significant at α=0.05." if p_value < 0.05
                   else "not statistically significant at α=0.05 — see power note below.")
            )

            # Power / MDE note (scipy-based two-proportion power via normal approx).
            avg_n = (n1 + n2) / 2
            alpha, power_target = 0.05, 0.8
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power_target)
            p_bar = (p1 + p2) / 2
            mde = (z_alpha + z_beta) * np.sqrt(2 * p_bar * (1 - p_bar) / avg_n)
            # Achieved power at the observed effect size, given actual n per arm.
            observed_effect = abs(p1 - p2)
            se_effect = np.sqrt(2 * p_bar * (1 - p_bar) / avg_n) if avg_n > 0 else np.nan
            achieved_power = (
                stats.norm.cdf(observed_effect / se_effect - z_alpha) if se_effect and se_effect > 0 else np.nan
            )
            st.caption(
                f"Power/MDE note: with n≈{avg_n:.0f} per arm at baseline conversion "
                f"≈{p_bar:.1%}, the minimum detectable effect at 80% power is "
                f"±{mde:.1%} points. The observed effect ({observed_effect:.1%} points) "
                f"implies achieved power ≈ {achieved_power:.0%}" +
                (" — likely under-powered." if achieved_power < 0.8 else ".")
            )

# --------------------------------------------------------------------------
# Tab 6 — Routing Simulator
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Route a share of simple-task frontier runs to balanced-medium")
    st.caption(
        "Simple tasks: test_generation, bug_fix, refactor, dependency_upgrade. "
        "Savings and success-rate impact are computed from each task's observed, "
        "stratified success/cost rates for frontier-large vs. balanced-medium."
    )
    stratified = d.simple_task_frontier_stats(f)

    if stratified.empty:
        st.info("No simple-task runs for the selected filters.")
    else:
        piv_cost = stratified.pivot_table(index="task_type", columns="model", values="avg_cost")
        piv_succ = stratified.pivot_table(index="task_type", columns="model", values="success_rate")
        piv_n = stratified.pivot_table(index="task_type", columns="model", values="n_runs")

        available = [t for t in piv_cost.index if "frontier-large" in piv_cost.columns and t in piv_cost.index]
        if "frontier-large" not in piv_cost.columns or not available:
            st.info("Not enough frontier-large / balanced-medium data to simulate.")
        else:
            pct = st.slider("% of simple-task frontier-large runs moved to balanced-medium",
                             min_value=0, max_value=100, value=25, step=5)
            frac = pct / 100.0

            frontier_n = piv_n.get("frontier-large", pd.Series(dtype=float)).fillna(0)
            frontier_cost = piv_cost.get("frontier-large", pd.Series(dtype=float))
            frontier_succ = piv_succ.get("frontier-large", pd.Series(dtype=float))
            balanced_cost = piv_cost.get("balanced-medium", pd.Series(dtype=float))
            balanced_succ = piv_succ.get("balanced-medium", pd.Series(dtype=float))

            moved_n = frontier_n * frac
            current_cost = (frontier_n * frontier_cost).sum()
            new_cost = ((frontier_n - moved_n) * frontier_cost).sum() + (moved_n * balanced_cost.reindex(frontier_n.index)).sum()
            savings = current_cost - new_cost

            current_success_weighted = (frontier_n * frontier_succ).sum()
            new_success_weighted = (
                ((frontier_n - moved_n) * frontier_succ).sum()
                + (moved_n * balanced_succ.reindex(frontier_n.index)).sum()
            )
            total_n = frontier_n.sum()
            current_rate = current_success_weighted / total_n if total_n else float("nan")
            new_rate = new_success_weighted / total_n if total_n else float("nan")

            c1, c2, c3 = st.columns(3)
            c1.metric("Projected monthly savings", f"${savings:,.0f}")
            c2.metric("Current frontier success rate (simple tasks)", f"{current_rate:.1%}")
            c3.metric("Projected success rate after routing", f"{new_rate:.1%}",
                      f"{(new_rate - current_rate):+.1%}")

            st.plotly_chart(charts.routing_savings_gauge(current_cost, new_cost), width='stretch')
            st.caption(
                f"Takeaway: moving {pct}% of simple-task frontier-large runs to "
                f"balanced-medium saves ~${savings:,.0f}/period while success rate "
                f"moves {(new_rate - current_rate):+.1%} points, based on observed "
                f"stratified rates."
            )
