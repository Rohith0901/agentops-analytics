"""test_sql.py — every SQL analysis file must run and return a non-empty
result with its expected columns; one metric is spot-checked against a
pandas recomputation from raw tables, to catch a SQL bug that still happens
to "run" without error. Skips gracefully if data/agentops.db is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db import DB_PATH, query, run_sql_file  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(), reason="data/agentops.db not generated yet — run src/generate_data.py first."
)

SQL_DIR = ROOT / "sql"

# Every file's minimum expected columns (a subset check, not exact-match, so
# the SQL can grow extra descriptive columns without breaking this test).
EXPECTED_COLUMNS = {
    "01_monthly_kpis": {"month", "mrr_usd", "revenue_usd", "active_paid_customers", "arpa_usd",
                         "new_logos", "churned_logos"},
    "02_nrr_grr": {"month", "cohort_customers", "base_revenue_usd", "retained_revenue_usd", "nrr_pct", "grr_pct"},
    "03_cohort_retention": {"cohort_month", "cohort_size", "months_since_signup", "retained_customers",
                            "retention_pct"},
    "04_activation": {"successes_out_of_5", "customers", "self_serve_conversion_pct", "retained_6mo_pct_of_paid"},
    "05_channel_performance": {"acquisition_channel", "signups", "self_serve_conversion_pct",
                                "paid_logo_churn_pct"},
    "06_llm_model_task_matrix": {"model", "task_type", "n_runs", "success_pct", "cost_per_success_usd",
                                  "p50_latency_sec", "p90_latency_sec"},
    "07_migration_pairs": {"migration_pair", "n_runs", "success_pct", "llm_cost_per_credit_usd"},
    "08_unit_economics_plan": {"plan", "total_revenue_usd", "total_llm_cost_usd", "gross_margin_pct",
                                "llm_cost_per_credit_usd", "revenue_per_credit_usd"},
    "09_task_cost_per_credit": {"task_type", "llm_cost_per_credit_usd", "blended_revenue_per_credit_usd",
                                 "margin_per_credit_usd"},
    "10_customer_margin_deciles": {"margin_decile", "customers", "avg_margin_pct"},
    "11_experiment_results": {"variant", "n_assigned", "n_converted_30d", "conversion_pct"},
    "12_failure_cost": {"model", "task_type", "n_failed_runs", "failed_llm_cost_usd", "pct_of_total_llm_spend"},
}


@pytest.mark.parametrize("name", sorted(EXPECTED_COLUMNS.keys()))
def test_sql_file_runs_and_has_expected_columns(name: str) -> None:
    path = SQL_DIR / f"{name}.sql"
    assert path.exists(), f"missing {path}"
    df = run_sql_file(path)
    assert not df.empty, f"{name}.sql returned an empty result"
    missing = EXPECTED_COLUMNS[name] - set(df.columns)
    assert not missing, f"{name}.sql is missing expected columns: {missing}"


def test_all_sql_files_are_covered_by_this_test() -> None:
    """Guards against a new sql/*.sql file being added without a column
    expectation (and therefore without real test coverage) here."""
    on_disk = {p.stem for p in SQL_DIR.glob("*.sql")}
    assert on_disk == set(EXPECTED_COLUMNS.keys())


# ======================================================================================
# Spot-check: recompute one metric independently in pandas and compare.
# ======================================================================================

def test_monthly_kpis_mrr_matches_pandas_recomputation() -> None:
    sql_result = run_sql_file(SQL_DIR / "01_monthly_kpis.sql")

    months = query("SELECT month, plan, monthly_fee_usd FROM customer_months WHERE plan <> 'Free'")
    pandas_mrr = months.groupby("month")["monthly_fee_usd"].sum().rename("mrr_usd").reset_index()

    merged = sql_result[["month", "mrr_usd"]].merge(pandas_mrr, on="month", suffixes=("_sql", "_pandas"))
    assert len(merged) == len(sql_result), "month set mismatch between SQL and pandas recomputation"
    assert (merged["mrr_usd_sql"] - merged["mrr_usd_pandas"]).abs().max() < 1e-6


def test_unit_economics_plan_margin_matches_pandas_recomputation() -> None:
    sql_result = run_sql_file(SQL_DIR / "08_unit_economics_plan.sql")

    months = query("SELECT customer_id, month, plan, revenue_usd FROM customer_months WHERE plan <> 'Free'")
    runs = query("SELECT customer_id, started_at, llm_cost_usd FROM agent_runs")
    runs["month"] = pd.to_datetime(runs["started_at"]).dt.strftime("%Y-%m-01")
    run_cost = runs.groupby(["customer_id", "month"])["llm_cost_usd"].sum().reset_index()

    merged = months.merge(run_cost, on=["customer_id", "month"], how="left")
    merged["llm_cost_usd"] = merged["llm_cost_usd"].fillna(0.0)
    by_plan = merged.groupby("plan").agg(total_revenue_usd=("revenue_usd", "sum"),
                                          total_llm_cost_usd=("llm_cost_usd", "sum"))
    by_plan["gross_margin_pct"] = (
        (by_plan["total_revenue_usd"] - by_plan["total_llm_cost_usd"]) / by_plan["total_revenue_usd"] * 100
    )

    for _, row in sql_result.iterrows():
        pandas_margin = by_plan.loc[row["plan"], "gross_margin_pct"]
        assert abs(row["gross_margin_pct"] - pandas_margin) < 0.05, (
            f"{row['plan']} margin mismatch: SQL={row['gross_margin_pct']} pandas={pandas_margin}"
        )
