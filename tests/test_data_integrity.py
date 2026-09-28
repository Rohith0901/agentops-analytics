"""test_data_integrity.py — sanity checks on data/agentops.db itself: no runs
before signup or after churn, tests_passed <= tests_total, costs/tokens >= 0,
revenue = fee + overage, unique IDs, FK integrity, and that experiment
assignments are restricted to self-serve customers within the assignment
window. Skips gracefully if the database hasn't been generated yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db import DB_PATH, query  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(), reason="data/agentops.db not generated yet — run src/generate_data.py first."
)


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    return query("SELECT * FROM customers")


@pytest.fixture(scope="module")
def customer_months() -> pd.DataFrame:
    return query("SELECT * FROM customer_months")


@pytest.fixture(scope="module")
def agent_runs() -> pd.DataFrame:
    return query("SELECT * FROM agent_runs")


@pytest.fixture(scope="module")
def experiment_assignments() -> pd.DataFrame:
    return query("SELECT * FROM experiment_assignments")


# ======================================================================================
# Uniqueness / referential integrity
# ======================================================================================

def test_customer_id_unique(customers: pd.DataFrame) -> None:
    assert customers["customer_id"].is_unique


def test_run_id_unique(agent_runs: pd.DataFrame) -> None:
    assert agent_runs["run_id"].is_unique


def test_agent_runs_customer_fk_integrity(agent_runs: pd.DataFrame, customers: pd.DataFrame) -> None:
    unknown = set(agent_runs["customer_id"]) - set(customers["customer_id"])
    assert not unknown, f"agent_runs reference unknown customer_ids: {sorted(unknown)[:5]}"


def test_customer_months_customer_fk_integrity(customer_months: pd.DataFrame, customers: pd.DataFrame) -> None:
    unknown = set(customer_months["customer_id"]) - set(customers["customer_id"])
    assert not unknown, f"customer_months reference unknown customer_ids: {sorted(unknown)[:5]}"


def test_experiment_assignments_customer_fk_integrity(
    experiment_assignments: pd.DataFrame, customers: pd.DataFrame
) -> None:
    unknown = set(experiment_assignments["customer_id"]) - set(customers["customer_id"])
    assert not unknown


def test_no_duplicate_customer_month_rows(customer_months: pd.DataFrame) -> None:
    dupes = customer_months.duplicated(subset=["customer_id", "month"]).sum()
    assert dupes == 0


# ======================================================================================
# Value-range invariants
# ======================================================================================

def test_costs_and_tokens_non_negative(agent_runs: pd.DataFrame) -> None:
    assert (agent_runs["llm_cost_usd"] >= 0).all()
    assert (agent_runs["input_tokens"] >= 0).all()
    assert (agent_runs["output_tokens"] >= 0).all()
    assert (agent_runs["steps"] >= 0).all()
    assert (agent_runs["latency_sec"] >= 0).all()
    assert (agent_runs["lines_changed"] >= 0).all()
    assert (agent_runs["credits_charged"] >= 0).all()


def test_tests_passed_le_tests_total(agent_runs: pd.DataFrame) -> None:
    assert (agent_runs["tests_passed"] <= agent_runs["tests_total"]).all()
    assert (agent_runs["tests_passed"] >= 0).all()


def test_human_rating_in_range_or_null(agent_runs: pd.DataFrame) -> None:
    rated = agent_runs["human_rating"].dropna()
    assert rated.between(1, 5).all()


def test_status_values_are_known(agent_runs: pd.DataFrame) -> None:
    assert set(agent_runs["status"].unique()) <= {"success", "partial", "failed"}


def test_credits_used_non_negative(customer_months: pd.DataFrame) -> None:
    assert (customer_months["credits_used"] >= 0).all()
    assert (customer_months["overage_credits"] >= 0).all()
    assert (customer_months["overage_revenue_usd"] >= 0).all()
    assert (customer_months["monthly_fee_usd"] >= 0).all()


# ======================================================================================
# Business-logic invariants
# ======================================================================================

def test_revenue_equals_fee_plus_overage(customer_months: pd.DataFrame) -> None:
    expected = customer_months["monthly_fee_usd"] + customer_months["overage_revenue_usd"]
    assert (customer_months["revenue_usd"] - expected).abs().max() < 1e-6


def test_runs_not_before_signup(agent_runs: pd.DataFrame, customers: pd.DataFrame) -> None:
    merged = agent_runs.merge(customers[["customer_id", "signup_date"]], on="customer_id")
    started = pd.to_datetime(merged["started_at"])
    signup = pd.to_datetime(merged["signup_date"])
    assert (started.dt.normalize() >= signup.dt.normalize()).all(), "found runs starting before the customer signed up"


def test_runs_not_after_churn(agent_runs: pd.DataFrame, customers: pd.DataFrame) -> None:
    churned = customers[customers["churn_date"].notna()][["customer_id", "churn_date"]]
    merged = agent_runs.merge(churned, on="customer_id", how="inner")
    started = pd.to_datetime(merged["started_at"])
    churn = pd.to_datetime(merged["churn_date"])
    # churn_date is the END of the last active month, so runs must not start after it.
    assert (started <= churn + pd.Timedelta(days=1)).all(), "found runs starting after the customer churned"


def test_free_plan_has_zero_fee_and_overage(customer_months: pd.DataFrame) -> None:
    free = customer_months[customer_months["plan"] == "Free"]
    assert (free["monthly_fee_usd"] == 0).all()
    assert (free["overage_revenue_usd"] == 0).all()


def test_experiment_assignments_are_self_serve_only(experiment_assignments: pd.DataFrame, customers: pd.DataFrame) -> None:
    merged = experiment_assignments.merge(customers[["customer_id", "is_sales_led"]], on="customer_id")
    assert (merged["is_sales_led"] == 0).all(), "experiment assigned to a sales-led (non self-serve) customer"


def test_experiment_assignments_within_window(experiment_assignments: pd.DataFrame) -> None:
    assigned = pd.to_datetime(experiment_assignments["assigned_at"])
    assert (assigned >= pd.Timestamp("2026-03-01")).all()
    assert (assigned <= pd.Timestamp("2026-05-31")).all()


def test_experiment_price_matches_variant(experiment_assignments: pd.DataFrame) -> None:
    control = experiment_assignments[experiment_assignments["variant"] == "control"]
    treatment = experiment_assignments[experiment_assignments["variant"] == "treatment"]
    assert (control["pro_price_shown"] == 29).all()
    assert (treatment["pro_price_shown"] == 39).all()
