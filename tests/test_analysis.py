"""test_analysis.py — fast unit tests for src/analysis: the routing policy
picks a valid model per task, the A/B test function gives a known p-value on
a known synthetic input, and KMeans persona output has the right shape.
Skips gracefully if data/agentops.db is missing (some tests need the real DB
to build routing/persona features; the A/B-test-function test does not).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db import DB_PATH, query  # noqa: E402

needs_db = pytest.mark.skipif(
    not DB_PATH.exists(), reason="data/agentops.db not generated yet — run src/generate_data.py first."
)

sys.path.insert(0, str(ROOT / "src" / "analysis"))
import llm_analytics  # noqa: E402
import pricing_analytics  # noqa: E402


# ======================================================================================
# Routing policy: valid model choice per task
# ======================================================================================

@needs_db
def test_routing_policy_picks_valid_models() -> None:
    runs = llm_analytics.load_runs()
    routing = llm_analytics.routing_policy(runs)

    valid_models = set(runs["model"].unique())
    valid_tasks = set(runs["task_type"].unique())

    recommended = routing[routing["is_recommended"]]
    # Exactly one recommendation per task type.
    assert set(recommended["task_type"]) == valid_tasks
    assert recommended["task_type"].is_unique
    assert set(recommended["model"]) <= valid_models

    # The recommended model's success rate must be within 2pp of the best
    # success rate observed for that task (the policy's own definition).
    for task_type, grp in routing.groupby("task_type"):
        best = grp["success_rate"].max()
        rec = grp[grp["is_recommended"]].iloc[0]
        assert rec["success_rate"] >= best - 0.02 - 1e-9


@needs_db
def test_routing_policy_recommends_cheapest_within_threshold() -> None:
    runs = llm_analytics.load_runs()
    routing = llm_analytics.routing_policy(runs)
    for task_type, grp in routing.groupby("task_type"):
        best = grp["success_rate"].max()
        eligible = grp[grp["success_rate"] >= best - 0.02]
        cheapest_eligible_cost = eligible["avg_cost_usd"].min()
        rec = grp[grp["is_recommended"]].iloc[0]
        assert abs(rec["avg_cost_usd"] - cheapest_eligible_cost) < 1e-9


# ======================================================================================
# A/B test function: known synthetic input -> known p-value
# ======================================================================================

def test_two_proportion_z_test_on_known_synthetic_input() -> None:
    """Two arms with identical, large sample size and a clearly different
    conversion rate should be extremely statistically significant; two arms
    with identical conversion rates should give p close to 1.
    """
    rng = np.random.default_rng(0)

    # Arm A: 40% conversion, Arm B: 60% conversion, n=500 each -> should be
    # a highly significant difference.
    control = pd.DataFrame({
        "variant": ["control"] * 500,
        "converted": (rng.random(500) < 0.40).astype(int),
    })
    treatment = pd.DataFrame({
        "variant": ["treatment"] * 500,
        "converted": (rng.random(500) < 0.60).astype(int),
    })
    exp = pd.concat([control, treatment], ignore_index=True)
    result = pricing_analytics.two_proportion_z_test(exp)
    assert result["p_value"] < 0.001
    assert result["diff_control_minus_treatment"] < 0  # treatment converts more

    # Identical conversion rates -> p-value should be large (not significant).
    rng2 = np.random.default_rng(1)
    same_rate = pd.DataFrame({
        "variant": ["control"] * 1000 + ["treatment"] * 1000,
        "converted": (rng2.random(2000) < 0.30).astype(int),
    })
    result_same = pricing_analytics.two_proportion_z_test(same_rate)
    assert result_same["p_value"] > 0.10


def test_two_proportion_z_test_exact_known_values() -> None:
    """A hand-computable case: 100 vs 100, 50 vs 30 conversions.
    p1=0.5, p2=0.3, pooled p=0.4, se=sqrt(0.4*0.6*(1/100+1/100))=0.06928...
    z = 0.2 / 0.069282 = 2.8868 -> two-sided p ~= 0.00389.
    """
    exp = pd.DataFrame({
        "variant": ["control"] * 100 + ["treatment"] * 100,
        "converted": [1] * 50 + [0] * 50 + [1] * 30 + [0] * 70,
    })
    result = pricing_analytics.two_proportion_z_test(exp)
    assert result["conversion_control"] == pytest.approx(0.5)
    assert result["conversion_treatment"] == pytest.approx(0.3)
    assert result["z_statistic"] == pytest.approx(2.8868, abs=1e-3)
    assert result["p_value"] == pytest.approx(0.00389, abs=1e-3)


# ======================================================================================
# KMeans personas: output shape
# ======================================================================================

@needs_db
def test_kmeans_personas_output_shape() -> None:
    sys.path.insert(0, str(ROOT / "src" / "analysis"))
    import customer_analytics

    assignments, profile, k, diagnostics = customer_analytics.kmeans_personas(k_range=range(3, 5))

    assert 3 <= k <= 4
    assert set(assignments["cluster"].unique()) == set(range(k))
    assert len(profile) == k
    assert profile["n_customers"].sum() == len(assignments)
    for col in customer_analytics.PERSONA_FEATURES:
        assert col in profile.columns
    assert "persona_name" in profile.columns
    assert profile["persona_name"].notna().all()
    assert {"k", "silhouette_score", "smallest_cluster_share", "meets_min_cluster_share"} <= set(diagnostics.columns)


@needs_db
def test_kmeans_personas_have_no_tiny_cluster_and_unique_names() -> None:
    """Guards against the earlier degenerate fit where one cluster absorbed
    96% of customers: every cluster in the CHOSEN k must hold at least 5% of
    the clustered (ever-paid) population, and persona names must be unique."""
    sys.path.insert(0, str(ROOT / "src" / "analysis"))
    import customer_analytics

    assignments, profile, k, diagnostics = customer_analytics.kmeans_personas()

    chosen_row = diagnostics[diagnostics["k"] == k].iloc[0]
    min_share = profile["n_customers"].min() / profile["n_customers"].sum()
    assert min_share >= customer_analytics.MIN_CLUSTER_SHARE - 1e-9, (
        f"smallest cluster share {min_share:.3f} is below the {customer_analytics.MIN_CLUSTER_SHARE} floor"
    )
    assert chosen_row["meets_min_cluster_share"]

    names = profile["persona_name"].tolist()
    assert len(names) == len(set(names)), f"duplicate persona names: {names}"

    # Customers who never paid must never appear in the clustered assignments.
    customers = query("SELECT customer_id, first_paid_date FROM customers")
    never_paid_ids = set(customers.loc[customers["first_paid_date"].isna(), "customer_id"])
    assert not (set(assignments["customer_id"]) & never_paid_ids)


@needs_db
def test_never_paid_segment_is_disjoint_and_has_zero_revenue() -> None:
    import customer_analytics

    df_all = customer_analytics.build_customer_usage_features()
    seg = customer_analytics.never_paid_segment(df_all)

    assert seg["n_customers"] > 0
    assert seg["lifetime_revenue_usd"] == 0.0
    assert 0.0 < seg["share_of_all_customers"] < 1.0


# ======================================================================================
# Churn discrete-time survival (hazard) model
# ======================================================================================

@needs_db
def test_survival_panel_excludes_right_censored_row_and_labels_events_correctly() -> None:
    import customer_analytics

    panel = customer_analytics.build_survival_panel()

    # No panel row should be a still-active customer's LAST paid row sitting
    # on the last calendar month present in the data — that observation is
    # right-censored and must be dropped, not labeled as a non-event.
    max_month = panel["month"].max()
    assert not ((panel["status"] == "active") & panel["is_last_row"] & (panel["month"] == max_month)).any()

    # Every churned customer contributes exactly one event row, and it must
    # be their last observed paid-month row.
    churned_customers = panel[panel["status"] == "churned"]
    events_per_customer = churned_customers.groupby("customer_id")["churned_event"].sum()
    assert (events_per_customer == 1).all()
    event_rows = churned_customers[churned_customers["churned_event"] == 1]
    assert (event_rows["is_last_row"]).all()

    # No customer contributes more than one event, and non-final rows never carry the event label.
    non_final_rows = panel[~panel["is_last_row"]]
    assert (non_final_rows["churned_event"] == 0).all()


@needs_db
def test_churn_hazard_model_grouped_holdout_auc_in_range() -> None:
    import customer_analytics

    odds, auc, panel = customer_analytics.churn_hazard_model()

    assert 0.0 <= auc <= 1.0
    assert {"feature", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high", "p_value"} <= set(odds.columns)
    assert (odds["odds_ratio"] > 0).all()
    assert len(panel) > 0
    assert panel["churned_event"].isin([0, 1]).all()


def test_kaplan_meier_manual_matches_hand_computed_case() -> None:
    """4 subjects: durations [1, 2, 2, 3], events [1, 0, 1, 1] (subject at
    duration 1 is censored... let's use an unambiguous hand-computable case:
    durations=[1,1,2,3], events=[1,1,1,0].
    n@1=4, d@1=2 -> S(1)=1-2/4=0.5
    n@2=2, d@2=1 -> S(2)=0.5*(1-1/2)=0.25
    n@3=1, d@3=0 (censored) -> S(3)=0.25*(1-0/1)=0.25
    """
    import customer_analytics

    km = customer_analytics.kaplan_meier(
        durations=np.array([1, 1, 2, 3]), events=np.array([1, 1, 1, 0])
    )
    km = km.set_index("t")
    assert km.loc[1, "survival"] == pytest.approx(0.5)
    assert km.loc[2, "survival"] == pytest.approx(0.25)
    assert km.loc[3, "survival"] == pytest.approx(0.25)
    assert km.loc[3, "events"] == 0
    assert km.loc[3, "n_at_risk"] == 1
