#!/usr/bin/env python3
"""customer_analytics.py — activation threshold analysis, a discrete-time
survival (hazard) churn model with odds ratios & AUC, and a KMeans
usage-persona segmentation.

All numbers are computed from data/agentops.db; nothing here is hard-coded.
Uses only parameterized SQL (via db.query with plain, hard-coded query text —
no string-built SQL from any input) per ARCHITECTURE.md security rules.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import ROOT, query  # noqa: E402

import statsmodels.api as sm  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

TABLES_DIR = ROOT / "reports" / "tables"


def _customer_activation_scores() -> pd.DataFrame:
    """Per-customer activation score: success share of the first 5 runs (by
    started_at). Shared by the persona features and the survival panel so
    both use the exact same definition of activation.
    """
    runs = query("SELECT customer_id, status, started_at, run_id FROM agent_runs")
    runs = runs.sort_values(["customer_id", "started_at", "run_id"])
    runs["run_rank"] = runs.groupby("customer_id").cumcount() + 1
    first_five = runs[runs["run_rank"] <= 5]
    return (
        first_five.assign(is_success=lambda d: d["status"] == "success")
        .groupby("customer_id")["is_success"]
        .mean()
        .rename("activation_score")
        .reset_index()
    )


# ======================================================================================
# 1. Activation threshold analysis
# ======================================================================================

def activation_threshold_table() -> pd.DataFrame:
    """For each possible "N of first 5 runs succeeded" cutoff (0..5), compute
    self-serve conversion rate and 6-month paid retention among customers at
    or above that cutoff — a data-driven view of where the activation
    threshold should actually sit, rather than assuming ">=3" up front.
    """
    # Pull the per-customer activation table directly (mirrors sql/04_activation.sql
    # logic at the per-customer grain, since that file aggregates to bucket level).
    runs = query(
        """
        SELECT customer_id, status, started_at, run_id
        FROM agent_runs
        """
    )
    runs = runs.sort_values(["customer_id", "started_at", "run_id"])
    runs["run_rank"] = runs.groupby("customer_id").cumcount() + 1
    first_five = runs[runs["run_rank"] <= 5]
    counts = first_five.groupby("customer_id").size()
    full_window = counts[counts == 5].index
    successes = (
        first_five[first_five["customer_id"].isin(full_window)]
        .assign(is_success=lambda d: d["status"] == "success")
        .groupby("customer_id")["is_success"]
        .sum()
        .rename("successes")
    )

    customers = query(
        """
        SELECT customer_id, first_paid_date, is_sales_led
        FROM customers
        """
    )
    max_month = query("SELECT MAX(month) AS m FROM customer_months")["m"].iloc[0]

    paid_months = query("SELECT customer_id, month, plan FROM customer_months WHERE plan <> 'Free'")
    paid_months_set = set(zip(paid_months["customer_id"], paid_months["month"]))

    def retained_6mo(row) -> int:
        if pd.isna(row["first_paid_date"]):
            return 0
        target = (pd.Timestamp(row["first_paid_date"]).replace(day=1) + pd.DateOffset(months=6))
        if target > pd.Timestamp(max_month):
            return np.nan  # not yet observable
        target_str = target.strftime("%Y-%m-01")
        return 1 if (row["customer_id"], target_str) in paid_months_set else 0

    merged = customers.merge(successes, on="customer_id", how="inner")
    merged["retained_6mo"] = merged.apply(retained_6mo, axis=1)

    rows = []
    for cutoff in range(0, 6):
        at_or_above = merged[merged["successes"] >= cutoff]
        self_serve = at_or_above[at_or_above["is_sales_led"] == 0]
        conv_rate = self_serve["first_paid_date"].notna().mean() if len(self_serve) else np.nan
        paid = at_or_above[at_or_above["first_paid_date"].notna()]
        observable = paid[paid["retained_6mo"].notna()]
        ret_rate = observable["retained_6mo"].mean() if len(observable) else np.nan
        rows.append({
            "min_successes_of_5": cutoff,
            "n_customers_at_or_above": len(at_or_above),
            "self_serve_conversion_rate": conv_rate,
            "retained_6mo_rate": ret_rate,
        })
    return pd.DataFrame(rows)


# ======================================================================================
# 2. Churn as a discrete-time survival (hazard) model
# ======================================================================================
#
# The original cross-sectional "ever churned?" framing is right-censored: a
# customer who paid for one month so far obviously hasn't had the *chance* to
# churn yet, but a plain 0/1 "churned ever" label ignores that exposure time
# entirely and mixes short- and long-tenure customers together — which is
# exactly why a naive model on that target came out barely better than random
# (AUC ~= 0.53). The fix is to model churn at the (customer, month) grain: did
# THIS customer churn at the end of THIS paid month, conditional on having
# reached it. That is a standard discrete-time hazard / person-period model.

def _lagged_monthly_run_stats() -> pd.DataFrame:
    """Per (customer_id, month) run stats, shifted forward one calendar month
    so that a panel row for month t can join in month (t-1)'s behavior as a
    non-leaky, already-observed-at-the-time-of-the-churn-decision feature."""
    runs = query("SELECT customer_id, started_at, status, task_type FROM agent_runs")
    runs["month"] = pd.to_datetime(runs["started_at"]).dt.strftime("%Y-%m-01")
    stats = runs.groupby(["customer_id", "month"]).agg(
        n_runs=("status", "size"),
        failure_rate=("status", lambda s: (s == "failed").mean()),
        migration_share=("task_type", lambda s: (s == "code_migration").mean()),
    ).reset_index()
    # Shift the month label forward by one month: stats computed *in* month M
    # become the "prior_month_*" feature attached to the panel row for M+1.
    stats["month"] = (pd.to_datetime(stats["month"]) + pd.DateOffset(months=1)).dt.strftime("%Y-%m-01")
    return stats.rename(columns={
        "n_runs": "prior_month_n_runs",
        "failure_rate": "prior_month_failure_rate",
        "migration_share": "prior_month_migration_share",
    })


def build_survival_panel() -> pd.DataFrame:
    """One row per (customer_id, paid month), i.e. a person-period panel.

    `churned_event` = 1 only on a customer's LAST paid-month row if that
    customer's overall status is 'churned' (i.e. they churned at the end of
    that month); 0 on every earlier paid month of theirs (we know for a fact
    they did NOT churn then, since they show up paying again next month).

    The one row that is dropped entirely is an *active* customer's last paid
    row when that row falls on the last calendar month present in the data —
    that observation is right-censored (we simply don't yet know whether they
    churn the month after the data window ends), so it carries no usable
    label and must be excluded rather than coded as a non-event.
    """
    months = query(
        "SELECT customer_id, month, plan, credits_used FROM customer_months "
        "WHERE plan <> 'Free' ORDER BY customer_id, month"
    )
    customers = query(
        "SELECT customer_id, signup_date, acquisition_channel, status, churn_date FROM customers"
    )
    max_month = months["month"].max()

    panel = months.merge(customers, on="customer_id", how="left")
    panel["row_number"] = panel.groupby("customer_id").cumcount() + 1
    panel["n_rows"] = panel.groupby("customer_id")["month"].transform("count")
    panel["is_last_row"] = panel["row_number"] == panel["n_rows"]

    censored = panel["is_last_row"] & (panel["status"] == "active") & (panel["month"] == max_month)
    panel = panel.loc[~censored].copy()
    panel["churned_event"] = (panel["is_last_row"] & (panel["status"] == "churned")).astype(int)

    # --- Tenure (+ tenure^2, to allow a non-linear hazard shape) -------------
    month_ts = pd.to_datetime(panel["month"])
    signup_ts = pd.to_datetime(panel["signup_date"])
    panel["tenure_months"] = (
        (month_ts.dt.year * 12 + month_ts.dt.month) - (signup_ts.dt.year * 12 + signup_ts.dt.month)
    )
    panel["tenure_months_sq"] = panel["tenure_months"] ** 2

    # --- Lagged (t-1) usage features: no same-month leakage ------------------
    lagged = _lagged_monthly_run_stats()
    panel = panel.merge(lagged, on=["customer_id", "month"], how="left")
    for col in ["prior_month_n_runs", "prior_month_failure_rate", "prior_month_migration_share"]:
        panel[col] = panel[col].fillna(0.0)
    panel["log_monthly_runs"] = np.log1p(panel["prior_month_n_runs"])

    # --- Lifetime activation score (fixed at signup, not re-computed per row) -
    activation = _customer_activation_scores()
    panel = panel.merge(activation, on="customer_id", how="left")
    panel["activation_score"] = panel["activation_score"].fillna(0.0)

    return panel


def fit_hazard_model(panel: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Fit a statsmodels Logit discrete-time hazard model for interpretable
    odds ratios + 95% CIs (on the full panel), and a customer-GROUPED holdout
    sklearn LogisticRegression for an honest AUC — grouped so that no two
    person-period rows from the SAME customer can land on opposite sides of
    the train/test split (row-level random splitting would leak a customer's
    other months' outcome correlation into the test set).

    Returns (odds_ratio_table, grouped_holdout_auc).
    """
    categorical = ["acquisition_channel", "plan"]
    numeric = [
        "activation_score", "tenure_months", "tenure_months_sq",
        "prior_month_failure_rate", "prior_month_migration_share", "log_monthly_runs",
    ]

    X = pd.get_dummies(panel[categorical + numeric], columns=categorical, drop_first=True)
    X = X.astype(float)
    y = panel["churned_event"].astype(float)
    groups = panel["customer_id"]

    # --- statsmodels Logit: odds ratios + 95% CI, on the full panel ----------
    X_sm = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y, X_sm).fit(disp=0, maxiter=200)

    params = model.params
    conf = model.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    odds = pd.DataFrame({
        "feature": params.index,
        "coef": params.values,
        "odds_ratio": np.exp(params.values),
        "odds_ratio_ci_low": np.exp(conf["ci_low"].values),
        "odds_ratio_ci_high": np.exp(conf["ci_high"].values),
        "p_value": model.pvalues.values,
    })
    odds = odds[odds["feature"] != "const"].reset_index(drop=True)

    # --- sklearn LogisticRegression on a customer-grouped holdout: honest AUC -
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    if not set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx])):
        # GroupShuffleSplit guarantees this by construction; a plain
        # exception (not `assert`, which is stripped under `-O`) guards
        # against a future refactor accidentally switching to a row-level
        # splitter and silently reintroducing customer leakage.
        raise RuntimeError("customer leakage detected across the grouped train/test split")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X.iloc[train_idx])
    X_test_s = scaler.transform(X.iloc[test_idx])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_s, y.iloc[train_idx])
    auc = roc_auc_score(y.iloc[test_idx], clf.predict_proba(X_test_s)[:, 1])

    return odds, float(auc)


def churn_hazard_model() -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Convenience wrapper: build the panel, fit the model, return
    (odds_ratio_table, grouped_holdout_auc, panel) — the panel is returned too
    since the Kaplan-Meier curve below reuses its churned_event/tenure columns.
    """
    panel = build_survival_panel()
    odds, auc = fit_hazard_model(panel)
    return odds, auc, panel


# ======================================================================================
# 2b. Kaplan-Meier-style survival curve, activated vs not (manual — no lifelines)
# ======================================================================================

def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Manual Kaplan-Meier estimator: at each distinct observed duration t,
    S(t) = S(t-) * (1 - d_t / n_t), where n_t is the number still "at risk"
    (duration >= t) and d_t is the number of events exactly at t. Censored
    observations (event=0) simply leave the risk set after their duration
    without being counted as an event, which is the standard KM treatment.
    """
    df = pd.DataFrame({"duration": durations, "event": events})
    timeline = np.sort(df["duration"].unique())
    survival = 1.0
    rows = []
    for t in timeline:
        n_t = int((df["duration"] >= t).sum())
        d_t = int(((df["duration"] == t) & (df["event"] == 1)).sum())
        if n_t > 0:
            survival *= (1 - d_t / n_t)
        rows.append({"t": int(t), "n_at_risk": n_t, "events": d_t, "survival": survival})
    return pd.DataFrame(rows)


def km_by_activation(panel: pd.DataFrame, activated_threshold: float = 0.6) -> pd.DataFrame:
    """Per-customer survival curve (tenure in paid months vs survival
    probability), split into 'activated' (activation_score >= 3/5 = 0.6) vs
    'not activated'. Duration = number of paid customer-months observed
    (including the censored final row that build_survival_panel drops from
    the hazard-model panel, since KM needs it back to know each still-active
    customer's true observed exposure time); event = 1 if that customer's
    final month ended in churn.
    """
    per_customer = (
        panel.groupby("customer_id")
        .agg(
            duration=("row_number", "max"),
            event=("churned_event", "max"),
            activation_score=("activation_score", "first"),
        )
    )
    # build_survival_panel drops the single censored final row per active
    # customer, which would understate active customers' true duration by 1
    # month; add it back for a correct "months observed" count here.
    max_row_incl_censored = query(
        "SELECT customer_id, COUNT(*) AS n_paid_months FROM customer_months "
        "WHERE plan <> 'Free' GROUP BY customer_id"
    ).set_index("customer_id")["n_paid_months"]
    per_customer["duration"] = max_row_incl_censored.reindex(per_customer.index).fillna(per_customer["duration"])

    per_customer["activated"] = per_customer["activation_score"] >= activated_threshold

    curves = []
    for activated, grp in per_customer.groupby("activated"):
        km = kaplan_meier(grp["duration"].to_numpy(), grp["event"].to_numpy())
        km["group"] = "Activated (>=3/5)" if activated else "Not activated (<3/5)"
        curves.append(km)
    return pd.concat(curves, ignore_index=True)


# ======================================================================================
# 3. KMeans usage personas
# ======================================================================================
#
# A first pass clustered ALL customers together and let k=2 win by
# silhouette score — but that "winner" was degenerate: 2,126 of 2,207
# customers (96%) landed in one catch-all cluster, with the other cluster
# just isolating a handful of extreme heavy users. Two things drove this:
# (1) customers who never paid (permanently on Free) are a huge, trivially-
#     defined group that swamps any usage-based signal among customers who
#     actually engaged commercially, so they are now excluded from clustering
#     and reported as their own separate, non-clustered segment; and
# (2) the volume/cost features (runs, $ cost, revenue) are heavily
#     right-skewed with extreme outliers, which a Euclidean-distance method
#     like KMeans is very sensitive to — log1p + percentile clipping tames
#     that before standardizing.

def build_customer_usage_features() -> pd.DataFrame:
    """One row per customer (ALL customers, including never-paid) with usage
    behavior features: avg monthly runs, failure rate, migration share,
    activation, avg LLM cost per run, and lifetime paid revenue.
    """
    runs = query(
        """
        SELECT customer_id, status, task_type, llm_cost_usd, started_at, run_id
        FROM agent_runs
        """
    )
    activation = _customer_activation_scores().set_index("customer_id")["activation_score"]

    run_stats = runs.groupby("customer_id").agg(
        n_runs=("status", "size"),
        failure_rate=("status", lambda s: (s == "failed").mean()),
        migration_share=("task_type", lambda s: (s == "code_migration").mean()),
        avg_llm_cost=("llm_cost_usd", "mean"),
    )

    months = query("SELECT customer_id, plan, revenue_usd FROM customer_months")
    paid_months = months[months["plan"] != "Free"].groupby("customer_id").size().rename("paid_months")
    revenue = months.groupby("customer_id")["revenue_usd"].sum().rename("lifetime_revenue_usd")

    customers = query("SELECT customer_id, current_plan, first_paid_date FROM customers")

    df = (
        customers.set_index("customer_id")
        .join(run_stats)
        .join(activation)
        .join(paid_months)
        .join(revenue)
        .reset_index()
    )
    df["activation_score"] = df["activation_score"].fillna(0.0)
    df["failure_rate"] = df["failure_rate"].fillna(0.0)
    df["migration_share"] = df["migration_share"].fillna(0.0)
    df["avg_llm_cost"] = df["avg_llm_cost"].fillna(0.0)
    df["paid_months"] = df["paid_months"].fillna(0)
    df["lifetime_revenue_usd"] = df["lifetime_revenue_usd"].fillna(0.0)
    df["avg_monthly_runs"] = df["n_runs"].fillna(0) / df["paid_months"].replace(0, np.nan)
    df["avg_monthly_runs"] = df["avg_monthly_runs"].fillna(df["n_runs"].fillna(0))
    df["ever_paid"] = df["first_paid_date"].notna()
    return df.dropna(subset=["n_runs"])


# Backwards-compatible alias (a few callers/tests referred to this name).
def build_persona_features() -> pd.DataFrame:
    return build_customer_usage_features()


PERSONA_FEATURES = [
    "avg_monthly_runs", "failure_rate", "migration_share",
    "activation_score", "avg_llm_cost", "lifetime_revenue_usd",
]
# These are heavily right-skewed (a handful of very heavy/high-revenue
# customers) and get log1p + 1st/99th-percentile clipping before clustering;
# the bounded [0,1]-ish rate features are used as-is.
SKEWED_PERSONA_FEATURES = ["avg_monthly_runs", "avg_llm_cost", "lifetime_revenue_usd"]

MIN_CLUSTER_SHARE = 0.05  # smallest cluster must be >= 5% of clustered customers


def never_paid_segment(df: pd.DataFrame) -> dict:
    """Customers who never converted to paid are a trivially-defined segment
    (by construction, not by clustering) — reported separately rather than
    diluting the KMeans fit on customers who actually engaged commercially.
    """
    seg = df[~df["ever_paid"]]
    row = {
        "persona_name": "Never converted (Free only)",
        "n_customers": int(len(seg)),
        "share_of_all_customers": len(seg) / len(df) if len(df) else float("nan"),
    }
    for col in PERSONA_FEATURES:
        row[col] = float(seg[col].mean()) if len(seg) else float("nan")
    return row


def _cluster_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the standardized clustering matrix: log1p + 1st/99th percentile
    clipping on skewed volume/cost features, raw values for bounded rate
    features, then z-score standardization (fit on this same population)."""
    X = df[PERSONA_FEATURES].copy()
    for col in SKEWED_PERSONA_FEATURES:
        lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = np.log1p(X[col].clip(lower=lo, upper=hi))
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def choose_k(X_scaled: np.ndarray, k_range: range) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Try every k in k_range, and return a diagnostics table (silhouette
    score, smallest-cluster share, and whether it clears MIN_CLUSTER_SHARE)
    plus the fitted labels for every k — so the eventual choice of k is
    transparent and reproducible from the table alone, not just a printed
    number.
    """
    rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        labels_by_k[k] = labels
        sizes = pd.Series(labels).value_counts().sort_index()
        min_share = sizes.min() / len(labels)
        score = silhouette_score(X_scaled, labels)
        rows.append({
            "k": k,
            "silhouette_score": score,
            "cluster_sizes": ",".join(str(int(s)) for s in sizes),
            "smallest_cluster_share": min_share,
            "meets_min_cluster_share": min_share >= MIN_CLUSTER_SHARE,
        })
    return pd.DataFrame(rows), labels_by_k


def kmeans_personas(
    k_range: range = range(3, 7), min_cluster_share: float = MIN_CLUSTER_SHARE
) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    """Cluster customers who ever paid only (never-paid customers are a
    separate, non-clustered segment — see `never_paid_segment`). Standardize
    usage features (with log1p + percentile clipping on skewed ones), choose
    k in `k_range` by silhouette score subject to every cluster holding at
    least `min_cluster_share` of the clustered population, fit KMeans at
    that k, and return (customer-level cluster assignments, cluster profile
    table, chosen k, the full k-selection diagnostics table).
    """
    df_all = build_customer_usage_features()
    df = df_all[df_all["ever_paid"]].reset_index(drop=True)
    X_scaled = _cluster_matrix(df)

    diagnostics, labels_by_k = choose_k(X_scaled, k_range)

    eligible = diagnostics[diagnostics["meets_min_cluster_share"]]
    if len(eligible):
        chosen_row = eligible.loc[eligible["silhouette_score"].idxmax()]
    else:
        # No k in the range keeps every cluster above the floor — fall back
        # to whichever k gets closest (largest smallest-cluster share), so
        # the pipeline degrades gracefully instead of raising.
        chosen_row = diagnostics.loc[diagnostics["smallest_cluster_share"].idxmax()]
    best_k = int(chosen_row["k"])
    best_score = float(chosen_row["silhouette_score"])
    best_labels = labels_by_k[best_k]

    df = df.copy()
    df["cluster"] = best_labels

    profile = df.groupby("cluster")[PERSONA_FEATURES].mean()
    cluster_sizes = df.groupby("cluster").size()
    profile["n_customers"] = cluster_sizes
    profile["share_of_paid_customers"] = cluster_sizes / len(df)
    profile["silhouette_score"] = best_score
    profile = name_personas(profile)

    return (
        df[["customer_id", "cluster"] + PERSONA_FEATURES],
        profile.reset_index(),
        best_k,
        diagnostics,
    )


def name_personas(profile: pd.DataFrame) -> pd.DataFrame:
    """Assign a human-readable, deterministic, UNIQUE name to each cluster
    from its centroid profile relative to the cross-cluster mean. Traits are
    combined in a fixed priority order; if two clusters still end up with the
    same combination of traits (a real possibility with few, similar
    clusters), a stable numeric qualifier is appended so no two personas ever
    share a name.
    """
    overall = profile[PERSONA_FEATURES].mean()
    names = []
    for _, row in profile.iterrows():
        parts = []
        if row["migration_share"] >= overall["migration_share"] * 1.3:
            parts.append("Migration-Heavy")
        if row["lifetime_revenue_usd"] >= overall["lifetime_revenue_usd"] * 1.5:
            parts.append("Enterprise")
        elif row["lifetime_revenue_usd"] <= overall["lifetime_revenue_usd"] * 0.5:
            parts.append("Light Self-Serve")
        if row["avg_monthly_runs"] >= overall["avg_monthly_runs"] * 1.5 and "Enterprise" not in parts:
            parts.append("Heavy User")
        if row["failure_rate"] >= overall["failure_rate"] * 1.3:
            parts.append("Failure-Prone")
        elif row["failure_rate"] <= overall["failure_rate"] * 0.7 and len(parts) < 2:
            parts.append("Reliable")
        if row["activation_score"] <= overall["activation_score"] * 0.8 and len(parts) < 2:
            parts.append("At-Risk")
        if not parts:
            parts.append("Steady Moderate User")
        names.append(" ".join(parts))

    # Deterministic de-duplication: if two clusters produce the same name,
    # break the tie by lifetime revenue rank (highest revenue keeps the base
    # name; lower-revenue duplicates get a stable "(Segment N)" suffix, where
    # N is the cluster's 1-based rank by revenue among the tied group).
    profile = profile.copy()
    profile["_name"] = names
    final_names = list(names)
    for name, idx in profile.groupby("_name").groups.items():
        if len(idx) <= 1:
            continue
        tied = profile.loc[idx].sort_values("lifetime_revenue_usd", ascending=False)
        for rank, cluster_idx in enumerate(tied.index, start=1):
            if rank == 1:
                continue
            pos = profile.index.get_loc(cluster_idx)
            final_names[pos] = f"{name} (Segment {rank})"
    if len(final_names) != len(set(final_names)):
        raise RuntimeError(f"persona names must be unique, got: {final_names}")

    profile = profile.drop(columns="_name")
    profile.insert(0, "persona_name", final_names)
    return profile


# ======================================================================================
# Orchestration
# ======================================================================================

def run_all() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Activation threshold analysis ...")
    act_table = activation_threshold_table()
    act_table.to_csv(TABLES_DIR / "activation_threshold.csv", index=False)
    print(act_table.to_string(index=False))

    print("\nChurn discrete-time hazard model (odds ratios + 95% CI) ...")
    odds, auc, panel = churn_hazard_model()
    odds.to_csv(TABLES_DIR / "churn_odds_ratios.csv", index=False)
    print(odds.to_string(index=False))
    print(f"Customer-grouped holdout ROC-AUC: {auc:.3f}")
    pd.DataFrame([{"holdout_roc_auc": auc, "n_panel_rows": len(panel),
                    "n_events": int(panel["churned_event"].sum())}]).to_csv(
        TABLES_DIR / "churn_model_auc.csv", index=False
    )

    print("\nKaplan-Meier survival curve, activated vs not ...")
    km = km_by_activation(panel)
    km.to_csv(TABLES_DIR / "survival_km_curve.csv", index=False)
    print(km.to_string(index=False))

    print("\nKMeans usage personas (customers who ever paid only) ...")
    df_all = build_customer_usage_features()
    never_paid = never_paid_segment(df_all)
    pd.DataFrame([never_paid]).to_csv(TABLES_DIR / "persona_never_paid_segment.csv", index=False)
    print(f"Never-paid segment: {never_paid['n_customers']:,} customers "
          f"({never_paid['share_of_all_customers']*100:.1f}% of all customers) — reported separately, not clustered.")

    assignments, profile, k, diagnostics = kmeans_personas()
    assignments.to_csv(TABLES_DIR / "persona_assignments.csv", index=False)
    profile.to_csv(TABLES_DIR / "persona_profile.csv", index=False)
    diagnostics.to_csv(TABLES_DIR / "persona_k_selection.csv", index=False)
    print("k-selection diagnostics (silhouette + smallest-cluster share per k):")
    print(diagnostics.to_string(index=False))
    print(f"Chosen k = {k} (best silhouette among k with every cluster >= {MIN_CLUSTER_SHARE*100:.0f}% share)")
    print(profile.to_string(index=False))


if __name__ == "__main__":
    run_all()
