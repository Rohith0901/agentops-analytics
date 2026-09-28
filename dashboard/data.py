"""dashboard/data.py — cached, parameterized, read-only data access for the Streamlit app.

Security rules (see ARCHITECTURE.md "Testing & security requirements"):
- The SQLite connection is opened **read-only** via the `file:{path}?mode=ro` URI form
  (`uri=True`). Any attempted write raises `sqlite3.OperationalError`.
- No SQL is ever built from free text. Every query is a hard-coded, parameterized
  statement using `?` placeholders. Where a query needs an `IN (...)` list (e.g. for
  multi-select sidebar filters), the placeholders are built from the *count* of
  already allow-listed values — never by interpolating the values themselves.
- Every filter value coming from the UI is validated against an allow-list pulled
  fresh from the database before it is used in any query.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "agentops.db"

# Task types considered "simple" for routing/margin analyses (shared definition).
SIMPLE_TASKS = ("test_generation", "bug_fix", "refactor", "dependency_upgrade")


def db_exists() -> bool:
    """Whether the SQLite database file is present."""
    return DB_PATH.exists()


def get_read_only_conn() -> sqlite3.Connection:
    """Open a strictly read-only connection to data/agentops.db.

    Uses the `file:...?mode=ro` URI form with `uri=True`. Any write attempted
    on the returned connection raises `sqlite3.OperationalError: attempt to
    write a readonly database` — this is exercised directly in
    tests/test_dashboard.py.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(str(DB_PATH))
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _read_sql(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run one parameterized, hard-coded SELECT against the read-only connection."""
    conn = get_read_only_conn()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def _placeholders(values: tuple) -> str:
    """Build a `?, ?, ...` placeholder string sized to len(values)."""
    return ", ".join(["?"] * len(values))


# --------------------------------------------------------------------------
# Allow-lists (sidebar filter option lists, pulled straight from the DB)
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def get_filter_options() -> dict[str, list]:
    """Fetch the allow-listed values for every sidebar filter, straight from the DB.

    Every value a user later picks in the sidebar must appear in one of these
    lists before it is used in any query (see `_validate`).
    """
    months = _read_sql(
        "SELECT DISTINCT month FROM customer_months ORDER BY month"
    )["month"].tolist()
    plans = _read_sql("SELECT plan FROM plans ORDER BY plan")["plan"].tolist()
    regions = _read_sql(
        "SELECT DISTINCT region FROM customers ORDER BY region"
    )["region"].tolist()
    channels = _read_sql(
        "SELECT DISTINCT acquisition_channel FROM customers ORDER BY acquisition_channel"
    )["acquisition_channel"].tolist()
    industries = _read_sql(
        "SELECT DISTINCT industry FROM customers ORDER BY industry"
    )["industry"].tolist()
    models = _read_sql("SELECT model FROM models ORDER BY model")["model"].tolist()
    task_types = _read_sql(
        "SELECT DISTINCT task_type FROM agent_runs ORDER BY task_type"
    )["task_type"].tolist()
    return {
        "months": months,
        "plans": plans,
        "regions": regions,
        "channels": channels,
        "industries": industries,
        "models": models,
        "task_types": task_types,
    }


def _validate(selected: list[str], allowed: list[str]) -> list[str]:
    """Keep only values that are present in the allow-list; silently drop the rest."""
    allowed_set = set(allowed)
    return [v for v in selected if v in allowed_set]


@dataclass(frozen=True)
class Filters:
    """A validated, immutable bundle of sidebar filter selections.

    Every field is guaranteed (via `Filters.build`) to be a subset of the
    corresponding allow-list fetched from the database — values from a
    malformed or tampered widget state are dropped, never used directly.
    """

    months: tuple = field(default_factory=tuple)
    plans: tuple = field(default_factory=tuple)
    regions: tuple = field(default_factory=tuple)
    channels: tuple = field(default_factory=tuple)
    industries: tuple = field(default_factory=tuple)

    @staticmethod
    def build(
        months: list[str],
        plans: list[str],
        regions: list[str],
        channels: list[str],
        industries: list[str],
        options: dict[str, list],
    ) -> "Filters":
        return Filters(
            months=tuple(_validate(months, options["months"])),
            plans=tuple(_validate(plans, options["plans"])),
            regions=tuple(_validate(regions, options["regions"])),
            channels=tuple(_validate(channels, options["channels"])),
            industries=tuple(_validate(industries, options["industries"])),
        )


def _customer_filter_sql(f: Filters, alias: str = "c") -> tuple[str, tuple]:
    """Build a `WHERE`-clause fragment (and its params) for customer-level filters."""
    clauses = []
    params: list = []
    if f.regions:
        clauses.append(f"{alias}.region IN ({_placeholders(f.regions)})")
        params.extend(f.regions)
    if f.channels:
        clauses.append(f"{alias}.acquisition_channel IN ({_placeholders(f.channels)})")
        params.extend(f.channels)
    if f.industries:
        clauses.append(f"{alias}.industry IN ({_placeholders(f.industries)})")
        params.extend(f.industries)
    sql = (" AND " + " AND ".join(clauses)) if clauses else ""
    return sql, tuple(params)


# --------------------------------------------------------------------------
# Executive Overview
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def monthly_financials(f: Filters) -> pd.DataFrame:
    """Per-month MRR, revenue, paid customer count, and LLM cost (paid months only).

    MRR = SUM(monthly_fee_usd) of paid months (plan <> 'Free').
    Revenue = SUM(revenue_usd) of paid months.
    LLM cost is derived by joining agent_runs on (customer_id, month(started_at)).
    Gross margin = (revenue - llm_cost) / revenue.
    """
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    month_clause, month_params = "", ()
    if f.months:
        month_clause = f" AND cm.month IN ({_placeholders(f.months)})"
        month_params = f.months
    plan_clause, plan_params = "", ()
    if f.plans:
        plan_clause = f" AND cm.plan IN ({_placeholders(f.plans)})"
        plan_params = f.plans

    sql = f"""
        WITH billing AS (
            SELECT cm.month, cm.customer_id, cm.plan, cm.monthly_fee_usd, cm.revenue_usd
            FROM customer_months cm
            JOIN customers c ON c.customer_id = cm.customer_id
            WHERE cm.plan <> 'Free'{cust_sql}{month_clause}{plan_clause}
        ),
        cost AS (
            SELECT c.customer_id, substr(a.started_at, 1, 7) || '-01' AS month,
                   SUM(a.llm_cost_usd) AS llm_cost_usd
            FROM agent_runs a
            JOIN customers c ON c.customer_id = a.customer_id
            WHERE 1=1{cust_sql}
            GROUP BY c.customer_id, substr(a.started_at, 1, 7) || '-01'
        )
        SELECT
            b.month,
            COUNT(DISTINCT b.customer_id) AS active_paid_customers,
            SUM(b.monthly_fee_usd) AS mrr,
            SUM(b.revenue_usd) AS revenue,
            COALESCE(SUM(co.llm_cost_usd), 0) AS llm_cost
        FROM billing b
        LEFT JOIN cost co ON co.customer_id = b.customer_id AND co.month = b.month
        GROUP BY b.month
        ORDER BY b.month
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    params = cust_params + month_params + plan_params + cust_params
    df = _read_sql(sql, params)
    df["gross_margin"] = (df["revenue"] - df["llm_cost"]) / df["revenue"].replace(0, pd.NA)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def run_success_rate(f: Filters) -> pd.DataFrame:
    """Per-month run success rate (success / total runs), for filtered customers."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    month_clause, month_params = "", ()
    if f.months:
        month_clause = f" AND (substr(a.started_at, 1, 7) || '-01') IN ({_placeholders(f.months)})"
        month_params = f.months
    plan_clause, plan_params = "", ()
    if f.plans:
        plan_clause = f" AND a.plan_at_run IN ({_placeholders(f.plans)})"
        plan_params = f.plans

    sql = f"""
        SELECT substr(a.started_at, 1, 7) || '-01' AS month,
               AVG(CASE WHEN a.status = 'success' THEN 1.0 ELSE 0.0 END) AS success_rate,
               COUNT(*) AS n_runs
        FROM agent_runs a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE 1=1{cust_sql}{month_clause}{plan_clause}
        GROUP BY month
        ORDER BY month
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    params = cust_params + month_params + plan_params
    return _read_sql(sql, params)


@st.cache_data(ttl=600, show_spinner=False)
def nrr(f: Filters) -> float | None:
    """Net revenue retention: revenue_usd(month t) / revenue_usd(month t-1) for the
    cohort of customers paid & active in month t-1, averaged across consecutive
    month pairs in the filtered range. Returns None if not enough data."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        SELECT cm.customer_id, cm.month, cm.revenue_usd
        FROM customer_months cm
        JOIN customers c ON c.customer_id = cm.customer_id
        WHERE cm.plan <> 'Free'{cust_sql}
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    df = _read_sql(sql, cust_params)
    if df.empty:
        return None
    if f.months:
        df = df[df["month"].isin(f.months)]
    if df.empty:
        return None
    piv = df.pivot_table(index="customer_id", columns="month", values="revenue_usd", fill_value=0.0)
    months_sorted = sorted(piv.columns)
    if len(months_sorted) < 2:
        return None
    ratios = []
    for prev, cur in zip(months_sorted[:-1], months_sorted[1:]):
        cohort = piv.index[piv[prev] > 0]
        if len(cohort) == 0:
            continue
        prev_rev = piv.loc[cohort, prev].sum()
        cur_rev = piv.loc[cohort, cur].sum()
        if prev_rev > 0:
            ratios.append(cur_rev / prev_rev)
    if not ratios:
        return None
    return float(sum(ratios) / len(ratios))


# --------------------------------------------------------------------------
# Customers & Retention
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def cohort_retention(f: Filters) -> pd.DataFrame:
    """Logo-retention triangle: signup-month cohort x months-since-signup -> % still
    billed (any plan, including Free) that month."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        SELECT c.customer_id, substr(c.signup_date, 1, 7) || '-01' AS cohort_month,
               cm.month
        FROM customers c
        JOIN customer_months cm ON cm.customer_id = c.customer_id
        WHERE 1=1{cust_sql}
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    df = _read_sql(sql, cust_params)
    if df.empty:
        return df
    if f.months:
        df = df[df["month"].isin(f.months)]
    df["cohort_month"] = pd.to_datetime(df["cohort_month"])
    df["month"] = pd.to_datetime(df["month"])
    df["months_since_signup"] = (
        (df["month"].dt.year - df["cohort_month"].dt.year) * 12
        + (df["month"].dt.month - df["cohort_month"].dt.month)
    )
    cohort_sizes = df.groupby("cohort_month")["customer_id"].nunique()
    active = (
        df.groupby(["cohort_month", "months_since_signup"])["customer_id"]
        .nunique()
        .reset_index(name="active")
    )
    active["cohort_size"] = active["cohort_month"].map(cohort_sizes)
    active["retention"] = active["active"] / active["cohort_size"]
    return active


@st.cache_data(ttl=600, show_spinner=False)
def activation_by_customer(f: Filters) -> pd.DataFrame:
    """Per customer: activation flag (>=3/5 successes in first 5 runs by started_at),
    self-serve conversion flag, and whether the customer is still active (retention)."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        SELECT a.customer_id, a.started_at, a.status
        FROM agent_runs a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE 1=1{cust_sql}
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    runs = _read_sql(sql, cust_params)
    cust_sql2 = (
        f"SELECT customer_id, is_sales_led, first_paid_date, status, acquisition_channel "
        f"FROM customers c WHERE 1=1{cust_sql}"  # nosec B608 - identifiers/clauses hardcoded; values via ? params
    )
    cust = _read_sql(cust_sql2, cust_params)
    if runs.empty or cust.empty:
        return pd.DataFrame()

    runs = runs.sort_values(["customer_id", "started_at"])
    first5 = runs.groupby("customer_id").head(5)
    agg = first5.groupby("customer_id").agg(
        n_runs=("status", "size"),
        n_success=("status", lambda s: (s == "success").sum()),
    )
    agg["activated"] = (agg["n_success"] / agg["n_runs"].clip(lower=1)) >= 0.6
    agg = agg.reset_index()

    out = cust.merge(agg[["customer_id", "activated"]], on="customer_id", how="left")
    out["activated"] = out["activated"].fillna(False)
    out["self_serve_converted"] = out["first_paid_date"].notna() & (out["is_sales_led"] == 0)
    out["retained"] = out["status"] == "active"
    return out


@st.cache_data(ttl=600, show_spinner=False)
def channel_summary(f: Filters) -> pd.DataFrame:
    """Per acquisition channel: customer count, self-serve conversion rate, retention rate."""
    act = activation_by_customer(f)
    if act.empty:
        return act
    selfserve = act[act["is_sales_led"] == 0]
    grp = selfserve.groupby("acquisition_channel").agg(
        n_customers=("customer_id", "nunique"),
        conversion_rate=("self_serve_converted", "mean"),
        retention_rate=("retained", "mean"),
        activation_rate=("activated", "mean"),
    ).reset_index()
    return grp


# --------------------------------------------------------------------------
# LLM Performance
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def model_task_success(f: Filters) -> pd.DataFrame:
    """Success rate and run count by model x task_type."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    month_clause, month_params = "", ()
    if f.months:
        month_clause = f" AND (substr(a.started_at, 1, 7) || '-01') IN ({_placeholders(f.months)})"
        month_params = f.months
    sql = f"""
        SELECT a.model, a.task_type,
               AVG(CASE WHEN a.status = 'success' THEN 1.0 ELSE 0.0 END) AS success_rate,
               AVG(a.llm_cost_usd) AS avg_cost,
               COUNT(*) AS n_runs
        FROM agent_runs a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE 1=1{cust_sql}{month_clause}
        GROUP BY a.model, a.task_type
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    return _read_sql(sql, cust_params + month_params)


@st.cache_data(ttl=600, show_spinner=False)
def cost_per_successful_run(f: Filters) -> pd.DataFrame:
    """Cost per *successful* run, by model: SUM(llm_cost_usd) / COUNT(successes)."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        SELECT a.model,
               SUM(a.llm_cost_usd) AS total_cost,
               SUM(CASE WHEN a.status = 'success' THEN 1 ELSE 0 END) AS n_success,
               COUNT(*) AS n_runs
        FROM agent_runs a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE 1=1{cust_sql}
        GROUP BY a.model
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    df = _read_sql(sql, cust_params)
    df["cost_per_success"] = df["total_cost"] / df["n_success"].replace(0, pd.NA)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def migration_pairs(f: Filters) -> pd.DataFrame:
    """Migration-pair (source_stack -> target_stack) success/cost table, migrations only."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        SELECT a.source_stack, a.target_stack,
               AVG(CASE WHEN a.status = 'success' THEN 1.0 ELSE 0.0 END) AS success_rate,
               AVG(a.llm_cost_usd) AS avg_cost,
               COUNT(*) AS n_runs
        FROM agent_runs a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE a.task_type = 'code_migration'{cust_sql}
        GROUP BY a.source_stack, a.target_stack
        ORDER BY n_runs DESC
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    return _read_sql(sql, cust_params)


@st.cache_data(ttl=600, show_spinner=False)
def failure_cost(f: Filters) -> dict:
    """Total LLM $ spent on failed/partial runs (the 'failure tax')."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        SELECT
            SUM(CASE WHEN a.status != 'success' THEN a.llm_cost_usd ELSE 0 END) AS failed_cost,
            SUM(a.llm_cost_usd) AS total_cost,
            SUM(CASE WHEN a.status != 'success' THEN 1 ELSE 0 END) AS n_failed,
            COUNT(*) AS n_runs
        FROM agent_runs a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE 1=1{cust_sql}
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    df = _read_sql(sql, cust_params)
    if df.empty or df["total_cost"].iloc[0] in (None, 0):
        return {"failed_cost": 0.0, "total_cost": 0.0, "share": 0.0, "n_failed": 0, "n_runs": 0}
    row = df.iloc[0]
    return {
        "failed_cost": float(row["failed_cost"] or 0.0),
        "total_cost": float(row["total_cost"] or 0.0),
        "share": float((row["failed_cost"] or 0.0) / row["total_cost"]) if row["total_cost"] else 0.0,
        "n_failed": int(row["n_failed"] or 0),
        "n_runs": int(row["n_runs"] or 0),
    }


# --------------------------------------------------------------------------
# Pricing & Unit Economics
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def margin_by_plan(f: Filters) -> pd.DataFrame:
    """Gross margin by plan over paid months: (revenue - llm_cost) / revenue."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        WITH billing AS (
            SELECT cm.customer_id, cm.month, cm.plan, cm.revenue_usd
            FROM customer_months cm
            JOIN customers c ON c.customer_id = cm.customer_id
            WHERE cm.plan <> 'Free'{cust_sql}
        ),
        cost AS (
            SELECT c.customer_id, substr(a.started_at, 1, 7) || '-01' AS month,
                   SUM(a.llm_cost_usd) AS llm_cost_usd
            FROM agent_runs a
            JOIN customers c ON c.customer_id = a.customer_id
            WHERE 1=1{cust_sql}
            GROUP BY c.customer_id, substr(a.started_at, 1, 7) || '-01'
        )
        SELECT b.plan,
               SUM(b.revenue_usd) AS revenue,
               COALESCE(SUM(co.llm_cost_usd), 0) AS llm_cost
        FROM billing b
        LEFT JOIN cost co ON co.customer_id = b.customer_id AND co.month = b.month
        GROUP BY b.plan
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    df = _read_sql(sql, cust_params + cust_params)
    df["gross_margin"] = (df["revenue"] - df["llm_cost"]) / df["revenue"].replace(0, pd.NA)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def cost_revenue_per_credit_by_task(f: Filters) -> pd.DataFrame:
    """Cost-per-credit vs revenue-per-credit by task_type.

    Revenue-per-credit is approximated using each run's plan-at-run blended
    rate: revenue_usd / credits_used for that customer-month, applied evenly
    across that month's runs is over-complex; instead we use the plan's
    monthly_fee/included_credits as the "list" revenue-per-credit reference,
    and actual observed cost-per-credit from agent_runs.
    """
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        SELECT a.task_type,
               SUM(a.llm_cost_usd) AS total_cost,
               SUM(a.credits_charged) AS total_credits,
               COUNT(*) AS n_runs
        FROM agent_runs a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE 1=1{cust_sql}
        GROUP BY a.task_type
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    df = _read_sql(sql, cust_params)
    df["cost_per_credit"] = df["total_cost"] / df["total_credits"].replace(0, pd.NA)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def team_revenue_per_credit(f: Filters) -> float | None:
    """Team plan's revenue-per-credit reference line: SUM(revenue_usd)/SUM(credits_used)
    over Team-plan paid months."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        SELECT SUM(cm.revenue_usd) AS revenue, SUM(cm.credits_used) AS credits
        FROM customer_months cm
        JOIN customers c ON c.customer_id = cm.customer_id
        WHERE cm.plan = 'Team'{cust_sql}
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    df = _read_sql(sql, cust_params)
    if df.empty or not df["credits"].iloc[0]:
        return None
    return float(df["revenue"].iloc[0] / df["credits"].iloc[0])


@st.cache_data(ttl=600, show_spinner=False)
def customer_margin_deciles(f: Filters) -> pd.DataFrame:
    """Per-customer gross margin (paid months only), bucketed into deciles."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    sql = f"""
        WITH billing AS (
            SELECT cm.customer_id, cm.month, cm.revenue_usd
            FROM customer_months cm
            JOIN customers c ON c.customer_id = cm.customer_id
            WHERE cm.plan <> 'Free'{cust_sql}
        ),
        cost AS (
            SELECT c.customer_id, substr(a.started_at, 1, 7) || '-01' AS month,
                   SUM(a.llm_cost_usd) AS llm_cost_usd
            FROM agent_runs a
            JOIN customers c ON c.customer_id = a.customer_id
            WHERE 1=1{cust_sql}
            GROUP BY c.customer_id, substr(a.started_at, 1, 7) || '-01'
        )
        SELECT b.customer_id,
               SUM(b.revenue_usd) AS revenue,
               COALESCE(SUM(co.llm_cost_usd), 0) AS llm_cost
        FROM billing b
        LEFT JOIN cost co ON co.customer_id = b.customer_id AND co.month = b.month
        GROUP BY b.customer_id
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    df = _read_sql(sql, cust_params + cust_params)
    df = df[df["revenue"] > 0].copy()
    if df.empty:
        return df
    df["gross_margin"] = (df["revenue"] - df["llm_cost"]) / df["revenue"]
    df["decile"] = pd.qcut(df["gross_margin"].rank(method="first"), 10, labels=False) + 1
    return df


# --------------------------------------------------------------------------
# Pricing Experiment
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def experiment_conversion(f: Filters) -> pd.DataFrame:
    """Per-variant conversion: first_paid_date within 30 days of assigned_at."""
    sql = """
        SELECT ea.customer_id, ea.variant, ea.assigned_at, c.first_paid_date
        FROM experiment_assignments ea
        JOIN customers c ON c.customer_id = ea.customer_id
        WHERE ea.experiment_name = 'pro_price_2026Q2'
    """
    df = _read_sql(sql)
    if df.empty:
        return df
    assigned = pd.to_datetime(df["assigned_at"])
    paid = pd.to_datetime(df["first_paid_date"])
    df["converted"] = paid.notna() & ((paid - assigned).dt.days.between(0, 30))
    return df.groupby("variant").agg(
        n=("customer_id", "nunique"), conversions=("converted", "sum")
    ).reset_index()


# --------------------------------------------------------------------------
# Routing Simulator
# --------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def simple_task_frontier_stats(f: Filters) -> pd.DataFrame:
    """Observed stratified (task_type) success rate & avg cost for frontier-large vs
    balanced-medium runs on simple tasks -- inputs for the routing simulator."""
    cust_sql, cust_params = _customer_filter_sql(f, alias="c")
    placeholders = _placeholders(SIMPLE_TASKS)
    sql = f"""
        SELECT a.task_type, a.model,
               AVG(CASE WHEN a.status = 'success' THEN 1.0 ELSE 0.0 END) AS success_rate,
               AVG(a.llm_cost_usd) AS avg_cost,
               COUNT(*) AS n_runs
        FROM agent_runs a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE a.task_type IN ({placeholders}) AND a.model IN ('frontier-large', 'balanced-medium'){cust_sql}
        GROUP BY a.task_type, a.model
    """  # nosec B608 - identifiers/clauses are hardcoded; all values pass through ? params
    return _read_sql(sql, SIMPLE_TASKS + cust_params)
