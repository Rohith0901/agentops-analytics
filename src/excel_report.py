"""excel_report.py — builds reports/AgentOps_Report.xlsx.

A recruiter-facing Excel workbook for the (synthetic) CodeShift dataset:
KPI dashboard with native charts, monthly/plan/model-task breakdowns, a
customer-level summary table, a live-formula pricing simulator, and a
pivot-ready flat fact table. See ARCHITECTURE.md for the shared metric
definitions this module recomputes from `data/agentops.db`.

Run directly: `.venv/bin/python src/excel_report.py [--out PATH]`.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, NamedStyle, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo

from db import ROOT, get_conn

# --------------------------------------------------------------------- constants

SIMPLE_TASKS = ["test_generation", "bug_fix", "refactor", "dependency_upgrade"]
ALL_PLANS = ["Free", "Pro", "Team", "Enterprise"]
PAID_PLANS = ["Pro", "Team", "Enterprise"]

DEFAULT_OUT_PATH = ROOT / "reports" / "AgentOps_Report.xlsx"

# --------------------------------------------------------------------- styling

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True)
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
INPUT_FONT = Font(color="7F6000", bold=True)
LABEL_FONT = Font(bold=True)
TITLE_FONT = Font(size=16, bold=True, color="1F3864")
SUBTITLE_FONT = Font(size=10, italic=True, color="595959")
NOTE_FONT = Font(size=9, italic=True, color="808080")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
MONEY_FMT = '"$"#,##0'
MONEY2_FMT = '"$"#,##0.00'
PCT_FMT = "0.0%"
INT_FMT = "#,##0"


# --------------------------------------------------------------------- security guard

_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_cell(value):
    """Neutralise CSV/formula injection in a single data-derived cell value.

    Any string that *starts with* one of `= + - @ \\t \\r` gets a leading
    apostrophe so Excel treats it as literal text instead of a formula.
    Non-string values (numbers, None/NaN, Excel formula strings we author
    ourselves and pass in separately) are returned unchanged. This is used
    for every cell whose text originates in the database — never for the
    author-written formulas in the Pricing Simulator sheet.
    """
    if not isinstance(value, str):
        return value
    if value.startswith(_DANGEROUS_PREFIXES):
        return "'" + value
    return value


def sanitize_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a copy of `df` with `sanitize_cell` applied to text columns."""
    df = df.copy()
    cols = columns if columns is not None else list(df.select_dtypes(include=["object", "string"]).columns)
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(sanitize_cell)
    return df


# --------------------------------------------------------------------- data fetchers
#
# Every metric here is computed straight from `data/agentops.db` via
# parameterized SQL (never string-built from external input), matching the
# shared metric definitions in ARCHITECTURE.md:
#   - LLM cost per customer-month = SUM(agent_runs.llm_cost_usd) grouped by
#     customer_id and substr(started_at,1,7)||'-01' = customer_months.month.
#   - Paid month = plan <> 'Free'. MRR = SUM(monthly_fee_usd) of paid months;
#     revenue = SUM(revenue_usd). Gross margin = (revenue - LLM cost) /
#     revenue over paid months.
#   - Activation = success share of a customer's first 5 runs by started_at;
#     activated = >= 3/5.
#   - Simple tasks = test_generation, bug_fix, refactor, dependency_upgrade.


# NOTE on the repeated "WITH run_cost AS (...)" block below: every query
# that needs LLM cost per (customer_id, month) writes out the same CTE text
# in full, as a *fully static* triple-quoted string literal — no f-string,
# `.format()`, `%`, or `+` concatenation touches these queries at all, per
# ARCHITECTURE.md's SQL-safety rule ("never build SQL with f-strings/%/+").
# That costs a little duplication; the alternative (interpolating a shared
# helper's return value into an f-string) is exactly the pattern that rule
# forbids, even though the interpolated text itself is 100% hardcoded.


def fetch_monthly_kpis(conn) -> pd.DataFrame:
    sql = """
        WITH run_cost AS (
            SELECT customer_id, substr(started_at, 1, 7) || '-01' AS month,
                   SUM(llm_cost_usd) AS llm_cost
            FROM agent_runs
            GROUP BY customer_id, substr(started_at, 1, 7)
        )
        SELECT
            cm.month,
            SUM(CASE WHEN cm.plan <> 'Free' THEN cm.monthly_fee_usd ELSE 0 END) AS mrr,
            SUM(CASE WHEN cm.plan <> 'Free' THEN cm.revenue_usd ELSE 0 END) AS paid_revenue,
            SUM(CASE WHEN cm.plan <> 'Free' THEN COALESCE(rc.llm_cost, 0) ELSE 0 END) AS paid_llm_cost,
            SUM(COALESCE(rc.llm_cost, 0)) AS total_llm_cost,
            COUNT(DISTINCT cm.customer_id) AS active_customers,
            SUM(CASE WHEN cm.plan <> 'Free' THEN 1 ELSE 0 END) AS paid_customers
        FROM customer_months cm
        LEFT JOIN run_cost rc ON rc.customer_id = cm.customer_id AND rc.month = cm.month
        GROUP BY cm.month
        ORDER BY cm.month
    """
    df = pd.read_sql_query(sql, conn)
    df["gross_margin"] = (df["paid_revenue"] - df["paid_llm_cost"]) / df["paid_revenue"].replace(0, pd.NA)
    return df


def fetch_plan_economics(conn) -> pd.DataFrame:
    sql = """
        WITH run_cost AS (
            SELECT customer_id, substr(started_at, 1, 7) || '-01' AS month,
                   SUM(llm_cost_usd) AS llm_cost
            FROM agent_runs
            GROUP BY customer_id, substr(started_at, 1, 7)
        )
        SELECT
            cm.plan,
            COUNT(DISTINCT cm.customer_id) AS customers,
            COUNT(*) AS customer_months,
            SUM(cm.revenue_usd) AS revenue,
            SUM(COALESCE(rc.llm_cost, 0)) AS llm_cost
        FROM customer_months cm
        LEFT JOIN run_cost rc ON rc.customer_id = cm.customer_id AND rc.month = cm.month
        GROUP BY cm.plan
    """
    df = pd.read_sql_query(sql, conn)
    df["gross_margin"] = (df["revenue"] - df["llm_cost"]) / df["revenue"].replace(0, pd.NA)
    order = {p: i for i, p in enumerate(ALL_PLANS)}
    df["_order"] = df["plan"].map(order)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def fetch_model_task(conn) -> pd.DataFrame:
    sql = """
        SELECT
            model,
            task_type,
            COUNT(*) AS n_runs,
            AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0 END) AS success_rate,
            AVG(llm_cost_usd) AS avg_cost_per_run,
            SUM(llm_cost_usd) * 1.0
                / NULLIF(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0) AS cost_per_success,
            AVG(latency_sec) AS avg_latency_sec
        FROM agent_runs
        GROUP BY model, task_type
        ORDER BY model, task_type
    """
    return pd.read_sql_query(sql, conn)


def fetch_customers(conn) -> pd.DataFrame:
    customers = pd.read_sql_query(
        """
        SELECT customer_id, company_name, industry, company_size, region,
               acquisition_channel, current_plan, status, signup_date, churn_date
        FROM customers
        """,
        conn,
    )
    revenue = pd.read_sql_query(
        "SELECT customer_id, SUM(revenue_usd) AS total_revenue FROM customer_months GROUP BY customer_id",
        conn,
    )
    runs = pd.read_sql_query(
        """
        SELECT customer_id, COUNT(*) AS total_runs,
               AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0 END) AS success_rate,
               SUM(llm_cost_usd) AS total_llm_cost
        FROM agent_runs
        GROUP BY customer_id
        """,
        conn,
    )
    # Activation: success share of each customer's first 5 runs by started_at.
    activation = pd.read_sql_query(
        """
        WITH ranked AS (
            SELECT customer_id, status,
                   ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY started_at) AS rn
            FROM agent_runs
        )
        SELECT customer_id,
               AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0 END) AS first5_success_rate
        FROM ranked
        WHERE rn <= 5
        GROUP BY customer_id
        """,
        conn,
    )
    activation["activated"] = activation["first5_success_rate"] >= 0.6

    df = (
        customers.merge(revenue, on="customer_id", how="left")
        .merge(runs, on="customer_id", how="left")
        .merge(activation, on="customer_id", how="left")
    )
    for col in ["total_revenue", "total_runs", "success_rate", "total_llm_cost", "first5_success_rate"]:
        df[col] = df[col].fillna(0)
    df["activated"] = df["activated"].fillna(False)
    df["gross_margin"] = (df["total_revenue"] - df["total_llm_cost"]) / df["total_revenue"].replace(0, pd.NA)
    return df.sort_values("customer_id").reset_index(drop=True)


def fetch_pivot_data(conn) -> pd.DataFrame:
    """Flat customer-month fact table (customer attributes denormalised in) for PivotTables."""
    cm = pd.read_sql_query(
        """
        SELECT customer_id, month, plan, monthly_fee_usd, included_credits, credits_used,
               overage_credits, overage_revenue_usd, revenue_usd
        FROM customer_months
        """,
        conn,
    )
    run_cost = pd.read_sql_query(
        """
        SELECT customer_id, substr(started_at, 1, 7) || '-01' AS month,
               SUM(llm_cost_usd) AS llm_cost
        FROM agent_runs
        GROUP BY customer_id, substr(started_at, 1, 7)
        """,
        conn,
    )
    cm = cm.merge(run_cost, on=["customer_id", "month"], how="left")
    cm["llm_cost"] = cm["llm_cost"].fillna(0.0)
    cm["gross_margin"] = (cm["revenue_usd"] - cm["llm_cost"]) / cm["revenue_usd"].replace(0, pd.NA)

    attrs = pd.read_sql_query(
        "SELECT customer_id, industry, company_size, region, acquisition_channel FROM customers",
        conn,
    )
    df = cm.merge(attrs, on="customer_id", how="left")
    cols = [
        "customer_id", "month", "plan", "industry", "company_size", "region",
        "acquisition_channel", "monthly_fee_usd", "included_credits", "credits_used",
        "overage_credits", "overage_revenue_usd", "revenue_usd", "llm_cost", "gross_margin",
    ]
    return df[cols].sort_values(["customer_id", "month"]).reset_index(drop=True)


def fetch_simulator_baseline(conn) -> dict:
    """Every baseline number the Pricing Simulator sheet's formulas reference.

    All figures are *averages per customer-month* (or per-run, or global
    factors) so the simulator's formulas simply multiply/adjust these by
    live input cells — see `build_pricing_simulator_sheet` for the formula
    wiring.
    """
    # PAID_PLANS / SIMPLE_TASKS are hard-coded module-level constants (never
    # user input); every "IN (...)" below uses a *literal* "?" placeholder
    # list sized to match them, with the actual values bound via `params=`.
    # This keeps every query below a fully static string (no f-string/%/+
    # SQL construction anywhere), matching ARCHITECTURE.md's SQL-safety rule.
    if len(PAID_PLANS) != 3:
        raise ValueError("literal '?,?,?' placeholders below assume 3 paid plans")
    if len(SIMPLE_TASKS) != 4:
        raise ValueError("literal '?,?,?,?' placeholders below assume 4 simple task types")

    plans = pd.read_sql_query(
        "SELECT plan, monthly_fee_usd, included_credits, overage_usd_per_credit FROM plans", conn
    )
    plans = plans.set_index("plan")

    active_customers = pd.read_sql_query(
        """
        SELECT current_plan AS plan, COUNT(*) AS n
        FROM customers
        WHERE status = 'active' AND current_plan IN (?, ?, ?)
        GROUP BY current_plan
        """,
        conn,
        params=PAID_PLANS,
    ).set_index("plan")["n"]

    n_months = pd.read_sql_query(
        """
        SELECT plan, COUNT(*) AS n_months
        FROM customer_months
        WHERE plan IN (?, ?, ?)
        GROUP BY plan
        """,
        conn,
        params=PAID_PLANS,
    ).set_index("plan")["n_months"]

    avg_llm_cost = pd.read_sql_query(
        """
        WITH run_cost AS (
            SELECT customer_id, substr(started_at, 1, 7) || '-01' AS month,
                   SUM(llm_cost_usd) AS llm_cost
            FROM agent_runs
            GROUP BY customer_id, substr(started_at, 1, 7)
        )
        SELECT cm.plan, AVG(COALESCE(rc.llm_cost, 0)) AS avg_llm_cost
        FROM customer_months cm
        LEFT JOIN run_cost rc ON rc.customer_id = cm.customer_id AND rc.month = cm.month
        WHERE cm.plan IN (?, ?, ?)
        GROUP BY cm.plan
        """,
        conn,
        params=PAID_PLANS,
    ).set_index("plan")["avg_llm_cost"]

    avg_frontier_simple_cost = pd.read_sql_query(
        """
        WITH sf AS (
            SELECT customer_id, substr(started_at, 1, 7) || '-01' AS month, SUM(llm_cost_usd) AS cost
            FROM agent_runs
            WHERE model = 'frontier-large' AND task_type IN (?, ?, ?, ?)
            GROUP BY customer_id, substr(started_at, 1, 7)
        )
        SELECT cm.plan, AVG(COALESCE(sf.cost, 0)) AS avg_frontier_simple_cost
        FROM customer_months cm
        LEFT JOIN sf ON sf.customer_id = cm.customer_id AND sf.month = cm.month
        WHERE cm.plan IN (?, ?, ?)
        GROUP BY cm.plan
        """,
        conn,
        params=SIMPLE_TASKS + PAID_PLANS,
    ).set_index("plan")["avg_frontier_simple_cost"]

    runs_per_plan_task = pd.read_sql_query(
        """
        SELECT cm.plan, ar.task_type, COUNT(*) AS n_runs
        FROM agent_runs ar
        JOIN customer_months cm
            ON cm.customer_id = ar.customer_id AND cm.month = substr(ar.started_at, 1, 7) || '-01'
        WHERE cm.plan IN (?, ?, ?)
        GROUP BY cm.plan, ar.task_type
        """,
        conn,
        params=PAID_PLANS,
    )
    task_types = pd.read_sql_query("SELECT DISTINCT task_type FROM agent_runs ORDER BY task_type", conn)[
        "task_type"
    ].tolist()
    avg_runs_per_task = (
        runs_per_plan_task.pivot(index="plan", columns="task_type", values="n_runs")
        .reindex(index=PAID_PLANS, columns=task_types)
        .fillna(0)
        .astype(float)
    )
    for plan in PAID_PLANS:
        avg_runs_per_task.loc[plan] = avg_runs_per_task.loc[plan] / n_months.get(plan, 1)

    credits_per_task = pd.read_sql_query(
        "SELECT task_type, AVG(credits_charged) AS avg_credits FROM agent_runs GROUP BY task_type", conn
    ).set_index("task_type")["avg_credits"].reindex(task_types)

    reduction_row = pd.read_sql_query(
        """
        SELECT model, AVG(llm_cost_usd) AS avg_cost
        FROM agent_runs
        WHERE model IN ('frontier-large', 'balanced-medium') AND task_type IN (?, ?, ?, ?)
        GROUP BY model
        """,
        conn,
        params=SIMPLE_TASKS,
    ).set_index("model")["avg_cost"]
    reduction_factor = 1 - (reduction_row["balanced-medium"] / reduction_row["frontier-large"])

    elasticity_pro = fetch_pro_price_elasticity(conn)

    actual_margin = fetch_plan_economics(conn).set_index("plan")["gross_margin"].reindex(PAID_PLANS)

    return {
        "plans": plans,  # DataFrame indexed by plan: monthly_fee_usd, included_credits, overage_usd_per_credit
        "active_customers": active_customers,  # Series indexed by plan
        "avg_llm_cost": avg_llm_cost,  # Series indexed by plan
        "avg_frontier_simple_cost": avg_frontier_simple_cost,  # Series indexed by plan
        "avg_runs_per_task": avg_runs_per_task,  # DataFrame: index=plan, columns=task_type
        "credits_per_task": credits_per_task,  # Series indexed by task_type
        "task_types": task_types,
        "reduction_factor": float(reduction_factor),
        "elasticity_pro": elasticity_pro,  # float: arc elasticity of 30-day conversion, $29 vs $39 (A/B test)
        "actual_margin": actual_margin,  # Series indexed by plan: actual historical gross margin from the real data
    }


def fetch_pro_price_elasticity(conn) -> float:
    """Arc elasticity of 30-day paid conversion, from the Pro $29-vs-$39 A/B test.

    Joins `experiment_assignments` (control shown $29, treatment shown $39)
    to `customers.first_paid_date` and defines "converted" as first_paid_date
    falling within 30 days of `assigned_at`. Arc elasticity uses the midpoint
    (Marshallian) formula so it's symmetric in the direction of the price
    change:

        E = [(Q2 - Q1) / ((Q1 + Q2) / 2)] / [(P2 - P1) / ((P1 + P2) / 2)]

    where (P1, Q1) = (29, control conversion) and (P2, Q2) = (39, treatment
    conversion). This value seeds the Pricing Simulator's default "price
    elasticity of paid customers" input for Pro; Team/Enterprise reuse it as
    a labelled assumption since there's no A/B test for those plans.
    """
    df = pd.read_sql_query(
        """
        SELECT ea.variant, ea.pro_price_shown, ea.assigned_at, c.first_paid_date
        FROM experiment_assignments ea
        JOIN customers c ON c.customer_id = ea.customer_id
        WHERE ea.experiment_name = 'pro_price_2026Q2'
        """,
        conn,
    )
    df["assigned_at"] = pd.to_datetime(df["assigned_at"])
    df["first_paid_date"] = pd.to_datetime(df["first_paid_date"])
    days_to_paid = (df["first_paid_date"] - df["assigned_at"]).dt.days
    df["converted_30d"] = df["first_paid_date"].notna() & (days_to_paid >= 0) & (days_to_paid <= 30)

    rates = df.groupby("variant")["converted_30d"].mean()
    q1, q2 = float(rates["control"]), float(rates["treatment"])
    p1, p2 = 29.0, 39.0

    delta_q = q2 - q1
    avg_q = (q1 + q2) / 2
    delta_p = p2 - p1
    avg_p = (p1 + p2) / 2
    return (delta_q / avg_q) / (delta_p / avg_p)


# --------------------------------------------------------------------- sheet helpers


def _set_col_widths(ws, widths: dict[str, float]) -> None:
    for col, width in widths.items():
        ws.column_dimensions[col].width = width


def _write_header_row(ws, row: int, headers: list[str], start_col: int = 1) -> None:
    for i, h in enumerate(headers):
        cell = ws.cell(row=row, column=start_col + i, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def write_dataframe(
    ws, df: pd.DataFrame, start_row: int, start_col: int = 1,
    sanitize_columns: list[str] | None = None, number_formats: dict[str, str] | None = None,
) -> tuple[int, int]:
    """Write a DataFrame's header + rows starting at (start_row, start_col).

    Text columns named in `sanitize_columns` (default: all object columns)
    are passed through `sanitize_cell`. Returns (last_row, last_col).
    """
    df = sanitize_frame(df, sanitize_columns)
    headers = list(df.columns)
    _write_header_row(ws, start_row, headers, start_col)
    number_formats = number_formats or {}
    for r, (_, row) in enumerate(df.iterrows(), start=start_row + 1):
        for c, col in enumerate(headers, start=start_col):
            value = row[col]
            if pd.isna(value):
                value = None
            cell = ws.cell(row=r, column=c, value=value)
            cell.border = BORDER
            if col in number_formats:
                cell.number_format = number_formats[col]
    last_row = start_row + len(df)
    last_col = start_col + len(headers) - 1
    return last_row, last_col


def add_excel_table(ws, name: str, first_row: int, first_col: int, last_row: int, last_col: int,
                     style: str = "TableStyleMedium9") -> None:
    ref = (
        f"{get_column_letter(first_col)}{first_row}:{get_column_letter(last_col)}{last_row}"
    )
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name=style, showRowStripes=True, showFirstColumn=False, showLastColumn=False
    )
    ws.add_table(table)


def color_scale(ws, cell_range: str, low="F8696B", mid="FFEB84", high="63BE7B") -> None:
    rule = ColorScaleRule(
        start_type="min", start_color=low,
        mid_type="percentile", mid_value=50, mid_color=mid,
        end_type="max", end_color=high,
    )
    ws.conditional_formatting.add(cell_range, rule)


def input_cell(ws, row: int, col: int, value, number_format: str | None = None, comment: str | None = None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.fill = INPUT_FILL
    cell.font = INPUT_FONT
    cell.border = BORDER
    if number_format:
        cell.number_format = number_format
    return cell


# --------------------------------------------------------------------- README sheet


def build_readme_sheet(wb: Workbook) -> None:
    ws = wb.active
    ws.title = "README"
    _set_col_widths(ws, {"A": 4, "B": 110})

    ws["A1"] = "AgentOps Analytics — Excel Report"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:B1")

    ws["A3"] = (
        "Purpose: a recruiter-facing summary of CodeShift's (fictional AI coding-agent SaaS) "
        "customer, LLM-usage, and pricing data — built to demonstrate SQL/Python analysis, "
        "Excel modelling, and Power BI / Tableau readiness for an Analytics Intern role."
    )
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[3].height = 45
    ws.merge_cells("A3:B3")

    ws["A5"] = "IMPORTANT: All data in this workbook is SYNTHETIC (generated by src/generate_data.py)."
    ws["A5"].font = Font(bold=True, color="C00000")
    ws.merge_cells("A5:B5")

    ws["A7"] = "Sheet guide"
    ws["A7"].font = Font(bold=True, size=12)

    guide = [
        ("KPI Dashboard", "Headline KPIs + native charts: monthly MRR/revenue/LLM cost trend, margin by plan."),
        ("Monthly KPIs", "Month-by-month MRR, revenue, LLM cost, gross margin, active/paid customers."),
        ("Plan Economics", "Revenue, LLM cost, and margin rolled up by pricing plan."),
        ("Model x Task", "Success rate, cost, and latency for every (LLM model, task type) pair — the routing dataset."),
        ("Customers", "One row per customer: plan, segment, activation, total revenue/cost/margin (Excel Table, filters, conditional formatting)."),
        ("Pricing Simulator", "Live-formula what-if model: edit plan prices, credits, overage rate, price elasticity, model-routing %, and churn assumptions and watch revenue/cost/margin recompute. A locked 'Baseline outputs' block mirrors the same formulas at status-quo inputs, so at defaults Δ margin is exactly 0 for every plan — a genuine model-vs-model check, not model-vs-data."),
        ("Pivot-ready Data", "Flat customer-month fact table (Excel Table) — build your own PivotTables/PivotCharts from this."),
        ("_check", "Hidden sheet: Python-recomputed values for every Pricing Simulator output, used to verify the live formulas are correct."),
    ]
    r = 8
    for name, desc in guide:
        ws.cell(row=r, column=1, value=name).font = LABEL_FONT
        ws.cell(row=r, column=2, value=desc).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[r].height = 30
        r += 1

    ws.cell(row=r + 1, column=1, value=f"Generated: {dt.date.today().isoformat()}").font = NOTE_FONT


# --------------------------------------------------------------------- KPI Dashboard


def build_kpi_dashboard_sheet(wb: Workbook, monthly: pd.DataFrame, plan_econ: pd.DataFrame) -> None:
    ws = wb.create_sheet("KPI Dashboard")
    _set_col_widths(ws, {c: 16 for c in "ABCDEFGH"})

    ws["A1"] = "AgentOps — KPI Dashboard"
    ws["A1"].font = TITLE_FONT
    ws.merge_cells("A1:H1")

    latest = monthly.iloc[-1]
    total_revenue = monthly["paid_revenue"].sum()
    total_llm_cost = monthly["paid_llm_cost"].sum()
    overall_margin = (total_revenue - total_llm_cost) / total_revenue if total_revenue else 0
    paid_customers_latest = int(latest["paid_customers"])

    kpis = [
        ("Latest month MRR", latest["mrr"], MONEY_FMT),
        ("Latest month paid revenue", latest["paid_revenue"], MONEY_FMT),
        ("Latest month LLM cost (paid)", latest["paid_llm_cost"], MONEY_FMT),
        ("Latest month gross margin", latest["gross_margin"], PCT_FMT),
        ("Latest month paid customers", paid_customers_latest, INT_FMT),
        ("Lifetime paid revenue", total_revenue, MONEY_FMT),
        ("Lifetime LLM cost (paid)", total_llm_cost, MONEY_FMT),
        ("Lifetime gross margin", overall_margin, PCT_FMT),
    ]
    row, col = 3, 1
    for label, value, fmt in kpis:
        c1 = ws.cell(row=row, column=col, value=label)
        c1.font = Font(bold=True, size=9, color="595959")
        c1.alignment = Alignment(wrap_text=True)
        c2 = ws.cell(row=row + 1, column=col, value=value)
        c2.font = Font(bold=True, size=14, color="1F3864")
        c2.number_format = fmt
        col += 2
        if col > 8:
            col = 1
            row += 3

    # Data block backing the charts (kept on-sheet so charts can reference it directly).
    data_start = 10
    ws.cell(row=data_start, column=1, value="Monthly trend data (for charts below)").font = Font(bold=True)
    headers = ["month", "mrr", "paid_revenue", "paid_llm_cost", "gross_margin"]
    _write_header_row(ws, data_start + 1, headers)
    for i, (_, r) in enumerate(monthly.iterrows(), start=data_start + 2):
        ws.cell(row=i, column=1, value=str(r["month"])[:7])
        ws.cell(row=i, column=2, value=float(r["mrr"])).number_format = MONEY_FMT
        ws.cell(row=i, column=3, value=float(r["paid_revenue"])).number_format = MONEY_FMT
        ws.cell(row=i, column=4, value=float(r["paid_llm_cost"])).number_format = MONEY_FMT
        gm = r["gross_margin"]
        ws.cell(row=i, column=5, value=None if pd.isna(gm) else float(gm)).number_format = PCT_FMT
    data_end = data_start + 1 + len(monthly)

    line = LineChart()
    line.title = "Monthly MRR, Revenue & LLM Cost"
    line.style = 2
    line.y_axis.title = "USD"
    line.x_axis.title = "Month"
    cats = Reference(ws, min_col=1, min_row=data_start + 2, max_row=data_end)
    for col_idx, name in [(2, "MRR"), (3, "Revenue"), (4, "LLM Cost")]:
        data = Reference(ws, min_col=col_idx, min_row=data_start + 1, max_row=data_end)
        line.add_data(data, titles_from_data=True)
    line.set_categories(cats)
    line.width, line.height = 24, 11
    ws.add_chart(line, "G3")

    # Bar chart: margin by plan, from Plan Economics data written just below.
    plan_start = data_end + 3
    ws.cell(row=plan_start, column=1, value="Margin by plan (for chart)").font = Font(bold=True)
    _write_header_row(ws, plan_start + 1, ["plan", "gross_margin"])
    paid_plan_econ = plan_econ[plan_econ["plan"] != "Free"].reset_index(drop=True)
    for i, (_, r) in enumerate(paid_plan_econ.iterrows(), start=plan_start + 2):
        ws.cell(row=i, column=1, value=r["plan"])
        gm = r["gross_margin"]
        ws.cell(row=i, column=2, value=None if pd.isna(gm) else float(gm)).number_format = PCT_FMT
    plan_end = plan_start + 1 + len(paid_plan_econ)

    bar = BarChart()
    bar.title = "Gross Margin by Plan"
    bar.y_axis.title = "Gross margin"
    bar.y_axis.numFmt = "0%"
    cats2 = Reference(ws, min_col=1, min_row=plan_start + 2, max_row=plan_end)
    data2 = Reference(ws, min_col=2, min_row=plan_start + 1, max_row=plan_end)
    bar.add_data(data2, titles_from_data=True)
    bar.set_categories(cats2)
    bar.width, bar.height = 24, 11
    ws.add_chart(bar, "G20")


# --------------------------------------------------------------------- Monthly KPIs / Plan Economics / Model x Task


def build_monthly_kpis_sheet(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Monthly KPIs")
    out = df[["month", "mrr", "paid_revenue", "paid_llm_cost", "gross_margin",
              "active_customers", "paid_customers"]].copy()
    out["month"] = out["month"].str[:7]
    last_row, last_col = write_dataframe(
        ws, out, 1,
        number_formats={"mrr": MONEY_FMT, "paid_revenue": MONEY_FMT, "paid_llm_cost": MONEY_FMT,
                         "gross_margin": PCT_FMT, "active_customers": INT_FMT, "paid_customers": INT_FMT},
    )
    add_excel_table(ws, "MonthlyKPIs", 1, 1, last_row, last_col)
    color_scale(ws, f"E2:E{last_row}")
    ws.freeze_panes = "A2"
    _set_col_widths(ws, {c: 16 for c in "ABCDEFG"})


def build_plan_economics_sheet(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Plan Economics")
    out = df[["plan", "customers", "customer_months", "revenue", "llm_cost", "gross_margin"]].copy()
    last_row, last_col = write_dataframe(
        ws, out, 1,
        number_formats={"customers": INT_FMT, "customer_months": INT_FMT, "revenue": MONEY_FMT,
                         "llm_cost": MONEY_FMT, "gross_margin": PCT_FMT},
    )
    add_excel_table(ws, "PlanEconomics", 1, 1, last_row, last_col)
    color_scale(ws, f"F2:F{last_row}")
    ws.freeze_panes = "A2"
    _set_col_widths(ws, {c: 16 for c in "ABCDEF"})


def build_model_task_sheet(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Model x Task")
    out = df.copy()
    last_row, last_col = write_dataframe(
        ws, out, 1,
        number_formats={"n_runs": INT_FMT, "success_rate": PCT_FMT, "avg_cost_per_run": MONEY2_FMT,
                         "cost_per_success": MONEY2_FMT, "avg_latency_sec": "#,##0.0"},
    )
    add_excel_table(ws, "ModelTask", 1, 1, last_row, last_col)
    color_scale(ws, f"C2:C{last_row}")
    ws.freeze_panes = "A2"
    _set_col_widths(ws, {"A": 20, "B": 20, "C": 12, "D": 14, "E": 16, "F": 16, "G": 16})


# --------------------------------------------------------------------- Customers


def build_customers_sheet(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Customers")
    out = df[[
        "customer_id", "company_name", "industry", "company_size", "region",
        "acquisition_channel", "current_plan", "status", "total_revenue",
        "total_llm_cost", "gross_margin", "total_runs", "success_rate",
        "first5_success_rate", "activated",
    ]].copy()
    sanitize_cols = ["customer_id", "company_name", "industry", "company_size", "region",
                      "acquisition_channel", "current_plan", "status"]
    last_row, last_col = write_dataframe(
        ws, out, 1, sanitize_columns=sanitize_cols,
        number_formats={"total_revenue": MONEY_FMT, "total_llm_cost": MONEY_FMT, "gross_margin": PCT_FMT,
                         "total_runs": INT_FMT, "success_rate": PCT_FMT, "first5_success_rate": PCT_FMT},
    )
    add_excel_table(ws, "Customers", 1, 1, last_row, last_col)
    color_scale(ws, f"K2:K{last_row}")  # gross_margin
    color_scale(ws, f"M2:M{last_row}")  # success_rate
    ws.freeze_panes = "A2"
    _set_col_widths(ws, {
        "A": 12, "B": 24, "C": 14, "D": 14, "E": 12, "F": 16, "G": 12, "H": 10,
        "I": 14, "J": 14, "K": 12, "L": 10, "M": 12, "N": 14, "O": 10,
    })


# --------------------------------------------------------------------- Pivot-ready Data


def build_pivot_data_sheet(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Pivot-ready Data")
    sanitize_cols = ["customer_id", "month", "plan", "industry", "company_size",
                      "region", "acquisition_channel"]
    last_row, last_col = write_dataframe(
        ws, df, 1, sanitize_columns=sanitize_cols,
        number_formats={"monthly_fee_usd": MONEY_FMT, "included_credits": INT_FMT,
                         "credits_used": INT_FMT, "overage_credits": INT_FMT,
                         "overage_revenue_usd": MONEY_FMT, "revenue_usd": MONEY_FMT,
                         "llm_cost": MONEY2_FMT, "gross_margin": PCT_FMT},
    )
    add_excel_table(ws, "PivotData", 1, 1, last_row, last_col, style="TableStyleMedium2")
    ws.freeze_panes = "A2"
    _set_col_widths(ws, {c: 16 for c in "ABCDEFGHIJKLMNO"})


# --------------------------------------------------------------------- Pricing Simulator


def build_pricing_simulator_sheet(wb: Workbook, baseline: dict) -> dict:
    """Build the showpiece Pricing Simulator sheet.

    Design note: every "Projected" output has an apples-to-apples "Baseline
    (locked)" counterpart built from the *same formula structure*, but always
    referencing a gray, non-editable copy of the status-quo inputs (and
    literal 0s for routing/churn). At the sheet's default input state the
    editable inputs equal the locked ones, so Δ margin is exactly 0.00 for
    every plan and blended — a model-vs-model comparison, not model-vs-data.
    A separate "Actual margin in data" row shows the real historical margin
    (from `fetch_plan_economics`) for comparison, with a note explaining why
    it differs from the model's average-usage baseline.

    Returns a dict of the key output cell addresses and the Python-recomputed
    "expected" values for the default (baseline) input state, so `_check` and
    the test suite can confirm the live formulas evaluate to the right numbers.
    """
    ws = wb.create_sheet("Pricing Simulator")
    _set_col_widths(ws, {"A": 30, "B": 14, "C": 14, "D": 14, "E": 14, "F": 14, "G": 14, "H": 16})

    ws["A1"] = "Pricing Simulator"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = (
        "Edit any YELLOW cell (plan prices, included credits, overage rate, credits per task, "
        "price elasticity, % of simple tasks routed to a cheaper model, and churn assumption) — "
        "every number below recomputes live via Excel formulas. Gray cells are locked reference "
        "values pulled from the actual data (either today's baseline, or the status quo the model "
        "compares against — never touch them)."
    )
    ws["A2"].font = SUBTITLE_FONT
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("A2:H2")
    ws.row_dimensions[2].height = 42

    task_types = baseline["task_types"]
    plans_df = baseline["plans"]
    active_customers = baseline["active_customers"]
    avg_llm_cost = baseline["avg_llm_cost"]
    avg_frontier_simple_cost = baseline["avg_frontier_simple_cost"]
    avg_runs_per_task = baseline["avg_runs_per_task"]
    credits_per_task = baseline["credits_per_task"]
    reduction_factor = baseline["reduction_factor"]
    elasticity_pro = baseline["elasticity_pro"]
    actual_margin = baseline["actual_margin"]

    refs: dict = {"task_types": task_types, "plans": PAID_PLANS}
    GRAY = PatternFill("solid", fgColor="F2F2F2")

    # ---- Section A: credits-per-task-type input row + avg runs/month by plan (reference) ----
    row = 4
    ws.cell(row=row, column=1, value="Task mix & credits (drives overage via SUMPRODUCT)").font = Font(bold=True, size=11)
    row += 1
    header_row = row
    ws.cell(row=header_row, column=1, value="Avg runs / customer-month by plan →")
    for j, task in enumerate(task_types):
        ws.cell(row=header_row, column=2 + j, value=task.replace("_", " ").title())
        ws.cell(row=header_row, column=2 + j).font = Font(bold=True, size=9)
        ws.cell(row=header_row, column=2 + j).alignment = Alignment(wrap_text=True)
    row += 1
    avg_runs_rows: dict[str, int] = {}
    for plan in PAID_PLANS:
        ws.cell(row=row, column=1, value=f"{plan} (baseline avg)").font = LABEL_FONT
        for j, task in enumerate(task_types):
            cell = ws.cell(row=row, column=2 + j, value=round(float(avg_runs_per_task.loc[plan, task]), 4))
            cell.number_format = "0.00"
            cell.fill = GRAY
            cell.border = BORDER
        avg_runs_rows[plan] = row
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Credits per task type (INPUT)").font = LABEL_FONT
    row += 1
    credits_row = row
    credit_input_cells: dict[str, str] = {}
    for j, task in enumerate(task_types):
        col = 2 + j
        ws.cell(row=credits_row - 1, column=col, value=task.replace("_", " ").title()).font = Font(bold=True, size=9)
        c = input_cell(ws, credits_row, col, float(credits_per_task[task]), number_format="0.0")
        dv = DataValidation(type="decimal", operator="greaterThanOrEqual", formula1="0")
        dv.error = "Credits per task must be >= 0"
        ws.add_data_validation(dv)
        dv.add(c)
        credit_input_cells[task] = f"{get_column_letter(col)}{credits_row}"
    row += 1
    ws.cell(row=row, column=1, value="Credits per task type (baseline, locked)").font = Font(size=9, italic=True)
    row += 1
    locked_credits_row = row
    for j, task in enumerate(task_types):
        col = 2 + j
        c = ws.cell(row=locked_credits_row, column=col, value=float(credits_per_task[task]))
        c.number_format = "0.0"
        c.fill = GRAY
        c.border = BORDER
    row = locked_credits_row + 2

    # ---- Section B: global routing / churn levers ----
    ws.cell(row=row, column=1, value="Global levers").font = Font(bold=True, size=11)
    row += 1
    pct_routed_label_row = row
    ws.cell(row=row, column=1, value="% of simple-task frontier runs routed to balanced-medium")
    # Default 0 — status quo has no routing. Scenarios below set their own %.
    pct_routed_cell = input_cell(ws, row, 2, 0.0, number_format="0%")
    dv_pct = DataValidation(type="decimal", operator="between", formula1="0", formula2="1")
    dv_pct.error = "Enter a percentage between 0% and 100%"
    ws.add_data_validation(dv_pct)
    dv_pct.add(pct_routed_cell)
    row += 1
    ws.cell(row=row, column=1, value="Cost reduction from routing (baseline, from data)")
    reduction_cell = ws.cell(row=row, column=2, value=round(reduction_factor, 4))
    reduction_cell.number_format = "0.0%"
    reduction_cell.fill = GRAY
    reduction_addr = f"B{row}"
    row += 1
    ws.cell(row=row, column=1, value="Net customer count change from churn improvement")
    churn_cell = input_cell(ws, row, 2, 0.0, number_format="0%")
    dv_churn = DataValidation(type="decimal", operator="between", formula1="-1", formula2="1")
    dv_churn.error = "Enter a percentage between -100% and 100%"
    ws.add_data_validation(dv_churn)
    dv_churn.add(churn_cell)
    churn_addr = "B" + str(row)
    pct_routed_addr = f"B{pct_routed_label_row}"
    row += 2

    # ---- Section C: per-plan pricing inputs (incl. price elasticity) ----
    ws.cell(row=row, column=1, value="Plan pricing inputs").font = Font(bold=True, size=11)
    row += 1
    ws.cell(row=row, column=1, value=(
        "Elasticity default = arc elasticity of 30-day conversion from the Pro $29-vs-$39 A/B test "
        "(experiment_assignments + customers.first_paid_date). Team/Enterprise reuse it as an "
        "ASSUMPTION — there's no A/B test for those plans."
    )).font = NOTE_FONT
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    ws.row_dimensions[row].height = 28
    row += 1
    _write_header_row(ws, row, ["Plan", "Price ($/mo)", "Included credits", "Overage ($/credit)",
                                 "Price elasticity of customers", "Active customers (baseline)"])
    row += 1
    plan_input_rows: dict[str, int] = {}
    for plan in PAID_PLANS:
        ws.cell(row=row, column=1, value=plan).font = LABEL_FONT
        price_cell = input_cell(ws, row, 2, float(plans_df.loc[plan, "monthly_fee_usd"]), number_format=MONEY_FMT)
        incl_cell = input_cell(ws, row, 3, float(plans_df.loc[plan, "included_credits"]), number_format=INT_FMT)
        overage_cell = input_cell(ws, row, 4, float(plans_df.loc[plan, "overage_usd_per_credit"]), number_format="0.00")
        elasticity_cell = input_cell(ws, row, 5, round(float(elasticity_pro), 4), number_format="0.00")
        for c in (price_cell, incl_cell, overage_cell):
            dv = DataValidation(type="decimal", operator="greaterThanOrEqual", formula1="0")
            dv.error = "Must be >= 0"
            ws.add_data_validation(dv)
            dv.add(c)
        dv_elas = DataValidation(type="decimal", operator="between", formula1="-5", formula2="5")
        dv_elas.error = "Enter an elasticity between -5 and 5"
        ws.add_data_validation(dv_elas)
        dv_elas.add(elasticity_cell)
        cust_cell = ws.cell(row=row, column=6, value=int(active_customers.get(plan, 0)))
        cust_cell.number_format = INT_FMT
        cust_cell.fill = GRAY
        cust_cell.border = BORDER
        plan_input_rows[plan] = row
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Baseline plan inputs (locked — status quo the model compares against)").font = Font(size=9, italic=True)
    row += 1
    _write_header_row(ws, row, ["Plan", "Price ($/mo)", "Included credits", "Overage ($/credit)"])
    row += 1
    locked_plan_rows: dict[str, int] = {}
    for plan in PAID_PLANS:
        ws.cell(row=row, column=1, value=plan)
        vals = [
            float(plans_df.loc[plan, "monthly_fee_usd"]),
            float(plans_df.loc[plan, "included_credits"]),
            float(plans_df.loc[plan, "overage_usd_per_credit"]),
        ]
        fmts = [MONEY_FMT, INT_FMT, "0.00"]
        for j, (v, fmt) in enumerate(zip(vals, fmts)):
            c = ws.cell(row=row, column=2 + j, value=v)
            c.number_format = fmt
            c.fill = GRAY
            c.border = BORDER
        locked_plan_rows[plan] = row
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Baseline avg LLM cost / customer-month (from data)").font = Font(size=9, italic=True)
    row += 1
    _write_header_row(ws, row, ["Plan", "Avg LLM cost/mo", "Avg frontier+simple-task cost/mo"])
    row += 1
    baseline_cost_rows: dict[str, int] = {}
    for plan in PAID_PLANS:
        ws.cell(row=row, column=1, value=plan)
        c1 = ws.cell(row=row, column=2, value=round(float(avg_llm_cost[plan]), 4))
        c1.number_format = MONEY2_FMT
        c1.fill = GRAY
        c2 = ws.cell(row=row, column=3, value=round(float(avg_frontier_simple_cost[plan]), 4))
        c2.number_format = MONEY2_FMT
        c2.fill = GRAY
        for c in (c1, c2):
            c.border = BORDER
        baseline_cost_rows[plan] = row
        row += 1

    # ---- Section D: baseline outputs (locked model — the Δ margin=0 reference) ----
    row += 1
    ws.cell(row=row, column=1, value="Baseline outputs (locked model — reference for Δ margin)").font = Font(bold=True, size=11)
    row += 1
    _write_header_row(ws, row, ["Plan", "Adj. customers", "Avg credits used/mo", "Revenue/mo",
                                 "LLM cost/mo", "Gross margin"])
    row += 1
    baseline_output_rows: dict[str, int] = {}
    for plan in PAID_PLANS:
        lp = locked_plan_rows[plan]
        pr = plan_input_rows[plan]  # for the "Active customers (baseline)" reference column F
        bc = baseline_cost_rows[plan]
        locked_credits_terms = "+".join(
            f"{get_column_letter(2 + j)}${locked_credits_row}*{get_column_letter(2 + j)}{avg_runs_rows[plan]}"
            for j in range(len(task_types))
        )
        ws.cell(row=row, column=1, value=plan).font = LABEL_FONT

        # Locked churn (0) and locked routing (0) are literal, not cell refs —
        # the baseline is *by definition* the no-routing, no-churn-change,
        # no-price-change status quo, structurally identical to the projected
        # formula below but with every lever pinned at its status-quo value.
        adj_cust_formula = f"=F{pr}*(1+0)*(1+E{pr}*(B{lp}-B{lp})/B{lp})"
        ws.cell(row=row, column=2, value=adj_cust_formula).number_format = "#,##0.0"

        credits_formula = "=" + locked_credits_terms
        ws.cell(row=row, column=3, value=credits_formula).number_format = "0.0"

        rev_formula = f"=(B{lp}+MAX(0,C{row}-C{lp})*D{lp})*B{row}"
        ws.cell(row=row, column=4, value=rev_formula).number_format = MONEY_FMT

        cost_formula = f"=(B{bc}-C{bc}*0*{reduction_addr})*B{row}"
        ws.cell(row=row, column=5, value=cost_formula).number_format = MONEY_FMT

        margin_formula = f"=IF(D{row}=0,0,(D{row}-E{row})/D{row})"
        ws.cell(row=row, column=6, value=margin_formula).number_format = PCT_FMT

        for c in range(1, 7):
            ws.cell(row=row, column=c).border = BORDER
            if c > 1:
                ws.cell(row=row, column=c).fill = GRAY
        baseline_output_rows[plan] = row
        row += 1

    baseline_total_row = row
    ws.cell(row=baseline_total_row, column=1, value="Total / blended (baseline)").font = Font(bold=True)
    b_first, b_last = baseline_output_rows[PAID_PLANS[0]], baseline_output_rows[PAID_PLANS[-1]]
    ws.cell(row=baseline_total_row, column=2, value=f"=SUM(B{b_first}:B{b_last})").number_format = "#,##0.0"
    ws.cell(row=baseline_total_row, column=4, value=f"=SUM(D{b_first}:D{b_last})").number_format = MONEY_FMT
    ws.cell(row=baseline_total_row, column=5, value=f"=SUM(E{b_first}:E{b_last})").number_format = MONEY_FMT
    ws.cell(row=baseline_total_row, column=6,
            value=f"=IF(D{baseline_total_row}=0,0,(D{baseline_total_row}-E{baseline_total_row})/D{baseline_total_row})").number_format = PCT_FMT
    for c in range(1, 7):
        ws.cell(row=baseline_total_row, column=c).border = BORDER
        ws.cell(row=baseline_total_row, column=c).font = Font(bold=True)
    row = baseline_total_row + 2

    # ---- Section E: live projected outputs ----
    ws.cell(row=row, column=1, value="Projected outputs (LIVE FORMULAS)").font = Font(bold=True, size=11, color="C00000")
    row += 1
    _write_header_row(ws, row, ["Plan", "Adj. customers", "Avg credits used/mo", "Projected revenue/mo",
                                 "Projected LLM cost/mo", "Gross margin", "Δ margin vs baseline (pp)",
                                 "Actual margin in data (reference)"])
    output_header_row = row
    row += 1
    output_rows: dict[str, int] = {}
    for plan in PAID_PLANS:
        pr = plan_input_rows[plan]
        lp = locked_plan_rows[plan]
        bc = baseline_cost_rows[plan]
        brow = baseline_output_rows[plan]
        credits_terms = "+".join(
            f"{credit_input_cells[t]}*{get_column_letter(2 + j)}{avg_runs_rows[plan]}"
            for j, t in enumerate(task_types)
        )
        ws.cell(row=row, column=1, value=plan).font = LABEL_FONT

        # Adjusted customers = baseline customers, scaled by (1 + churn change)
        # and by a price-elasticity demand response to the price change vs the
        # LOCKED (status-quo) price — so editing price alone moves customer
        # count, and at default inputs (price == locked price) this term is 1.
        adj_cust_formula = f"=F{pr}*(1+{churn_addr})*(1+E{pr}*(B{pr}-B{lp})/B{lp})"
        ws.cell(row=row, column=2, value=adj_cust_formula).number_format = "#,##0.0"

        credits_formula = "=" + credits_terms
        ws.cell(row=row, column=3, value=credits_formula).number_format = "0.0"

        rev_formula = f"=(B{pr}+MAX(0,C{row}-C{pr})*D{pr})*B{row}"
        ws.cell(row=row, column=4, value=rev_formula).number_format = MONEY_FMT

        cost_formula = f"=(B{bc}-C{bc}*{pct_routed_addr}*{reduction_addr})*B{row}"
        ws.cell(row=row, column=5, value=cost_formula).number_format = MONEY_FMT

        margin_formula = f"=IF(D{row}=0,0,(D{row}-E{row})/D{row})"
        ws.cell(row=row, column=6, value=margin_formula).number_format = PCT_FMT

        # Δ margin vs the LOCKED baseline model (Section D) — same formula
        # structure on both sides, so at default inputs this is exactly 0.
        delta_formula = f"=(F{row}-F{brow})*100"
        ws.cell(row=row, column=7, value=delta_formula).number_format = "0.00"

        actual_val = actual_margin.get(plan)
        actual_cell = ws.cell(row=row, column=8, value=None if pd.isna(actual_val) else round(float(actual_val), 6))
        actual_cell.number_format = PCT_FMT
        actual_cell.fill = GRAY

        for c in range(1, 9):
            ws.cell(row=row, column=c).border = BORDER
        output_rows[plan] = row
        row += 1

    total_row = row
    ws.cell(row=total_row, column=1, value="Total / blended").font = Font(bold=True)
    first_out, last_out = output_rows[PAID_PLANS[0]], output_rows[PAID_PLANS[-1]]
    ws.cell(row=total_row, column=2, value=f"=SUM(B{first_out}:B{last_out})").number_format = "#,##0.0"
    ws.cell(row=total_row, column=4, value=f"=SUM(D{first_out}:D{last_out})").number_format = MONEY_FMT
    ws.cell(row=total_row, column=5, value=f"=SUM(E{first_out}:E{last_out})").number_format = MONEY_FMT
    ws.cell(row=total_row, column=6, value=f"=IF(D{total_row}=0,0,(D{total_row}-E{total_row})/D{total_row})").number_format = PCT_FMT
    ws.cell(row=total_row, column=7, value=f"=(F{total_row}-F{baseline_total_row})*100").number_format = "0.00"
    for c in range(1, 9):
        ws.cell(row=total_row, column=c).border = BORDER
        ws.cell(row=total_row, column=c).font = Font(bold=True)
    color_scale(ws, f"F{output_header_row + 1}:F{total_row}")

    row = total_row + 1
    note = ws.cell(row=row, column=1, value=(
        "Note: the model applies MAX(0, usage − included) to each plan's AVERAGE usage per "
        "customer-month, which understates the overage a real, skewed customer base generates "
        "(Jensen's inequality: E[max(0, X − included)] ≥ max(0, E[X] − included) for a "
        "convex payoff like overage billing) — heavy users individually cross the credit cap far "
        "more often than the average customer appears to. That's why \"Actual margin in data\" "
        "(computed from real per-customer overage) tends to run ABOVE the model's baseline margin: "
        "the model is a conservative, average-usage approximation, not a per-customer simulation."
    ))
    note.font = NOTE_FONT
    note.alignment = Alignment(wrap_text=True)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    ws.row_dimensions[row].height = 40

    refs.update({
        "plan_input_rows": plan_input_rows,
        "locked_plan_rows": locked_plan_rows,
        "baseline_cost_rows": baseline_cost_rows,
        "baseline_output_rows": baseline_output_rows,
        "baseline_total_row": baseline_total_row,
        "output_rows": output_rows,
        "total_row": total_row,
        "credit_input_cells": credit_input_cells,
        "locked_credits_row": locked_credits_row,
        "avg_runs_rows": avg_runs_rows,
        "pct_routed_addr": pct_routed_addr,
        "reduction_addr": reduction_addr,
        "churn_addr": churn_addr,
    })

    # ---- Section E: scenario comparison block ----
    row = total_row + 3
    ws.cell(row=row, column=1, value="Scenario comparison (status quo vs 3 scenarios)").font = Font(bold=True, size=11, color="C00000")
    row += 1
    scenario_cols = {"Status Quo": 2, "Scenario A": 3, "Scenario B": 4, "Scenario C": 5}
    _write_header_row(ws, row, ["Input"] + list(scenario_cols.keys()), start_col=1)
    scenario_header_row = row
    row += 1

    scenario_defaults = {
        "Pro price": {"Status Quo": float(plans_df.loc["Pro", "monthly_fee_usd"]), "Scenario A": 34.0, "Scenario B": 29.0, "Scenario C": 39.0},
        "Team price": {"Status Quo": float(plans_df.loc["Team", "monthly_fee_usd"]), "Scenario A": 159.0, "Scenario B": 149.0, "Scenario C": 179.0},
        "Enterprise price": {"Status Quo": float(plans_df.loc["Enterprise", "monthly_fee_usd"]), "Scenario A": 1200.0, "Scenario B": 1250.0, "Scenario C": 1350.0},
        "% simple tasks routed": {"Status Quo": 0.0, "Scenario A": 0.30, "Scenario B": 0.60, "Scenario C": 0.60},
        "Churn improvement %": {"Status Quo": 0.0, "Scenario A": 0.0, "Scenario B": 0.03, "Scenario C": 0.03},
    }
    scenario_input_rows: dict[str, int] = {}
    for label, values in scenario_defaults.items():
        ws.cell(row=row, column=1, value=label).font = LABEL_FONT
        for scen_name, col in scenario_cols.items():
            fmt = "0%" if "%" in label or "routed" in label or "improvement" in label else MONEY_FMT
            c = input_cell(ws, row, col, values[scen_name], number_format=fmt)
            if scen_name == "Status Quo":
                c.fill = PatternFill("solid", fgColor="F2F2F2")
                c.font = Font(color="595959")
        scenario_input_rows[label] = row
        row += 1

    row += 1
    _write_header_row(ws, row, ["Output"] + list(scenario_cols.keys()), start_col=1)
    scenario_output_header = row
    row += 1

    def scenario_formula_block(metric_row_label: str, formula_builder):
        nonlocal row
        ws.cell(row=row, column=1, value=metric_row_label).font = LABEL_FONT
        for scen_name, col in scenario_cols.items():
            col_letter = get_column_letter(col)
            formula = formula_builder(col_letter)
            ws.cell(row=row, column=col, value=formula)
        return row

    pro_price_r = scenario_input_rows["Pro price"]
    team_price_r = scenario_input_rows["Team price"]
    ent_price_r = scenario_input_rows["Enterprise price"]
    routed_r = scenario_input_rows["% simple tasks routed"]
    churn_r = scenario_input_rows["Churn improvement %"]
    price_row_for_plan = {"Pro": pro_price_r, "Team": team_price_r, "Enterprise": ent_price_r}

    def adj_customers_formula(plan: str, col_letter: str) -> str:
        pr = plan_input_rows[plan]  # F{pr} = baseline active customers; E{pr} = elasticity input
        lp = locked_plan_rows[plan]  # B{lp} = locked (status-quo) price
        price_r = price_row_for_plan[plan]
        return (
            f"F${pr}*(1+{col_letter}${churn_r})"
            f"*(1+E${pr}*({col_letter}${price_r}-B${lp})/B${lp})"
        )

    def total_revenue_formula(col_letter: str) -> str:
        terms = []
        for plan in PAID_PLANS:
            price_r = price_row_for_plan[plan]
            terms.append(f"{col_letter}${price_r}*({adj_customers_formula(plan, col_letter)})")
        return "=" + "+".join(terms)

    def total_cost_formula(col_letter: str) -> str:
        terms = []
        for plan in PAID_PLANS:
            bc = baseline_cost_rows[plan]
            terms.append(
                f"(B${bc}-C${bc}*{col_letter}${routed_r}*{reduction_addr})"
                f"*({adj_customers_formula(plan, col_letter)})"
            )
        return "=" + "+".join(terms)

    rev_row = scenario_formula_block("Total revenue/mo", total_revenue_formula)
    for scen_name, col in scenario_cols.items():
        ws.cell(row=rev_row, column=col).number_format = MONEY_FMT
    row += 1

    cost_row = scenario_formula_block("Total LLM cost/mo", total_cost_formula)
    for scen_name, col in scenario_cols.items():
        ws.cell(row=cost_row, column=col).number_format = MONEY_FMT
    row += 1

    margin_row = scenario_formula_block(
        "Blended gross margin",
        lambda cl: f"=IF({cl}{rev_row}=0,0,({cl}{rev_row}-{cl}{cost_row})/{cl}{rev_row})",
    )
    for scen_name, col in scenario_cols.items():
        ws.cell(row=margin_row, column=col).number_format = PCT_FMT
    row += 1

    sq_col = get_column_letter(scenario_cols["Status Quo"])
    delta_row = scenario_formula_block(
        "Δ margin vs Status Quo (pp)",
        lambda cl: f"=({cl}{margin_row}-{sq_col}{margin_row})*100",
    )
    for scen_name, col in scenario_cols.items():
        ws.cell(row=delta_row, column=col).number_format = "0.0"

    refs["scenario"] = {
        "cols": scenario_cols,
        "input_rows": scenario_input_rows,
        "revenue_row": rev_row,
        "cost_row": cost_row,
        "margin_row": margin_row,
        "delta_row": delta_row,
    }

    ws.freeze_panes = "A3"
    return refs


# --------------------------------------------------------------------- _check sheet (hidden)


def compute_expected_simulator_outputs(
    baseline: dict,
    price_overrides: dict[str, float] | None = None,
    pct_routed: float = 0.0,
    churn_change: float = 0.0,
) -> dict:
    """Reimplements the Pricing Simulator's formula logic in Python.

    Used both to populate the hidden `_check` sheet and by the test suite to
    confirm the workbook's live formulas will evaluate to the same numbers as
    the analytics they're built from — since no LibreOffice/Excel engine is
    available in this environment to actually recalculate the .xlsx.

    `price_overrides` lets a caller move a plan's price away from the locked
    baseline (e.g. to check the elasticity response) without touching any
    other input; defaults reproduce the sheet's own default input state,
    where every editable input equals its locked counterpart and the
    "projected" and "baseline (locked)" blocks are therefore identical —
    i.e. delta_pp == 0.0 for every plan and blended.
    """
    plans_df = baseline["plans"]
    active_customers = baseline["active_customers"]
    avg_llm_cost = baseline["avg_llm_cost"]
    avg_frontier_simple_cost = baseline["avg_frontier_simple_cost"]
    avg_runs_per_task = baseline["avg_runs_per_task"]
    credits_per_task = baseline["credits_per_task"]
    reduction_factor = baseline["reduction_factor"]
    task_types = baseline["task_types"]
    elasticity_pro = float(baseline["elasticity_pro"])
    price_overrides = price_overrides or {}

    per_plan = {}
    total_revenue = 0.0
    total_cost = 0.0
    total_customers = 0.0
    baseline_total_revenue = 0.0
    baseline_total_cost = 0.0
    for plan in PAID_PLANS:
        locked_price = float(plans_df.loc[plan, "monthly_fee_usd"])
        included = float(plans_df.loc[plan, "included_credits"])
        overage_rate = float(plans_df.loc[plan, "overage_usd_per_credit"])
        price = price_overrides.get(plan, locked_price)
        elasticity = elasticity_pro  # same default assumption for all three plans
        baseline_customers = float(active_customers.get(plan, 0))

        credits_used = sum(
            float(credits_per_task[t]) * float(avg_runs_per_task.loc[plan, t]) for t in task_types
        )
        base_llm = float(avg_llm_cost[plan])
        base_frontier_simple = float(avg_frontier_simple_cost[plan])

        # ---- baseline (locked): status-quo price, 0% routed, 0% churn change ----
        baseline_adj_customers = baseline_customers * (1 + 0.0) * (1 + elasticity * 0.0)
        baseline_revenue_per_customer = locked_price + max(0.0, credits_used - included) * overage_rate
        baseline_revenue = baseline_revenue_per_customer * baseline_adj_customers
        baseline_cost_per_customer = base_llm - base_frontier_simple * 0.0 * reduction_factor
        baseline_cost = baseline_cost_per_customer * baseline_adj_customers
        baseline_margin = 0.0 if baseline_revenue == 0 else (baseline_revenue - baseline_cost) / baseline_revenue

        # ---- projected: editable inputs (default to locked values) ----
        pct_price_change = (price - locked_price) / locked_price
        adj_customers = baseline_customers * (1 + churn_change) * (1 + elasticity * pct_price_change)
        revenue_per_customer = price + max(0.0, credits_used - included) * overage_rate
        revenue = revenue_per_customer * adj_customers
        cost_per_customer = base_llm - base_frontier_simple * pct_routed * reduction_factor
        cost = cost_per_customer * adj_customers
        margin = 0.0 if revenue == 0 else (revenue - cost) / revenue

        per_plan[plan] = {
            "customers": adj_customers, "credits_used": credits_used, "revenue": revenue,
            "cost": cost, "margin": margin, "baseline_margin": baseline_margin,
            "baseline_customers": baseline_adj_customers, "baseline_revenue": baseline_revenue,
            "baseline_cost": baseline_cost,
            "delta_pp": (margin - baseline_margin) * 100,
        }
        total_revenue += revenue
        total_cost += cost
        total_customers += adj_customers
        baseline_total_revenue += baseline_revenue
        baseline_total_cost += baseline_cost

    total_margin = 0.0 if total_revenue == 0 else (total_revenue - total_cost) / total_revenue
    baseline_total_margin = (
        0.0 if baseline_total_revenue == 0
        else (baseline_total_revenue - baseline_total_cost) / baseline_total_revenue
    )

    return {
        "per_plan": per_plan,
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "total_margin": total_margin,
        "total_customers": total_customers,
        "baseline_total_revenue": baseline_total_revenue,
        "baseline_total_cost": baseline_total_cost,
        "baseline_total_margin": baseline_total_margin,
        "total_delta_pp": (total_margin - baseline_total_margin) * 100,
    }


def build_check_sheet(wb: Workbook, expected: dict) -> None:
    ws = wb.create_sheet("_check")
    ws.sheet_state = "hidden"
    ws["A1"] = "Python-recomputed Pricing Simulator outputs (default input state)"
    ws["A1"].font = Font(bold=True)
    _write_header_row(ws, 3, ["Plan", "Customers", "Credits used/mo", "Revenue/mo", "LLM cost/mo",
                                "Margin", "Baseline margin", "Delta (pp)"])
    row = 4
    for plan in PAID_PLANS:
        p = expected["per_plan"][plan]
        ws.cell(row=row, column=1, value=plan)
        ws.cell(row=row, column=2, value=round(p["customers"], 4))
        ws.cell(row=row, column=3, value=round(p["credits_used"], 4))
        ws.cell(row=row, column=4, value=round(p["revenue"], 4))
        ws.cell(row=row, column=5, value=round(p["cost"], 4))
        ws.cell(row=row, column=6, value=round(p["margin"], 6))
        ws.cell(row=row, column=7, value=round(p["baseline_margin"], 6))
        ws.cell(row=row, column=8, value=round(p["delta_pp"], 4))
        row += 1
    ws.cell(row=row, column=1, value="TOTAL")
    ws.cell(row=row, column=4, value=round(expected["total_revenue"], 4))
    ws.cell(row=row, column=5, value=round(expected["total_cost"], 4))
    ws.cell(row=row, column=6, value=round(expected["total_margin"], 6))
    ws.cell(row=row, column=7, value=round(expected["baseline_total_margin"], 6))
    ws.cell(row=row, column=8, value=round(expected["total_delta_pp"], 4))


# --------------------------------------------------------------------- build & main


def build_workbook(out_path: Path) -> None:
    wb = Workbook()
    with get_conn() as conn:
        monthly = fetch_monthly_kpis(conn)
        plan_econ = fetch_plan_economics(conn)
        model_task = fetch_model_task(conn)
        customers = fetch_customers(conn)
        pivot_data = fetch_pivot_data(conn)
        baseline = fetch_simulator_baseline(conn)

    build_readme_sheet(wb)
    build_kpi_dashboard_sheet(wb, monthly, plan_econ)
    build_monthly_kpis_sheet(wb, monthly)
    build_plan_economics_sheet(wb, plan_econ)
    build_model_task_sheet(wb, model_task)
    build_customers_sheet(wb, customers)
    build_pricing_simulator_sheet(wb, baseline)
    build_pivot_data_sheet(wb, pivot_data)
    expected = compute_expected_simulator_outputs(baseline)
    build_check_sheet(wb, expected)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH,
                         help="Output path for the workbook (default: reports/AgentOps_Report.xlsx)")
    args = parser.parse_args()
    build_workbook(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
