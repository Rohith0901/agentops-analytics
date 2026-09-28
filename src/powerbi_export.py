"""powerbi_export.py — star-schema CSV export for Power BI / Tableau.

Reads the AgentOps SQLite database (via `src/db.py`) and writes a small star
schema of CSVs to `data/powerbi/`:

    dim_customer, dim_date, dim_plan, dim_model, dim_task   (dimensions)
    fact_agent_runs, fact_customer_month                    (facts, surrogate FKs)
    README.md                                               (relationships + DAX)

Design notes
------------
- Every dimension gets a small integer surrogate key (`*_key`) so the fact
  tables never carry natural-key strings as foreign keys — the standard
  Power BI star-schema pattern.
- `dim_date` is a full daily calendar spanning the data's date range (with a
  little padding) so date-based FKs (including `churn_date`) always resolve.
- All text pulled from the database is passed through `sanitize_cell` before
  being written, guarding against CSV/formula injection if the CSV is later
  opened directly in Excel/Sheets (see ARCHITECTURE.md "Testing & security
  requirements"). Numbers are left untouched.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from db import ROOT, get_conn

# Task types considered "simple" per the shared metric definitions — kept in
# sync with excel_report.py and ARCHITECTURE.md.
SIMPLE_TASKS = ["test_generation", "bug_fix", "refactor", "dependency_upgrade"]

DEFAULT_OUT_DIR = ROOT / "data" / "powerbi"

# Formula/CSV-injection guard: a leading apostrophe forces Excel/Sheets to
# treat the cell as plain text instead of evaluating it as a formula.
_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_cell(value):
    """Neutralise CSV/formula injection in a single cell value.

    Any string that *starts with* one of `= + - @ \\t \\r` is prefixed with a
    leading apostrophe so spreadsheet software renders it as literal text
    rather than evaluating it as a formula. Non-string values (including
    ``None``/NaN and numbers) pass through unchanged.
    """
    if not isinstance(value, str):
        return value
    if value.startswith(_DANGEROUS_PREFIXES):
        return "'" + value
    return value


def sanitize_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Apply `sanitize_cell` to the given (or all object-dtype) columns of a copy of `df`."""
    df = df.copy()
    cols = columns if columns is not None else list(df.select_dtypes(include=["object", "string"]).columns)
    for col in cols:
        if col in df.columns:
            df[col] = df[col].map(sanitize_cell)
    return df


# --------------------------------------------------------------------- dims


def build_dim_customer(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT customer_id, company_name, signup_date, industry, company_size,
               region, acquisition_channel, is_sales_led, first_paid_date,
               current_plan, churn_date, status
        FROM customers
        ORDER BY customer_id
        """,
        conn,
    )
    df.insert(0, "customer_key", range(1, len(df) + 1))
    text_cols = ["company_name", "industry", "company_size", "region",
                 "acquisition_channel", "current_plan", "status"]
    return sanitize_frame(df, text_cols)


def build_dim_plan(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT plan, monthly_fee_usd, included_credits, overage_usd_per_credit "
        "FROM plans ORDER BY monthly_fee_usd",
        conn,
    )
    df.insert(0, "plan_key", range(1, len(df) + 1))
    return sanitize_frame(df, ["plan"])


def build_dim_model(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT model, usd_per_m_input, usd_per_m_output, sec_per_step "
        "FROM models ORDER BY usd_per_m_input DESC",
        conn,
    )
    df.insert(0, "model_key", range(1, len(df) + 1))
    return sanitize_frame(df, ["model"])


def build_dim_task(conn) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT DISTINCT task_type FROM agent_runs ORDER BY task_type", conn
    )
    df.insert(0, "task_key", range(1, len(df) + 1))
    df["is_simple_task"] = df["task_type"].isin(SIMPLE_TASKS).astype(int)
    return sanitize_frame(df, ["task_type"])


def build_dim_date(conn) -> pd.DataFrame:
    (min_run, max_run) = conn.execute(
        "SELECT MIN(started_at), MAX(started_at) FROM agent_runs"
    ).fetchone()
    (min_month, max_month) = conn.execute(
        "SELECT MIN(month), MAX(month) FROM customer_months"
    ).fetchone()
    (min_signup, max_signup) = conn.execute(
        "SELECT MIN(signup_date), MAX(signup_date) FROM customers"
    ).fetchone()
    (min_churn, max_churn) = conn.execute(
        "SELECT MIN(churn_date), MAX(churn_date) FROM customers"
    ).fetchone()

    candidates = [min_run, max_run, min_month, max_month, min_signup, max_signup,
                  min_churn, max_churn]
    dates = [dt.date.fromisoformat(c[:10]) for c in candidates if c]
    start = min(dates).replace(day=1)
    end_raw = max(dates)
    # Pad to the end of the max month so partial trailing months resolve too.
    if end_raw.month == 12:
        end = dt.date(end_raw.year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        end = dt.date(end_raw.year, end_raw.month + 1, 1) - dt.timedelta(days=1)

    calendar = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": calendar})
    df["date_key"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["year"] = calendar.year
    df["quarter"] = calendar.quarter
    df["month"] = calendar.month
    df["month_name"] = calendar.strftime("%B")
    df["month_start"] = calendar.to_period("M").to_timestamp().strftime("%Y-%m-%d")
    df["day"] = calendar.day
    df["day_of_week"] = calendar.strftime("%A")
    df["is_month_start"] = (calendar.day == 1).astype(int)
    return df[["date_key", "date", "year", "quarter", "month", "month_name",
               "month_start", "day", "day_of_week", "is_month_start"]]


def _date_to_key(iso_date: str) -> int:
    return int(iso_date[:10].replace("-", ""))


# --------------------------------------------------------------------- facts


def build_fact_agent_runs(conn, dim_plan: pd.DataFrame, dim_model: pd.DataFrame,
                           dim_task: pd.DataFrame, customer_key_map: dict) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT run_id, customer_id, started_at, task_type, source_stack, target_stack,
               model, plan_at_run, steps, input_tokens, output_tokens, llm_cost_usd,
               latency_sec, lines_changed, tests_total, tests_passed, status,
               credits_charged, human_rating
        FROM agent_runs
        """,
        conn,
    )

    plan_key_map = dict(zip(dim_plan["plan"], dim_plan["plan_key"]))
    model_key_map = dict(zip(dim_model["model"], dim_model["model_key"]))
    task_key_map = dict(zip(dim_task["task_type"], dim_task["task_key"]))

    df["customer_key"] = df["customer_id"].map(customer_key_map)
    df["date_key"] = df["started_at"].str[:10].str.replace("-", "", regex=False).astype(int)
    df["plan_key"] = df["plan_at_run"].map(plan_key_map)
    df["model_key"] = df["model"].map(model_key_map)
    df["task_key"] = df["task_type"].map(task_key_map)

    ordered = df[[
        "run_id", "customer_key", "date_key", "plan_key", "model_key", "task_key",
        "started_at", "source_stack", "target_stack", "steps", "input_tokens",
        "output_tokens", "llm_cost_usd", "latency_sec", "lines_changed",
        "tests_total", "tests_passed", "status", "credits_charged", "human_rating",
    ]]
    return sanitize_frame(ordered, ["run_id", "started_at", "source_stack", "target_stack", "status"])


def build_fact_customer_month(conn, dim_plan: pd.DataFrame,
                               customer_key_map: dict) -> pd.DataFrame:
    cm = pd.read_sql_query(
        """
        SELECT customer_id, month, plan, monthly_fee_usd, included_credits,
               credits_used, overage_credits, overage_revenue_usd, revenue_usd
        FROM customer_months
        """,
        conn,
    )
    # LLM cost per customer-month: SUM(llm_cost_usd) grouped by customer_id and
    # the calendar month of started_at (shared metric definition).
    run_cost = pd.read_sql_query(
        """
        SELECT customer_id, substr(started_at, 1, 7) || '-01' AS month,
               SUM(llm_cost_usd) AS llm_cost_usd
        FROM agent_runs
        GROUP BY customer_id, substr(started_at, 1, 7)
        """,
        conn,
    )
    cm = cm.merge(run_cost, on=["customer_id", "month"], how="left")
    cm["llm_cost_usd"] = cm["llm_cost_usd"].fillna(0.0)

    plan_key_map = dict(zip(dim_plan["plan"], dim_plan["plan_key"]))
    cm["customer_key"] = cm["customer_id"].map(customer_key_map)
    cm["date_key"] = cm["month"].map(_date_to_key)
    cm["plan_key"] = cm["plan"].map(plan_key_map)

    ordered = cm[[
        "customer_key", "date_key", "plan_key", "monthly_fee_usd", "included_credits",
        "credits_used", "overage_credits", "overage_revenue_usd", "revenue_usd",
        "llm_cost_usd",
    ]]
    return ordered


# --------------------------------------------------------------------- readme


README_TEMPLATE = """# AgentOps Power BI star schema

Synthetic data (see project README / ARCHITECTURE.md) for the fictional
CodeShift SaaS platform. Load all CSVs below into Power BI (or Tableau) and
wire up the relationships exactly as listed — every fact table uses small
integer surrogate keys, never natural-key strings, as foreign keys.

## Tables

| Table | Grain | Notes |
|---|---|---|
| `dim_customer` | one row per customer | `customer_key` surrogate PK |
| `dim_date` | one row per calendar day | `date_key` = `YYYYMMDD` int PK; spans the full data range |
| `dim_plan` | one row per pricing plan | `plan_key` surrogate PK |
| `dim_model` | one row per LLM model | `model_key` surrogate PK |
| `dim_task` | one row per agent task type | `task_key` surrogate PK, includes `is_simple_task` flag |
| `fact_agent_runs` | one row per agent run | grain = `run_id`; `date_key` is the run's day |
| `fact_customer_month` | one row per customer per active month | includes derived `llm_cost_usd` (summed from `fact_agent_runs`' parent runs for that customer + month) |

## Relationships (build these in Power BI's Model view)

- `fact_agent_runs[customer_key]` → `dim_customer[customer_key]` (many-to-one)
- `fact_agent_runs[date_key]` → `dim_date[date_key]` (many-to-one)
- `fact_agent_runs[plan_key]` → `dim_plan[plan_key]` (many-to-one) — plan *at the time of the run*
- `fact_agent_runs[model_key]` → `dim_model[model_key]` (many-to-one)
- `fact_agent_runs[task_key]` → `dim_task[task_key]` (many-to-one)
- `fact_customer_month[customer_key]` → `dim_customer[customer_key]` (many-to-one)
- `fact_customer_month[date_key]` → `dim_date[date_key]` (many-to-one) — always the 1st of the month
- `fact_customer_month[plan_key]` → `dim_plan[plan_key]` (many-to-one)

All relationships are single-direction (dimension → fact) to keep filter
context predictable. Mark `dim_date[date]` as the model's Date Table.

## Suggested DAX measures

```
MRR :=
CALCULATE(
    SUM(fact_customer_month[monthly_fee_usd]),
    dim_plan[plan] <> "Free"
)
```

```
Gross Margin % :=
VAR PaidRevenue =
    CALCULATE(SUM(fact_customer_month[revenue_usd]), dim_plan[plan] <> "Free")
VAR PaidLLMCost =
    CALCULATE(SUM(fact_customer_month[llm_cost_usd]), dim_plan[plan] <> "Free")
RETURN
    DIVIDE(PaidRevenue - PaidLLMCost, PaidRevenue)
```

```
NRR :=
VAR StartMRR =
    CALCULATE(
        [MRR],
        DATEADD(dim_date[date], -12, MONTH),
        dim_customer[first_paid_date] <> BLANK()
    )
VAR EndMRR = [MRR]
RETURN
    DIVIDE(EndMRR, StartMRR)
```

```
Success Rate :=
DIVIDE(
    CALCULATE(COUNTROWS(fact_agent_runs), fact_agent_runs[status] = "success"),
    COUNTROWS(fact_agent_runs)
)
```

```
Cost per Successful Run :=
DIVIDE(
    SUM(fact_agent_runs[llm_cost_usd]),
    CALCULATE(COUNTROWS(fact_agent_runs), fact_agent_runs[status] = "success")
)
```

```
Activation Rate :=
VAR FirstFive =
    FILTER(
        ADDCOLUMNS(
            fact_agent_runs,
            "@rn",
            RANKX(
                FILTER(fact_agent_runs, EARLIER(fact_agent_runs[customer_key]) = fact_agent_runs[customer_key]),
                fact_agent_runs[started_at],
                ,
                ASC
            )
        ),
        [@rn] <= 5
    )
VAR PerCustomer =
    ADDCOLUMNS(
        SUMMARIZE(FirstFive, dim_customer[customer_key]),
        "@SuccessShare",
        AVERAGEX(FILTER(FirstFive, fact_agent_runs[customer_key] = dim_customer[customer_key]),
            IF(fact_agent_runs[status] = "success", 1, 0))
    )
RETURN
    DIVIDE(COUNTROWS(FILTER(PerCustomer, [@SuccessShare] >= 0.6)), COUNTROWS(PerCustomer))
```

(`Activation Rate` is easiest to pre-compute in Power Query or a calculated
table if `RANKX`-per-partition proves slow on the full fact table — the DAX
above documents the intended logic per the shared metric definitions in
ARCHITECTURE.md: success share of a customer's first 5 runs by `started_at`;
activated = success share ≥ 3/5.)

## Data notes

- All data is **synthetic**, generated by `src/generate_data.py` for a
  fictional company ("CodeShift"). No real customers, revenue, or LLM usage.
- "Simple" task types (`is_simple_task` in `dim_task`): {simple_tasks}.
- A "paid month" is any `fact_customer_month` row whose `dim_plan[plan] <> "Free"`.
"""


def write_readme(out_dir: Path) -> None:
    text = README_TEMPLATE.format(simple_tasks=", ".join(SIMPLE_TASKS))
    (out_dir / "README.md").write_text(text)


# --------------------------------------------------------------------- main


def export(out_dir: Path) -> dict[str, pd.DataFrame]:
    """Build the full star schema and write every table as a CSV under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        dim_customer = build_dim_customer(conn)
        dim_plan = build_dim_plan(conn)
        dim_model = build_dim_model(conn)
        dim_task = build_dim_task(conn)
        dim_date = build_dim_date(conn)

        customer_key_map = dict(zip(dim_customer["customer_id"], dim_customer["customer_key"]))
        fact_agent_runs = build_fact_agent_runs(conn, dim_plan, dim_model, dim_task, customer_key_map)
        fact_customer_month = build_fact_customer_month(conn, dim_plan, customer_key_map)

    tables = {
        "dim_customer": dim_customer,
        "dim_date": dim_date,
        "dim_plan": dim_plan,
        "dim_model": dim_model,
        "dim_task": dim_task,
        "fact_agent_runs": fact_agent_runs,
        "fact_customer_month": fact_customer_month,
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    write_readme(out_dir)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory for the star-schema CSVs (default: data/powerbi/)",
    )
    args = parser.parse_args()
    tables = export(args.out)
    total_rows = sum(len(df) for df in tables.values())
    print(f"Wrote {len(tables)} tables ({total_rows:,} rows total) + README.md to {args.out}")


if __name__ == "__main__":
    main()
