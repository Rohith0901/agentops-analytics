"""Tests for src/powerbi_export.py: star-schema integrity and the CSV/formula-injection guard.

Every test that touches the real database is skipped if `data/agentops.db`
doesn't exist yet (per ARCHITECTURE.md: "never the full DB" isn't quite right
here — the star-schema export legitimately needs the real, generated
dataset — but we still skip cleanly rather than fail when it's absent, e.g.
before `make data` has run). Outputs are always written into pytest's
`tmp_path`, never into the shared `data/powerbi/` directory.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import powerbi_export as pbi  # noqa: E402
from db import DB_PATH  # noqa: E402

requires_db = pytest.mark.skipif(
    not DB_PATH.exists(), reason="data/agentops.db not generated yet — run `make data` first"
)


@pytest.fixture(scope="module")
def exported_tables(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("powerbi_out")
    return pbi.export(out_dir), out_dir


# --------------------------------------------------------------------- sanitize_cell


@pytest.mark.parametrize(
    "raw",
    [
        "=cmd|' /C calc'!A0",
        "+1+1",
        "-2+3",
        "@SUM(1,2)",
        "\tsneaky",
        "\rsneaky",
    ],
)
def test_sanitize_cell_neutralises_dangerous_prefixes(raw):
    out = pbi.sanitize_cell(raw)
    assert isinstance(out, str)
    assert out.startswith("'")
    assert out == "'" + raw


@pytest.mark.parametrize(
    "raw",
    ["Acme Corp", "Bluewave Solutions", "SaaS", "North America", "e-commerce (retail)"],
)
def test_sanitize_cell_leaves_normal_strings_untouched(raw):
    assert pbi.sanitize_cell(raw) == raw


@pytest.mark.parametrize("raw", [-5, -5.25, 0, 42, 3.14])
def test_sanitize_cell_leaves_numbers_untouched(raw):
    out = pbi.sanitize_cell(raw)
    assert out == raw
    assert type(out) is type(raw)


def test_sanitize_cell_passes_through_none():
    assert pbi.sanitize_cell(None) is None


def test_sanitize_frame_only_touches_requested_or_object_columns():
    df = pd.DataFrame({"text": ["=evil", "fine"], "num": [-1, 2]})
    out = pbi.sanitize_frame(df)
    assert out["text"].tolist() == ["'=evil", "fine"]
    assert out["num"].tolist() == [-1, 2]


# --------------------------------------------------------------------- files exist


@requires_db
def test_all_expected_files_written(exported_tables):
    _, out_dir = exported_tables
    expected = {
        "dim_customer.csv", "dim_date.csv", "dim_plan.csv", "dim_model.csv",
        "dim_task.csv", "fact_agent_runs.csv", "fact_customer_month.csv", "README.md",
    }
    present = {p.name for p in out_dir.iterdir()}
    assert expected <= present


@requires_db
def test_readme_mentions_relationships_and_dax(exported_tables):
    _, out_dir = exported_tables
    text = (out_dir / "README.md").read_text()
    assert "Relationships" in text
    for measure in ["MRR", "Gross Margin", "NRR", "Success Rate", "Cost per Successful Run", "Activation Rate"]:
        assert measure in text


# --------------------------------------------------------------------- surrogate keys unique


@requires_db
@pytest.mark.parametrize("table,key", [
    ("dim_customer", "customer_key"),
    ("dim_date", "date_key"),
    ("dim_plan", "plan_key"),
    ("dim_model", "model_key"),
    ("dim_task", "task_key"),
])
def test_dim_surrogate_keys_are_unique(exported_tables, table, key):
    tables, _ = exported_tables
    df = tables[table]
    assert df[key].is_unique
    assert df[key].notna().all()


@requires_db
def test_fact_agent_runs_run_id_is_unique(exported_tables):
    tables, _ = exported_tables
    assert tables["fact_agent_runs"]["run_id"].str.lstrip("'").is_unique


# --------------------------------------------------------------------- FK integrity


@requires_db
def test_fact_agent_runs_foreign_keys_resolve(exported_tables):
    tables, _ = exported_tables
    runs = tables["fact_agent_runs"]
    assert runs["customer_key"].notna().all()
    assert runs["customer_key"].isin(tables["dim_customer"]["customer_key"]).all()
    assert runs["date_key"].isin(tables["dim_date"]["date_key"]).all()
    assert runs["plan_key"].isin(tables["dim_plan"]["plan_key"]).all()
    assert runs["model_key"].isin(tables["dim_model"]["model_key"]).all()
    assert runs["task_key"].isin(tables["dim_task"]["task_key"]).all()


@requires_db
def test_fact_customer_month_foreign_keys_resolve(exported_tables):
    tables, _ = exported_tables
    cm = tables["fact_customer_month"]
    assert cm["customer_key"].notna().all()
    assert cm["customer_key"].isin(tables["dim_customer"]["customer_key"]).all()
    assert cm["date_key"].isin(tables["dim_date"]["date_key"]).all()
    assert cm["plan_key"].isin(tables["dim_plan"]["plan_key"]).all()


# --------------------------------------------------------------------- row counts match DB


@requires_db
def test_row_counts_match_source_tables(exported_tables):
    tables, _ = exported_tables
    with sqlite3.connect(DB_PATH) as conn:
        n_customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        n_runs = conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
        n_cm = conn.execute("SELECT COUNT(*) FROM customer_months").fetchone()[0]
        n_plans = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        n_models = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        n_tasks = conn.execute("SELECT COUNT(DISTINCT task_type) FROM agent_runs").fetchone()[0]

    assert len(tables["dim_customer"]) == n_customers
    assert len(tables["fact_agent_runs"]) == n_runs
    assert len(tables["fact_customer_month"]) == n_cm
    assert len(tables["dim_plan"]) == n_plans
    assert len(tables["dim_model"]) == n_models
    assert len(tables["dim_task"]) == n_tasks


@requires_db
def test_fact_customer_month_llm_cost_matches_definition(exported_tables):
    """Spot-check the shared metric definition: LLM cost per customer-month =
    SUM(agent_runs.llm_cost_usd) grouped by customer_id and month."""
    tables, _ = exported_tables
    cm = tables["fact_customer_month"]
    with sqlite3.connect(DB_PATH) as conn:
        expected = pd.read_sql_query(
            """
            SELECT customer_id, substr(started_at, 1, 7) || '-01' AS month,
                   SUM(llm_cost_usd) AS llm_cost_usd
            FROM agent_runs GROUP BY customer_id, substr(started_at, 1, 7)
            """,
            conn,
        )
    total_expected = expected["llm_cost_usd"].sum()
    total_actual = cm["llm_cost_usd"].sum()
    assert total_actual == pytest.approx(total_expected, rel=1e-9)


@requires_db
def test_dim_date_covers_full_range_with_no_gaps(exported_tables):
    tables, _ = exported_tables
    dates = pd.to_datetime(tables["dim_date"]["date"])
    diffs = dates.sort_values().diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all()
