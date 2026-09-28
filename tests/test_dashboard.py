"""tests/test_dashboard.py — tests for the Streamlit dashboard (Builder D).

Covers:
- The app runs end-to-end via `streamlit.testing.v1.AppTest` with no exceptions
  and all six tabs render.
- `dashboard/data.py` unit tests: filters work, invalid/unknown filter values
  are silently dropped (never used in a query), a value containing a quote or
  semicolon cannot alter query results (no SQL injection), and the dashboard's
  connection helper is provably read-only.
- `.streamlit/config.toml` carries every required security setting.
- A live `streamlit run` smoke test: launch the real server, curl it for a
  200, then tear it down.
"""

from __future__ import annotations

import socket
import urllib.request
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

import data as d  # noqa: E402  (path must be extended first)


# --------------------------------------------------------------------------
# App-level smoke test (AppTest)
# --------------------------------------------------------------------------


def test_app_runs_without_exceptions_and_renders_tabs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
    assert len(at.tabs) == 6


# --------------------------------------------------------------------------
# dashboard/data.py — filters
# --------------------------------------------------------------------------


def test_filter_options_come_from_db_allowlists():
    options = d.get_filter_options()
    for key in ("months", "plans", "regions", "channels", "industries", "models", "task_types"):
        assert key in options
        assert isinstance(options[key], list)
        assert len(options[key]) > 0
    assert "Free" in options["plans"]


def test_invalid_filter_values_are_dropped():
    options = d.get_filter_options()
    real_region = options["regions"][0]
    f = d.Filters.build(
        months=[],
        plans=["Not A Real Plan", "'; DROP TABLE customers; --"],
        regions=[real_region, "Nowhere Land"],
        channels=["Fake Channel"],
        industries=[],
        options=options,
    )
    assert f.plans == ()
    assert f.regions == (real_region,)
    assert f.channels == ()


def test_quote_or_semicolon_value_does_not_alter_results():
    """A value containing a quote/semicolon must not change the result set —
    it should simply fail the allow-list check and be dropped, same as any
    other unknown value."""
    options = d.get_filter_options()
    baseline = d.Filters.build([], [], [], [], [], options)
    injected = d.Filters.build(
        months=[],
        plans=[],
        regions=["' OR '1'='1"],
        channels=["x'; DROP TABLE customers; --"],
        industries=[],
        options=options,
    )
    assert injected.regions == ()
    assert injected.channels == ()

    base_df = d.monthly_financials(baseline)
    inj_df = d.monthly_financials(injected)
    pd_assert_frame_equal_ignoring_index(base_df, inj_df)


def pd_assert_frame_equal_ignoring_index(a, b):
    import pandas as pd

    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False
    )


def test_valid_filter_values_pass_through():
    options = d.get_filter_options()
    plan = options["plans"][0]
    f = d.Filters.build([], [plan], [], [], [], options)
    assert f.plans == (plan,)


# --------------------------------------------------------------------------
# dashboard/data.py — read-only connection
# --------------------------------------------------------------------------


def test_connection_is_read_only():
    import sqlite3

    conn = d.get_read_only_conn()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE should_fail (x INTEGER)")
    finally:
        conn.close()


def test_read_only_connection_can_still_select():
    conn = d.get_read_only_conn()
    try:
        cur = conn.execute("SELECT COUNT(*) FROM customers")
        assert cur.fetchone()[0] > 0
    finally:
        conn.close()


# --------------------------------------------------------------------------
# dashboard/data.py — a sampling of the query functions run without error
# --------------------------------------------------------------------------


def test_query_functions_return_dataframes():
    options = d.get_filter_options()
    f = d.Filters.build(options["months"][:3], [], [], [], [], options)

    assert not d.monthly_financials(f).empty
    assert not d.run_success_rate(f).empty
    assert not d.model_task_success(f).empty
    assert not d.cost_per_successful_run(f).empty
    assert not d.margin_by_plan(f).empty
    assert not d.cost_revenue_per_credit_by_task(f).empty
    assert not d.simple_task_frontier_stats(f).empty

    exp = d.experiment_conversion(f)
    assert set(exp["variant"]) <= {"control", "treatment"}


def test_db_exists_and_missing_db_message():
    assert d.db_exists() is True
    # Simulate a missing DB path without touching the real file.
    old_path = d.DB_PATH
    try:
        d.DB_PATH = ROOT / "data" / "does_not_exist.db"
        assert d.db_exists() is False
    finally:
        d.DB_PATH = old_path


# --------------------------------------------------------------------------
# .streamlit/config.toml — mandatory security settings
# --------------------------------------------------------------------------


def test_streamlit_config_has_required_security_settings():
    cfg_path = ROOT / ".streamlit" / "config.toml"
    assert cfg_path.exists()
    cfg = tomllib.loads(cfg_path.read_text())

    assert cfg["server"]["enableXsrfProtection"] is True
    assert cfg["server"]["enableCORS"] is False
    assert cfg["server"]["headless"] is True
    assert cfg["server"]["fileWatcherType"] == "none"
    assert cfg["browser"]["gatherUsageStats"] is False
    assert cfg["client"]["toolbarMode"] == "viewer"
    assert cfg["client"]["showErrorDetails"] is False


# --------------------------------------------------------------------------
# Live server smoke test
# --------------------------------------------------------------------------


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _free_port() -> int:
    """Ask the OS for an unused port so parallel test runs never collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_live_server_smoke():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", str(port), "--server.address", "127.0.0.1",
         "--server.headless", "true"],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,  # not PIPE: an undrained pipe can block the server
        stderr=subprocess.DEVNULL,
    )
    try:
        # Generous deadline: cold Streamlit start-up is slow on a busy machine.
        deadline = time.time() + 90
        status = None
        while time.time() < deadline and proc.poll() is None:
            if _port_open(port):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=10) as resp:  # nosec B310 - fixed local http URL
                        status = resp.status
                    break
                except OSError:
                    pass
            time.sleep(0.5)
        assert status == 200, f"streamlit server not healthy (status={status}, exit={proc.poll()})"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
