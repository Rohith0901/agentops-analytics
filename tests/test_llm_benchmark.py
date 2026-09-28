"""Tests for llm_benchmark: grading correctness and sandbox containment.

The containment tests use harmless probes (a sleep loop, a local file write, an
env-var read) to confirm the grading subprocess keeps untrusted model output
away from the host.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm_benchmark"))

import run_benchmark as rb  # noqa: E402
from tasks import TASKS  # noqa: E402

SIMPLE_TESTS = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]


def grade(code: str, tests=SIMPLE_TESTS, timeout: int = 5) -> dict:
    return rb.grade_in_subprocess(code, tests, timeout)


# --------------------------------------------------------------------- grading


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_reference_passes_and_buggy_fails(task):
    assert grade(task["reference_solution"], task["tests"], 10)["status"] == "success"
    assert grade(task["buggy_solution"], task["tests"], 10)["status"] != "success"


def test_statuses():
    assert grade("def add(a, b):\n    return a + b\n")["status"] == "success"
    partial = grade("def add(a, b):\n    return a + b + (1 if a < 0 else 0)\n")
    assert partial["status"] == "partial"
    assert 0 < partial["tests_passed"] < partial["tests_total"]
    assert grade("def add(a, b):\n    return None\n")["status"] == "failed"
    assert grade("def add(a, b)\n    return a + b\n")["status"] == "failed"  # syntax error


def test_extract_code_variants():
    fenced, clean = rb.extract_code("Here:\n```python\ndef f():\n    return 1\n```\nDone.")
    assert "def f()" in fenced and clean
    unfenced, _ = rb.extract_code("def f():\n    return 1\n")
    assert "def f()" in unfenced
    garbage, clean = rb.extract_code("I cannot help with that.")
    assert not clean


# ----------------------------------------------------------------- containment


def test_infinite_loop_is_killed():
    result = grade("while True:\n    pass\n", timeout=2)
    assert result["status"] == "failed"
    assert result["error_type"] in {"Timeout", "RunnerCrash"}


def test_write_outside_sandbox_is_blocked(tmp_path):
    target = tmp_path / "should_not_exist.txt"
    grade(f"open({str(target)!r}, 'w').write('x')\ndef add(a, b):\n    return a + b\n")
    assert not target.exists()


def test_read_outside_sandbox_is_blocked(tmp_path):
    secret = tmp_path / "private.txt"
    secret.write_text("top-secret")
    code = (
        f"DATA = open({str(secret)!r}).read()\n"
        "def add(a, b):\n    return a + b\n"
    )
    assert grade(code)["status"] == "failed"


def test_parent_env_not_inherited(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    code = "import os\ndef add(a, b):\n    assert 'SECRET_TOKEN' not in os.environ\n    return a + b\n"
    assert grade(code)["status"] == "success"


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.run(['true'])\n",
        "import os\nos.system('true')\n",
        "import socket\nsocket.socket()\n",
    ],
    ids=["subprocess", "os.system", "socket"],
)
def test_process_and_network_calls_are_blocked(code):
    result = grade(code + "def add(a, b):\n    return a + b\n")
    assert result["status"] == "failed"


def test_large_output_is_capped():
    result = grade("import sys\nsys.stdout.write('x' * 5_000_000)\ndef add(a, b):\n    return a + b\n")
    # Either the run is cut off, or it finishes; the parent must never buffer it all.
    assert result["status"] in {"success", "failed"}


# ----------------------------------------------------------------- host checks


@pytest.mark.parametrize("host", ["http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_local_hosts_allowed(host):
    rb.validate_host(host, allow_remote=False)


@pytest.mark.parametrize("host", ["http://example.com:11434", "http://10.0.0.5", "file:///etc/passwd", "ftp://localhost"])
def test_remote_or_odd_hosts_rejected(host):
    with pytest.raises(SystemExit):
        rb.validate_host(host, allow_remote=False)


# ------------------------------------------------------------------ end to end


def test_mock_end_to_end(tmp_path):
    out = tmp_path / "results.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "llm_benchmark" / "run_benchmark.py"), "--models", "demo-model",
         "--repeats", "1", "--mock", "--out", str(out)],
        check=True, capture_output=True, timeout=300,
    )
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(TASKS)
    assert {"model", "task_id", "status", "tests_passed", "tests_total", "cost_usd"} <= set(rows[0])
    assert all(r["model"] == "MOCK/demo-model" for r in rows)
    assert all(int(r["tests_passed"]) <= int(r["tests_total"]) for r in rows)
