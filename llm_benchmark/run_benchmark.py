#!/usr/bin/env python
"""
run_benchmark.py — CLI harness that runs the task bank (tasks.py) against one or
more local Ollama models, grades the results in a sandboxed subprocess, and writes
a CSV shaped so it can sit next to CodeShift's `agent_runs` table.

Usage
-----
    # Against a real local Ollama server (http://localhost:11434):
    .venv/bin/python llm_benchmark/run_benchmark.py \\
        --models llama3.2 qwen2.5-coder:1.5b \\
        --repeats 3 \\
        --out llm_benchmark/results/results.csv

    # Fully offline, no Ollama required (simulates plausible model behavior):
    .venv/bin/python llm_benchmark/run_benchmark.py \\
        --models llama3.2 qwen2.5-coder:1.5b mistral \\
        --repeats 3 --mock \\
        --out llm_benchmark/results/results.csv

Each (model, task, repeat) triple becomes one CSV row, mirroring one `agent_runs`
row in the main dataset: model, task_type, tokens in/out, latency, cost, and
success/partial/failed status with tests_passed/tests_total.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tasks import TASKS  # noqa: E402

# --------------------------------------------------------------------------- config

DEFAULT_HOST = "http://localhost:11434"
SUBPROCESS_TIMEOUT_SEC = 10

# ----------------------------------------------------------------- sandbox hardening
#
# Threat model / honesty note: run_benchmark.py executes *untrusted, LLM-generated
# Python*. The controls below (isolated interpreter flags, a stripped environment,
# POSIX resource limits, an audit hook, and a hard wall-clock + output-size cap
# enforced by killing the whole process group) are defense in depth for a
# single-developer benchmark harness running on a trusted machine. They are a
# *speed bump*, not a real security boundary: a determined attacker with arbitrary
# code execution inside CPython can find gaps in an audit-hook-based sandbox
# (e.g. via low-level bytecode tricks, or resource exhaustion patterns the rlimits
# don't cover). For untrusted code at scale, or in any multi-tenant/CI setting,
# use a real isolation boundary instead — e.g. Docker/Podman with
# `--network none --read-only --pids-limit --memory` (see --docker below), a VM,
# or gVisor/nsjail. This module always keeps that recommendation next to the code.

# Grading subprocess gets a minimal, secret-free environment: PATH so `python`'s
# own dynamic linker / dlopen lookups keep working, and a fixed C locale so stdlib
# text handling is deterministic. Nothing else is inherited (no API keys, no
# HOME-based config, no PYTHONPATH).
_CHILD_PATH = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

# Hard ceiling on captured stdout/stderr from the grading subprocess, so a
# candidate that tries to print gigabytes of output can't OOM the parent.
MAX_OUTPUT_BYTES = 200_000

# Hard ceiling on an Ollama HTTP response body, so a malicious/misbehaving server
# can't stream unbounded data into the parent's memory.
MAX_HTTP_RESPONSE_BYTES = 25_000_000

# Hosts the harness will talk to without an explicit opt-in. Ollama's HTTP API
# has no auth by default, so pointing this at an arbitrary remote host is an
# SSRF-shaped foot-gun (a malicious/typo'd --host could be used to probe internal
# services from whatever machine runs this benchmark).
_LOCAL_HOSTNAMES = {"localhost"}
_LOCAL_IPS = {"127.0.0.1", "::1"}

# $ per million tokens, chosen to sit in the same range as the main dataset's
# `models` price table (frontier-large / balanced-medium / fast-small /
# open-weights-hosted) so LLM benchmark costs are comparable to agent_runs.llm_cost_usd.
# Ollama models are all "open-weights, self-hosted" in spirit, so we default them
# into that price tier unless the user supplies --price-table.
DEFAULT_PRICE_TABLE = {
    "llama3.2": {"usd_per_m_input": 0.15, "usd_per_m_output": 0.60},
    "llama3.2:1b": {"usd_per_m_input": 0.10, "usd_per_m_output": 0.40},
    "llama3.1": {"usd_per_m_input": 0.20, "usd_per_m_output": 0.80},
    "qwen2.5-coder:1.5b": {"usd_per_m_input": 0.10, "usd_per_m_output": 0.40},
    "qwen2.5-coder:7b": {"usd_per_m_input": 0.18, "usd_per_m_output": 0.70},
    "mistral": {"usd_per_m_input": 0.20, "usd_per_m_output": 0.75},
    "codellama": {"usd_per_m_input": 0.18, "usd_per_m_output": 0.70},
    # fallback tier used for any model not listed above (see get_price)
    "_default": {"usd_per_m_input": 0.15, "usd_per_m_output": 0.60},
}

# Runner script executed *inside the sandboxed subprocess*. It loads the model's
# extracted code, then executes each hidden test independently (a fresh copy of
# the candidate's namespace per test) so a bug in one test doesn't cascade into
# false failures for the others. This is what makes "partial" status possible.
#
# Before exec'ing a single byte of candidate code, it installs a `sys.addaudithook`
# tripwire (see module docstring / README "Security model" for the honesty caveat:
# this is a speed bump, not a real sandbox) that blocks the obvious escape hatches
# a generated snippet might reach for: opening a socket, shelling out via
# subprocess/os.system/os.exec*/os.posix_spawn, forking, loading arbitrary native
# code via ctypes, or reading/writing files outside this attempt's temp directory
# (stdlib/site import paths are allow-listed for reads so `import` keeps working).
_RUNNER_SRC = r'''
import sys, os, json, traceback

def _install_audit_hook():
    allowed_root = os.path.realpath(os.path.dirname(os.path.abspath(sys.argv[1])))
    read_ok_roots = tuple(
        os.path.realpath(p) for p in list(sys.path) + [sys.prefix, sys.base_prefix,
                                                          sys.exec_prefix, sys.base_exec_prefix]
        if p
    )

    def _under(path, roots):
        try:
            rp = os.path.realpath(path)
        except Exception:
            return False
        return any(rp == r or rp.startswith(r + os.sep) for r in roots)

    def hook(event, args):
        if event in ("os.system", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn"):
            raise PermissionError("blocked by sandbox audit hook: " + event)
        if event.startswith("subprocess."):
            raise PermissionError("blocked by sandbox audit hook: " + event)
        if event.startswith("socket.") or event == "socket.gethostbyname":
            raise PermissionError("blocked by sandbox audit hook: " + event)
        if event.startswith("ctypes."):
            raise PermissionError("blocked by sandbox audit hook: " + event)
        if event == "open":
            path, mode, flags = (args + (None, None, None))[:3]
            mode = mode or ""
            is_write = any(c in mode for c in "wax+") or bool(
                flags and (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
            )
            if is_write:
                if not _under(path, (allowed_root,)):
                    raise PermissionError("blocked by sandbox audit hook: write outside sandbox: " + str(path))
            else:
                if not _under(path, (allowed_root,) + read_ok_roots):
                    raise PermissionError("blocked by sandbox audit hook: read outside sandbox: " + str(path))

    sys.addaudithook(hook)


def main():
    _install_audit_hook()
    candidate_path, tests_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3]
    tests = json.load(open(tests_path, "r", encoding="utf-8"))
    result = {
        "tests_total": len(tests),
        "tests_passed": 0,
        "status": "failed",
        "error_type": None,
        "first_error": None,
    }

    namespace = {}
    try:
        code = open(candidate_path, "r", encoding="utf-8").read()
        exec(compile(code, "candidate.py", "exec"), namespace)
    except BaseException as e:
        result["error_type"] = type(e).__name__
        result["first_error"] = str(e)[:300]
        json.dump(result, open(result_path, "w"))
        return

    passed = 0
    first_err = None
    first_err_type = None
    for t in tests:
        try:
            exec(compile(t, "<hidden_test>", "exec"), dict(namespace))
            passed += 1
        except BaseException as e:
            if first_err is None:
                first_err = str(e)[:300]
                first_err_type = type(e).__name__

    result["tests_passed"] = passed
    result["error_type"] = first_err_type
    result["first_error"] = first_err
    if len(tests) > 0 and passed == len(tests):
        result["status"] = "success"
    elif passed > 0:
        result["status"] = "partial"
    else:
        result["status"] = "failed"

    json.dump(result, open(result_path, "w"))

main()
'''


@dataclass
class RunResult:
    """One row of the output CSV — one (model, task, repeat) attempt."""

    model: str
    task_id: str
    task_type: str
    repeat: int
    status: str  # success | partial | failed
    tests_passed: int
    tests_total: int
    error_type: str | None
    prompt_tokens: int  # Ollama's prompt_eval_count
    output_tokens: int  # Ollama's eval_count
    total_duration_sec: float
    eval_duration_sec: float
    tokens_per_sec: float
    cost_usd: float
    code_extracted: bool
    timestamp: str


# --------------------------------------------------------------------------- pricing


def get_price(model: str, price_table: dict) -> dict:
    """Look up {usd_per_m_input, usd_per_m_output} for a model, falling back to
    the generic open-weights-hosted tier if the model isn't in the table."""
    return price_table.get(model, price_table.get("_default", DEFAULT_PRICE_TABLE["_default"]))


def compute_cost(prompt_tokens: int, output_tokens: int, price: dict) -> float:
    cost = (prompt_tokens / 1_000_000) * price["usd_per_m_input"]
    cost += (output_tokens / 1_000_000) * price["usd_per_m_output"]
    return round(cost, 8)


# ------------------------------------------------------------------------- ollama I/O


def validate_host(host: str, allow_remote: bool) -> None:
    """Guard against SSRF-shaped misuse of --host: Ollama's HTTP API has no
    auth by default, so this harness will silently POST prompts to, and read
    responses from, whatever --host points at. By default we only allow
    http(s) to localhost/127.0.0.1/::1; anything else needs an explicit
    --allow-remote-host opt-in. Raises SystemExit with a clear message."""
    parsed = urllib.parse.urlsplit(host)
    if parsed.scheme not in ("http", "https"):
        sys.exit(f"[run_benchmark] --host must be http:// or https://, got {host!r}")
    hostname = (parsed.hostname or "").lower()
    if allow_remote:
        return
    is_local = hostname in _LOCAL_HOSTNAMES or hostname in _LOCAL_IPS
    if not is_local:
        try:
            is_local = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_local = False
    if not is_local:
        sys.exit(
            f"[run_benchmark] --host {host!r} is not localhost. Refusing to send prompts to a "
            f"remote, unauthenticated Ollama-style endpoint by default (this could be used for "
            f"SSRF-style probing of other hosts). Pass --allow-remote-host if you really mean it."
        )


def call_ollama(host: str, model: str, prompt: str, timeout_sec: int = 60) -> dict:
    """Call Ollama's /api/generate with stream=false. Returns the parsed JSON
    response, which includes 'response', 'prompt_eval_count', 'eval_count',
    'total_duration' (ns), 'eval_duration' (ns).

    Raises RuntimeError with a clear, actionable message if the server is
    unreachable — callers should catch this and suggest --mock.
    """
    url = host.rstrip("/") + "/api/generate"
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310 - scheme/host restricted to http(s) localhost by validate_host()
            # Bound how much we'll ever read into memory from a single response,
            # in case the server (or a misconfigured/malicious --host) streams
            # back something huge.
            raw = resp.read(MAX_HTTP_RESPONSE_BYTES + 1)
            if len(raw) > MAX_HTTP_RESPONSE_BYTES:
                raise RuntimeError(
                    f"response from {host} exceeded {MAX_HTTP_RESPONSE_BYTES} bytes; refusing to read further"
                )
            return json.loads(raw.decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host} (model={model!r}): {e.reason}. "
            f"Is the Ollama server running? Start it with `ollama serve` and pull "
            f"the model with `ollama pull {model}`, or re-run with --mock to test "
            f"the pipeline fully offline without a server."
        ) from e
    except Exception as e:  # malformed JSON, HTTP error body, etc.
        raise RuntimeError(
            f"Ollama call failed for model={model!r} at {host}: {e}. "
            f"Re-run with --mock to test the pipeline offline."
        ) from e


# --------------------------------------------------------------------- code extraction

_CODE_FENCE_RE = re.compile(r"```(?:python|python3|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> tuple[str, bool]:
    """Pull a Python code block out of a model's free-text response.

    Tries fenced ```python blocks first (the common case for real models).
    Falls back to "everything from the first top-level `def`/`import` onward"
    for models that forget the fence (also exercised by --mock's 'malformed'
    category). Returns (code, extracted_cleanly).
    """
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        # Prefer the block containing a function definition, else take the first.
        for m in matches:
            if "def " in m:
                return m.strip(), True
        return matches[0].strip(), True

    # No fences at all — look for a def/import line to salvage something.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "import ", "from ")):
            return "\n".join(lines[i:]).strip(), False

    # Nothing recognizable as code.
    return text.strip(), False


# --------------------------------------------------------------------------- grading

_IS_POSIX = os.name == "posix"


def _child_env() -> dict:
    """A minimal, secret-free environment for the grading subprocess: no
    inherited API keys/tokens/config, just enough PATH to resolve the
    interpreter's own dynamic-linking needs and a fixed locale."""
    return {"PATH": _CHILD_PATH, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


def _make_preexec_fn(cpu_seconds: int):
    """Best-effort POSIX resource limits applied in the child right after fork,
    before the runner script is exec'd. Every limit is set defensively (each in
    its own try/except) because platforms disagree on what's supported —
    notably macOS rejects/ignores some RLIMIT_AS configurations and treats
    RLIMIT_NPROC as a per-*user* limit rather than per-process, so we lower it
    cautiously (relative to the current limit) instead of pinning an absolute
    value that could starve the rest of the user's processes."""

    def _preexec():
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # no core dumps
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))  # ~1MB max file size
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except Exception:
            pass
        try:
            # 1 GiB address space cap. Skipped gracefully where the platform
            # rejects RLIMIT_AS (some macOS configurations do).
            mem_bytes = 1 * 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
        try:
            # Guard against fork bombs without clobbering the parent's own
            # (per-user, on macOS) process budget: tighten relative to
            # whatever the current soft/hard limits already are.
            soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            cap = 32
            new_soft = cap if soft in (resource.RLIM_INFINITY,) or soft > cap else soft
            resource.setrlimit(resource.RLIMIT_NPROC, (new_soft, hard))
        except Exception:
            pass

    return _preexec


def _read_capped(stream, cap: int) -> tuple[bytes, bool]:
    """Read from a subprocess pipe into memory up to `cap` bytes, then keep
    draining (and discarding) so the child never blocks on a full pipe buffer.
    Returns (captured_bytes, was_truncated)."""
    chunks: list[bytes] = []
    total = 0
    truncated = False
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            if total < cap:
                chunks.append(chunk[: cap - total])
            else:
                truncated = True
            total += len(chunk)
            if total > cap:
                truncated = True
    except Exception:
        pass
    return b"".join(chunks), truncated


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the whole process group the grading subprocess started (defense in
    depth beyond killing just the direct child: the audit hook is meant to
    block fork/exec/subprocess entirely, but if it's ever bypassed we still
    want a runaway descendant reaped on timeout)."""
    try:
        if _IS_POSIX:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def grade_in_subprocess(code: str, tests: list[str], timeout_sec: int) -> dict:
    """Write the candidate code + hidden tests to a temp dir and grade them in a
    hardened subprocess: isolated Python interpreter (`-I -S`), a stripped
    secret-free environment, its own process group + a fresh temp directory as
    cwd, no stdin, POSIX resource limits (CPU/memory/file size/fd count/no core
    dumps/process count — see `_make_preexec_fn`), an in-process audit-hook
    tripwire against network/subprocess/exec/ctypes/out-of-sandbox file access
    (see `_RUNNER_SRC`), a hard wall-clock timeout, and a captured-output size
    cap — all enforced by killing the entire process group. See the module
    docstring and README "Security model" section: this is defense in depth
    for a trusted single-developer machine, not a hardened multi-tenant
    sandbox; use Docker/VM isolation for untrusted code at scale."""
    with tempfile.TemporaryDirectory(prefix="llm_bench_") as tmpdir:
        tmp = Path(tmpdir)
        candidate_path = tmp / "candidate.py"
        tests_path = tmp / "tests.json"
        result_path = tmp / "result.json"
        runner_path = tmp / "_runner.py"

        candidate_path.write_text(code, encoding="utf-8")
        tests_path.write_text(json.dumps(tests), encoding="utf-8")
        runner_path.write_text(_RUNNER_SRC, encoding="utf-8")

        cmd = [sys.executable, "-I", "-S", str(runner_path), str(candidate_path), str(tests_path), str(result_path)]
        popen_kwargs = dict(
            cwd=tmpdir,
            env=_child_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if _IS_POSIX:
            popen_kwargs["start_new_session"] = True  # own process group, for killpg
            popen_kwargs["preexec_fn"] = _make_preexec_fn(cpu_seconds=timeout_sec)

        try:
            proc = subprocess.Popen(cmd, **popen_kwargs)
        except Exception as e:
            return {
                "tests_total": len(tests),
                "tests_passed": 0,
                "status": "failed",
                "error_type": "SpawnError",
                "first_error": f"could not start grading subprocess: {e}"[:300],
            }

        out_holder: dict = {}
        err_holder: dict = {}
        t_out = threading.Thread(target=lambda: out_holder.update(zip(("data", "truncated"), _read_capped(proc.stdout, MAX_OUTPUT_BYTES))))
        t_err = threading.Thread(target=lambda: err_holder.update(zip(("data", "truncated"), _read_capped(proc.stderr, MAX_OUTPUT_BYTES))))
        t_out.start()
        t_err.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

        t_out.join(timeout=5)
        t_err.join(timeout=5)
        truncated = out_holder.get("truncated") or err_holder.get("truncated")

        if timed_out:
            return {
                "tests_total": len(tests),
                "tests_passed": 0,
                "status": "failed",
                "error_type": "Timeout",
                "first_error": f"execution exceeded {timeout_sec}s",
            }

        if truncated and not result_path.exists():
            # Killed for flooding stdout/stderr before it could report a result.
            return {
                "tests_total": len(tests),
                "tests_passed": 0,
                "status": "failed",
                "error_type": "OutputLimitExceeded",
                "first_error": f"captured output exceeded {MAX_OUTPUT_BYTES} bytes",
            }

        if result_path.exists():
            try:
                return json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as e:
                return {
                    "tests_total": len(tests),
                    "tests_passed": 0,
                    "status": "failed",
                    "error_type": "ResultParseError",
                    "first_error": str(e)[:300],
                }

        # Runner itself crashed (e.g. segfault, killed by an rlimit) before
        # writing a result file.
        return {
            "tests_total": len(tests),
            "tests_passed": 0,
            "status": "failed",
            "error_type": "RunnerCrash",
            "first_error": "grading subprocess produced no result file",
        }


# ------------------------------------------------------------------------------ mock


MOCK_PROFILES = {
    # Rough, illustrative quality/speed tiers for a few common Ollama models.
    # Any --models name not listed here gets a deterministic profile derived
    # from a hash of its name, so the tool works with arbitrary model names.
    "llama3.2": {"p_correct": 0.55, "p_buggy": 0.30, "p_malformed": 0.15, "tokens_per_sec": 45},
    "qwen2.5-coder:1.5b": {"p_correct": 0.72, "p_buggy": 0.20, "p_malformed": 0.08, "tokens_per_sec": 60},
    "mistral": {"p_correct": 0.40, "p_buggy": 0.35, "p_malformed": 0.25, "tokens_per_sec": 38},
    "codellama": {"p_correct": 0.62, "p_buggy": 0.28, "p_malformed": 0.10, "tokens_per_sec": 40},
}


def mock_profile_for(model: str) -> dict:
    if model in MOCK_PROFILES:
        return MOCK_PROFILES[model]
    # Deterministic fallback profile from a hash of the model name, so unknown
    # model names still get consistent (not random-every-run) behavior.
    h = int(hashlib.sha256(model.encode()).hexdigest(), 16)
    p_correct = 0.35 + (h % 40) / 100.0  # in [0.35, 0.74]
    remainder = 1.0 - p_correct
    p_buggy = remainder * 0.7
    p_malformed = remainder * 0.3
    tokens_per_sec = 25 + (h % 50)
    return {"p_correct": p_correct, "p_buggy": p_buggy, "p_malformed": p_malformed, "tokens_per_sec": tokens_per_sec}


def mock_generate(model: str, task: dict, repeat: int, seed: int) -> dict:
    """Simulate an Ollama /api/generate response for --mock mode, without any
    network access. Produces a plausible mix of correct / subtly-buggy /
    malformed code, plus plausible token counts and timing, deterministically
    seeded so runs are reproducible."""
    key = f"{seed}:{model}:{task['id']}:{repeat}"
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32))
    profile = mock_profile_for(model)

    roll = rng.random()
    if roll < profile["p_correct"]:
        category = "correct"
    elif roll < profile["p_correct"] + profile["p_buggy"]:
        category = "buggy"
    else:
        category = "malformed"

    commentary = rng.choice(
        ["Here's the fix:", "Sure, here is the code:", "Here you go:", "Solution below:", ""]
    )

    if category == "correct":
        body = task["reference_solution"]
        text = f"{commentary}\n\n```python\n{body}\n```"
    elif category == "buggy":
        body = task["buggy_solution"]
        text = f"{commentary}\n\n```python\n{body}\n```"
    else:
        variant = rng.choice(["no_fence", "truncated", "bad_syntax"])
        body = task["reference_solution"]
        if variant == "no_fence":
            # Model forgets the code fence entirely (still recoverable via fallback).
            text = f"{commentary}\n{body}"
        elif variant == "truncated":
            lines = body.splitlines()
            cut = max(1, len(lines) // 2)
            text = f"{commentary}\n\n```python\n" + "\n".join(lines[:cut]) + "\n```"
        else:  # bad_syntax
            broken = body.replace(":", "", 1)  # drop the first colon -> SyntaxError
            text = f"{commentary}\n\n```python\n{broken}\n```"

    prompt_tokens = max(8, len(task["prompt"]) // 4)
    output_tokens = max(4, len(text) // 4)
    tok_per_sec = max(5.0, rng.gauss(profile["tokens_per_sec"], profile["tokens_per_sec"] * 0.15))
    eval_duration_sec = output_tokens / tok_per_sec
    # Prompt processing is typically much faster than generation; model this loosely.
    prompt_eval_duration_sec = prompt_tokens / (tok_per_sec * 6)
    total_duration_sec = prompt_eval_duration_sec + eval_duration_sec

    return {
        "response": text,
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
        "total_duration": int(total_duration_sec * 1e9),
        "eval_duration": int(eval_duration_sec * 1e9),
    }


# --------------------------------------------------------------------------- driver


def run_one(
    model: str,
    task: dict,
    repeat: int,
    host: str,
    mock: bool,
    seed: int,
    price_table: dict,
    grading_timeout: int,
) -> RunResult:
    if mock:
        resp = mock_generate(model, task, repeat, seed)
    else:
        resp = call_ollama(host, model, task["prompt"])

    text = resp.get("response", "")
    prompt_tokens = int(resp.get("prompt_eval_count", 0) or 0)
    output_tokens = int(resp.get("eval_count", 0) or 0)
    total_duration_sec = float(resp.get("total_duration", 0) or 0) / 1e9
    eval_duration_sec = float(resp.get("eval_duration", 0) or 0) / 1e9
    tokens_per_sec = (output_tokens / eval_duration_sec) if eval_duration_sec > 0 else 0.0

    code, extracted_cleanly = extract_code(text)
    grade = grade_in_subprocess(code, task["tests"], grading_timeout)

    price = get_price(model, price_table)
    cost = compute_cost(prompt_tokens, output_tokens, price)

    return RunResult(
        # Mock rows are tagged so simulated numbers can never pass as real model results.
        model=f"MOCK/{model}" if mock else model,
        task_id=task["id"],
        task_type=task["task_type"],
        repeat=repeat,
        status=grade["status"],
        tests_passed=grade["tests_passed"],
        tests_total=grade["tests_total"],
        error_type=grade.get("error_type"),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_duration_sec=round(total_duration_sec, 4),
        eval_duration_sec=round(eval_duration_sec, 4),
        tokens_per_sec=round(tokens_per_sec, 2),
        cost_usd=cost,
        code_extracted=extracted_cleanly,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model names, e.g. llama3.2 qwen2.5-coder:1.5b")
    parser.add_argument("--repeats", type=int, default=3, help="Samples per (model, task) pair")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "results.csv", help="Output CSV path")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server base URL")
    parser.add_argument(
        "--allow-remote-host",
        action="store_true",
        help="Allow --host to point at a non-localhost address (default: refused, to avoid SSRF-style misuse)",
    )
    parser.add_argument("--mock", action="store_true", help="Simulate model responses offline (no Ollama server needed)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --mock determinism")
    parser.add_argument("--timeout", type=int, default=SUBPROCESS_TIMEOUT_SEC, help="Grading subprocess timeout, seconds")
    parser.add_argument("--price-table", type=Path, default=None, help="Optional JSON file overriding the default $/M-token price table")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Restrict to specific task ids (default: all)")
    args = parser.parse_args()

    price_table = dict(DEFAULT_PRICE_TABLE)
    if args.price_table:
        with open(args.price_table) as f:
            price_table.update(json.load(f))

    tasks = TASKS if not args.task_ids else [t for t in TASKS if t["id"] in args.task_ids]
    if not tasks:
        sys.exit(f"No matching tasks for --task-ids {args.task_ids}")

    if not args.mock:
        validate_host(args.host, args.allow_remote_host)
        # Fail fast with a clear message rather than erroring 15*N*repeats times.
        try:
            call_ollama(args.host, args.models[0], "print('ping')")
        except RuntimeError as e:
            sys.exit(f"[run_benchmark] {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = len(args.models) * len(tasks) * args.repeats
    done = 0
    rows: list[RunResult] = []

    print(
        f"Running {len(args.models)} model(s) x {len(tasks)} task(s) x {args.repeats} repeat(s) "
        f"= {total} attempts {'[MOCK MODE]' if args.mock else f'against {args.host}'}"
    )

    for model in args.models:
        for task in tasks:
            for repeat in range(1, args.repeats + 1):
                try:
                    result = run_one(
                        model=model,
                        task=task,
                        repeat=repeat,
                        host=args.host,
                        mock=args.mock,
                        seed=args.seed,
                        price_table=price_table,
                        grading_timeout=args.timeout,
                    )
                except RuntimeError as e:
                    sys.exit(f"[run_benchmark] {e}")
                rows.append(result)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total} ({result.model} / {result.task_id} rep{repeat}: {result.status})")

    fieldnames = list(asdict(rows[0]).keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    n_success = sum(1 for r in rows if r.status == "success")
    print(f"\nWrote {len(rows)} rows to {args.out}")
    print(f"Overall: {n_success}/{len(rows)} succeeded ({100 * n_success / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
