# llm_benchmark — real LLM-output evaluation harness (Ollama)

This folder is a small, self-contained benchmark that runs *real* LLM output
through the same evaluation lens as CodeShift's `agent_runs` table: does the
generated code work, how many tokens did it take, how fast was it, and what
would it have cost on a hosted API. It uses a local [Ollama](https://ollama.com)
server so it links directly to hands-on experience with Ollama + Llama 3.2,
rather than only analyzing simulated data (see `../ARCHITECTURE.md`).

It does not depend on the rest of the repo (no imports from `src/`, no shared
database) — it's a standalone add-on that happens to speak the same schema.

## What it measures

For each `(model, task, repeat)` triple, one Ollama call is made and the
resulting code is graded, producing one row that mirrors a CodeShift
`agent_runs` record:

| This harness's column | `agent_runs` equivalent | Meaning |
|---|---|---|
| `model` | `model` | Ollama model name |
| `task_type` | `task_type` | code_migration / bug_fix / test_generation / refactor / feature_build |
| `prompt_tokens` | `input_tokens` | Ollama `prompt_eval_count` |
| `output_tokens` | `output_tokens` | Ollama `eval_count` |
| `total_duration_sec` | `latency_sec` | Ollama `total_duration` (ns → sec) |
| `tests_passed` / `tests_total` | `tests_passed` / `tests_total` | hidden assert-based unit tests |
| `status` | `status` | success / partial / failed |
| `cost_usd` | `llm_cost_usd` | tokens x a configurable $/M-token price table |

`cost_usd` is an **API-equivalent cost**: Ollama itself is free/local, so we
price the token counts it reports against a small table of $/million-token
rates (`DEFAULT_PRICE_TABLE` in `run_benchmark.py`, same order of magnitude as
the main dataset's `open-weights-hosted` tier) so the numbers are comparable
to `agent_runs.llm_cost_usd` even though no money actually changes hands.

The 15 tasks (`tasks.py`) mirror the platform's task-type mix: Python 2 → 3
conversions, Java 8 → idiomatic Python translation, COBOL-style logic
modernization, buggy-function fixes, test-writing exercises, refactors, and
new small features. Every task ships hidden, assert-based unit tests that are
executed against the model's *own* generated code — never against a reference
solution — so grading is fully objective.

## How to run it

### 1. Real Ollama (needs the candidate's actual Ollama setup)

```bash
ollama serve                       # if not already running
ollama pull llama3.2
ollama pull qwen2.5-coder:1.5b

.venv/bin/python llm_benchmark/run_benchmark.py \
    --models llama3.2 qwen2.5-coder:1.5b \
    --repeats 3 \
    --out llm_benchmark/results/results.csv

.venv/bin/python llm_benchmark/analyze_benchmark.py \
    --results llm_benchmark/results/results.csv \
    --out-dir llm_benchmark/results
```

If the Ollama server isn't reachable, `run_benchmark.py` fails fast with a
specific message (host, model, suggested `ollama pull`/`ollama serve`
commands, and a pointer to `--mock`) rather than erroring out partway through
a long run.

### 2. Offline / mock mode (no Ollama required — this is how it was tested here)

```bash
.venv/bin/python llm_benchmark/run_benchmark.py \
    --models llama3.2 qwen2.5-coder:1.5b mistral \
    --repeats 3 --mock \
    --out llm_benchmark/results/results.csv

.venv/bin/python llm_benchmark/analyze_benchmark.py \
    --results llm_benchmark/results/results.csv \
    --out-dir llm_benchmark/results
```

`--mock` synthesizes plausible responses per model (a mix of fully-correct,
subtly-buggy, and malformed/unparseable code, with per-model "quality
profiles" and plausible token counts/timing) so the entire pipeline —
generation → code extraction → sandboxed grading → cost accounting →
statistics → chart — runs and is verifiable without Ollama installed. This
build machine does not have Ollama installed, so the committed
`results/results.csv`, `results/summary.md`, and
`results/pass_rate_by_model.png` in this folder were produced with `--mock`
using 3 mock models x 15 tasks x 3 repeats (135 attempts). Swap in real model
names and drop `--mock` to get real numbers once Ollama is available.

## How grading works (sandbox-ish)

1. The model's free-text response is scanned for a fenced ` ```python ` block
   (falls back to scanning for the first `def`/`import` line if the model
   forgot the fence).
2. The extracted code, plus the task's hidden tests, are written to a fresh
   temp directory and executed in a **separate subprocess** (`python
   _runner.py`) with a 10-second wall-clock timeout.
3. Inside that subprocess, the candidate code is exec'd once, then **each
   hidden test is exec'd independently** against a fresh copy of the
   resulting namespace. This is what makes partial credit possible: a model
   that gets 2 of 3 asserts right is graded `partial` (`tests_passed=2,
   tests_total=3`), not just pass/fail.
4. `status` is `success` (all tests pass), `partial` (some pass), or `failed`
   (none pass, code didn't parse, or the subprocess timed out /
   crashed — reported as `error_type="Timeout"` / `"RunnerCrash"`).

## Security model

Model output is **untrusted code**, so grading uses several independent layers:

| Layer | What it does |
|---|---|
| Separate process | Candidate code never runs in the harness process: no `eval`/`exec` in the parent, no shell. |
| Isolated interpreter | `python -I -S`: ignores `PYTHON*` env vars, user site-packages and `site`. |
| Clean environment | The child gets only `PATH` and locale, so API keys and tokens in your shell are not visible. |
| Fresh temp directory | Each attempt runs in its own `TemporaryDirectory`, which is deleted afterwards. |
| Resource limits (POSIX) | CPU time, memory, max file size (~1 MB), open files, process count, no core dumps. |
| Audit hook | Blocks sockets, `subprocess`, `os.system`/`exec`/`fork`/`posix_spawn`, `ctypes`, and file reads/writes outside the temp dir (stdlib reads allowed). |
| Wall-clock timeout | Kills the whole process group, including any children. |
| Output cap | stdout/stderr are read with a size cap so a flood can't exhaust memory. |
| Host allow-list | `--host` must be localhost/127.0.0.1/::1 unless you pass `--allow-remote-host`, and HTTP responses are size-capped. |

These layers are covered by `tests/test_llm_benchmark.py` (timeouts, blocked writes and reads, blocked process and network calls, env isolation, host validation).

**Residual risk, stated honestly:** Python audit hooks are a speed bump, not a security boundary. Determined native-code tricks can bypass them. The candidate also shares its temp directory with the grader, so malicious output could fake its *own* score (it can't reach your machine beyond the limits above). For running untrusted models at scale, wrap the grader in a container (`docker run --network none --read-only --memory 256m ...`) or a VM.

## Analysis (`analyze_benchmark.py`)

Reads the results CSV and writes `results/summary.md` +
`results/pass_rate_by_model.png`:

- **pass@1 / pass@k** — the unbiased estimator from Chen et al. 2021 (the
  Codex paper), `1 - C(n-c, k) / C(n, k)`, averaged per task then across
  tasks, rather than the biased "just run once" empirical pass rate.
- **Wilson 95% confidence intervals** on pass rate per model and per
  model x task_type — more reliable than a normal approximation at the small
  sample sizes a few repeats produce.
- **Chi-square test of independence** between model and outcome
  (success/partial/failed), to check whether the observed differences
  between models are more than sampling noise.
- **Tokens, latency, tokens/sec, cost per successful task** per model.
- A bar chart of pass rate by model with Wilson error bars
  (`pass_rate_by_model.png`).

## Files

```
llm_benchmark/
  tasks.py               15 tasks: id, task_type, prompt, entry_func, hidden tests,
                          reference_solution + buggy_solution (mock-mode only)
  run_benchmark.py        CLI: calls Ollama (or --mock), extracts code, grades in
                          a sandboxed subprocess, writes results/results.csv
  analyze_benchmark.py    CLI: stats + chart from results.csv
  README.md               this file
  results/
    results.csv            one row per (model, task, repeat) attempt
    summary.md             tables: pass@1/k, CIs, chi-square, cost, error types
    pass_rate_by_model.png bar chart with Wilson 95% CI error bars
```

Run `.venv/bin/python llm_benchmark/tasks.py` at any time as a fast sanity
check: it asserts every `reference_solution` passes its own hidden tests and
every `buggy_solution` fails at least one (catches a broken task before it
ever reaches a model).
