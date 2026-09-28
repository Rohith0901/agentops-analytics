#!/usr/bin/env python3
"""generate_data.py — simulates the CodeShift AI-coding-agent SaaS platform.

CodeShift (fictional) sells AI agents that run coding tasks (legacy migrations,
bug fixes, test generation, ...) for customers on a credits-based subscription.
This script simulates ~20 months of the business (customers, billing, and
individual agent runs) and writes the result to `data/raw/*.csv` and
`data/agentops.db` (SQLite). See ARCHITECTURE.md for the full data-model and
simulation-design contract this script implements.

Design notes for readers
-------------------------
* Every tunable knob lives in the CONFIG section below — nothing is a magic
  number buried in a function body.
* The simulation is a per-customer, month-by-month lifecycle: signup -> first
  month of usage -> conversion / churn / expansion decisions -> ... -> the
  end of the observation window or churn.
* Within a single customer-month, individual agent runs are generated with
  vectorized numpy operations (arrays of size n_runs), not a Python loop per
  run. The only Python-level loop is over (customer, month) pairs, which
  ARCHITECTURE.md explicitly allows ("loop over customers & months is fine,
  arrays within").
* A handful of "hidden truths" are baked in on purpose, for analysts to
  rediscover later: balanced-medium ~matches frontier-large on simple tasks
  at a quarter of the cost; activation predicts conversion & retention;
  Referral/Community customers stick around longer than Paid Search; Team
  plan margins are the thinnest of the paid plans because flat per-task
  credits under-price expensive migrations; the Q2 2026 price A/B test is
  under-powered.
"""

from __future__ import annotations

import argparse
import calendar
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "agentops.db"

# ======================================================================================
# CONFIG — every tunable parameter for the simulation lives here.
# ======================================================================================

# ---- Observation window ----------------------------------------------------
WINDOW_START = pd.Timestamp("2025-01-01")
WINDOW_END = pd.Timestamp("2026-08-31")
N_MONTHS = 20  # Jan 2025 .. Aug 2026 inclusive

# ---- Signup growth ----------------------------------------------------------
SIGNUP_BASE_PER_MONTH = 60.0
SIGNUP_GROWTH_RATE = 0.06  # ~6%/month compounding -> ~2,200 signups over 20 months

# ---- Plans (Builder A may tune these; ARCHITECTURE.md tables kept in sync) --
PLANS = {
    "Free": {"monthly_fee_usd": 0, "included_credits": 40, "overage_usd_per_credit": None},
    "Pro": {"monthly_fee_usd": 29, "included_credits": 150, "overage_usd_per_credit": 0.25},
    "Team": {"monthly_fee_usd": 149, "included_credits": 500, "overage_usd_per_credit": 0.22},
    "Enterprise": {"monthly_fee_usd": 1200, "included_credits": 1000, "overage_usd_per_credit": 0.18},
}
PLAN_ORDER = ["Free", "Pro", "Team", "Enterprise"]

# ---- Models (cost per million tokens, seconds per agent "step") -------------
MODELS = {
    "frontier-large": {"usd_per_m_input": 3.00, "usd_per_m_output": 15.00, "sec_per_step": 11},
    "balanced-medium": {"usd_per_m_input": 0.80, "usd_per_m_output": 4.00, "sec_per_step": 6},
    "fast-small": {"usd_per_m_input": 0.25, "usd_per_m_output": 1.25, "sec_per_step": 3},
    "open-weights-hosted": {"usd_per_m_input": 0.15, "usd_per_m_output": 0.60, "sec_per_step": 4.5},
}

# Router: probability a run on a given plan is served by each model.
# Free never gets the frontier model; Enterprise is mostly frontier.
MODEL_ROUTING_BY_PLAN = {
    "Free": {"frontier-large": 0.00, "balanced-medium": 0.00, "fast-small": 0.55, "open-weights-hosted": 0.45},
    "Pro": {"frontier-large": 0.05, "balanced-medium": 0.45, "fast-small": 0.35, "open-weights-hosted": 0.15},
    "Team": {"frontier-large": 0.30, "balanced-medium": 0.45, "fast-small": 0.15, "open-weights-hosted": 0.10},
    "Enterprise": {"frontier-large": 0.65, "balanced-medium": 0.25, "fast-small": 0.07, "open-weights-hosted": 0.03},
}

# ---- Industries: legacy score drives migration-heavy task mix --------------
INDUSTRY_LEGACY_SCORE = {
    "Banking": 0.90,
    "Government": 0.85,
    "Insurance": 0.80,
    "Telecom": 0.60,
    "Healthcare": 0.55,
    "Fintech": 0.35,
    "E-commerce": 0.30,
    "SaaS": 0.20,
}
INDUSTRY_WEIGHTS = {
    "Banking": 0.12, "Government": 0.08, "Insurance": 0.10, "Telecom": 0.12,
    "Healthcare": 0.12, "Fintech": 0.14, "E-commerce": 0.16, "SaaS": 0.16,
}

# ---- Task types: flat credits per task (2..6) -------------------------------
TASK_CREDITS = {
    "test_generation": 2,
    "bug_fix": 3,
    "dependency_upgrade": 3,
    "refactor": 4,
    "feature_build": 5,
    "code_migration": 6,
}
SIMPLE_TASKS = {"test_generation", "bug_fix", "dependency_upgrade"}
COMPLEX_TASKS = {"refactor", "feature_build", "code_migration"}

# Non-migration tasks use source == target from this set of languages.
NON_MIGRATION_LANGUAGES = ["Python", "Java", "TypeScript", "Go", "C#"]

# Migration pairs: (difficulty_mult, token_mult). Lower difficulty = harder
# (drags down success probability more and needs more agent steps).
MIGRATION_PAIRS = {
    "COBOL->Java": {"source": "COBOL", "target": "Java", "difficulty": 0.70, "token_mult": 1.8},
    "Java 8->Java 21": {"source": "Java 8", "target": "Java 21", "difficulty": 1.05, "token_mult": 1.0},
    "Python 2->Python 3": {"source": "Python 2", "target": "Python 3", "difficulty": 1.15, "token_mult": 0.8},
    "AngularJS->React": {"source": "AngularJS", "target": "React", "difficulty": 0.85, "token_mult": 1.3},
".NET Framework->.NET 8": {"source": ".NET Framework", "target": ".NET 8", "difficulty": 0.95, "token_mult": 1.1},
    "PHP->Node.js": {"source": "PHP", "target": "Node.js", "difficulty": 0.85, "token_mult": 1.2},
}
# Baseline pick weights before the legacy-score tilt toward COBOL->Java.
MIGRATION_PAIR_BASE_WEIGHT = {
    "COBOL->Java": 0.30, "Java 8->Java 21": 0.20, "Python 2->Python 3": 0.15,
    "AngularJS->React": 0.15, ".NET Framework->.NET 8": 0.12, "PHP->Node.js": 0.08,
}

# ---- Model quality: base success rate (on the frontier model) --------------
BASE_SUCCESS_FRONTIER = {
    "test_generation": 0.95, "bug_fix": 0.87, "refactor": 0.88,
    "dependency_upgrade": 0.85, "feature_build": 0.76,
    "code_migration": 0.83,  # further multiplied by the migration pair's difficulty
}
# Success multiplier by model, split by simple vs. complex task category.
# This is the "balanced-medium ~= frontier on simple tasks, at ~1/4 the cost"
# routing opportunity analysts should discover.
MODEL_SUCCESS_MULT = {
    "frontier-large": {"simple": 1.00, "complex": 1.00},
    "balanced-medium": {"simple": 0.98, "complex": 0.86},
    "fast-small": {"simple": 0.90, "complex": 0.66},
    "open-weights-hosted": {"simple": 0.84, "complex": 0.55},
}
# Smaller/weaker models grind through more agent steps for the same task.
MODEL_STEP_MULT = {
    "frontier-large": 1.00, "balanced-medium": 1.30, "fast-small": 1.80, "open-weights-hosted": 2.10,
}
# Failed runs loop and re-try -> take substantially more steps than a clean run.
STATUS_STEP_MULT = {"success": 1.00, "partial": 1.20, "failed": 1.60}
# Of the non-success probability mass, how much becomes "partial" vs "failed".
PARTIAL_SHARE_OF_NONSUCCESS = 0.55

# Base agent-steps per task type (before model/difficulty/status multipliers).
# Tuned down from an initial pass so overall gross margin lands in the
# 45-65% calibration target while keeping ~20-25k realistic input tokens/step.
BASE_STEPS_BY_TASK = {
    "test_generation": 4, "bug_fix": 6, "dependency_upgrade": 4,
    "refactor": 8, "feature_build": 12, "code_migration": 16,
}
# Roughly realistic per-step token footprint for an agentic coding step.
TOKENS_PER_STEP_INPUT_MEAN = 22_500
OUTPUT_INPUT_RATIO_MEAN = 0.35
LINES_PER_STEP_BASE = {
    "test_generation": 10, "bug_fix": 8, "dependency_upgrade": 6,
    "refactor": 18, "feature_build": 22, "code_migration": 16,
}

# ---- Customer attributes -----------------------------------------------------
COMPANY_SIZES = ["Startup", "SMB", "Mid-Market", "Enterprise"]
REGIONS = ["North America", "Europe", "India", "APAC"]
REGION_WEIGHTS = [0.40, 0.25, 0.20, 0.15]
CHANNELS = ["Organic", "Paid Search", "Referral", "Community", "Outbound Sales"]
CHANNEL_WEIGHTS = [0.30, 0.22, 0.20, 0.18, 0.10]

# Company-size mix, conditional on acquisition channel (sales-led skews big).
SIZE_WEIGHTS_BY_CHANNEL = {
    "Outbound Sales": [0.00, 0.15, 0.35, 0.50],
    "Organic": [0.40, 0.32, 0.20, 0.08],
    "Paid Search": [0.42, 0.33, 0.19, 0.06],
    "Referral": [0.32, 0.32, 0.24, 0.12],
    "Community": [0.38, 0.30, 0.22, 0.10],
}
# Sales-led (Outbound Sales) customers skip Free entirely and start paid.
SALES_LED_STARTING_PLAN_WEIGHTS = {"Team": 0.65, "Enterprise": 0.35}

# ---- Usage intensity: mean monthly runs per plan, times a per-customer ------
# lognormal "engagement" multiplier (captures heavy vs. light users).
PLAN_MONTHLY_RUN_LAMBDA = {"Free": 6, "Pro": 18, "Team": 52, "Enterprise": 160}
ENGAGEMENT_LOGNORMAL_SIGMA = 0.55
MIN_FIRST_MONTH_RUNS = 5  # activation is computed on the first 5 runs

# ---- Activation -> conversion (logistic regression, hand-tuned coefficients) -
CONVERSION_LOGIT_INTERCEPT = -2.95
CONVERSION_LOGIT_ACTIVATION_COEF = 2.6
CONVERSION_LOGIT_CHANNEL = {
    "Organic": 0.0, "Paid Search": -0.55, "Referral": 0.65, "Community": 0.45, "Outbound Sales": 0.0,
}
CONVERSION_LOGIT_SIZE = {"Startup": -0.10, "SMB": 0.0, "Mid-Market": 0.15, "Enterprise": 0.30}
LATE_CONVERSION_DECAY = 0.35  # attempt probability shrinks by this factor each extra month on Free

# ---- Churn hazards -----------------------------------------------------------
FREE_CHURN_BASE_HAZARD = 0.30
PAID_CHURN_BASE_HAZARD = {"Pro": 0.058, "Team": 0.048, "Enterprise": 0.020}
CHURN_CHANNEL_MULT = {
    "Organic": 1.00, "Paid Search": 1.35, "Referral": 0.72, "Community": 0.82, "Outbound Sales": 0.75,
}
CHURN_FAILURE_RATE_THRESHOLD = 0.30  # a month with >30% failed runs hurts retention
CHURN_FAILURE_RATE_MULT = 1.5

# ---- Expansion (NRR): heavy usage vs. included credits nudges an upgrade ----
EXPANSION_USAGE_THRESHOLD = 1.15  # credits_used > 115% of included_credits
EXPANSION_PROB = {"Pro": 0.15, "Team": 0.08}  # Pro->Team, Team->Enterprise
UPGRADE_PATH = {"Pro": "Team", "Team": "Enterprise"}

# ---- Human ratings ------------------------------------------------------------
RATING_COVERAGE = 0.25  # ~25% of runs get a human rating
RATING_WEIGHTS_BY_STATUS = {
    "success": [0.02, 0.03, 0.10, 0.35, 0.50],
    "partial": [0.05, 0.15, 0.30, 0.35, 0.15],
    "failed": [0.35, 0.35, 0.20, 0.08, 0.02],
}

# ---- Price A/B test (Builder B analyzes this; Builder A only generates it) --
EXPERIMENT_NAME = "pro_price_2026Q2"
EXPERIMENT_START = pd.Timestamp("2026-03-01")
EXPERIMENT_END = pd.Timestamp("2026-05-31")
EXPERIMENT_TREATMENT_PRICE = 39
EXPERIMENT_CONTROL_PRICE = 29
EXPERIMENT_TREATMENT_CONVERSION_MULT = 0.85


# ======================================================================================
# Small helpers
# ======================================================================================

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def weighted_choice(rng: np.random.Generator, options: list, weights: list, size: int = 1):
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    return rng.choice(options, size=size, p=p)


def month_add(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    """Return the first-of-month timestamp `n` months after `ts`'s month."""
    total = ts.year * 12 + (ts.month - 1) + n
    return pd.Timestamp(year=total // 12, month=total % 12 + 1, day=1)


def random_day_in_month(rng: np.random.Generator, month_start: pd.Timestamp) -> pd.Timestamp:
    days_in_month = (month_start + pd.offsets.MonthEnd(0)).day
    day = int(rng.integers(1, days_in_month + 1))
    seconds = int(rng.integers(0, 24 * 3600))
    return month_start + pd.Timedelta(days=day - 1, seconds=seconds)


# ======================================================================================
# Static reference tables
# ======================================================================================

def build_plans_df() -> pd.DataFrame:
    rows = []
    for plan, cfg in PLANS.items():
        rows.append({
            "plan": plan,
            "monthly_fee_usd": cfg["monthly_fee_usd"],
            "included_credits": cfg["included_credits"],
            "overage_usd_per_credit": cfg["overage_usd_per_credit"],
        })
    return pd.DataFrame(rows)


def build_models_df() -> pd.DataFrame:
    rows = []
    for model, cfg in MODELS.items():
        rows.append({
            "model": model,
            "usd_per_m_input": cfg["usd_per_m_input"],
            "usd_per_m_output": cfg["usd_per_m_output"],
            "sec_per_step": cfg["sec_per_step"],
        })
    return pd.DataFrame(rows)


# ======================================================================================
# Customer generation
# ======================================================================================

@dataclass
class CustomerSeed:
    """Static, immutable attributes decided once at signup time."""
    customer_id: str
    company_name: str
    signup_date: pd.Timestamp
    industry: str
    company_size: str
    region: str
    acquisition_channel: str
    is_sales_led: int
    # Filled in during simulation:
    experiment_variant: str | None = None
    experiment_price: int | None = None


NAME_PREFIXES = [
    "Nimbus", "Vertex", "Cobalt", "Lattice", "Anchor", "Summit", "Halcyon", "Meridian",
    "Pinegrove", "Ironclad", "Bluewave", "Solstice", "Keystone", "Redwood", "Amberline",
    "Northfield", "Cascade", "Brightpath", "Foundry", "Silverline", "Crestpoint", "Harbor",
    "Wellspring", "Ridgeway", "Clearview", "Beacon", "Timberline", "Glacier", "Highfield",
    "Cornerstone",
]
NAME_SUFFIXES = ["Systems", "Labs", "Group", "Technologies", "Partners", "Networks", "Solutions", "Works", "Analytics", "Co"]


def make_company_name(rng: np.random.Generator, used: set[str]) -> str:
    """Return a plausible, unique fake company name.

    There are only len(NAME_PREFIXES) * len(NAME_SUFFIXES) base combinations,
    which is fewer than the number of customers we simulate, so once the base
    pool is exhausted we fall back to appending a numeric suffix (still
    plausible-looking, e.g. "Nimbus Labs II") rather than looping forever.
    """
    base = f"{rng.choice(NAME_PREFIXES)} {rng.choice(NAME_SUFFIXES)}"
    if base not in used:
        used.add(base)
        return base
    n = 2
    while f"{base} {n}" in used:
        n += 1
    name = f"{base} {n}"
    used.add(name)
    return name


def generate_customer_seeds(rng: np.random.Generator, scale: float) -> list[CustomerSeed]:
    """Decide, for each customer: when they signed up and their static profile."""
    seeds: list[CustomerSeed] = []
    used_names: set[str] = set()
    customer_counter = 1

    industries = list(INDUSTRY_WEIGHTS.keys())
    industry_p = np.array(list(INDUSTRY_WEIGHTS.values()))
    industry_p = industry_p / industry_p.sum()

    for m in range(N_MONTHS):
        month_start = month_add(WINDOW_START, m)
        n_signups = int(round(SIGNUP_BASE_PER_MONTH * (1 + SIGNUP_GROWTH_RATE) ** m * scale))
        if n_signups <= 0:
            continue

        channels = rng.choice(CHANNELS, size=n_signups, p=np.array(CHANNEL_WEIGHTS) / sum(CHANNEL_WEIGHTS))
        industries_draw = rng.choice(industries, size=n_signups, p=industry_p)
        regions_draw = rng.choice(REGIONS, size=n_signups, p=np.array(REGION_WEIGHTS) / sum(REGION_WEIGHTS))

        for i in range(n_signups):
            channel = channels[i]
            size_weights = SIZE_WEIGHTS_BY_CHANNEL[channel]
            company_size = rng.choice(COMPANY_SIZES, p=np.array(size_weights) / sum(size_weights))
            signup_date = random_day_in_month(rng, month_start)
            is_sales_led = 1 if channel == "Outbound Sales" else 0

            seeds.append(CustomerSeed(
                customer_id=f"C{customer_counter:05d}",
                company_name=make_company_name(rng, used_names),
                signup_date=signup_date,
                industry=industries_draw[i],
                company_size=company_size,
                region=regions_draw[i],
                acquisition_channel=channel,
                is_sales_led=is_sales_led,
            ))
            customer_counter += 1
    return seeds


# ======================================================================================
# Per-customer-month run generation (vectorized within a month)
# ======================================================================================

def task_type_weights_for_legacy(legacy: float) -> dict[str, float]:
    """Higher legacy score -> more code_migration tasks; remaining mass split
    among the other task types with fixed relative weights."""
    p_migration = 0.10 + 0.5 * legacy
    other_weights = {
        "bug_fix": 0.30, "test_generation": 0.20, "refactor": 0.20,
        "dependency_upgrade": 0.15, "feature_build": 0.15,
    }
    total_other = 1 - p_migration
    weights = {k: v * total_other for k, v in other_weights.items()}
    weights["code_migration"] = p_migration
    return weights


def migration_pair_weights_for_legacy(legacy: float) -> dict[str, float]:
    """Higher legacy score tilts migration mix further toward COBOL->Java."""
    weights = dict(MIGRATION_PAIR_BASE_WEIGHT)
    weights["COBOL->Java"] *= (0.5 + legacy)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def generate_runs(
    rng: np.random.Generator,
    customer_id: str,
    plan: str,
    month_start_np: np.datetime64,
    days_in_month: int,
    n_runs: int,
    task_weights: dict[str, float],
    migration_weights: dict[str, float],
    run_id_counter: list[int],
) -> dict[str, np.ndarray] | None:
    """Vectorized generation of `n_runs` agent_runs rows for one customer-month.

    Returns a dict of equal-length numpy arrays (one per output column) rather
    than a DataFrame — this function runs tens of thousands of times, so
    avoiding a pandas object per call matters a lot for total runtime. The
    caller batches these dicts and builds one DataFrame at the very end.
    """
    if n_runs <= 0:
        return None

    # --- Task type & migration pair (or plain language) per run --------------
    task_names = list(task_weights.keys())
    task_p = np.array(list(task_weights.values()))
    task_p = task_p / task_p.sum()
    task_type = rng.choice(task_names, size=n_runs, p=task_p)

    is_migration = task_type == "code_migration"
    n_migration = int(is_migration.sum())

    source_stack = np.empty(n_runs, dtype=object)
    target_stack = np.empty(n_runs, dtype=object)
    difficulty = np.ones(n_runs)
    token_mult = np.ones(n_runs)

    if n_migration > 0:
        pair_names = list(migration_weights.keys())
        pair_p = np.array(list(migration_weights.values()))
        pair_p = pair_p / pair_p.sum()
        chosen_pairs = rng.choice(pair_names, size=n_migration, p=pair_p)
        for idx, pair_name in zip(np.where(is_migration)[0], chosen_pairs):
            pair_cfg = MIGRATION_PAIRS[pair_name]
            source_stack[idx] = pair_cfg["source"]
            target_stack[idx] = pair_cfg["target"]
            difficulty[idx] = pair_cfg["difficulty"]
            token_mult[idx] = pair_cfg["token_mult"]

    n_non_migration = n_runs - n_migration
    if n_non_migration > 0:
        langs = rng.choice(NON_MIGRATION_LANGUAGES, size=n_non_migration)
        non_mig_idx = np.where(~is_migration)[0]
        source_stack[non_mig_idx] = langs
        target_stack[non_mig_idx] = langs

    # --- Model routing by plan --------------------------------------------------
    routing = MODEL_ROUTING_BY_PLAN[plan]
    model_names = list(routing.keys())
    model_p = np.array(list(routing.values()))
    model_p = model_p / model_p.sum()
    model = rng.choice(model_names, size=n_runs, p=model_p)

    # --- Success probability & status ------------------------------------------
    base_success = np.array([BASE_SUCCESS_FRONTIER[t] for t in task_type])
    base_success = np.where(is_migration, base_success * difficulty, base_success)

    task_category = np.where(np.isin(task_type, list(SIMPLE_TASKS)), "simple", "complex")
    model_mult = np.array([MODEL_SUCCESS_MULT[m][c] for m, c in zip(model, task_category)])
    success_prob = np.clip(base_success * model_mult, 0.03, 0.97)

    u = rng.random(n_runs)
    partial_prob = (1 - success_prob) * PARTIAL_SHARE_OF_NONSUCCESS
    status = np.where(
        u < success_prob, "success",
        np.where(u < success_prob + partial_prob, "partial", "failed"),
    )

    # --- Steps, tokens, cost, latency, lines, tests -----------------------------
    steps_base = np.array([BASE_STEPS_BY_TASK[t] for t in task_type], dtype=float)
    difficulty_step_mult = np.where(is_migration, 2.0 - difficulty, 1.0)
    model_step_mult = np.array([MODEL_STEP_MULT[m] for m in model])
    status_step_mult = np.array([STATUS_STEP_MULT[s] for s in status])
    step_noise = rng.lognormal(mean=0.0, sigma=0.20, size=n_runs)
    steps = np.maximum(1, np.round(
        steps_base * difficulty_step_mult * model_step_mult * status_step_mult * step_noise
    )).astype(int)

    input_tokens = np.maximum(200, np.round(
        steps * TOKENS_PER_STEP_INPUT_MEAN * token_mult * rng.lognormal(mean=0.0, sigma=0.15, size=n_runs)
    )).astype(int)
    output_tokens = np.maximum(50, np.round(
        input_tokens * OUTPUT_INPUT_RATIO_MEAN * rng.lognormal(mean=0.0, sigma=0.20, size=n_runs)
    )).astype(int)

    model_cost_in = np.array([MODELS[m]["usd_per_m_input"] for m in model])
    model_cost_out = np.array([MODELS[m]["usd_per_m_output"] for m in model])
    llm_cost_usd = (input_tokens / 1e6) * model_cost_in + (output_tokens / 1e6) * model_cost_out

    sec_per_step = np.array([MODELS[m]["sec_per_step"] for m in model])
    latency_sec = np.round(steps * sec_per_step * rng.lognormal(mean=0.0, sigma=0.15, size=n_runs), 1)

    lines_per_step = np.array([LINES_PER_STEP_BASE[t] for t in task_type])
    lines_changed = np.maximum(1, np.round(
        steps * lines_per_step * rng.lognormal(mean=0.0, sigma=0.30, size=n_runs)
    )).astype(int)

    tests_total = rng.poisson(8, size=n_runs) + 2
    tests_passed = np.empty(n_runs, dtype=int)
    success_mask = status == "success"
    partial_mask = status == "partial"
    failed_mask = status == "failed"
    tests_passed[success_mask] = tests_total[success_mask]
    tests_passed[partial_mask] = rng.binomial(tests_total[partial_mask], 0.55)
    tests_passed[failed_mask] = rng.binomial(tests_total[failed_mask], 0.15)

    credits_charged = np.array([TASK_CREDITS[t] for t in task_type])

    # --- Human ratings: ~25% coverage, correlated with status -------------------
    has_rating = rng.random(n_runs) < RATING_COVERAGE
    human_rating = np.full(n_runs, np.nan)
    for s, weights in RATING_WEIGHTS_BY_STATUS.items():
        mask = (status == s) & has_rating
        n_mask = int(mask.sum())
        if n_mask:
            human_rating[mask] = rng.choice([1, 2, 3, 4, 5], size=n_mask, p=np.array(weights) / sum(weights))

    # --- Timestamps & IDs --------------------------------------------------------
    # Pure-numpy datetime arithmetic: no per-row Timestamp objects, no per-row
    # isoformat() calls. This is the single biggest runtime win in the whole
    # generator, since this function runs once per customer-month (tens of
    # thousands of times) and formerly built a Python list of Timestamps here.
    day_offsets = rng.integers(0, days_in_month, size=n_runs).astype("timedelta64[D]")
    second_offsets = rng.integers(0, 24 * 3600, size=n_runs).astype("timedelta64[s]")
    started_dt64 = month_start_np + day_offsets + second_offsets

    start_id = run_id_counter[0]
    run_ids = np.array([f"R{start_id + i:08d}" for i in range(n_runs)])
    run_id_counter[0] += n_runs

    # A single, cheap sort by time so downstream code can treat "first N runs"
    # as "first N runs in time order" without re-sorting a DataFrame later.
    order = np.argsort(started_dt64)

    return {
        "run_id": run_ids[order],
        "customer_id": np.full(n_runs, customer_id),
        "started_at_dt64": started_dt64[order],
        "task_type": task_type[order],
        "source_stack": source_stack[order],
        "target_stack": target_stack[order],
        "model": model[order],
        "plan_at_run": np.full(n_runs, plan),
        "steps": steps[order],
        "input_tokens": input_tokens[order],
        "output_tokens": output_tokens[order],
        "llm_cost_usd": np.round(llm_cost_usd, 6)[order],
        "latency_sec": latency_sec[order],
        "lines_changed": lines_changed[order],
        "tests_total": tests_total[order],
        "tests_passed": tests_passed[order],
        "status": status[order],
        "credits_charged": credits_charged[order],
        "human_rating": human_rating[order],
    }


# ======================================================================================
# Main per-customer lifecycle simulation
# ======================================================================================

def simulate_customer(
    rng: np.random.Generator,
    seed: CustomerSeed,
    run_id_counter: list[int],
) -> tuple[dict, list[dict], dict[str, np.ndarray] | None, dict | None]:
    """Simulate one customer's full lifecycle: monthly billing rows, agent runs,
    the final customer record, and (if applicable) an experiment assignment row.

    Agent runs are returned as a single dict-of-arrays (one key per column),
    already concatenated across all of this customer's months, rather than a
    DataFrame — see `generate_runs` for why.
    """
    legacy = INDUSTRY_LEGACY_SCORE[seed.industry]
    task_weights = task_type_weights_for_legacy(legacy)
    migration_weights = migration_pair_weights_for_legacy(legacy)

    signup_month = pd.Timestamp(year=seed.signup_date.year, month=seed.signup_date.month, day=1)
    last_month = month_add(WINDOW_START, N_MONTHS - 1)

    # Experiment assignment (self-serve customers who sign up in the window only).
    experiment_row = None
    if not seed.is_sales_led and EXPERIMENT_START <= signup_month <= EXPERIMENT_END:
        variant = "treatment" if rng.random() < 0.5 else "control"
        price = EXPERIMENT_TREATMENT_PRICE if variant == "treatment" else EXPERIMENT_CONTROL_PRICE
        seed.experiment_variant = variant
        seed.experiment_price = price
        experiment_row = {
            "customer_id": seed.customer_id,
            "experiment_name": EXPERIMENT_NAME,
            "assigned_at": seed.signup_date.strftime("%Y-%m-%d"),
            "variant": variant,
            "pro_price_shown": price,
        }

    # Per-customer engagement multiplier (heavy vs. light users of the product).
    engagement = rng.lognormal(mean=0.0, sigma=ENGAGEMENT_LOGNORMAL_SIGMA)

    if seed.is_sales_led:
        plan = rng.choice(
            list(SALES_LED_STARTING_PLAN_WEIGHTS.keys()),
            p=list(SALES_LED_STARTING_PLAN_WEIGHTS.values()),
        )
        first_paid_date = seed.signup_date
    else:
        plan = "Free"
        first_paid_date = None

    activation = None  # success share of the first 5 runs; set after month 1
    months_on_free = 0
    churn_date = None
    monthly_rows = []
    run_chunks: list[dict[str, np.ndarray]] = []

    month = signup_month
    while month <= last_month:
        included_credits = PLANS[plan]["included_credits"]
        days_in_month = calendar.monthrange(month.year, month.month)[1]
        month_start_np = np.datetime64(f"{month.year:04d}-{month.month:02d}-01")
        if month == signup_month:
            # Signup month: runs can only happen from the signup day onwards.
            month_start_np = np.datetime64(seed.signup_date.strftime("%Y-%m-%d"))
            days_in_month = days_in_month - seed.signup_date.day + 1

        # --- Decide how much they use the product this month --------------------
        base_lambda = PLAN_MONTHLY_RUN_LAMBDA[plan] * engagement
        if plan == "Free" and month == signup_month:
            n_runs = max(MIN_FIRST_MONTH_RUNS, rng.poisson(base_lambda))
        else:
            n_runs = max(1, rng.poisson(base_lambda))

        month_runs = generate_runs(
            rng, seed.customer_id, plan, month_start_np, days_in_month, n_runs,
            task_weights, migration_weights, run_id_counter,
        )

        # Free plan is a hard cap: truncate runs once included credits are used up
        # (runs already come back time-ordered from generate_runs).
        if month_runs is not None and plan == "Free":
            cum_credits = np.cumsum(month_runs["credits_charged"])
            keep = cum_credits <= included_credits
            if not keep.all():
                # Always keep at least the first MIN_FIRST_MONTH_RUNS runs so
                # activation can still be computed from the first month.
                keep[:MIN_FIRST_MONTH_RUNS] = True
            if not keep.all():
                month_runs = {k: v[keep] for k, v in month_runs.items()}

        if month_runs is not None:
            run_chunks.append(month_runs)

        # --- Activation: success share of the first 5 runs of the customer's life --
        if activation is None and month_runs is not None and len(month_runs["status"]) >= MIN_FIRST_MONTH_RUNS:
            first_five_status = month_runs["status"][:MIN_FIRST_MONTH_RUNS]
            activation = float(np.mean(first_five_status == "success"))

        # --- Billing for this month ----------------------------------------------
        credits_used = int(month_runs["credits_charged"].sum()) if month_runs is not None else 0
        if plan == "Free":
            overage_credits = 0
            overage_revenue = 0.0
            monthly_fee = 0
        else:
            overage_credits = max(0, credits_used - included_credits)
            overage_rate = PLANS[plan]["overage_usd_per_credit"]
            overage_revenue = overage_credits * overage_rate
            if plan == "Pro" and seed.experiment_price is not None:
                monthly_fee = seed.experiment_price
            else:
                monthly_fee = PLANS[plan]["monthly_fee_usd"]
        revenue = monthly_fee + overage_revenue

        monthly_rows.append({
            "customer_id": seed.customer_id,
            "month": month.strftime("%Y-%m-01"),
            "plan": plan,
            "monthly_fee_usd": monthly_fee,
            "included_credits": included_credits,
            "credits_used": credits_used,
            "overage_credits": overage_credits,
            "overage_revenue_usd": round(overage_revenue, 2),
            "revenue_usd": round(revenue, 2),
        })

        failed_rate = float(np.mean(month_runs["status"] == "failed")) if month_runs is not None else 0.0

        # --- Decide next month's state -------------------------------------------
        if plan == "Free":
            months_on_free += 1
            act = activation if activation is not None else 0.0
            logit = (
                CONVERSION_LOGIT_INTERCEPT
                + CONVERSION_LOGIT_ACTIVATION_COEF * act
                + CONVERSION_LOGIT_CHANNEL[seed.acquisition_channel]
                + CONVERSION_LOGIT_SIZE[seed.company_size]
            )
            p_conv = float(sigmoid(logit)) * (LATE_CONVERSION_DECAY ** (months_on_free - 1))
            if seed.experiment_variant == "treatment":
                p_conv *= EXPERIMENT_TREATMENT_CONVERSION_MULT
            p_churn_free = np.clip(FREE_CHURN_BASE_HAZARD * (1.3 - 0.8 * act), 0.05, 0.60)

            u = rng.random()
            if u < p_conv:
                plan = "Pro"
                first_paid_date = month_add(month, 1)
            elif u < p_conv + p_churn_free:
                churn_date = (month + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
                break
            # else: stay Free, continue loop
        else:
            act = activation if activation is not None else 0.5
            p_churn = PAID_CHURN_BASE_HAZARD[plan] * (1.5 - 0.9 * act) * CHURN_CHANNEL_MULT[seed.acquisition_channel]
            if failed_rate > CHURN_FAILURE_RATE_THRESHOLD:
                p_churn *= CHURN_FAILURE_RATE_MULT
            p_churn = float(np.clip(p_churn, 0.003, 0.60))

            if rng.random() < p_churn:
                churn_date = (month + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
                break

            # Expansion: heavy usage vs. included credits nudges an upgrade.
            if plan in EXPANSION_PROB and included_credits > 0:
                if credits_used > EXPANSION_USAGE_THRESHOLD * included_credits:
                    if rng.random() < EXPANSION_PROB[plan]:
                        plan = UPGRADE_PATH[plan]

        month = month_add(month, 1)

    status = "churned" if churn_date is not None else "active"
    customer_row = {
        "customer_id": seed.customer_id,
        "company_name": seed.company_name,
        "signup_date": seed.signup_date.strftime("%Y-%m-%d"),
        "industry": seed.industry,
        "company_size": seed.company_size,
        "region": seed.region,
        "acquisition_channel": seed.acquisition_channel,
        "is_sales_led": seed.is_sales_led,
        "first_paid_date": first_paid_date.strftime("%Y-%m-%d") if first_paid_date is not None else None,
        "current_plan": plan,
        "churn_date": churn_date,
        "status": status,
    }

    # Concatenate this customer's monthly run-chunks into one dict-of-arrays.
    runs_combined: dict[str, np.ndarray] | None = None
    if run_chunks:
        keys = run_chunks[0].keys()
        runs_combined = {k: np.concatenate([c[k] for c in run_chunks]) for k in keys}

    return customer_row, monthly_rows, runs_combined, experiment_row


# ======================================================================================
# Orchestration
# ======================================================================================

def run_simulation(seed: int, scale: float) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    seeds = generate_customer_seeds(rng, scale)

    run_id_counter = [1]
    customer_rows = []
    month_rows: list[dict] = []
    run_chunks: list[dict[str, np.ndarray]] = []
    experiment_rows = []

    for cust_seed in seeds:
        customer_row, months, runs_chunk, experiment_row = simulate_customer(rng, cust_seed, run_id_counter)
        customer_rows.append(customer_row)
        month_rows.extend(months)
        if runs_chunk is not None:
            run_chunks.append(runs_chunk)
        if experiment_row is not None:
            experiment_rows.append(experiment_row)

    customers_df = pd.DataFrame(customer_rows)
    customer_months_df = pd.DataFrame(month_rows)
    experiment_assignments_df = pd.DataFrame(experiment_rows)

    # Build agent_runs in one shot: concatenate each column across all
    # per-customer chunks, then construct a single DataFrame.
    keys = run_chunks[0].keys()
    combined = {k: np.concatenate([c[k] for c in run_chunks]) for k in keys}
    started_at = combined.pop("started_at_dt64").astype("datetime64[s]").astype(str)
    combined["started_at"] = started_at
    # Column order matches ARCHITECTURE.md's agent_runs spec.
    column_order = [
        "run_id", "customer_id", "started_at", "task_type", "source_stack", "target_stack",
        "model", "plan_at_run", "steps", "input_tokens", "output_tokens", "llm_cost_usd",
        "latency_sec", "lines_changed", "tests_total", "tests_passed", "status",
        "credits_charged", "human_rating",
    ]
    agent_runs_df = pd.DataFrame({k: combined[k] for k in column_order})

    return {
        "plans": build_plans_df(),
        "models": build_models_df(),
        "customers": customers_df,
        "customer_months": customer_months_df,
        "agent_runs": agent_runs_df,
        "experiment_assignments": experiment_assignments_df,
    }


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        for name, df in tables.items():
            df.to_sql(name, conn, index=False)
        # Indexes analysts will actually query on.
        conn.execute("CREATE INDEX idx_customer_months_customer_id ON customer_months(customer_id)")
        conn.execute("CREATE INDEX idx_customer_months_month ON customer_months(month)")
        conn.execute("CREATE INDEX idx_agent_runs_customer_id ON agent_runs(customer_id)")
        conn.execute("CREATE INDEX idx_agent_runs_started_at ON agent_runs(started_at)")
        conn.execute("CREATE INDEX idx_agent_runs_model ON agent_runs(model)")
        conn.execute("CREATE INDEX idx_agent_runs_task_type ON agent_runs(task_type)")
        conn.commit()
    finally:
        conn.close()


def print_calibration_summary(tables: dict[str, pd.DataFrame], runtime_sec: float) -> None:
    customers = tables["customers"]
    months = tables["customer_months"]
    runs = tables["agent_runs"]

    n_customers = len(customers)
    n_runs = len(runs)
    overall_success_rate = (runs["status"] == "success").mean()

    # Cost per run, joined implicitly (llm_cost_usd is already per run).
    total_revenue = months["revenue_usd"].sum()
    total_llm_cost = runs["llm_cost_usd"].sum()
    overall_margin = (total_revenue - total_llm_cost) / total_revenue if total_revenue else float("nan")

    # Margin by plan: join runs -> customer_months by (customer_id, month) to
    # attribute LLM cost to the plan active in that billing month.
    runs_month = runs.copy()
    runs_month["month"] = pd.to_datetime(runs_month["started_at"]).dt.strftime("%Y-%m-01")
    cost_by_plan_month = (
        runs_month.groupby(["customer_id", "month"])["llm_cost_usd"].sum().reset_index()
    )
    merged = months.merge(cost_by_plan_month, on=["customer_id", "month"], how="left")
    merged["llm_cost_usd"] = merged["llm_cost_usd"].fillna(0.0)
    margin_by_plan = merged.groupby("plan").apply(
        lambda g: (g["revenue_usd"].sum() - g["llm_cost_usd"].sum()) / g["revenue_usd"].sum()
        if g["revenue_usd"].sum() > 0 else float("nan"),
        include_groups=False,
    )

    n_free = (customers["is_sales_led"] == 0).sum()
    n_converted = customers["first_paid_date"].notna().sum() - (customers["is_sales_led"] == 1).sum()
    conversion_rate = n_converted / n_free if n_free else float("nan")

    # Paid monthly logo churn, approximated as a hazard rate: churn events among
    # paying customers, divided by the paid customer-months "at risk" of churning
    # that month (every paid customer-month is one at-risk observation).
    paid_months = months[months["plan"] != "Free"]
    churned_customers = customers[(customers["status"] == "churned") & (customers["first_paid_date"].notna())]
    paid_churn_rate = len(churned_customers) / len(paid_months) if len(paid_months) else float("nan")

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Customers:              {n_customers:,}")
    print(f"Customer-months:        {len(months):,}")
    print(f"Agent runs:             {n_runs:,}")
    print(f"Experiment assignments: {len(tables['experiment_assignments']):,}")
    print("-" * 70)
    print(f"Overall run success rate:      {overall_success_rate:.1%}  (target 70-78%)")
    print(f"Overall gross margin:           {overall_margin:.1%}  (target 45-65%)")
    print("Gross margin by plan:")
    for plan_name in PLAN_ORDER:
        if plan_name in margin_by_plan.index:
            print(f"    {plan_name:<12s}{margin_by_plan[plan_name]:.1%}")
    print(f"Free -> paid conversion rate:   {conversion_rate:.1%}  (target 25-35%)")
    print(f"Paid monthly logo churn (approx): {paid_churn_rate:.1%}  (target 3-6%)")
    print("-" * 70)
    print(f"Runtime: {runtime_sec:.1f}s  (target < 60s)")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CodeShift AgentOps synthetic dataset.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor on signup volume (default: 1.0)")
    args = parser.parse_args()

    t0 = time.time()
    tables = run_simulation(seed=args.seed, scale=args.scale)
    write_outputs(tables)
    runtime = time.time() - t0

    print_calibration_summary(tables, runtime)


if __name__ == "__main__":
    main()
