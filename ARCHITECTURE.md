# AgentOps Analytics — Architecture & Build Spec

Portfolio project analyzing a simulated AI coding-agent SaaS company
(autonomous agents for software migration/modernization). Core topics covered:
customer data · LLM output data · pricing data · reports/visualizations · Excel · SQL ·
Tableau/Power BI · statistics · Python.

The project analyses **CodeShift** (fictional company, fictional data) — a SaaS platform whose
AI agents run coding tasks (legacy migrations, bug fixes, test generation...) for customers.

## Hard rules for every builder
- Python 3.14, venv at `.venv/` (pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib,
  seaborn, plotly, streamlit, openpyxl already installed). Run with `.venv/bin/python`.
- All paths via `pathlib`, relative to project root (`ROOT = Path(__file__).resolve().parents[1]`).
- Deterministic: `np.random.default_rng(42)`.
- No hard-coded "findings" — every number in reports is computed from data.
- Clean, commented, readable code, meant to be read by other developers. Type hints, docstrings, small functions.
- Money in USD. Dates ISO `YYYY-MM-DD`.

## Directory layout
```
agentops-analytics/
  ARCHITECTURE.md
  README.md                    # final write-up (orchestrator)
  requirements.txt
  Makefile                     # make data | analysis | excel | dashboard | all
  data/
    raw/*.csv                  # generated tables (gitignored)
    agentops.db                # SQLite, same tables (gitignored)
    powerbi/                   # star-schema CSV export for Power BI / Tableau
  src/
    generate_data.py           # [Builder A]
    db.py                      # [Builder A] helper: get_conn(), run_sql_file(path) -> DataFrame
    analysis/
      run_sql.py               # [Builder B] executes sql/*.sql -> reports/tables/*.csv
      customer_analytics.py    # [Builder B] activation, retention, churn model, segmentation
      llm_analytics.py         # [Builder B] model x task performance, cost-quality, routing
      pricing_analytics.py     # [Builder B] unit economics, A/B test, power analysis
      make_figures.py          # [Builder B] reports/figures/*.png
      build_insights.py        # [Builder B] reports/insights.md (auto-generated numbers)
    excel_report.py            # [Builder C] reports/AgentOps_Report.xlsx
    powerbi_export.py          # [Builder C] data/powerbi/*.csv star schema
  sql/                         # [Builder B] schema-aware analytical SQL (SQLite dialect)
  dashboard/app.py             # [Builder D] Streamlit
  llm_benchmark/               # [Builder E] optional real LLM benchmark via Ollama
  reports/{figures,tables}/
```

## Data model (SQLite + CSV, produced by Builder A)

### `plans`
| plan | monthly_fee_usd | included_credits | overage_usd_per_credit |
|---|---|---|---|
| Free | 0 | 40 | NULL (hard cap) |
| Pro | 29 | 150 | 0.25 |
| Team | 149 | 500 | 0.22 |
| Enterprise | 1200 | 1000 | 0.18 |
(Builder A may tune numbers during calibration — see targets.)

### `models`
| model | usd_per_m_input | usd_per_m_output | sec_per_step |
|---|---|---|---|
| frontier-large | 3.00 | 15.00 | 11 |
| balanced-medium | 0.80 | 4.00 | 6 |
| fast-small | 0.25 | 1.25 | 3 |
| open-weights-hosted | 0.15 | 0.60 | 4.5 |

### `customers`
customer_id (C00001), company_name (plausible fake), signup_date, industry, company_size
(Startup/SMB/Mid-Market/Enterprise), region (India/North America/Europe/APAC), acquisition_channel
(Organic/Paid Search/Referral/Community/Outbound Sales), is_sales_led (0/1),
first_paid_date (NULL if never), current_plan, churn_date (NULL if active), status (active/churned).
**Do NOT store activation or engagement** — analysts must derive them.

### `customer_months` (billing, one row per customer per active month incl. Free)
customer_id, month (YYYY-MM-01), plan, monthly_fee_usd (actual price paid; Pro may be 39 for
experiment treatment), included_credits, credits_used, overage_credits, overage_revenue_usd,
revenue_usd (fee + overage). LLM cost is NOT stored here (derive by joining runs).

### `agent_runs` (~250k–400k rows)
run_id, customer_id, started_at (ISO datetime), task_type, source_stack, target_stack, model,
plan_at_run, steps, input_tokens, output_tokens, llm_cost_usd, latency_sec, lines_changed,
tests_total, tests_passed, status (success/partial/failed), credits_charged, human_rating (1–5, NULL ~75%).

### `experiment_assignments`
customer_id, experiment_name ('pro_price_2026Q2'), assigned_at, variant (control/treatment),
pro_price_shown (29/39). Conversion is derived by joining customers.first_paid_date.

## Simulation design (the "hidden truths" analysts should rediscover)
Window: 2025-01-01 → 2026-08-31 (20 months). Signups grow ~6%/month from ~60/month (~2,200 total).

1. **Task mix & legacy:** industries carry a legacy score (Banking .9, Government .85, Insurance .8,
   Telecom .6, Healthcare .55, Fintech .35, E-commerce .3, SaaS .2). Higher legacy → more
   `code_migration` and more COBOL→Java. Task types: code_migration, bug_fix, test_generation,
   refactor, dependency_upgrade, feature_build (credits flat per task: 2..6).
   Migration pairs (difficulty mult / token mult): COBOL→Java (.70/1.8), Java 8→Java 21 (1.05/1.0),
   Python 2→Python 3 (1.15/.8), AngularJS→React (.85/1.3), .NET Framework→.NET 8 (.95/1.1),
   PHP→Node.js (.85/1.2). Non-migration tasks: source=target in {Python, Java, TypeScript, Go, C#}.
2. **Model quality vs cost:** base success (frontier) test_gen .93, bug_fix .84, refactor .86,
   dep_upgrade .82, feature_build .72, migration .80×pair difficulty. Multipliers
   (simple tasks / complex tasks): balanced .98/.86, fast-small .90/.66, open-weights .84/.55.
   → **balanced-medium ≈ frontier on simple tasks at ~1/4 cost** (routing opportunity).
   Smaller models take more steps; failed runs take ~1.6× steps (agent loops) → failures are costly.
3. **Router assigns model randomly by plan** (Free: no frontier; Enterprise: mostly frontier).
4. **Activation (aha moment):** success rate of a customer's first 5 runs drives free→paid conversion
   and lowers paid churn. Free users mostly on small models → lower activation.
5. **Channel effects:** Referral/Community retain & convert best; Paid Search worst.
6. **Reliability → churn:** a month with high failure rate raises next-month churn hazard.
7. **Expansion:** credits_used > 115% of included → chance to upgrade Pro→Team, Team→Enterprise (NRR).
8. **Pricing flaw:** flat credits per task means COBOL migrations on frontier cost more LLM $ than
   the credit revenue they earn on Team → Team/migration-heavy customers have thin/negative margin.
9. **Price A/B test** 2026-03-01..2026-05-31, self-serve signups 50/50: Pro $29 vs $39;
   treatment conversion ×0.85 (for Pro-destined users). Likely under-powered → power analysis matters.

**Calibration targets:** overall gross margin (revenue − LLM cost) 45–65%; Team plan margin clearly
the weakest paid plan; overall run success ~70–78%; free→paid conversion ~25–35%; paid monthly
logo churn ~3–6%; total runs 250k–400k; generator runs in < 60s.

## Analysis questions (Builder B) — each covers a core analytics topic
- Customer: MRR/ARR trend, NRR & GRR, cohort logo retention triangle, activation threshold vs
  conversion & 6-month retention, churn drivers (logistic regression w/ odds ratios + AUC),
  KMeans personas on usage features.
- LLM: success / cost / latency by model × task (stratified — beware Simpson's paradox),
  cost per *successful* run, cost-quality frontier, optimal routing policy (min cost s.t. success
  within 2pp of frontier per task) + projected $ savings, chi-square model vs outcome, rating vs status.
- Pricing: revenue vs LLM cost per plan & per task (cost per credit vs revenue per credit),
  margin by customer decile, A/B test (two-proportion z-test, bootstrap CI on revenue per signup,
  power / MDE / required sample size), price elasticity estimate, pricing scenarios.

## Testing & security requirements (all builders)
- **Tests:** every module gets pytest tests in `tests/test_<module>.py` (fast: use a small
  `--scale` dataset or a temp SQLite fixture, never the full DB). Must pass:
  `.venv/bin/python -m pytest -q`. Test data invariants too (no negative costs, tokens ≥ 0,
  tests_passed ≤ tests_total, runs only while customer active, revenue = fee + overage).
- **SQL safety:** never build SQL with f-strings/`%`/`+` from any input. Use `?` parameters.
  Identifiers (table names) only from a hard-coded allowlist. Dashboard opens SQLite
  **read-only** (`file:...?mode=ro` URI, `uri=True`).
- **Dashboard:** no free-text SQL box, no file upload, no `unsafe_allow_html=True` with data,
  no `eval`/`exec`/`pickle` of anything. Filters are allow-listed widget values only.
  `.streamlit/config.toml`: `server.enableXsrfProtection=true`, `server.enableCORS=false`,
  `browser.gatherUsageStats=false`, `client.toolbarMode="viewer"`, `server.fileWatcherType="none"`.
- **Excel:** values written as data, never as formulas derived from data strings
  (guard against CSV/formula injection: prefix cells starting with `= + - @` from text fields
  with `'`). Only author-written formulas in the simulator sheet. Same guard in Power BI CSVs.
- **Secrets & repo hygiene:** no API keys/tokens anywhere; `.gitignore` covers `.venv/`,
  `data/raw/`, `data/*.db`, `__pycache__/`, `.env*`, `.DS_Store`, `*.pyc`.
- **Supply chain:** pinned versions in `requirements.txt` (`==`), checked with `pip-audit`.
- **Static analysis:** `bandit -r src dashboard llm_benchmark -q` clean or justified `# nosec`.
