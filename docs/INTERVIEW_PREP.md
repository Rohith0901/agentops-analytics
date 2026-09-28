# Interview Prep — AgentOps Analytics

Everything here is plain-English study material for Rohith. It assumes you already know the project exists — the goal is to be able to *defend* every number and every method choice, not just recite them.

---

## 1. The 60-second pitch

"I built an end-to-end analytics project on a simulated AI-coding-agent SaaS company called CodeShift. I generated a synthetic dataset with deliberately planted business dynamics — things like 'customers who succeed early stick around longer' or 'one pricing plan quietly loses money on a specific workload' — and then built the whole analytics stack an analyst would actually use to *find* those things without being told they're there: SQL queries, Python statistical models, an Excel workbook with a live pricing simulator, a Power BI star schema, and a Streamlit dashboard. I also built a side project that benchmarks real local LLMs (Llama 3.2 via Ollama) with the same evaluation lens, since I've used local LLMs before in two other projects. The point wasn't to build a flashy dashboard — it was to practice the actual judgment calls: how do you avoid Simpson's paradox, how do you handle right-censoring in a churn model, how do you tell someone their A/B test failed because it was underpowered rather than because there's no effect."

## 2. The 2-minute walkthrough script

1. **Start with the README's Key Findings.** "Here are five things the analysis surfaced — each one links to a chart and a recommendation."
2. **Point at the architecture diagram.** "Data flows one way: a generator produces a SQLite database, SQL and Python analyze it, and three different outputs consume the same numbers — an auto-generated markdown report, an Excel workbook, and a Streamlit dashboard. Nothing is entered twice."
3. **Open one figure that shows a 'twist'** — the churn odds-ratio forest plot is a good one: "This took two tries. My first churn model was a plain classifier and only just beat a coin flip. Reframing it as a proper survival problem fixed that — I'll explain why if you want."
4. **Open the Pricing Simulator sheet in Excel.** "Every output cell is a live formula, not a value I typed in. There's a locked 'baseline' block next to every live input so that at default settings the delta is exactly zero — that's a model-vs-model sanity check, not a model-vs-data one."
5. **Close on the honest limitations.** "It's synthetic data, so the findings describe the simulator's planted rules, not a real market — but the methods (survival modeling, power analysis, Pareto routing) are the transferable part, and I'd apply the same discipline to real data."

---

## 3. Analyses, one at a time

For each: what question, which method, why that method (vs. the obvious alternative), how to read the output, one key number, and the honest caveat.

### MRR / CMGR / NRR / GRR
- **Question:** How is recurring revenue trending, and is it growth from new logos or expansion of existing ones?
- **Method:** MRR = sum of `monthly_fee_usd` over paid customer-months (excludes overage — that's "revenue," a separate, larger number). CMGR = compound monthly growth rate over a trailing window, more stable than a single month-over-month % for a report headline. NRR/GRR computed month N vs. month N+1 revenue on the cohort that was paying in month N (GRR floors expansion at 100%, so it isolates pure retention from upsell).
- **Why not just "% change month over month"?** A single MoM % is noisy and doesn't distinguish "we lost logos" from "we lost expansion revenue" — NRR/GRR split that apart, which is the standard SaaS lens.
- **Read it as:** MRR trend chart, `figures/01_mrr_trend.png`; the table in `reports/tables/01_monthly_kpis.csv` and `02_nrr_grr.csv`.
- **Key number:** current MRR and trailing-6-month CMGR are both stated at the top of `reports/insights.md` (executive summary) — read that file for the live figure rather than memorizing a number that regenerates.
- **Caveat:** NRR/GRR here is month-over-month, not the more common trailing-12-month cohort definition, because the 20-month observation window is too short for a clean trailing-12m cohort near the start of the series (documented in `sql/02_nrr_grr.sql`).

### Cohort retention (logo retention triangle)
- **Question:** Of everyone who signed up in month X, what fraction is still a *paying* customer N months later?
- **Method:** Signup-month cohort × months-since-signup grid, gated on "paying," not "any activity" — a SQL self-join / grouping, no special library needed.
- **Why paid, not "active"?** For a subscription business, logo retention is the number that maps directly to revenue risk; "still logged in" doesn't.
- **Read it as:** the heatmap (`figures/02_cohort_retention_heatmap.png`) — darker/higher = more of that cohort still paying at that month mark.
- **Caveat:** early cohorts have more months of data than recent ones, so don't compare a 1-month-old cohort's month-12 cell to nothing.

### Activation threshold
- **Question:** Is there a "aha moment" number of early successes that predicts who sticks around?
- **Method:** Bucket customers by successes-out-of-first-5-runs (0 through 5), then look at self-serve conversion rate and 6-month paid retention in each bucket.
- **Why buckets, not a continuous correlation?** Buckets make a threshold effect visible (e.g., "3 of 5" is where retention visibly steps up) — a single correlation coefficient would hide that shape.
- **Read it as:** `figures/03_activation_vs_conversion_retention.png` — two lines, both rising with activation bucket.
- **Key number:** conversion rises from 30% (0/5 successes) to 46% (5/5); 6-month retention rises from 77% to 84% (`reports/insights.md`).
- **Caveat:** this is descriptive, not causal — customers who happen to succeed early may also be inherently better-fit customers (e.g., simpler task mix), not purely "activation causes retention."

### Hazard model, censoring, odds ratios, AUC
- **Question:** What actually predicts which customer churns in which month?
- **Method — the interview-relevant one.** First attempt: one row per customer, label = "did they ever churn" (a plain binary classifier). That's **right-censored**: a customer who's only paid one month hasn't had the *chance* to churn yet, but gets the same "0" label as a 20-month veteran who also didn't churn. That silently teaches the model "short tenure = safe," which is backwards. Fix: rebuild as a **discrete-time survival / hazard model** — a person-period panel with one row per *paid customer-month*, label = "did they churn at the end of *this* month" (1 only on a churned customer's last paid month; 0 on every earlier month, because we *know* they didn't churn then — they show up paying again next month). The one row you must drop, not label, is a still-active customer's most recent month (we don't yet know their fate).
- **Why logistic regression over something fancier (Cox PH, gradient boosting)?** Logistic on the person-period panel gives odds ratios that are directly interpretable to a non-technical stakeholder ("activation cuts the monthly odds of churn by 65%"), and is the standard, well-understood way to do discrete-time survival without a dedicated survival library.
- **Odds ratio, plainly:** an odds ratio of 0.35 for `activation_score` means each one-unit increase in activation is associated with the odds of churning that month being multiplied by 0.35 (a 65% relative reduction) — not that churn probability itself drops 65 points.
- **AUC:** area under the ROC curve on a **customer-grouped holdout** (`GroupShuffleSplit`, not a row-level random split — randomly splitting person-period rows would leak a customer's own correlated other months into the test set, inflating the score). Getting the split unit right dropped a misleadingly optimistic score down to an honest ~0.62.
- **Read it as:** the forest plot (`figures/04_churn_odds_ratios_forest_plot.png`) — bars crossing 1.0 are not significant; `activation_score`, `tenure_months`, `tenure_months_sq`, and `acquisition_channel_Paid Search` are the ones with confidence intervals that exclude 1.0 (`reports/tables/churn_odds_ratios.csv`).
- **Caveat:** AUC ≈ 0.62 is only modestly predictive out-of-sample — good enough to say *which levers matter, directionally, with a confidence interval*, not good enough to flag an individual customer as "about to churn."

### KMeans + silhouette (customer personas)
- **Question:** Are there natural usage-based customer segments worth treating differently?
- **Method:** KMeans on lifetime usage features (run volume, success rate, migration share, etc.), with the number of clusters (k) chosen by **silhouette score** rather than picked arbitrarily.
- **Why silhouette over the elbow method?** Silhouette gives a single number per k that balances within-cluster tightness against between-cluster separation, so "which k is best" is itself a data-driven, defensible choice rather than an eyeballed elbow.
- **Read it as:** the persona scatter/summary (`figures/10_customer_personas.png`); `reports/tables/persona_profile.csv` for what defines each cluster.
- **Caveat:** clusters are built from *lifetime-average* behavior — a customer's persona can drift over its life, and this snapshot won't show that movement.

### Simpson's paradox check
- **Question:** Does the "best model overall" ranking hold up once you split by task type, or does it flip (a real risk whenever a confound — here, task difficulty — differs by group)?
- **Method:** Compute the pooled model success ranking, then the same ranking within every task-type stratum, and check they agree everywhere (`reports/tables/llm_simpsons_reversal_check.csv`). Backed by a chi-square test of model vs. outcome for statistical significance of the association.
- **Why check this at all?** Because it's a classic real trap: if, say, the frontier model gets *disproportionately* assigned the hardest tasks (which it does here — Enterprise plans skew frontier and skew complex), its pooled success rate could look artificially *worse* than a cheaper model that only ever sees easy tasks — the reversal is not hypothetical, it's the exact risk this dataset was built to create the *conditions* for.
- **Read it as:** the heatmap (`figures/05_model_task_success_heatmap.png`) — read each row (task type) independently rather than only the marginal/pooled row.
- **Caveat:** only pairwise ranking *agreement* was checked per stratum, not a full task-mix-adjusted regression — a subtler confound (e.g., by customer segment) could still exist.

### Chi-square (model × outcome)
- **Question:** Is the observed difference in outcome mix (success/partial/failed) across models more than sampling noise?
- **Method:** Chi-square test of independence on the model × outcome contingency table, plus Cramér's V for effect size (chi-square alone doesn't tell you if a significant result is *practically* large).
- **Why chi-square over, say, a t-test?** The outcome variable here is categorical (three levels), and we're testing association between two categorical variables — chi-square is the standard tool for that, not a t-test (which needs a continuous outcome).
- **Read it as:** `reports/tables/llm_chi_square_model_outcome.csv` — a large chi2 and small Cramér's V together mean "statistically real but only a modest effect size," which is exactly this project's result.

### Pareto cost-quality frontier & routing policy (+ bootstrap CI)
- **Question:** For each task type, which model gives the best success rate per dollar, and is there a cheaper model that's "good enough"?
- **Method:** Plot cost vs. success per (model, task) pair; the routing policy picks, per task type, the **cheapest model whose success rate is within 2 percentage points of the best observed** — an explicit, defensible threshold rather than "always pick the frontier model" or "always pick the cheapest."
- **Why a fixed pp threshold rather than optimizing a cost-per-success-point trade-off directly?** A simple, stated threshold ("within 2pp of the ceiling") is transparent and easy for a non-technical stakeholder to sanity-check and adjust, versus a black-box optimization with an implicit price-per-quality-point that's harder to defend in a room.
- **Read it as:** `figures/06_cost_quality_pareto_frontier.png` (points toward the top-left dominate); `reports/tables/llm_routing_policy.csv` has the per-task recommendation and the projected annual dollar impact (positive = savings from routing to cheaper model; negative = the cost of *investing* in reliability on complex tasks by concentrating them on the frontier model).
- **Key number:** simple tasks (test_generation, bug_fix) route to a cheaper model at positive dollar impact; complex tasks (migration, feature_build, refactor, dependency_upgrade) show a *negative* dollar impact when concentrated on the frontier model — that's a deliberate reliability spend, not a mistake in the policy.
- **Caveat:** projected dollars scale the observed window's run volume to a monthly/annual rate assuming task mix and volume stay constant.

### Unit economics & cost-per-credit
- **Question:** Is the flat "credits per task" billing model mispricing certain task types?
- **Method:** Compare cost-per-credit (LLM $ spent ÷ credits consumed) against revenue-per-credit (blended platform-wide $/credit, since billing charges a single flat rate) by task type and by plan.
- **Why a blended revenue-per-credit rate rather than a plan-specific one?** Credits are billed identically across task types within a plan (that's the "flaw" being surfaced) — using a single blended rate as the comparison baseline is what makes the mismatch visible; using a task-specific rate would hide the very problem being diagnosed.
- **Read it as:** `figures/08_cost_vs_revenue_per_credit_by_task.png` — bars above the reference line cost more per credit than they earn.
- **Caveat:** this is a lifetime-average view; a customer's task mix can shift over time in ways this snapshot doesn't capture.

### A/B test: two-proportion z-test, power, MDE, required n
- **Question:** Does raising Pro's price from $29 to $39 change self-serve conversion, and can we trust that answer?
- **Method:** Two-proportion z-test on control vs. treatment conversion (pooled-proportion standard error, standard for comparing two independent binomial rates). Then, because the result wasn't significant, a **power analysis**: given the actual sample sizes and observed effect, what power did the test actually have, and what sample size would be needed to detect that same effect at the conventional 80% power target?
- **Why do the power analysis at all — isn't "not significant" the answer?** "Not significant" and "no effect" are different claims. A test can fail to reach significance either because there's truly no effect, *or* because it didn't have enough data to reliably detect a real one. The power calculation is what tells them apart here.
- **Read it as:** `reports/tables/ab_test_two_proportion_z.csv` (control/treatment n, conversion, p-value) and `reports/tables/ab_test_power_analysis.csv` (achieved power, MDE at 80% power, required n per arm).
- **Key numbers:** control converts higher than treatment (a multi-point gap), p=0.15, achieved power ≈30%, and the required sample size per arm to detect the *observed* effect at 80% power is roughly 3.5–4x the actual per-arm sample size that was run.
- **Caveat:** this reads as "directionally consistent with a real effect, but underpowered" — it is *not* evidence that $39 is safe, nor definitive proof the gap is real; the honest conclusion is "run it longer, or run a properly powered follow-up."

### Price elasticity (arc elasticity)
- **Question:** How sensitive is conversion to the Pro price change, expressed as a single elasticity number the pricing simulator can reuse?
- **Method:** **Arc elasticity** — % change in quantity (conversion) over % change in price, using the *midpoint* of the two prices/quantities as the denominator, rather than point elasticity anchored at one endpoint.
- **Why arc over point elasticity?** Arc elasticity is symmetric — it gives the same magnitude whether you frame the move as $29→$39 or $39→$29 — which matters because the Excel pricing simulator's slider can move price either direction from either baseline; point elasticity would silently depend on which price you anchored at.
- **Read it as:** `reports/tables/pricing_arc_elasticity.csv` — a negative number is expected (higher price → lower conversion); the magnitude is what feeds the simulator's default elasticity input.
- **Caveat:** estimated from one price move over one quarter for one plan — it's a starting assumption for the simulator, not a robust, externally-validated elasticity curve.

### Kaplan–Meier
- **Question:** Non-parametric cross-check — does paid-tenure survival actually look different for activated vs. non-activated customers, without assuming any particular model form?
- **Method:** A manually implemented Kaplan–Meier estimator (no external survival library) computing the step-function survival probability at each observed churn month, split by activation status.
- **Why build it by hand instead of trusting the hazard model alone?** It's an independent, assumption-light cross-check — the hazard/logistic model assumes a particular functional form (log-odds linear in the features); Kaplan–Meier makes no such assumption, so agreement between the two increases confidence the finding is real and not a modeling artifact.
- **Read it as:** `figures/11_km_survival_by_activation.png` — two step-down curves; the activated-customer curve should sit visibly above the non-activated one at every month.
- **Caveat:** unlike the hazard model, this doesn't control for any other feature (tenure, channel, etc.) — it's a single-variable comparison.

---

## 4. SQL concepts used (with real snippets)

**CTEs (Common Table Expressions)** — `WITH name AS (...)` blocks that name an intermediate result so a long query reads top-to-bottom like a pipeline instead of one deeply nested subquery. From `sql/10_customer_margin_deciles.sql`:

```sql
WITH run_cost_by_customer AS (
    SELECT customer_id, SUM(llm_cost_usd) AS llm_cost_usd,
           SUM(CASE WHEN task_type = 'code_migration' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS migration_share
    FROM agent_runs
    GROUP BY customer_id
),
customer_econ AS (
    SELECT cm.customer_id, SUM(cm.revenue_usd) AS lifetime_revenue_usd, ...
    FROM customer_months cm
    LEFT JOIN run_cost_by_customer rc ON rc.customer_id = cm.customer_id
    WHERE cm.plan <> 'Free'
    GROUP BY cm.customer_id
)
```
Plain English: "first figure out each customer's lifetime LLM cost and migration mix, *then* join that onto their billing history" — each CTE is a named, reusable step.

**Window functions** — functions that compute a value *per row* while still seeing other rows in a group, without collapsing the result the way `GROUP BY` does. Two used here:

- `NTILE(10)` — splits customers into 10 equal-sized buckets ordered by margin, used for the margin-decile analysis (`sql/10_customer_margin_deciles.sql`).
- `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` plus `COUNT(*) OVER (PARTITION BY ...)` — SQLite has no built-in percentile function, so `sql/06_llm_model_task_matrix.sql` ranks every row's latency within its (model, task_type) group and picks the row sitting at the 50th/90th percentile rank — a standard workaround:

```sql
ranked_latency AS (
    SELECT model, task_type, latency_sec,
           ROW_NUMBER() OVER (PARTITION BY model, task_type ORDER BY latency_sec) AS rn,
           COUNT(*) OVER (PARTITION BY model, task_type) AS n
    FROM base
)
```
Plain English: "within each model+task group, rank every run by how slow it was, and remember how big that group is — then pick the row at the 50%/90% mark of that ranking."

---

## 5. 25 likely interview questions with concise model answers

1. **Walk me through this project.** → Use the 60-second pitch above.
2. **Why did you make up your own data instead of using a public dataset?** → I wanted planted, known ground-truth effects so I could verify the analysis actually *finds* them, not just produces plausible-looking numbers — and a coding-agent SaaS domain lets me speak directly to the JD (LLM output data, pricing, coding-agent product).
3. **Isn't synthetic data kind of a cop-out?** → It's a controlled experiment, not a cop-out — the honest framing (stated up front in the README and every report) is that it validates method, not market truth; the next step with real data is in Q25.
4. **What would you do differently with real company data?** → Expect messier joins, missing/duplicate keys, actual outliers and fraud-like anomalies, privacy/PII handling, and I'd need to validate my "known" effects against domain experts instead of a generator script — plus real elasticity and churn drivers won't be as clean as a single dominant feature.
5. **What's the single most important finding and why?** → Activation → retention/conversion link, because it's actionable (product/onboarding investment) and statistically corroborated two independent ways (a hazard model *and* a non-parametric Kaplan-Meier curve agreeing).
6. **Explain the churn model in one minute.** → Use the hazard-model section above: person-period panel, right-censoring, customer-grouped holdout, odds ratios.
7. **Why not just use "did they churn ever" as a label?** → Right-censoring — see hazard model caveat above.
8. **What's an odds ratio, in plain English?** → Multiplies the odds of the event; 0.35 means the odds of churning that month are cut to about a third, not that churn probability drops by 65 percentage points.
9. **AUC of 0.62 — is that good?** → Modest but real; better than chance (0.5), meaningfully lower than a "great" model (~0.8+); good enough to prioritize levers, not to flag individuals.
10. **What is Simpson's paradox and did you find one?** → A trend that reverses when you split the data into subgroups because a confound differs across groups; checked explicitly here and *did not* find a reversal — the pooled ranking held in every task stratum.
11. **Why chi-square for the model/outcome test?** → Two categorical variables (model, outcome) — chi-square tests whether they're independent; paired with Cramér's V so a "significant" result is also sized honestly.
12. **How did you choose k for KMeans?** → Ran multiple k values and picked the one with the highest silhouette score, rather than eyeballing an elbow chart.
13. **What's the Pareto/cost-quality frontier routing policy, and what would you tell a PM about it?** → Route simple tasks to the cheapest model within 2pp of best success (real savings); keep complex tasks on the frontier model (a deliberate reliability spend, not a missed savings opportunity) — the two should not be pitched the same way.
14. **The A/B test wasn't significant — so the new price doesn't matter?** → No — it means the test didn't have enough power to tell either way; achieved power was low and I calculated exactly how much more data would be needed.
15. **What's statistical power, plainly?** → The probability your test detects a real effect of a given size, given your sample size — low power means a "no effect" result is uninformative, not reassuring.
16. **What is arc elasticity and why not point elasticity?** → Uses the midpoint of the two prices/quantities as the base, so it gives the same magnitude regardless of which direction you frame the price move — matters because the pricing simulator lets you move price either way from either starting point.
17. **Why bootstrap the revenue confidence interval instead of just using the t-test's CI?** → Bootstrapping makes no normality assumption about the revenue distribution (which is right-skewed — a few large customers) — a safer cross-check than a parametric interval on skewed data.
18. **How is the Pricing Simulator's Excel sheet trustworthy — how do I know the formulas are right?** → Every live formula has a "Baseline (locked)" reference computed with the identical formula structure at status-quo inputs; at default inputs, Δ margin is exactly zero everywhere by construction, and a hidden `_check` sheet independently recomputes every simulator output in Python to catch any formula drift — that's a model-vs-model check as well as a model-vs-data one.
19. **What does the Pricing Simulator actually let you change?** → Per-plan price, included credits, overage rate, price elasticity of customers, the % of simple-task frontier runs routed to a cheaper model, and a churn-rate assumption — all editable inputs with Excel data-validation guards (e.g., elasticity clamped to a sane range), with margin/revenue recomputing live via author-written formulas.
20. **How did you make sure the Excel/Power BI exports are safe?** → Formula-injection guard: any text value starting with `= + - @` is prefixed with `'` before writing, in both the Excel and Power BI CSV export paths — otherwise a stray customer name like `=CMD(...)` could execute as a formula when opened.
21. **How is the dashboard secured?** → Opens SQLite strictly read-only (`file:...?mode=ro`, `uri=True` — any write attempt raises an error), every query is parameterized and hard-coded (no free-text SQL box), and every filter widget's choices come from a hard-coded allow-list re-validated server-side.
22. **How is the local-LLM benchmark's untrusted code execution kept safe?** → Runs in a separate subprocess with an isolated interpreter (`python -I -S`), a stripped environment, a fresh temp directory, POSIX resource limits, an audit hook blocking sockets/subprocess/exec/ctypes, and a wall-clock timeout — layered defenses, explicitly documented as "a speed bump, not a hard security boundary" for untrusted code at scale.
23. **What did the AI tools do vs. what did you do?** → Be honest: AI-assisted tools helped generate code faster and drafted this document, but *I* chose which analyses to run and why (survival modeling over a naive churn classifier, arc elasticity over point elasticity, the 2pp routing threshold, the locked-baseline simulator design), decided what the calibration targets should be, and I need to be able to defend every one of those choices in my own words — which is the actual point of this prep doc.
24. **What was the hardest bug or design decision?** → The churn model's censoring bug is the strongest story: a naive "did they ever churn" model looked almost like a coin flip (AUC ~0.53); recognizing it as a *censoring* problem, not a modeling problem, and rebuilding as a person-period panel with a customer-grouped holdout, lifted it to a materially more honest ~0.62 — and changed which features looked significant.
25. **What would you build next if given more time / real data?** → A time-varying (not lifetime-average) persona model that tracks segment drift; a properly powered, longer-running version of the pricing A/B test; a task-mix-adjusted regression to fully rule out confounded Simpson's-paradox risk; and wiring the real Ollama benchmark results in for a head-to-head with the simulated model-cost assumptions.

---

## 6. Things to practise yourself (don't just read this)

- [ ] Re-run `make analysis` from scratch and watch the numbers regenerate — confirm you understand every step in `src/analysis/run_all.py`'s order.
- [ ] Open the Pricing Simulator sheet in `reports/AgentOps_Report.xlsx`, change one input (e.g. the routing %), and narrate out loud what should happen to margin *before* you look at the recomputed number.
- [ ] Pick one figure in `reports/figures/` at random and explain it fully, from memory, without opening `reports/insights.md`.
- [ ] Re-run the SQL behind one finding directly (e.g. `sql/05_channel_performance.sql`) via `sqlite3 data/agentops.db < sql/05_channel_performance.sql` (or through `src/analysis/run_sql.py`) and check it matches the CSV in `reports/tables/`.
- [ ] Explain the censoring bug in the churn model to a non-technical friend in under 90 seconds, with no notes.
- [ ] Run `make benchmark-mock` and then read `llm_benchmark/results/summary.md` — be ready to explain pass@k and Wilson confidence intervals without looking anything up.

---

## 7. Resume bullets (XYZ format: accomplished X, measured by Y, by doing Z)

1. Built an end-to-end analytics pipeline (SQL + Python + Excel + Power BI + Streamlit) on a 2,200-customer, ~292k-agent-run simulated SaaS dataset, surfacing a churn-driving activation effect (odds ratio 0.35, customer-grouped holdout AUC 0.62) by reframing a naive churn classifier as a right-censoring-aware discrete-time survival model.
2. Designed a cost-quality routing policy across 4 LLM models and 6 coding-agent task types that identified real, task-specific savings opportunities on simple tasks while correctly flagging complex tasks (migrations, feature builds) as requiring continued investment in the highest-success model, backed by a Simpson's-paradox stratification check and a chi-square significance test.
3. Diagnosed an underpowered pricing A/B test (p=0.15 at ~30% achieved power) via a two-proportion z-test, bootstrap confidence intervals, and a formal power/minimum-detectable-effect analysis, then built a live-formula Excel pricing simulator (locked baseline, data-validated inputs, Python-verified `_check` sheet) with 158 passing pytest tests covering data integrity, SQL correctness, security (parameterized queries, read-only DB access, formula-injection guards), and a sandboxed local-LLM benchmark harness.
