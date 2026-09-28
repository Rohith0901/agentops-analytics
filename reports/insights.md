# AgentOps Analytics — Insights Report

> **All data in this report is synthetic / simulated** (see `src/generate_data.py`). CodeShift is a fictional company built for this portfolio project; no real customers, revenue, or usage are represented. Every number below is computed live from `data/agentops.db` — nothing is hard-coded.

## Executive summary

- Current MRR is **$118,864/mo** ($1,426,368 ARR), growing at a trailing 6-month CMGR of **9.3%/mo** (net new MRR of $49,093 over the last 6 months), across **560** active paying customers, with average month-over-month net revenue retention of **99%**.
- **Team plan gross margin (42%) is 27pp below Enterprise (69%)** — flat per-task credits under-price expensive workloads, most visibly **COBOL -> Java** migrations at $0.73 LLM-cost per credit earned.
- Activation (≥3 of first 5 runs succeeding) predicts outcomes: customers with all 5 first runs succeeding convert **16pp** more than those with zero, and in a discrete-time survival (hazard) model of monthly churn, the strongest driver of a customer's month-to-month churn hazard is **activation_score** (odds ratio 0.35, customer-grouped holdout AUC 0.62).
- Routing simple tasks to a cheaper model saves an estimated **$10,401/year**; failed runs alone burn **$77,300** in LLM spend (frontier-large/code_migration is the single costliest combo). Concentrating complex tasks on the highest-success model would cost an additional **$193,948/year** but raise success materially — a deliberate reliability trade-off, not a savings opportunity.
- The Pro $29→$39 price A/B test is **not statistically significant** (p=0.150) at an achieved power of only **30%** — the experiment was underpowered; detecting the observed effect at 80% power would need ~709 customers per arm versus the ~189 per arm actually assigned (196 control / 182 treatment).

## Customer analytics

### Finding: activation in the first 5 runs is the clearest predictor of who converts and stays
**Evidence:** self-serve conversion rises from 30% (0/5 successes) to 46% (5/5 successes); 6-month paid retention rises from 77% to 84% over the same range ([activation vs conversion figure](figures/03_activation_vs_conversion_retention.png)). In the discrete-time survival (hazard) model of monthly churn (one row per paid customer-month; see Methodology notes), `activation_score` has an odds ratio of 0.35 per paid customer-month (95% CI 0.17–0.69, p=0.003); the customer-grouped holdout ROC-AUC is 0.62 ([odds ratio forest plot](figures/04_churn_odds_ratios_forest_plot.png)). A Kaplan-Meier survival curve confirms the same story non-parametrically: activated customers show visibly higher paid-tenure survival than unactivated ones at every month observed ([Kaplan-Meier figure](figures/11_km_survival_by_activation.png)).
**Recommendation:** invest in first-session success (better task templates, clearer scoping guidance, a "golden path" onboarding task) since it is the single lever most tightly linked to both conversion and retention.
**Caveat:** the customer-grouped holdout AUC (0.62) shows the feature set is only modestly predictive out-of-sample for any given customer-month — churn here is driven by many small factors, not one dominant one, so this model should inform prioritization (which levers matter, directionally and with a confidence interval), not individual customer-level action.

### Finding: acquisition channel quality varies sharply
**Evidence:** Referral converts self-serve signups at 41% vs Paid Search at 17%, with paid-logo churn of 23.6% vs 39.0% respectively (`sql/05_channel_performance.sql`).
**Recommendation:** shift acquisition spend toward Referral/Community-style channels and treat Paid Search leads with extra onboarding support given their weaker downstream economics.
**Caveat:** channel is correlated with company size and industry in this dataset; the raw channel effect may partly reflect who each channel attracts rather than the channel itself.

### Finding: usage personas among paying customers — largest is 'Light Self-Serve'
**Evidence:** 1,399 customers (63% of all customers) never converted to paid and are reported as their own non-clustered segment (trivially defined, not from KMeans). Among the 808 customers who ever paid, KMeans (k=3, chosen by silhouette score subject to every cluster holding at least 5% of paying customers — see `reports/tables/persona_k_selection.csv` for the full k-search) finds 'Light Self-Serve' as the largest paid segment (362 customers, 45% of paying customers, silhouette=0.26) ([personas figure](figures/10_customer_personas.png)).
**Recommendation:** use these personas to prioritize product and support investment — e.g. proactive reliability outreach for failure-prone heavy users, migration tooling investment for migration-heavy segments, and a lighter-weight self-serve activation push aimed at the large never-converted Free segment.
**Caveat:** clusters are drawn from lifetime-average behavior on log-transformed, percentile-clipped features (to tame the heavy right skew in usage volume and cost) — a customer's persona can drift over its lifecycle, and the silhouette score (0.26) indicates real but moderate cluster separation, not perfectly distinct groups.

## LLM output analytics

### Finding: model ranking is consistent across task types (no Simpson's-paradox reversal detected)
**Evidence:** the pooled model success ranking (frontier-large, balanced-medium, fast-small, open-weights-hosted) agrees with the per-task-type ranking in every stratum checked (`reports/tables/llm_simpsons_reversal_check.csv`); a chi-square test of model vs outcome is significant (chi2=13,217, p<0.001, Cramér's V=0.15) ([model x task heatmap](figures/05_model_task_success_heatmap.png)).
**Recommendation:** it is safe to use the pooled model leaderboard for coarse decisions, but still route per-task since the *size* of each model's advantage varies a lot by task even though the *ranking* does not reverse.
**Caveat:** we only checked pairwise ranking agreement across task strata, not a full task-mix-adjusted regression; a stronger confound could still exist by customer segment.

### Finding: an explicit routing policy saves money on simple tasks, but complex tasks need the frontier model
**Evidence:** cheapest-model-within-2pp-of-best routing ([cost-quality frontier](figures/06_cost_quality_pareto_frontier.png)) saves an estimated **$10,401/year** on simple tasks (test_generation, bug_fix), while no cheaper model comes within 2pp of frontier-large's success rate on migrations, feature builds, refactors, or dependency upgrades — concentrating those on frontier-large would cost an additional **$193,948/year** but raise success materially.
**Recommendation:** implement automatic routing for simple tasks now (low risk, immediate savings); treat the reliability investment on complex tasks as a separate, deliberate pricing/margin decision (see Pricing section).
**Caveat:** projected dollars scale the observed window's run volume to a monthly/annual rate and assume task-type volume and mix stay constant — a genuinely new pricing or product motion would change both.

### Finding: failed runs are a large, concentrated cost
**Evidence:** failed runs burn **$77,300** in LLM cost across the observed window (18.9% of total LLM spend); frontier-large on code_migration is the single most expensive combo ($37,406).
**Recommendation:** add an early-exit / step-budget cap for runs trending toward failure, since failed runs already take materially more steps than successful ones before giving up.
**Caveat:** this is LLM $ cost only; it excludes the larger business cost of a failed task (support load, customer trust), which is not directly observable in this dataset.

### Finding: human ratings track run outcome closely
**Evidence:** Spearman correlation between human rating and outcome (failed<partial<success) is ρ=0.55 (p<0.001, n=72,875).
**Recommendation:** human ratings are a reasonable low-cost quality proxy where automated status isn't available (e.g. spot-checking partial/ambiguous outcomes).
**Caveat:** only ~25% of runs are rated, and coverage is not random with respect to outcome by design in the simulator; a real deployment should audit rating-selection bias.

## Pricing analytics

### Finding: Team plan margin is thinnest, driven by migration-heavy usage
**Evidence:** Team gross margin is 42% vs 69% for Enterprise ([margin by plan figure](figures/07_margin_by_plan.png)); at the task level, `code_migration` earns 0.528 USD margin per credit (negative/thin) vs the platform-wide blended rate ([cost vs revenue per credit figure](figures/08_cost_vs_revenue_per_credit_by_task.png)); customer-level margin deciles range from -6% to 86%.
**Recommendation:** re-price migration tasks with complexity-weighted credits (see pricing scenarios below: raising `code_migration` credits recovers $398,329 of margin in one scenario) rather than a flat per-task credit schedule.
**Caveat:** scenario projections apply a new price/credit schedule to *historical* volume and assume no behavior change; real repricing would change usage patterns (credit-conscious customers may shift task mix or plan).

### Finding: the Pro price A/B test is directionally consistent with a real effect, but underpowered
**Evidence:** control (196 customers) converts at 26.0%, treatment (182 customers) at 19.8% (diff 6.2pp, 95% CI [-2.2, 14.7]pp, p=0.150); achieved power is only 30% and detecting this effect at 80% power would need ~709 customers per arm (vs ~189 per arm actually assigned: 196 control / 182 treatment). Arc price elasticity of conversion is estimated at -0.93 ([A/B test figure](figures/09_ab_test_conversion_with_ci.png)).
**Recommendation:** do not conclude the $39 price is safe or unsafe from this test alone — either run it longer / at higher traffic, or treat $39 as a hypothesis to validate with a properly powered follow-up before a full rollout.
**Caveat:** the bootstrap CI on first-3-month revenue per assigned user (`reports/tables/ab_test_bootstrap_revenue.csv`) is wide and includes zero difference, consistent with the underpowered read from the proportion test.

### Pricing scenarios (see `reports/tables/pricing_scenarios.csv` for full assumptions)

| Scenario | Revenue (USD) | LLM cost (USD) | Gross margin % |
|---|---:|---:|---:|
| status_quo | 1,084,057 | 408,608 | 62.3% |
| complexity_weighted_migration_credits | 1,482,386 | 408,608 | 72.4% |
| route_simple_tasks_to_balanced_medium | 1,084,057 | 387,259 | 64.3% |
| pro_39 | 1,116,097 | 408,608 | 63.4% |

Each scenario's assumptions are stated in full in the `assumption` column of the CSV; none of these projections account for demand response to the new pricing.

## Methodology notes

- **Churn is modeled as a discrete-time survival (hazard) problem, not a cross-sectional 'ever churned?' classifier.** A first pass modeled churn as one row per customer with a plain 'did they ever churn' label — but that framing is right-censored: a customer who has only paid for one month clearly hasn't had the *chance* to churn yet, and lumping them in with a 20-month-tenured customer under the same 0/1 label discards that exposure-time information entirely (that model's holdout AUC was ~0.53, barely better than chance). The fix used here is the standard actuarial/epidemiological approach: build a person-period panel with one row per *paid customer-month*, where the label is 'did this customer churn at the end of THIS specific month' (1 for a churned customer's final paid month, 0 for every earlier month of theirs — we know for a fact they didn't churn then, since they show up paying again the next month). The one row that must be dropped rather than labeled is a still-active customer's most recent paid month, which is right-censored (we don't yet know whether they'll churn right after the data window ends). Features include tenure (+ tenure²) and a lagged (month t-1) failure rate / migration share so no same-month outcome information leaks into the prediction. The holdout AUC is evaluated with a **customer-grouped** split (`GroupShuffleSplit`), not a row-level random split, since randomly splitting person-period rows would let a customer's other months leak correlated outcome information into the test set. This lifted holdout AUC from ~0.53 to a more honest ~0.6 — getting the unit of analysis and the censoring right materially changes both the model's apparent skill and which features come out significant. A manual Kaplan-Meier curve (no external survival-analysis library) provides a non-parametric cross-check of the same conclusion.
- All monetary figures are in USD; all dates are ISO `YYYY-MM-DD`.
- SQL analyses live in `sql/*.sql` (run via `src/analysis/run_sql.py`); statistical analyses live in `src/analysis/{customer_analytics,llm_analytics,pricing_analytics}.py`; figures in `src/analysis/make_figures.py`.
- This report (`reports/insights.md`) is generated by `src/analysis/build_insights.py` by reading the CSVs those modules already wrote to `reports/tables/` — regenerate everything with `python src/analysis/run_all.py`.
