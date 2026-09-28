-- 09_task_cost_per_credit.sql
-- Cost per credit vs revenue per credit, by task_type. Flat credits per task
-- (2..6) mean every task type is priced the same per credit at the billing
-- layer, but LLM cost per credit varies enormously by task — this is the
-- direct evidence for the "flat credits mis-price migrations" pricing flaw.
--
-- revenue_per_credit_usd uses a single blended platform-wide rate (total paid
-- revenue / total credits used) since revenue is billed at the
-- customer-month level, not per task; it is a benchmark, not a per-task
-- revenue attribution.

WITH blended_rate AS (
    SELECT SUM(revenue_usd) * 1.0 / NULLIF(SUM(credits_used), 0) AS revenue_per_credit_usd
    FROM customer_months
    WHERE plan <> 'Free'
),
task_agg AS (
    SELECT
        task_type,
        COUNT(*) AS n_runs,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS n_success,
        SUM(llm_cost_usd) AS total_llm_cost_usd,
        SUM(credits_charged) AS total_credits_charged,
        AVG(credits_charged) AS credits_per_task
    FROM agent_runs
    GROUP BY task_type
)
SELECT
    t.task_type,
    t.n_runs,
    ROUND(t.n_success * 100.0 / t.n_runs, 2) AS success_pct,
    t.credits_per_task,
    ROUND(t.total_llm_cost_usd, 2) AS total_llm_cost_usd,
    ROUND(t.total_llm_cost_usd / NULLIF(t.total_credits_charged, 0), 4) AS llm_cost_per_credit_usd,
    ROUND(b.revenue_per_credit_usd, 4) AS blended_revenue_per_credit_usd,
    ROUND(
        b.revenue_per_credit_usd - (t.total_llm_cost_usd / NULLIF(t.total_credits_charged, 0)),
        4
    ) AS margin_per_credit_usd,
    -- Window function: rank task types by margin-per-credit, worst first.
    RANK() OVER (
        ORDER BY (b.revenue_per_credit_usd - (t.total_llm_cost_usd / NULLIF(t.total_credits_charged, 0))) ASC
    ) AS margin_per_credit_rank_worst_first
FROM task_agg t
CROSS JOIN blended_rate b
ORDER BY margin_per_credit_usd ASC;
