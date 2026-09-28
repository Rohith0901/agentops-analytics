-- 12_failure_cost.sql
-- LLM $ burned on failed runs, by model and task_type — failed runs loop and
-- retry (STATUS_STEP_MULT in generate_data.py), so they are the most
-- expensive-per-outcome category. Quantifies the direct dollar cost of
-- unreliability, separate from the opportunity cost of the unfinished task.

WITH failed AS (
    SELECT model, task_type, llm_cost_usd, steps
    FROM agent_runs
    WHERE status = 'failed'
),
failed_agg AS (
    SELECT
        model,
        task_type,
        COUNT(*) AS n_failed_runs,
        SUM(llm_cost_usd) AS failed_llm_cost_usd,
        AVG(steps) AS avg_steps_on_failure
    FROM failed
    GROUP BY model, task_type
),
total_cost AS (
    SELECT SUM(llm_cost_usd) AS total_llm_cost_usd FROM agent_runs
)
SELECT
    f.model,
    f.task_type,
    f.n_failed_runs,
    ROUND(f.failed_llm_cost_usd, 2) AS failed_llm_cost_usd,
    ROUND(f.avg_steps_on_failure, 2) AS avg_steps_on_failure,
    ROUND(f.failed_llm_cost_usd * 100.0 / t.total_llm_cost_usd, 3) AS pct_of_total_llm_spend,
    -- Window function: rank (model, task_type) combos by absolute dollars
    -- burned on failure — the "where to focus reliability work first" list.
    RANK() OVER (ORDER BY f.failed_llm_cost_usd DESC) AS failure_cost_rank,
    -- Running cumulative share of total failed-run spend, ordered by rank,
    -- to answer "how many combos account for 80% of wasted spend".
    ROUND(
        SUM(f.failed_llm_cost_usd) OVER (ORDER BY f.failed_llm_cost_usd DESC
                                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        * 100.0 / SUM(f.failed_llm_cost_usd) OVER (),
        2
    ) AS cumulative_pct_of_failed_spend
FROM failed_agg f
CROSS JOIN total_cost t
ORDER BY failed_llm_cost_usd DESC;
