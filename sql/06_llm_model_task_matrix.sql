-- 06_llm_model_task_matrix.sql
-- Model x task_type performance matrix: outcome mix, cost, cost per
-- *successful* run, and approximate p50/p90 latency (SQLite has no
-- PERCENTILE_CONT, so we rank rows within each group and pick the row at the
-- target rank — a standard window-function trick for percentile approximation).

WITH base AS (
    SELECT
        model,
        task_type,
        status,
        llm_cost_usd,
        latency_sec,
        input_tokens,
        output_tokens
    FROM agent_runs
),
group_counts AS (
    SELECT model, task_type, COUNT(*) AS n FROM base GROUP BY model, task_type
),
ranked_latency AS (
    -- Window functions: ROW_NUMBER over latency within each (model, task_type)
    -- group, plus the group's row count, let us pick the p50/p90-th ranked
    -- latency value directly (nearest-rank percentile method).
    SELECT
        model,
        task_type,
        latency_sec,
        ROW_NUMBER() OVER (PARTITION BY model, task_type ORDER BY latency_sec) AS rn,
        COUNT(*) OVER (PARTITION BY model, task_type) AS n
    FROM base
),
percentiles AS (
    SELECT
        model,
        task_type,
        MAX(CASE WHEN rn = CAST(0.5 * n AS INTEGER) + 1 THEN latency_sec END) AS p50_latency_sec,
        MAX(CASE WHEN rn = CAST(0.9 * n AS INTEGER) + 1 THEN latency_sec END) AS p90_latency_sec
    FROM ranked_latency
    GROUP BY model, task_type
),
agg AS (
    SELECT
        model,
        task_type,
        COUNT(*) AS n_runs,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS n_success,
        SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END) AS n_partial,
        SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END) AS n_failed,
        SUM(llm_cost_usd) AS total_llm_cost_usd,
        AVG(llm_cost_usd) AS avg_llm_cost_usd,
        AVG(input_tokens) AS avg_input_tokens,
        AVG(output_tokens) AS avg_output_tokens,
        SUM(CASE WHEN status = 'success' THEN llm_cost_usd ELSE 0 END) AS success_llm_cost_usd
    FROM base
    GROUP BY model, task_type
)
SELECT
    a.model,
    a.task_type,
    a.n_runs,
    a.n_success,
    a.n_partial,
    a.n_failed,
    ROUND(a.n_success * 100.0 / a.n_runs, 2) AS success_pct,
    ROUND(a.n_partial * 100.0 / a.n_runs, 2) AS partial_pct,
    ROUND(a.n_failed  * 100.0 / a.n_runs, 2) AS failed_pct,
    ROUND(a.avg_llm_cost_usd, 4) AS avg_llm_cost_usd,
    ROUND(a.total_llm_cost_usd, 2) AS total_llm_cost_usd,
    -- Cost per *successful* run: the real "cost of getting a working result",
    -- which penalizes cheap-but-unreliable models correctly.
    ROUND(a.total_llm_cost_usd / NULLIF(a.n_success, 0), 4) AS cost_per_success_usd,
    ROUND(a.avg_input_tokens, 0) AS avg_input_tokens,
    ROUND(a.avg_output_tokens, 0) AS avg_output_tokens,
    p.p50_latency_sec,
    p.p90_latency_sec
FROM agg a
JOIN percentiles p ON p.model = a.model AND p.task_type = a.task_type
ORDER BY a.task_type, a.model;
