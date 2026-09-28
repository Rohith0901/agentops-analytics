-- 07_migration_pairs.sql
-- Legacy-migration pair performance: which source->target migrations are
-- hardest (lowest success), most expensive, and most common. Evidence for
-- the "COBOL->Java is disproportionately costly" pricing flaw.

WITH migrations AS (
    SELECT
        source_stack,
        target_stack,
        status,
        llm_cost_usd,
        steps,
        credits_charged
    FROM agent_runs
    WHERE task_type = 'code_migration'
),
agg AS (
    SELECT
        source_stack || ' -> ' || target_stack AS migration_pair,
        COUNT(*) AS n_runs,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS n_success,
        AVG(llm_cost_usd) AS avg_llm_cost_usd,
        SUM(llm_cost_usd) AS total_llm_cost_usd,
        AVG(steps) AS avg_steps,
        AVG(credits_charged) AS avg_credits_charged,
        SUM(credits_charged) AS total_credits_charged
    FROM migrations
    GROUP BY migration_pair
)
SELECT
    migration_pair,
    n_runs,
    n_success,
    ROUND(n_success * 100.0 / n_runs, 2) AS success_pct,
    ROUND(avg_llm_cost_usd, 4) AS avg_llm_cost_usd,
    ROUND(total_llm_cost_usd, 2) AS total_llm_cost_usd,
    ROUND(avg_steps, 2) AS avg_steps,
    avg_credits_charged,
    -- LLM $ burned per credit earned on this migration pair — the direct
    -- evidence that flat per-task credits mis-price harder migrations.
    ROUND(total_llm_cost_usd / NULLIF(total_credits_charged, 0), 4) AS llm_cost_per_credit_usd,
    -- Window function: rank pairs by cost-per-credit so the worst-priced
    -- migration is immediately identifiable.
    RANK() OVER (ORDER BY total_llm_cost_usd * 1.0 / NULLIF(total_credits_charged, 0) DESC) AS cost_per_credit_rank,
    -- Share of all migration volume this pair represents.
    ROUND(n_runs * 100.0 / SUM(n_runs) OVER (), 2) AS pct_of_migration_volume
FROM agg
ORDER BY llm_cost_per_credit_usd DESC;
