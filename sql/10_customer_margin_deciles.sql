-- 10_customer_margin_deciles.sql
-- Customer-level gross margin deciles (NTILE window function): bucket paying
-- customers into 10 equal-sized groups by lifetime gross margin, and profile
-- each decile's plan mix and migration-task share, to show which customer
-- segments actually make money.

WITH run_cost_by_customer AS (
    SELECT customer_id, SUM(llm_cost_usd) AS llm_cost_usd,
           SUM(CASE WHEN task_type = 'code_migration' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS migration_share
    FROM agent_runs
    GROUP BY customer_id
),
customer_econ AS (
    SELECT
        cm.customer_id,
        SUM(cm.revenue_usd) AS lifetime_revenue_usd,
        COALESCE(MAX(rc.llm_cost_usd), 0.0) AS lifetime_llm_cost_usd,
        COALESCE(MAX(rc.migration_share), 0.0) AS migration_share
    FROM customer_months cm
    LEFT JOIN run_cost_by_customer rc ON rc.customer_id = cm.customer_id
    WHERE cm.plan <> 'Free'
    GROUP BY cm.customer_id
),
customer_margin AS (
    SELECT
        customer_id,
        lifetime_revenue_usd,
        lifetime_llm_cost_usd,
        migration_share,
        (lifetime_revenue_usd - lifetime_llm_cost_usd) AS lifetime_gross_profit_usd,
        (lifetime_revenue_usd - lifetime_llm_cost_usd) * 1.0 / NULLIF(lifetime_revenue_usd, 0) AS margin_pct
    FROM customer_econ
    WHERE lifetime_revenue_usd > 0
),
deciled AS (
    -- Window function: NTILE(10) splits customers into decile buckets
    -- ordered by margin_pct, lowest-margin first (decile 1 = worst margin).
    SELECT
        customer_id,
        lifetime_revenue_usd,
        lifetime_llm_cost_usd,
        lifetime_gross_profit_usd,
        margin_pct,
        migration_share,
        NTILE(10) OVER (ORDER BY margin_pct) AS margin_decile
    FROM customer_margin
)
SELECT
    d.margin_decile,
    COUNT(*) AS customers,
    ROUND(AVG(d.margin_pct) * 100, 2) AS avg_margin_pct,
    ROUND(MIN(d.margin_pct) * 100, 2) AS min_margin_pct,
    ROUND(MAX(d.margin_pct) * 100, 2) AS max_margin_pct,
    ROUND(SUM(d.lifetime_revenue_usd), 2) AS total_revenue_usd,
    ROUND(SUM(d.lifetime_gross_profit_usd), 2) AS total_gross_profit_usd,
    ROUND(AVG(d.migration_share) * 100, 2) AS avg_migration_share_pct,
    -- Dominant plan in this decile (most common current_plan among its customers).
    (
        SELECT c.current_plan
        FROM customers c
        WHERE c.customer_id IN (SELECT customer_id FROM deciled d2 WHERE d2.margin_decile = d.margin_decile)
        GROUP BY c.current_plan
        ORDER BY COUNT(*) DESC
        LIMIT 1
    ) AS most_common_plan
FROM deciled d
GROUP BY d.margin_decile
ORDER BY d.margin_decile;
