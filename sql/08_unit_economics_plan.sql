-- 08_unit_economics_plan.sql
-- Unit economics by plan: revenue, LLM cost, gross margin, and cost-per-credit
-- vs revenue-per-credit. This is the SQL evidence for "Team plan margin is
-- the thinnest paid plan" (flat per-task credits under-price expensive
-- migrations, and Team customers run a heavier migration mix than Pro).

WITH run_cost_by_customer_month AS (
    -- Attribute each run's LLM cost to the billing month it happened in, so
    -- it can be joined onto customer_months (which does not store LLM cost).
    SELECT
        customer_id,
        substr(started_at, 1, 7) || '-01' AS month,
        SUM(llm_cost_usd) AS llm_cost_usd,
        SUM(credits_charged) AS credits_charged
    FROM agent_runs
    GROUP BY customer_id, substr(started_at, 1, 7)
),
plan_month AS (
    SELECT
        cm.plan,
        cm.customer_id,
        cm.month,
        cm.revenue_usd,
        cm.credits_used,
        COALESCE(rc.llm_cost_usd, 0.0) AS llm_cost_usd
    FROM customer_months cm
    LEFT JOIN run_cost_by_customer_month rc
        ON rc.customer_id = cm.customer_id AND rc.month = cm.month
    WHERE cm.plan <> 'Free'
),
agg AS (
    SELECT
        plan,
        COUNT(DISTINCT customer_id) AS paying_customers,
        COUNT(*) AS customer_months,
        SUM(revenue_usd) AS total_revenue_usd,
        SUM(llm_cost_usd) AS total_llm_cost_usd,
        SUM(credits_used) AS total_credits_used
    FROM plan_month
    GROUP BY plan
)
SELECT
    plan,
    paying_customers,
    customer_months,
    ROUND(total_revenue_usd, 2) AS total_revenue_usd,
    ROUND(total_llm_cost_usd, 2) AS total_llm_cost_usd,
    ROUND(total_revenue_usd - total_llm_cost_usd, 2) AS gross_profit_usd,
    ROUND((total_revenue_usd - total_llm_cost_usd) * 100.0 / NULLIF(total_revenue_usd, 0), 2) AS gross_margin_pct,
    ROUND(total_llm_cost_usd / NULLIF(total_credits_used, 0), 4) AS llm_cost_per_credit_usd,
    ROUND(total_revenue_usd / NULLIF(total_credits_used, 0), 4) AS revenue_per_credit_usd,
    -- Window function: rank plans by margin, worst first — pulls the thin-
    -- margin plan straight to the top of the report.
    RANK() OVER (ORDER BY (total_revenue_usd - total_llm_cost_usd) * 1.0 / NULLIF(total_revenue_usd, 0) ASC)
        AS margin_rank_worst_first
FROM agg
ORDER BY gross_margin_pct ASC;
