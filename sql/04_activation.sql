-- 04_activation.sql
-- Per-customer activation (success share of first 5 runs), bucketed by the
-- number of successes out of 5, joined to self-serve conversion and 6-month
-- paid retention. This is the SQL evidence behind "activation predicts
-- conversion and retention" (ARCHITECTURE.md hidden truth #4).
--
-- Activation      = successes among a customer's first 5 runs (by started_at)
-- Activated       = >= 3 of first 5 runs succeeded
-- Self-serve conv.= first_paid_date IS NOT NULL among is_sales_led = 0
-- 6-month paid ret.= customer has a paid (plan <> 'Free') customer_months row
--                    at exactly 6 calendar months after first_paid_date

WITH ranked_runs AS (
    -- Window function: ROW_NUMBER gives each customer's runs in time order so
    -- we can isolate exactly their first 5 without a LIMIT-per-group hack.
    SELECT
        customer_id,
        status,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY started_at, run_id) AS run_rank
    FROM agent_runs
),
first_five AS (
    SELECT customer_id, status
    FROM ranked_runs
    WHERE run_rank <= 5
),
activation_per_customer AS (
    SELECT
        customer_id,
        COUNT(*)                                   AS runs_considered,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
        ROUND(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 3) AS activation_score
    FROM first_five
    GROUP BY customer_id
    -- Only customers with a full first-5-runs window are scored (matches the
    -- generator's own MIN_FIRST_MONTH_RUNS=5 activation definition).
    HAVING COUNT(*) = 5
),
retention_check AS (
    SELECT
        c.customer_id,
        CASE WHEN EXISTS (
            SELECT 1 FROM customer_months cm
            WHERE cm.customer_id = c.customer_id
              AND cm.plan <> 'Free'
              AND cm.month = date(substr(c.first_paid_date, 1, 7) || '-01', '+6 months')
        ) THEN 1 ELSE 0 END AS retained_6mo
    FROM customers c
    WHERE c.first_paid_date IS NOT NULL
      -- Only score customers who have actually had the chance to reach the
      -- 6-month mark within the observed data window (else a recent signup
      -- would be counted as "not retained" purely from data truncation).
      AND date(substr(c.first_paid_date, 1, 7) || '-01', '+6 months') <= (SELECT MAX(month) FROM customer_months)
)
SELECT
    a.successes AS successes_out_of_5,
    COUNT(*) AS customers,
    SUM(CASE WHEN c.first_paid_date IS NOT NULL AND c.is_sales_led = 0 THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN c.is_sales_led = 0 THEN 1 ELSE 0 END), 0) * 100
        AS self_serve_conversion_pct,
    SUM(CASE WHEN c.is_sales_led = 0 THEN 1 ELSE 0 END) AS self_serve_customers,
    ROUND(AVG(r.retained_6mo) * 100, 2) AS retained_6mo_pct_of_paid,
    SUM(r.retained_6mo) AS retained_6mo_count,
    COUNT(r.customer_id) AS paid_customers_with_6mo_window,
    -- Window function: rank buckets by conversion so the report can call out
    -- the monotonic activation -> conversion relationship directly.
    RANK() OVER (
        ORDER BY SUM(CASE WHEN c.first_paid_date IS NOT NULL AND c.is_sales_led = 0 THEN 1 ELSE 0 END) * 1.0
            / NULLIF(SUM(CASE WHEN c.is_sales_led = 0 THEN 1 ELSE 0 END), 0) DESC
    ) AS conversion_rank
FROM activation_per_customer a
JOIN customers c ON c.customer_id = a.customer_id
LEFT JOIN retention_check r ON r.customer_id = a.customer_id
GROUP BY a.successes
ORDER BY a.successes;
