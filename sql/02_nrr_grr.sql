-- 02_nrr_grr.sql
-- Net Revenue Retention (NRR) and Gross Revenue Retention (GRR), computed
-- month-over-month on the cohort of customers who were paying in the prior
-- month (a 20-month observation window is too short for a clean trailing-12m
-- cohort near the start of the series, so we use the standard SaaS
-- "month N vs month N+1" cohort definition and let analysts roll it up).
--
-- NRR = SUM(next-month revenue, 0 if churned) / SUM(this-month revenue)
--       — expansion is included, so NRR can exceed 100%.
-- GRR = SUM(MIN(next-month revenue, this-month revenue)) / SUM(this-month revenue)
--       — expansion is capped out, so GRR maxes at 100% (contraction + churn only).

WITH paid_months AS (
    SELECT customer_id, month, revenue_usd
    FROM customer_months
    WHERE plan <> 'Free'
),
with_next AS (
    -- Window function: LEAD looks at each customer's *next row* in paid_months.
    -- Because a customer never returns to Free once paid (see generate_data.py
    -- lifecycle), consecutive paid_months rows for the same customer are always
    -- consecutive calendar months, so LEAD is a safe stand-in for an explicit
    -- "next calendar month" join.
    SELECT
        customer_id,
        month AS base_month,
        revenue_usd AS base_revenue,
        LEAD(month) OVER (PARTITION BY customer_id ORDER BY month)        AS next_month,
        LEAD(revenue_usd) OVER (PARTITION BY customer_id ORDER BY month)  AS next_revenue
    FROM paid_months
),
resolved AS (
    SELECT
        customer_id,
        base_month,
        base_revenue,
        -- Target month is always base_month + 1 calendar month; if there is no
        -- contiguous next row, the customer churned (or the window ended) and
        -- contributes 0 retained revenue to that target month.
        date(base_month, '+1 month') AS target_month,
        CASE
            WHEN next_month = date(base_month, '+1 month') THEN next_revenue
            ELSE 0.0
        END AS retained_revenue
    FROM with_next
)
SELECT
    target_month AS month,
    COUNT(*)                                    AS cohort_customers,
    ROUND(SUM(base_revenue), 2)                 AS base_revenue_usd,
    ROUND(SUM(retained_revenue), 2)             AS retained_revenue_usd,
    ROUND(SUM(MIN(retained_revenue, base_revenue)), 2) AS grr_capped_revenue_usd,
    ROUND(SUM(retained_revenue) * 100.0 / NULLIF(SUM(base_revenue), 0), 2)             AS nrr_pct,
    ROUND(SUM(MIN(retained_revenue, base_revenue)) * 100.0 / NULLIF(SUM(base_revenue), 0), 2) AS grr_pct
FROM resolved
-- Exclude the last observed base_month: its "target_month" falls outside the
-- data window, so we cannot yet tell churn from data-truncation.
WHERE target_month <= (SELECT MAX(month) FROM customer_months)
GROUP BY target_month
ORDER BY target_month;
