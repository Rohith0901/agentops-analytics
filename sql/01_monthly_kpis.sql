-- 01_monthly_kpis.sql
-- Monthly SaaS KPI trend: MRR, revenue, active paid customers, new/churned
-- logos, ARPA, and month-over-month growth.
--
-- Definitions (shared across all builders, see ARCHITECTURE.md):
--   MRR      = SUM(monthly_fee_usd) over paid (plan <> 'Free') customer_months
--   revenue  = SUM(revenue_usd) (fee + overage) over paid customer_months
--   ARPA     = MRR / active paid customers that month
-- New logos in month M   = customers whose first_paid_date falls in month M.
-- Churned logos in month M = customers whose churn_date falls in month M,
--   restricted to customers who were ever paid (first_paid_date IS NOT NULL) —
--   a Free user churning is not a "logo" loss from a revenue standpoint.

WITH paid_months AS (
    -- One row per customer per paid billing month.
    SELECT
        customer_id,
        month,
        monthly_fee_usd,
        revenue_usd
    FROM customer_months
    WHERE plan <> 'Free'
),
monthly_agg AS (
    SELECT
        month,
        SUM(monthly_fee_usd)         AS mrr_usd,
        SUM(revenue_usd)             AS revenue_usd,
        COUNT(DISTINCT customer_id)  AS active_paid_customers
    FROM paid_months
    GROUP BY month
),
new_logos AS (
    SELECT substr(first_paid_date, 1, 7) || '-01' AS month,
           COUNT(*) AS new_logo_count
    FROM customers
    WHERE first_paid_date IS NOT NULL
    GROUP BY substr(first_paid_date, 1, 7)
),
churned_logos AS (
    SELECT substr(churn_date, 1, 7) || '-01' AS month,
           COUNT(*) AS churned_logo_count
    FROM customers
    WHERE churn_date IS NOT NULL AND first_paid_date IS NOT NULL
    GROUP BY substr(churn_date, 1, 7)
)
SELECT
    m.month,
    m.mrr_usd,
    m.revenue_usd,
    m.active_paid_customers,
    ROUND(m.mrr_usd * 1.0 / NULLIF(m.active_paid_customers, 0), 2) AS arpa_usd,
    COALESCE(nl.new_logo_count, 0)      AS new_logos,
    COALESCE(cl.churned_logo_count, 0)  AS churned_logos,
    -- Window functions: MoM growth via LAG, ordered by calendar month.
    ROUND(
        (m.mrr_usd - LAG(m.mrr_usd) OVER (ORDER BY m.month))
        * 100.0 / NULLIF(LAG(m.mrr_usd) OVER (ORDER BY m.month), 0),
        2
    ) AS mrr_mom_growth_pct,
    ROUND(
        (m.active_paid_customers - LAG(m.active_paid_customers) OVER (ORDER BY m.month))
        * 100.0 / NULLIF(LAG(m.active_paid_customers) OVER (ORDER BY m.month), 0),
        2
    ) AS active_customers_mom_growth_pct
FROM monthly_agg m
LEFT JOIN new_logos nl ON nl.month = m.month
LEFT JOIN churned_logos cl ON cl.month = m.month
ORDER BY m.month;
