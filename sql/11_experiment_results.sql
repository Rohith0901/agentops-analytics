-- 11_experiment_results.sql
-- Price A/B test ('pro_price_2026Q2') summary by variant: assignment counts,
-- conversion within 30 days of assignment, and first-3-month revenue among
-- converters. The Python A/B test (two-proportion z-test, bootstrap CI,
-- power analysis) in src/analysis/pricing_analytics.py consumes the raw
-- per-customer rows directly from the database; this file is the
-- human-readable variant-level summary for reports/tables/.

WITH assignments AS (
    SELECT
        ea.customer_id,
        ea.variant,
        ea.pro_price_shown,
        ea.assigned_at,
        c.first_paid_date,
        CASE
            WHEN c.first_paid_date IS NOT NULL
                 AND julianday(c.first_paid_date) - julianday(ea.assigned_at) <= 30
            THEN 1 ELSE 0
        END AS converted_within_30d
    FROM experiment_assignments ea
    JOIN customers c ON c.customer_id = ea.customer_id
    WHERE ea.experiment_name = 'pro_price_2026Q2'
),
first_3mo_revenue AS (
    SELECT
        a.customer_id,
        SUM(cm.revenue_usd) AS revenue_first_3mo_usd
    FROM assignments a
    JOIN customer_months cm ON cm.customer_id = a.customer_id
    WHERE a.converted_within_30d = 1
      AND cm.plan <> 'Free'
      AND (
            CAST(strftime('%Y', cm.month) AS INTEGER) * 12 + CAST(strftime('%m', cm.month) AS INTEGER)
          ) - (
            CAST(strftime('%Y', substr(a.first_paid_date, 1, 7) || '-01') AS INTEGER) * 12
            + CAST(strftime('%m', substr(a.first_paid_date, 1, 7) || '-01') AS INTEGER)
          ) BETWEEN 0 AND 2
    GROUP BY a.customer_id
)
SELECT
    a.variant,
    a.pro_price_shown,
    COUNT(*) AS n_assigned,
    SUM(a.converted_within_30d) AS n_converted_30d,
    ROUND(SUM(a.converted_within_30d) * 100.0 / COUNT(*), 2) AS conversion_pct,
    ROUND(AVG(COALESCE(f.revenue_first_3mo_usd, 0)), 2) AS avg_revenue_per_assigned_first_3mo_usd,
    ROUND(SUM(COALESCE(f.revenue_first_3mo_usd, 0)), 2) AS total_revenue_first_3mo_usd,
    -- Window function: share of total converters each variant represents.
    ROUND(SUM(a.converted_within_30d) * 100.0 / SUM(SUM(a.converted_within_30d)) OVER (), 2)
        AS pct_of_all_converters
FROM assignments a
LEFT JOIN first_3mo_revenue f ON f.customer_id = a.customer_id
GROUP BY a.variant, a.pro_price_shown
ORDER BY a.variant;
