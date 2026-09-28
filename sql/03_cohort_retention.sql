-- 03_cohort_retention.sql
-- Signup-month cohort x months-since-signup PAID-logo retention triangle.
--
-- For every signup cohort (customers grouped by the calendar month they
-- signed up in), what fraction of that cohort is an active *paying*
-- customer N months after signup? This is the classic retention triangle,
-- but gated on "paying" rather than "any activity" since that is the
-- business-relevant definition of retention for a subscription product.

WITH cohorts AS (
    -- One row per customer with its signup cohort month.
    SELECT
        customer_id,
        substr(signup_date, 1, 7) || '-01' AS cohort_month
    FROM customers
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM cohorts
    GROUP BY cohort_month
),
paid_months AS (
    SELECT customer_id, month
    FROM customer_months
    WHERE plan <> 'Free'
),
retention_raw AS (
    -- months_since_signup computed via integer year*12+month arithmetic
    -- (SQLite has no native "month diff" function).
    SELECT
        c.cohort_month,
        (CAST(strftime('%Y', pm.month) AS INTEGER) * 12 + CAST(strftime('%m', pm.month) AS INTEGER))
        - (CAST(strftime('%Y', c.cohort_month) AS INTEGER) * 12 + CAST(strftime('%m', c.cohort_month) AS INTEGER))
            AS months_since_signup,
        pm.customer_id
    FROM paid_months pm
    JOIN cohorts c ON c.customer_id = pm.customer_id
),
retained_counts AS (
    SELECT cohort_month, months_since_signup, COUNT(*) AS retained_customers
    FROM retention_raw
    GROUP BY cohort_month, months_since_signup
)
SELECT
    r.cohort_month,
    cs.cohort_size,
    r.months_since_signup,
    r.retained_customers,
    ROUND(r.retained_customers * 100.0 / cs.cohort_size, 2) AS retention_pct,
    -- Window function: month-over-month change in retention within the same
    -- cohort, i.e. how much logo retention drops between consecutive
    -- months-since-signup buckets.
    ROUND(
        r.retained_customers * 100.0 / cs.cohort_size
        - LAG(r.retained_customers * 100.0 / cs.cohort_size) OVER (
            PARTITION BY r.cohort_month ORDER BY r.months_since_signup
          ),
        2
    ) AS retention_pct_change_vs_prior_month
FROM retained_counts r
JOIN cohort_sizes cs ON cs.cohort_month = r.cohort_month
ORDER BY r.cohort_month, r.months_since_signup;
