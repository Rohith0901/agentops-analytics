-- 05_channel_performance.sql
-- Acquisition-channel scorecard: signups, self-serve conversion, paid logo
-- churn rate, and lifetime revenue per signup. Evidence for the hidden truth
-- that Referral/Community customers retain & convert best while Paid Search
-- is worst.

WITH channel_customers AS (
    SELECT
        customer_id,
        acquisition_channel,
        is_sales_led,
        first_paid_date,
        churn_date,
        status
    FROM customers
),
revenue_per_customer AS (
    SELECT customer_id, SUM(revenue_usd) AS lifetime_revenue_usd
    FROM customer_months
    WHERE plan <> 'Free'
    GROUP BY customer_id
),
channel_stats AS (
    SELECT
        cc.acquisition_channel,
        COUNT(*) AS signups,
        SUM(CASE WHEN cc.is_sales_led = 0 THEN 1 ELSE 0 END) AS self_serve_signups,
        SUM(CASE WHEN cc.is_sales_led = 0 AND cc.first_paid_date IS NOT NULL THEN 1 ELSE 0 END)
            AS self_serve_conversions,
        SUM(CASE WHEN cc.first_paid_date IS NOT NULL THEN 1 ELSE 0 END) AS paid_logos,
        SUM(CASE WHEN cc.first_paid_date IS NOT NULL AND cc.status = 'churned' THEN 1 ELSE 0 END)
            AS paid_logos_churned,
        AVG(COALESCE(rpc.lifetime_revenue_usd, 0)) AS avg_lifetime_revenue_per_signup_usd,
        SUM(COALESCE(rpc.lifetime_revenue_usd, 0)) AS total_lifetime_revenue_usd
    FROM channel_customers cc
    LEFT JOIN revenue_per_customer rpc ON rpc.customer_id = cc.customer_id
    GROUP BY cc.acquisition_channel
)
SELECT
    acquisition_channel,
    signups,
    self_serve_signups,
    self_serve_conversions,
    ROUND(self_serve_conversions * 100.0 / NULLIF(self_serve_signups, 0), 2) AS self_serve_conversion_pct,
    paid_logos,
    paid_logos_churned,
    ROUND(paid_logos_churned * 100.0 / NULLIF(paid_logos, 0), 2) AS paid_logo_churn_pct,
    ROUND(avg_lifetime_revenue_per_signup_usd, 2) AS avg_lifetime_revenue_per_signup_usd,
    ROUND(total_lifetime_revenue_usd, 2) AS total_lifetime_revenue_usd,
    -- Window functions: rank each channel on conversion and on churn so the
    -- "best/worst channel" story is explicit in the output, not left to the
    -- reader to eyeball from raw percentages.
    RANK() OVER (ORDER BY self_serve_conversions * 1.0 / NULLIF(self_serve_signups, 0) DESC) AS conversion_rank,
    RANK() OVER (ORDER BY paid_logos_churned * 1.0 / NULLIF(paid_logos, 0) ASC) AS retention_rank
FROM channel_stats
ORDER BY self_serve_conversion_pct DESC;
