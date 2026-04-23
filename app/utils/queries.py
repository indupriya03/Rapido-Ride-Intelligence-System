# =============================================================================
# QUERIES.PY — All SQL queries for Rapido Ride Intelligence System
# =============================================================================
# Usage:
#   from utils.queries import Q1_KPI_SUMMARY, Q4_DRIVER_DEMAND, ...
#
# Every query is consumed by run_query() which returns a pandas DataFrame.
# All heavy joins and aggregations happen in SQL — not in Python.
#
# Organised by page:
#   Q0  → Shared / reusable helpers
#   Q1  → Overview page
#   Q2  → Predictions page
#   Q3  → Analytics page
#   Q4  → Strategy page   ← fully updated with driver allocation strategy
# =============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ─────────────────────────────────────────────────────────────────────────────
# Q0  SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

Q0_CITIES = "SELECT DISTINCT city FROM bookings ORDER BY city"

Q0_VEHICLE_TYPES = "SELECT DISTINCT vehicle_type FROM bookings ORDER BY vehicle_type"

Q0_DATE_RANGE = """
    SELECT
        MIN(booking_datetime) AS min_date,
        MAX(booking_datetime) AS max_date
    FROM bookings
"""


# ─────────────────────────────────────────────────────────────────────────────
# Q1  OVERVIEW PAGE
# ─────────────────────────────────────────────────────────────────────────────

# KPI cards — single-row summary
Q1_KPI_SUMMARY = """
    SELECT
        COUNT(*)                                                       AS total_rides,
        ROUND(AVG(booking_value), 2)                                   AS avg_fare,
        ROUND(
            100.0 * SUM(booking_status = 'Cancelled') / COUNT(*), 2
        )                                                              AS cancel_rate_pct,
        ROUND(
            AVG(CASE WHEN actual_ride_time_min > 0
                     THEN actual_ride_time_min - estimated_ride_time_min
                     ELSE NULL END), 2
        )                                                              AS avg_delay_min,
        ROUND(AVG(surge_multiplier), 3)                                AS avg_surge,
        SUM(booking_status = 'Completed')                              AS completed_rides,
        SUM(booking_status = 'Cancelled')                              AS cancelled_rides,
        SUM(booking_status NOT IN ('Completed','Cancelled'))           AS incomplete_rides
    FROM bookings
"""

# City heatmap — rides + cancellation rate by city
Q1_CITY_HEATMAP = """
    SELECT
        city,
        COUNT(*)                                                       AS total_rides,
        SUM(booking_status = 'Completed')                              AS completed,
        SUM(booking_status = 'Cancelled')                              AS cancelled,
        ROUND(
            100.0 * SUM(booking_status = 'Cancelled') / COUNT(*), 2
        )                                                              AS cancel_rate_pct,
        ROUND(AVG(booking_value), 2)                                   AS avg_fare,
        ROUND(AVG(surge_multiplier), 3)                                AS avg_surge
    FROM bookings
    GROUP BY city
    ORDER BY total_rides DESC
"""

# Pickup demand by city (top pickup locations)
Q1_PICKUP_DEMAND = """
    SELECT
        city,
        pickup_location,
        COUNT(*)                                                       AS ride_count,
        ROUND(
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY city), 1
        )                                                              AS pct_of_city
    FROM bookings
    GROUP BY city, pickup_location
    ORDER BY city, ride_count DESC
"""

# Hourly trends — rides + cancellations by hour
Q1_HOURLY_TRENDS = """
    SELECT
        hour_of_day,
        COUNT(*)                                                       AS total_rides,
        SUM(booking_status = 'Completed')                              AS completed,
        SUM(booking_status = 'Cancelled')                              AS cancelled,
        ROUND(
            100.0 * SUM(booking_status = 'Cancelled') / COUNT(*), 2
        )                                                              AS cancel_rate_pct,
        ROUND(AVG(booking_value), 2)                                   AS avg_fare,
        ROUND(AVG(surge_multiplier), 3)                                AS avg_surge
    FROM bookings
    GROUP BY hour_of_day
    ORDER BY hour_of_day
"""

# Vehicle-type breakdown
Q1_VEHICLE_BREAKDOWN = """
    SELECT
        vehicle_type,
        COUNT(*)                                                       AS total_rides,
        ROUND(AVG(booking_value), 2)                                   AS avg_fare,
        ROUND(AVG(ride_distance_km), 2)                                AS avg_dist_km,
        ROUND(
            100.0 * SUM(booking_status = 'Cancelled') / COUNT(*), 2
        )                                                              AS cancel_rate_pct
    FROM bookings
    GROUP BY vehicle_type
    ORDER BY total_rides DESC
"""

# Status distribution
Q1_STATUS_DIST = """
    SELECT
        booking_status,
        COUNT(*) AS count
    FROM bookings
    GROUP BY booking_status
"""


# ─────────────────────────────────────────────────────────────────────────────
# Q2  PREDICTIONS PAGE
# ─────────────────────────────────────────────────────────────────────────────

# Predictions summary KPIs
Q2_PREDICTION_KPIS = """
    SELECT
        COUNT(*)                                                               AS total_predictions,
        ROUND(AVG(predicted_fare), 2)                                          AS avg_predicted_fare,
        ROUND(AVG(actual_fare), 2)                                             AS avg_actual_fare,
        ROUND(AVG(CASE WHEN actual_completed_flag = 1
               THEN ABS(predicted_fare - actual_fare) END), 2)                 AS avg_fare_mae,
        ROUND(
            100.0 * SUM(cancel_risk_tier = 'High') / COUNT(*), 2
        )                                                                      AS high_risk_pct,
        SUM(cancel_risk_tier = 'High')                                         AS high_risk_count,
        SUM(cancel_risk_tier = 'Medium')                                       AS medium_risk_count,
        SUM(cancel_risk_tier = 'Low')                                          AS low_risk_count
    FROM predictions
"""

# Risk tier distribution
Q2_RISK_TIER_DIST = """
    SELECT
        cancel_risk_tier,
        COUNT(*)                                                               AS count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)                    AS pct
    FROM predictions
    GROUP BY cancel_risk_tier
    ORDER BY FIELD(cancel_risk_tier, 'High', 'Medium', 'Low')
"""

# Recommended action distribution
Q2_ACTIONS_DIST = """
    SELECT
        recommended_action,
        cancel_risk_tier,
        COUNT(*)                                                               AS count
    FROM predictions
    GROUP BY recommended_action, cancel_risk_tier
    ORDER BY count DESC
"""

# Fare prediction accuracy: predicted vs actual by city
Q2_FARE_ACCURACY_BY_CITY = """
    SELECT
        city,
        ROUND(AVG(predicted_fare), 2)                                          AS avg_predicted,
        ROUND(AVG(actual_fare), 2)                                             AS avg_actual,
        ROUND(AVG(ABS(predicted_fare - actual_fare)), 2)                       AS mae,
        ROUND(
            100.0 * AVG(ABS(predicted_fare - actual_fare)) / AVG(actual_fare), 1
        )                                                                      AS mape_pct,
        COUNT(*)                                                               AS n
    FROM predictions
    WHERE actual_fare > 0
      AND actual_completed_flag = 1
    GROUP BY city
    ORDER BY mae DESC
"""

# Cancel risk distribution by city
Q2_RISK_BY_CITY = """
    SELECT
        city,
        cancel_risk_tier,
        COUNT(*)                                                               AS count
    FROM predictions
    GROUP BY city, cancel_risk_tier
    ORDER BY city, FIELD(cancel_risk_tier, 'High', 'Medium', 'Low')
"""

# Cancel probability distribution (histogram buckets)
Q2_CANCEL_PROBA_DIST = """
    SELECT
        FLOOR(cancel_probability * 10) / 10                                    AS prob_bucket,
        COUNT(*)                                                               AS count
    FROM predictions
    GROUP BY prob_bucket
    ORDER BY prob_bucket
"""

# High-risk bookings sample table
Q2_HIGH_RISK_SAMPLE = """
    SELECT
        p.booking_id,
        p.city,
        p.vehicle_type,
        p.hour_of_day,
        ROUND(p.ride_distance_km, 1)                                           AS distance_km,
        ROUND(p.surge_multiplier, 2)                                           AS surge,
        ROUND(p.cancel_probability * 100, 1)                                   AS cancel_prob_pct,
        p.cancel_risk_tier,
        p.recommended_action,
        ROUND(p.predicted_fare, 2)                                             AS predicted_fare,
        ROUND(p.actual_fare, 2)                                                AS actual_fare
    FROM predictions p
    WHERE p.cancel_risk_tier = 'High'
    ORDER BY p.cancel_probability DESC
    LIMIT 200
"""

# Model performance: confusion matrix data
Q2_MODEL_ACCURACY = """
    SELECT
        actual_cancelled_flag,
        CASE WHEN cancel_probability >= uc3_threshold_used THEN 1 ELSE 0 END   AS predicted_cancelled,
        COUNT(*)                                                               AS count
    FROM predictions
    GROUP BY actual_cancelled_flag, predicted_cancelled
"""

# Fare scatter data (predicted vs actual) — sample for plot
Q2_FARE_SCATTER = """
    SELECT
        ROUND(actual_fare, 2)                                                  AS actual_fare,
        ROUND(predicted_fare, 2)                                               AS predicted_fare,
        city,
        vehicle_type,
        cancel_risk_tier
    FROM predictions
    WHERE actual_fare > 0
        AND actual_fare < 2000
        AND actual_completed_flag = 1
    ORDER BY RAND()
    LIMIT 2000
"""


# Live inference — averages over all matching bookings in the predictions table
# instead of returning one random booking (which gave unstable results).
# Derives cancel_risk_tier from the averaged probability so the tier is
# statistically meaningful rather than reflecting a single booking's outcome.
Q2_LIVE_INFERENCE = """
    SELECT
        p.city,
        p.vehicle_type,
        p.hour_of_day,
        p.traffic_level,
        p.weather_condition,
        ROUND(AVG(p.ride_distance_km), 1)                                      AS ride_distance_km,
        ROUND(AVG(p.surge_multiplier), 2)                                      AS surge_multiplier,
        ROUND(AVG(p.predicted_fare), 2)                                        AS predicted_fare,
        ROUND(AVG(p.cancel_probability) * 100, 1)                              AS cancel_probability_pct,
        ROUND(AVG(p.cancel_probability), 4)                                    AS cancel_probability_raw,
        CASE
            WHEN AVG(p.cancel_probability) >= 0.70 THEN 'High'
            WHEN AVG(p.cancel_probability) >= 0.40 THEN 'Medium'
            ELSE 'Low'
        END                                                                    AS cancel_risk_tier,
        CASE
            WHEN AVG(p.cancel_probability) >= 0.70 THEN 'Reassign Driver'
            WHEN AVG(p.cancel_probability) >= 0.40 THEN 'Send Reminder'
            ELSE 'Proceed'
        END                                                                    AS recommended_action,
        MAX(p.uc3_threshold_used)                                              AS uc3_threshold_used,
        COUNT(*)                                                               AS matched_bookings
    FROM predictions p
    WHERE p.city              = :city
      AND p.vehicle_type      = :vehicle_type
      AND p.hour_of_day       = :hour_of_day
      AND p.traffic_level     = :traffic_level
      AND p.weather_condition = :weather_condition
      AND ABS(p.ride_distance_km - :distance_km) <= 5.0
    GROUP BY p.city, p.vehicle_type, p.hour_of_day,
             p.traffic_level, p.weather_condition
    LIMIT 1
"""


# ─────────────────────────────────────────────────────────────────────────────
# Q3  ANALYTICS PAGE
# ─────────────────────────────────────────────────────────────────────────────

# Cancellation heatmap: hour × day_of_week
Q3_CANCEL_HEATMAP = """
    SELECT
        b.hour_of_day,
        b.day_of_week,
        COUNT(*)                                                               AS total_rides,
        SUM(b.booking_status = 'Cancelled')                                    AS cancellations,
        ROUND(
            100.0 * SUM(b.booking_status = 'Cancelled') / COUNT(*), 2
        )                                                                      AS cancel_rate_pct
    FROM bookings b
    GROUP BY b.hour_of_day, b.day_of_week
    ORDER BY b.hour_of_day, b.day_of_week
"""

# Surge patterns: surge multiplier by hour and city
Q3_SURGE_BY_HOUR_CITY = """
    SELECT
        hour_of_day,
        city,
        ROUND(AVG(surge_multiplier), 3)                                        AS avg_surge,
        ROUND(MAX(surge_multiplier), 2)                                        AS max_surge,
        COUNT(*)                                                               AS ride_count
    FROM bookings
    GROUP BY hour_of_day, city
    ORDER BY hour_of_day, city
"""

# Cancel probability vs surge multiplier buckets
Q3_CANCEL_PROB_BY_SURGE = """
    SELECT
        ROUND(surge_multiplier, 1)                                             AS surge_bucket,
        ROUND(AVG(cancel_probability), 4)                                      AS avg_cancel_prob,
        COUNT(*)                                                               AS count
    FROM predictions
    GROUP BY surge_bucket
    ORDER BY surge_bucket
"""

# Model-predicted cancel risk vs actual cancels by hour
Q3_CANCEL_PROB_BY_HOUR = """
    SELECT
        hour_of_day,
        ROUND(AVG(cancel_probability), 4)                                      AS avg_cancel_prob,
        SUM(actual_cancelled_flag)                                             AS actual_cancels,
        COUNT(*)                                                               AS total
    FROM predictions
    GROUP BY hour_of_day
    ORDER BY hour_of_day
"""

# Cancel reasons (incomplete_ride_reason)
Q3_CANCEL_REASONS = """
    SELECT
        COALESCE(incomplete_ride_reason, 'Not Specified')                      AS reason,
        booking_status,
        COUNT(*)                                                               AS count
    FROM bookings
    WHERE booking_status IN ('Cancelled', 'Incomplete')
    GROUP BY reason, booking_status
    ORDER BY count DESC
    LIMIT 30
"""

# Customer vs driver cancellations (inferred from reason)
Q3_CANCEL_BY_PARTY = """
    SELECT
        CASE
            WHEN incomplete_ride_reason LIKE '%customer%'
              OR incomplete_ride_reason LIKE '%Customer%'   THEN 'Customer No Show'
            WHEN incomplete_ride_reason LIKE '%driver%'
              OR incomplete_ride_reason LIKE '%Driver%'     THEN 'Driver'
            WHEN booking_status = 'Cancelled'               THEN 'Cancelled By Customer (Assumed)'
            ELSE 'System / Unknown'
        END                                                                    AS cancelled_by,
        COUNT(*)                                                               AS count
    FROM bookings
    WHERE booking_status IN ('Cancelled', 'Incomplete')
    GROUP BY cancelled_by
    ORDER BY count DESC
"""

# Distance vs cancel probability
Q3_DIST_VS_CANCEL = """
    SELECT
        FLOOR(ride_distance_km / 5) * 5                                        AS dist_bucket_km,
        ROUND(AVG(cancel_probability), 4)                                      AS avg_cancel_prob,
        COUNT(*)                                                               AS count
    FROM predictions
    WHERE ride_distance_km < 60
    GROUP BY dist_bucket_km
    ORDER BY dist_bucket_km
"""

# Weekend vs weekday comparison
Q3_WEEKEND_COMPARISON = """
    SELECT
        is_weekend,
        COUNT(*)                                                               AS total_rides,
        ROUND(AVG(booking_value), 2)                                           AS avg_fare,
        ROUND(AVG(surge_multiplier), 3)                                        AS avg_surge,
        ROUND(
            100.0 * SUM(booking_status = 'Cancelled') / COUNT(*), 2
        )                                                                      AS cancel_rate_pct
    FROM bookings
    GROUP BY is_weekend
"""

# Weather impact
Q3_WEATHER_IMPACT = """
    SELECT
        weather_condition,
        COUNT(*)                                                               AS total_rides,
        ROUND(AVG(surge_multiplier), 3)                                        AS avg_surge,
        ROUND(
            100.0 * SUM(booking_status = 'Cancelled') / COUNT(*), 2
        )                                                                      AS cancel_rate_pct,
        ROUND(AVG(booking_value), 2)                                           AS avg_fare
    FROM bookings
    GROUP BY weather_condition
    ORDER BY total_rides DESC
"""

# Traffic impact
Q3_TRAFFIC_IMPACT = """
    SELECT
        traffic_level,
        COUNT(*)                                                               AS total_rides,
        ROUND(AVG(actual_ride_time_min - estimated_ride_time_min), 2)          AS avg_delay_min,
        ROUND(AVG(surge_multiplier), 3)                                        AS avg_surge,
        ROUND(
            100.0 * SUM(booking_status = 'Cancelled') / COUNT(*), 2
        )                                                                      AS cancel_rate_pct
    FROM bookings
    WHERE actual_ride_time_min > 0
    GROUP BY traffic_level
    ORDER BY avg_delay_min DESC
"""


# ─────────────────────────────────────────────────────────────────────────────
# Q4  STRATEGY PAGE
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# DRIVER ALLOCATION — TAB 2
# ---------------------------------------------------------------------------

# 1. Rides-per-driver heatmap (city × hour) — NULLIF guard prevents divide-by-zero
#    allocation_status classifies each slot for ops action
Q4_DRIVER_DEMAND = """
SELECT
    b.city,
    b.hour_of_day,

    COUNT(DISTINCT b.driver_id)                                AS active_drivers,
    COUNT(*)                                                   AS total_rides,

    ROUND(
        COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0), 2
    )                                                          AS rides_per_driver,

    ROUND(AVG(p.cancel_probability), 3)                        AS avg_cancel_risk,

    -- 🔥 Demand pressure score (NEW)
    ROUND(
        (COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0)) 
        * (1 + AVG(p.cancel_probability)),
        2
    )                                                          AS demand_pressure,

    -- ⏰ Time band (NEW)
    CASE
        WHEN b.hour_of_day BETWEEN 7 AND 10 THEN 'Morning Peak'
        WHEN b.hour_of_day BETWEEN 18 AND 23 THEN 'Evening Peak'
        ELSE 'Off-Peak'
    END                                                        AS time_band,

    -- 🎯 Refined allocation logic (UPDATED)
    CASE
        WHEN COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0) >= 1.15 THEN 'High Demand'
        WHEN COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0) >= 1.08 THEN 'Peak Lean'
        WHEN COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0) >= 0.95 THEN 'Balanced'
        ELSE 'Low Demand'
    END                                                        AS allocation_status,

    -- 🚀 Action layer (NEW)
    CASE
        WHEN COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0) >= 1.15
             AND AVG(p.cancel_probability) > 0.25
            THEN 'Add Drivers Immediately'

        WHEN COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0) >= 1.10
            THEN 'Pre-position Drivers'

        WHEN COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0) < 0.95
            THEN 'Shift Drivers Out'

        ELSE 'Maintain'
    END                                                        AS recommended_action,


    CONCAT(
        LPAD(GREATEST(b.hour_of_day - 1, 0), 2, '0'),
        ':',
        '30'
    ) AS pre_position_by
FROM bookings b
JOIN predictions p ON b.booking_id = p.booking_id

GROUP BY b.city, b.hour_of_day
HAVING rides_per_driver IS NOT NULL
ORDER BY demand_pressure DESC, b.city, b.hour_of_day
"""

# 2. Cancel risk heatmap — avg cancel probability by city × vehicle type
#    Tells fleet managers which segment to prioritise for reassignment
Q4_RISK_BY_CITY_VEHICLE = """
    SELECT
        city,
        vehicle_type,
        cancel_risk_tier,
        COUNT(*)                                                               AS ride_count,
        ROUND(AVG(cancel_probability) * 100, 2)                               AS avg_cancel_prob_pct,
        ROUND(AVG(predicted_fare), 2)                                          AS avg_predicted_fare,
        ROUND(AVG(surge_multiplier), 3)                                        AS avg_surge
    FROM predictions
    GROUP BY city, vehicle_type, cancel_risk_tier
    ORDER BY city, vehicle_type, FIELD(cancel_risk_tier, 'High', 'Medium', 'Low')
"""

#3. Reassignment candidates — one best replacement driver per high-risk booking
#   Uses ROW_NUMBER to pick only the top-scoring driver per booking_id
Q4_REASSIGNMENT_CANDIDATES = """
    WITH best_driver_per_city AS (
        SELECT
            driver_id,
            driver_city,
            ROUND(avg_driver_rating, 2)  AS driver_rating,
            ROUND(acceptance_rate, 2)    AS acceptance_rate,
            ROUND(
                (avg_driver_rating * 0.4)
              + (acceptance_rate   * 0.3)
              + ((1 - delay_rate)  * 0.3), 3
            )                            AS replacement_score
        FROM drivers
        WHERE ROUND(
                (avg_driver_rating * 0.4)
              + (acceptance_rate   * 0.3)
              + ((1 - delay_rate)  * 0.3), 3
              ) >= 0.8
        ORDER BY replacement_score DESC
        LIMIT 5
    )
    SELECT
        p.booking_id,
        p.city,
        p.vehicle_type,
        p.hour_of_day,
        ROUND(p.cancel_probability * 100, 1)  AS cancel_prob_pct,
        p.recommended_action,
        ROUND(p.predicted_fare, 2)            AS predicted_fare,
        d.driver_id                           AS best_replacement_driver,
        d.driver_city,
        d.driver_rating,
        d.acceptance_rate,
        d.replacement_score
    FROM (
        SELECT booking_id, city, vehicle_type, hour_of_day,
               cancel_probability, recommended_action, predicted_fare
        FROM predictions
        WHERE cancel_risk_tier   = 'High'
          AND recommended_action = 'Reassign Driver'
        ORDER BY cancel_probability DESC
        LIMIT 200
    ) p
    JOIN best_driver_per_city d
      ON d.driver_city = p.city
    ORDER BY p.cancel_probability DESC
"""

# 4. Driver efficiency — completion rate vs total fare earned per driver
#    Used for efficiency scatter plot and tier classification
Q4_DRIVER_EFFICIENCY = """
    SELECT
        b.driver_id,
        b.city,
        b.vehicle_type,
        COUNT(*)                                                               AS total_rides,
        SUM(b.booking_status = 'Completed')                                    AS completed_rides,
        ROUND(
            100.0 * SUM(b.booking_status = 'Completed') / COUNT(*), 1
        )                                                                      AS completion_rate_pct,
        ROUND(
            SUM(CASE WHEN b.booking_status = 'Completed'
                     THEN b.booking_value ELSE 0 END), 2
        )                                                                      AS total_earned,
        ROUND(AVG(b.surge_multiplier), 2)                                      AS avg_surge_worked,
        CASE
            WHEN 100.0 * SUM(b.booking_status = 'Completed') / COUNT(*) >= 90 THEN 'Elite'
            WHEN 100.0 * SUM(b.booking_status = 'Completed') / COUNT(*) >= 75 THEN 'Reliable'
            WHEN 100.0 * SUM(b.booking_status = 'Completed') / COUNT(*) >= 60 THEN 'Developing'
            ELSE 'At Risk'
        END                                                                    AS driver_tier
    FROM bookings b
    GROUP BY b.driver_id, b.city, b.vehicle_type
    HAVING total_rides >= 5
    ORDER BY completion_rate_pct DESC, total_earned DESC
    LIMIT 300
"""

# 5. Driver tier summary — counts per tier per city for KPI cards
Q4_DRIVER_TIER_SUMMARY = """
    SELECT
        city,
        CASE
            WHEN 100.0 * SUM(booking_status = 'Completed') / COUNT(*) >= 90 THEN 'Elite'
            WHEN 100.0 * SUM(booking_status = 'Completed') / COUNT(*) >= 75 THEN 'Reliable'
            WHEN 100.0 * SUM(booking_status = 'Completed') / COUNT(*) >= 60 THEN 'Developing'
            ELSE 'At Risk'
        END                                                                    AS driver_tier,
        COUNT(DISTINCT driver_id)                                              AS driver_count
    FROM bookings
    GROUP BY city, driver_id
    HAVING COUNT(*) >= 5
    ORDER BY city, driver_tier
"""

# 6. Understaffed slots — city + hour slots needing urgent driver deployment
Q4_UNDERSTAFFED_SLOTS = """
    SELECT
        b.city,
        b.hour_of_day,
        COUNT(DISTINCT b.driver_id)                                            AS active_drivers,
        COUNT(*)                                                               AS total_rides,
        ROUND(COUNT(*) / NULLIF(COUNT(DISTINCT b.driver_id), 0), 1)           AS rides_per_driver,
        ROUND(AVG(p.cancel_probability) * 100, 1)                             AS avg_cancel_prob_pct,
        SUM(p.cancel_risk_tier = 'High')                                       AS high_risk_rides,
        CONCAT(
            LPAD(GREATEST(b.hour_of_day - 1, 0), 2, '0'), CHAR(58), '30'
        )                                                                      AS pre_position_by
    FROM bookings b
    JOIN predictions p ON b.booking_id = p.booking_id
    GROUP BY b.city, b.hour_of_day
    HAVING rides_per_driver > 1.15
    ORDER BY rides_per_driver DESC, high_risk_rides DESC
    LIMIT 30
"""

# ---------------------------------------------------------------------------
# OPS ALERTS — TAB 1
# ---------------------------------------------------------------------------

# Rule-based operational alerts from predictions
Q4_OPS_ALERTS = """
    SELECT
        p.city,
        p.hour_of_day,
        p.vehicle_type,
        COUNT(*)                                                               AS ride_count,
        ROUND(AVG(p.surge_multiplier), 2)                                      AS avg_surge,
        SUM(p.cancel_risk_tier = 'High')                                       AS high_risk_count,
        ROUND(
            100.0 * SUM(p.cancel_risk_tier = 'High') / COUNT(*), 1
        )                                                                      AS high_risk_pct,
        ROUND(AVG(p.cancel_probability), 3)                                    AS avg_cancel_prob,
        CONCAT(
            LPAD(GREATEST(p.hour_of_day - 1, 0), 2, '0'), CHAR(58), '30'
        )                                                                      AS pre_position_by,
        CASE
            WHEN AVG(p.surge_multiplier) > 1.8
             AND SUM(p.cancel_risk_tier = 'High') > 5  THEN 'CRITICAL — Surge + High Cancel Risk'
            WHEN AVG(p.surge_multiplier) > 1.8         THEN 'WARNING — High Surge Demand'
            WHEN SUM(p.cancel_risk_tier = 'High') > 10 THEN 'WARNING — High Cancellation Risk'
            WHEN AVG(p.surge_multiplier) > 1.4         THEN 'INFO — Moderate Surge'
            ELSE 'OK'
        END                                                                    AS alert_level
    FROM predictions p
    GROUP BY p.city, p.hour_of_day, p.vehicle_type
    HAVING alert_level != 'OK'
    ORDER BY FIELD(alert_level,
        'CRITICAL — Surge + High Cancel Risk',
        'WARNING — High Surge Demand',
        'WARNING — High Cancellation Risk',
        'INFO — Moderate Surge'
    ), high_risk_count DESC
"""

# Peak hours needing surge intervention
Q4_SURGE_STRATEGY = """
    SELECT
        hour_of_day,
        city,
        ROUND(AVG(surge_multiplier), 3)                                        AS avg_surge,
        COUNT(*)                                                               AS total_rides,
        SUM(booking_status = 'Cancelled')                                      AS cancellations,
        ROUND(
            100.0 * SUM(booking_status = 'Cancelled') / COUNT(*), 2
        )                                                                      AS cancel_rate_pct,
        CASE
            WHEN AVG(surge_multiplier) > 1.8 THEN 'Reduce Surge to Retain Riders'
            WHEN AVG(surge_multiplier) > 1.4 THEN 'Monitor — Surge Moderate'
            ELSE 'Normal Operations'
        END                                                                    AS recommendation
    FROM bookings
    GROUP BY hour_of_day, city
    ORDER BY avg_surge DESC
    LIMIT 50
"""

# ---------------------------------------------------------------------------
# DRIVER PERFORMANCE — TAB 2 (existing, kept + enhanced)
# ---------------------------------------------------------------------------

# Top driver performance scores from drivers table
Q4_DRIVER_PERFORMANCE = """
    SELECT
        d.driver_id,
        d.driver_city                                                          AS city,
        d.driver_experience_years,
        d.avg_driver_rating,
        d.acceptance_rate,
        d.delay_rate,
        d.total_assigned_rides,
        d.accepted_rides,
        ROUND(
            (d.avg_driver_rating * 0.4)
          + (d.acceptance_rate   * 0.3)
          + ((1 - d.delay_rate)  * 0.3), 3
        )                                                                      AS performance_score,
        CASE
            WHEN ROUND(
                    (d.avg_driver_rating * 0.4)
                  + (d.acceptance_rate   * 0.3)
                  + ((1 - d.delay_rate)  * 0.3), 3
                 ) >= 0.8 THEN 'Elite'
            WHEN ROUND(
                    (d.avg_driver_rating * 0.4)
                  + (d.acceptance_rate   * 0.3)
                  + ((1 - d.delay_rate)  * 0.3), 3
                 ) >= 0.6 THEN 'Reliable'
            WHEN ROUND(
                    (d.avg_driver_rating * 0.4)
                  + (d.acceptance_rate   * 0.3)
                  + ((1 - d.delay_rate)  * 0.3), 3
                 ) >= 0.4 THEN 'Developing'
            ELSE 'At Risk'
        END                                                                    AS driver_tier
    FROM drivers d
    ORDER BY performance_score DESC
    LIMIT 100
"""

# ---------------------------------------------------------------------------
# REVENUE — TAB 3
# ---------------------------------------------------------------------------

# Revenue impact of cancellations by city
Q4_REVENUE_IMPACT = """
    SELECT
        b.city,
        COUNT(CASE WHEN b.booking_status = 'Cancelled' THEN 1 END)            AS cancelled_rides,
        ROUND(AVG(b.booking_value), 2)                                         AS avg_booking_value,
        ROUND(
            COUNT(CASE WHEN b.booking_status = 'Cancelled' THEN 1 END)
          * AVG(b.booking_value), 2
        )                                                                      AS estimated_lost_revenue,
        SUM(CASE WHEN b.booking_status = 'Completed' THEN b.booking_value ELSE 0 END)
                                                                               AS actual_revenue
    FROM bookings b
    GROUP BY b.city
    ORDER BY estimated_lost_revenue DESC
"""

# ---------------------------------------------------------------------------
# CUSTOMERS — TAB 4
# ---------------------------------------------------------------------------

# Customer LTV segmentation (based on total bookings + cancellation rate)
Q4_CUSTOMER_SEGMENTS = """
    SELECT
        CASE
            WHEN total_bookings >= 20 AND cancellation_rate < 0.2 THEN 'Champion'
            WHEN total_bookings >= 10 AND cancellation_rate < 0.3 THEN 'Loyal'
            WHEN total_bookings >= 5  AND cancellation_rate < 0.4 THEN 'Potential'
            WHEN cancellation_rate >= 0.5                          THEN 'At Risk'
            ELSE 'New / Occasional'
        END                                                                    AS segment,
        COUNT(*)                                                               AS customer_count,
        ROUND(AVG(total_bookings), 1)                                          AS avg_bookings,
        ROUND(AVG(cancellation_rate), 3)                                       AS avg_cancel_rate,
        ROUND(AVG(avg_customer_rating), 2)                                     AS avg_rating
    FROM customers
    GROUP BY segment
    ORDER BY customer_count DESC
"""