# =============================================================================
# FEATURE ENGINEERING PIPELINE — LEAKAGE-FREE VERSION
# =============================================================================
# UC1 — Ride Outcome Prediction     (Multi-Class Classification)
# UC2 — Fare Prediction              (Regression)
# UC3 — Customer Cancellation Risk   (Binary Classification)
# UC4 — Driver Delay Prediction      (Binary Classification)
# =============================================================================
#
# ZONE STRUCTURE:
#   ZONE 1  → Safe feature engineering on full df (no aggregations)
#   ZONE 2  → Derive targets, define X/y, train_test_split
#   ZONE 3  → Leaky operations fit on X_train only, applied to both
#
# =============================================================================

import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
os.makedirs("models", exist_ok=True)


#project_root = os.path.abspath(os.getcwd())  # one level up
#print(f"Project root set to: {project_root}")
# Add src folder to path
sys.path.append("src")
print("Current working directory:", os.getcwd())
from data_loader import load_cleaned_data
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 0 — LOAD DATA
# =============================================================================

bookings_df, customers_df, drivers_df, location_demand_df, time_features_df = load_cleaned_data()



# =============================================================================
# SECTION 1 — MERGE ALL 5 TABLES
# =============================================================================

def merge_all_tables(bookings_df, customers_df, drivers_df,
                     location_demand_df, time_features_df):

    print("Original shapes:")
    print(f"  bookings_df       : {bookings_df.shape}")
    print(f"  customers_df      : {customers_df.shape}")
    print(f"  drivers_df        : {drivers_df.shape}")
    print(f"  location_demand_df: {location_demand_df.shape}")
    print(f"  time_features_df  : {time_features_df.shape}")

    df = bookings_df.copy()

    # ----------------------------------------------------------------
    # 1. customers_df — drop anything already in bookings
    # ----------------------------------------------------------------
    cust_bring = [
        'customer_id',                  # join key
        'customer_gender',
        'customer_age',
        'customer_city',                # rename below — different from pickup city
        'customer_signup_days_ago',
        'preferred_vehicle_type',
        'total_bookings',
        'completed_rides',
        'cancelled_rides',
        'incomplete_rides',
        'cancellation_rate',
        'avg_customer_rating',
        'customer_cancel_flag',
    ]
    cust_bring = [c for c in cust_bring if c in customers_df.columns]
    df = df.merge(customers_df[cust_bring], on='customer_id', how='left')
    print(f"After customers : {df.shape}")

    # ----------------------------------------------------------------
    # 2. drivers_df
    #    vehicle_type     → already in bookings (per-ride), DROP from drivers
    #    incomplete_rides → exists in bookings too, RENAME drivers version
    #    city             → rename to driver_city to avoid clash with booking city
    # ----------------------------------------------------------------
    drivers_clean = drivers_df.rename(columns={
        'incomplete_rides' : 'driver_incomplete_rides',
        'city'             : 'driver_city',
    }).drop(columns=[
        'vehicle_type',     # synthetic data totally different from bookings — Keeping only in bookings
    ], errors='ignore')

    driv_bring = [
        'driver_id',                    # join key
        'driver_age',
        'driver_city',
        'driver_experience_years',
        'total_assigned_rides',
        'accepted_rides',
        'driver_incomplete_rides',
        'delay_count',
        'acceptance_rate',
        'delay_rate',
        'avg_driver_rating',
        'avg_pickup_delay_min',
        'driver_delay_flag',
        'experience_outlier_flag',
        'rejected_rides',
    ]
    driv_bring = [c for c in driv_bring if c in drivers_clean.columns]
    df = df.merge(drivers_clean[driv_bring], on='driver_id', how='left')
    print(f"After drivers   : {df.shape}")

    # ----------------------------------------------------------------
    # 3. time_features_df
    #    hour_of_day, day_of_week, is_weekend → already in bookings, DROP
    #    Only bring: is_holiday, peak_time_flag, season_peak_lbl
    # ----------------------------------------------------------------
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['merge_hour'] = df['booking_datetime'].dt.floor('H')

    time_clean = time_features_df.copy()
    time_clean['merge_hour'] = pd.to_datetime(time_clean['datetime'])

    time_bring = [
        'merge_hour',                   # join key
        'is_holiday',                   # not in bookings
        'peak_time_flag',               # not in bookings
        'season',                       # not in bookings
        # DROP: hour_of_day, day_of_week, is_weekend — already in bookings(redundant)
    ]
    time_bring = [c for c in time_bring if c in time_clean.columns]
    time_clean = time_clean[time_bring].drop_duplicates('merge_hour')

    df = df.merge(time_clean, on='merge_hour', how='left')
    df = df.drop(columns=['merge_hour'], errors='ignore')
    print(f"After time      : {df.shape}")

    # ----------------------------------------------------------------
    # 4. location_demand_df
    #    completed_rides, cancelled_rides → location-level aggregates,
    #    different meaning from booking-level, RENAME them
    #    total_requests → rename to avoid any clash
    # ----------------------------------------------------------------
    loc_clean = location_demand_df.rename(columns={
        'completed_rides' : 'loc_completed_rides',
        'cancelled_rides' : 'loc_cancelled_rides',
        'total_requests'  : 'loc_total_requests',
    })

    loc_bring = [
        'city', 'pickup_location', 'hour_of_day', 'vehicle_type',  # join keys
        'loc_total_requests',
        'loc_completed_rides',
        'loc_cancelled_rides',
        'avg_wait_time_min',
        'avg_surge_multiplier',
        'demand_level',
    ]
    loc_bring = [c for c in loc_bring if c in loc_clean.columns]
    loc_clean = loc_clean[loc_bring].drop_duplicates(
        ['city', 'pickup_location', 'hour_of_day', 'vehicle_type']
    )

    df = df.merge(loc_clean,
                  on=['city', 'pickup_location', 'hour_of_day', 'vehicle_type'],
                  how='left')
    print(f"After location  : {df.shape}")
    # ----------------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------------
    suffix_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    if suffix_cols:
        print(f"\n⚠️  Suffix columns still present: {suffix_cols}")
    else:
        print("\n✅ No suffix columns.")

    if len(df) != len(bookings_df):
        print(f"❌ Row count changed! {len(bookings_df)} → {len(df)}")
    else:
        print(f"✅ Row count preserved: {len(df):,}")

    print(f"\nFinal shape: {df.shape}")
    return df



# =============================================================================
# ZONE 1 — SAFE FEATURE ENGINEERING (on full df, no aggregations)
# =============================================================================

# -----------------------------------------------------------------------------
# SECTION 2 — DATETIME FEATURES  [ALL MODELS]
# Safe: fixed math, no cross-row aggregation
# -----------------------------------------------------------------------------

def add_datetime_features(df):

    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])

    df['hour_of_day']     = df['booking_datetime'].dt.hour
    df['day_of_week_num'] = df['booking_datetime'].dt.dayofweek
    df['booking_month']   = df['booking_datetime'].dt.month
    df['booking_quarter'] = df['booking_datetime'].dt.quarter
    df['booking_week']    = df['booking_datetime'].dt.isocalendar().week.astype(int)

    # Fixed-threshold flags — safe
    df['is_morning_peak'] = df['hour_of_day'].between(7, 10).astype(int)
    df['is_evening_peak'] = df['hour_of_day'].between(17, 21).astype(int)
    df['is_night_ride']   = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    df['is_weekend']      = df['day_of_week_num'].isin([5, 6]).astype(int)

    # Cyclical encoding — safe (fixed formula)
    df['hour_sin']  = np.sin(2 * np.pi * df['hour_of_day']     / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour_of_day']     / 24)
    df['day_sin']   = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['day_cos']   = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['booking_month']   / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['booking_month']   / 12)

    print("✅ Datetime features added.")
    return df


# -----------------------------------------------------------------------------
# SECTION 3 — RIDE FEATURES  [UC1, UC2, UC3]
# Safe: row-wise math only
# -----------------------------------------------------------------------------

def add_ride_features(df):

    df['actual_ride_time_recorded'] = df['actual_ride_time_min'].notna().astype(int)

    # Row-wise ride time comparison — safe
    actual_time = df['actual_ride_time_min'].fillna(0)

    df['ride_time_ratio'] = actual_time / (df['estimated_ride_time_min'] + 1)
    df['ride_time_error'] = actual_time - df['estimated_ride_time_min']
    df['ride_time_overrun'] = (df['ride_time_error'] > 0).astype(int)

    # Row-wise fare decomposition — safe
    df['expected_fare']  = df['base_fare'] * df['surge_multiplier']
    df['fare_deviation'] = df['booking_value'] - df['expected_fare']
    df['fare_per_km']    = df['booking_value'] / (df['ride_distance_km'] + 0.1)
    df['fare_per_min']   = df['booking_value'] / (df['estimated_ride_time_min'] + 1)
    df['surge_impact']   = df['booking_value'] - df['base_fare']

    # same_loc_flag — safe (already exists, just cast)
    if 'same_loc_flag' in df.columns:
        df['same_loc_flag'] = df['same_loc_flag'].astype(int)

    # --- Fare vs. location average ---
    if 'avg_surge_multiplier' in df.columns:
        df['fare_above_loc_avg'] = (
            df['booking_value'] > (df['base_fare'] * df['avg_surge_multiplier'])
        ).astype(int)

    # --- Base fare per km (pre-surge) ---
    df['base_fare_per_km'] = df['base_fare'] / (df['ride_distance_km'] + 0.1)

   
 
    # --- Fare efficiency: value per unit time AND distance combined ---
    df['fare_per_km_per_surge'] = df['fare_per_km'] / (df['surge_multiplier'] + 0.01)


    # --- Delay burden per km ---
    # Drivers who delay more on longer rides are different from those who delay on short ones.
    df['delay_per_km'] = (df['delay_rate'] * df['avg_pickup_delay_min']) / (
        df['ride_distance_km'] + 0.1
    )

    # ⚠️ REMOVED from here (moved to Zone 3):
    #   is_short_ride, is_long_ride  → quantile(0.25 / 0.75) on full df = leakage
    #   distance_bin                 → pd.qcut on full df = leakage

    print("✅ Ride features added.")
    return df


# -----------------------------------------------------------------------------
# SECTION 4 — CUSTOMER FEATURES  [UC1, UC3]
# Safe: row-wise ratios and fixed-threshold flags
# Removed: is_high_cancel_customer (quantile-based → Zone 3)
# -----------------------------------------------------------------------------

def add_customer_features(df):

    df['customer_tenure_years'] = df['customer_signup_days_ago'] / 365
    df['is_new_customer']       = (df['customer_signup_days_ago'] < 90).astype(int)

    # Row-wise ratios — safe
    df['cancel_to_booking_ratio'] = df['cancelled_rides']   / (df['total_bookings'] + 1)
    df['cust_completion_rate']    = df['completed_rides']   / (df['total_bookings'] + 1)
    df['incomplete_ride_share']   = df['incomplete_rides']  / (df['total_bookings'] + 1)

    # Fixed-threshold flags — safe
    df['is_low_rated_customer'] = (df['avg_customer_rating'] < 3.5).astype(int)

    # Vehicle preference match — safe (row-wise comparison)
    if 'preferred_vehicle_type' in df.columns and 'vehicle_type' in df.columns:
        df['vehicle_preference_match'] = (
            df['preferred_vehicle_type'] == df['vehicle_type']
        ).astype(int)

    # Fixed-bin age cut — safe (fixed bin edges, not quantile)
    df['customer_age_bin'] = pd.cut(
        df['customer_age'],
        bins=[0, 25, 35, 50, 100],
        labels=['young', 'adult', 'mid', 'senior']
    )

    # --- Cancellation Risk Score (UC3 primary signal) ---
    # Higher = more likely to cancel.
    df['cancel_risk_score'] = (
        df['cancellation_rate']         * 0.35 +
        df['cancel_to_booking_ratio']   * 0.25 +
        df['is_low_rated_customer']     * 0.10 
    ).clip(0, 1)
 
    df['cancel_risk_tier'] = pd.cut(
        df['cancel_risk_score'],
        bins=[0, 0.3, 0.5, 0.7, 1.01],
        labels=['low', 'medium', 'high', 'very_high'],
        include_lowest=True
    )

    # --- Delay Risk Score (UC4 primary signal) ---
    df['delay_risk_score'] = (
        df['delay_rate']               * 0.35 +
        (df['ride_distance_km'] > 15).astype(int)     * 0.10 +  # fixed threshold instead of quantile
        df['is_night_ride']            * 0.10
    ).clip(0, 1)
 
    df['delay_risk_tier'] = pd.cut(
        df['delay_risk_score'],
        bins=[0, 0.3, 0.5, 0.7, 1.01],
        labels=['low', 'medium', 'high', 'very_high'],
        include_lowest=True
    )

    # ⚠️ REMOVED from here (moved to Zone 3):
    #   is_high_cancel_customer → quantile(0.75) on full df = leakage

    print("✅ Customer features added.")
    return df


# -----------------------------------------------------------------------------
# SECTION 5 — DRIVER FEATURES  [UC1, UC4]
# Safe: row-wise ratios and fixed-threshold flags
# -----------------------------------------------------------------------------

def add_driver_features(df):

    df['rejection_rate']         = df['rejected_rides'] / (df['total_assigned_rides'] + 1)
    df['driver_incomplete_rate'] = (
        df['incomplete_rides_driver'] if 'incomplete_rides_driver' in df.columns
        else df['incomplete_rides']
    ) / (df['accepted_rides'] + 1)
    df['delay_per_ride']         = df['delay_count'] / (df['accepted_rides'] + 1)

    # Weighted composite score — row-wise, safe
    df['driver_reliability_score'] = (
        df['acceptance_rate'] * 0.35 +
        (1 - df['delay_rate']) * 0.3 +
        (df['avg_driver_rating'] / 5) * 0.2 +
        (1 - df['driver_incomplete_rate']) * 0.15
    ).clip(0, 1)

    # Fixed-bin cuts — safe (fixed edges, not quantile)
    df['driver_acceptance_tier'] = pd.cut(
        df['acceptance_rate'],
        bins=[0, 0.6, 0.75, 0.9, 1.01],
        labels=['low', 'medium', 'high', 'very_high']
    )
    df['driver_experience_bin'] = pd.cut(
        df['driver_experience_years'],
        bins=[0, 3, 7, 14],
        labels=['junior', 'mid', 'senior']
    )

    # Fixed-threshold flags — safe
    df['is_unreliable_driver'] = (
        (df['delay_rate'] > 0.15) | (df['acceptance_rate'] < 0.6)
    ).astype(int)
    df['is_low_rated_driver']  = (df['avg_driver_rating'] < 3.5).astype(int)

    print("✅ Driver features added.")
    return df


# -----------------------------------------------------------------------------
# SECTION 6 — LOCATION FEATURES  [UC2, UC3]
# Safe: row-wise math and fixed-value flags
# Removed: pickup_hotspot_flag (value_counts on full df = leakage)
#          pickup/drop location_freq (value_counts on full df = leakage)
# -----------------------------------------------------------------------------

def add_location_features(df):

    # Row-wise demand ratio — safe
    if 'total_requests' in df.columns and 'completed_rides_loc' in df.columns:
        df['demand_supply_ratio'] = df['total_requests'] / (df['completed_rides_loc'] + 1)
    elif 'total_requests' in df.columns:
        df['demand_supply_ratio'] = df['total_requests'] / (df['completed_rides'] + 1)

    # Row-wise surge deviation vs location average — safe
    if 'avg_surge_multiplier' in df.columns:
        df['location_surge_deviation'] = df['surge_multiplier'] - df['avg_surge_multiplier']

    # Fixed-bin wait time cut — safe (fixed edges, not quantile)
    if 'avg_wait_time_min' in df.columns:
        df['wait_time_bin'] = pd.cut(
            df['avg_wait_time_min'],
            bins=[0, 5, 10, 20, 999],
            labels=['fast', 'normal', 'slow', 'very_slow']
        )

    # Demand level flag — safe (maps existing column, no aggregation)
    if 'demand_level' in df.columns:
        df['is_high_demand_location'] = (df['demand_level'] == 'High').astype(int)

    # ⚠️ REMOVED from here (moved to Zone 3):
    #   pickup_hotspot_flag      → value_counts().head(10) on full df = leakage
    #   pickup_location_freq     → value_counts(normalize=True) on full df = leakage
    #   drop_location_freq       → value_counts(normalize=True) on full df = leakage

    print("✅ Location features added.")
    return df


# -----------------------------------------------------------------------------
# SECTION 7 — INTERACTION FEATURES  [ALL MODELS]
# Safe: row-wise multiplications of existing columns
# -----------------------------------------------------------------------------

def add_interaction_features(df):

    traffic_map = {'Low': 1, 'Medium': 2, 'High': 3}
    weather_map = {'Clear': 0, 'Rain': 1, 'Heavy Rain': 2}
    df['traffic_num'] = df['traffic_level'].map(traffic_map)
    df['weather_num'] = df['weather_condition'].map(weather_map)

    # Row-wise interactions — all safe
    df['surge_x_distance']      = df['surge_multiplier']          * df['ride_distance_km']
    df['surge_x_traffic']       = df['surge_multiplier']          * df['traffic_num']
    df['peak_x_surge']          = df['peak_time_flag']            * df['surge_multiplier']
    df['base_x_surge']          = df['base_fare']                 * df['surge_multiplier']
    df['traffic_x_ridetime']    = df['traffic_num']               * df['estimated_ride_time_min']
    df['weather_x_ridetime']    = df['weather_num']               * df['estimated_ride_time_min']
    df['cancel_risk_x_peak']    = df['cancellation_rate']         * df['peak_time_flag']
    df['cancel_risk_x_night']   = df['cancellation_rate']         * df['is_night_ride']
    df['reliability_x_distance']= df['driver_reliability_score']  * df['ride_distance_km']
    df['delay_x_traffic']       = df['delay_rate']                * df['traffic_num']

    
    # Fixed compound flags — safe
    df['rain_high_traffic']      = (
        df['weather_condition'].isin(['Rain', 'Heavy Rain']) &
        (df['traffic_level'] == 'High')
    ).astype(int)
    df['high_delay_high_traffic']= (
        (df['delay_rate'] > 0.1) & (df['traffic_level'] == 'High')
    ).astype(int)
    df['holiday_x_peak']         = df['is_holiday']    * df['peak_time_flag']
    df['night_x_high_traffic']   = df['is_night_ride'] * (df['traffic_num'] == 3).astype(int)

    print("✅ Interaction features added.")
    return df


# -----------------------------------------------------------------------------
# SECTION 8 — ENCODING  [ALL MODELS]
# Safe: fixed mappings and one-hot (no data-driven encoding)
# Removed: frequency encoding → Zone 3
# -----------------------------------------------------------------------------

def encode_features(df):

    df['is_cancelled'] = (df['booking_status'] == 'Cancelled').astype(int)

    # Ordinal — fixed maps, safe
    df['traffic_enc'] = df['traffic_level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['weather_enc'] = df['weather_condition'].map({'Clear': 0, 'Rain': 1, 'Heavy Rain': 2})
    if 'demand_level' in df.columns:
        df['demand_enc'] = df['demand_level'].map({'Low': 0, 'Medium': 1, 'High': 2})

    # Target encoding for UC1 — safe (fixed map)
    df['booking_status_enc'] = df['booking_status'].map(
        {'Completed': 0, 'Cancelled': 1, 'Incomplete': 2}
    )
    
    # One-hot — safe (no learned statistics)
    df = pd.get_dummies(df, columns=['incomplete_ride_reason'], prefix='reason', drop_first=True)
    ohe_cols = [
        'customer_city',
        'driver_city',
        'preferred_vehicle_type',
        'season',
        'day_of_week',
        'vehicle_type', 
        'city', 
        'customer_gender'
    ]

    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)



    # Ordinal bins — fixed maps, safe
    # .astype(int) required: .map() on a pd.cut() Series inherits the
    # categorical dtype even when all mapped values are integers.
    if 'driver_experience_bin' in df.columns:
        df['driver_experience_enc'] = df['driver_experience_bin'].map(
            {'junior': 0, 'mid': 1, 'senior': 2}).astype(int)
    if 'driver_acceptance_tier' in df.columns:
        df['driver_acceptance_enc'] = df['driver_acceptance_tier'].map(
            {'low': 0, 'medium': 1, 'high': 2, 'very_high': 3}).astype(int)
    if 'customer_age_bin' in df.columns:
        df['customer_age_enc'] = df['customer_age_bin'].map(
            {'young': 0, 'adult': 1, 'mid': 2, 'senior': 3}).astype(int)
    if 'wait_time_bin' in df.columns:
        df['wait_time_enc'] = df['wait_time_bin'].map(
            {'fast': 0, 'normal': 1, 'slow': 2, 'very_slow': 3}).astype(int)

    # ⚠️ REMOVED from here (moved to Zone 3):
    #   pickup_location_freq, drop_location_freq → value_counts = leakage

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    print("✅ Encoding done.")
    return df


def add_new_uc1_interactions(df):
    """
    New interaction features targeting UC1 minority class (incomplete rides).
    Called AFTER encode_features so demand_enc is available.
    """
    # Uses demand_enc (available after encoding)
    if 'demand_enc' in df.columns and 'is_unreliable_driver' in df.columns:
        df['demand_x_unreliable'] = df['is_unreliable_driver'] * df['demand_enc']

    if 'demand_enc' in df.columns and 'acceptance_rate' in df.columns:
        df['demand_x_low_acceptance'] = df['demand_enc'] * (1 - df['acceptance_rate'])

    if 'demand_enc' in df.columns and 'driver_incomplete_rides' in df.columns and 'total_assigned_rides' in df.columns:
        df['demand_x_incomplete_rate'] = (
            df['demand_enc'] * df['driver_incomplete_rides'] / (df['total_assigned_rides'] + 1)
        )

    # These don't need demand_enc — safe anywhere but keeping together
    if 'ride_distance_km' in df.columns and 'is_night_ride' in df.columns and 'weather_enc' in df.columns:
        df['ride_difficulty'] = (
            df['ride_distance_km'] * df['is_night_ride'] * df['weather_enc']
        )

    if 'ride_distance_km' in df.columns and 'traffic_enc' in df.columns:
        df['distance_x_traffic'] = df['ride_distance_km'] * df['traffic_enc']

    if 'driver_experience_years' in df.columns and 'ride_distance_km' in df.columns:
        df['exp_x_distance'] = df['driver_experience_years'] * df['ride_distance_km']

    if 'avg_driver_rating' in df.columns and 'estimated_ride_time_min' in df.columns:
        df['rating_x_ridetime'] = df['avg_driver_rating'] * df['estimated_ride_time_min']

    if 'is_low_rated_customer' in df.columns and 'is_low_rated_driver' in df.columns:
        df['dual_low_rating'] = df['is_low_rated_customer'] * df['is_low_rated_driver']

    if 'is_new_customer' in df.columns and 'is_unreliable_driver' in df.columns:
        df['new_cust_x_unreliable'] = df['is_new_customer'] * df['is_unreliable_driver']

    if 'estimated_ride_time_min' in df.columns and 'avg_wait_time_min' in df.columns:
        df['time_pressure'] = df['estimated_ride_time_min'] / (df['avg_wait_time_min'] + 1)

    if 'peak_time_flag' in df.columns and 'ride_distance_km' in df.columns:
        df['peak_x_distance'] = df['peak_time_flag'] * df['ride_distance_km']

    print("✅ New UC1 interaction features added.")
    return df


# -----------------------------------------------------------------------------
# SECTION 9 — LOG TRANSFORM  [UC2]
# Safe: row-wise math
# -----------------------------------------------------------------------------
def log_transform(df):
    log_cols = [
        'base_fare',
        'ride_distance_km',
        'total_bookings',
        'total_requests',
        'estimated_ride_time_min'
    ]

    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # TARGET — keep separate
    if 'booking_value' in df.columns:
        df['booking_value_log'] = np.log1p(df['booking_value'].clip(lower=0))

    print("✅ Log transforms applied (safe).")
    return df


# -----------------------------------------------------------------------------
# SECTION 10 — DROP RAW COLUMNS
# -----------------------------------------------------------------------------

def drop_raw_columns(df):
    drop_cols = [
        'booking_id', 'customer_id', 'driver_id',
        'booking_datetime', 'actual_time_available', 'merge_hour',
        'pickup_location', 'drop_location',
        'traffic_level', 'weather_condition', 'demand_level',
        'booking_status',
        'driver_experience_bin', 'driver_acceptance_tier',
        'customer_age_bin', 'wait_time_bin',
        'traffic_num', 'weather_num','delay_risk_tier',
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"✅ Dropped {len(drop_cols)} raw/redundant columns.")
    return df


# =============================================================================
# ZONE 1 MASTER FUNCTION
# =============================================================================

def run_zone1_engineering(bookings_df, customers_df, drivers_df,
                          location_demand_df, time_features_df):
    print("\n" + "="*60)
    print("ZONE 1 — SAFE FEATURE ENGINEERING")
    print("="*60)

    df = merge_all_tables(bookings_df, customers_df, drivers_df,
                          location_demand_df, time_features_df)
    df = add_datetime_features(df)
    df = add_ride_features(df)
    df = add_customer_features(df)
    df = add_driver_features(df)
    df = add_location_features(df)
    df = add_interaction_features(df)
    df = encode_features(df)
    df = add_new_uc1_interactions(df)   # new features that need demand_enc
    df = log_transform(df)
    df = drop_raw_columns(df)

    print(f"\n✅ Zone 1 complete. Shape: {df.shape}")
    return df


# =============================================================================
# ZONE 2 — DEFINE TARGETS, LEAKAGE DROPS, SPLIT
# =============================================================================

# Leakage columns per use case
# (columns derived from the target — must never appear in X)


UC1_LEAKAGE = [
    
    # Target and direct derivations
    'booking_status_enc',
    'booking_status',
    'is_cancelled',

    # Post-ride time info
    'actual_ride_time_min',
    'actual_ride_time_recorded',
    'ride_time_ratio',
    'ride_time_error',
    'ride_time_overrun',

    # booking_value derived — only known after booking is finalised
    'fare_per_km',
    'fare_per_min',
    'fare_deviation',
    'surge_impact',
    'fare_above_loc_avg',
    'fare_per_km_per_surge',

    # Redundant with delay_rate — binary summary of a feature already in X
    'driver_delay_flag',

    # Risk tier cols — pd.cut() categoricals built from leaky inputs
    # cancel_risk_tier uses cancellation_rate (indirect leakage for UC1)
    # delay_risk_tier  uses delay_rate (indirect leakage for UC1)
    'cancel_risk_tier',
    'cancel_risk_score',
    'delay_risk_tier',
    'delay_risk_score',

    'customer_cancel_flag',     # directly flags cancelling customers
    'incomplete_ride_share',    # derived from incomplete rides (class 2)
    'incomplete_rides',         # raw count — reconstructs class 2
    'cancel_to_booking_ratio',  # derived from cancelled rides (class 1)
    'cust_completion_rate',     # derived from completed/total
    'cancelled_rides',          # raw count — reconstructs class 1
    'cancellation_rate',        # same family
    'total_bookings',           # reconstructs the above
    'completed_rides',          # completes the picture

]


UC2_LEAKAGE = [

    # Target
    'booking_value',
    'booking_value_log',

    # Derived from booking_value
    'fare_deviation',
    'fare_per_km',
    'fare_per_min',
    'surge_impact',
    'fare_above_loc_avg',
    'fare_per_km_per_surge',
    'surge_cost_share',

    # base_fare — 0.92 corr with target, back-calculated in synthetic data
    'base_fare',
    'base_fare_per_km',    # base_fare / distance
    'base_x_surge',        # base_fare * surge ≈ booking_value
    'expected_fare',       # base_fare * surge — near-perfect target proxy

    # Irrelevant to fare
    'cancel_risk_score',
    'cancel_risk_tier',
    'delay_risk_score',
    'delay_risk_tier',
    'Customer_Loyalty_Score',
    'loyalty_x_cancel',
    'new_high_cancel',
    'driver_delay_flag',

    #trial without dominance of these two features
    # 'vehicle_type_Cab', 
    # 'vehicle_type_Bike',

    #newly added for UC1
    'demand_x_unreliable',
    'demand_x_low_acceptance',
    'ride_difficulty',
    'distance_x_traffic',
    'exp_x_distance',
    'rating_x_ridetime',
    'dual_low_rating',
    'new_cust_x_unreliable',
    'time_pressure',
    'peak_x_distance',
    'demand_x_incomplete_rate',
    ]

UC3_LEAKAGE = [
    'customer_cancel_flag',  
    'is_cancelled'    # the target itself
    # Redundant with delay_rate — binary summary of a feature already in X
    'driver_delay_flag',
    # Direct cancellation metrics — used to create the target
    'cancelled_rides',
    'cancellation_rate',
    'cancel_to_booking_ratio',
    'incomplete_ride_share', # includes cancelled rides in numerator, very close proxy
    'is_high_cancel_customer',
    'new_high_cancel',
    # Interactions built from cancellation metrics
    'cancel_risk_x_peak',
    'cancel_risk_x_night',
    # Composite score built from cancellation metrics
    'cancel_risk_score',         # uses cancellation_rate — indirect leakage
    'cancel_risk_tier',          # uses cancel_risk_score — indirect leakage

    # UC1 loyalty uses cancellation_rate — indirect leakage for UC3
    'Customer_Loyalty_Score_uc1',
    'loyalty_x_cancel',

    # Direct target reconstructors
    'booking_status_enc',
    'completed_rides',
    'incomplete_rides',
    'total_bookings',
    'cust_completion_rate',
    'booking_value_log',
    'booking_value',         # check if this is pre or post-ride
    # Post-hoc ride outcome features
    'actual_ride_time_min',
    'actual_ride_time_recorded',
    'ride_time_ratio',
    'ride_time_error',
    'ride_time_overrun',
    'fare_deviation',
    'fare_above_loc_avg',
    # Cancellation reason dummies
    'reason_Customer No-Show',
    'reason_Driver Delay',
    'reason_Not Applicable',
    'reason_Vehicle Issue',

    #newly added for UC1
    'demand_x_unreliable',
    'demand_x_low_acceptance',
    'ride_difficulty',
    'distance_x_traffic',
    'exp_x_distance',
    'rating_x_ridetime',
    'time_pressure',
    'peak_x_distance',
    'demand_x_incomplete_rate',

]

UC4_LEAKAGE = [
    'driver_delay_flag',         # the target itself
    # Direct delay metrics — used to create the target
    'delay_count',
    'delay_rate',
    'delay_per_ride',
    # Interactions built from delay metrics
    'high_delay_high_traffic',
    'delay_x_traffic',
    # New features derived from delay_rate
    'delay_per_km',              # uses delay_rate
    'delay_risk_score',          # uses delay_rate
    'delay_risk_tier',           # uses delay_risk_score
    'is_unreliable_driver',
    'driver_reliability_score',
    'driver_incomplete_rate',

    # Not relevant to driver delay
    'cancel_risk_score',        # built from cancellation_rate — unrelated to delay
    'cancel_risk_tier',         # pd.cut() categorical of cancel_risk_score
    'Customer_Loyalty_Score_uc3',

    # Direct outcome labels
    'is_cancelled',
    'booking_status_enc',
    'reason_Driver Delay',
    'reason_Customer No-Show',
    'reason_Not Applicable',
    'reason_Vehicle Issue',
    # Post-ride actuals
    'actual_ride_time_min',
    'actual_ride_time_recorded',
    'ride_time_ratio',
    'ride_time_error',
    'ride_time_overrun',
    'fare_deviation',
    'fare_above_loc_avg',
    # Leaky driver aggregate
    'avg_pickup_delay_min',
    # Leaky customer aggregates
    'cancelled_rides',
    'incomplete_rides',
    'completed_rides',
    'cancellation_rate',
    'customer_cancel_flag',
    'cancel_to_booking_ratio',
    'cust_completion_rate',
    'incomplete_ride_share',
    'total_bookings',        # reconstructs the above
    'booking_value_log',     # post-ride value
    'booking_value',

    # newly added for UC1
    'demand_x_unreliable',
    'demand_x_low_acceptance',
    'ride_difficulty',
    'distance_x_traffic',
    'new_cust_x_unreliable',
]


def get_splits(df, use_case='UC1', test_size=0.2, random_state=42):
    """
    Zone 2: drop leakage columns, define X and y, split.
    Zone 3: fit quantile flags, frequency encoding, scaler on train only.
    """

    leakage_map = {
        'UC1': (UC1_LEAKAGE, 'booking_status_enc'),
        'UC2': (UC2_LEAKAGE, 'booking_value_log'),
        'UC3': (UC3_LEAKAGE, 'is_cancelled'),
        'UC4': (UC4_LEAKAGE, 'driver_delay_flag'),
    }

    leakage_cols, target_col = leakage_map[use_case]

    if use_case == 'UC1':
        reason_cols = [c for c in df.columns if c.startswith('reason_')]
        leakage_cols = leakage_cols + reason_cols



    # Work on a copy — never mutate the original df
    df_uc = df.copy()

    leakage_to_drop = [c for c in leakage_cols if c in df_uc.columns and c != target_col]
    X = df_uc.drop(columns=leakage_to_drop + [target_col], errors='ignore')
    y = df_uc[target_col].copy()

    # Drop rows where target is null
    mask = y.notna()
    X, y = X[mask], y[mask]

    # Stratify only for classification
    stratify = y if use_case in ['UC1', 'UC3', 'UC4'] else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print(f"\n{'='*60}")
    print(f"ZONE 2 — {use_case} Split done")
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    if use_case == 'UC2':
        corr = X.corrwith(y).sort_values(ascending=False)
        print("\nCorrelations with target:\n")
        print(corr.head(10))

    # ------------------------------------------------------------------
    # ZONE 3 — Leaky operations: fit on X_train, apply to both
    # ------------------------------------------------------------------
    print(f"ZONE 3 — Post-split feature engineering ({use_case})")

    X_train, X_test = apply_zone3_features(X_train, X_test, use_case)

    print(f"  Final train shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# ZONE 3 — POST-SPLIT LEAKY FEATURE ENGINEERING
# Fit ONLY on X_train. Apply same thresholds/mappings to X_test.
# =============================================================================

def apply_zone3_features(X_train, X_test, use_case):
    """
    Applies all data-driven feature engineering after the split.
    Everything here is fit on X_train only.
    """
 
 
    if 'driver_experience_years' in X_train.columns:
    # Cap experience at (driver_age - 18) — fit cap on train only
        for df_ in [X_train, X_test]:
            max_valid_exp = df_['driver_age'] - 18
            df_['driver_experience_years'] = df_['driver_experience_years'].clip(
                upper=max_valid_exp
            )
        print("  ✅ driver_experience_years capped at (driver_age - 18).")
 
    # ------------------------------------------------------------------
    # 3B — Quantile-based ride distance flags  [UC1, UC2, UC3]
    # ------------------------------------------------------------------
    if 'ride_distance_km' in X_train.columns and use_case in ['UC1', 'UC2', 'UC3']:

        short_thresh = X_train['ride_distance_km'].quantile(0.25)
        long_thresh  = X_train['ride_distance_km'].quantile(0.75)

        for df_ in [X_train, X_test]:
            df_['is_short_ride'] = (df_['ride_distance_km'] < short_thresh).astype(int)
            df_['is_long_ride']  = (df_['ride_distance_km'] > long_thresh).astype(int)
            df_['distance_bin_enc'] = pd.cut(
                df_['ride_distance_km'],
                bins=[-np.inf, short_thresh, long_thresh, np.inf],
                labels=[0, 1, 2]
            ).astype(float)

        print("  ✅ Quantile distance flags applied (train thresholds used on test).")

    # ------------------------------------------------------------------
    # 3C — Quantile-based high cancellation flag  [UC1, UC3]
    # ------------------------------------------------------------------
    if 'cancellation_rate' in X_train.columns and use_case == 'UC1':

        cancel_thresh = X_train['cancellation_rate'].quantile(0.75)

        for df_ in [X_train, X_test]:
            df_['is_high_cancel_customer'] = (
                df_['cancellation_rate'] > cancel_thresh
            ).astype(int)

            # --- Customer segment: new + high cancel = highest risk ---
            df_['new_high_cancel'] = (
            (df_['is_new_customer'] == 1) & (df_['is_high_cancel_customer'] == 1)).astype(int)
        print("  ✅ High cancel flag applied (train threshold used on test).")


        

    # ------------------------------------------------------------------
    # 3D — Frequency encoding: pickup & drop location  [UC1, UC2]
    # ------------------------------------------------------------------
    for col in ['pickup_location', 'drop_location']:
        if col in X_train.columns and use_case in ['UC1', 'UC2']:
            freq_map = X_train[col].value_counts(normalize=True)
            X_train[col + '_freq'] = X_train[col].map(freq_map)
            # Unseen locations in test get 0
            X_test[col  + '_freq'] = X_test[col].map(freq_map).fillna(0)

    if use_case in ['UC1', 'UC2']:
        print("  ✅ Frequency encoding applied (train frequencies used on test).")

    # ------------------------------------------------------------------
    # 3E — Pickup hotspot flag  [UC1, UC2]
    # Hotspot = top 10 pickup locations by frequency in TRAIN only
    # ------------------------------------------------------------------
    if 'pickup_location' in X_train.columns and use_case in ['UC1', 'UC2']:
        top_locs = X_train['pickup_location'].value_counts().head(10).index
        X_train['pickup_hotspot_flag'] = X_train['pickup_location'].isin(top_locs).astype(int)
        X_test['pickup_hotspot_flag']  = X_test['pickup_location'].isin(top_locs).astype(int)
        print("  ✅ Pickup hotspot flag applied (train top-10 used on test).")

        # City_Pair freq
        freq_map = X_train['City_Pair'].value_counts(normalize=True)
        X_train['City_Pair_freq'] = X_train['City_Pair'].map(freq_map)
        X_test['City_Pair_freq']  = X_test['City_Pair'].map(freq_map).fillna(0)



    
    # Customer_Loyalty_Score — UC1 only (cancellation_rate is leaked in UC3)
    if 'cancellation_rate' in X_train.columns and use_case == 'UC1':
        max_tenure = X_train['customer_signup_days_ago'].max() / 365
        for df_ in [X_train, X_test]:
            tenure_norm     = (df_['customer_signup_days_ago'] / 365) / (max_tenure + 1e-9)
            completion_rate = df_['completed_rides'] / (df_['total_bookings'] + 1)
            rating_norm     = df_['avg_customer_rating'] / 5
            low_cancel      = 1 - df_['cancellation_rate'].clip(0, 1)
            df_['Customer_Loyalty_Score'] = (
                tenure_norm     * 0.20 +
                completion_rate * 0.40 +
                rating_norm     * 0.20 +
                low_cancel      * 0.20
            ).clip(0, 1)
            df_['loyalty_x_cancel'] = df_['Customer_Loyalty_Score'] * df_['cancellation_rate']
        print("  ✅ Customer Loyalty Score applied.")

    # UC3 — cancellation_rate is leaked so build loyalty without it
    # UC3 — completed_rides/total_bookings/cancellation_rate all dropped (leakage)
    elif use_case == 'UC3':
        max_tenure = X_train['customer_signup_days_ago'].max() / 365
        for df_ in [X_train, X_test]:
            tenure_norm = (df_['customer_signup_days_ago'] / 365) / (max_tenure + 1e-9)
            rating_norm = df_['avg_customer_rating'] / 5
            df_['Customer_Loyalty_Score'] = (
                tenure_norm * 0.50 +
                rating_norm * 0.50
            ).clip(0, 1)
        print("  ✅ Customer Loyalty Score (tenure + rating only) applied for UC3.")
    # ------------------------------------------------------------------
    # 3F — StandardScaler on numeric columns
    # Fit on X_train, transform both
    # ------------------------------------------------------------------

    SCALE_COLS = [
        'surge_multiplier', 'estimated_ride_time_min',
        'acceptance_rate', 'delay_rate', 'avg_pickup_delay_min',
        'customer_age', 'customer_tenure_years', 'customer_signup_days_ago',
        'fare_per_km', 'fare_per_min',
        'driver_reliability_score', 'demand_supply_ratio',
        'cancel_to_booking_ratio', 'cust_completion_rate',
        'rejection_rate', 'delay_per_ride', 'driver_incomplete_rate',
        'location_surge_deviation', 'surge_x_distance',
        'traffic_x_ridetime', 'reliability_x_distance',
        'ride_distance_km', 'base_fare',
    ]
    cols_to_scale = [c for c in SCALE_COLS if c in X_train.columns]

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])
    print(f"  ✅ StandardScaler applied to {len(cols_to_scale)} columns.")

    # Add this in get_splits() after fitting the scaler
    joblib.dump(scaler, f"models/scaler_{use_case.lower()}.pkl")

    
    return X_train, X_test


# =============================================================================
# USAGE
# =============================================================================

if __name__ == '__main__':

    #Step 1 — Zone 1: safe engineering on full df
    df = run_zone1_engineering(
        bookings_df, customers_df, drivers_df,
        location_demand_df, time_features_df
    )

    
    #Step 2 — Zone 2 + 3: split + post-split engineering per model
    X_train_uc1, X_test_uc1, y_train_uc1, y_test_uc1 = get_splits(df.copy(), 'UC1')
    X_train_uc2, X_test_uc2, y_train_uc2, y_test_uc2 = get_splits(df.copy(), 'UC2')
    X_train_uc3, X_test_uc3, y_train_uc3, y_test_uc3 = get_splits(df.copy(), 'UC3')
    X_train_uc4, X_test_uc4, y_train_uc4, y_test_uc4 = get_splits(df.copy(), 'UC4')

    print("Pipeline loaded. Call run_zone1_engineering() then get_splits().")


    # 1. No target leakage in X
    assert 'booking_status_enc'  not in X_train_uc1.columns
    assert 'booking_value_log'   not in X_train_uc2.columns
    assert 'customer_cancel_flag' not in X_train_uc3.columns
    assert 'driver_delay_flag'   not in X_train_uc4.columns
    print("✅ No target leakage")

    # 2. No nulls in targets
    assert y_train_uc1.isna().sum() == 0
    assert y_train_uc2.isna().sum() == 0
    assert y_train_uc3.isna().sum() == 0
    assert y_train_uc4.isna().sum() == 0
    print("✅ No null targets")

    # 3. Class balance check
    print("\nClass balance in training sets:")
    print("UC1:", y_train_uc1.value_counts(normalize=True).round(3))
    print("UC3:", y_train_uc3.value_counts(normalize=True).round(3))
    print("UC4:", y_train_uc4.value_counts(normalize=True).round(3))


    





        