# =============================================================================
# src/feature_engineering/zone1_features.py
# =============================================================================
# All Zone 1 feature derivation functions.
# Safe to run on the full dataset — no data-driven stats, no targets.
# =============================================================================

import numpy as np
import pandas as pd


# =============================================================================
# DATETIME FEATURES  [ALL MODELS]
# =============================================================================

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])

    df['day_of_week_num'] = df['booking_datetime'].dt.dayofweek
    df['booking_month']   = df['booking_datetime'].dt.month
    df['booking_quarter'] = df['booking_datetime'].dt.quarter
    df['booking_week']    = df['booking_datetime'].dt.isocalendar().week.astype(int)

    df['is_morning_peak'] = df['hour_of_day'].between(7, 10).astype(int)
    df['is_evening_peak'] = df['hour_of_day'].between(17, 21).astype(int)
    df['is_night_ride']   = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    df['is_weekend']      = df['day_of_week_num'].isin([5, 6]).astype(int)

    df['hour_sin']  = np.sin(2 * np.pi * df['hour_of_day']     / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour_of_day']     / 24)
    df['day_sin']   = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
    df['day_cos']   = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['booking_month']   / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['booking_month']   / 12)

    print("✅ Datetime features added.")
    return df


# =============================================================================
# RIDE FEATURES  [UC1, UC2, UC3]
# =============================================================================

def add_ride_features(df: pd.DataFrame) -> pd.DataFrame:
    df['expected_fare'] = df['base_fare'] * df['surge_multiplier']
    df['fare_per_km']   = df['expected_fare'] / (df['ride_distance_km'] + 0.1)
    df['fare_per_min']  = df['expected_fare'] / (df['estimated_ride_time_min'] + 1)

    if 'same_loc_flag' in df.columns:
        df['same_loc_flag'] = df['same_loc_flag'].astype(int)

    df['delay_per_km'] = (df['delay_rate'] * df['avg_pickup_delay_min']) / (
        df['ride_distance_km'] + 0.1
    )
    print("✅ Ride features added.")
    return df


# =============================================================================
# CUSTOMER FEATURES  [UC1, UC3]
# =============================================================================

def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['customer_tenure_years'] = df['customer_signup_days_ago'] / 365
    df['is_new_customer']       = (df['customer_signup_days_ago'] < 90).astype(int)

    df['cancel_to_booking_ratio'] = df['cancelled_rides']  / (df['total_bookings'] + 1)
    df['cust_completion_rate']    = df['completed_rides']  / (df['total_bookings'] + 1)
    df['incomplete_ride_share']   = df['incomplete_rides'] / (df['total_bookings'] + 1)

    df['is_low_rated_customer'] = (df['avg_customer_rating'] < 3.5).astype(int)

    if 'preferred_vehicle_type' in df.columns and 'vehicle_type' in df.columns:
        df['vehicle_preference_match'] = (
            df['preferred_vehicle_type'] == df['vehicle_type']
        ).astype(int)

    df['customer_age_bin'] = pd.cut(
        df['customer_age'],
        bins=[0, 25, 35, 50, 100],
        labels=['young', 'adult', 'mid', 'senior']
    )
    print("✅ Customer features added.")
    return df


# =============================================================================
# DRIVER FEATURES  [UC1, UC4]
# =============================================================================

def add_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    df['rejection_rate'] = df['rejected_rides'] / (df['total_assigned_rides'] + 1)
    df['driver_incomplete_rate'] = (
        df['driver_incomplete_rides'] if 'driver_incomplete_rides' in df.columns
        else df['incomplete_rides']
    ) / (df['accepted_rides'] + 1)
    df['delay_per_ride'] = df['delay_count'] / (df['accepted_rides'] + 1)

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

    df['is_unreliable_driver'] = (
        (df['delay_rate'] > 0.15) | (df['acceptance_rate'] < 0.6)
    ).astype(int)
    df['is_low_rated_driver'] = (df['avg_driver_rating'] < 3.5).astype(int)

    print("✅ Driver features added.")
    return df


# =============================================================================
# LOCATION FEATURES  [UC2, UC3]
# =============================================================================

def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'loc_total_requests' in df.columns and 'loc_completed_rides' in df.columns:
        df['demand_supply_ratio'] = df['loc_total_requests'] / (df['loc_completed_rides'] + 1)

    if 'avg_surge_multiplier' in df.columns:
        df['location_surge_deviation'] = df['surge_multiplier'] - df['avg_surge_multiplier']

    if 'avg_wait_time_min' in df.columns:
        df['wait_time_bin'] = pd.cut(
            df['avg_wait_time_min'],
            bins=[0, 5, 10, 20, 999],
            labels=['fast', 'normal', 'slow', 'very_slow']
        )

    if 'demand_level' in df.columns:
        df['is_high_demand_location'] = (df['demand_level'] == 'High').astype(int)

    print("✅ Location features added.")
    return df


# =============================================================================
# INTERACTION FEATURES  [ALL MODELS]
# =============================================================================

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    traffic_map = {'Low': 1, 'Medium': 2, 'High': 3}
    weather_map = {'Clear': 0, 'Rain': 1, 'Heavy Rain': 2}
    df['traffic_num'] = df['traffic_level'].map(traffic_map)
    df['weather_num'] = df['weather_condition'].map(weather_map)

    df['surge_x_distance']   = df['surge_multiplier']       * df['ride_distance_km']
    df['surge_x_traffic']    = df['surge_multiplier']       * df['traffic_num']
    df['peak_x_surge']       = df['peak_time_flag']         * df['surge_multiplier']
    df['base_x_surge']       = df['base_fare']              * df['surge_multiplier']
    df['traffic_x_ridetime'] = df['traffic_num']            * df['estimated_ride_time_min']
    df['weather_x_ridetime'] = df['weather_num']            * df['estimated_ride_time_min']
    df['cancel_risk_x_peak'] = df['cancellation_rate']      * df['peak_time_flag']
    df['cancel_risk_x_night']= df['cancellation_rate']      * df['is_night_ride']
    df['delay_x_traffic']    = df['delay_rate']             * df['traffic_num']

    df['rain_high_traffic'] = (
        df['weather_condition'].isin(['Rain', 'Heavy Rain']) &
        (df['traffic_level'] == 'High')
    ).astype(int)
    df['high_delay_high_traffic'] = (
        (df['delay_rate'] > 0.1) & (df['traffic_level'] == 'High')
    ).astype(int)
    df['night_x_high_traffic'] = (
        df['is_night_ride'].astype(int) * (df['traffic_num'] == 3).astype(int)
    )
    print("✅ Interaction features added.")
    return df


# =============================================================================
# ENCODING  [ALL MODELS]
# =============================================================================

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_cancelled']       = (df['booking_status'] == 'Cancelled').astype(int)
    df['traffic_enc']        = df['traffic_level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['weather_enc']        = df['weather_condition'].map({'Clear': 0, 'Rain': 1, 'Heavy Rain': 2})
    if 'demand_level' in df.columns:
        df['demand_enc']     = df['demand_level'].map({'Low': 0, 'Medium': 1, 'High': 2})

    df['booking_status_enc'] = df['booking_status'].map(
        {'Completed': 0, 'Cancelled': 1, 'Incomplete': 2}
    )

    df = pd.get_dummies(df, columns=['incomplete_ride_reason'], prefix='reason', drop_first=True)

    ohe_cols = [
        'customer_city', 'driver_city', 'preferred_vehicle_type',
        'season', 'day_of_week', 'vehicle_type', 'city', 'customer_gender'
    ]
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

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

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    print("✅ Encoding done.")
    return df


# =============================================================================
# UC1 INTERACTION FEATURES (post-encoding)
# =============================================================================

def add_new_uc1_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Called AFTER encode_features so demand_enc is available."""
    if 'demand_enc' in df.columns and 'is_unreliable_driver' in df.columns:
        df['demand_x_unreliable']      = df['is_unreliable_driver'] * df['demand_enc']
    if 'demand_enc' in df.columns and 'acceptance_rate' in df.columns:
        df['demand_x_low_acceptance']  = df['demand_enc'] * (1 - df['acceptance_rate'])
    if ('demand_enc' in df.columns and 'driver_incomplete_rides' in df.columns
            and 'total_assigned_rides' in df.columns):
        df['demand_x_incomplete_rate'] = (
            df['demand_enc'] * df['driver_incomplete_rides'] / (df['total_assigned_rides'] + 1)
        )
    if all(c in df.columns for c in ['ride_distance_km', 'is_night_ride', 'weather_enc']):
        df['ride_difficulty']    = df['ride_distance_km'] * df['is_night_ride'] * df['weather_enc']
    if 'ride_distance_km' in df.columns and 'traffic_enc' in df.columns:
        df['distance_x_traffic'] = df['ride_distance_km'] * df['traffic_enc']
    if 'driver_experience_years' in df.columns and 'ride_distance_km' in df.columns:
        df['exp_x_distance']     = df['driver_experience_years'] * df['ride_distance_km']
    if 'avg_driver_rating' in df.columns and 'estimated_ride_time_min' in df.columns:
        df['rating_x_ridetime']  = df['avg_driver_rating'] * df['estimated_ride_time_min']
    if 'is_low_rated_customer' in df.columns and 'is_low_rated_driver' in df.columns:
        df['dual_low_rating']    = df['is_low_rated_customer'] * df['is_low_rated_driver']
    if 'is_new_customer' in df.columns and 'is_unreliable_driver' in df.columns:
        df['new_cust_x_unreliable'] = df['is_new_customer'] * df['is_unreliable_driver']
    if 'estimated_ride_time_min' in df.columns and 'avg_wait_time_min' in df.columns:
        df['time_pressure']      = df['estimated_ride_time_min'] / (df['avg_wait_time_min'] + 1)
    if 'peak_time_flag' in df.columns and 'ride_distance_km' in df.columns:
        df['peak_x_distance']    = df['peak_time_flag'] * df['ride_distance_km']

    print("✅ New UC1 interaction features added.")
    return df


# =============================================================================
# LOG TRANSFORM  [UC2 target + skewed inputs]
# =============================================================================

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    log_cols = [
        'base_fare', 'ride_distance_km', 'total_bookings',
        'total_requests', 'estimated_ride_time_min'
    ]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    if 'booking_value' in df.columns:
        df['booking_value_log'] = np.log1p(df['booking_value'].clip(lower=0))

    print("✅ Log transforms applied.")
    return df


# =============================================================================
# DROP RAW / REDUNDANT COLUMNS
# =============================================================================

def drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        'booking_id', 'customer_id', 'driver_id', 'actual_ride_time_min',
        'booking_datetime', 'actual_time_available', 'merge_hour',
        'pickup_location', 'drop_location',
        'traffic_level', 'weather_condition', 'demand_level',
        # Matches old pipeline exactly:
        # - booking_status, completed_rides, cust_completion_rate, total_bookings dropped here
        # - cancelled_rides, incomplete_rides intentionally kept — Zone 2 UC1 leakage drops them
        'booking_status', 'completed_rides', 'cust_completion_rate', 'total_bookings',
        'driver_experience_bin', 'driver_acceptance_tier',
        'customer_age_bin', 'wait_time_bin',
        'traffic_num', 'weather_num',
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"✅ Dropped {len(drop_cols)} raw/redundant columns.")
    return df