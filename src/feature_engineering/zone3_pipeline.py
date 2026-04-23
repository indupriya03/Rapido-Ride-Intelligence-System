# =============================================================================
# src/feature_engineering/zone3_pipeline.py
# =============================================================================
# Post-split feature engineering.
# ALL statistics (quantiles, frequency maps, scaler) are fit on X_train ONLY
# and applied identically to X_test.
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

SCALE_COLS = [
    'surge_multiplier', 'estimated_ride_time_min',
    'acceptance_rate', 'delay_rate', 'avg_pickup_delay_min',
    'customer_age', 'customer_tenure_years', 'customer_signup_days_ago',
    'fare_per_km', 'fare_per_min', 'demand_supply_ratio',
    'cancel_to_booking_ratio', 'cust_completion_rate',
    'rejection_rate', 'delay_per_ride', 'driver_incomplete_rate',
    'location_surge_deviation', 'surge_x_distance',
    'traffic_x_ridetime', 'ride_distance_km', 'base_fare',
]


def apply_zone3_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_case: str,
    model_dir: str = 'models',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply all post-split, data-driven transformations.

    Parameters
    ----------
    X_train   : training features (post Zone 2 split)
    X_test    : test features
    use_case  : 'UC1' | 'UC2_Cab' | 'UC2_Auto' | 'UC2_Bike' | 'UC3' | 'UC4'
    model_dir : directory to save the fitted scaler

    Returns
    -------
    X_train, X_test  (both transformed in-place copies)
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    # ── 3A: Cap driver experience at (driver_age - 18) ───────────────────────
    if 'driver_experience_years' in X_train.columns:
        for df_ in [X_train, X_test]:
            max_valid_exp = df_['driver_age'] - 18
            df_['driver_experience_years'] = df_['driver_experience_years'].clip(
                lower=0, upper=max_valid_exp
            )
        print("  ✅ driver_experience_years capped at (driver_age - 18).")

    # ── 3B: Quantile ride-distance flags  [UC1, UC2, UC3] ────────────────────
    base_uc = use_case.split('_')[0]
    if 'ride_distance_km' in X_train.columns and base_uc in ['UC1', 'UC2', 'UC3']:
        short_thresh = X_train['ride_distance_km'].quantile(0.25)
        long_thresh  = X_train['ride_distance_km'].quantile(0.75)

        for df_ in [X_train, X_test]:
            df_['is_short_ride']     = (df_['ride_distance_km'] < short_thresh).astype(int)
            df_['is_long_ride']      = (df_['ride_distance_km'] > long_thresh).astype(int)
            df_['distance_bin_enc']  = pd.cut(
                df_['ride_distance_km'],
                bins=[-np.inf, short_thresh, long_thresh, np.inf],
                labels=[0, 1, 2]
            ).astype(float)
        print("  ✅ Quantile distance flags applied (train thresholds → test).")

    # ── 3C: Quantile high-cancel flag  [UC1] ─────────────────────────────────
    if 'cancellation_rate' in X_train.columns and use_case == 'UC1':
        cancel_thresh = X_train['cancellation_rate'].quantile(0.75)
        for df_ in [X_train, X_test]:
            df_['is_high_cancel_customer'] = (
                df_['cancellation_rate'] > cancel_thresh
            ).astype(int)
            df_['new_high_cancel'] = (
                (df_['is_new_customer'] == 1) &
                (df_['is_high_cancel_customer'] == 1)
            ).astype(int)
        print("  ✅ High cancel flag applied (train threshold → test).")

    # ── 3D: Frequency encoding — pickup & drop location  [UC1, UC2*] ─────────
    for col in ['pickup_location', 'drop_location']:
        if col in X_train.columns and base_uc in ['UC1', 'UC2']:
            freq_map = X_train[col].value_counts(normalize=True)
            X_train[col + '_freq'] = X_train[col].map(freq_map)
            X_test[col  + '_freq'] = X_test[col].map(freq_map).fillna(0)

    if base_uc in ['UC1', 'UC2']:
        print("  ✅ Frequency encoding applied (train frequencies → test).")

    # ── 3E: Pickup hotspot flag & City_Pair freq  [UC1, UC2*] ────────────────
    if 'pickup_location' in X_train.columns and base_uc in ['UC1', 'UC2']:
        top_locs = X_train['pickup_location'].value_counts().head(10).index
        X_train['pickup_hotspot_flag'] = X_train['pickup_location'].isin(top_locs).astype(int)
        X_test['pickup_hotspot_flag']  = X_test['pickup_location'].isin(top_locs).astype(int)
        print("  ✅ Pickup hotspot flag applied (train top-10 → test).")

    if 'City_Pair' in X_train.columns and base_uc in ['UC1', 'UC2']:
        freq_map = X_train['City_Pair'].value_counts(normalize=True)
        X_train['City_Pair_freq'] = X_train['City_Pair'].map(freq_map)
        X_test['City_Pair_freq']  = X_test['City_Pair'].map(freq_map).fillna(0)

    # ── 3F: Customer Loyalty Score ────────────────────────────────────────────
    if 'cancellation_rate' in X_train.columns and use_case == 'UC1':
        max_tenure = X_train['customer_signup_days_ago'].max() / 365
        for df_ in [X_train, X_test]:
            tenure_norm     = (df_['customer_signup_days_ago'] / 365) / (max_tenure + 1e-9)
            completion_rate = df_['completed_rides'] / (df_['total_bookings'] + 1)
            rating_norm     = df_['avg_customer_rating'] / 5
            low_cancel      = 1 - df_['cancellation_rate'].clip(0, 1)
            df_['Customer_Loyalty_Score'] = (
                tenure_norm * 0.20 + completion_rate * 0.40 +
                rating_norm * 0.20 + low_cancel * 0.20
            ).clip(0, 1)
            df_['loyalty_x_cancel'] = df_['Customer_Loyalty_Score'] * df_['cancellation_rate']
        print("  ✅ Customer Loyalty Score (full) applied.")

    elif use_case == 'UC3':
        max_tenure = X_train['customer_signup_days_ago'].max() / 365
        for df_ in [X_train, X_test]:
            tenure_norm = (df_['customer_signup_days_ago'] / 365) / (max_tenure + 1e-9)
            rating_norm = df_['avg_customer_rating'] / 5
            df_['Customer_Loyalty_Score'] = (
                tenure_norm * 0.50 + rating_norm * 0.50
            ).clip(0, 1)
        print("  ✅ Customer Loyalty Score (tenure + rating only) applied for UC3.")

    # ── 3G: StandardScaler on numeric columns ─────────────────────────────────
    cols_to_scale = [c for c in SCALE_COLS if c in X_train.columns]
    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    scaler_path = os.path.join(model_dir, f'scaler_{use_case.lower()}.pkl')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ StandardScaler fitted on {len(cols_to_scale)} cols → {scaler_path}")

    return X_train, X_test
