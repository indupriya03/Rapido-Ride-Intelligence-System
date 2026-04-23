# =============================================================================
# src/inference/preprocessor.py
# =============================================================================
# Transforms a single raw booking row into model-ready features at inference.
#
# Pipeline mirrors training exactly:
#   1. Zone 1  — feature derivation (same functions as training)
#   2. Zone 2  — drop leakage columns (same lists as training)
#   3. Zone 3  — apply fitted artifacts: scaler, freq maps, quantile thresholds
#   4. Align   — reindex to the exact columns the model was trained on
#                (loaded from models/feature_cols_{use_case}.json)
#
# feature_cols_{use_case}.json is written by run_training.py → save_feature_cols().
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib

from ..feature_engineering.zone1_features import (
    add_datetime_features, add_ride_features, add_customer_features,
    add_driver_features, add_location_features, add_interaction_features,
    encode_features, add_new_uc1_interactions, log_transform, drop_raw_columns,
)
from ..feature_engineering.zone2_config import LEAKAGE_MAP, FEATURE_SELECT_MAP
from ..modeling.model_io import load_feature_cols

MODEL_DIR = 'models'

TARGET_MAP = {
    'UC1': 'booking_status_enc',
    'UC2': 'booking_value_log',
    'UC3': 'is_cancelled',
    'UC4': 'driver_delay_flag',
}


def preprocess_row(
    row: dict,
    use_case: str,
    model_dir: str = MODEL_DIR,
) -> pd.DataFrame:
    """
    Transform a single raw booking dict into a model-ready feature DataFrame.

    Parameters
    ----------
    row       : dict — raw pre-joined fields (bookings + customers + drivers +
                location + time), same schema used at training time
    use_case  : 'UC1' | 'UC2_Cab' | 'UC2_Auto' | 'UC2_Bike' | 'UC3' | 'UC4'
    model_dir : root models directory

    Returns
    -------
    X : pd.DataFrame — 1 row, columns aligned exactly to model training columns
    """
    df = pd.DataFrame([row])

    # ── Zone 1: feature derivation ────────────────────────────────────────────
    df = add_datetime_features(df)
    df = add_ride_features(df)
    df = add_customer_features(df)
    df = add_driver_features(df)
    df = add_location_features(df)
    df = add_interaction_features(df)
    df = encode_features(df)
    df = add_new_uc1_interactions(df)
    df = log_transform(df)
    df = drop_raw_columns(df)

    # ── Zone 2: drop leakage + feature-select columns ─────────────────────────
    base_uc       = use_case.split('_')[0]
    leakage_cols  = list(LEAKAGE_MAP.get(base_uc, []))
    feat_sel_cols = list(FEATURE_SELECT_MAP.get(base_uc, []))
    target_col    = TARGET_MAP.get(base_uc, '')

    if base_uc == 'UC1':
        reason_cols  = [c for c in df.columns if c.startswith('reason_')]
        leakage_cols = leakage_cols + reason_cols

    cols_to_drop = [
        c for c in leakage_cols + feat_sel_cols
        if c in df.columns and c != target_col
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    if base_uc == 'UC2':
        vtype_ohe = [c for c in df.columns if c.startswith('vehicle_type_')]
        df = df.drop(columns=vtype_ohe, errors='ignore')

    # ── Zone 3: apply fitted artifacts ────────────────────────────────────────
    uc_dir = os.path.join(model_dir, use_case.lower())

    # 3A: Cap driver experience
    if 'driver_experience_years' in df.columns and 'driver_age' in df.columns:
        df['driver_experience_years'] = df['driver_experience_years'].clip(
            lower=0, upper=(df['driver_age'] - 18)
        )

    # 3B: Distance quantile flags
    dist_path = os.path.join(uc_dir, 'distance_thresholds.pkl')
    if os.path.exists(dist_path) and 'ride_distance_km' in df.columns:
        short_thresh, long_thresh = joblib.load(dist_path)
        df['is_short_ride']    = (df['ride_distance_km'] < short_thresh).astype(int)
        df['is_long_ride']     = (df['ride_distance_km'] > long_thresh).astype(int)
        df['distance_bin_enc'] = pd.cut(
            df['ride_distance_km'],
            bins=[-np.inf, short_thresh, long_thresh, np.inf],
            labels=[0, 1, 2]
        ).astype(float)

    # 3C: High-cancel flag (UC1)
    cancel_thresh_path = os.path.join(uc_dir, 'cancel_threshold.pkl')
    if os.path.exists(cancel_thresh_path) and 'cancellation_rate' in df.columns:
        cancel_thresh = joblib.load(cancel_thresh_path)
        df['is_high_cancel_customer'] = (df['cancellation_rate'] > cancel_thresh).astype(int)
        if 'is_new_customer' in df.columns:
            df['new_high_cancel'] = (
                (df['is_new_customer'] == 1) & (df['is_high_cancel_customer'] == 1)
            ).astype(int)

    # 3D: Frequency encoding
    for col in ['pickup_location', 'drop_location']:
        freq_path = os.path.join(uc_dir, f'freq_map_{col}.pkl')
        if os.path.exists(freq_path) and col in df.columns:
            freq_map = joblib.load(freq_path)
            df[col + '_freq'] = df[col].map(freq_map).fillna(0)

    # 3E: Pickup hotspot + City_Pair freq
    hotspot_path = os.path.join(uc_dir, 'pickup_hotspot_locs.pkl')
    if os.path.exists(hotspot_path) and 'pickup_location' in df.columns:
        top_locs = joblib.load(hotspot_path)
        df['pickup_hotspot_flag'] = df['pickup_location'].isin(top_locs).astype(int)

    city_pair_path = os.path.join(uc_dir, 'freq_map_City_Pair.pkl')
    if os.path.exists(city_pair_path) and 'City_Pair' in df.columns:
        freq_map = joblib.load(city_pair_path)
        df['City_Pair_freq'] = df['City_Pair'].map(freq_map).fillna(0)

    # 3F: Customer Loyalty Score (UC3 lean version)
    if use_case == 'UC3' and 'customer_signup_days_ago' in df.columns:
        max_tenure  = df['customer_signup_days_ago'].max() / 365
        tenure_norm = (df['customer_signup_days_ago'] / 365) / (max_tenure + 1e-9)
        rating_norm = df['avg_customer_rating'] / 5
        df['Customer_Loyalty_Score'] = (tenure_norm * 0.50 + rating_norm * 0.50).clip(0, 1)

    # 3G: StandardScaler — use scaler.feature_names_in_ to scale exactly the right cols
    scaler_path = os.path.join(uc_dir, f'scaler_{use_case.lower()}.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            f"Run run_feature_engineering.py first."
        )
    scaler      = joblib.load(scaler_path)
    scaler_cols = list(scaler.feature_names_in_)

    # Add any scaler cols missing from this row (e.g. derived cols that were NaN-dropped)
    for c in scaler_cols:
        if c not in df.columns:
            df[c] = 0.0

    df[scaler_cols] = scaler.transform(df[scaler_cols])

    # ── Align to exact training columns ───────────────────────────────────────
    feature_cols = load_feature_cols(use_case, model_dir=model_dir)

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    return X