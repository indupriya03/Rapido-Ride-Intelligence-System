# =============================================================================
# INSERT PREDICTIONS — Rapido Ride Intelligence System
# =============================================================================
# Runs UC2 (fare) and UC3 (cancellation) models on all bookings
# and inserts results into the predictions table.
#
# Run AFTER:
#   1. create_database.py  (tables exist + data loaded)
#   2. model_training.py   (models saved to models/)
#
# Usage:
#   python insert_predictions.py
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from sqlalchemy import create_engine, text
from datetime import datetime
sys.path.append("app/utils")
from db import get_engine, test_connection

sys.path.append("src")
from data_loader import load_cleaned_data
from feature_engineering import run_zone1_engineering, get_splits

# =============================================================================
# CONFIG
# =============================================================================
DB_URL       = "mysql+pymysql://root:your_password@localhost/rapido_db"
MODEL_DIR    = "models"
MODEL_VERSION = "v1.0"

# =============================================================================
# STEP 1 — Load models and thresholds
# =============================================================================
def load_models():
    print("Loading models...")
    uc2_model = joblib.load(f"{MODEL_DIR}/uc2_fare_model.pkl")
    uc3_model = joblib.load(f"{MODEL_DIR}/uc3_cancel_model.pkl")

    with open(f"{MODEL_DIR}/thresholds.json") as f:
        thresholds = json.load(f)

    uc3_threshold = thresholds['uc3_threshold']
    print(f"  ✅ UC2 fare model loaded.")
    print(f"  ✅ UC3 cancel model loaded. Threshold: {uc3_threshold:.3f}")
    return uc2_model, uc3_model, uc3_threshold


# =============================================================================
# STEP 2 — Prepare features (same pipeline as training)
# =============================================================================
def prepare_features():
    print("\nRunning feature engineering pipeline...")
    bookings_df, customers_df, drivers_df, location_demand_df, time_features_df = load_cleaned_data()

    # ── Fix booking_id to match database (BIGINT) ──
    bookings_df['booking_id'] = bookings_df['booking_id'].str.extract(r'(\d+)').astype('int64')
    bookings_df['actual_ride_time_min'] = bookings_df['actual_ride_time_min'].fillna(0)
    bookings_df.loc[bookings_df['booking_status'] != 'Completed', 'actual_ride_time_min'] = 0

    df = run_zone1_engineering(
        bookings_df, customers_df, drivers_df,
        location_demand_df, time_features_df
    )

    # ── Save booking_ids BEFORE split using the same random_state=42 ──
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=42
    )
    all_idx = list(idx_train) + list(idx_test)

    # ── Align BOTH df and bookings_df to same order BEFORE anything else ──
    df               = df.loc[all_idx].reset_index(drop=True)
    bookings_aligned = bookings_df.loc[all_idx].reset_index(drop=True)
    booking_ids      = bookings_aligned['booking_id'].copy()

    # ── Get UC2 and UC3 splits on the already-aligned df ──
    X_train_uc2, X_test_uc2, y_train_uc2, y_test_uc2 = get_splits(df.copy(), 'UC2')
    X_train_uc3, X_test_uc3, y_train_uc3, y_test_uc3 = get_splits(df.copy(), 'UC3')

    # ── Combine train + test, reset index so .values lines up ──
    X_uc2 = pd.concat([X_train_uc2, X_test_uc2], axis=0).reset_index(drop=True)
    X_uc3 = pd.concat([X_train_uc3, X_test_uc3], axis=0).reset_index(drop=True)

    # ── Sanity check before returning ──
    assert len(booking_ids) == len(X_uc2) == len(X_uc3) == len(bookings_aligned), \
        f"Length mismatch: booking_ids={len(booking_ids)}, X_uc2={len(X_uc2)}, " \
        f"X_uc3={len(X_uc3)}, bookings_aligned={len(bookings_aligned)}"

    print(f"  ✅ booking_ids       : {len(booking_ids)}")
    print(f"  ✅ UC2 features      : {X_uc2.shape}")
    print(f"  ✅ UC3 features      : {X_uc3.shape}")
    print(f"  ✅ bookings_aligned  : {bookings_aligned.shape}")

    return X_uc2, X_uc3, booking_ids, bookings_aligned, df


# =============================================================================
# STEP 3 — Run predictions
# =============================================================================
def run_predictions(uc2_model, uc3_model, uc3_threshold,
                    X_uc2, X_uc3):
    print("\nRunning predictions...")

    # Clean column names (XGBoost dislikes special chars)
    def clean_cols(X):
        X = X.copy()
        X.columns = [c.replace('[','_').replace(']','_')
                      .replace('<','_').replace('>','_')
                      .replace(' ','_').replace(',','_')
                      for c in X.columns]
        # Drop non-numeric
        bad = X.select_dtypes(include=['object','category']).columns.tolist()
        if bad:
            X = X.drop(columns=bad)
        # Fill nulls
        X = X.fillna(X.median())
        return X

    X_uc2_clean = clean_cols(X_uc2)
    X_uc3_clean = clean_cols(X_uc3)

    # UC2 — fare prediction (log scale, back-transform)
    fare_log_pred = uc2_model.predict(X_uc2_clean)
    fare_pred     = np.expm1(fare_log_pred)

    # UC3 — cancellation probability
    cancel_proba  = uc3_model.predict_proba(X_uc3_clean)[:, 1]
    cancel_flag   = (cancel_proba >= uc3_threshold).astype(int)

    # Risk tier based on probability
    def risk_tier(p):
        if p >= 0.7:   return 'High'
        elif p >= 0.4: return 'Medium'
        else:          return 'Low'

    cancel_tiers = [risk_tier(p) for p in cancel_proba]

    # Recommended action
    def recommended_action(tier):
        if tier == 'High':   return 'Reassign Driver'
        elif tier == 'Medium': return 'Send Reminder'
        else:                return 'Proceed'

    actions = [recommended_action(t) for t in cancel_tiers]

    print(f"  ✅ Fare predictions: min={fare_pred.min():.2f}, "
          f"max={fare_pred.max():.2f}, mean={fare_pred.mean():.2f}")
    print(f"  ✅ Cancel predictions: "
          f"{cancel_flag.sum():,} high-risk out of {len(cancel_flag):,}")

    return fare_pred, cancel_proba, cancel_tiers, actions


# =============================================================================
# STEP 4 — Build predictions dataframe
# =============================================================================
def build_predictions_df(booking_ids, bookings_aligned, df,
                          fare_pred, cancel_proba, cancel_tiers,
                          actions, uc3_threshold):

    predictions_df = pd.DataFrame({
        'booking_id'             : booking_ids,
        'predicted_at'           : datetime.now(),
        'model_version'          : MODEL_VERSION,
        'prediction_type'        : 'ride_inference',
        'city'                   : bookings_aligned['city'].values,
        'vehicle_type'           : bookings_aligned['vehicle_type'].values,
        'hour_of_day'            : bookings_aligned['hour_of_day'].values,
        'ride_distance_km'       : bookings_aligned['ride_distance_km'].values,
        'surge_multiplier'       : bookings_aligned['surge_multiplier'].values,
        'is_weekend'             : bookings_aligned['is_weekend'].values,
        'demand_level'           : df['demand_level'].values if 'demand_level' in df.columns else None,
        'peak_time_flag'         : df['peak_time_flag'].values if 'peak_time_flag' in df.columns else None,
        'predicted_fare'         : fare_pred.round(2),
        'predicted_ride_time_min': bookings_aligned['estimated_ride_time_min'].values,
        'cancel_probability'     : cancel_proba.round(4),
        'cancel_risk_tier'       : cancel_tiers,
        'uc3_threshold_used'     : uc3_threshold,
        'recommended_action'     : actions,
        'actual_completed_flag'  : (bookings_aligned['booking_status'] == 'Completed').astype(int).values,
        'actual_fare'            : bookings_aligned['booking_value'].values,
        'actual_ride_time_min'   : bookings_aligned['actual_ride_time_min'].values,
        'actual_cancelled_flag'  : (bookings_aligned['booking_status'] == 'Cancelled').astype(int).values,
    })

    print(f"\n  ✅ Predictions dataframe: {predictions_df.shape}")
    return predictions_df


# =============================================================================
# STEP 5 —  insert predictions into MySQL
# =============================================================================
def insert_predictions(predictions_df):
    print("\nInserting into MySQL...")
    engine = get_engine()

    # ── Clear existing predictions before re-inserting ──
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE predictions"))
        conn.commit()
    print("  ✅ Existing predictions cleared.")

    # ── Clean up types before insert ──
    float_cols = ['ride_distance_km', 'surge_multiplier', 'predicted_fare',
                  'predicted_ride_time_min', 'cancel_probability',
                  'uc3_threshold_used', 'actual_fare', 'actual_ride_time_min']
    for col in float_cols:
        predictions_df[col] = predictions_df[col].astype('float64').round(4)

    bool_cols = ['is_weekend', 'peak_time_flag',
                 'actual_completed_flag', 'actual_cancelled_flag']
    for col in bool_cols:
        predictions_df[col] = predictions_df[col].fillna(0).astype(int)

    predictions_df['demand_level'] = predictions_df['demand_level'].fillna('Unknown')
    predictions_df['hour_of_day']  = predictions_df['hour_of_day'].fillna(0).astype(int)

    predictions_df.to_sql(
        name      = 'predictions',
        con       = engine,
        if_exists = 'append',
        index     = False,
        chunksize = 1000,
        method    = None,
    )
    print(f"  ✅ {len(predictions_df):,} predictions inserted.")

    # Verify
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
        high  = conn.execute(text("SELECT COUNT(*) FROM predictions WHERE cancel_risk_tier='High'")).scalar()
        med   = conn.execute(text("SELECT COUNT(*) FROM predictions WHERE cancel_risk_tier='Medium'")).scalar()
        low   = conn.execute(text("SELECT COUNT(*) FROM predictions WHERE cancel_risk_tier='Low'")).scalar()

    print(f"\n  Predictions summary:")
    print(f"    Total     : {count:,}")
    print(f"    High risk : {high:,}")
    print(f"    Med risk  : {med:,}")
    print(f"    Low risk  : {low:,}")
    engine.dispose()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("="*55)
    print("Rapido — Insert Predictions")
    print("="*55)

    uc2_model, uc3_model, uc3_threshold = load_models()

    X_uc2, X_uc3, booking_ids, bookings_aligned, df = prepare_features()

    fare_pred, cancel_proba, cancel_tiers, actions = run_predictions(
        uc2_model, uc3_model, uc3_threshold,
        X_uc2, X_uc3
    )

    predictions_df = build_predictions_df(
        booking_ids, bookings_aligned, df,
        fare_pred, cancel_proba, cancel_tiers,
        actions, uc3_threshold
    )

    insert_predictions(predictions_df)

    print("\n✅ Done. predictions table is ready for Streamlit.")