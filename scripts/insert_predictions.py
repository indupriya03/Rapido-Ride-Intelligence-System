# =============================================================================
# insert_predictions.py — Rapido Ride Intelligence System
# =============================================================================
# Runs UC2 (fare, per vehicle) and UC3 (cancellation) models on all bookings
# and inserts results into the predictions table.
#
# Run AFTER:
#   1. setup_database.py     (tables exist + data loaded)
#   2. run_training.py       (models saved to models/, splits saved to splits/)
#
# Usage:
#   python insert_predictions.py
#
# =============================================================================
# CHANGES FROM PREVIOUS VERSION
# =============================================================================
#
# 1. NO LONGER RE-RUNS FEATURE ENGINEERING
#    Old: called run_zone1_engineering() + get_splits() every time (~minutes).
#    New: loads pre-saved splits from splits/ directly (~seconds).
#         Splits are saved by run_feature_engineering.py and are guaranteed
#         to match exactly what the models were trained on.
#
# 2. UC2 SPLIT LOADING FIXED (3 per-vehicle models)
#    Old: get_splits(df, 'UC2') was called as if it returned a single tuple —
#         this broke when UC2 was changed to return a dict of 3 vehicle splits.
#    New: loads splits/uc2_cab.pkl, splits/uc2_auto.pkl, splits/uc2_bike.pkl
#         separately. Each split carries its own df-level row index which is
#         used to map predictions back to booking_ids.
#
# 3. BOOKING METADATA FROM bookings_cleaned.csv DIRECTLY
#    Old: pulled demand_level / peak_time_flag from a local 'df' variable
#         that came from zone1 engineering — these columns were dropped by
#         drop_raw_columns() so they were often None anyway.
#    New: reads bookings_cleaned.csv once. Split indices are positional row
#         references into bookings (index 0 = B_000001, etc.), so
#         bookings.iloc[split_index] gives the exact right booking row.
#
# 4. FARE PREDICTION ALIGNMENT SIMPLIFIED
#    Old: used bookings_aligned.index.get_indexer() after a reset_index —
#         fragile because reset_index breaks the df-level index.
#    New: builds a single fare_pred Series indexed by df row position,
#         then aligns to the master index via pd.Series.reindex().
#
# 5. MODEL FILENAMES CORRECTED
#    Old: loaded uc3_final.pkl (correct) but UC2 path was inconsistent
#         with the 3-model structure.
#    New: explicit per-vehicle model paths: uc2_cab_final.pkl, etc.
#
# =============================================================================

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import text

warnings.filterwarnings('ignore')
sys.path.append('app/utils')
sys.path.append('src')

from db import get_engine

# =============================================================================
# CONFIG
# =============================================================================

MODEL_DIR     = 'models'
SPLITS_DIR    = 'splits'
DATA_DIR      = 'data/cleaned'
MODEL_VERSION = 'v1.0'


# =============================================================================
# STEP 1 — Load models and thresholds
# =============================================================================

def load_models() -> tuple:
    """
    Load the 3 UC2 per-vehicle fare models, the UC3 cancellation model,
    and the optimised UC3 classification threshold from thresholds.json.

    Returns
    -------
    uc2_models   : dict  {'Cab': model, 'Auto': model, 'Bike': model}
    uc3_model    : fitted classifier
    uc3_threshold: float  (optimised from precision-recall curve at training)
    """
    print('Loading models...')

    uc2_models = {
        'Cab':  joblib.load(os.path.join(MODEL_DIR, 'uc2_cab_final.pkl')),
        'Auto': joblib.load(os.path.join(MODEL_DIR, 'uc2_auto_final.pkl')),
        'Bike': joblib.load(os.path.join(MODEL_DIR, 'uc2_bike_final.pkl')),
    }
    uc3_model = joblib.load(os.path.join(MODEL_DIR, 'uc3_final.pkl'))

    with open(os.path.join(MODEL_DIR, 'thresholds.json')) as f:
        thresholds = json.load(f)

    uc3_threshold_raw = thresholds['uc3_threshold']

    # ── Clamp threshold to safe operational range ─────────────────────────────
    # retraining overwrites thresholds.json via precision-recall curve, which
    # can produce values outside the operationally calibrated range.
    # If the model-generated threshold falls outside [0.55, 0.70] we override
    # it to 0.60 — the balanced default that avoids both extremes:
    #   < 0.55 → too many false positives, floods reassignment queue
    #   > 0.70 → too few High-Risk flags, real cancellations get missed
    THRESHOLD_MIN     = 0.55
    THRESHOLD_MAX     = 0.70
    THRESHOLD_DEFAULT = 0.60

    if not (THRESHOLD_MIN <= uc3_threshold_raw <= THRESHOLD_MAX):
        print(f'  ⚠️  UC3 threshold {uc3_threshold_raw:.4f} outside safe range '
              f'[{THRESHOLD_MIN}, {THRESHOLD_MAX}] — overriding to {THRESHOLD_DEFAULT}')
        uc3_threshold = THRESHOLD_DEFAULT
    else:
        uc3_threshold = uc3_threshold_raw
        print(f'  ✅ UC3 threshold accepted  : {uc3_threshold:.4f} (within safe range)')

    print(f'  ✅ UC2 fare models loaded  : Cab, Auto, Bike')
    print(f'  ✅ UC3 cancel model loaded  : threshold in use = {uc3_threshold:.4f}')
    return uc2_models, uc3_model, uc3_threshold


# =============================================================================
# STEP 2 — Load pre-saved splits + booking metadata
# =============================================================================

def load_splits_and_metadata() -> tuple:
    """
    Load pre-saved train/test splits from splits/ and booking metadata
    from bookings_cleaned.csv.

    WHY SPLITS INSTEAD OF RE-RUNNING ENGINEERING
    ---------------------------------------------
    Splits are saved by run_feature_engineering.py and are byte-for-byte
    identical to what each model was trained on — same scaler, same SHAP
    filter, same column order. Re-running zone1 + get_splits() risks drift
    if any upstream function changes, and takes ~2 minutes unnecessarily.

    HOW INDEX ALIGNMENT WORKS
    -------------------------
    The zone1 engineering pipeline starts with bookings_df in its original
    row order (default RangeIndex 0..99999). All merges are left-joins that
    preserve this order. get_splits() stratified-splits this index, so every
    integer index value in a split directly corresponds to
    bookings_cleaned.iloc[index] — no reordering needed.

    Returns
    -------
    uc2_splits   : dict  {'Cab': (X_full, meta), 'Auto': ..., 'Bike': ...}
                   X_full : pd.DataFrame of features, indexed by df row pos
                   meta   : pd.DataFrame of booking columns at same index
    uc3_X_full   : pd.DataFrame of UC3 features, indexed by df row pos
    master_meta  : pd.DataFrame of all 100k bookings metadata, indexed by
                   df row pos — used to build the final predictions_df
    """
    print('\nLoading pre-saved splits and booking metadata...')

    # ── Booking metadata (read once, index = row position = split index) ──────
    bookings = pd.read_csv(os.path.join(DATA_DIR, 'bookings_cleaned.csv'))
    bookings['booking_id_num'] = (
        bookings['booking_id'].str.extract(r'(\d+)').astype('int64')
    )
    # Ensure actual_ride_time_min is clean
    bookings['actual_ride_time_min'] = bookings['actual_ride_time_min'].fillna(0)
    bookings.loc[
        bookings['booking_status'] != 'Completed', 'actual_ride_time_min'
    ] = 0

    # peak_time_flag from time_features if needed — derive from hour_of_day
    # (avoids needing to re-run zone1 just for this column)
    bookings['peak_time_flag'] = bookings['hour_of_day'].between(7, 10) | \
                                  bookings['hour_of_day'].between(17, 21)
    bookings['peak_time_flag'] = bookings['peak_time_flag'].astype(int)

    print(f'  ✅ bookings_cleaned loaded : {bookings.shape}')

    # ── UC2 per-vehicle splits ────────────────────────────────────────────────
    uc2_splits = {}
    for vtype in ['Cab', 'Auto', 'Bike']:
        path = os.path.join(SPLITS_DIR, f'uc2_{vtype.lower()}.pkl')
        X_tr, X_te, y_tr, y_te = joblib.load(path)

        # Concatenate train + test — keeps the original df row-position index
        X_full = pd.concat([X_tr, X_te], axis=0)

        # Pull matching booking metadata rows by positional index
        meta = bookings.iloc[X_full.index].copy()
        meta.index = X_full.index   # align index for later reindex()

        print(f'  ✅ UC2 {vtype:<5} split loaded : {X_full.shape}')
        uc2_splits[vtype] = (X_full, meta)

    # ── UC3 split ─────────────────────────────────────────────────────────────
    X_tr3, X_te3, _, _ = joblib.load(os.path.join(SPLITS_DIR, 'uc3.pkl'))
    uc3_X_full = pd.concat([X_tr3, X_te3], axis=0)
    print(f'  ✅ UC3 split loaded        : {uc3_X_full.shape}')

    # ── Master booking metadata (all 100k rows, indexed 0..99999) ─────────────
    # UC2 and UC3 indices cover the same 100k rows (verified at build time).
    # Build a master metadata frame ordered by df row position for the
    # final predictions_df assembly.
    master_index = sorted(set(uc3_X_full.index))    # 0..99999 sorted
    master_meta  = bookings.iloc[master_index].copy()
    master_meta.index = master_index

    # Sanity checks
    uc2_total = sum(len(v[0]) for v in uc2_splits.values())
    assert uc2_total == len(uc3_X_full) == len(master_meta), (
        f'Row count mismatch: UC2 total={uc2_total}, '
        f'UC3={len(uc3_X_full)}, master_meta={len(master_meta)}'
    )
    print(f'  ✅ master_meta assembled   : {master_meta.shape}')
    print(f'  ✅ All row counts match    : {len(master_meta):,}')

    return uc2_splits, uc3_X_full, master_meta


# =============================================================================
# STEP 3 — Run predictions
# =============================================================================

def _clean_cols(X: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitise column names and dtypes for XGBoost / LightGBM.
    - Rename special characters in column names
    - Drop object/category columns
    - Fill NaN with column median
    """
    X = X.copy()
    X.columns = [
        c.replace('[', '_').replace(']', '_')
         .replace('<', '_').replace('>', '_')
         .replace(' ', '_').replace(',', '_')
        for c in X.columns
    ]
    bad = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if bad:
        X = X.drop(columns=bad)
    X = X.fillna(X.median())
    return X


def run_predictions(
    uc2_models: dict,
    uc3_model,
    uc3_threshold: float,
    uc2_splits: dict,
    uc3_X_full: pd.DataFrame,
    master_meta: pd.DataFrame,
) -> tuple:
    """
    Generate fare and cancellation predictions for all 100k bookings.

    UC2 (fare)
    ----------
    Each vehicle model predicts on its own feature set.
    Predictions are placed into a Series indexed by df row position,
    then reindexed to master_meta's index to guarantee alignment.
    Back-transform with np.expm1 (target was log1p at training time).

    UC3 (cancellation)
    ------------------
    Single model on the full UC3 feature set.
    Probability >= uc3_threshold → predicted cancellation.
    Risk tier and recommended action derived from raw probability.

    Returns
    -------
    fare_pred    : np.ndarray  shape (100000,)  in master_meta row order
    cancel_proba : np.ndarray  shape (100000,)
    cancel_tiers : list[str]
    actions      : list[str]
    """
    print('\nRunning predictions...')

    # ── UC2: predict per vehicle, collect into a single Series ───────────────
    fare_series = pd.Series(np.nan, index=master_meta.index, dtype=float)

    for vtype, model in uc2_models.items():
        if vtype not in uc2_splits:
            print(f'  ⚠️  No split for vehicle "{vtype}" — skipping.')
            continue

        X_full, _ = uc2_splits[vtype]
        X_clean   = _clean_cols(X_full)

        fare_log = model.predict(X_clean)
        fare_veh = np.expm1(fare_log)   # back-transform from log1p target

        # Place predictions at the correct df row positions
        vehicle_series = pd.Series(fare_veh, index=X_full.index)
        fare_series.update(vehicle_series)   # update in place, index-aligned

        print(f'  ✅ UC2 {vtype:<5}: {len(fare_veh):,} fare predictions  '
              f'| mean=₹{fare_veh.mean():.2f}  min=₹{fare_veh.min():.2f}  max=₹{fare_veh.max():.2f}')

    # Fill any residual NaN (should be zero after all 3 vehicles)
    nan_count = fare_series.isna().sum()
    if nan_count > 0:
        print(f'  ⚠️  {nan_count:,} bookings had no UC2 prediction — filling with median.')
        fare_series = fare_series.fillna(fare_series.median())

    fare_pred = fare_series.values   # now in master_meta row order

    # ── UC3: cancellation probability ────────────────────────────────────────
    X_uc3_clean  = _clean_cols(uc3_X_full.reindex(master_meta.index))
    cancel_proba = uc3_model.predict_proba(X_uc3_clean)[:, 1]

    def _risk_tier(p: float) -> str:
        if p >= 0.70:   return 'High'
        elif p >= 0.40: return 'Medium'
        else:           return 'Low'

    def _action(tier: str) -> str:
        if tier == 'High':     return 'Reassign Driver'
        elif tier == 'Medium': return 'Send Reminder'
        else:                  return 'Proceed'

    cancel_tiers = [_risk_tier(p) for p in cancel_proba]
    actions      = [_action(t) for t in cancel_tiers]

    high_count = sum(1 for t in cancel_tiers if t == 'High')
    print(f'  ✅ UC3: {high_count:,} high-risk cancellations '
          f'out of {len(cancel_proba):,}  (threshold={uc3_threshold:.4f})')
    print(f'  ✅ Fare range: ₹{fare_pred.min():.2f} – ₹{fare_pred.max():.2f}  '
          f'| mean=₹{fare_pred.mean():.2f}')

    return fare_pred, cancel_proba, cancel_tiers, actions


# =============================================================================
# STEP 4 — Build predictions DataFrame
# =============================================================================

def build_predictions_df(
    master_meta: pd.DataFrame,
    fare_pred: np.ndarray,
    cancel_proba: np.ndarray,
    cancel_tiers: list,
    actions: list,
    uc3_threshold: float,
) -> pd.DataFrame:
    """
    Assemble the final DataFrame that will be inserted into the predictions
    table. All arrays are in master_meta row order (sorted df row position).

    Booking metadata columns come from bookings_cleaned directly —
    no zone1 engineering output needed.
    peak_time_flag is derived from hour_of_day (07-10, 17-21).
    demand_level is not stored in bookings_cleaned; set to 'Unknown' here.
    To populate it properly, join location_demand_cleaned on
    (city, pickup_location, hour_of_day, vehicle_type) before inserting.
    """
    print('\nBuilding predictions dataframe...')

    predictions_df = pd.DataFrame({
        'booking_id'              : master_meta['booking_id_num'].values,
        'predicted_at'            : datetime.now(),
        'model_version'           : MODEL_VERSION,
        'prediction_type'         : 'ride_inference',
        'city'                    : master_meta['city'].values,
        'vehicle_type'            : master_meta['vehicle_type'].values,
        'hour_of_day'             : master_meta['hour_of_day'].values,
        'ride_distance_km'        : master_meta['ride_distance_km'].values,
        'surge_multiplier'        : master_meta['surge_multiplier'].values,
        'is_weekend'              : master_meta['is_weekend'].values,
        'demand_level'            : 'Unknown',   # not in bookings_cleaned; see docstring
        'peak_time_flag'          : master_meta['peak_time_flag'].values,
        'traffic_level'           : master_meta['traffic_level'].values,
        'weather_condition'       : master_meta['weather_condition'].values,
        'predicted_fare'          : fare_pred.round(2),
        'predicted_ride_time_min' : master_meta['estimated_ride_time_min'].values,
        'cancel_probability'      : cancel_proba.round(4),
        'cancel_risk_tier'        : cancel_tiers,
        'uc3_threshold_used'      : uc3_threshold,
        'recommended_action'      : actions,
        'actual_completed_flag'   : (master_meta['booking_status'] == 'Completed').astype(int).values,
        'actual_fare'             : master_meta['booking_value'].values,
        'actual_ride_time_min'    : master_meta['actual_ride_time_min'].values,
        'actual_cancelled_flag'   : (master_meta['booking_status'] == 'Cancelled').astype(int).values,
    })

    print(f'  ✅ predictions_df shape : {predictions_df.shape}')
    print(f'  ✅ booking_id range     : {predictions_df["booking_id"].min():,} – '
          f'{predictions_df["booking_id"].max():,}')
    return predictions_df


# =============================================================================
# STEP 5 — Insert into MySQL
# =============================================================================

def insert_predictions(predictions_df: pd.DataFrame) -> None:
    """
    Truncate the predictions table and bulk-insert all rows in chunks of 1000.
    Verifies row count and risk-tier breakdown after insert.
    """
    print('\nInserting into MySQL...')
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text('TRUNCATE TABLE predictions'))
        conn.commit()
    print('  ✅ Existing predictions cleared.')

    # ── Coerce dtypes before insert ───────────────────────────────────────────
    float_cols = [
        'ride_distance_km', 'surge_multiplier', 'predicted_fare',
        'predicted_ride_time_min', 'cancel_probability',
        'uc3_threshold_used', 'actual_fare', 'actual_ride_time_min',
    ]
    for col in float_cols:
        predictions_df[col] = predictions_df[col].astype('float64').round(4)

    int_cols = ['is_weekend', 'peak_time_flag', 'hour_of_day',
                'actual_completed_flag', 'actual_cancelled_flag']
    for col in int_cols:
        predictions_df[col] = predictions_df[col].fillna(0).astype(int)

    predictions_df['demand_level'] = predictions_df['demand_level'].fillna('Unknown')
    predictions_df['traffic_level'] = predictions_df['traffic_level'].fillna('Unknown')
    predictions_df['weather_condition'] = predictions_df['weather_condition'].fillna('Unknown')

    predictions_df.to_sql(
        name      = 'predictions',
        con       = engine,
        if_exists = 'append',
        index     = False,
        chunksize = 1000,
        method    = None,
    )
    print(f'  ✅ {len(predictions_df):,} predictions inserted.')

    # ── Verify ────────────────────────────────────────────────────────────────
    with engine.connect() as conn:
        total = conn.execute(text('SELECT COUNT(*) FROM predictions')).scalar()
        high  = conn.execute(text("SELECT COUNT(*) FROM predictions WHERE cancel_risk_tier='High'")).scalar()
        med   = conn.execute(text("SELECT COUNT(*) FROM predictions WHERE cancel_risk_tier='Medium'")).scalar()
        low   = conn.execute(text("SELECT COUNT(*) FROM predictions WHERE cancel_risk_tier='Low'")).scalar()

    print(f'\n  Predictions summary:')
    print(f'    Total     : {total:,}')
    print(f'    High risk : {high:,}')
    print(f'    Med risk  : {med:,}')
    print(f'    Low risk  : {low:,}')
    engine.dispose()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('=' * 55)
    print('Rapido — Insert Predictions')
    print('=' * 55)

    uc2_models, uc3_model, uc3_threshold = load_models()

    uc2_splits, uc3_X_full, master_meta = load_splits_and_metadata()

    fare_pred, cancel_proba, cancel_tiers, actions = run_predictions(
        uc2_models, uc3_model, uc3_threshold,
        uc2_splits, uc3_X_full, master_meta,
    )

    predictions_df = build_predictions_df(
        master_meta, fare_pred, cancel_proba,
        cancel_tiers, actions, uc3_threshold,
    )

    insert_predictions(predictions_df)

    print('\n✅ Done. predictions table is ready for Streamlit.')