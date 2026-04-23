# =============================================================================
# run_feature_engineering.py  — Stage 1
# =============================================================================
# Zone 1 → Zone 2+3 → Feature Selection → save splits to disk.
#
# Run:  python run_feature_engineering.py
# =============================================================================

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.append('src')

import joblib
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader                    import load_cleaned_data
from src.feature_engineering            import run_zone1_engineering, get_splits

os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('splits',  exist_ok=True)

# =============================================================================
# STEP 1 — Load
# =============================================================================
bookings_df, customers_df, drivers_df, location_demand_df, time_features_df = (
    load_cleaned_data()
)

# =============================================================================
# STEP 2 — Zone 1
# =============================================================================
df = run_zone1_engineering(
    bookings_df, customers_df, drivers_df,
    location_demand_df, time_features_df,
)

# =============================================================================
# STEP 3 — Re-attach vehicle_type for UC2 per-vehicle splitting
# Zone 1 OHE-encodes and drops vehicle_type; UC2 needs the raw label to subset.
# All other UCs must NOT see this column — it would add a spurious extra feature.
# =============================================================================
df_uc2 = df.copy()
df_uc2['vehicle_type'] = bookings_df['vehicle_type'].values

# df (without vehicle_type) is used for UC1 / UC3 / UC4
# df_uc2 (with vehicle_type) is used only for UC2

# =============================================================================
# STEP 4 — Zone 2 + 3 + Feature Selection per use case
# =============================================================================

# UC1 — Ride Outcome (multi-class)
X_train_uc1, X_test_uc1, y_train_uc1, y_test_uc1 = get_splits(
    df.copy(), 'UC1',
    run_corr_filter=True, run_shap_filter=True, shap_top_n=40,
)

# UC2 — Fare Prediction (per vehicle) — uses df_uc2 which has vehicle_type
uc2_splits = get_splits(
    df_uc2.copy(), 'UC2',
    run_corr_filter=True, run_shap_filter=True, shap_top_n=35,
)
X_train_cab,  X_test_cab,  y_train_cab,  y_test_cab  = uc2_splits['Cab']
X_train_auto, X_test_auto, y_train_auto, y_test_auto = uc2_splits['Auto']
X_train_bike, X_test_bike, y_train_bike, y_test_bike = uc2_splits['Bike']

# UC3 — Customer Cancellation Risk (binary)
X_train_uc3, X_test_uc3, y_train_uc3, y_test_uc3 = get_splits(
    df.copy(), 'UC3',
    run_corr_filter=True, run_shap_filter=True, shap_top_n=35,
)

# UC4 — Driver Delay Prediction (binary)
X_train_uc4, X_test_uc4, y_train_uc4, y_test_uc4 = get_splits(
    df.copy(), 'UC4',
    run_corr_filter=True, run_shap_filter=True, shap_top_n=30,
)

# =============================================================================
# STEP 5 — Sanity assertions
# =============================================================================
assert 'booking_status_enc'   not in X_train_uc1.columns,  "UC1 target leaked!"
assert 'booking_value_log'    not in X_train_cab.columns,  "UC2-Cab target leaked!"
assert 'booking_value_log'    not in X_train_auto.columns, "UC2-Auto target leaked!"
assert 'booking_value_log'    not in X_train_bike.columns, "UC2-Bike target leaked!"
assert 'is_cancelled'         not in X_train_uc3.columns,  "UC3 target leaked!"
assert 'driver_delay_flag'    not in X_train_uc4.columns,  "UC4 target leaked!"
print("\n✅ No target leakage")

assert y_train_uc1.isna().sum() == 0
assert y_train_cab.isna().sum() == 0
assert y_train_uc3.isna().sum() == 0
assert y_train_uc4.isna().sum() == 0
print("✅ No null targets")

# =============================================================================
# STEP 6 — Save splits for downstream stages
# =============================================================================
joblib.dump((X_train_uc1, X_test_uc1, y_train_uc1, y_test_uc1), 'splits/uc1.pkl')
joblib.dump((X_train_cab,  X_test_cab,  y_train_cab,  y_test_cab),  'splits/uc2_cab.pkl')
joblib.dump((X_train_auto, X_test_auto, y_train_auto, y_test_auto), 'splits/uc2_auto.pkl')
joblib.dump((X_train_bike, X_test_bike, y_train_bike, y_test_bike), 'splits/uc2_bike.pkl')
joblib.dump((X_train_uc3, X_test_uc3, y_train_uc3, y_test_uc3), 'splits/uc3.pkl')
joblib.dump((X_train_uc4, X_test_uc4, y_train_uc4, y_test_uc4), 'splits/uc4.pkl')
print("✅ Splits saved to splits/")

# =============================================================================
# STEP 7 — Summary
# =============================================================================
print("\nClass balance (train):")
print("  UC1:", y_train_uc1.value_counts(normalize=True).round(3).to_dict())
print("  UC3:", y_train_uc3.value_counts(normalize=True).round(3).to_dict())
print("  UC4:", y_train_uc4.value_counts(normalize=True).round(3).to_dict())

print("\nFeature counts:")
for label, X in [('UC1', X_train_uc1), ('UC2-Cab', X_train_cab),
                  ('UC2-Auto', X_train_auto), ('UC2-Bike', X_train_bike),
                  ('UC3', X_train_uc3), ('UC4', X_train_uc4)]:
    print(f"  {label:<10}: {X.shape[1]} features  |  {len(X):,} train rows")

print("\n✅ Stage 1 complete. Run run_training.py next.")