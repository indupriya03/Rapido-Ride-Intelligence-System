# =============================================================================
# run_training.py  — Stage 2
# =============================================================================
# Load saved splits → train all baseline models → save baselines + thresholds.
#
# Run:  python run_training.py
# Requires: run_feature_engineering.py to have been run first.
# =============================================================================

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import joblib

from src.modeling.model_trainers  import train_uc1, train_uc2, train_uc3, train_uc4
print("current dir:", os.getcwd())
from src.modeling.postprocessing  import threshold_tuning, precision_targeted_threshold
from src.modeling.model_io        import save_thresholds, save_feature_cols
from src.modeling.model_utils     import clean_for_sklearn

os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# =============================================================================
# STEP 1 — Load splits
# =============================================================================
print("Loading splits...")
X_train_uc1, X_test_uc1, y_train_uc1, y_test_uc1 = joblib.load('splits/uc1.pkl')
X_train_cab,  X_test_cab,  y_train_cab,  y_test_cab  = joblib.load('splits/uc2_cab.pkl')
X_train_auto, X_test_auto, y_train_auto, y_test_auto = joblib.load('splits/uc2_auto.pkl')
X_train_bike, X_test_bike, y_train_bike, y_test_bike = joblib.load('splits/uc2_bike.pkl')
X_train_uc3, X_test_uc3, y_train_uc3, y_test_uc3 = joblib.load('splits/uc3.pkl')
X_train_uc4, X_test_uc4, y_train_uc4, y_test_uc4 = joblib.load('splits/uc4.pkl')
print("✅ Splits loaded.")

# =============================================================================
# STEP 2 — Train baselines
# =============================================================================

# UC1 — multi-class classification
uc1_model, X_tr1, X_te1, uc1_results = train_uc1(
    X_train_uc1, X_test_uc1, y_train_uc1, y_test_uc1,
    model_dir='models', output_dir='outputs',
)
save_feature_cols(X_tr1.columns.tolist(), 'UC1', model_dir='models')

# UC2 — regression, per vehicle
uc2_results = train_uc2(
    uc2_splits={
        'Cab' : (X_train_cab,  X_test_cab,  y_train_cab,  y_test_cab),
        'Auto': (X_train_auto, X_test_auto, y_train_auto, y_test_auto),
        'Bike': (X_train_bike, X_test_bike, y_train_bike, y_test_bike),
    },
    model_dir='models', output_dir='outputs',
)
for vtype, (_, X_tr_v, _, _) in uc2_results.items():
    save_feature_cols(X_tr_v.columns.tolist(), f'UC2_{vtype}', model_dir='models')

# UC3 — binary classification (cancellation risk)
uc3_model, X_tr3, X_te3, uc3_results = train_uc3(
    X_train_uc3, X_test_uc3, y_train_uc3, y_test_uc3,
    model_dir='models', output_dir='outputs',
)
save_feature_cols(X_tr3.columns.tolist(), 'UC3', model_dir='models')

# UC4 — binary classification (driver delay)
uc4_model, X_tr4, X_te4, uc4_results = train_uc4(
    X_train_uc4, X_test_uc4, y_train_uc4, y_test_uc4,
    model_dir='models', output_dir='outputs',
)
save_feature_cols(X_tr4.columns.tolist(), 'UC4', model_dir='models')

# =============================================================================
# STEP 3 — Threshold tuning
# UC3: precision-targeted — live risk system, minimise false alarms on bookings
# UC4: F1-optimal — delay flag, balanced precision/recall acceptable
# =============================================================================

# UC3 — target precision >= 0.65: when we flag a booking as cancel-risk,
#        we are right at least 65% of the time.
#        Raise min_precision to reduce FP further (fewer catches but more reliable).
#        Lower it to catch more cancellations (more FP accepted).
thresh_uc3, _ = precision_targeted_threshold(
    uc3_model, X_te3, y_test_uc3,
    min_precision=0.65,
    use_case='UC3',
)

# UC4 — F1-optimal threshold
thresh_uc4, _ = threshold_tuning(uc4_model, X_te4, y_test_uc4, use_case='UC4')

save_thresholds(
    {'uc3_threshold': float(thresh_uc3), 'uc4_threshold': float(thresh_uc4)},
    model_dir='models',
)

# =============================================================================
# STEP 4 — SMOTE retrain for UC1 (optional, compare with baseline)
# =============================================================================
from src.modeling.postprocessing import smote_retrain
from xgboost import XGBClassifier

smote_model, _ = smote_retrain(
    model_class=XGBClassifier,
    model_kwargs=dict(
        n_estimators=400, max_depth=6, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric='mlogloss', use_label_encoder=False,
        random_state=42, n_jobs=-1,
    ),
    X_train=X_tr1, y_train=y_train_uc1,
    X_test=X_te1,  y_test=y_test_uc1,
    use_case='UC1',
)

# =============================================================================
# STEP 5 — Summary
# =============================================================================
import pandas as pd
print("\n" + "=" * 60)
print("BASELINE SUMMARY")
print("=" * 60)

print("\n── UC1 (Multi-Class) ──")
print(pd.DataFrame(uc1_results).T.to_string())

for vtype, (model, X_tr, X_te, res) in uc2_results.items():
    print(f"\n── UC2-{vtype} (Regression) ──")
    print(pd.DataFrame(res).T.to_string())

print("\n── UC3 (Binary) ──")
print(pd.DataFrame(uc3_results).T.to_string())

print("\n── UC4 (Binary) ──")
print(pd.DataFrame(uc4_results).T.to_string())

print("\n✅ Stage 2 complete. Run run_tuning.py next.")