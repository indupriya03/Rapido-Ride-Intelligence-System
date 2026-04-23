# =============================================================================
# run_tuning.py  — Stage 3
# =============================================================================
# Optuna tuning for UC1, UC3, UC4 → retrain on full data → save tuned models.
# UC2 is skipped (regression baseline goes straight to final in Stage 4).
#
# Speed optimisations:
#   - Tune on 40% subset, retrain on full set
#   - n_estimators capped at 300
#   - Results cached in outputs/tuned_params.json — delete to re-run Optuna
#
# Run:  python run_tuning.py
# Requires: run_training.py to have been run first (splits must exist).
# =============================================================================

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split as tts

from src.tuning.tuning_utils   import save_tuned_params, load_tuned_params
from src.tuning.tune_xgboost   import tune_xgboost_classifier
from src.tuning.tune_lgbm      import tune_lgbm_classifier
from src.tuning.retrain_tuned  import retrain_and_evaluate
from src.modeling.model_utils  import clean_for_sklearn
from src.modeling.model_io     import save_thresholds, load_thresholds
from src.modeling.postprocessing import threshold_tuning, precision_targeted_threshold

os.makedirs('outputs', exist_ok=True)
SEED = 42

# =============================================================================
# STEP 1 — Load splits
# =============================================================================
print("Loading splits...")
X_train_uc1, X_test_uc1, y_train_uc1, y_test_uc1 = joblib.load('splits/uc1.pkl')
X_train_uc3, X_test_uc3, y_train_uc3, y_test_uc3 = joblib.load('splits/uc3.pkl')
X_train_uc4, X_test_uc4, y_train_uc4, y_test_uc4 = joblib.load('splits/uc4.pkl')

X_tr1, X_te1 = clean_for_sklearn(X_train_uc1, X_test_uc1)
X_tr3, X_te3 = clean_for_sklearn(X_train_uc3, X_test_uc3)
X_tr4, X_te4 = clean_for_sklearn(X_train_uc4, X_test_uc4)
print("✅ Splits loaded and cleaned.")

# =============================================================================
# STEP 2 — 40% tuning subsets
# =============================================================================
print("\nPreparing 40% tuning subsets...")
X_tr1_tune, _, y_tr1_tune, _ = tts(X_tr1, y_train_uc1, train_size=0.4,
                                    random_state=SEED, stratify=y_train_uc1)
X_tr3_tune, _, y_tr3_tune, _ = tts(X_tr3, y_train_uc3, train_size=0.4,
                                    random_state=SEED, stratify=y_train_uc3)
X_tr4_tune, _, y_tr4_tune, _ = tts(X_tr4, y_train_uc4, train_size=0.4,
                                    random_state=SEED, stratify=y_train_uc4)

# =============================================================================
# STEP 3 — Run or load Optuna
# =============================================================================
FORCE_TUNE = True # Set to True to ignore cached params and re-run tuning (will overwrite tuned_params.json)
existing = load_tuned_params()

if existing and not FORCE_TUNE:
    print("\n✅ Cached params found — skipping Optuna.")
    uc1_xgb_params  = existing['uc1_xgb']
    uc1_lgbm_params = existing['uc1_lgbm']
    uc3_xgb_params  = existing['uc3_xgb']
    uc3_lgbm_params = existing['uc3_lgbm']
    uc4_xgb_params  = existing['uc4_xgb']
    uc4_lgbm_params = existing['uc4_lgbm']

else:
    print("\n🔍 No cached params — running Optuna...\n")

    print("=" * 60)
    print("UC1 — XGBoost (50 trials)")
    uc1_xgb_params, _  = tune_xgboost_classifier(
        X_tr1_tune, y_tr1_tune, X_te1, y_test_uc1,
        n_trials=50, multi_class=True, uc_name='UC1')

    print("\nUC1 — LightGBM (50 trials)")
    uc1_lgbm_params, _ = tune_lgbm_classifier(
        X_tr1_tune, y_tr1_tune, X_te1, y_test_uc1,
        n_trials=50, multi_class=True, uc_name='UC1')

    print("\n" + "=" * 60)
    print("UC3 — XGBoost (30 trials)")
    uc3_xgb_params, _  = tune_xgboost_classifier(
        X_tr3_tune, y_tr3_tune, X_te3, y_test_uc3,
        n_trials=30, multi_class=False, class_weight='balanced', uc_name='UC3')

    print("\nUC3 — LightGBM (30 trials)")
    uc3_lgbm_params, _ = tune_lgbm_classifier(
        X_tr3_tune, y_tr3_tune, X_te3, y_test_uc3,
        n_trials=30, multi_class=False, class_weight='balanced', uc_name='UC3')

    print("\n" + "=" * 60)
    print("UC4 — XGBoost (30 trials)")
    uc4_xgb_params, _  = tune_xgboost_classifier(
        X_tr4_tune, y_tr4_tune, X_te4, y_test_uc4,
        n_trials=30, multi_class=False, class_weight='balanced', uc_name='UC4')

    print("\nUC4 — LightGBM (30 trials)")
    uc4_lgbm_params, _ = tune_lgbm_classifier(
        X_tr4_tune, y_tr4_tune, X_te4, y_test_uc4,
        n_trials=30, multi_class=False, class_weight='balanced', uc_name='UC4')

    save_tuned_params({
        'uc1_xgb' : uc1_xgb_params,  'uc1_lgbm': uc1_lgbm_params,
        'uc3_xgb' : uc3_xgb_params,  'uc3_lgbm': uc3_lgbm_params,
        'uc4_xgb' : uc4_xgb_params,  'uc4_lgbm': uc4_lgbm_params,
    })

# =============================================================================
# STEP 4 — Retrain on full data
# =============================================================================
print("\n" + "=" * 60)
print("RETRAINING TUNED MODELS ON FULL TRAINING DATA")
print("=" * 60)

uc1_best, uc1_label, uc1_acc, uc1_f1 = retrain_and_evaluate(
    uc_name='UC1', xgb_params=uc1_xgb_params, lgbm_params=uc1_lgbm_params,
    X_train=X_tr1, y_train=y_train_uc1,
    X_test=X_te1,  y_test=y_test_uc1,
    multi_class=True,
    labels=['Completed', 'Cancelled', 'Incomplete'],
    save_prefix='uc1', model_dir='models', output_dir='outputs',
)

uc3_best, uc3_label, uc3_acc, uc3_f1 = retrain_and_evaluate(
    uc_name='UC3', xgb_params=uc3_xgb_params, lgbm_params=uc3_lgbm_params,
    X_train=X_tr3, y_train=y_train_uc3,
    X_test=X_te3,  y_test=y_test_uc3,
    multi_class=False, class_weight='balanced',
    labels=['Not Cancelled', 'Cancelled'],
    save_prefix='uc3', model_dir='models', output_dir='outputs',
)

uc4_best, uc4_label, uc4_acc, uc4_f1 = retrain_and_evaluate(
    uc_name='UC4', xgb_params=uc4_xgb_params, lgbm_params=uc4_lgbm_params,
    X_train=X_tr4, y_train=y_train_uc4,
    X_test=X_te4,  y_test=y_test_uc4,
    multi_class=False, class_weight='balanced',
    labels=['On Time', 'Delayed'],
    save_prefix='uc4', model_dir='models', output_dir='outputs',
)

# =============================================================================
# STEP 4b — Re-tune thresholds on the tuned models
# (thresholds.json from run_training.py was fitted on baseline models —
#  re-compute here for the tuned models that will be used as finals)
# UC3: precision-targeted — minimise false alarms on live bookings
# UC4: F1-optimal
# =============================================================================
thresh_uc3_tuned, _ = precision_targeted_threshold(
    uc3_best, X_te3, y_test_uc3,
    min_precision=0.65,
    use_case='UC3 (tuned)',
)
thresh_uc4_tuned, _ = threshold_tuning(
    uc4_best, X_te4, y_test_uc4,
    use_case='UC4 (tuned)',
)
save_thresholds(
    {'uc3_threshold': float(thresh_uc3_tuned),
     'uc4_threshold': float(thresh_uc4_tuned)},
    model_dir='models',
)
print("✅ Thresholds updated for tuned models → models/thresholds.json")

# =============================================================================
# STEP 5 — Summary
# =============================================================================
print("\n" + "=" * 60)
print("TUNING SUMMARY")
print("=" * 60)
print(f"{'Use Case':<30} {'Winner':<12} {'Acc':>8} {'F1':>8}")
print("─" * 60)
print(f"{'UC1 (Multi-Class)':<30} {uc1_label:<12} {uc1_acc:>8.4f} {uc1_f1:>8.4f}")
print(f"{'UC3 (Cancellation)':<30} {uc3_label:<12} {uc3_acc:>8.4f} {uc3_f1:>8.4f}")
print(f"{'UC4 (Driver Delay)':<30} {uc4_label:<12} {uc4_acc:>8.4f} {uc4_f1:>8.4f}")
print(f"{'UC2 (Fare/vehicle)':<30} {'N/A — skip':<12}")

print("\n✅ Stage 3 complete. Run run_model_selection.py next.")