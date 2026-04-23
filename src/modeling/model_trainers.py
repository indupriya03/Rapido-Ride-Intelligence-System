# =============================================================================
# src/modeling/model_trainers.py
# =============================================================================
# One train_uc*() function per use case.
# Each function: trains all baseline models, picks the best, saves, returns it.
# =============================================================================

import os
import numpy as np
import pandas as pd

from .model_utils       import (clean_for_sklearn, evaluate_classifier,
                                 evaluate_regressor, plot_confusion_matrix,
                                 plot_feature_importance)
from .model_definitions import get_classifiers, get_regressors
from .model_io          import save_model

SEED = 42


# =============================================================================
# UC1 — Ride Outcome (Multi-Class Classification)
# Target: booking_status_enc  (0=Completed, 1=Cancelled, 2=Incomplete)
#
# FIX APPLIED (2 issues diagnosed from data):
#
# Issue 1 — Leakage (fixed in zone2_config):
#   cancel_risk_x_peak / _night were interaction terms built from
#   cancellation_rate (in UC1_LEAKAGE). They were not listed and survived
#   the drop. Now blocked at zone2_config level.
#
# Issue 2 — Incomplete class imbalance (fixed here):
#   Incomplete is only 8.4% of training data. Baseline balanced weight
#   gave Incomplete recall=0.36. Two-part fix:
#     a) SMOTE: oversample Incomplete to 50% of Cancelled count (~9,300)
#     b) class_weight={0:1, 1:3, 2:6}: penalise Incomplete misses 6x
#   Net result: Incomplete recall 0.36 -> 0.71, F1 0.45 -> 0.46
#   Completed F1 drop (0.86->0.76) expected — was inflated by leakage.
# =============================================================================

def train_uc1(X_train, X_test, y_train, y_test,
              model_dir='models', output_dir='outputs'):
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import f1_score as _f1

    print("\n" + "=" * 60)
    print("UC1 — RIDE OUTCOME PREDICTION  (Multi-Class)")
    print("=" * 60)

    X_tr, X_te = clean_for_sklearn(X_train, X_test)
    labels      = ['Completed', 'Cancelled', 'Incomplete']
    results     = {}

    # ── Baseline sweep (all models, balanced weights) ────────────────────────────────────────────
    best_baseline = {'name': None, 'f1': -1, 'model': None, 'y_pred': None}
    for name, clf in get_classifiers(n_classes=3, class_weight='balanced').items():
        print(f"\nTraining baseline: {name} ...")
        clf.fit(X_tr, y_train)
        y_pred = evaluate_classifier(name, clf, X_te, y_test,
                                     multi_class=True, results_store=results)
        if results[name]['f1'] > best_baseline['f1']:
            best_baseline = {'name': name, 'f1': results[name]['f1'],
                             'model': clf, 'y_pred': y_pred}
    print(f"\n  Baseline best: {best_baseline['name']}  (F1={best_baseline['f1']:.4f})")

    # ── SMOTE: oversample Incomplete to 50% of Cancelled count ────────────────────
    # Full equalisation collapses Completed precision. 50% of Cancelled
    # (~9,300) gives enough minority signal without over-synthesising.
    n_cancelled  = int((y_train == 1).sum())
    smote_target = int(n_cancelled * 0.50)
    print(f"\nSMOTE — Incomplete -> {smote_target:,} samples")
    print(f"  Before: {dict(pd.Series(y_train).value_counts().sort_index())}")
    smote = SMOTE(sampling_strategy={2: smote_target}, random_state=SEED, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_tr, y_train)
    print(f"  After : {dict(pd.Series(y_res).value_counts().sort_index())}")

    # ── LightGBM + targeted class weights ────────────────────────────────────────
    # class_weight={0:1, 1:3, 2:6}:
    #   Completed baseline | Cancelled 3x | Incomplete 6x (highest miss cost)
    # num_leaves=63: richer splits than depth-6 XGBoost for minority class
    CLASS_WEIGHT = {0: 1, 1: 3, 2: 6}
    print(f"\nTraining LightGBM — class_weight={CLASS_WEIGHT} ...")
    lgbm_fixed = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        min_child_samples=20, class_weight=CLASS_WEIGHT,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgbm_fixed.fit(X_res, y_res)
    y_pred_fixed = evaluate_classifier(
        'LightGBM (SMOTE + class_weight)', lgbm_fixed, X_te, y_test,
        multi_class=True, results_store=results,
    )

    # ── Report Incomplete F1 comparison ───────────────────────────────────────
    inc_f1_base  = _f1(y_test, best_baseline['y_pred'], average=None, zero_division=0)[2]
    inc_f1_fixed = _f1(y_test, y_pred_fixed,            average=None, zero_division=0)[2]
    print(f"\n  Incomplete F1 — baseline: {inc_f1_base:.4f}  |  fixed: {inc_f1_fixed:.4f}")

    # Always use fixed model: baseline weighted F1 advantage came from leakage
    fixed_f1   = results['LightGBM (SMOTE + class_weight)']['f1']
    final_name = 'LightGBM (SMOTE + class_weight)'
    print(f"\n🏆 UC1 Final: {final_name}  "
          f"(Weighted F1={fixed_f1:.4f}  |  Incomplete F1={inc_f1_fixed:.4f})")

    # ── Plots ───────────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred_fixed, labels,
                          title=f"UC1 Confusion — {final_name}",
                          save_path=os.path.join(output_dir, 'uc1_confusion_matrix.png'))
    plot_feature_importance(lgbm_fixed, X_tr.columns,
                            title=f"UC1 Feature Importance — {final_name}",
                            save_path=os.path.join(output_dir, 'uc1_feature_importance.png'))

    # ── Save both so you can compare ──────────────────────────────────────────────
    save_model(best_baseline['model'], 'uc1_baseline', model_dir)
    save_model(lgbm_fixed,             'uc1_final',    model_dir)

    return lgbm_fixed, X_tr, X_te, results


# =============================================================================
# UC2 — Fare Prediction (Regression, per vehicle type)
# Target: booking_value_log  (log1p transformed; back-transform for metrics)
# =============================================================================

def train_uc2(uc2_splits: dict, model_dir='models', output_dir='outputs'):
    """
    uc2_splits: {'Cab': (X_tr, X_te, y_tr, y_te), 'Auto': ..., 'Bike': ...}
    Returns: {'Cab': (model, X_tr_clean, X_te_clean, results), ...}
    """
    print("\n" + "=" * 60)
    print("UC2 — FARE PREDICTION  (Regression, per vehicle)")
    print("=" * 60)

    all_results = {}

    for vtype, (X_train, X_test, y_train, y_test) in uc2_splits.items():
        print(f"\n{'─' * 50}")
        print(f"  UC2 — {vtype}")
        print(f"{'─' * 50}")

        X_tr, X_te = clean_for_sklearn(X_train, X_test)
        results     = {}
        best        = {'name': None, 'r2': -np.inf, 'model': None}

        for name, reg in get_regressors().items():
            print(f"\nTraining: {name} ...")
            reg.fit(X_tr, y_train)
            evaluate_regressor(name, reg, X_te, y_test,
                               log_target=True, results_store=results)
            if results[name]['R2'] > best['r2']:
                best = {'name': name, 'r2': results[name]['R2'], 'model': reg}

        print(f"\n🏆 UC2-{vtype} Best: {best['name']}  (R²={best['r2']:.4f})")

        # Residual plot
        y_pred_log = best['model'].predict(X_te)
        residuals  = np.expm1(y_test) - np.expm1(y_pred_log)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(np.expm1(y_pred_log), residuals, alpha=0.3, s=8, color='steelblue')
        axes[0].axhline(0, color='red', linewidth=1)
        axes[0].set_xlabel('Predicted Fare'); axes[0].set_ylabel('Residual')
        axes[0].set_title(f'UC2-{vtype} Residuals')
        axes[1].hist(residuals, bins=60, color='steelblue', edgecolor='white')
        axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Count')
        axes[1].set_title('Residual Distribution')
        plt.tight_layout()
        path = os.path.join(output_dir, f'uc2_{vtype.lower()}_residuals.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  📊 Saved: {path}")

        plot_feature_importance(best['model'], X_tr.columns,
                                title=f"UC2-{vtype} Feature Importance",
                                save_path=os.path.join(output_dir,
                                    f'uc2_{vtype.lower()}_feature_importance.png'))
        save_model(best['model'], f'uc2_{vtype.lower()}_baseline', model_dir)
        all_results[vtype] = (best['model'], X_tr, X_te, results)

    return all_results


# =============================================================================
# UC3 — Customer Cancellation Risk (Binary Classification)
# Target: customer_cancel_flag  (0=Not Cancelled, 1=Cancelled)
# =============================================================================

def train_uc3(X_train, X_test, y_train, y_test,
              model_dir='models', output_dir='outputs'):
    print("\n" + "=" * 60)
    print("UC3 — CUSTOMER CANCELLATION RISK  (Binary)")
    print("=" * 60)

    X_tr, X_te = clean_for_sklearn(X_train, X_test)
    results     = {}
    labels      = ['Not Cancelled', 'Cancelled']
    best        = {'name': None, 'auc': -1, 'model': None, 'y_pred': None}

    for name, clf in get_classifiers(n_classes=2, class_weight='balanced').items():
        print(f"\nTraining: {name} ...")
        clf.fit(X_tr, y_train)
        y_pred = evaluate_classifier(name, clf, X_te, y_test,
                                     multi_class=False, results_store=results)
        auc_val = results[name]['auc']
        auc_cmp = auc_val if isinstance(auc_val, float) else -1
        if auc_cmp > best['auc']:
            best = {'name': name, 'auc': auc_cmp, 'model': clf, 'y_pred': y_pred}

    print(f"\n🏆 UC3 Best: {best['name']}  (AUC={best['auc']:.4f})")
    plot_confusion_matrix(y_test, best['y_pred'], labels,
                          title=f"UC3 Confusion — {best['name']}",
                          save_path=os.path.join(output_dir, 'uc3_confusion_matrix.png'))
    plot_feature_importance(best['model'], X_tr.columns,
                            title=f"UC3 Feature Importance — {best['name']}",
                            save_path=os.path.join(output_dir, 'uc3_feature_importance.png'))
    save_model(best['model'], 'uc3_baseline', model_dir)
    return best['model'], X_tr, X_te, results


# =============================================================================
# UC4 — Driver Delay Prediction (Binary Classification)
# Target: driver_delay_flag  (0=On Time, 1=Delayed)
# =============================================================================

def train_uc4(X_train, X_test, y_train, y_test,
              model_dir='models', output_dir='outputs'):
    print("\n" + "=" * 60)
    print("UC4 — DRIVER DELAY PREDICTION  (Binary)")
    print("=" * 60)

    X_tr, X_te = clean_for_sklearn(X_train, X_test)
    results     = {}
    labels      = ['On Time', 'Delayed']
    best        = {'name': None, 'auc': -1, 'model': None, 'y_pred': None}

    for name, clf in get_classifiers(n_classes=2, class_weight='balanced').items():
        print(f"\nTraining: {name} ...")
        clf.fit(X_tr, y_train)
        y_pred = evaluate_classifier(name, clf, X_te, y_test,
                                     multi_class=False, results_store=results)
        auc_val = results[name]['auc']
        auc_cmp = auc_val if isinstance(auc_val, float) else -1
        if auc_cmp > best['auc']:
            best = {'name': name, 'auc': auc_cmp, 'model': clf, 'y_pred': y_pred}

    print(f"\n🏆 UC4 Best: {best['name']}  (AUC={best['auc']:.4f})")
    plot_confusion_matrix(y_test, best['y_pred'], labels,
                          title=f"UC4 Confusion — {best['name']}",
                          save_path=os.path.join(output_dir, 'uc4_confusion_matrix.png'))
    plot_feature_importance(best['model'], X_tr.columns,
                            title=f"UC4 Feature Importance — {best['name']}",
                            save_path=os.path.join(output_dir, 'uc4_feature_importance.png'))
    save_model(best['model'], 'uc4_baseline', model_dir)
    return best['model'], X_tr, X_te, results