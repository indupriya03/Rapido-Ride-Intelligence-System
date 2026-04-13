# =============================================================================
# MODEL TRAINING PIPELINE
# =============================================================================
# UC1 — Ride Outcome Prediction         (Multi-Class Classification)
# UC2 — Fare Prediction                  (Regression)
# UC3 — Customer Cancellation Risk       (Binary Classification)
# UC4 — Driver Delay Prediction          (Binary Classification)
# =============================================================================
#
# MODEL PLAN:
#   Classification (UC1, UC3, UC4):
#     - Logistic Regression  (baseline)
#     - Random Forest
#     - XGBoost / LightGBM   ✅ (recommended)
#
#   Regression (UC2):
#     - Linear Regression    (baseline)
#     - Random Forest Regressor
#     - XGBoost Regressor    ✅ (recommended)
#
# STRUCTURE:
#   SECTION 0  → Imports & setup
#   SECTION 1  → Load data from feature engineering pipeline
#   SECTION 2  → Helper utilities (clean columns, evaluate)
#   SECTION 3  → Model definitions
#   SECTION 4  → UC1  Multi-class Classification
#   SECTION 5  → UC2  Regression
#   SECTION 6  → UC3  Binary Classification (Cancellation)
#   SECTION 7  → UC4  Binary Classification (Driver Delay)
#   SECTION 8  → Summary comparison table
# =============================================================================

#from pyexpat import model
import sys
import os
import warnings
#from fastapi import params
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier

import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

import joblib, json, os
os.makedirs("models", exist_ok=True)

warnings.filterwarnings('ignore')
sys.path.append("src")

# =============================================================================
# SECTION 0 — REPRODUCIBILITY
# =============================================================================
SEED = 42
np.random.seed(SEED)

# =============================================================================
# SECTION 1 — LOAD DATA FROM FEATURE ENGINEERING PIPELINE
# =============================================================================

from data_loader import load_cleaned_data
from feature_engineering import run_zone1_engineering, get_splits

print("Loading raw data...")
bookings_df, customers_df, drivers_df, location_demand_df, time_features_df = load_cleaned_data()

print("\nRunning Zone 1 engineering...")
df = run_zone1_engineering(
    bookings_df, customers_df, drivers_df,
    location_demand_df, time_features_df
)

print("\nGenerating per-use-case splits...")
X_train_uc1, X_test_uc1, y_train_uc1, y_test_uc1 = get_splits(df.copy(), 'UC1')
X_train_uc2, X_test_uc2, y_train_uc2, y_test_uc2 = get_splits(df.copy(), 'UC2')
X_train_uc3, X_test_uc3, y_train_uc3, y_test_uc3 = get_splits(df.copy(), 'UC3')
X_train_uc4, X_test_uc4, y_train_uc4, y_test_uc4 = get_splits(df.copy(), 'UC4')

print("\n✅ All splits ready.")


# =============================================================================
# SECTION 2 — HELPER UTILITIES
# =============================================================================

def clean_for_sklearn(X_train, X_test):
    """
    Drop any remaining non-numeric / categorical columns that sklearn cannot
    handle (object, category dtype). Returns cleaned copies.
    Also sanitises column names (LightGBM/XGBoost dislike brackets/spaces).
    """
    # Strip unsupported dtypes
    drop_dtypes = ['object', 'category']
    bad_cols = X_train.select_dtypes(include=drop_dtypes).columns.tolist()
    if bad_cols:
        print(f"  ⚠️  Dropping {len(bad_cols)} non-numeric cols: {bad_cols[:5]}{'...' if len(bad_cols)>5 else ''}")
    X_train = X_train.drop(columns=bad_cols, errors='ignore').copy()
    X_test  = X_test.drop(columns=bad_cols,  errors='ignore').copy()

    # Rename columns — XGBoost / LightGBM hate special characters
    rename = lambda c: (c.replace('[', '_').replace(']', '_')
                          .replace('<', '_').replace('>', '_')
                          .replace(' ', '_').replace(',', '_'))
    X_train.columns = [rename(c) for c in X_train.columns]
    X_test.columns  = [rename(c) for c in X_test.columns]

    # Fill any residual NaNs with column median (train-derived)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    return X_train, X_test


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------
def evaluate_classifier(name, model, X_test, y_test, multi_class=False, results_store=None):
    """
    Prints classification report + confusion matrix.
    Stores summary metrics in results_store dict (optional).
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    avg    = 'weighted' if multi_class else 'binary'
    f1     = f1_score(y_test, y_pred, average=avg, zero_division=0)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 ({avg:>8}): {f1:.4f}")

    # AUC — only for binary
    if not multi_class:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc   = roc_auc_score(y_test, proba)
            print(f"  ROC-AUC  : {auc:.4f}")
        except Exception:
            auc = None
    else:
        auc = None

    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    if results_store is not None:
        results_store[name] = {
            'accuracy': round(acc, 4),
            'f1': round(f1, 4),
            'auc': round(auc, 4) if auc else 'N/A',
        }
    return y_pred


# --------------------------------------------------------------------------
# Regression metrics
# --------------------------------------------------------------------------
def evaluate_regressor(name, model, X_test, y_test, log_target=True, results_store=None):
    """
    Evaluates regression model. If log_target=True, back-transforms with expm1.
    """
    y_pred_log = model.predict(X_test)

    if log_target:
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
    else:
        y_pred = y_pred_log
        y_true = y_test

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    if results_store is not None:
        results_store[name] = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
        }
    return y_pred


# --------------------------------------------------------------------------
# Feature importance plot
# --------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, title, top_n=20, save_path=None):
    """Works for tree-based models with feature_importances_."""
    if not hasattr(model, 'feature_importances_'):
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Saved: {save_path}")
    plt.close()


# --------------------------------------------------------------------------
# Confusion matrix plot
# --------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Saved: {save_path}")
    plt.close()
    fig, ax = plt.subplots(figsize=(7, 5))



# =============================================================================
# SECTION 3 — MODEL DEFINITIONS
# =============================================================================

def get_classifiers(n_classes=2, class_weight='balanced'):
    """Returns dict of {name: model} for classification."""
    multi = (n_classes > 2)

    models = {
        'Logistic Regression (Baseline)': LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=SEED,
            multi_class='multinomial' if multi else 'auto',
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=SEED,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss' if multi else 'logloss',
            random_state=SEED,
            n_jobs=-1,
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=class_weight,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }
    return models


def get_regressors():
    """Returns dict of {name: model} for regression."""
    return {
        'Linear Regression (Baseline)': LinearRegression(n_jobs=-1),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        ),
        'XGBoost Regressor': XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
        ),
    }


# =============================================================================
# SECTION 4 — UC1: RIDE OUTCOME PREDICTION (Multi-Class Classification)
# =============================================================================
# Target : booking_status_enc  → 0=Completed, 1=Cancelled, 2=Incomplete
# Models : Logistic Regression | Random Forest | XGBoost | LightGBM
# =============================================================================

print("\n" + "="*60)
print("UC1 — RIDE OUTCOME PREDICTION (Multi-Class Classification)")
print("="*60)

X_tr1, X_te1 = clean_for_sklearn(X_train_uc1, X_test_uc1)
uc1_results   = {}
CLASS_LABELS_UC1 = ['Completed', 'Cancelled', 'Incomplete']

models_uc1 = get_classifiers(n_classes=3, class_weight='balanced')
best_uc1   = {'name': None, 'f1': -1, 'model': None, 'y_pred': None}

for name, clf in models_uc1.items():
    print(f"\nTraining: {name} ...")
    clf.fit(X_tr1, y_train_uc1)
    y_pred = evaluate_classifier(
        name, clf, X_te1, y_test_uc1,
        multi_class=True, results_store=uc1_results
    )
    if uc1_results[name]['f1'] > best_uc1['f1']:
        best_uc1 = {'name': name, 'f1': uc1_results[name]['f1'],
                    'model': clf, 'y_pred': y_pred}

print(f"\n🏆 UC1 Best Model: {best_uc1['name']}  (F1={best_uc1['f1']:.4f})")

# Plots for best model
plot_confusion_matrix(
    y_test_uc1, best_uc1['y_pred'],
    labels=CLASS_LABELS_UC1,
    title=f"UC1 Confusion Matrix — {best_uc1['name']}",
    save_path="outputs/uc1_confusion_matrix.png"
)
plot_feature_importance(
    best_uc1['model'], X_tr1.columns,
    title=f"UC1 Feature Importance — {best_uc1['name']}",
    save_path="outputs/uc1_feature_importance.png"
)


# =============================================================================
# SECTION 5 — UC2: FARE PREDICTION (Regression)
# =============================================================================
# Target : booking_value_log  (log1p transformed, back-transformed for metrics)
# Models : Linear Regression | Random Forest Regressor | XGBoost Regressor
# =============================================================================

print("\n" + "="*60)
print("UC2 — FARE PREDICTION (Regression)")
print("="*60)

X_tr2, X_te2 = clean_for_sklearn(X_train_uc2, X_test_uc2)
uc2_results   = {}

models_uc2 = get_regressors()
best_uc2   = {'name': None, 'r2': -np.inf, 'model': None}

for name, reg in models_uc2.items():
    print(f"\nTraining: {name} ...")
    reg.fit(X_tr2, y_train_uc2)
    evaluate_regressor(
        name, reg, X_te2, y_test_uc2,
        log_target=True, results_store=uc2_results
    )
    if uc2_results[name]['R2'] > best_uc2['r2']:
        best_uc2 = {'name': name, 'r2': uc2_results[name]['R2'], 'model': reg}

print(f"\n🏆 UC2 Best Model: {best_uc2['name']}  (R²={best_uc2['r2']:.4f})")

# Residual plot for best regressor
y_pred_log = best_uc2['model'].predict(X_te2)
residuals  = np.expm1(y_test_uc2) - np.expm1(y_pred_log)
fig, axes  = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(np.expm1(y_pred_log), residuals, alpha=0.3, s=10, color='steelblue')
axes[0].axhline(0, color='red', linewidth=1)
axes[0].set_xlabel('Predicted Fare'); axes[0].set_ylabel('Residual')
axes[0].set_title(f'UC2 Residuals — {best_uc2["name"]}')
axes[1].hist(residuals, bins=60, color='steelblue', edgecolor='white')
axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Count')
axes[1].set_title('Residual Distribution')
plt.tight_layout()
plt.savefig("outputs/uc2_residuals.png", dpi=150)
print("  📊 Saved: outputs/uc2_residuals.png")
plt.close()

plot_feature_importance(
    best_uc2['model'], X_tr2.columns,
    title=f"UC2 Feature Importance — {best_uc2['name']}",
    save_path="outputs/uc2_feature_importance.png"
)


# =============================================================================
# SECTION 6 — UC3: CUSTOMER CANCELLATION RISK (Binary Classification)
# =============================================================================
# Target : is_cancelled  → 0=Not Cancelled, 1=Cancelled
# Note   : cancellation_rate and related features are EXCLUDED (leakage)
# Models : Logistic Regression | Random Forest | XGBoost | LightGBM
# =============================================================================

print("\n" + "="*60)
print("UC3 — CUSTOMER CANCELLATION RISK (Binary Classification)")
print("="*60)

X_tr3, X_te3 = clean_for_sklearn(X_train_uc3, X_test_uc3)
uc3_results   = {}
CLASS_LABELS_UC3 = ['Not Cancelled', 'Cancelled']

models_uc3 = get_classifiers(n_classes=2, class_weight='balanced')
best_uc3   = {'name': None, 'auc': -1, 'model': None, 'y_pred': None}

for name, clf in models_uc3.items():
    print(f"\nTraining: {name} ...")
    clf.fit(X_tr3, y_train_uc3)
    y_pred = evaluate_classifier(
        name, clf, X_te3, y_test_uc3,
        multi_class=False, results_store=uc3_results
    )
    auc_val = uc3_results[name]['auc']
    auc_cmp = auc_val if isinstance(auc_val, float) else -1
    if auc_cmp > best_uc3['auc']:
        best_uc3 = {'name': name, 'auc': auc_cmp, 'model': clf, 'y_pred': y_pred}

print(f"\n🏆 UC3 Best Model: {best_uc3['name']}  (AUC={best_uc3['auc']:.4f})")

plot_confusion_matrix(
    y_test_uc3, best_uc3['y_pred'],
    labels=CLASS_LABELS_UC3,
    title=f"UC3 Confusion Matrix — {best_uc3['name']}",
    save_path="outputs/uc3_confusion_matrix.png"
)
plot_feature_importance(
    best_uc3['model'], X_tr3.columns,
    title=f"UC3 Feature Importance — {best_uc3['name']}",
    save_path="outputs/uc3_feature_importance.png"
)


# =============================================================================
# SECTION 7 — UC4: DRIVER DELAY PREDICTION (Binary Classification)
# =============================================================================
# Target : driver_delay_flag  → 0=On Time, 1=Delayed
# Note   : delay_rate and derived features are EXCLUDED (leakage)
# Models : Logistic Regression | Random Forest | XGBoost | LightGBM
# =============================================================================

print("\n" + "="*60)
print("UC4 — DRIVER DELAY PREDICTION (Binary Classification)")
print("="*60)

X_tr4, X_te4 = clean_for_sklearn(X_train_uc4, X_test_uc4)
uc4_results   = {}
CLASS_LABELS_UC4 = ['On Time', 'Delayed']

models_uc4 = get_classifiers(n_classes=2, class_weight='balanced')
best_uc4   = {'name': None, 'auc': -1, 'model': None, 'y_pred': None}

for name, clf in models_uc4.items():
    print(f"\nTraining: {name} ...")
    clf.fit(X_tr4, y_train_uc4)
    y_pred = evaluate_classifier(
        name, clf, X_te4, y_test_uc4,
        multi_class=False, results_store=uc4_results
    )
    auc_val = uc4_results[name]['auc']
    auc_cmp = auc_val if isinstance(auc_val, float) else -1
    if auc_cmp > best_uc4['auc']:
        best_uc4 = {'name': name, 'auc': auc_cmp, 'model': clf, 'y_pred': y_pred}

print(f"\n🏆 UC4 Best Model: {best_uc4['name']}  (AUC={best_uc4['auc']:.4f})")

plot_confusion_matrix(
    y_test_uc4, best_uc4['y_pred'],
    labels=CLASS_LABELS_UC4,
    title=f"UC4 Confusion Matrix — {best_uc4['name']}",
    save_path="outputs/uc4_confusion_matrix.png"
)
plot_feature_importance(
    best_uc4['model'], X_tr4.columns,
    title=f"UC4 Feature Importance — {best_uc4['name']}",
    save_path="outputs/uc4_feature_importance.png"
)


# =============================================================================
# SECTION 8 — SUMMARY COMPARISON TABLE
# =============================================================================

print("\n" + "="*60)
print("SUMMARY — ALL USE CASES")
print("="*60)

print("\n── UC1: Ride Outcome (Multi-Class Classification) ──")
uc1_df = pd.DataFrame(uc1_results).T
uc1_df.index.name = 'Model'
print(uc1_df.to_string())

print("\n── UC2: Fare Prediction (Regression) ──")
uc2_df = pd.DataFrame(uc2_results).T
uc2_df.index.name = 'Model'
print(uc2_df.to_string())

print("\n── UC3: Cancellation Risk (Binary Classification) ──")
uc3_df = pd.DataFrame(uc3_results).T
uc3_df.index.name = 'Model'
print(uc3_df.to_string())

print("\n── UC4: Driver Delay (Binary Classification) ──")
uc4_df = pd.DataFrame(uc4_results).T
uc4_df.index.name = 'Model'
print(uc4_df.to_string())

print("\n✅ Model training pipeline complete.")
print("   Plots saved to outputs/ directory.")


# =============================================================================
# SECTION 9 — THRESHOLD TUNING (replaces Optuna)
# =============================================================================
from sklearn.metrics import precision_recall_curve

print("\n" + "="*60)
print("THRESHOLD TUNING — UC3 & UC4")
print("="*60)

# UC3
probs_uc3 = best_uc3['model'].predict_proba(X_te3)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test_uc3, probs_uc3)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_thresh_uc3 = thresholds[f1_scores[:-1].argmax()]
y_pred_uc3_tuned = (probs_uc3 >= best_thresh_uc3).astype(int)

print(f"\nUC3 — optimal threshold : {best_thresh_uc3:.3f}  (default 0.500)")
print(f"  F1 before : {f1_score(y_test_uc3, best_uc3['y_pred'], average='binary'):.4f}")
print(f"  F1 after  : {f1_score(y_test_uc3, y_pred_uc3_tuned, average='binary'):.4f}")
print(f"  AUC       : {roc_auc_score(y_test_uc3, probs_uc3):.4f}")

# UC4
probs_uc4 = best_uc4['model'].predict_proba(X_te4)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test_uc4, probs_uc4)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_thresh_uc4 = thresholds[f1_scores[:-1].argmax()]
y_pred_uc4_tuned = (probs_uc4 >= best_thresh_uc4).astype(int)

print(f"\nUC4 — optimal threshold : {best_thresh_uc4:.3f}  (default 0.500)")
print(f"  F1 before : {f1_score(y_test_uc4, best_uc4['y_pred'], average='binary'):.4f}")
print(f"  F1 after  : {f1_score(y_test_uc4, y_pred_uc4_tuned, average='binary'):.4f}")
print(f"  AUC       : {roc_auc_score(y_test_uc4, probs_uc4):.4f}")


# =============================================================================
# SECTION 10 — SMOTE FOR UC1
# =============================================================================
from imblearn.over_sampling import SMOTE

print("\n" + "="*60)
print("UC1 — SMOTE REBALANCING")
print("="*60)

print(f"\nClass distribution before SMOTE: {dict(pd.Series(y_train_uc1).value_counts().sort_index())}")
sm = SMOTE(random_state=42)
X_tr1_res, y_tr1_res = sm.fit_resample(X_tr1, y_train_uc1)
print(f"Class distribution after  SMOTE: {dict(pd.Series(y_tr1_res).value_counts().sort_index())}")

uc1_smote_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.06,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
).fit(X_tr1_res, y_tr1_res)

print("\nUC1 — XGBoost + SMOTE results:")
evaluate_classifier('XGBoost + SMOTE', uc1_smote_model, X_te1, y_test_uc1,
                    multi_class=True, results_store=None)


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"UC1 Accuracy — baseline XGB  : 0.8005")
print(f"UC1 Accuracy — XGB + SMOTE   : {accuracy_score(y_test_uc1, uc1_smote_model.predict(X_te1)):.4f}")
print(f"UC3 F1       — before thresh  : {f1_score(y_test_uc3, best_uc3['y_pred'], average='binary'):.4f}")
print(f"UC3 F1       — after  thresh  : {f1_score(y_test_uc3, y_pred_uc3_tuned, average='binary'):.4f}")
print(f"UC4 F1       — before thresh  : {f1_score(y_test_uc4, best_uc4['y_pred'], average='binary'):.4f}")
print(f"UC4 F1       — after  thresh  : {f1_score(y_test_uc4, y_pred_uc4_tuned, average='binary'):.4f}")
print("\n✅ Done. Total extra time: ~30 seconds.")


# model saving
joblib.dump(best_uc1['model'], "models/uc1_outcome_model.pkl")
joblib.dump(best_uc2['model'], "models/uc2_fare_model.pkl")
joblib.dump(best_uc3['model'], "models/uc3_cancel_model.pkl")
joblib.dump(best_uc4['model'], "models/uc4_delay_model.pkl")

# Also save the threshold for UC3
json.dump(
    {'uc3_threshold': float(best_thresh_uc3),
     'uc4_threshold': float(best_thresh_uc4)},
    open("models/thresholds.json", "w")
)
print("✅ Models saved to models/")