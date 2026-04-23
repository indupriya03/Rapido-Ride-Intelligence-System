# =============================================================================
# src/modeling/model_utils.py
# =============================================================================
# Shared utilities used across all model training scripts:
#   clean_for_sklearn()      — strip non-numeric cols, sanitise names, fill NaN
#   evaluate_classifier()    — print report + store metrics
#   evaluate_regressor()     — print MAE/RMSE/R² (with log back-transform)
#   plot_confusion_matrix()  — seaborn heatmap → save PNG
#   plot_feature_importance()— horizontal bar → save PNG
# =============================================================================

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

SEED = 42


# =============================================================================
# DATA CLEANING
# =============================================================================

def clean_for_sklearn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop non-numeric columns, sanitise column names, fill residual NaN with
    train-derived medians. Returns cleaned copies ready for any sklearn estimator.
    """
    drop_dtypes = ['object', 'category']
    bad_cols = X_train.select_dtypes(include=drop_dtypes).columns.tolist()
    if bad_cols:
        print(f"  ⚠️  Dropping {len(bad_cols)} non-numeric cols: "
              f"{bad_cols[:5]}{'...' if len(bad_cols) > 5 else ''}")

    X_train = X_train.drop(columns=bad_cols, errors='ignore').copy()
    X_test  = X_test.drop(columns=bad_cols,  errors='ignore').copy()

    def _rename(c):
        return (c.replace('[', '_').replace(']', '_')
                 .replace('<', '_').replace('>', '_')
                 .replace(' ', '_').replace(',', '_'))

    X_train.columns = [_rename(c) for c in X_train.columns]
    X_test.columns  = [_rename(c) for c in X_test.columns]

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    return X_train, X_test


# =============================================================================
# EVALUATION — CLASSIFICATION
# =============================================================================

def evaluate_classifier(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    multi_class: bool = False,
    results_store: dict | None = None,
) -> np.ndarray:
    """
    Print classification report. Optionally store summary metrics in results_store.
    Returns y_pred array.
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    avg    = 'weighted' if multi_class else 'binary'
    f1     = f1_score(y_test, y_pred, average=avg, zero_division=0)

    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 ({avg:>8}): {f1:.4f}")

    auc = None
    if not multi_class:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc   = roc_auc_score(y_test, proba)
            print(f"  ROC-AUC  : {auc:.4f}")
        except Exception:
            pass

    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    if results_store is not None:
        results_store[name] = {
            'accuracy': round(acc, 4),
            'f1'      : round(f1, 4),
            'auc'     : round(auc, 4) if auc is not None else 'N/A',
        }
    return y_pred


# =============================================================================
# EVALUATION — REGRESSION
# =============================================================================

def evaluate_regressor(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_target: bool = True,
    results_store: dict | None = None,
) -> np.ndarray:
    """
    Print MAE / RMSE / R². If log_target=True, back-transforms with expm1.
    Returns y_pred (original scale).
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

    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    if results_store is not None:
        results_store[name] = {
            'MAE' : round(mae,  4),
            'RMSE': round(rmse, 4),
            'R2'  : round(r2,   4),
        }
    return y_pred


# =============================================================================
# PLOTS
# =============================================================================

def plot_confusion_matrix(
    y_true, y_pred,
    labels: list,
    title: str,
    save_path: str | None = None,
) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Saved: {save_path}")
    plt.close()


def plot_feature_importance(
    model,
    feature_names,
    title: str,
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    if not hasattr(model, 'feature_importances_'):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  📊 Saved: {save_path}")
    plt.close()
