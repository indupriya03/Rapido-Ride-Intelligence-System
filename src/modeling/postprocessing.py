# =============================================================================
# src/modeling/postprocessing.py
# =============================================================================
# Post-training improvements:
#   threshold_tuning()          — F1-optimal threshold (UC4)
#   precision_targeted_threshold() — finds threshold at a minimum precision
#                                    constraint (UC3 live risk system)
#   smote_retrain()             — oversample minority class then retrain (UC1)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

SEED = 42


def threshold_tuning(
    model, X_test, y_test, use_case: str = ''
) -> tuple[float, np.ndarray]:
    """
    Find the threshold that maximises F1. Used for UC4.
    """
    probs      = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    f1_scores  = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx   = f1_scores[:-1].argmax()
    best_thresh = thresholds[best_idx]

    y_pred_default = model.predict(X_test)
    y_pred_tuned   = (probs >= best_thresh).astype(int)

    f1_before = f1_score(y_test, y_pred_default, average='binary', zero_division=0)
    f1_after  = f1_score(y_test, y_pred_tuned,   average='binary', zero_division=0)
    auc       = roc_auc_score(y_test, probs)

    print(f"\n{use_case} — Threshold Tuning (F1-optimal)")
    print(f"  Default threshold : 0.500  →  F1 = {f1_before:.4f}")
    print(f"  Best threshold    : {best_thresh:.3f}  →  F1 = {f1_after:.4f}")
    print(f"  ROC-AUC           : {auc:.4f}")

    return best_thresh, y_pred_tuned


def precision_targeted_threshold(
    model,
    X_test,
    y_test,
    min_precision: float = 0.65,
    use_case: str = '',
) -> tuple[float, np.ndarray]:
    """
    Find the lowest threshold that still meets a minimum precision requirement.

    For live risk scoring (UC3), we want to flag bookings as high-risk only
    when we're confident they'll actually cancel — minimising false alarms on
    real bookings. This trades some recall for meaningful precision.

    Strategy
    --------
    Walk the precision-recall curve from high threshold downward.
    Take the lowest threshold where precision >= min_precision.
    This maximises recall (catches more real cancellations) while keeping
    the false positive rate controlled.

    Falls back to F1-optimal threshold if no point meets min_precision.

    Parameters
    ----------
    model         : fitted binary classifier with predict_proba
    X_test        : test features
    y_test        : true binary labels
    min_precision : minimum acceptable precision (default 0.65)
                    Tune this based on business cost of false alarms vs misses.
    use_case      : label for printing only

    Returns
    -------
    best_threshold : float
    y_pred_tuned   : np.ndarray (0/1 at best_threshold)
    """
    probs = model.predict_proba(X_test)[:, 1]
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, probs)

    # precision_arr / recall_arr have one more element than thresholds
    # pair them up (drop last element of precision/recall which is the (1,0) point)
    pairs = list(zip(thresholds, precision_arr[:-1], recall_arr[:-1]))

    # Find all thresholds that meet min_precision, pick the one with best recall
    candidates = [(t, p, r) for t, p, r in pairs if p >= min_precision]

    if candidates:
        # Lowest threshold among candidates = highest recall while meeting precision
        best_thresh, best_prec, best_rec = min(candidates, key=lambda x: x[0])
        strategy = f"precision ≥ {min_precision}"
    else:
        # Fallback: F1-optimal
        f1_scores  = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-9)
        best_idx   = f1_scores[:-1].argmax()
        best_thresh = thresholds[best_idx]
        best_prec   = precision_arr[best_idx]
        best_rec    = recall_arr[best_idx]
        strategy    = "F1-optimal (fallback — precision target not achievable)"

    y_pred_default = model.predict(X_test)
    y_pred_tuned   = (probs >= best_thresh).astype(int)

    # Full metrics comparison
    auc    = roc_auc_score(y_test, probs)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_tuned).ravel()

    print(f"\n{use_case} — Precision-Targeted Threshold  ({strategy})")
    print(f"  AUC                    : {auc:.4f}")
    print(f"\n  {'Metric':<20} {'Default (0.5)':>15} {'Tuned ({:.3f})'.format(best_thresh):>15}")
    print(f"  {'─'*52}")
    print(f"  {'Threshold':<20} {'0.500':>15} {best_thresh:>15.3f}")
    print(f"  {'Precision':<20} {precision_score(y_test, y_pred_default, zero_division=0):>15.4f} "
          f"{precision_score(y_test, y_pred_tuned, zero_division=0):>15.4f}")
    print(f"  {'Recall':<20} {recall_score(y_test, y_pred_default, zero_division=0):>15.4f} "
          f"{recall_score(y_test, y_pred_tuned, zero_division=0):>15.4f}")
    print(f"  {'F1':<20} {f1_score(y_test, y_pred_default, zero_division=0):>15.4f} "
          f"{f1_score(y_test, y_pred_tuned, zero_division=0):>15.4f}")
    print(f"\n  Confusion matrix comparison:")
    print(f"  {'':20} {'Default':>15} {'Tuned':>15}")
    print(f"  {'TP (caught cancels)':<20} {tp:>15} {tp2:>15}")
    print(f"  {'FP (false alarms)':<20} {fp:>15} {fp2:>15}")
    print(f"  {'FN (missed cancels)':<20} {fn:>15} {fn2:>15}")
    print(f"  {'TN (correct clears)':<20} {tn:>15} {tn2:>15}")
    fp_rate_default = fp / (fp + tn) if (fp + tn) > 0 else 0
    fp_rate_tuned   = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0
    print(f"  {'FP rate':<20} {fp_rate_default:>15.4f} {fp_rate_tuned:>15.4f}")

    return best_thresh, y_pred_tuned


def smote_retrain(
    model_class, model_kwargs: dict,
    X_train, y_train, X_test, y_test,
    use_case: str = 'UC1'
) -> tuple:
    """
    Resample X_train with SMOTE then retrain model_class(**model_kwargs).
    """
    from .model_utils import evaluate_classifier

    print(f"\n{use_case} — SMOTE Rebalancing")
    print(f"  Before: {dict(pd.Series(y_train).value_counts().sort_index())}")

    sm = SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After : {dict(pd.Series(y_res).value_counts().sort_index())}")

    model = model_class(**model_kwargs)
    model.fit(X_res, y_res)

    multi  = len(np.unique(y_train)) > 2
    y_pred = evaluate_classifier(
        f'{model_class.__name__} + SMOTE', model, X_test, y_test,
        multi_class=multi, results_store=None
    )
    return model, y_pred