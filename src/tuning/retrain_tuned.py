# =============================================================================
# src/tuning/retrain_tuned.py
# =============================================================================
# After Optuna finds best params on the 40% tuning subset,
# retrain_and_evaluate() rebuilds both XGBoost and LightGBM on the FULL
# training set, picks the winner by F1, and returns the fitted model.
# =============================================================================

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

from ..modeling.model_utils import plot_confusion_matrix, plot_feature_importance
from ..modeling.model_io    import save_model

SEED = 42


def retrain_and_evaluate(
    uc_name:     str,
    xgb_params:  dict,
    lgbm_params: dict,
    X_train, y_train,
    X_test,  y_test,
    multi_class:  bool = False,
    class_weight: str | None = None,
    labels:       list | None = None,
    save_prefix:  str  | None = None,
    model_dir:    str = 'models',
    output_dir:   str = 'outputs',
    seed: int = SEED,
) -> tuple:
    """
    Retrain XGBoost and LightGBM on full training data using tuned params.
    Pick the winner by F1, print full evaluation, save as uc*_tuned.pkl.

    Returns
    -------
    best_model, best_label, accuracy, f1
    """
    avg = 'weighted' if multi_class else 'binary'

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_kw = {
        **xgb_params,
        'eval_metric'      : 'mlogloss' if multi_class else 'logloss',
        'use_label_encoder': False,
        'random_state'     : seed,
        'n_jobs'           : -1,
    }
    # Remove early_stopping_rounds — not valid without eval_set at predict time
    xgb_kw.pop('early_stopping_rounds', None)

    if not multi_class and class_weight == 'balanced':
        xgb_kw['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_final = XGBClassifier(**xgb_kw)
    xgb_final.fit(X_train, y_train, verbose=False)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgbm_kw = {
        **lgbm_params,
        'random_state': seed,
        'n_jobs'      : -1,
        'verbose'     : -1,
    }
    if not multi_class and class_weight == 'balanced':
        lgbm_kw['is_unbalance'] = True

    lgbm_final = LGBMClassifier(**lgbm_kw)
    lgbm_final.fit(X_train, y_train)

    # ── Pick winner ───────────────────────────────────────────────────────────
    xgb_f1  = f1_score(y_test, xgb_final.predict(X_test),  average=avg, zero_division=0)
    lgbm_f1 = f1_score(y_test, lgbm_final.predict(X_test), average=avg, zero_division=0)

    best_model  = xgb_final  if xgb_f1 >= lgbm_f1 else lgbm_final
    best_label  = 'XGBoost'  if xgb_f1 >= lgbm_f1 else 'LightGBM'

    y_pred = best_model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average=avg, zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"{uc_name} — TUNED  [{best_label}]")
    print(f"{'=' * 60}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 ({avg:>8}): {f1:.4f}")
    if not multi_class:
        auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        print(f"  ROC-AUC       : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    if save_prefix:
        plot_confusion_matrix(
            y_test, y_pred,
            labels=labels or [],
            title=f"{uc_name} Confusion — Tuned {best_label}",
            save_path=f"{output_dir}/{save_prefix}_tuned_confusion_matrix.png",
        )
        plot_feature_importance(
            best_model,
            [c.replace('[', '_').replace(']', '_').replace('<', '_')
             for c in X_test.columns],
            title=f"{uc_name} Feature Importance — Tuned {best_label}",
            save_path=f"{output_dir}/{save_prefix}_tuned_feature_importance.png",
        )

    save_model(best_model, f'{save_prefix}_tuned' if save_prefix else uc_name.lower() + '_tuned',
               model_dir)
    return best_model, best_label, acc, f1
