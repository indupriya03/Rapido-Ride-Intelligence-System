# =============================================================================
# src/modeling/model_selection.py
# =============================================================================
# Compare baseline vs tuned metrics for each use case.
# Save the winner as uc*_final.pkl and write model_selection_report.json.
# =============================================================================

import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, r2_score

from .model_io import save_model, load_model, save_report
from .model_utils import evaluate_classifier, evaluate_regressor, clean_for_sklearn


def _classification_score(model, X_test, y_test, multi_class: bool) -> float:
    """Return weighted F1 (multi) or ROC-AUC (binary)."""
    if multi_class:
        return f1_score(y_test, model.predict(X_test),
                        average='weighted', zero_division=0)
    try:
        return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        return f1_score(y_test, model.predict(X_test),
                        average='binary', zero_division=0)


def _regression_score(model, X_test, y_test) -> float:
    """Return R² on back-transformed predictions."""
    from sklearn.metrics import r2_score
    return r2_score(np.expm1(y_test), np.expm1(model.predict(X_test)))


def select_and_save_finals(
    splits: dict,
    model_dir: str = 'models',
    output_dir: str = 'outputs',
) -> dict:
    """
    Load baseline and tuned models for each UC, compare on test set,
    save the winner as uc*_final.pkl.

    Parameters
    ----------
    splits : {
        'uc1': (X_train, X_test, y_train, y_test),
        'uc2': {'Cab': (...), 'Auto': (...), 'Bike': (...)},
        'uc3': (X_train, X_test, y_train, y_test),
        'uc4': (X_train, X_test, y_train, y_test),
    }

    Returns
    -------
    report : dict  — also written to outputs/model_selection_report.json
    """
    report  = {}
    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── UC1 ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = splits['uc1']
    X_tr, X_te = clean_for_sklearn(X_train, X_test)

    baseline = load_model('uc1_baseline', model_dir)
    b_score  = _classification_score(baseline, X_te, y_test, multi_class=True)

    try:
        tuned   = load_model('uc1_tuned', model_dir)
        t_score = _classification_score(tuned, X_te, y_test, multi_class=True)
    except FileNotFoundError:
        tuned, t_score = None, -1

    winner      = tuned if (tuned is not None and t_score > b_score) else baseline
    winner_name = 'tuned' if (tuned is not None and t_score > b_score) else 'baseline'
    save_model(winner, 'uc1_final', model_dir)

    report['UC1'] = {
        'metric': 'weighted_f1',
        'baseline': round(b_score, 4),
        'tuned'   : round(t_score, 4) if tuned else 'N/A',
        'winner'  : winner_name,
    }
    print(f"\nUC1 → baseline={b_score:.4f}  tuned={t_score:.4f}  winner={winner_name}")

    # ── UC2 (per vehicle — no tuning stage; baseline IS final) ───────────────
    for vtype in ['Cab', 'Auto', 'Bike']:
        X_train, X_test, y_train, y_test = splits['uc2'][vtype]
        X_tr, X_te = clean_for_sklearn(X_train, X_test)
        tag        = f'uc2_{vtype.lower()}'

        baseline   = load_model(f'{tag}_baseline', model_dir)
        b_score    = _regression_score(baseline, X_te, y_test)
        save_model(baseline, f'{tag}_final', model_dir)

        report[f'UC2_{vtype}'] = {
            'metric'  : 'r2',
            'baseline': round(b_score, 4),
            'tuned'   : 'N/A',
            'winner'  : 'baseline',
        }
        print(f"UC2-{vtype} → baseline R²={b_score:.4f}  (no tuning — baseline is final)")

    # ── UC3 ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = splits['uc3']
    X_tr, X_te = clean_for_sklearn(X_train, X_test)

    baseline = load_model('uc3_baseline', model_dir)
    b_score  = _classification_score(baseline, X_te, y_test, multi_class=False)

    try:
        tuned   = load_model('uc3_tuned', model_dir)
        t_score = _classification_score(tuned, X_te, y_test, multi_class=False)
    except FileNotFoundError:
        tuned, t_score = None, -1

    winner      = tuned if (tuned is not None and t_score > b_score) else baseline
    winner_name = 'tuned' if (tuned is not None and t_score > b_score) else 'baseline'
    save_model(winner, 'uc3_final', model_dir)

    report['UC3'] = {
        'metric'  : 'roc_auc',
        'baseline': round(b_score, 4),
        'tuned'   : round(t_score, 4) if tuned else 'N/A',
        'winner'  : winner_name,
    }
    print(f"UC3 → baseline={b_score:.4f}  tuned={t_score:.4f}  winner={winner_name}")

    # ── UC4 ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = splits['uc4']
    X_tr, X_te = clean_for_sklearn(X_train, X_test)

    baseline = load_model('uc4_baseline', model_dir)
    b_score  = _classification_score(baseline, X_te, y_test, multi_class=False)

    try:
        tuned   = load_model('uc4_tuned', model_dir)
        t_score = _classification_score(tuned, X_te, y_test, multi_class=False)
    except FileNotFoundError:
        tuned, t_score = None, -1

    winner      = tuned if (tuned is not None and t_score > b_score) else baseline
    winner_name = 'tuned' if (tuned is not None and t_score > b_score) else 'baseline'
    save_model(winner, 'uc4_final', model_dir)

    report['UC4'] = {
        'metric'  : 'roc_auc',
        'baseline': round(b_score, 4),
        'tuned'   : round(t_score, 4) if tuned else 'N/A',
        'winner'  : winner_name,
    }
    print(f"UC4 → baseline={b_score:.4f}  tuned={t_score:.4f}  winner={winner_name}")

    save_report(report, output_dir)
    print("\n✅ Model selection complete.")
    return report
