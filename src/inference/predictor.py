# =============================================================================
# src/inference/predictor.py
# =============================================================================
# Load the final model for a given use case and predict on a preprocessed row.
# Applies the stored probability threshold for UC3 and UC4.
#
# Usage
# -----
#   from src.inference.predictor import predict
#   result = predict(row_dict, use_case='UC3')
# =============================================================================

import numpy as np
import pandas as pd

from .preprocessor      import preprocess_row
from ..modeling.model_io import load_model, load_thresholds

# Human-readable label maps
UC1_LABELS = {0: 'Completed', 1: 'Cancelled', 2: 'Incomplete'}
UC3_LABELS = {0: 'Not Cancelled', 1: 'Cancelled'}
UC4_LABELS = {0: 'On Time', 1: 'Delayed'}


def predict(
    row: dict,
    use_case: str,
    model_dir: str = 'models',
    return_proba: bool = False,
) -> dict:
    """
    End-to-end prediction for a single booking row.

    Parameters
    ----------
    row          : raw booking dict (pre-join schema)
    use_case     : 'UC1' | 'UC2_Cab' | 'UC2_Auto' | 'UC2_Bike' | 'UC3' | 'UC4'
    model_dir    : directory containing *_final.pkl and thresholds.json
    return_proba : if True, include raw probabilities in the result dict

    Returns
    -------
    result : dict with keys:
        use_case, prediction, label (human-readable), probability (optional)
    """
    # ── Preprocess ────────────────────────────────────────────────────────────
    X = preprocess_row(row, use_case=use_case, model_dir=model_dir)

    # ── Load model ────────────────────────────────────────────────────────────
    model_name = use_case.lower() + '_final'
    model      = load_model(model_name, model_dir)

    # ── Predict ───────────────────────────────────────────────────────────────
    base_uc = use_case.split('_')[0]

    if base_uc == 'UC1':
        pred_class = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0].tolist()
        result = {
            'use_case'   : use_case,
            'prediction' : pred_class,
            'label'      : UC1_LABELS.get(pred_class, str(pred_class)),
        }
        if return_proba:
            result['probabilities'] = {UC1_LABELS[i]: round(p, 4)
                                       for i, p in enumerate(proba)}

    elif base_uc == 'UC2':
        pred_log  = float(model.predict(X)[0])
        pred_fare = float(np.expm1(pred_log))
        result = {
            'use_case'       : use_case,
            'predicted_fare' : round(pred_fare, 2),
        }

    elif base_uc in ('UC3', 'UC4'):
        thresholds = load_thresholds(model_dir)
        thresh_key = 'uc3_threshold' if base_uc == 'UC3' else 'uc4_threshold'
        threshold  = thresholds.get(thresh_key, 0.5)

        proba      = float(model.predict_proba(X)[0, 1])
        pred_class = int(proba >= threshold)
        labels_map = UC3_LABELS if base_uc == 'UC3' else UC4_LABELS

        result = {
            'use_case'  : use_case,
            'prediction': pred_class,
            'label'     : labels_map[pred_class],
            'threshold' : threshold,
        }
        if return_proba:
            result['probability'] = round(proba, 6)

    else:
        raise ValueError(f"Unknown use_case: {use_case}")

    return result
