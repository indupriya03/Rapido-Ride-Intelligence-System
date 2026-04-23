# =============================================================================
# src/modeling/model_io.py
# =============================================================================
# Save / load all model artifacts:
#   save_model / load_model            — joblib pkl files
#   save_feature_cols / load_feature_cols — list of training columns per UC
#   save_thresholds / load_thresholds  — thresholds.json
#   save_report                        — model_selection_report.json
# =============================================================================

import os
import json
import joblib


def save_model(model, name: str, model_dir: str = 'models') -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'{name}.pkl')
    joblib.dump(model, path)
    print(f"  💾 Saved: {path}")
    return path


def load_model(name: str, model_dir: str = 'models'):
    path = os.path.join(model_dir, f'{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def save_feature_cols(columns: list, use_case: str, model_dir: str = 'models') -> str:
    """Save the exact column list the model was trained on."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'feature_cols_{use_case.lower()}.json')
    with open(path, 'w') as f:
        json.dump(list(columns), f, indent=2)
    print(f"  💾 Saved feature cols ({len(columns)}): {path}")
    return path


def load_feature_cols(use_case: str, model_dir: str = 'models') -> list:
    """Load the exact column list the model was trained on."""
    path = os.path.join(model_dir, f'feature_cols_{use_case.lower()}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature columns not found: {path}\n"
            f"Re-run run_training.py to regenerate."
        )
    with open(path) as f:
        return json.load(f)


def save_thresholds(thresholds: dict, model_dir: str = 'models') -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'thresholds.json')
    with open(path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"  💾 Saved: {path}")
    return path


def load_thresholds(model_dir: str = 'models') -> dict:
    path = os.path.join(model_dir, 'thresholds.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"thresholds.json not found in {model_dir}")
    with open(path) as f:
        return json.load(f)


def save_report(report: dict, output_dir: str = 'outputs') -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'model_selection_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  💾 Saved: {path}")
    return path