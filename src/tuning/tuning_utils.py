# =============================================================================
# src/tuning/tuning_utils.py
# =============================================================================
# Save / load tuned hyperparameters to outputs/tuned_params.json.
# If the file exists, the tuning stage is skipped on re-runs.
# Delete the file to force a fresh Optuna run.
# =============================================================================

import os
import json

TUNED_PARAMS_PATH = os.path.join('outputs', 'tuned_params.json')


def save_tuned_params(params_dict: dict, path: str = TUNED_PARAMS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    print(f"  ✅ Tuned params saved → {path}")


def load_tuned_params(path: str = TUNED_PARAMS_PATH) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
        print(f"  ✅ Tuned params loaded from {path}")
        print(f"     Delete this file to re-run Optuna fresh.")
        return params
    return None
