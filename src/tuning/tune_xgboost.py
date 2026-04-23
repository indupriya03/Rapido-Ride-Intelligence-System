# =============================================================================
# src/tuning/tune_xgboost.py
# =============================================================================
# Optuna hyperparameter search for XGBoostClassifier.
# Tuning is run on a 40% subset of training data for speed;
# the best params are returned for retraining on the full set.
# =============================================================================

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

SEED = 42


def tune_xgboost_classifier(
    X_train, y_train,
    X_test,  y_test,
    n_trials:    int  = 50,
    multi_class: bool = False,
    class_weight: str | None = None,
    seed: int = SEED,
    uc_name: str = '',
) -> tuple[dict, optuna.Study]:
    """
    Run Optuna TPE search over XGBoost hyperparameters.

    Objective metric:
        multi_class=True  → weighted F1
        multi_class=False → ROC-AUC

    Parameters
    ----------
    X_train, y_train : tuning subset (caller should pass 40% of full train)
    X_test,  y_test  : validation set (full test split)
    n_trials         : Optuna trial count
    multi_class      : True for UC1 (3-class), False for UC3/UC4 (binary)
    class_weight     : 'balanced' → sets scale_pos_weight for binary tasks

    Returns
    -------
    best_params : dict  (ready to unpack into XGBClassifier(**params, ...))
    study       : optuna.Study
    """
    def objective(trial):
        params = {
            'n_estimators'         : trial.suggest_int('n_estimators', 75, 300),
            'max_depth'            : trial.suggest_int('max_depth', 4, 7),
            'learning_rate'        : trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
            'subsample'            : trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree'     : trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_weight'     : trial.suggest_int('min_child_weight', 1, 10),
            'gamma'                : trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha'            : trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda'           : trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'early_stopping_rounds': 30,
            'random_state'         : seed,
            'eval_metric'          : 'mlogloss' if multi_class else 'logloss',
            'use_label_encoder'    : False,
            'n_jobs'               : -1,
        }
        if not multi_class and class_weight == 'balanced':
            params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        if multi_class:
            return f1_score(y_test, model.predict(X_test),
                            average='weighted', zero_division=0)
        else:
            return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    metric = 'F1 (weighted)' if multi_class else 'ROC-AUC'
    print(f"\n  {'─' * 50}")
    print(f"  {uc_name} XGBoost — Optuna best")
    print(f"  {'─' * 50}")
    print(f"  Trials    : {len(study.trials)}")
    print(f"  Best {metric:<15}: {study.best_value:.4f}")
    print(f"  Params    : {study.best_params}")

    return study.best_params, study
