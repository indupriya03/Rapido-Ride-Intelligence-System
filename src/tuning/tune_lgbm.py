# =============================================================================
# src/tuning/tune_lgbm.py
# =============================================================================
# Optuna hyperparameter search for LGBMClassifier.
# Mirrors tune_xgboost.py interface exactly so retrain_tuned.py can treat
# both results identically.
# =============================================================================

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import f1_score, roc_auc_score
from lightgbm import LGBMClassifier

SEED = 42


def tune_lgbm_classifier(
    X_train, y_train,
    X_test,  y_test,
    n_trials:     int  = 50,
    multi_class:  bool = False,
    class_weight: str | None = None,
    seed: int = SEED,
    uc_name: str = '',
) -> tuple[dict, optuna.Study]:
    """
    Run Optuna TPE search over LightGBM hyperparameters.

    Objective metric:
        multi_class=True  → weighted F1
        multi_class=False → ROC-AUC

    Returns
    -------
    best_params : dict
    study       : optuna.Study
    """
    def objective(trial):
        params = {
            'n_estimators'      : trial.suggest_int('n_estimators', 75, 300),
            'max_depth'         : trial.suggest_int('max_depth', 4, 7),
            'learning_rate'     : trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
            'num_leaves'        : trial.suggest_int('num_leaves', 20, 100),
            'subsample'         : trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree'  : trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_samples' : trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha'         : trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda'        : trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state'      : seed,
            'n_jobs'            : -1,
            'verbose'           : -1,
        }
        if not multi_class and class_weight == 'balanced':
            params['is_unbalance'] = True

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)])

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
    print(f"  {uc_name} LightGBM — Optuna best")
    print(f"  {'─' * 50}")
    print(f"  Trials    : {len(study.trials)}")
    print(f"  Best {metric:<15}: {study.best_value:.4f}")
    print(f"  Params    : {study.best_params}")

    return study.best_params, study
