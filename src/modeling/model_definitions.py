# =============================================================================
# src/modeling/model_definitions.py
# =============================================================================
# Factory functions that return dicts of {name: estimator}.
# All hyperparameters are sensible baselines — tuning lives in src/tuning/.
# =============================================================================

from sklearn.linear_model  import LogisticRegression, LinearRegression
from sklearn.ensemble      import RandomForestClassifier, RandomForestRegressor
from xgboost               import XGBClassifier, XGBRegressor
from lightgbm              import LGBMClassifier

SEED = 42


def get_classifiers(n_classes: int = 2, class_weight: str = 'balanced') -> dict:
    """
    Return baseline classifiers for UC1 (n_classes=3) or UC3/UC4 (n_classes=2).
    """
    multi = n_classes > 2

    return {
        'Logistic Regression': LogisticRegression(
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


def get_regressors() -> dict:
    """Return baseline regressors for UC2."""
    return {
        'Linear Regression': LinearRegression(n_jobs=-1),
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
