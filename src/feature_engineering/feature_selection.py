# =============================================================================
# src/feature_engineering/feature_selection.py
# =============================================================================
# Two feature selection steps applied after Zone 3, before model training:
#   Step A — Correlation filter  (near-zero target corr + inter-feature redund.)
#   Step B — SHAP filter         (fit fast RF → rank mean |SHAP| → keep top-N)
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =============================================================================
# STEP A — CORRELATION FILTER
# =============================================================================

def correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    target_corr_threshold: float = 0.01,
    inter_feature_threshold: float = 0.95,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Two-stage correlation-based feature selection.

    Stage 1 — drop features with |corr(feature, target)| < target_corr_threshold.
    Stage 2 — among survivors, if |corr(fi, fj)| >= inter_feature_threshold,
               drop the one with lower target correlation.

    Returns
    -------
    X_train_filtered, X_test_filtered, report (dict)
    """
    print("\n" + "-" * 50)
    print("FEATURE SELECTION — Correlation Filter")
    print("-" * 50)

    num_cols    = X_train.select_dtypes(include=[np.number]).columns.tolist()
    target_corr = X_train[num_cols].corrwith(y_train).abs()

    # Stage 1
    low_corr_cols = target_corr[target_corr < target_corr_threshold].index.tolist()
    if verbose:
        print(f"\nStage 1 — target corr < {target_corr_threshold}: dropping {len(low_corr_cols)}")

    # Stage 2
    surviving    = [c for c in num_cols if c not in low_corr_cols]
    corr_matrix  = X_train[surviving].corr().abs()
    redundant    = set()
    upper        = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    for col in upper.columns:
        if col in redundant:
            continue
        highly_corr = upper[col][upper[col] >= inter_feature_threshold].index.tolist()
        for partner in highly_corr:
            if partner in redundant:
                continue
            if target_corr.get(col, 0) >= target_corr.get(partner, 0):
                redundant.add(partner)
            else:
                redundant.add(col)

    if verbose:
        print(f"Stage 2 — inter-feature |corr| >= {inter_feature_threshold}: "
              f"dropping {len(redundant)}")

    cols_to_drop  = list(set(low_corr_cols) | redundant)
    X_train_out   = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
    X_test_out    = X_test.drop( columns=[c for c in cols_to_drop if c in X_test.columns])
    top_corr      = (target_corr
                     .drop(labels=cols_to_drop, errors='ignore')
                     .sort_values(ascending=False))

    print(f"\n✅ Correlation filter: {X_train.shape[1]} → {X_train_out.shape[1]} features")
    print(f"Top 15 by |target corr|:\n{top_corr.head(15).to_string()}")

    report = {
        'low_target_corr_dropped': low_corr_cols,
        'redundant_dropped'      : sorted(redundant),
        'all_dropped'            : cols_to_drop,
        'target_corr_ranking'    : top_corr,
    }
    return X_train_out, X_test_out, report


# =============================================================================
# STEP B — SHAP FILTER
# =============================================================================

def shap_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    use_case: str,
    top_n: int | None = None,
    shap_threshold: float | None = None,
    model_dir: str = 'models',
    n_estimators: int = 100,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    SHAP-based feature selection using a fast Random Forest.

    Exactly one of `top_n` or `shap_threshold` must be set.

    Returns
    -------
    X_train_filtered, X_test_filtered, report (dict)
    """
    if top_n is None and shap_threshold is None:
        raise ValueError("Provide either top_n or shap_threshold.")

    print("\n" + "-" * 50)
    print(f"FEATURE SELECTION — SHAP Filter  ({use_case})")
    print("-" * 50)

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Subsample for speed (max 5 000 rows)
    n_shap   = min(5_000, len(X_train))
    X_shap   = X_train[num_cols].sample(n=n_shap, random_state=random_state)
    y_shap   = y_train.loc[X_shap.index]

    base_uc = use_case.split('_')[0]
    if base_uc == 'UC2':
        model = RandomForestRegressor(
            n_estimators=50, max_depth=6, n_jobs=-1, random_state=random_state
        )
    else:
        model = RandomForestClassifier(
            n_estimators=50, max_depth=6, n_jobs=-1,
            random_state=random_state, class_weight='balanced'
        )

    model.fit(X_shap, y_shap)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        shap_arr = np.abs(shap_values)
        if shap_arr.ndim == 3:
            mean_abs_shap = shap_arr.mean(axis=0).mean(axis=-1)
        else:
            mean_abs_shap = shap_arr.mean(axis=0)

    shap_series = pd.Series(mean_abs_shap, index=num_cols).sort_values(ascending=False)

    if top_n is not None:
        keep_features = shap_series.head(top_n).index.tolist()
        print(f"Keeping top {top_n} features by mean |SHAP|.")
    else:
        keep_features = shap_series[shap_series >= shap_threshold].index.tolist()
        print(f"Keeping {len(keep_features)} features with mean |SHAP| >= {shap_threshold}.")

    dropped_features = [c for c in num_cols if c not in keep_features]
    non_num_cols     = [c for c in X_train.columns if c not in num_cols]
    final_keep       = keep_features + non_num_cols

    X_train_out = X_train[[c for c in final_keep if c in X_train.columns]]
    X_test_out  = X_test[[c  for c in final_keep if c in X_test.columns]]

    if verbose:
        print(f"\nTop 20 features by mean |SHAP|:\n{shap_series.head(20).to_string()}")
        print(f"\nDropped {len(dropped_features)} low-SHAP features.")

    print(f"\n✅ SHAP filter: {X_train.shape[1]} → {X_train_out.shape[1]} features")

    # Persist importance ranking
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(shap_series, os.path.join(model_dir, f'shap_importance_{use_case.lower()}.pkl'))

    report = {
        'shap_importance'  : shap_series,
        'kept_features'    : keep_features,
        'dropped_features' : dropped_features,
    }
    return X_train_out, X_test_out, report
