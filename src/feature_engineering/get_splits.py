# =============================================================================
# src/feature_engineering/get_splits.py
# =============================================================================
# Zone 2 + Zone 3 + Feature Selection orchestrator.
#
# Public API
# ----------
# get_splits(df, use_case, ...)
#   UC1 / UC3 / UC4 → (X_train, X_test, y_train, y_test)
#   UC2             → {'Cab': (...), 'Auto': (...), 'Bike': (...)}
#
# Steps per use-case (per vehicle for UC2)
# ----------------------------------------
#   1. Drop LEAKAGE columns
#   2. Drop FEATURE SELECTION columns
#   3. Drop vehicle OHE cols + subset by vehicle  (UC2 only)
#   4. Train/test split  (stratified for classification)
#   5. Zone 3 engineering
#   6. Correlation filter
#   7. SHAP filter
# =============================================================================

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .zone2_config import (
    UC_CONFIG, LEAKAGE_MAP, FEATURE_SELECT_MAP, PIPELINE_DEFAULTS
)
from .zone3_pipeline import apply_zone3_features
from .feature_selection import correlation_filter, shap_filter


# =============================================================================
# INTERNAL — single pipeline pass
# =============================================================================

def _build_single_split(
    df_uc: pd.DataFrame,
    target_col: str,
    use_case: str,
    stratify: bool,
    test_size: float,
    random_state: int,
    run_corr_filter: bool,
    run_shap_filter: bool,
    shap_top_n: int | None,
    shap_threshold: float | None,
    corr_target_thresh: float,
    corr_inter_thresh: float,
    model_dir: str,
) -> tuple:
    X = df_uc.drop(columns=[target_col], errors='ignore')
    y = df_uc[target_col].copy()

    mask = y.notna()
    X, y = X[mask], y[mask]

    stratify_arr = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arr
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    nan_cols = X_train.columns[X_train.isna().any()]
    if len(nan_cols):
        print(f"\nNaN cols in X_train:\n"
              f"{X_train[nan_cols].isna().sum().sort_values(ascending=False)}")

    # Zone 3
    print(f"\nZONE 3 — {use_case}")
    X_train, X_test = apply_zone3_features(
        X_train, X_test, use_case=use_case, model_dir=model_dir
    )
    print(f"  After Zone 3: {X_train.shape}")

    # Correlation filter
    if run_corr_filter:
        X_train, X_test, corr_report = correlation_filter(
            X_train, X_test, y_train,
            target_corr_threshold=corr_target_thresh,
            inter_feature_threshold=corr_inter_thresh,
        )
        joblib.dump(corr_report,
                    os.path.join(model_dir, f'corr_report_{use_case.lower()}.pkl'))

    # SHAP filter
    if run_shap_filter:
        X_train, X_test, shap_report = shap_filter(
            X_train, X_test, y_train,
            use_case=use_case,
            top_n=shap_top_n,
            shap_threshold=shap_threshold,
            model_dir=model_dir,
        )
        joblib.dump(shap_report,
                    os.path.join(model_dir, f'shap_report_{use_case.lower()}.pkl'))

    return X_train, X_test, y_train, y_test


# =============================================================================
# PUBLIC — get_splits()
# =============================================================================

def get_splits(
    df: pd.DataFrame,
    use_case: str = 'UC1',
    test_size: float        = PIPELINE_DEFAULTS['test_size'],
    random_state: int       = PIPELINE_DEFAULTS['random_state'],
    run_corr_filter: bool   = PIPELINE_DEFAULTS['run_corr_filter'],
    run_shap_filter: bool   = PIPELINE_DEFAULTS['run_shap_filter'],
    shap_top_n: int | None      = None,
    shap_threshold: float | None = None,
    corr_target_thresh: float = PIPELINE_DEFAULTS['corr_target_thresh'],
    corr_inter_thresh: float  = PIPELINE_DEFAULTS['corr_inter_thresh'],
    model_dir: str = 'models',
):
    """
    Zone 2 + Zone 3 + Feature Selection for one use case.

    Returns
    -------
    UC1 / UC3 / UC4 : (X_train, X_test, y_train, y_test)
    UC2             : {'Cab': (...), 'Auto': (...), 'Bike': (...)}
    """
    uc_cfg      = UC_CONFIG[use_case]
    target_col  = uc_cfg['target']
    stratify    = uc_cfg['stratify']
    _shap_top_n = shap_top_n if shap_top_n is not None else uc_cfg['shap_top_n']

    leakage_cols     = LEAKAGE_MAP[use_case]
    feat_select_cols = FEATURE_SELECT_MAP[use_case]

    if use_case == 'UC1':
        reason_cols  = [c for c in df.columns if c.startswith('reason_')]
        leakage_cols = leakage_cols + reason_cols

    df_uc = df.copy()

    leakage_to_drop = [c for c in leakage_cols if c in df_uc.columns and c != target_col]
    fs_to_drop      = [c for c in feat_select_cols
                       if c in df_uc.columns and c != target_col and c not in leakage_to_drop]

    print(f"\n{'=' * 60}")
    print(f"ZONE 2 — {use_case}  ({uc_cfg['description']})")
    print(f"  Cols entering Zone 2       : {df_uc.shape[1]}")
    print(f"  Leakage cols dropped       : {len(leakage_to_drop)}")
    print(f"  Feature-select cols dropped: {len(fs_to_drop)}")

    df_uc = df_uc.drop(columns=leakage_to_drop + fs_to_drop, errors='ignore')

    # ── UC2: per-vehicle split ────────────────────────────────────────────────
    if use_case == 'UC2':
        vehicle_types  = uc_cfg['vehicle_types']

        vtype_ohe_cols = [c for c in df_uc.columns if c.startswith('vehicle_type_')]
        if vtype_ohe_cols:
            df_uc = df_uc.drop(columns=vtype_ohe_cols)
            print(f"  Vehicle OHE cols dropped   : {len(vtype_ohe_cols)}")

        if 'vehicle_type' not in df_uc.columns:
            raise KeyError(
                "UC2 requires 'vehicle_type' in df. Re-attach it from bookings_df "
                "after Zone 1 before calling get_splits(). See run_feature_engineering.py."
            )

        uc2_splits = {}
        for vtype in vehicle_types:
            subset = df_uc[df_uc['vehicle_type'] == vtype].drop(
                columns=['vehicle_type'], errors='ignore'
            ).copy()

            if len(subset) == 0:
                print(f"  ⚠️  No rows for vehicle_type='{vtype}' — skipping.")
                continue

            label   = f'UC2_{vtype}'
            veh_dir = os.path.join(model_dir, label.lower())
            os.makedirs(veh_dir, exist_ok=True)

            print(f"\n{'─' * 50}")
            print(f"  UC2 → {vtype}  ({len(subset):,} rows)")
            print(f"{'─' * 50}")

            uc2_splits[vtype] = _build_single_split(
                df_uc=subset,
                target_col=target_col,
                use_case=label,
                stratify=False,
                test_size=test_size,
                random_state=random_state,
                run_corr_filter=run_corr_filter,
                run_shap_filter=run_shap_filter,
                shap_top_n=_shap_top_n,
                shap_threshold=shap_threshold,
                corr_target_thresh=corr_target_thresh,
                corr_inter_thresh=corr_inter_thresh,
                model_dir=veh_dir,
            )

        print(f"\n✅ UC2 per-vehicle pipeline complete.")
        for vtype, (X_tr, X_te, _, _) in uc2_splits.items():
            print(f"  {vtype}: train {X_tr.shape}  test {X_te.shape}")
        return uc2_splits

    # ── UC1 / UC3 / UC4 ──────────────────────────────────────────────────────
    uc_dir = os.path.join(model_dir, use_case.lower())
    os.makedirs(uc_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = _build_single_split(
        df_uc=df_uc,
        target_col=target_col,
        use_case=use_case,
        stratify=stratify,
        test_size=test_size,
        random_state=random_state,
        run_corr_filter=run_corr_filter,
        run_shap_filter=run_shap_filter,
        shap_top_n=_shap_top_n,
        shap_threshold=shap_threshold,
        corr_target_thresh=corr_target_thresh,
        corr_inter_thresh=corr_inter_thresh,
        model_dir=uc_dir,
    )

    print(f"\n✅ {use_case} pipeline complete.")
    print(f"  Final train: {X_train.shape}  |  Final test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
