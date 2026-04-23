# =============================================================================
# src/feature_engineering/zone1_pipeline.py
# =============================================================================
# Orchestrates all Zone 1 feature engineering steps in the correct order.
# Safe to run on the full dataset — no targets, no data-driven aggregations.
# =============================================================================

import pandas as pd
from .zone1_merge import merge_all_tables
from .zone1_features import (
    add_datetime_features,
    add_ride_features,
    add_customer_features,
    add_driver_features,
    add_location_features,
    add_interaction_features,
    encode_features,
    add_new_uc1_interactions,
    log_transform,
    drop_raw_columns,
)


def run_zone1_engineering(
    bookings_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    location_demand_df: pd.DataFrame,
    time_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full Zone 1 pipeline.

    Steps
    -----
    1.  Merge all 5 tables
    2.  Datetime features
    3.  Ride features
    4.  Customer features
    5.  Driver features
    6.  Location features
    7.  Interaction features
    8.  Encoding (OHE, ordinal, binary flags)
    9.  UC1 post-encoding interaction features
    10. Log transforms
    11. Drop raw/redundant columns

    Returns
    -------
    df : pd.DataFrame  — fully engineered, ready for Zone 2 splitting
    """
    print("\n" + "=" * 60)
    print("ZONE 1 — SAFE FEATURE ENGINEERING")
    print("=" * 60)

    df = merge_all_tables(bookings_df, customers_df, drivers_df,
                          location_demand_df, time_features_df)
    df = add_datetime_features(df)
    df = add_ride_features(df)
    df = add_customer_features(df)
    df = add_driver_features(df)
    df = add_location_features(df)
    df = add_interaction_features(df)
    df = encode_features(df)
    df = add_new_uc1_interactions(df)
    df = log_transform(df)
    df = drop_raw_columns(df)

    print(f"\n✅ Zone 1 complete. Final shape: {df.shape}")
    return df
