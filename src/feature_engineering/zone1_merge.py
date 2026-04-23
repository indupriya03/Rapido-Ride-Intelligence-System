# =============================================================================
# src/feature_engineering/zone1_merge.py
# =============================================================================
# Merges all 5 source tables into one flat DataFrame.
# All joins are LEFT joins on bookings_df to preserve every booking row.
# =============================================================================

import pandas as pd


def merge_all_tables(bookings_df, customers_df, drivers_df,
                     location_demand_df, time_features_df) -> pd.DataFrame:
    """
    Join bookings ← customers ← drivers ← time_features ← location_demand.

    Returns
    -------
    df : pd.DataFrame  (same row count as bookings_df)
    """
    print("Original shapes:")
    print(f"  bookings_df       : {bookings_df.shape}")
    print(f"  customers_df      : {customers_df.shape}")
    print(f"  drivers_df        : {drivers_df.shape}")
    print(f"  location_demand_df: {location_demand_df.shape}")
    print(f"  time_features_df  : {time_features_df.shape}")

    df = bookings_df.copy()

    # ── 1. Customers ──────────────────────────────────────────────────────────
    cust_bring = [
        'customer_id', 'customer_gender', 'customer_age', 'customer_city',
        'customer_signup_days_ago', 'preferred_vehicle_type',
        'total_bookings', 'completed_rides', 'cancelled_rides',
        'incomplete_rides', 'cancellation_rate', 'avg_customer_rating',
        'customer_cancel_flag',
    ]
    cust_bring = [c for c in cust_bring if c in customers_df.columns]
    df = df.merge(customers_df[cust_bring], on='customer_id', how='left')
    print(f"After customers : {df.shape}")

    # ── 2. Drivers ────────────────────────────────────────────────────────────
    drivers_clean = drivers_df.rename(columns={
        'incomplete_rides': 'driver_incomplete_rides',
        'city':             'driver_city',
    }).drop(columns=['vehicle_type'], errors='ignore')

    driv_bring = [
        'driver_id', 'driver_age', 'driver_city', 'driver_experience_years',
        'total_assigned_rides', 'accepted_rides', 'driver_incomplete_rides',
        'delay_count', 'acceptance_rate', 'delay_rate', 'avg_driver_rating',
        'avg_pickup_delay_min', 'driver_delay_flag', 'experience_outlier_flag',
        'rejected_rides',
    ]
    driv_bring = [c for c in driv_bring if c in drivers_clean.columns]
    df = df.merge(drivers_clean[driv_bring], on='driver_id', how='left')
    print(f"After drivers   : {df.shape}")

    # ── 3. Time features ──────────────────────────────────────────────────────
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['merge_hour'] = df['booking_datetime'].dt.floor('h')

    time_clean = time_features_df.copy()
    time_clean['merge_hour'] = pd.to_datetime(time_clean['datetime'])

    assert time_clean['merge_hour'].is_unique, \
        "time_features must have unique hourly timestamps."

    time_bring = ['merge_hour', 'peak_time_flag', 'season']
    time_bring = [c for c in time_bring if c in time_clean.columns]
    time_clean = time_clean[time_bring].drop_duplicates('merge_hour')

    df = df.merge(time_clean, on='merge_hour', how='left')
    df = df.drop(columns=['merge_hour'], errors='ignore')
    print(f"After time      : {df.shape}")

    # ── 4. Location demand ────────────────────────────────────────────────────
    loc_clean = location_demand_df.rename(columns={
        'completed_rides': 'loc_completed_rides',
        'cancelled_rides': 'loc_cancelled_rides',
        'total_requests':  'loc_total_requests',
    })
    loc_bring = [
        'city', 'pickup_location', 'hour_of_day', 'vehicle_type',
        'loc_total_requests', 'loc_completed_rides', 'loc_cancelled_rides',
        'avg_wait_time_min', 'avg_surge_multiplier', 'demand_level',
    ]
    loc_bring = [c for c in loc_bring if c in loc_clean.columns]
    loc_clean = loc_clean[loc_bring].drop_duplicates(
        ['city', 'pickup_location', 'hour_of_day', 'vehicle_type']
    )
    df = df.merge(loc_clean,
                  on=['city', 'pickup_location', 'hour_of_day', 'vehicle_type'],
                  how='left')
    print(f"After location  : {df.shape}")

    # Sanity checks
    suffix_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    if suffix_cols:
        print(f"\n⚠️  Suffix columns still present: {suffix_cols}")
    else:
        print("\n✅ No suffix columns.")

    if len(df) != len(bookings_df):
        print(f"❌ Row count changed! {len(bookings_df)} → {len(df)}")
    else:
        print(f"✅ Row count preserved: {len(df):,}")

    print(f"\nFinal merged shape: {df.shape}")
    return df
