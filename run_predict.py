# =============================================================================
# run_predict.py  — Stage 5
# =============================================================================
# Two modes:
#
#   python run_predict.py
#       → Single SAMPLE_ROW inference across UC1, UC2_Cab, UC3, UC4
#
#   python run_predict.py --batch [--n 100] [--seed 42] [--uc UC1 UC3]
#       → Load real test-split data, sample N rows, run all (or selected) UCs,
#         print a summary table + flag suspicious predictions (e.g. prob=0.0)
#
# Requires: model training + feature engineering to have completed.
# =============================================================================

import sys, os, argparse,warnings
import numpy as np
import pandas as pd

# Suppress sklearn pickle version mismatch warnings from joblib model loads
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*feature_names.*', category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))
from src.inference.predictor import predict

# =============================================================================
# SAMPLE INPUT — single-row demo
# =============================================================================

SAMPLE_ROW = {
    # Booking fields
    'booking_datetime'        : '2024-06-15 08:30:00',
    'hour_of_day'             : 8,
    'day_of_week'             : 5,
    'vehicle_type'            : 'Cab',
    'ride_distance_km'        : 12.5,
    'base_fare'               : 150.0,
    'surge_multiplier'        : 1.5,
    'estimated_ride_time_min' : 25,
    'booking_value'           : 225.0,
    'traffic_level'           : 'High',
    'weather_condition'       : 'Clear',
    'pickup_location'         : 'Andheri',
    'drop_location'           : 'Bandra',
    'city'                    : 'Mumbai',

    # Customer fields
    'customer_id'             : 'C001',
    'customer_age'            : 32,
    'customer_gender'         : 'Male',
    'customer_city'           : 'Mumbai',
    'customer_signup_days_ago': 400,
    'preferred_vehicle_type'  : 'Cab',
    'total_bookings'          : 45,
    'completed_rides'         : 38,
    'cancelled_rides'         : 5,
    'incomplete_rides'        : 2,
    'cancellation_rate'       : 0.11,
    'avg_customer_rating'     : 4.2,
    'customer_cancel_flag'    : 0,

    # Driver fields
    'driver_id'               : 'D001',
    'driver_age'              : 28,
    'driver_city'             : 'Mumbai',
    'driver_experience_years' : 1,
    'total_assigned_rides'    : 80,
    'accepted_rides'          : 52,
    'driver_incomplete_rides' : 8,
    'delay_count'             : 12,
    'rejected_rides'          : 28,
    'acceptance_rate'         : 0.55,
    'delay_rate'              : 0.04,
    'avg_driver_rating'       : 3.2,
    'avg_pickup_delay_min'    : 2.1,
    'driver_delay_flag'       : 0,
    'experience_outlier_flag' : 0,

    # Location / demand fields
    'loc_total_requests'      : 180,
    'loc_completed_rides'     : 145,
    'loc_cancelled_rides'     : 25,
    'avg_wait_time_min'       : 4.5,
    'avg_surge_multiplier'    : 1.3,
    'demand_level'            : 'High',

    # Time fields
    'peak_time_flag'          : 1,
    'season'                  : 'Summer',

    # Misc
    'incomplete_ride_reason'  : 'Not Applicable',
    'same_loc_flag'           : 0,
    'booking_status'          : 'Completed',  # known for demo; excluded as leakage at train time
}


# =============================================================================
# SINGLE-ROW MODE
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("INFERENCE — SAMPLE BOOKING")
    print("=" * 60)

    for uc in ['UC1', 'UC2_Cab', 'UC3', 'UC4']:
        print(f"\n-- {uc} --")
        try:
            result = predict(SAMPLE_ROW, use_case=uc, return_proba=True)
            for k, v in result.items():
                print(f"  {k:<22}: {v}")
        except FileNotFoundError as e:
            print(f"  WARNING  {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nInference complete.")





   