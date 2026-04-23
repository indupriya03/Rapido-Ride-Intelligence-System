# =============================================================================
# src/feature_engineering/zone2_config.py
# =============================================================================
# Central config for all Zone 2 decisions:
#   UC_CONFIG         — per-use-case target, task type, stratify flag
#   LEAKAGE_MAP       — columns that ALWAYS get dropped (reconstruct target)
#   FEATURE_SELECT_MAP— columns dropped for model quality, not leakage
#   PIPELINE_DEFAULTS — default knobs for get_splits()
# =============================================================================

UC_CONFIG = {
    'UC1': {
        'description' : 'Ride Outcome — multi-class classification',
        'target'      : 'booking_status_enc',
        'task'        : 'classification',
        'stratify'    : True,
        'shap_top_n'  : 40,
        'vehicle_types': None,
    },
    'UC2': {
        'description' : 'Fare Prediction — regression (per vehicle type)',
        'target'      : 'booking_value_log',
        'task'        : 'regression',
        'stratify'    : False,
        'shap_top_n'  : 35,
        'vehicle_types': ['Cab', 'Auto', 'Bike'],
    },
    'UC3': {
        'description' : 'Customer Cancellation Risk — binary classification',
        'target'      : 'is_cancelled',          # booking-level flag: booking_status == 'Cancelled'
        'task'        : 'classification',
        'stratify'    : True,
        'shap_top_n'  : 35,
        'vehicle_types': None,
    },
    'UC4': {
        'description' : 'Driver Delay Prediction — binary classification',
        'target'      : 'driver_delay_flag',
        'task'        : 'classification',
        'stratify'    : True,
        'shap_top_n'  : 30,
        'vehicle_types': None,
    },
}

# =============================================================================
# LEAKAGE MAP — columns that directly reconstruct the target
# These are ALWAYS removed regardless of any toggle flags.
# =============================================================================

LEAKAGE_MAP = {
    'UC1': [
        'booking_status_enc', 'booking_status', 'is_cancelled',   # target + aliases
        'fare_deviation', 'fare_above_loc_avg', 'fare_per_km_per_surge',
        'customer_cancel_flag',          # flags customers who cancelled → reconstructs class 1
        'cancelled_rides',               # raw count → reconstructs class 1
        'cancellation_rate',             # derived from cancelled_rides / total
        'cancel_to_booking_ratio',       # same family
        'completed_rides',               # reconstructs class 0
        'cust_completion_rate',          # derived from completed / total
        'incomplete_rides',              # raw count → reconstructs class 2
        'incomplete_ride_share',         # derived from incomplete / total
        'total_bookings',                # denominates all of the above
        'driver_delay_flag',
        # ── Interaction terms built from leaky cancellation_rate ──────────────
        # cancellation_rate is in leakage above; these interaction terms
        # were not listed but carry the same signal and survived the drop.
        # Confirmed corr with target: cancel_risk_x_peak=0.08, _night=0.04
        'cancel_risk_x_peak',            # cancellation_rate × peak_time_flag
        'cancel_risk_x_night',           # cancellation_rate × is_night_ride
        # reason_ dummies added dynamically in get_splits (only known post-ride)
    ],
    'UC2': [
        'booking_value', 'booking_value_log',
        'base_fare', 'fare_per_km', 'fare_per_min', 'expected_fare', 'base_x_surge',
        'fare_deviation', 'fare_above_loc_avg', 'fare_per_km_per_surge', 'surge_cost_share',
        'booking_status_enc', 'booking_status', 'is_cancelled', 'driver_delay_flag',
        
    ],
    'UC3': [
        # Target is is_cancelled — DO NOT list it here (get_splits skips the target automatically)
        'customer_cancel_flag',          # customer-level aggregate of cancellations — leakage
        'cancelled_rides', 'cancellation_rate', 'cancel_to_booking_ratio',
        'incomplete_ride_share', 'completed_rides', 'incomplete_rides',
        'total_bookings', 'cust_completion_rate',
        'cancel_risk_x_peak', 'cancel_risk_x_night', 'new_high_cancel', 'loyalty_x_cancel',
        'Customer_Loyalty_Score_uc1',
        'booking_status_enc', 'booking_status', 'booking_value_log', 'booking_value',
        'fare_deviation', 'fare_above_loc_avg',
        'reason_Customer No-Show', 'reason_Driver Delay',
        'reason_Not Applicable', 'reason_Vehicle Issue',
        'driver_delay_flag', 'delay_per_km',
    ],
    'UC4': [
        'driver_delay_flag',
        'delay_count', 'delay_rate', 'delay_per_ride', 'avg_pickup_delay_min',
        'high_delay_high_traffic', 'delay_x_traffic', 'delay_per_km',
        'is_unreliable_driver',
        'is_cancelled', 'booking_status_enc', 'booking_status',
        'fare_deviation', 'fare_above_loc_avg', 'booking_value_log', 'booking_value',
        'reason_Driver Delay', 'reason_Customer No-Show',
        'reason_Not Applicable', 'reason_Vehicle Issue',
        'cancelled_rides', 'incomplete_rides', 'completed_rides',
        'cancellation_rate', 'customer_cancel_flag', 'cancel_to_booking_ratio',
        'cust_completion_rate', 'incomplete_ride_share', 'total_bookings',
        'Customer_Loyalty_Score_uc3',
        # ── Synthetic DGP leakage — co-generated with delay_flag ─────────────
        # In the synthetic dataset, driver_delay_flag is defined as delay_rate > 0.10
        # (100% match confirmed). driver_incomplete_rides was generated from the
        # SAME latent "bad driver" variable — incomplete_rides==0 → zero delayed
        # drivers; incomplete_rides>=2 → near-perfect separation. This is not a
        # real causal relationship; it is an artefact of the synthetic DGP.
        # Confirmed: model with these features achieves AUC=1.0 (impossible in prod).
        'driver_incomplete_rides',       # co-generated with delay_flag in synthetic DGP
        'driver_incomplete_rate',        # driver_incomplete_rides / (accepted_rides + 1)
        'demand_x_incomplete_rate',      # interaction built on leaky column above
    ],
}

# =============================================================================
# FEATURE SELECTION MAP — excluded for model quality, not leakage
# =============================================================================

FEATURE_SELECT_MAP = {
    'UC1': [],
    'UC2': [
        'Customer_Loyalty_Score', 'loyalty_x_cancel', 'new_high_cancel', 'delay_per_km',
        'incomplete_ride_share', 'cancel_to_booking_ratio', 'cust_completion_rate',
        'is_low_rated_customer', 'vehicle_preference_match', 'customer_age_enc',
        'rejection_rate', 'demand_x_unreliable', 'demand_x_low_acceptance',
        'ride_difficulty', 'distance_x_traffic', 'exp_x_distance', 'rating_x_ridetime',
        'dual_low_rating', 'new_cust_x_unreliable', 'time_pressure', 'peak_x_distance',
        'demand_x_incomplete_rate', 'customer_tenure_years', 'is_new_customer',
        'customer_cancel_flag', 'cancel_risk_x_night', 'avg_wait_time_min', 'rejected_rides',
        'demand_supply_ratio',
    ],
    'UC3': [
        'demand_x_unreliable', 'demand_x_low_acceptance', 'ride_difficulty',
        'distance_x_traffic', 'exp_x_distance', 'rating_x_ridetime',
        'time_pressure', 'peak_x_distance', 'demand_x_incomplete_rate',
    ],
    'UC4': [
        'demand_x_unreliable', 'demand_x_low_acceptance', 'ride_difficulty',
        'distance_x_traffic', 'new_cust_x_unreliable',
        'customer_tenure_years', 'is_new_customer',
        'vehicle_preference_match', 'customer_age_enc',
    ],
}

# =============================================================================
# PIPELINE DEFAULTS
# =============================================================================

PIPELINE_DEFAULTS = {
    'test_size'          : 0.20,
    'random_state'       : 42,
    'run_corr_filter'    : True,
    'run_shap_filter'    : True,
    'corr_target_thresh' : 0.01,
    'corr_inter_thresh'  : 0.95,
}