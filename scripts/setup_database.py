# =============================================================================
# SETUP DATABASE PIPELINE — Rapido Ride Intelligence System
# =============================================================================
# Single script that:
#   1. Connects to MySQL
#   2. Creates all tables
#   3. Cleans and optimizes data types
#   4. Inserts all 5 tables
#   5. Verifies row counts
#
# Run ONCE from project root:
#   python setup_database.py
# =============================================================================

import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
import sys
import os

project_root = os.path.abspath(os.getcwd())
print(f"Project root: {project_root}")
sys.path.append(os.path.join(project_root, "app/utils"))
from db import get_engine, test_connection

sys.path.append(os.path.join(project_root, "src"))
from data_loader import load_cleaned_data

# =============================================================================
# SECTION 1 — TEST CONNECTION
# =============================================================================
print("=" * 55)
print("Rapido — Database Setup Pipeline")
print("=" * 55)

if not test_connection():
    print("❌ Cannot connect to MySQL. Check your db.py credentials.")
    sys.exit(1)

# =============================================================================
# SECTION 2 — CREATE TABLES
# =============================================================================
CREATE_STATEMENTS = {

'bookings': """
CREATE TABLE IF NOT EXISTS bookings (
    booking_id              BIGINT          PRIMARY KEY,
    day_of_week             VARCHAR(10),
    is_weekend              TINYINT,
    hour_of_day             TINYINT,
    city                    VARCHAR(50),
    pickup_location         VARCHAR(50),
    drop_location           VARCHAR(50),
    vehicle_type            VARCHAR(20),
    ride_distance_km        FLOAT,
    estimated_ride_time_min FLOAT,
    actual_ride_time_min    FLOAT,
    traffic_level           VARCHAR(20),
    weather_condition       VARCHAR(50),
    base_fare               FLOAT,
    surge_multiplier        FLOAT,
    booking_value           FLOAT,
    booking_status          VARCHAR(20),
    incomplete_ride_reason  VARCHAR(100),
    customer_id             VARCHAR(20),
    driver_id               VARCHAR(20),
    booking_datetime        DATETIME,
    actual_time_available   TINYINT,
    same_loc_flag           TINYINT,
    INDEX idx_city          (city),
    INDEX idx_hour          (hour_of_day),
    INDEX idx_status        (booking_status),
    INDEX idx_customer      (customer_id),
    INDEX idx_driver        (driver_id),
    INDEX idx_datetime      (booking_datetime)
)
""",

'customers': """
CREATE TABLE IF NOT EXISTS customers (
    customer_id              VARCHAR(20) PRIMARY KEY,
    customer_gender          VARCHAR(20),
    customer_age             SMALLINT,
    customer_city            VARCHAR(50),
    customer_signup_days_ago INT,
    preferred_vehicle_type   VARCHAR(20),
    total_bookings           INT,
    completed_rides          INT,
    cancelled_rides          INT,
    incomplete_rides         INT,
    cancellation_rate        FLOAT,
    avg_customer_rating      FLOAT,
    customer_cancel_flag     TINYINT,
    INDEX idx_city           (customer_city),
    INDEX idx_cancel_flag    (customer_cancel_flag)
)
""",

'drivers': """
CREATE TABLE IF NOT EXISTS drivers (
    driver_id               VARCHAR(20) PRIMARY KEY,
    driver_age              SMALLINT,
    driver_city             VARCHAR(50),
    vehicle_type            VARCHAR(20),
    driver_experience_years TINYINT,
    total_assigned_rides    INT,
    accepted_rides          INT,
    incomplete_rides        INT,
    delay_count             INT,
    acceptance_rate         FLOAT,
    delay_rate              FLOAT,
    avg_driver_rating       FLOAT,
    avg_pickup_delay_min    FLOAT,
    driver_delay_flag       TINYINT,
    experience_outlier_flag TINYINT,
    rejected_rides          INT,
    INDEX idx_city          (driver_city),
    INDEX idx_delay_flag    (driver_delay_flag)
)
""",

'location_demand': """
CREATE TABLE IF NOT EXISTS location_demand (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    city                 VARCHAR(50),
    pickup_location      VARCHAR(50),
    hour_of_day          TINYINT,
    vehicle_type         VARCHAR(20),
    total_requests       INT,
    completed_rides      INT,
    cancelled_rides      INT,
    avg_wait_time_min    FLOAT,
    avg_surge_multiplier FLOAT,
    demand_level         VARCHAR(20),
    INDEX idx_city_hour  (city, hour_of_day),
    INDEX idx_location   (pickup_location)
)
""",

'time_features': """
CREATE TABLE IF NOT EXISTS time_features (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    datetime       DATETIME,
    hour_of_day    TINYINT,
    day_of_week    VARCHAR(15),
    is_weekend     TINYINT,
    is_holiday     TINYINT,
    peak_time_flag TINYINT,
    season         VARCHAR(20),
    INDEX idx_hour (hour_of_day),
    INDEX idx_dt   (datetime)
)
""",

'predictions': """
CREATE TABLE IF NOT EXISTS predictions (
    id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
    booking_id              BIGINT NOT NULL,
    predicted_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_version           VARCHAR(50) NOT NULL,
    prediction_type         VARCHAR(30) DEFAULT 'ride_inference',
    city                    VARCHAR(50),
    vehicle_type            VARCHAR(20),
    hour_of_day             TINYINT,
    ride_distance_km        FLOAT,
    surge_multiplier        FLOAT,
    demand_level            VARCHAR(20),
    is_weekend              TINYINT(1),
    peak_time_flag          TINYINT(1),
    predicted_fare          FLOAT,
    predicted_ride_time_min FLOAT,
    cancel_probability      FLOAT,
    cancel_risk_tier        VARCHAR(20),
    uc3_threshold_used      FLOAT,
    recommended_action      VARCHAR(50),
    actual_completed_flag   TINYINT(1),
    actual_fare             FLOAT,
    actual_ride_time_min    FLOAT,
    actual_cancelled_flag   TINYINT(1),
    FOREIGN KEY (booking_id) REFERENCES bookings(booking_id),
    UNIQUE KEY uq_booking_model (booking_id, model_version),
    INDEX idx_booking_id    (booking_id),
    INDEX idx_predicted_at  (predicted_at),
    INDEX idx_model_version (model_version),
    INDEX idx_risk_tier     (cancel_risk_tier),
    INDEX idx_city          (city)
)
""",
}

print("\n📋 Creating tables...")
engine = get_engine()
with engine.connect() as conn:
    # Drop in reverse order to respect foreign keys
    for table in ['predictions', 'time_features', 'location_demand',
                  'drivers', 'customers', 'bookings']:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    conn.commit()

    for table_name, ddl in CREATE_STATEMENTS.items():
        conn.execute(text(ddl))
        conn.commit()
        print(f"  ✅ {table_name}")

# =============================================================================
# SECTION 3 — LOAD RAW DATA
# =============================================================================
print("\n📂 Loading cleaned data...")
bookings_df, customers_df, drivers_df, location_demand_df, time_features_df = load_cleaned_data()
print(f"  bookings        : {bookings_df.shape}")
print(f"  customers       : {customers_df.shape}")
print(f"  drivers         : {drivers_df.shape}")
print(f"  location_demand : {location_demand_df.shape}")
print(f"  time_features   : {time_features_df.shape}")

# =============================================================================
# SECTION 4 — CLEAN & OPTIMIZE DATA TYPES
# =============================================================================
print("\n🔧 Cleaning and optimizing data types...")

# ── Bookings ──────────────────────────────────────────────
bookings_df['actual_ride_time_min'] = bookings_df['actual_ride_time_min'].fillna(0)
bookings_df.loc[bookings_df['booking_status'] != 'Completed', 'actual_ride_time_min'] = 0
bookings_df['booking_id']       = bookings_df['booking_id'].str.extract(r'(\d+)').astype('int64')
bookings_df['booking_datetime'] = pd.to_datetime(bookings_df['booking_datetime'])

for col in ['day_of_week', 'city', 'pickup_location', 'drop_location',
            'vehicle_type', 'traffic_level', 'weather_condition',
            'booking_status', 'incomplete_ride_reason']:
    bookings_df[col] = bookings_df[col].astype('category')

bookings_df['is_weekend']             = bookings_df['is_weekend'].astype('int8')
bookings_df['same_loc_flag']          = bookings_df['same_loc_flag'].astype('int8')
bookings_df['actual_time_available']  = bookings_df['actual_time_available'].astype('int8')
bookings_df['hour_of_day']            = bookings_df['hour_of_day'].astype('int8')
bookings_df['ride_distance_km']       = bookings_df['ride_distance_km'].astype('float32')
bookings_df['estimated_ride_time_min']= bookings_df['estimated_ride_time_min'].astype('float32')
bookings_df['actual_ride_time_min']   = bookings_df['actual_ride_time_min'].astype('float32')
bookings_df['base_fare']              = bookings_df['base_fare'].astype('float32')
bookings_df['surge_multiplier']       = bookings_df['surge_multiplier'].astype('float32')
bookings_df['booking_value']          = bookings_df['booking_value'].astype('float32')
print("  ✅ bookings")

# ── Customers ─────────────────────────────────────────────
for col in ['customer_gender', 'customer_city', 'preferred_vehicle_type']:
    customers_df[col] = customers_df[col].astype('category')

customers_df['customer_age']             = customers_df['customer_age'].astype('int16')
customers_df['customer_signup_days_ago'] = customers_df['customer_signup_days_ago'].astype('int32')
customers_df['cancellation_rate']        = customers_df['cancellation_rate'].astype('float32')
customers_df['avg_customer_rating']      = customers_df['avg_customer_rating'].astype('float32')
customers_df['customer_cancel_flag']     = customers_df['customer_cancel_flag'].astype('int8')
customers_df['total_bookings']           = customers_df['total_bookings'].astype('int32')
customers_df['completed_rides']          = customers_df['completed_rides'].astype('int32')
customers_df['cancelled_rides']          = customers_df['cancelled_rides'].astype('int32')
customers_df['incomplete_rides']         = customers_df['incomplete_rides'].astype('int32')
print("  ✅ customers")

# ── Drivers ───────────────────────────────────────────────
for col in ['driver_city', 'vehicle_type']:
    drivers_df[col] = drivers_df[col].astype('category')

drivers_df['driver_age']              = drivers_df['driver_age'].astype('int16')
drivers_df['driver_experience_years'] = drivers_df['driver_experience_years'].astype('int8')
drivers_df['acceptance_rate']         = drivers_df['acceptance_rate'].astype('float32')
drivers_df['delay_rate']              = drivers_df['delay_rate'].astype('float32')
drivers_df['avg_driver_rating']       = drivers_df['avg_driver_rating'].astype('float32')
drivers_df['avg_pickup_delay_min']    = drivers_df['avg_pickup_delay_min'].astype('float32')
drivers_df['driver_delay_flag']       = drivers_df['driver_delay_flag'].astype('int8')
drivers_df['experience_outlier_flag'] = drivers_df['experience_outlier_flag'].astype('int8')
drivers_df['total_assigned_rides']    = drivers_df['total_assigned_rides'].astype('int32')
drivers_df['accepted_rides']          = drivers_df['accepted_rides'].astype('int32')
drivers_df['incomplete_rides']        = drivers_df['incomplete_rides'].astype('int32')
drivers_df['delay_count']             = drivers_df['delay_count'].astype('int32')
drivers_df['rejected_rides']          = drivers_df['rejected_rides'].astype('int32')
print("  ✅ drivers")

# ── Location Demand ───────────────────────────────────────
for col in ['city', 'pickup_location', 'vehicle_type', 'demand_level']:
    location_demand_df[col] = location_demand_df[col].astype('category')

location_demand_df['hour_of_day']         = location_demand_df['hour_of_day'].astype('int8')
location_demand_df['avg_wait_time_min']   = location_demand_df['avg_wait_time_min'].astype('float32')
location_demand_df['avg_surge_multiplier']= location_demand_df['avg_surge_multiplier'].astype('float32')
location_demand_df['total_requests']      = location_demand_df['total_requests'].astype('int32')
location_demand_df['completed_rides']     = location_demand_df['completed_rides'].astype('int32')
location_demand_df['cancelled_rides']     = location_demand_df['cancelled_rides'].astype('int32')
print("  ✅ location_demand")

# ── Time Features ─────────────────────────────────────────
time_features_df['datetime']       = pd.to_datetime(time_features_df['datetime'])
for col in ['day_of_week', 'season']:
    time_features_df[col] = time_features_df[col].astype('category')

time_features_df['hour_of_day']    = time_features_df['hour_of_day'].astype('int8')
time_features_df['is_weekend']     = time_features_df['is_weekend'].astype('int8')
time_features_df['is_holiday']     = time_features_df['is_holiday'].astype('int8')
time_features_df['peak_time_flag'] = time_features_df['peak_time_flag'].astype('int8')
print("  ✅ time_features")

# =============================================================================
# SECTION 5 — SANITY CHECK NULLS
# =============================================================================
print("\n🔍 Null check...")
all_clean = True
for name, df in [('bookings', bookings_df), ('customers', customers_df),
                 ('drivers', drivers_df), ('location_demand', location_demand_df),
                 ('time_features', time_features_df)]:
    nulls = df.isnull().sum().sum()
    status = "✅" if nulls == 0 else f"⚠️  {nulls} nulls"
    print(f"  {name:<20}: {status}")
    if nulls > 0:
        all_clean = False

if not all_clean:
    print("  ⚠️  Some nulls remain — they will be inserted as NULL in MySQL.")

# =============================================================================
# SECTION 6 — INSERT INTO MYSQL
# =============================================================================
print("\n📥 Inserting into MySQL...")

table_map = {
    'bookings'       : bookings_df,
    'customers'      : customers_df,
    'drivers'        : drivers_df,
    'location_demand': location_demand_df,
    'time_features'  : time_features_df,
}

for table_name, df in table_map.items():
    start = datetime.now()
    try:
        df.to_sql(
            name      = table_name,
            con       = engine,
            if_exists = 'append',
            index     = False,
            chunksize = 5000,
            method    = 'multi',
        )
        elapsed = (datetime.now() - start).seconds
        print(f"  ✅ {table_name:<20} {len(df):>10,} rows  ({elapsed}s)")
    except Exception as e:
        print(f"  ❌ {table_name:<20} failed: {e}")

# =============================================================================
# SECTION 7 — VERIFY
# =============================================================================
print("\n" + "=" * 55)
print("VERIFICATION — Row counts")
print("=" * 55)
with engine.connect() as conn:
    for t in ['bookings', 'customers', 'drivers', 'location_demand', 'time_features']:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        print(f"  {t:<25}: {count:>10,} rows")

print("\n✅ Database setup complete. Run insert_predictions.py next.")