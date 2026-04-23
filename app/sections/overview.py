# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================
# File: app/pages/1_Overview.py
#
# Shows:
#   - KPI cards (total rides, cancellation rate, avg fare, avg delay)
#   - Rides by city (bar chart)
#   - Cancellations by hour (line chart)
#   - Booking status distribution (pie chart)
#   - Surge multiplier by city (bar chart)
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.utils.db import run_query

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Overview — Rapido",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Rapido Ride Intelligence — Overview")
st.markdown("High-level snapshot of ride operations across all cities.")
st.divider()

# =============================================================================
# LOAD DATA
# =============================================================================
@st.cache_data(ttl=300)
def load_overview_data():
    bookings = run_query("""
        SELECT
            b.booking_id,
            b.city,
            b.hour_of_day,
            b.day_of_week,
            b.is_weekend,
            b.vehicle_type,
            b.ride_distance_km,
            b.booking_value,
            b.booking_status,
            b.surge_multiplier,
            b.base_fare,
            b.traffic_level,
            b.weather_condition,
            b.booking_datetime,
            d.avg_pickup_delay_min,
            d.driver_delay_flag
        FROM bookings b
        LEFT JOIN drivers d ON b.driver_id = d.driver_id
    """)
    return bookings

@st.cache_data(ttl=300)
def load_location_data():
    return run_query("""
        SELECT city, pickup_location, hour_of_day,
               vehicle_type, total_requests,
               completed_rides, cancelled_rides,
               avg_wait_time_min, avg_surge_multiplier, demand_level
        FROM location_demand
    """)

with st.spinner("Loading data..."):
    df      = load_overview_data()
    loc_df  = load_location_data()

if df.empty:
    st.error("No data loaded. Check your database connection.")
    st.stop()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
st.sidebar.header("Filters")

cities = ["All"] + sorted(df['city'].dropna().unique().tolist())
sel_city = st.sidebar.selectbox("City", cities)

vehicles = ["All"] + sorted(df['vehicle_type'].dropna().unique().tolist())
sel_vehicle = st.sidebar.selectbox("Vehicle Type", vehicles)

hour_range = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

# Apply filters
filtered = df.copy()
if sel_city != "All":
    filtered = filtered[filtered['city'] == sel_city]
if sel_vehicle != "All":
    filtered = filtered[filtered['vehicle_type'] == sel_vehicle]
filtered = filtered[
    (filtered['hour_of_day'] >= hour_range[0]) &
    (filtered['hour_of_day'] <= hour_range[1])
]

# =============================================================================
# SECTION 1 — KPI CARDS
# =============================================================================
total_rides      = len(filtered)
completed        = (filtered['booking_status'] == 'Completed').sum()
cancelled        = (filtered['booking_status'] == 'Cancelled').sum()
cancel_rate      = (cancelled / total_rides * 100) if total_rides > 0 else 0
avg_fare         = filtered['booking_value'].mean()
avg_surge        = filtered['surge_multiplier'].mean()
avg_delay        = filtered['avg_pickup_delay_min'].fillna(0).mean()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Rides", f"{total_rides:,}")
with col2:
    st.metric("Cancellation Rate", f"{cancel_rate:.1f}%",
              delta=f"{cancelled:,} rides", delta_color="inverse")
with col3:
    st.metric("Avg Fare (₹)", f"₹{avg_fare:.0f}")
with col4:
    st.metric("Avg Delay (min)", f"{avg_delay:.1f}" if pd.notna(avg_delay) else "N/A")
with col5:
    st.metric("Avg Surge", f"{avg_surge:.2f}x")

st.divider()

# =============================================================================
# SECTION 2 — RIDES BY CITY + BOOKING STATUS
# =============================================================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Cancellation rate by city")
    city_cancel = (
        filtered.groupby('city').apply(
            lambda x: pd.Series({
                'cancel_rate': (x['booking_status'] == 'Cancelled').sum() / len(x) * 100,
                'total_rides': len(x)
            })
        ).reset_index().sort_values('cancel_rate', ascending=True)
    )
    fig_city = px.bar(
        city_cancel, x='cancel_rate', y='city',
        orientation='h',
        color='cancel_rate',
        color_continuous_scale='RdYlGn_r',
        labels={'cancel_rate': 'Cancellation Rate (%)', 'city': 'City'},
        text=city_cancel['cancel_rate'].apply(lambda x: f"{x:.1f}%"),
    )
    fig_city.update_traces(textposition='outside')
    fig_city.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, r=60),
        height=320,
        xaxis=dict(range=[0, city_cancel['cancel_rate'].max() * 1.2])
    )
    st.plotly_chart(fig_city, use_container_width=True)

with col_right:
    st.subheader("Booking status distribution")
    status_counts = (
        filtered['booking_status']
        .value_counts().reset_index()
    )
    fig_pie = px.pie(
        status_counts, names='booking_status', values='count',
        color_discrete_sequence=['#2ecc71', '#e74c3c', '#f39c12'],
        hole=0.4,
    )
    fig_pie.update_layout(
        margin=dict(t=20, b=20),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 3 — CANCELLATIONS BY HOUR
# =============================================================================
st.subheader("Ride activity by hour of day")

hourly = (
    filtered.groupby('hour_of_day')['booking_status']
    .value_counts().unstack(fill_value=0)
    .reset_index()
)
# Ensure all status columns exist
for col in ['Completed', 'Cancelled', 'Incomplete']:
    if col not in hourly.columns:
        hourly[col] = 0

fig_hourly = go.Figure()
fig_hourly.add_trace(go.Scatter(
    x=hourly['hour_of_day'], y=hourly['Completed'],
    name='Completed', mode='lines+markers',
    line=dict(color='#2ecc71', width=2),
    fill='tozeroy', fillcolor='rgba(46,204,113,0.1)'
))
fig_hourly.add_trace(go.Scatter(
    x=hourly['hour_of_day'], y=hourly['Cancelled'],
    name='Cancelled', mode='lines+markers',
    line=dict(color='#e74c3c', width=2),
    fill='tozeroy', fillcolor='rgba(231,76,60,0.1)'
))
fig_hourly.add_trace(go.Scatter(
    x=hourly['hour_of_day'], y=hourly['Incomplete'],
    name='Incomplete', mode='lines+markers',
    line=dict(color='#f39c12', width=2),
))
fig_hourly.update_layout(
    xaxis_title='Hour of Day',
    yaxis_title='Number of Rides',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation='h', y=1.1),
    margin=dict(t=20, b=20),
    height=350,
    xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[0, 23]),
)
st.plotly_chart(fig_hourly, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 4 — SURGE + DEMAND HEATMAP
# =============================================================================
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Avg surge multiplier by city & vehicle")
    surge_city = (
        filtered.groupby(['city', 'vehicle_type'])['surge_multiplier']
        .mean().reset_index()
    )
    fig_surge = px.bar(
        surge_city, x='city', y='surge_multiplier',
        color='vehicle_type', barmode='group',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={'surge_multiplier': 'Avg Surge', 'city': 'City'},
    )
    fig_surge.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20),
        height=320,
        legend_title='Vehicle',
    )
    st.plotly_chart(fig_surge, use_container_width=True)

with col_r:
    st.subheader("Demand level by city & hour")
    # Use location_demand table
    loc_filtered = loc_df.copy()
    if sel_city != "All":
        loc_filtered = loc_filtered[loc_filtered['city'] == sel_city]
    if sel_vehicle != "All":
        loc_filtered = loc_filtered[loc_filtered['vehicle_type'] == sel_vehicle]

    demand_heat = (
        loc_filtered.groupby(['city', 'hour_of_day'])['total_requests']
        .sum().reset_index()
    )
    fig_demand = px.density_heatmap(
        demand_heat, x='hour_of_day', y='city',
        z='total_requests', color_continuous_scale='YlOrRd',
        labels={'hour_of_day': 'Hour', 'city': 'City',
                'total_requests': 'Requests'},
    )
    fig_demand.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig_demand, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 5 — VEHICLE TYPE BREAKDOWN
# =============================================================================
st.subheader("Avg fare by vehicle type")
fare_vehicle = (
    filtered.groupby('vehicle_type')['booking_value']
    .agg(['mean', 'median', 'count']).reset_index()
    .rename(columns={'mean': 'Avg Fare', 'median': 'Median Fare', 'count': 'Rides'})
)
fare_vehicle['Avg Fare']    = fare_vehicle['Avg Fare'].round(2)
fare_vehicle['Median Fare'] = fare_vehicle['Median Fare'].round(2)
st.dataframe(fare_vehicle, use_container_width=True, hide_index=True)

st.caption("Data refreshes every 5 minutes. Filters apply to all charts.")