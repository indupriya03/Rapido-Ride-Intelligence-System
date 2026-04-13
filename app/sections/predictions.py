# -*- coding: utf-8 -*-
# =============================================================================
# pages/predictions.py — UC2 Fare + UC3 Cancel Risk predictions
# =============================================================================

from sklearn import base
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.db     import run_query
from utils.charts import (
    RAPIDO_YELLOW, RAPIDO_SURFACE, RAPIDO_BORDER, RAPIDO_TEXT, RAPIDO_MUTED,
    RISK_COLORS, STATUS_COLORS, apply_chart_theme,
    bar_chart, pie_chart, heatmap_chart, risk_gauge, fare_vs_actual_scatter,
)
from utils.theme  import LIGHT_THEME, DARK_THEME
from utils.queries import (
    Q0_CITIES, Q0_VEHICLE_TYPES,
    Q2_PREDICTION_KPIS, Q2_RISK_TIER_DIST, Q2_ACTIONS_DIST,
    Q2_FARE_ACCURACY_BY_CITY, Q2_RISK_BY_CITY, Q2_CANCEL_PROBA_DIST,
    Q2_HIGH_RISK_SAMPLE, Q2_MODEL_ACCURACY, Q2_FARE_SCATTER,
)

# Inherit theme from session state (set in app.py)
_is_dark = st.session_state.get("theme", "light") == "dark"
THEME    = DARK_THEME if _is_dark else LIGHT_THEME

_card_bg    = THEME["card"]
_card_bdr   = THEME["border"]
_text_main  = THEME["text"]
_text_muted = "#7A7A7A" if _is_dark else "#888888"
_accent     = THEME["accent"]

def _themed(fig, height=340):
    return apply_chart_theme(fig, THEME, height=height)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='font-size:28px;font-weight:700;margin-bottom:4px;color:{_text_main};'>Predictions</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{_text_muted};'>UC2 Fare Forecasting · UC3 Cancellation Risk · Live Ride Inference</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📦 Batch Results", "🎯 Live Inference", "📈 Model Performance"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — BATCH RESULTS
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    kpi = run_query(Q2_PREDICTION_KPIS).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions",   f"{int(kpi.total_predictions):,}")
    c2.metric("Avg Predicted Fare",  f"₹{kpi.avg_predicted_fare:,.2f}")
    c3.metric("Fare MAE",            f"₹{kpi.avg_fare_mae:,.2f}")
    c4.metric("High-Risk Rides",     f"{int(kpi.high_risk_count):,}  ({kpi.high_risk_pct:.1f}%)")

    st.markdown("")

    col_a, col_b = st.columns([1, 1.4])

    with col_a:
        st.markdown("<div class='section-title'>Cancel Risk Distribution</div>", unsafe_allow_html=True)
        risk_df = run_query(Q2_RISK_TIER_DIST)
        fig_risk = go.Figure(go.Pie(
            labels=risk_df["cancel_risk_tier"],
            values=risk_df["count"],
            hole=0.52,
            marker_colors=[RISK_COLORS.get(t, "#888") for t in risk_df["cancel_risk_tier"]],
            textinfo="percent+label",
        ))
        fig_risk = _themed(fig_risk, 300)
        fig_risk.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>Recommended Actions</div>", unsafe_allow_html=True)
        act_df = run_query(Q2_ACTIONS_DIST)
        fig_act = go.Figure()
        for tier, color in RISK_COLORS.items():
            sub = act_df[act_df["cancel_risk_tier"] == tier]
            if not sub.empty:
                fig_act.add_trace(go.Bar(
                    x=sub["recommended_action"], y=sub["count"],
                    name=tier, marker_color=color,
                ))
        fig_act  = _themed(fig_act, 300)
        fig_act.update_layout(
            barmode="stack",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=THEME["border"]),
            yaxis=dict(gridcolor=THEME["border"]),
        )
        st.plotly_chart(fig_act, use_container_width=True)

    st.markdown("<div class='section-title'>Cancel Probability Distribution</div>", unsafe_allow_html=True)
    prob_df = run_query(Q2_CANCEL_PROBA_DIST)
    fig_prob = go.Figure(go.Bar(
        x=prob_df["prob_bucket"].astype(float),
        y=prob_df["count"],
        marker_color=[
            RISK_COLORS["High"]   if p >= 0.7 else
            RISK_COLORS["Medium"] if p >= 0.4 else
            RISK_COLORS["Low"]
            for p in prob_df["prob_bucket"].astype(float)
        ],
        width=0.09,
    ))
    fig_prob.add_vline(x=0.4, line_dash="dash", line_color=RAPIDO_MUTED,
                       annotation_text="Medium threshold (0.4)")
    fig_prob.add_vline(x=0.7, line_dash="dash", line_color="#FF4B4B",
                       annotation_text="High threshold (0.7)")
    fig_prob = _themed(fig_prob, 300)
    fig_prob.update_layout(
        xaxis=dict(title="Cancel Probability", gridcolor=THEME["border"], tickformat=".0%"),
        yaxis=dict(title="Count", gridcolor=THEME["border"]),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("<div class='section-title'>Cancel Risk by City</div>", unsafe_allow_html=True)
    risk_city_df = run_query(Q2_RISK_BY_CITY)
    if not risk_city_df.empty:
        pivot = risk_city_df.pivot_table(
            index="city", columns="cancel_risk_tier", values="count", fill_value=0
        )
        pivot = pivot.reindex(columns=["High", "Medium", "Low"], fill_value=0)
        fig_hm   = heatmap_chart(pivot, title="", height=380, color_scale="YlOrRd")
        fig_hm   = apply_chart_theme(fig_hm, THEME, height=380)
        fig_hm.update_xaxes(title="Risk Tier")
        fig_hm.update_yaxes(title="")
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("<div class='section-title'>🔴 High-Risk Bookings</div>", unsafe_allow_html=True)
    hr_df = run_query(Q2_HIGH_RISK_SAMPLE)
    st.dataframe(
        hr_df.style.format({
            "cancel_prob_pct": "{:.1f}%",
            "surge": "{:.2f}x",
            "predicted_fare": "₹{:.0f}",
            "actual_fare": "₹{:.0f}",
        }).applymap(
            lambda v: "color: #FF4B4B; font-weight:600" if v == "Reassign Driver" else "",
            subset=["recommended_action"] if "recommended_action" in hr_df.columns else [],
        ),
        use_container_width=True, hide_index=True, height=320,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Ride Input — UC2 Fare & UC3 Cancel Risk</div>", unsafe_allow_html=True)

    MODEL_DIR = "models"
    models_exist = (
        os.path.exists(f"{MODEL_DIR}/uc2_fare_model.pkl") and
        os.path.exists(f"{MODEL_DIR}/uc3_cancel_model.pkl") and
        os.path.exists(f"{MODEL_DIR}/scaler_uc2.pkl") and
        os.path.exists(f"{MODEL_DIR}/scaler_uc3.pkl")
    )

    # ── Auto-compute surge based on ride conditions ──
    def compute_surge(hour, is_weekend, peak, traffic, weather):
        base = 1.0
        if 7 <= hour <= 10 or 17 <= hour <= 21:
            base += 0.4
        if hour >= 22 or hour <= 5:
            base += 0.2
        if is_weekend:
            base += 0.2
        if peak:
            base += 0.3
        if traffic == "High":
            base += 0.4
        elif traffic == "Medium":
            base += 0.2
        if weather == "Heavy Rain":
            base += 0.5
        elif weather == "Rain":
            base += 0.3
        return round(min(base, 3.0), 1)
    
    col_inp, col_out = st.columns([1, 1.2])

    with col_inp:
        city_sel     = st.selectbox("City",         ["Bangalore", "Delhi", "Mumbai", "Chennai", "Hyderabad"])
        vehicle_sel  = st.selectbox("Vehicle Type", ["Bike", "Auto", "Cab"])
        distance_sel = st.slider("Distance (km)",   1.0, 50.0, 10.0, 0.5)
        hour_sel     = st.slider("Hour of Day",     0,   23,   9)
        weekend_sel  = st.checkbox("Is Weekend?")
        peak_sel     = st.checkbox("Peak Time?")
        traffic_sel  = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        weather_sel  = st.selectbox("Weather",       ["Clear", "Rain", "Heavy Rain"])
        surge_sel = compute_surge(hour_sel, weekend_sel, peak_sel, traffic_sel, weather_sel)
        st.info(f"⚡ Auto-computed Surge: **{surge_sel}x**")
        predict_btn  = st.button("⚡ Predict", use_container_width=True)



    with col_out:
        if predict_btn:
            if models_exist:
                try:
                    # ── Load models + scalers + threshold ──
                    uc2_model  = joblib.load(f"{MODEL_DIR}/uc2_fare_model.pkl")
                    uc3_model  = joblib.load(f"{MODEL_DIR}/uc3_cancel_model.pkl")
                    scaler_uc2 = joblib.load(f"{MODEL_DIR}/scaler_uc2.pkl")
                    scaler_uc3 = joblib.load(f"{MODEL_DIR}/scaler_uc3.pkl")
                    with open(f"{MODEL_DIR}/thresholds.json") as f:
                        thresh = json.load(f).get("uc3_threshold", 0.5)

                    # ── Derived inputs ──
                    traffic_map  = {"Low": 1, "Medium": 2, "High": 3}
                    weather_map  = {"Clear": 0, "Rain": 1, "Heavy Rain": 2}
                    traffic_num  = traffic_map[traffic_sel]
                    weather_num  = weather_map[weather_sel]
                    est_ride_time = distance_sel * 3.0   # ~3 min/km heuristic

                    # ── Build raw feature dict ──
                    raw = {
                        # Core inputs
                        "ride_distance_km":             distance_sel,
                        "surge_multiplier":             surge_sel,
                        "estimated_ride_time_min":      est_ride_time,
                        "hour_of_day":                  hour_sel,
                        "is_weekend":                   int(weekend_sel),
                        "peak_time_flag":               int(peak_sel),
                        "is_night_ride":                int(hour_sel >= 22 or hour_sel <= 5),
                        "is_morning_peak":              int(7 <= hour_sel <= 10),
                        "is_evening_peak":              int(17 <= hour_sel <= 21),
                        # Encoded fields
                        "traffic_enc":                  traffic_num - 1,   # 0/1/2
                        "weather_enc":                  weather_num,
                        "demand_enc":                   1,                 # assume Medium
                        "traffic_num":                  traffic_num,
                        "weather_num":                  weather_num,
                        # Interaction features (raw — not in SCALE_COLS)
                        "surge_x_distance":             surge_sel * distance_sel,
                        "surge_x_traffic":              surge_sel * traffic_num,
                        "peak_x_surge":                 int(peak_sel) * surge_sel,
                        "traffic_x_ridetime":           traffic_num * est_ride_time,
                        "weather_x_ridetime":           weather_num * est_ride_time,
                        "rain_high_traffic":            int(weather_sel in ["Rain","Heavy Rain"] and traffic_sel == "High"),
                        "high_delay_high_traffic":      int(traffic_sel == "High"),
                        "holiday_x_peak":               0,
                        "night_x_high_traffic":         int((hour_sel >= 22 or hour_sel <= 5) and traffic_sel == "High"),
                        "peak_x_distance":              int(peak_sel) * distance_sel,
                        # Distance flags (use fixed thresholds — same as Zone 3 approx)
                        "is_short_ride":                int(distance_sel < 5),
                        "is_long_ride":                 int(distance_sel > 20),
                        "distance_bin_enc":             2 if distance_sel > 20 else (0 if distance_sel < 5 else 1),
                        # Vehicle type one-hot
                        "vehicle_type_Cab":             int(vehicle_sel == "Cab"),
                        "vehicle_type_Bike":            int(vehicle_sel == "Bike"),
                        # City one-hot
                        "city_Delhi":                   int(city_sel == "Delhi"),
                        "city_Mumbai":                  int(city_sel == "Mumbai"),
                        "city_Chennai":                 int(city_sel == "Chennai"),
                        "city_Hyderabad":               int(city_sel == "Hyderabad"),
                        # Driver defaults (median values — better than 0)
                        "acceptance_rate":              0.85,
                        "delay_rate":                   0.05,
                        "avg_driver_rating":            4.2,
                        "driver_reliability_score":     0.80,
                        "avg_pickup_delay_min":         3.0,
                        "rejection_rate":               0.10,
                        "driver_incomplete_rate":       0.05,
                        "delay_per_ride":               0.05,
                        "delay_per_km":                 0.01,
                        "delay_count":                  2,
                        "total_assigned_rides":         150,
                        "accepted_rides":               128,
                        "driver_incomplete_rides":      6,
                        "driver_experience_years":      4.0,
                        "driver_age":                   32,
                        "avg_driver_rating":            4.2,
                        "driver_experience_enc":        1,
                        "driver_acceptance_enc":        2,
                        "is_unreliable_driver":         0,
                        "is_low_rated_driver":          0,
                        "experience_outlier_flag":      0,
                        # Customer defaults
                        "customer_age":                 30,
                        "customer_signup_days_ago":     365,
                        "customer_tenure_years":        1.0,
                        "is_new_customer":              0,
                        "avg_customer_rating":          4.0,
                        "total_bookings":               15,
                        "completed_rides":              12,
                        "cancelled_rides":              2,
                        "incomplete_rides":             1,
                        "cancellation_rate":            0.13,
                        "cancel_to_booking_ratio":      0.13,
                        "cust_completion_rate":         0.80,
                        "incomplete_ride_share":        0.07,
                        "is_low_rated_customer":        0,
                        "is_high_cancel_customer":      0,
                        "vehicle_preference_match":     1,
                        "cancel_risk_score":            0.10,
                        "Customer_Loyalty_Score":       0.75,
                        "loyalty_x_cancel":             0.10,
                        # Location defaults
                        "loc_total_requests":           500,
                        "loc_completed_rides":          380,
                        "loc_cancelled_rides":          80,
                        "avg_wait_time_min":            5.0,
                        "avg_surge_multiplier":         surge_sel,
                        "location_surge_deviation":     0.0,
                        "is_high_demand_location":      0,
                        "demand_supply_ratio":          1.0,
                        # Derived ride features
                        "fare_per_km":                  surge_sel * 12,
                        "fare_per_min":                 surge_sel * 4,
                        "base_fare":                    distance_sel * 12,
                        "base_fare_per_km":             12.0,
                        "expected_fare":                distance_sel * 12 * surge_sel,
                        "surge_impact":                 distance_sel * 12 * (surge_sel - 1),
                        "base_x_surge":                 distance_sel * 12 * surge_sel,
                        "fare_per_km_per_surge":        12.0,
                        "fare_above_loc_avg":           int(surge_sel > 1.5),
                        "reliability_x_distance":       0.80 * distance_sel,
                        "delay_x_traffic":              0.05 * traffic_num,
                        "cancel_risk_x_peak":           0.13 * int(peak_sel),
                        "cancel_risk_x_night":          0.13 * int(hour_sel >= 22 or hour_sel <= 5),
                        "demand_x_incomplete_rate":     1 * 0.05,
                        "time_pressure":                est_ride_time / 6.0,
                        "exp_x_distance":               4.0 * distance_sel,
                        "ride_difficulty":              0,
                        "distance_x_traffic":           distance_sel * (traffic_num - 1),
                        "dual_low_rating":              0,
                        "new_cust_x_unreliable":        0,
                        # Cyclical time
                        "hour_sin":                     np.sin(2 * np.pi * hour_sel / 24),
                        "hour_cos":                     np.cos(2 * np.pi * hour_sel / 24),
                        "is_holiday":                   0,
                        "peak_x_distance":              int(peak_sel) * distance_sel,
                    }

                    # ── Align to model feature columns ──
                    uc2_cols = list(uc2_model.feature_names_in_)
                    uc3_cols = list(uc3_model.feature_names_in_)

                    def build_X(cols, raw_dict):
                        row = {c: raw_dict.get(c, 0) for c in cols}
                        return pd.DataFrame([row])[cols].astype(float)

                    X_uc2_raw = build_X(uc2_cols, raw)
                    X_uc3_raw = build_X(uc3_cols, raw)

                    # ── Apply scaler to columns it was fit on ──
                    SCALE_COLS = [
                        'surge_multiplier', 'estimated_ride_time_min',
                        'acceptance_rate', 'delay_rate', 'avg_pickup_delay_min',
                        'customer_age', 'customer_tenure_years', 'customer_signup_days_ago',
                        'fare_per_km', 'fare_per_min',
                        'driver_reliability_score', 'demand_supply_ratio',
                        'cancel_to_booking_ratio', 'cust_completion_rate',
                        'rejection_rate', 'delay_per_ride', 'driver_incomplete_rate',
                        'location_surge_deviation', 'surge_x_distance',
                        'traffic_x_ridetime', 'reliability_x_distance',
                        'ride_distance_km', 'base_fare',
                    ]

                    def apply_scaler(X, scaler, scale_cols):
                        cols_present = [c for c in scale_cols if c in X.columns]
                        if cols_present:
                            X = X.copy()
                            X[cols_present] = scaler.transform(X[cols_present])
                        return X

                    X_uc2 = apply_scaler(X_uc2_raw, scaler_uc2, SCALE_COLS)
                    X_uc3 = apply_scaler(X_uc3_raw, scaler_uc3, SCALE_COLS)

                    # ── Predict ──
                    fare_log  = uc2_model.predict(X_uc2)[0]
                    fare_pred = float(np.expm1(fare_log))
                    cancel_p  = float(uc3_model.predict_proba(X_uc3)[0, 1])

                    tier   = "High" if cancel_p >= 0.7 else "Medium" if cancel_p >= 0.4 else "Low"
                    action = {"High": "Reassign Driver", "Medium": "Send Reminder", "Low": "Proceed"}[tier]

                    st.plotly_chart(risk_gauge(cancel_p, "Cancellation Risk"), use_container_width=True)

                    tier_badge = f"<span class='badge-{tier.lower()}'>{tier} Risk</span>"
                    st.markdown(f"""
                    <div style='background:{_card_bg};border:1px solid {_card_bdr};border-radius:12px;padding:20px;'>
                      <div style='font-size:13px;color:{_text_muted};margin-bottom:4px;'>PREDICTED FARE</div>
                      <div style='font-size:36px;font-weight:700;color:{_accent};'>₹{fare_pred:,.0f}</div>
                      <div style='font-size:12px;color:{_text_muted};margin-top:4px;'>
                        Base est: ₹{(distance_sel * 12):,.0f} · Surge ×{surge_sel}
                      </div>
                      <hr style='border-color:{_card_bdr};margin:14px 0;'>
                      <div style='font-size:13px;color:{_text_muted};margin-bottom:6px;'>CANCEL RISK</div>
                      {tier_badge}
                      <div style='font-size:20px;font-weight:600;margin:8px 0;color:{_text_main};'>{cancel_p*100:.1f}%</div>
                      <div style='font-size:13px;color:{_text_muted};'>Threshold: {thresh:.3f} · Recommended: <b style='color:{_text_main};'>{action}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.exception(e)

            else:
                st.warning("Model files not found. Ensure `insert_predictions.py` has been run first.", icon="⚠️")

        else:
            st.markdown(f"""
            <div style='background:{_card_bg};border:1px dashed {_card_bdr};border-radius:12px;
                        padding:40px;text-align:center;color:{_text_muted};'>
              <div style='font-size:48px;margin-bottom:12px;'>⚡</div>
              <div>Fill in ride details and click <b>Predict</b></div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>UC2 — Fare Model Accuracy</div>", unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fare_acc_df = run_query(Q2_FARE_ACCURACY_BY_CITY)
        fig_fare_acc = go.Figure()
        fig_fare_acc.add_trace(go.Bar(
            x=fare_acc_df["city"], y=fare_acc_df["avg_actual"],
            name="Actual", marker_color="#4B9EFF",
        ))
        fig_fare_acc.add_trace(go.Bar(
            x=fare_acc_df["city"], y=fare_acc_df["avg_predicted"],
            name="Predicted", marker_color=RAPIDO_YELLOW,
        ))
        fig_fare_acc = _themed(fig_fare_acc, 350)
        fig_fare_acc.update_layout(
            barmode="group",
            xaxis=dict(gridcolor=THEME["border"]),
            yaxis=dict(title="₹ Fare", gridcolor=THEME["border"]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_fare_acc, use_container_width=True)

    with col_p2:
        fig_mape = go.Figure(go.Bar(
            x=fare_acc_df["city"],
            y=fare_acc_df["mape_pct"],
            marker_color=[
                "#FF4B4B" if m > 15 else "#FFB400" if m > 8 else "#00C48C"
                for m in fare_acc_df["mape_pct"]
            ],
            text=fare_acc_df["mape_pct"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
        fig_mape     = _themed(fig_mape, 350)
        fig_mape.update_layout(
            title="Mean Absolute Percentage Error by City",
            xaxis=dict(gridcolor=THEME["border"]),
            yaxis=dict(title="MAPE (%)", gridcolor=THEME["border"]),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_mape, use_container_width=True)

    st.markdown("<div class='section-title'>Predicted vs Actual Fare Scatter (UC2)</div>", unsafe_allow_html=True)
    scatter_df = run_query(Q2_FARE_SCATTER)
    if not scatter_df.empty:
        fig_scatter = fare_vs_actual_scatter(scatter_df)
        fig_scatter  = apply_chart_theme(fig_scatter, THEME, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<div class='section-title'>UC3 — Cancellation Model Confusion</div>", unsafe_allow_html=True)
    conf_df = run_query(Q2_MODEL_ACCURACY)
    if not conf_df.empty:
        pivot_conf = conf_df.pivot_table(
            index="actual_cancelled_flag",
            columns="predicted_cancelled",
            values="count", fill_value=0,
        )
        pivot_conf.index   = ["Not Cancelled" if i == 0 else "Cancelled" for i in pivot_conf.index]
        pivot_conf.columns = ["Predicted: No" if c == 0 else "Predicted: Yes" for c in pivot_conf.columns]
        fig_conf = heatmap_chart(pivot_conf, title="", height=280, color_scale="YlOrRd")
        fig_conf     = apply_chart_theme(fig_conf, THEME, height=280)
        st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown("<div class='section-title'>Model Summary</div>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("UC1 — Ride Outcome",  "Accuracy ~80%",  "XGBoost + SMOTE")
    mc2.metric("UC2 — Fare (R²)",     "~0.95+",         "XGBoost Regressor")
    mc3.metric("UC3 — Cancel (AUC)",  "~0.88+",         "LightGBM / XGBoost")
    mc4.metric("UC4 — Delay (AUC)",   "~0.86+",         "LightGBM / XGBoost")