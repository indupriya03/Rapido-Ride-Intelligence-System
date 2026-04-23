# -*- coding: utf-8 -*-
# =============================================================================
# pages/predictions.py — UC2 Fare + UC3 Cancel Risk predictions
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
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
    Q2_LIVE_INFERENCE,
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
    @st.cache_data(ttl=300)
    def load_prediction_kpis():
        return run_query(Q2_PREDICTION_KPIS).iloc[0]

    kpi = load_prediction_kpis()

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
            xaxis=dict(title="Recommended Action", gridcolor=THEME["border"]),
            yaxis=dict(title="Number of Rides", gridcolor=THEME["border"]),
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
        fig_hm   = heatmap_chart(pivot, title="", height=380, color_scale="YlOrRd_r")
        fig_hm   = apply_chart_theme(fig_hm, THEME, height=380)
        fig_hm.update_xaxes(title="Risk Tier")
        fig_hm.update_yaxes(title="")
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("<div class='section-title'>🔴 High-Risk Bookings</div>", unsafe_allow_html=True)
    hr_df = run_query(Q2_HIGH_RISK_SAMPLE)
    st.dataframe(
        hr_df.style.format({
            "cancel_prob_pct": "{:.1f}%",
            "distance_km":     "{:.1f}",
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
# TAB 2 — LIVE INFERENCE (SQL-based — no hardcoded defaults)
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Ride Input — UC2 Fare & UC3 Cancel Risk</div>", unsafe_allow_html=True)
    st.caption("Fetches the closest matching real booking from the predictions table.")

    col_inp, col_out = st.columns([1, 1.2])

    with col_inp:
        city_sel     = st.selectbox("City",          ["Bangalore", "Delhi", "Mumbai", "Chennai", "Hyderabad"])
        vehicle_sel  = st.selectbox("Vehicle Type",  ["Bike", "Auto", "Cab"])
        distance_sel = st.slider("Distance (km)",    1.0, 50.0, 10.0, 0.5)
        hour_sel     = st.slider("Hour of Day",      0, 23, 9)
        traffic_sel  = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        weather_sel  = st.selectbox("Weather",       ["Clear", "Rain", "Heavy Rain"])
        predict_btn  = st.button("⚡ Predict", use_container_width=True)

    with col_out:
        if predict_btn:
            result = run_query(
                Q2_LIVE_INFERENCE,
                params={
                    "city":             city_sel,
                    "vehicle_type":     vehicle_sel,
                    "hour_of_day":      hour_sel,
                    "traffic_level":    traffic_sel,
                    "weather_condition": weather_sel,
                    "distance_km":      distance_sel,
                }
            )

            if result.empty:
                st.warning(
                    "No matching booking found for this combination. "
                    "Try adjusting the distance, hour, or traffic level."
                )
            else:
                row = result.iloc[0]
                if int(row["matched_bookings"]) < 5:
                    st.warning(
                        f"Only {int(row['matched_bookings'])} matching bookings found — "
                        "result may not be representative. Try widening the distance or hour."
                    )
                tier   = row["cancel_risk_tier"]
                cancel_p = float(row["cancel_probability_raw"])

                st.plotly_chart(
                    risk_gauge(cancel_p, "Cancellation Risk", theme=THEME),
                    use_container_width=True
                )

                tier_badge = f"<span class='badge-{tier.lower()}'>{tier} Risk</span>"
                st.markdown(f"""
                <div style='background:{_card_bg};border:1px solid {_card_bdr};border-radius:12px;padding:20px;'>
                  <div style='font-size:13px;color:{_text_muted};margin-bottom:4px;'>PREDICTED FARE</div>
                  <div style='font-size:36px;font-weight:700;color:{_accent};'>₹{row['predicted_fare']:,.0f}</div>
                  <div style='font-size:12px;color:{_text_muted};margin-top:4px;'>
                    Surge ×{row['surge_multiplier']} · {row['vehicle_type']} · {row['ride_distance_km']} km
                  </div>
                  <hr style='border-color:{_card_bdr};margin:14px 0;'>
                  <div style='font-size:13px;color:{_text_muted};margin-bottom:6px;'>CANCEL RISK</div>
                  {tier_badge}
                  <div style='font-size:20px;font-weight:600;margin:8px 0;color:{_text_main};'>{row['cancel_probability_pct']:.1f}%</div>
                  <div style='font-size:13px;color:{_text_muted};'>
                    Threshold: {row['uc3_threshold_used']:.3f} ·
                    Recommended: <b style='color:{_text_main};'>{row['recommended_action']}</b>
                  </div>
                  <hr style='border-color:{_card_bdr};margin:14px 0;'>
                  <div style='font-size:12px;color:{_text_muted};'>
                    Based on {int(row['matched_bookings']):,} similar historical bookings
                  </div>
                </div>
                """, unsafe_allow_html=True)
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

    st.markdown("")

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
        fig_scatter = fare_vs_actual_scatter(scatter_df, theme=THEME)
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