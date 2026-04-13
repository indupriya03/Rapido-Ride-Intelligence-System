# -*- coding: utf-8 -*-
# =============================================================================
# pages/analytics.py — Deep analytics: cancellation windows, surge, features
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.db     import run_query
from utils.charts import (
    RAPIDO_YELLOW, RAPIDO_SURFACE, RAPIDO_BORDER, RAPIDO_TEXT, RAPIDO_MUTED,
    RISK_COLORS, apply_chart_theme, bar_chart, line_chart, heatmap_chart, pie_chart,
)
from utils.theme  import LIGHT_THEME, DARK_THEME
from utils.queries import (
    Q3_CANCEL_HEATMAP, Q3_SURGE_BY_HOUR_CITY,
    Q3_CANCEL_PROB_BY_SURGE, Q3_CANCEL_PROB_BY_HOUR,
    Q3_CANCEL_REASONS, Q3_CANCEL_BY_PARTY,
    Q3_DIST_VS_CANCEL, Q3_WEEKEND_COMPARISON,
    Q3_WEATHER_IMPACT, Q3_TRAFFIC_IMPACT,
)

# Inherit theme from session state (set in app.py)
_is_dark = st.session_state.get("theme", "light") == "dark"
THEME    = DARK_THEME if _is_dark else LIGHT_THEME

# Shorthand for inline HTML colours that adapt to theme
_card_bg   = THEME["card"]
_card_bdr  = THEME["border"]
_text_main = THEME["text"]
_text_muted = "#7A7A7A" if _is_dark else "#888888"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='font-size:28px;font-weight:700;margin-bottom:4px;color:{_text_main};'>Analytics</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{_text_muted};'>Cancellation windows · Surge patterns · Feature drivers · Operational insights</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🕐 Cancellations", "⚡ Surge", "🔍 Feature Drivers", "🌦 External Factors"])

# ═══════════════════════════════════════════════════════════════════════════
# HELPER — apply theme to a manually built go.Figure
# ═══════════════════════════════════════════════════════════════════════════
def _themed(fig, height=340):
    return apply_chart_theme(fig, THEME, height=height)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CANCELLATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("<div class='section-title'>Cancellation Rate — Hour × Day of Week</div>", unsafe_allow_html=True)
    cancel_hm_df = run_query(Q3_CANCEL_HEATMAP)

    DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if not cancel_hm_df.empty:
        pivot = cancel_hm_df.pivot_table(
            index="hour_of_day",
            columns="day_of_week",
            values="cancel_rate_pct",
            fill_value=0,
        )
        cols_ordered = [d for d in DAY_ORDER if d in pivot.columns]
        if cols_ordered:
            pivot = pivot[cols_ordered]

        fig_hm = heatmap_chart(pivot, title="", height=480, color_scale="RdYlGn_r")
        apply_chart_theme(fig_hm, THEME, height=480)
        fig_hm.update_xaxes(title="Day of Week")
        fig_hm.update_yaxes(title="Hour of Day", autorange="reversed")
        fig_hm.update_traces(
            hovertemplate="Hour: %{y}<br>Day: %{x}<br>Cancel Rate: %{z:.1f}%<extra></extra>"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("<div class='section-title'>Model-Predicted Cancel Risk by Hour</div>", unsafe_allow_html=True)
    cancel_hr_df = run_query(Q3_CANCEL_PROB_BY_HOUR)
    fig_chr = go.Figure()
    fig_chr.add_trace(go.Bar(
        x=cancel_hr_df["hour_of_day"],
        y=cancel_hr_df["actual_cancels"],
        name="Actual Cancels",
        marker_color="#FF4B4B", opacity=0.7,
    ))
    fig_chr.add_trace(go.Scatter(
        x=cancel_hr_df["hour_of_day"],
        y=cancel_hr_df["avg_cancel_prob"] * cancel_hr_df["total"],
        name="Model-Predicted Cancels",
        mode="lines+markers",
        line=dict(color=RAPIDO_YELLOW, width=2.5),
        yaxis="y",
    ))
    _themed(fig_chr, 340)
    fig_chr.update_layout(
        xaxis=dict(title="Hour of Day", gridcolor=THEME["border"], tickmode="linear"),
        yaxis=dict(title="Cancellation Count", gridcolor=THEME["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_chr, use_container_width=True)

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("<div class='section-title'>Cancellation Reasons</div>", unsafe_allow_html=True)
        reasons_df = run_query(Q3_CANCEL_REASONS)
        if not reasons_df.empty:
            top_reasons = reasons_df.groupby("reason")["count"].sum().nlargest(10).reset_index()
            fig_reasons = go.Figure(go.Bar(
                x=top_reasons["count"],
                y=top_reasons["reason"],
                orientation="h",
                marker_color=RAPIDO_YELLOW,
            ))
            _themed(fig_reasons, 360)
            fig_reasons.update_layout(
                xaxis=dict(gridcolor=THEME["border"]),
                yaxis=dict(gridcolor=THEME["border"]),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_reasons, use_container_width=True)

    with col_r2:
        st.markdown("<div class='section-title'>Cancelled by Party</div>", unsafe_allow_html=True)
        party_df = run_query(Q3_CANCEL_BY_PARTY)
        if not party_df.empty:
            fig_party = go.Figure(go.Pie(
                labels=party_df["cancelled_by"],
                values=party_df["count"],
                hole=0.5,
                marker_colors=[RAPIDO_YELLOW, "#FF4B4B", "#4B9EFF", "#00C48C"],
                textinfo="percent+label",
            ))
            _themed(fig_party, 360)
            fig_party.update_layout(
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_party, use_container_width=True)

    st.markdown("<div class='section-title'>Weekend vs Weekday</div>", unsafe_allow_html=True)
    wknd_df = run_query(Q3_WEEKEND_COMPARISON)
    if not wknd_df.empty:
        wknd_df["label"] = wknd_df["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        wc1, wc2, wc3 = st.columns(3)
        for row in wknd_df.itertuples():
            lbl = row.label
            col = wc1 if lbl == "Weekday" else wc2
            col.metric(f"{lbl} — Total Rides",  f"{int(row.total_rides):,}")
            col.metric(f"{lbl} — Avg Fare",     f"₹{row.avg_fare:,.2f}")
            col.metric(f"{lbl} — Cancel Rate",  f"{row.cancel_rate_pct:.1f}%")
            col.metric(f"{lbl} — Avg Surge",    f"{row.avg_surge:.2f}x")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — SURGE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Surge Multiplier by Hour & City</div>", unsafe_allow_html=True)
    surge_df = run_query(Q3_SURGE_BY_HOUR_CITY)

    if not surge_df.empty:
        pivot_surge = surge_df.pivot_table(
            index="hour_of_day", columns="city",
            values="avg_surge", fill_value=1.0,
        )
        fig_surge_hm = heatmap_chart(pivot_surge, title="", height=460, color_scale="YlOrRd")
        apply_chart_theme(fig_surge_hm, THEME, height=460)
        fig_surge_hm.update_xaxes(title="City")
        fig_surge_hm.update_yaxes(title="Hour of Day", autorange="reversed")
        fig_surge_hm.update_traces(
            hovertemplate="Hour: %{y}<br>City: %{x}<br>Avg Surge: %{z:.2f}x<extra></extra>"
        )
        st.plotly_chart(fig_surge_hm, use_container_width=True)

        st.markdown("<div class='section-title'>Hourly Surge Trend by City</div>", unsafe_allow_html=True)
        cities_avail = surge_df["city"].unique()
        fig_surge_line = go.Figure()
        palette = [RAPIDO_YELLOW, "#4B9EFF", "#FF4B4B", "#00C48C", "#9B59B6", "#E67E22"]
        for i, city in enumerate(cities_avail):
            sub = surge_df[surge_df["city"] == city]
            fig_surge_line.add_trace(go.Scatter(
                x=sub["hour_of_day"], y=sub["avg_surge"],
                name=city, mode="lines+markers",
                line=dict(color=palette[i % len(palette)], width=2),
            ))
        _themed(fig_surge_line, 380)
        fig_surge_line.update_layout(
            xaxis=dict(title="Hour of Day", gridcolor=THEME["border"], tickmode="linear"),
            yaxis=dict(title="Avg Surge Multiplier", gridcolor=THEME["border"]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_surge_line, use_container_width=True)

    st.markdown("<div class='section-title'>Surge vs Cancel Probability</div>", unsafe_allow_html=True)
    surge_cancel_df = run_query(Q3_CANCEL_PROB_BY_SURGE)
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=surge_cancel_df["surge_bucket"].astype(float),
        y=surge_cancel_df["avg_cancel_prob"].astype(float) * 100,
        mode="lines+markers",
        line=dict(color=RAPIDO_YELLOW, width=2.5),
        marker=dict(size=8, color=RAPIDO_YELLOW),
        name="Avg Cancel Prob",
    ))
    _themed(fig_sc, 320)
    fig_sc.update_layout(
        xaxis=dict(title="Surge Multiplier", gridcolor=THEME["border"]),
        yaxis=dict(title="Avg Cancel Probability (%)", gridcolor=THEME["border"]),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE DRIVERS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Feature Importance (UC3 — Cancel Risk)</div>", unsafe_allow_html=True)
    st.info("📂 Feature importance charts are generated by `model_training.py` and saved to `outputs/`. Load them below or view from the outputs directory.")

    fi_paths = {
        "UC1 — Ride Outcome":     "outputs/uc1_feature_importance.png",
        "UC2 — Fare Prediction":  "outputs/uc2_feature_importance.png",
        "UC3 — Cancel Risk":      "outputs/uc3_feature_importance.png",
        "UC4 — Driver Delay":     "outputs/uc4_feature_importance.png",
    }
    fi_cols = st.columns(2)
    for i, (label, path) in enumerate(fi_paths.items()):
        with fi_cols[i % 2]:
            st.markdown(f"**{label}**")
            if os.path.exists(path):
                st.image(path)
            else:
                st.markdown(f"""
                <div style='background:{_card_bg};border:1px dashed {_card_bdr};border-radius:10px;
                            padding:24px;text-align:center;color:{_text_muted};font-size:13px;'>
                  📊 {path}<br><span style='font-size:11px;'>Run model_training.py to generate</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ride Distance vs Cancel Probability</div>", unsafe_allow_html=True)
    dist_df = run_query(Q3_DIST_VS_CANCEL)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=dist_df["dist_bucket_km"].astype(float),
        y=dist_df["avg_cancel_prob"].astype(float) * 100,
        marker_color=[
            RISK_COLORS["High"]   if p*100 >= 30 else
            RISK_COLORS["Medium"] if p*100 >= 20 else
            RISK_COLORS["Low"]
            for p in dist_df["avg_cancel_prob"].astype(float)
        ],
        width=4.5,
    ))
    _themed(fig_dist, 320)
    fig_dist.update_layout(
        xaxis=dict(title="Distance (km bucket)", gridcolor=THEME["border"]),
        yaxis=dict(title="Avg Cancel Probability (%)", gridcolor=THEME["border"]),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("<div class='section-title'>Confusion Matrices</div>", unsafe_allow_html=True)
    cm_paths = {
        "UC1 Confusion": "outputs/uc1_confusion_matrix.png",
        "UC3 Confusion": "outputs/uc3_confusion_matrix.png",
    }
    cm_cols = st.columns(2)
    for i, (label, path) in enumerate(cm_paths.items()):
        with cm_cols[i]:
            st.markdown(f"**{label}**")
            if os.path.exists(path):
                st.image(path)
            else:
                st.markdown(f"""
                <div style='background:{_card_bg};border:1px dashed {_card_bdr};border-radius:10px;
                            padding:24px;text-align:center;color:{_text_muted};font-size:13px;'>
                  📊 {path}
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — EXTERNAL FACTORS
# ═══════════════════════════════════════════════════════════════════════════
with tab4:

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown("<div class='section-title'>Weather Impact on Operations</div>", unsafe_allow_html=True)
        weather_df = run_query(Q3_WEATHER_IMPACT)
        if not weather_df.empty:
            fig_wth = go.Figure()
            fig_wth.add_trace(go.Bar(
                x=weather_df["weather_condition"],
                y=weather_df["total_rides"],
                name="Total Rides",
                marker_color="#4B9EFF",
            ))
            fig_wth.add_trace(go.Scatter(
                x=weather_df["weather_condition"],
                y=weather_df["cancel_rate_pct"],
                name="Cancel Rate %",
                mode="lines+markers",
                line=dict(color="#FF4B4B", width=2),
                yaxis="y2",
            ))
            _themed(fig_wth, 380)
            fig_wth.update_layout(
                yaxis=dict(gridcolor=THEME["border"]),
                yaxis2=dict(overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor=THEME["border"], tickangle=-30),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=20, b=60),
            )
            st.plotly_chart(fig_wth, use_container_width=True)

    with col_w2:
        st.markdown("<div class='section-title'>Traffic Level Impact</div>", unsafe_allow_html=True)
        traffic_df = run_query(Q3_TRAFFIC_IMPACT)
        if not traffic_df.empty:
            fig_trf = go.Figure()
            fig_trf.add_trace(go.Bar(
                x=traffic_df["traffic_level"],
                y=traffic_df["avg_delay_min"],
                name="Avg Delay (min)",
                marker_color=RAPIDO_YELLOW,
            ))
            fig_trf.add_trace(go.Scatter(
                x=traffic_df["traffic_level"],
                y=traffic_df["cancel_rate_pct"],
                name="Cancel Rate %",
                mode="lines+markers",
                line=dict(color="#FF4B4B", width=2),
                yaxis="y2",
            ))
            _themed(fig_trf, 380)
            fig_trf.update_layout(
                yaxis=dict(title="Avg Delay (min)", gridcolor=THEME["border"]),
                yaxis2=dict(title="Cancel Rate %", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor=THEME["border"]),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_trf, use_container_width=True)