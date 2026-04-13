# =============================================================================
# app.py — Rapido Ride Intelligence System · Streamlit Dashboard
# =============================================================================
# Run:  streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from utils.db     import run_query
from utils.charts import (
    RAPIDO_YELLOW, RAPIDO_BLACK, RAPIDO_DARK, RAPIDO_SURFACE,
    RAPIDO_TEXT, RAPIDO_MUTED, RAPIDO_BORDER,
    STATUS_COLORS,
    apply_chart_theme,          # theme-aware helper for charts
    bar_chart, line_chart, pie_chart, heatmap_chart,
)
from utils.queries import (
    Q1_KPI_SUMMARY, Q1_CITY_HEATMAP, Q1_HOURLY_TRENDS,
    Q1_VEHICLE_BREAKDOWN, Q1_PICKUP_DEMAND, Q1_STATUS_DIST,
)
from utils.theme import apply_theme, LIGHT_THEME, DARK_THEME   # CSS injector

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rapido Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME SELECTION
# ─────────────────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# THEME resolved before CSS injection — toggle is rendered inside the sidebar block below
THEME = DARK_THEME if st.session_state.theme == "dark" else LIGHT_THEME

apply_theme(THEME)   # injects background / card / accent CSS from theme.py

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — adapts to the active theme
# ─────────────────────────────────────────────────────────────────────────────
_is_dark = st.session_state.theme == "dark"

_metric_bg      = "#1A1A1A"   if _is_dark else "#F7F7F7"
_metric_border  = "#2C2C2C"   if _is_dark else "#E0E0E0"
_metric_label   = "#7A7A7A"   if _is_dark else "#555555"
_metric_value   = "#FFD600"   if _is_dark else "#B8960C"   # muted gold on light
_hr_color       = "#2C2C2C"   if _is_dark else "#E0E0E0"
_section_color  = "#7A7A7A"   if _is_dark else "#888888"
_expander_bg    = "#1A1A1A"   if _is_dark else "#F7F7F7"
_expander_bdr   = "#2C2C2C"   if _is_dark else "#E0E0E0"
_sidebar_bg     = "#111111"   if _is_dark else "#F0F0F0"
_sidebar_bdr    = "#2C2C2C"   if _is_dark else "#DDDDDD"
_sidebar_text   = "#E8E8E8"   if _is_dark else "#1A1A1A"
_body_bg        = "#0D0D0D"   if _is_dark else "#FFFFFF"
_body_text      = "#E8E8E8"   if _is_dark else "#1A1A1A"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: {_body_bg} !important;
    color: {_body_text} !important;
}}

/* ── Top header bar & decoration strip ── */
[data-testid="stHeader"] {{
    background-color: {_body_bg} !important;
    border-bottom: 1px solid {_metric_border} !important;
}}
[data-testid="stDecoration"] {{
    display: none !important;
}}
[data-testid="stAppViewContainer"] {{
    background-color: {_body_bg} !important;
}}
[data-testid="stMain"] {{
    background-color: {_body_bg} !important;
}}
[data-testid="stToolbar"] button svg {{
    fill: {_body_text} !important;
}}

[data-testid="stSidebar"] {{
    background: {_sidebar_bg} !important;
    border-right: 1px solid {_sidebar_bdr} !important;
}}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {{
    color: {_sidebar_text} !important;
}}

[data-testid="stMetric"] {{
    background: {_metric_bg};
    border: 1px solid {_metric_border};
    border-radius: 12px;
    padding: 16px 20px;
}}
[data-testid="stMetric"] label      {{ color: {_metric_label} !important; font-size: 12px !important; }}
[data-testid="stMetricValue"]        {{ color: {_metric_value} !important; font-size: 28px !important; font-weight: 700 !important; }}
[data-testid="stMetricDelta"]        {{ font-size: 12px !important; }}

hr {{ border-color: {_hr_color} !important; }}

[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

[data-testid="stExpander"] {{
    background: {_expander_bg} !important;
    border: 1px solid {_expander_bdr} !important;
    border-radius: 10px !important;
}}

.section-title {{
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {_section_color};
    margin: 24px 0 12px;
}}

.badge-high   {{ background:#FF4B4B22; color:#FF4B4B; border:1px solid #FF4B4B44; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }}
.badge-medium {{ background:#FFB40022; color:#FFB400; border:1px solid #FFB40044; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }}
.badge-low    {{ background:#00C48C22; color:#00C48C; border:1px solid #00C48C44; border-radius:6px; padding:2px 10px; font-size:12px; font-weight:600; }}

.alert-critical {{ background:#FF4B4B18; border-left:4px solid #FF4B4B; border-radius:8px; padding:12px 16px; margin:6px 0; }}
.alert-warning  {{ background:#FFB40018; border-left:4px solid #FFB400; border-radius:8px; padding:12px 16px; margin:6px 0; }}
.alert-info     {{ background:#4B9EFF18; border-left:4px solid #4B9EFF; border-radius:8px; padding:12px 16px; margin:6px 0; }}

.stPlotlyChart {{ border-radius: 12px; overflow: hidden; }}
.stSelectbox > div > div, .stSlider {{ border-color: {_metric_border} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # 1 — Dark / light toggle at the very top
    theme_toggle = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"))
    st.session_state.theme = "dark" if theme_toggle else "light"

    st.markdown("---")

    # 2 — Company logo / name
    st.markdown(f"""
    <div style='padding:12px 0 20px;'>
      <div style='font-size:24px; font-weight:700; color:#FFD600; letter-spacing:-0.5px;'>⚡ Rapido</div>
      <div style='font-size:11px; color:{_section_color}; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;'>Ride Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)

    # 3 — Navigation
    page = st.radio(
        "Navigate",
        ["🏠  Overview", "🔮  Predictions", "📊  Analytics", "🎯  Strategy"],
        label_visibility="collapsed",
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if "Overview" in page:
    # =========================================================================
    # PAGE 1 — OVERVIEW
    # =========================================================================
    st.markdown(f"<h1 style='font-size:28px;font-weight:700;margin-bottom:4px;color:{_body_text};'>Platform Overview</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{_section_color};margin-top:0;'>Real-time KPIs and demand intelligence across all cities</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    with st.spinner("Loading KPIs…"):
        kpi = run_query(Q1_KPI_SUMMARY).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rides",       f"{int(kpi.total_rides):,}")
    c2.metric("Avg Fare",          f"₹{kpi.avg_fare:,.2f}")
    c3.metric("Cancellation Rate", f"{kpi.cancel_rate_pct:.1f}%")
    c4.metric("Avg Delay",         f"{kpi.avg_delay_min:.1f} min" if kpi.avg_delay_min else "—")
    c5.metric("Avg Surge",         f"{kpi.avg_surge:.2f}x")

    st.markdown("")

    # ── Row 2: Status donut + City bar ─────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("<div class='section-title'>Booking Status</div>", unsafe_allow_html=True)
        status_df = run_query(Q1_STATUS_DIST)
        # Build pie directly so we can set marker_colors per slice in the correct order
        fig_status = go.Figure(go.Pie(
            labels=status_df["booking_status"],
            values=status_df["count"],
            hole=0.45,
            marker_colors=[STATUS_COLORS.get(s, "#888") for s in status_df["booking_status"]],
            textposition="outside",
            textinfo="percent+label",
        ))
        apply_chart_theme(fig_status, THEME, height=320)
        st.plotly_chart(fig_status, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>Rides by City</div>", unsafe_allow_html=True)
        city_df = run_query(Q1_CITY_HEATMAP)
        fig_city = bar_chart(
            city_df.sort_values("total_rides", ascending=True).tail(12),
            x="total_rides", y="city",
            title="", height=320
        )
        fig_city.update_traces(marker_color=RAPIDO_YELLOW, orientation="h")
        fig_city.update_layout(xaxis_title="Total Rides", yaxis_title="")
        apply_chart_theme(fig_city, THEME, height=320)
        st.plotly_chart(fig_city, use_container_width=True)

    # ── Row 3: Hourly trends ───────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Hourly Ride Trends</div>", unsafe_allow_html=True)
    hourly_df = run_query(Q1_HOURLY_TRENDS)

    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly_df["hour_of_day"], y=hourly_df["total_rides"],
        name="Total Rides", marker_color=RAPIDO_YELLOW, opacity=0.8,
        yaxis="y",
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly_df["hour_of_day"], y=hourly_df["cancel_rate_pct"],
        name="Cancel Rate %", line=dict(color="#FF4B4B", width=2.5),
        mode="lines+markers", yaxis="y2",
    ))
    apply_chart_theme(fig_hourly, THEME, height=360)
    fig_hourly.update_layout(
        yaxis=dict(title="Total Rides",    gridcolor=THEME["border"]),
        yaxis2=dict(title="Cancel Rate %", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Hour of Day", tickmode="linear", gridcolor=THEME["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        barmode="overlay",
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

    # ── Row 4: Vehicle breakdown + City cancel rates ───────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("<div class='section-title'>Vehicle Type Performance</div>", unsafe_allow_html=True)
        veh_df = run_query(Q1_VEHICLE_BREAKDOWN)
        fig_veh = go.Figure()
        fig_veh.add_trace(go.Bar(
            x=veh_df["vehicle_type"], y=veh_df["total_rides"],
            name="Rides", marker_color=RAPIDO_YELLOW,
        ))
        fig_veh.add_trace(go.Scatter(
            x=veh_df["vehicle_type"], y=veh_df["cancel_rate_pct"],
            name="Cancel %", mode="lines+markers",
            line=dict(color="#FF4B4B", width=2), yaxis="y2",
        ))
        apply_chart_theme(fig_veh, THEME, height=320)
        fig_veh.update_layout(
            yaxis=dict(gridcolor=THEME["border"]),
            yaxis2=dict(overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=THEME["border"]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_veh, use_container_width=True)

    with col_d:
        st.markdown("<div class='section-title'>City Cancellation Rate</div>", unsafe_allow_html=True)
        fig_cancel = bar_chart(
            city_df.sort_values("cancel_rate_pct", ascending=True),
            x="cancel_rate_pct", y="city",
            title="", height=320,
        )
        fig_cancel.update_traces(
            marker_color=[
                "#FF4B4B" if r > 30 else "#FFB400" if r > 20 else "#00C48C"
                for r in city_df.sort_values("cancel_rate_pct", ascending=True)["cancel_rate_pct"]
            ],
            orientation="h",
        )
        fig_cancel.update_layout(xaxis_title="Cancellation Rate (%)", yaxis_title="")
        apply_chart_theme(fig_cancel, THEME, height=320)
        st.plotly_chart(fig_cancel, use_container_width=True)

    # ── Row 5: City demand table ───────────────────────────────────────────────
    with st.expander("📍 Full City Demand Table", expanded=False):
        display_cols = ["city", "total_rides", "completed", "cancelled",
                        "cancel_rate_pct", "avg_fare", "avg_surge"]
        st.dataframe(
            city_df[display_cols].style.format({
                "cancel_rate_pct": "{:.1f}%",
                "avg_fare": "₹{:.2f}",
                "avg_surge": "{:.2f}x",
            }),
            use_container_width=True, hide_index=True,
        )

elif "Predictions" in page:
    exec(open(os.path.join(os.path.dirname(__file__), "sections", "predictions.py"), encoding="utf-8").read())

elif "Analytics" in page:
    exec(open(os.path.join(os.path.dirname(__file__), "sections", "analytics.py"), encoding="utf-8").read())

elif "Strategy" in page:
    exec(open(os.path.join(os.path.dirname(__file__), "sections", "strategy.py"), encoding="utf-8").read())