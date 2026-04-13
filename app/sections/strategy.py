# -*- coding: utf-8 -*-
# =============================================================================
# pages/strategy.py — Ops recommendations · Driver allocation · Revenue · Customers
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
    RAPIDO_YELLOW, RAPIDO_MUTED,
    apply_chart_theme, heatmap_chart,
)
from utils.theme  import LIGHT_THEME, DARK_THEME
from utils.queries import (
    Q4_DRIVER_DEMAND,
    Q4_DRIVER_PERFORMANCE,
    Q4_OPS_ALERTS,
    Q4_SURGE_STRATEGY,
    Q4_CUSTOMER_SEGMENTS,
    Q4_REVENUE_IMPACT,
    Q4_RISK_BY_CITY_VEHICLE,
    Q4_REASSIGNMENT_CANDIDATES,
    Q4_DRIVER_EFFICIENCY,
    Q4_DRIVER_TIER_SUMMARY,
    Q4_UNDERSTAFFED_SLOTS,
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
_is_dark    = st.session_state.get("theme", "light") == "dark"
THEME       = DARK_THEME if _is_dark else LIGHT_THEME

_card_bg    = THEME["card"]
_card_bdr   = THEME["border"]
_text_main  = THEME["text"]
_text_muted = "#7A7A7A" if _is_dark else "#888888"
_text_sub   = "#B8B8B8" if _is_dark else "#555555"
_accent     = THEME["accent"]

RISK_COLOR  = {"High": "#FF4B4B", "Medium": "#FFB400", "Low": "#00C48C"}
TIER_COLOR  = {"Elite": RAPIDO_YELLOW, "Reliable": "#00C48C",
               "Developing": "#FFB400", "At Risk": "#FF4B4B"}
ALLOC_COLOR = {"Understaffed": "#FF4B4B", "Stretched": "#FFB400",
               "Balanced": "#00C48C", "Overstaffed": "#4B9EFF"}

def _themed(fig, height=380):
    return apply_chart_theme(fig, THEME, height=height)

def _card(content: str):
    """Thin wrapper for a themed info card."""
    return f"""
    <div style='background:{_card_bg};border:1px solid {_card_bdr};border-radius:10px;
                padding:14px 16px;margin:6px 0;font-size:13px;color:{_text_sub};
                line-height:1.7;'>
      {content}
    </div>"""

def _badge(label: str, color: str):
    return (f"<span style='background:{color}22;color:{color};font-size:11px;"
            f"font-weight:700;border-radius:4px;padding:2px 8px;'>{label}</span>")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='font-size:28px;font-weight:700;margin-bottom:4px;"
    f"color:{_text_main};'>Strategy</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:{_text_muted};'>"
    f"Ops alerts · Driver allocation · Revenue impact · Customer segments</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🚨 Ops Alerts", "🚗 Driver Allocation", "💰 Revenue", "👥 Customers"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OPS ALERTS
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Rule-Based Operational Alerts</div>",
                unsafe_allow_html=True)
    st.caption("Generated from UC2/UC3 model outputs — updated every 5 min")

    alerts_df = run_query(Q4_OPS_ALERTS)

    if alerts_df.empty:
        st.success("✅ No active alerts. All operations normal.")
    else:
        critical = alerts_df[alerts_df["alert_level"].str.startswith("CRITICAL")]
        warning  = alerts_df[alerts_df["alert_level"].str.startswith("WARNING")]
        info     = alerts_df[alerts_df["alert_level"].str.startswith("INFO")]

        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("🔴 Critical Alerts", len(critical))
        ac2.metric("🟡 Warnings",        len(warning))
        ac3.metric("🔵 Info Alerts",     len(info))

        st.markdown("")

        for _, row in alerts_df.iterrows():
            level = row["alert_level"]
            if "CRITICAL" in level:
                css_class, icon = "alert-critical", "🔴"
            elif "WARNING" in level:
                css_class, icon = "alert-warning", "🟡"
            else:
                css_class, icon = "alert-info", "🔵"

            pre_pos = row.get("pre_position_by", "—")
            st.markdown(f"""
            <div class='{css_class}'>
              <div style='font-size:13px;font-weight:700;margin-bottom:4px;'>{icon} {level}</div>
              <div style='font-size:13px;color:{_text_sub};'>
                <b>{row['city']}</b> · Hour {int(row['hour_of_day']):02d}:00
                · {row['vehicle_type']}
                · {int(row['ride_count'])} rides
                · {int(row['high_risk_count'])} high-risk ({row['high_risk_pct']:.0f}%)
                · Avg Surge <b>{row['avg_surge']:.2f}x</b>
                · Pre-position by <b>{pre_pos}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='section-title'>Surge Pricing Recommendations</div>",
                unsafe_allow_html=True)
    surge_strat_df = run_query(Q4_SURGE_STRATEGY)
    if not surge_strat_df.empty:
        def color_rec(val):
            if val == "Reduce Surge to Retain Riders":
                return "color:#FF4B4B;font-weight:600"
            elif val == "Monitor — Surge Moderate":
                return "color:#FFB400;font-weight:600"
            return "color:#00C48C"

        st.dataframe(
            surge_strat_df.style
                .format({
                    "avg_surge":       "{:.2f}x",
                    "cancel_rate_pct": "{:.1f}%",
                })
                .applymap(color_rec, subset=["recommendation"]),
            use_container_width=True,
            hide_index=True,
            height=340,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — DRIVER ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Load all data upfront ───────────────────────────────────────────────
    demand_df      = run_query(Q4_DRIVER_DEMAND)
    risk_veh_df    = run_query(Q4_RISK_BY_CITY_VEHICLE)
    reassign_df    = run_query(Q4_REASSIGNMENT_CANDIDATES)
    efficiency_df  = run_query(Q4_DRIVER_EFFICIENCY)
    tier_sum_df    = run_query(Q4_DRIVER_TIER_SUMMARY)
    understaff_df  = run_query(Q4_UNDERSTAFFED_SLOTS)
    perf_df        = run_query(Q4_DRIVER_PERFORMANCE)

    # ── KPI summary row ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Fleet Snapshot</div>",
                unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    if not demand_df.empty:
        understaffed_count = (demand_df["allocation_status"] == "Understaffed").sum()
        stretched_count    = (demand_df["allocation_status"] == "Stretched").sum()
        k1.metric("🔴 Understaffed Slots", int(understaffed_count))
        k2.metric("🟡 Stretched Slots",    int(stretched_count))

    if not efficiency_df.empty:
        elite_n   = (efficiency_df["driver_tier"] == "Elite").sum()
        atrisk_n  = (efficiency_df["driver_tier"] == "At Risk").sum()
        k3.metric("⭐ Elite Drivers",   int(elite_n))
        k4.metric("⚠️ At-Risk Drivers", int(atrisk_n))

    if not reassign_df.empty:
        k5.metric("🔄 Reassign Queue", len(reassign_df))

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION A — Demand Heatmap
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>Driver Demand Heatmap — Rides per Driver (Hour × City)</div>",
                unsafe_allow_html=True)
    st.caption("Red = high demand (≥1.15 rides/driver). Subtle peaks matter — act early with pre-positioning.")
    if not demand_df.empty:
        pivot_demand = demand_df.pivot_table(
            index="hour_of_day",
            columns="city",
            values="rides_per_driver",
            fill_value=0,
            aggfunc="mean",
        )

        # Map each cell to a 0-3 status bucket for discrete coloring
        def _alloc_val(v):
            if v >= 1.15: return 3   # High Demand
            if v >= 1.08: return 2   # Peak Lean
            if v >= 0.95: return 1   # Balanced
            return 0                 # Low Demand

        pivot_status = pivot_demand.applymap(_alloc_val)

        hours  = [f"{int(h):02d}:00" for h in pivot_demand.index.tolist()]
        cities = pivot_demand.columns.tolist()

        # Per-cell text shows the actual rides/driver value
        text_vals = [[f"{v:.2f}" if v > 0 else "" for v in row]
                     for row in pivot_demand.values.tolist()]

        fig_demand = go.Figure(go.Heatmap(
            z=pivot_status.values.tolist(),
            x=cities,
            y=hours,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=11, color="rgba(0,0,0,0.72)"),
            colorscale=[
                [0.00, "#4B9EFF"],  # Low Demand  — blue
                [0.33, "#4B9EFF"],
                [0.33, "#00C48C"],  # Balanced    — green
                [0.55, "#00C48C"],
                [0.55, "#FFB400"],  # Peak Lean   — amber
                [0.78, "#FFB400"],
                [0.78, "#FF4B4B"],  # High Demand — red
                [1.00, "#FF4B4B"],
            ],
            zmin=0, zmax=3,
            showscale=False,
            hovertemplate="Hour: %{y}<br>City: %{x}<br>Rides/Driver: %{text}<extra></extra>",
        ))
        fig_demand.update_layout(
            height=500,
            xaxis=dict(title="City", side="bottom", tickfont=dict(size=12)),
            yaxis=dict(title="Hour of Day", autorange="reversed",
                       tickfont=dict(size=11)),
            margin=dict(l=60, r=20, t=20, b=50),
            plot_bgcolor=THEME.get("bg", "#FFFFFF"),
            paper_bgcolor=THEME.get("bg", "#FFFFFF"),
            font=dict(color=THEME.get("text", "#111111")),
        )
        st.plotly_chart(fig_demand, use_container_width=True)

        # Allocation status legend
        leg1, leg2, leg3, leg4 = st.columns(4)
        for col, label, color, rule in [
            (leg1, "High Demand", "#FF4B4B", "≥ 1.15 — Add drivers immediately"),
            (leg2, "Peak Lean",   "#FFB400", "1.08–1.15 — Pre-position drivers"),
            (leg3, "Balanced",    "#00C48C", "0.95–1.08 — Healthy zone"),
            (leg4, "Low Demand",  "#4B9EFF", "< 0.95 — Shift drivers out"),
        ]:
            col.markdown(f"""
            <div style='background:{color}18;border-left:3px solid {color};
                        border-radius:6px;padding:8px 10px;font-size:12px;'>
              <b style='color:{color};'>{label}</b><br>
              <span style='color:{_text_muted};'>{rule}</span>
            </div>""", unsafe_allow_html=True)

    else:
        st.info("ℹ️ Driver demand data unavailable — check bookings ↔ predictions join.")

    st.markdown("")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION B — Understaffed Slots Table
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🚨 High Demand Slots — Action Required</div>",
                unsafe_allow_html=True)
    st.caption("High demand slots (≥1.15 rides/driver). Deploy drivers before the pre-position time.")
    if not understaff_df.empty:
        st.dataframe(
            understaff_df.style
                .format({
                    "rides_per_driver":    "{:.1f}",
                    "avg_cancel_prob_pct": "{:.1f}%",
                })
                .background_gradient(subset=["rides_per_driver"], cmap="YlOrRd")
                .background_gradient(subset=["high_risk_rides"],  cmap="Reds"),
            use_container_width=True,
            hide_index=True,
            height=320,
        )
    else:
        st.success("✅ No critically understaffed slots found.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION C — Cancel Risk Heatmap: City × Vehicle Type
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>Cancel Risk Heatmap — City × Vehicle Type</div>",
                unsafe_allow_html=True)
    st.caption("Avg predicted cancel probability (%). Darker red = reassign-first priority.")

    if not risk_veh_df.empty:
        pivot_risk = (
            risk_veh_df
            .groupby(["city", "vehicle_type"])["avg_cancel_prob_pct"]
            .mean()
            .reset_index()
            .pivot(index="vehicle_type", columns="city", values="avg_cancel_prob_pct")
            .round(1)
        )
        fig_risk_hm = heatmap_chart(pivot_risk, title="", height=300, color_scale="RdYlGn_r")
        fig_risk_hm = apply_chart_theme(fig_risk_hm, THEME, height=300)
        fig_risk_hm.update_xaxes(title="City")
        fig_risk_hm.update_yaxes(title="Vehicle Type")
        fig_risk_hm.update_traces(
            hovertemplate="Vehicle: %{y}<br>City: %{x}<br>Avg Cancel Prob: %{z:.1f}%<extra></extra>"
        )
        st.plotly_chart(fig_risk_hm, use_container_width=True)

        # Stacked bar: risk tier count by city
        st.markdown("<div class='section-title'>Cancel Risk Volume — City Breakdown</div>",
                    unsafe_allow_html=True)
        fig_stack = go.Figure()
        for tier in ["High", "Medium", "Low"]:
            sub      = risk_veh_df[risk_veh_df["cancel_risk_tier"] == tier]
            city_agg = sub.groupby("city")["ride_count"].sum().reset_index()
            fig_stack.add_trace(go.Bar(
                name=f"{tier} Risk",
                x=city_agg["city"],
                y=city_agg["ride_count"],
                marker_color=RISK_COLOR[tier],
                hovertemplate=f"<b>{tier} Risk</b><br>City: %{{x}}<br>Rides: %{{y:,}}<extra></extra>",
            ))
        fig_stack = _themed(fig_stack, 360)
        fig_stack.update_layout(
            barmode="stack",
            xaxis=dict(title="City", gridcolor=THEME["border"]),
            yaxis=dict(title="Number of Rides", gridcolor=THEME["border"]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        # Top 5 worst segments callout cards
        worst = (
            risk_veh_df[risk_veh_df["cancel_risk_tier"] == "High"]
            .groupby(["city", "vehicle_type"])
            .agg(
                high_risk_rides=("ride_count", "sum"),
                avg_prob=("avg_cancel_prob_pct", "mean"),
                avg_fare=("avg_predicted_fare", "mean"),
            )
            .reset_index()
            .sort_values("high_risk_rides", ascending=False)
            .head(5)
        )
        if not worst.empty:
            st.markdown(
                f"<div style='margin-top:10px;margin-bottom:6px;font-size:13px;"
                f"font-weight:600;color:{_text_main};'>"
                f"🔴 Top 5 High-Risk Segments — Prioritise Driver Reassignment</div>",
                unsafe_allow_html=True,
            )
            for _, row in worst.iterrows():
                st.markdown(f"""
                <div style='background:{_card_bg};border:1px solid {_card_bdr};
                            border-left:4px solid #FF4B4B;border-radius:8px;
                            padding:10px 14px;margin:5px 0;font-size:13px;'>
                  <b style='color:#FF4B4B;'>{row['city']} — {row['vehicle_type']}</b>
                  &nbsp;·&nbsp; {int(row['high_risk_rides']):,} high-risk rides
                  &nbsp;·&nbsp; Avg cancel prob <b>{row['avg_prob']:.1f}%</b>
                  &nbsp;·&nbsp; Avg fare <b>₹{row['avg_fare']:.0f}</b>
                  &nbsp;·&nbsp;
                  <span style='color:{_text_muted};'>Action: Reassign Elite/Reliable driver</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION D — Reassignment Candidates
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>🔄 Reassignment Queue — High-Risk Bookings + Best Available Driver</div>",
                unsafe_allow_html=True)
    st.caption("Each row is a high-risk booking paired with the top-scoring available driver in the same city.")

    if not reassign_df.empty:
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Bookings Needing Reassignment", len(reassign_df))
        rc2.metric("Avg Cancel Probability",
                   f"{reassign_df['cancel_prob_pct'].mean():.1f}%")
        rc3.metric("Avg Replacement Score",
                   f"{reassign_df['replacement_score'].mean():.2f}")

        st.dataframe(
            reassign_df.style
                .format({
                    "cancel_prob_pct":   "{:.1f}%",
                    "predicted_fare":    "₹{:.2f}",
                    "driver_rating":     "{:.2f}",
                    "acceptance_rate":   "{:.0%}",
                    "replacement_score": "{:.3f}",
                })
                .background_gradient(subset=["cancel_prob_pct"],   cmap="Reds")
                .background_gradient(subset=["replacement_score"],  cmap="Greens"),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
    else:
        st.info("ℹ️ No reassignment candidates found — either no high-risk rides or no "
                "elite drivers available in matching cities.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION E — Driver Efficiency Scatter
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>Driver Efficiency — Completion Rate vs Total Earnings</div>",
                unsafe_allow_html=True)
    st.caption("Each dot = one driver. Size = total rides. Colour = performance tier.")

    if not efficiency_df.empty:
        fig_eff = px.scatter(
            efficiency_df,
            x="completion_rate_pct",
            y="total_earned",
            size="total_rides",
            color="driver_tier",
            hover_data=["driver_id", "city", "vehicle_type", "avg_surge_worked"],
            color_discrete_map=TIER_COLOR,
            labels={
                "completion_rate_pct": "Completion Rate (%)",
                "total_earned":        "Total Fare Earned (₹)",
                "total_rides":         "Total Rides",
                "driver_tier":         "Tier",
            },
            category_orders={"driver_tier": ["Elite", "Reliable", "Developing", "At Risk"]},
        )
        fig_eff = _themed(fig_eff, 440)
        fig_eff.update_layout(
            xaxis=dict(gridcolor=THEME["border"], range=[0, 105]),
            yaxis=dict(gridcolor=THEME["border"]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_eff, use_container_width=True)

        # Tier breakdown table
        st.markdown("<div class='section-title'>Driver Tier Distribution</div>",
                    unsafe_allow_html=True)

        tier_counts = (
            efficiency_df.groupby("driver_tier")
            .agg(
                drivers=("driver_id", "count"),
                avg_completion=("completion_rate_pct", "mean"),
                avg_earned=("total_earned", "mean"),
                avg_rides=("total_rides", "mean"),
            )
            .reset_index()
            .sort_values("avg_completion", ascending=False)
        )

        tc1, tc2, tc3, tc4 = st.columns(4)
        for col, tier in zip([tc1, tc2, tc3, tc4],
                             ["Elite", "Reliable", "Developing", "At Risk"]):
            sub = tier_counts[tier_counts["driver_tier"] == tier]
            if not sub.empty:
                row   = sub.iloc[0]
                color = TIER_COLOR[tier]
                col.markdown(f"""
                <div style='background:{color}18;border:1px solid {color}44;
                            border-radius:10px;padding:14px;text-align:center;'>
                  <div style='font-size:22px;font-weight:700;color:{color};'>
                    {int(row['drivers'])}
                  </div>
                  <div style='font-size:13px;font-weight:600;color:{color};
                              margin:2px 0;'>{tier}</div>
                  <div style='font-size:11px;color:{_text_muted};'>
                    Avg {row['avg_completion']:.0f}% completion<br>
                    ₹{row['avg_earned']:,.0f} avg earned
                  </div>
                </div>""", unsafe_allow_html=True)

        # Allocation rules per tier
        st.markdown("")
        st.markdown(
            f"<div style='font-size:13px;font-weight:600;color:{_text_main};"
            f"margin:10px 0 6px;'>Allocation Rules by Tier</div>",
            unsafe_allow_html=True,
        )
        rules = [
            ("⭐", "Elite",      RAPIDO_YELLOW, "First pick for high-risk reassignments & peak hour surge slots"),
            ("🟢", "Reliable",   "#00C48C",     "Standard peak-hour deployment; eligible for reassignment"),
            ("🟡", "Developing", "#FFB400",     "Off-peak slots & low-surge zones; monitor progress"),
            ("🔴", "At Risk",    "#FF4B4B",     "Flag for coaching; restrict from high-surge assignments"),
        ]
        for icon, tier, color, rule in rules:
            st.markdown(f"""
            <div style='background:{_card_bg};border:1px solid {_card_bdr};
                        border-left:4px solid {color};border-radius:8px;
                        padding:10px 14px;margin:5px 0;display:flex;
                        align-items:center;gap:12px;'>
              <span style='font-size:18px;'>{icon}</span>
              <div>
                <span style='font-size:13px;font-weight:600;color:{color};'>{tier}</span>
                <span style='font-size:13px;color:{_text_sub};margin-left:10px;'>{rule}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION F — Top Driver Performance Scores (original, enhanced with tiers)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>Top Driver Performance Scores</div>",
                unsafe_allow_html=True)

    if not perf_df.empty:
        col_f1, col_f2 = st.columns([3, 1])

        with col_f1:
            bar_colors = [
                TIER_COLOR.get(t, RAPIDO_MUTED)
                for t in perf_df["driver_tier"][:20]
            ]
            fig_perf = go.Figure(go.Bar(
                x=perf_df["driver_id"][:20],
                y=perf_df["performance_score"][:20],
                marker_color=bar_colors,
                text=perf_df["performance_score"][:20].apply(lambda x: f"{x:.2f}"),
                textposition="outside",
                customdata=perf_df["driver_tier"][:20],
                hovertemplate=(
                    "Driver: %{x}<br>Score: %{y:.3f}<br>Tier: %{customdata}<extra></extra>"
                ),
            ))
            fig_perf = _themed(fig_perf, 380)
            fig_perf.update_layout(
                xaxis=dict(title="Driver ID", gridcolor=THEME["border"],
                           tickangle=-45, tickfont=dict(size=10)),
                yaxis=dict(title="Performance Score", gridcolor=THEME["border"],
                           range=[0, 1.1]),
                margin=dict(l=10, r=10, t=20, b=80),
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        with col_f2:
            st.markdown("**Score Formula**")
            st.markdown(f"""
            <div style='background:{_card_bg};border:1px solid {_card_bdr};
                        border-radius:10px;padding:16px;font-size:13px;
                        color:{_text_sub};line-height:1.8;'>
              <b style='color:{_accent};'>Perf Score =</b><br>
              Rating × 0.4<br>
              + Accept Rate × 0.3<br>
              + (1 − Delay Rate) × 0.3<br><br>
              <span style='font-size:11px;color:{_text_muted};'>Range: 0.0 — 1.0</span><br><br>
              <b style='color:{_accent};'>Tiers:</b><br>
              <span style='color:{RAPIDO_YELLOW};'>⭐ Elite</span> ≥ 0.80<br>
              <span style='color:#00C48C;'>🟢 Reliable</span> ≥ 0.60<br>
              <span style='color:#FFB400;'>🟡 Developing</span> ≥ 0.40<br>
              <span style='color:#FF4B4B;'>🔴 At Risk</span> &lt; 0.40
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — REVENUE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Revenue Impact of Cancellations by City</div>",
                unsafe_allow_html=True)
    rev_df = run_query(Q4_REVENUE_IMPACT)

    if not rev_df.empty:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(
            x=rev_df["city"], y=rev_df["actual_revenue"],
            name="Actual Revenue", marker_color="#4B9EFF",
        ))
        fig_rev.add_trace(go.Bar(
            x=rev_df["city"], y=rev_df["estimated_lost_revenue"],
            name="Est. Lost Revenue", marker_color="#FF4B4B",
        ))
        fig_rev = _themed(fig_rev, 380)
        fig_rev.update_layout(
            barmode="group",
            xaxis=dict(gridcolor=THEME["border"]),
            yaxis=dict(title="₹ Revenue", gridcolor=THEME["border"]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_rev, use_container_width=True)

        total_lost   = rev_df["estimated_lost_revenue"].sum()
        total_earned = rev_df["actual_revenue"].sum()
        lost_pct     = 100 * total_lost / (total_lost + total_earned)

        r1, r2, r3 = st.columns(3)
        r1.metric("Total Actual Revenue",   f"₹{total_earned:,.0f}")
        r2.metric("Estimated Lost Revenue", f"₹{total_lost:,.0f}")
        r3.metric("Revenue Loss Rate",      f"{lost_pct:.1f}%")

        st.markdown("<div class='section-title'>City Revenue Table</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            rev_df.sort_values("estimated_lost_revenue", ascending=False)
                  .style.format({
                      "avg_booking_value":      "₹{:.2f}",
                      "estimated_lost_revenue": "₹{:,.0f}",
                      "actual_revenue":         "₹{:,.0f}",
                  })
                  .background_gradient(subset=["estimated_lost_revenue"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Customer Segmentation</div>",
                unsafe_allow_html=True)
    seg_df = run_query(Q4_CUSTOMER_SEGMENTS)

    if not seg_df.empty:
        SEGMENT_COLORS = {
            "Champion":         RAPIDO_YELLOW,
            "Loyal":            "#4B9EFF",
            "Potential":        "#00C48C",
            "At Risk":          "#FF4B4B",
            "New / Occasional": RAPIDO_MUTED,
        }

        col_s1, col_s2 = st.columns([1.2, 1])

        with col_s1:
            fig_seg = go.Figure(go.Pie(
                labels=seg_df["segment"],
                values=seg_df["customer_count"],
                hole=0.48,
                marker_colors=[SEGMENT_COLORS.get(s, "#888") for s in seg_df["segment"]],
                textinfo="percent+label",
            ))
            fig_seg = _themed(fig_seg, 360)
            fig_seg.update_layout(
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        with col_s2:
            SEGMENT_DESC = {
                "Champion":         "High bookings, low cancel rate. Reward & retain.",
                "Loyal":            "Regular riders. Offer loyalty perks.",
                "Potential":        "Growing riders. Engage with promotions.",
                "At Risk":          "High cancellation. Investigate causes.",
                "New / Occasional": "First-timers. Onboard with incentives.",
            }
            for _, row in seg_df.iterrows():
                seg   = row["segment"]
                color = SEGMENT_COLORS.get(seg, "#888")
                desc  = SEGMENT_DESC.get(seg, "")
                st.markdown(f"""
                <div style='background:{_card_bg};border:1px solid {_card_bdr};
                            border-left:4px solid {color};border-radius:8px;
                            padding:12px 14px;margin:6px 0;'>
                  <div style='font-size:13px;font-weight:600;color:{color};'>{seg}</div>
                  <div style='font-size:12px;color:{_text_muted};margin-top:2px;'>{desc}</div>
                  <div style='font-size:12px;color:{_text_sub};margin-top:4px;'>
                    {int(row['customer_count']):,} customers ·
                    Avg {row['avg_bookings']:.0f} rides ·
                    Cancel {row['avg_cancel_rate']*100:.0f}% ·
                    ⭐ {row['avg_rating']:.1f}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Segment Breakdown Table</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            seg_df.style.format({
                "avg_bookings":    "{:.1f}",
                "avg_cancel_rate": "{:.1%}",
                "avg_rating":      "{:.2f}",
            }).background_gradient(subset=["customer_count"], cmap="YlOrRd"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<div class='section-title'>Recommended Actions by Segment</div>",
                    unsafe_allow_html=True)
        ACTIONS = [
            ("🏆", "Champion",         RAPIDO_YELLOW, "Launch VIP tier — early access, no surge cap, dedicated support"),
            ("💙", "Loyal",            "#4B9EFF",     "Introduce milestone rewards — free ride every 10th booking"),
            ("🌱", "Potential",        "#00C48C",     "Send push notifications with 10% off next 3 rides"),
            ("⚠️",  "At Risk",          "#FF4B4B",     "Trigger exit-intent survey + re-engagement promo"),
            ("👋", "New / Occasional", RAPIDO_MUTED,  "Onboarding sequence: 3 welcome rides with fare guarantee"),
        ]
        for icon, seg, color, action in ACTIONS:
            st.markdown(f"""
            <div style='background:{_card_bg};border:1px solid {_card_bdr};
                        border-radius:8px;padding:12px 16px;margin:6px 0;
                        display:flex;align-items:center;gap:12px;'>
              <span style='font-size:20px;'>{icon}</span>
              <div>
                <div style='font-size:13px;font-weight:600;color:{color};'>{seg}</div>
                <div style='font-size:13px;color:{_text_sub};'>{action}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)