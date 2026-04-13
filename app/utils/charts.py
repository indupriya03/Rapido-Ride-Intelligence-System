# =============================================================================
# utils/charts.py — Shared Plotly chart helpers & theme tokens
# =============================================================================

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Brand palette ─────────────────────────────────────────────────────────────
RAPIDO_YELLOW   = "#FFD600"
RAPIDO_BLACK    = "#0D0D0D"
RAPIDO_DARK     = "#161616"
RAPIDO_SURFACE  = "#1E1E1E"
RAPIDO_BORDER   = "#2C2C2C"
RAPIDO_TEXT     = "#E8E8E8"
RAPIDO_MUTED    = "#7A7A7A"

RISK_COLORS = {
    "High":   "#FF4B4B",
    "Medium": "#FFB400",
    "Low":    "#00C48C",
}

ALERT_COLORS = {
    "CRITICAL — Surge + High Cancel Risk": "#FF4B4B",
    "WARNING — High Surge Demand":         "#FF8C00",
    "WARNING — High Cancellation Risk":    "#FFB400",
    "INFO — Moderate Surge":               "#4B9EFF",
}

STATUS_COLORS = {
    "Completed":  "#00C48C",
    "Cancelled":  "#FF4B4B",
    "Incomplete": "#FFB400",
}

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color=RAPIDO_TEXT),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ── Internal helper used by chart builder functions ───────────────────────────
def apply_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply the default dark Rapido theme to a figure. Used internally by
    bar_chart, line_chart, etc. For theme-aware layouts use apply_chart_theme."""
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=title, font=dict(size=14, color=RAPIDO_TEXT)),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=RAPIDO_BORDER,
            font=dict(color=RAPIDO_TEXT),
        ),
    )
    fig.update_xaxes(gridcolor=RAPIDO_BORDER, zerolinecolor=RAPIDO_BORDER)
    fig.update_yaxes(gridcolor=RAPIDO_BORDER, zerolinecolor=RAPIDO_BORDER)
    return fig


# ── Theme-aware helper (accepts a LIGHT_THEME / DARK_THEME dict) ──────────────
def apply_chart_theme(fig: go.Figure, theme: dict, title: str = "", height: int = 380) -> go.Figure:
    """Apply a light or dark theme dict (from utils/theme.py) to a figure."""
    fig.update_layout(
        template=theme["plotly_template"],
        title=dict(text=title, font=dict(size=14, color=theme["text"])),
        height=height,
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["bg"],
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=theme["border"],
            font=dict(color=theme["text"]),
        ),
    )
    fig.update_xaxes(gridcolor=theme["border"], zerolinecolor=theme["border"])
    fig.update_yaxes(gridcolor=theme["border"], zerolinecolor=theme["border"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def bar_chart(df, x, y, color=None, title="", color_map=None, height=380):
    kwargs = dict(x=x, y=y, text_auto=".2s")
    if color:
        kwargs["color"] = color
    if color_map:
        kwargs["color_discrete_map"] = color_map
    fig = px.bar(df, **kwargs, color_discrete_sequence=[RAPIDO_YELLOW])
    return apply_theme(fig, title, height)


def line_chart(df, x, y, color=None, title="", height=380):
    kwargs = dict(x=x, y=y)
    if color:
        kwargs["color"] = color
    fig = px.line(df, **kwargs, markers=True,
                  color_discrete_sequence=[RAPIDO_YELLOW, "#4B9EFF", "#FF4B4B", "#00C48C"])
    return apply_theme(fig, title, height)


def scatter_chart(df, x, y, color=None, title="", height=380, opacity=0.5):
    kwargs = dict(x=x, y=y, opacity=opacity)
    if color:
        kwargs["color"] = color
    fig = px.scatter(df, **kwargs,
                     color_discrete_sequence=[RAPIDO_YELLOW, "#4B9EFF", "#FF4B4B", "#00C48C"])
    return apply_theme(fig, title, height)


def pie_chart(df, names, values, title="", height=340):
    fig = px.pie(
        df, names=names, values=values,
        color_discrete_sequence=[RAPIDO_YELLOW, "#4B9EFF", "#FF4B4B", "#00C48C", "#9B59B6"],
        hole=0.45,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return apply_theme(fig, title, height)


def heatmap_chart(df_pivot, title="", height=420, color_scale="YlOrRd"):
    fig = go.Figure(go.Heatmap(
        z=df_pivot.values,
        x=list(df_pivot.columns),
        y=list(df_pivot.index),
        colorscale=color_scale,
        showscale=True,
        hoverongaps=False,
        colorbar=dict(tickfont=dict(color=RAPIDO_TEXT)),
    ))
    return apply_theme(fig, title, height)


def risk_gauge(probability: float, title: str = "Cancel Risk") -> go.Figure:
    """Semi-circle gauge for a single cancel probability."""
    color = (
        RISK_COLORS["High"]   if probability >= 0.7 else
        RISK_COLORS["Medium"] if probability >= 0.4 else
        RISK_COLORS["Low"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        number=dict(suffix="%", font=dict(size=36, color=color)),
        title=dict(text=title, font=dict(size=14, color=RAPIDO_TEXT)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=RAPIDO_MUTED),
            bar=dict(color=color, thickness=0.25),
            bgcolor=RAPIDO_SURFACE,
            bordercolor=RAPIDO_BORDER,
            steps=[
                dict(range=[0,  40], color="#1A2E1A"),
                dict(range=[40, 70], color="#2E2A14"),
                dict(range=[70,100], color="#2E1414"),
            ],
            threshold=dict(
                line=dict(color=color, width=4),
                thickness=0.75,
                value=probability * 100,
            ),
        ),
    ))
    fig.update_layout(**PLOTLY_THEME, height=260)
    return fig


def fare_vs_actual_scatter(df: pd.DataFrame) -> go.Figure:
    """Predicted vs Actual fare scatter with perfect-fit line."""
    max_val = max(df["actual_fare"].max(), df["predicted_fare"].max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color=RAPIDO_MUTED, dash="dash", width=1),
        name="Perfect Fit",
    ))
    fig.add_trace(go.Scatter(
        x=df["actual_fare"], y=df["predicted_fare"],
        mode="markers",
        marker=dict(
            color=RAPIDO_YELLOW, size=4, opacity=0.4,
            line=dict(width=0),
        ),
        name="Predictions",
        hovertemplate="Actual: %{x:.0f}<br>Predicted: %{y:.0f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title="Predicted vs Actual Fare (UC2)",
        xaxis_title="Actual Fare (₹)",
        yaxis_title="Predicted Fare (₹)",
        height=400,
    )
    return fig