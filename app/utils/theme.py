import streamlit as st

LIGHT_THEME = {
    "bg": "#FFFFFF",
    "text": "#1A1A1A",
    "card": "#F7F7F7",
    "border": "#E0E0E0",
    "accent": "#FFD700",  # Rapido yellow
    "plotly_template": "plotly_white"
}

DARK_THEME = {
    "bg": "#0E1117",
    "text": "#FAFAFA",
    "card": "#1A1D24",
    "border": "#2A2F3A",
    "accent": "#FFD700",
    "plotly_template": "plotly_dark"
}

def apply_theme(theme: dict) -> None:
    """Inject CSS variables for the selected theme into the Streamlit app."""
    st.markdown(f"""
        <style>
        /* ── App background & base text ── */
        .stApp {{
            background-color: {theme['bg']};
            color: {theme['text']};
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: {theme['card']};
            border-right: 1px solid {theme['border']};
        }}

        /* ── Metric cards ── */
        [data-testid="stMetric"] {{
            background-color: {theme['card']};
            border: 1px solid {theme['border']};
            border-radius: 10px;
            padding: 12px;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {theme['card']};
            border-bottom: 1px solid {theme['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {theme['text']};
        }}

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {{
            background-color: {theme['card']};
        }}

        /* ── Generic card class ── */
        .card {{
            background-color: {theme['card']};
            border: 1px solid {theme['border']};
            padding: 1rem;
            border-radius: 10px;
        }}

        /* ── Section title helper class ── */
        .section-title {{
            font-size: 14px;
            font-weight: 600;
            color: {theme['text']};
            margin: 16px 0 8px 0;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        /* ── Accent ── */
        .accent {{
            color: {theme['accent']};
        }}

        /* ── Risk badges ── */
        .badge-high   {{ background:#FF4B4B22;color:#FF4B4B;font-size:11px;font-weight:700;border-radius:4px;padding:2px 8px; }}
        .badge-medium {{ background:#FFB40022;color:#FFB400;font-size:11px;font-weight:700;border-radius:4px;padding:2px 8px; }}
        .badge-low    {{ background:#00C48C22;color:#00C48C;font-size:11px;font-weight:700;border-radius:4px;padding:2px 8px; }}
        </style>
    """, unsafe_allow_html=True)