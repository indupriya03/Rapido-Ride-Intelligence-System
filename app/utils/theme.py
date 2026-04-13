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

def apply_theme(theme):
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {theme['bg']};
            color: {theme['text']};
        }}

        .card {{
            background-color: {theme['card']};
            border: 1px solid {theme['border']};
            padding: 1rem;
            border-radius: 10px;
        }}

        .accent {{
            color: {theme['accent']};
        }}
        </style>
    """, unsafe_allow_html=True)