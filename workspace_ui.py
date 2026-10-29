"""Shared product navigation. Kept independent of the stock-analysis package."""
import streamlit as st

WORKSPACES = {"stock": "Stock Research", "pe": "PE Deal Intelligence"}


def workspace_switch() -> str:
    """Render an always-selected, keyboard-accessible switch with shareable URLs."""
    if "workspace_choice" not in st.session_state:
        requested = st.query_params.get("workspace", "stock")
        st.session_state.workspace_choice = requested if requested in WORKSPACES else "stock"

    def update_url():
        st.query_params["workspace"] = st.session_state.workspace_choice

    st.markdown("""
    <style>
    .st-key-workspace_switch { max-width: 540px; margin: 0 auto 1.5rem; }
    .st-key-workspace_choice { width: 100% !important; }
    .st-key-workspace_switch [data-testid="stRadio"] { width: 100%; }
    .st-key-workspace_switch [role="radiogroup"] {
        width: 100%;
        display: flex; justify-content: center; gap: .3rem; padding: .3rem;
        border: 1px solid rgba(128,128,128,.25); border-radius: 999px;
        background: var(--secondary-background-color);
    }
    .st-key-workspace_switch [data-testid="stRadio"] label {
        flex: 1; justify-content: center; margin: 0; padding: .6rem .8rem;
        border-radius: 999px; transition: background .18s ease;
    }
    .st-key-workspace_switch label:has(input:checked) {
        background: #2a78d6; color: white;
    }
    .st-key-workspace_switch label[data-baseweb="radio"] > div:first-child,
    .st-key-workspace_switch label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
    .st-key-workspace_switch label:focus-within {
        outline: 3px solid #2a78d6; outline-offset: 3px;
    }
    .st-key-workspace_switch label p { font-size: .95rem; font-weight: 600; }
    @media (max-width: 480px) {
        .st-key-workspace_switch [data-testid="stRadio"] { width: 100%; }
    .st-key-workspace_switch [role="radiogroup"] {
        width: 100%; flex-wrap: nowrap; }
        .st-key-workspace_switch label { padding: .55rem .35rem; }
        .st-key-workspace_switch label p { font-size: .8rem; }
    }
    </style>
    """, unsafe_allow_html=True)
    with st.container(key="workspace_switch"):
        return st.radio("Lighthouse workspace", list(WORKSPACES),
                        format_func=WORKSPACES.get, key="workspace_choice",
                        horizontal=True, label_visibility="collapsed", on_change=update_url)
