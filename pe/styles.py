"""Reusable Streamlit styles and consistent loading/error/data presentation."""
from __future__ import annotations

import streamlit as st

PAGE_MAX_WIDTH = 1200
PAGE_SPACING = "1.25rem"
COLORS = {"ink": "#172b4d", "muted": "#62748a", "accent": "#2a78d6", "border": "rgba(128,128,128,.25)"}


def apply_shared_styles() -> None:
    st.markdown(f"""
    <style>
      .block-container {{ max-width: {PAGE_MAX_WIDTH}px; padding-top: 1.5rem; padding-bottom: 3rem; }}
      .pe-eyebrow {{ color: {COLORS['muted']}; font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }}
      .pe-panel {{ border: 1px solid {COLORS['border']}; border-radius: .8rem; padding: {PAGE_SPACING}; }}
      [data-testid="stDataFrame"] {{ border: 1px solid {COLORS['border']}; border-radius: .5rem; }}
    </style>
    """, unsafe_allow_html=True)


def loading(message: str = "Loading PE research data…") -> None:
    st.info(message, icon="⏳")


def show_error(message: str, *, detail: str | None = None) -> None:
    st.error(message, icon="⚠️")
    if detail:
        with st.expander("Technical details"):
            st.code(detail)


def show_metadata(metadata) -> None:
    """Show version and coverage consistently, including an explicit cached-mode warning."""
    st.caption(
        f"Data version: {metadata.data_version} · Records: {metadata.record_count} · "
        f"Sources: {metadata.source_count} · Last update: {metadata.last_updated or 'not provided'}"
    )
    st.caption(f"Coverage: {metadata.coverage_notes}")
    if metadata.mode == "cached":
        st.warning("Cached mode: the published dataset was unavailable or invalid. Showing the last-known-good copy.", icon="📦")


def show_table(frame, *, empty_message: str = "No records are available yet.") -> None:
    if frame is None or frame.empty:
        st.info(empty_message)
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)
