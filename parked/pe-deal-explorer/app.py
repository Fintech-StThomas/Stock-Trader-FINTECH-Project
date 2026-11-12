"""PE Deal Intelligence (parked): the PE workspace as its own Streamlit app.

    cd parked/pe-deal-explorer && streamlit run app.py

Parked in November 2026: it is not part of the deployed Lighthouse app and imports nothing from it. The AI Deal
Research page calls an OpenAI model only when a server-side key is configured (docs/PE_AI_SEARCH.md); without one
the page explains that the search is unavailable, and the saved deals and Deal Detail still work.
"""
import streamlit as st

st.set_page_config(page_title="PE Deal Intelligence (parked)", page_icon="🏛️", layout="wide",
                   menu_items={"About": "PE Deal Intelligence, parked separately from Lighthouse. Decision support, not investment advice."})

from pe.app import render_pe_app  # noqa: E402

render_pe_app()
