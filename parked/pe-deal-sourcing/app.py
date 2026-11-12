"""PE Deal Sourcing (parked): the deal-sourcing product as its own Streamlit app.

    cd parked/pe-deal-sourcing && streamlit run app.py

Parked in November 2026: it is not part of the deployed Lighthouse app and imports nothing from it. It runs offline
on the saved pages in data/pe_deals/cache/ and needs no API key.
"""
import streamlit as st

st.set_page_config(page_title="PE Deal Sourcing (parked)", page_icon="🎯", layout="wide",
                   menu_items={"About": "PE Deal Sourcing, parked separately from Lighthouse. Decision support, not investment advice."})

from pedeals import app_view  # noqa: E402

app_view.render()
