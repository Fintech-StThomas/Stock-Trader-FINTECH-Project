"""PE workspace shell: sidebar navigation between PE pages, with each page in its own module."""
import streamlit as st

PAGES = {"explorer": "AI Deal Research", "detail": "Deal Detail"}
PAGE_ICONS = {"explorer": "🔎", "detail": "🧾"}


def _sidebar_nav() -> str:
    """Sidebar page switch. The page is mirrored in `?pe_page=` so a section can be linked directly."""
    if st.session_state.get("pe_page") not in PAGES:
        requested = st.query_params.get("pe_page", "explorer")
        st.session_state.pe_page = requested if requested in PAGES else "explorer"

    if st.query_params.get("pe_page") and st.query_params["pe_page"] not in PAGES:
        st.query_params["pe_page"] = st.session_state.pe_page

    def update_url():
        st.query_params["pe_page"] = st.session_state.pe_page

    with st.sidebar:
        st.markdown("### 🏛️ PE Deal Intelligence")
        page = st.radio("PE section", list(PAGES), key="pe_page", on_change=update_url,
                        format_func=lambda p: f"{PAGE_ICONS[p]}  {PAGES[p]}")
        st.divider()
    return page


def render_pe_app() -> None:
    page = _sidebar_nav()
    st.caption("LIGHTHOUSE  /  PRIVATE EQUITY  /  " + PAGES[page].upper())
    st.title("PE Deal Intelligence")
    if page == "explorer":
        st.write("**Understand the deal. Follow the evidence.** A dedicated workspace for researching private equity acquisitions: "
                 "who bought whom, why the transaction matters, and what the source actually says.")
        from pe.pages.explorer import render_explorer
        render_explorer()
    elif page == "detail":
        from pe.repository import load_dataset
        from pe.pages.deal_detail import render_deal_detail
        snapshot = load_dataset()
        render_deal_detail(snapshot["deals"], snapshot.get("evidence", []))
