"""PE workspace shell: sidebar navigation between PE pages, with each page in its own module."""
import streamlit as st

PAGES = {"explorer": "Deal Explorer", "sources": "Sources and Coverage", "quality": "Quality and Method"}
PAGE_ICONS = {"explorer": "🔎", "sources": "🗂️", "quality": "✅"}


def _sidebar_nav() -> str:
    """Sidebar page switch. The page is mirrored in `?pe_page=` so a section can be linked directly."""
    if "pe_page" not in st.session_state:
        requested = st.query_params.get("pe_page", "explorer")
        st.session_state.pe_page = requested if requested in PAGES else "explorer"

    def update_url():
        st.query_params["pe_page"] = st.session_state.pe_page

    with st.sidebar:
        st.markdown("### 🏛️ PE Deal Intelligence")
        page = st.radio("PE section", list(PAGES), key="pe_page", on_change=update_url,
                        format_func=lambda p: f"{PAGE_ICONS[p]}  {PAGES[p]}")
        st.divider()
    return page


def _render_sources() -> None:
    st.subheader("Know where the information comes from")
    st.write("This page will show the monitored PE firms, collection period, source links, "
             "last successful refresh and any gaps in coverage.")
    st.caption("Source monitoring is being developed by the collection owner. Public page visits do not trigger scraping.")


def _render_quality() -> None:
    st.subheader("Research you can check")
    st.write("Deal facts should link to supporting announcement excerpts. Missing terms stay unknown, "
             "and repeated coverage of one transaction is grouped together.")
    st.write("This page will report measured extraction accuracy, review status and known limitations "
             "once the team has validated the data.")
    st.caption("A stated acquisition rationale is a research lead, not proof of improved earnings or returns.")


def render_pe_app() -> None:
    page = _sidebar_nav()
    st.caption("LIGHTHOUSE  /  PRIVATE EQUITY  /  " + PAGES[page].upper())
    st.title("PE Deal Intelligence")
    if page == "explorer":
        st.write("**Understand the deal. Follow the evidence.** A dedicated workspace for researching private equity acquisitions: "
                 "who bought whom, why the transaction matters, and what the source actually says.")
        from pe.pages.explorer import render_explorer
        render_explorer()
    elif page == "sources":
        _render_sources()
        st.sidebar.caption("Pick **Deal Explorer** above to search and filter deals.")
    else:
        _render_quality()
        st.sidebar.caption("Pick **Deal Explorer** above to search and filter deals.")
