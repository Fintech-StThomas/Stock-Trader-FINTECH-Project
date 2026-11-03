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
    from pe.pages.sources import render_sources
    render_sources()


def _render_quality() -> None:
    from pe.pages.quality import render_quality
    render_quality()


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
