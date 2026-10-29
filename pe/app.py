"""PE landing page; feature modules can grow here without editing stock research."""
import streamlit as st


def render_pe_app() -> None:
    st.caption("LIGHTHOUSE  /  PRIVATE EQUITY")
    st.title("PE Deal Intelligence")
    st.markdown("### Understand the deal. Follow the evidence.")
    st.write("A dedicated workspace for researching private equity acquisitions: "
             "who bought whom, why the transaction matters, and what the source actually says.")
    st.info("This workspace is under development. Deal collection and analysis are not yet connected.")

    explore, sources, quality = st.tabs(["Deal Explorer", "Sources and Coverage", "Quality and Method"])
    with explore:
        st.subheader("From announcements to a research shortlist")
        st.write("The Deal Explorer will bring public acquisition announcements into one searchable view, "
                 "with filters for firm, industry, geography and acquisition rationale.")
        for col, title, detail in zip(st.columns(3),
                ["Discover transactions", "Inspect the evidence", "Build a shortlist"],
                ["Find acquisitions relevant to your research question.",
                 "Open a deal to see the buyer, target, stated rationale and original source.",
                 "Compare relevant transactions and export them for further investigation."]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.write(detail)
        st.caption("No deal dataset has been published yet. Search and export will become available when records are ready.")
    with sources:
        st.subheader("Know where the information comes from")
        st.write("This page will show the monitored PE firms, collection period, source links, "
                 "last successful refresh and any gaps in coverage.")
        st.caption("Sources have not been configured yet.")
    with quality:
        st.subheader("Research you can check")
        st.write("Deal facts will link to supporting announcement excerpts. Missing terms will stay unknown, "
                 "and repeated coverage of one transaction will be grouped together.")
        st.write("This page will report measured extraction accuracy, review status and known limitations "
                 "once the team has validated the data.")
        st.caption("An announced acquisition rationale is a research lead, not proof of improved earnings or returns.")
