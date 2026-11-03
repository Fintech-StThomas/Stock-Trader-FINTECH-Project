"""Streamlit Deal Explorer page for Lighthouse PE Deal Intelligence."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from pe.analytics import (
    count_by_category,
    deals_over_time,
    filter_deals,
    filter_options,
    load_deals,
    summary_metrics,
)
from pe.exports import data_dictionary_csv, export_filtered

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "pe" / "sample_deals.json"


def load_explorer_dataset(path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, bool]:
    """Load saved PE deal records. Returns (records, is_demo_data)."""
    path = Path(path)
    if not path.exists():
        return load_deals([]), False
    raw = pd.read_json(path)
    is_demo = bool(raw.get("data_status", pd.Series(dtype=str)).astype(str).eq("demo").any())
    return load_deals(raw), is_demo


def _display(value: Any, fallback: str = "Unknown") -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return fallback
    return str(value)


def _fmt_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return pd.Timestamp(value).strftime("%b %d, %Y")


FILTER_KEYS = (
    "pe_explorer_search", "pe_explorer_publishers", "pe_explorer_types", "pe_explorer_industries",
    "pe_explorer_geographies", "pe_explorer_rationales", "pe_explorer_dates",
)
SHORTLIST_STATE_KEY = "pe_explorer_shortlist_ids"
SHORTLIST_RATIONALE_KEY = "pe_explorer_shortlist_rationales"


def _clear_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


def _date_bounds(value: Any) -> tuple[date | None, date | None]:
    """st.date_input returns a 1-tuple while the user is mid-way through picking a range."""
    if isinstance(value, (tuple, list)):
        start = value[0] if len(value) >= 1 else None
        end = value[1] if len(value) >= 2 else None
        return start, end
    if isinstance(value, date):
        return value, value
    return None, None


def render_filter_sidebar(data: pd.DataFrame) -> dict[str, Any]:
    """Draw the explorer's filters in the sidebar and return them as `filter_deals` keyword arguments."""
    options = filter_options(data)
    min_date = data["publication_date"].min()
    max_date = data["publication_date"].max()
    default_start = min_date.date() if pd.notna(min_date) else date.today()
    default_end = max_date.date() if pd.notna(max_date) else date.today()

    with st.sidebar:
        st.markdown("#### Filter deals")
        query = st.text_input("Search", key="pe_explorer_search",
                              placeholder="e.g., distribution, healthcare, buyer name",
                              help="Matches buyer, target, publisher, headline, industry, geography, "
                                   "transaction type, rationale and evidence text.")
        publishers = st.multiselect("Publisher / PE firm", options["publisher"], key="pe_explorer_publishers")
        transaction_types = st.multiselect("Transaction type", options["transaction_type"], key="pe_explorer_types")
        industries = st.multiselect("Target industry", options["target_industry"], key="pe_explorer_industries")
        geographies = st.multiselect("Geography", options["geography"], key="pe_explorer_geographies")
        rationales = st.multiselect("Acquisition rationale", options["rationale"], key="pe_explorer_rationales")
        date_range = st.date_input("Publication date range", value=(default_start, default_end),
                                   min_value=default_start, max_value=default_end, key="pe_explorer_dates")
        st.button("Clear filters", key="pe_explorer_clear", on_click=_clear_filters, width="stretch")

    start_date, end_date = _date_bounds(date_range)
    # The default full range means "no date filter"; otherwise records with unknown
    # publication dates would disappear even though the visitor did not filter by date.
    if (start_date, end_date) == (default_start, default_end):
        start_date, end_date = None, None
    return dict(search=query, publishers=publishers, start_date=start_date, end_date=end_date,
                transaction_types=transaction_types, industries=industries, geographies=geographies,
                rationales=rationales)


def _active_filter_count(filters: dict[str, Any], data: pd.DataFrame) -> int:
    count = sum(bool(filters[k]) for k in ("search", "publishers", "transaction_types", "industries",
                                           "geographies", "rationales"))
    if filters["start_date"] is not None or filters["end_date"] is not None:
        count += 1
    return count


def render_explorer(dataset: Any | None = None, *, demo_data: bool | None = None) -> None:
    """Render sidebar filters, overview charts, results, detail evidence, and CSV downloads.

    `dataset` is injectable so the team can use a published repository dataset and tests
    can use controlled records. Public page visits never trigger web scraping.
    """
    if dataset is None:
        data, inferred_demo = load_explorer_dataset()
        demo_data = inferred_demo if demo_data is None else demo_data
    else:
        data = load_deals(dataset)
        demo_data = bool(demo_data)
    data = load_deals(data)

    st.subheader("Deal Explorer")
    st.write("Search and filter announced transactions with the filters in the sidebar (tap » on small screens), "
             "inspect the supporting evidence, and export a research shortlist.")
    if demo_data:
        st.warning("Demo dataset: these records are illustrative and must not be treated as verified transactions.")

    if data.empty:
        st.info("No deal records are available yet. Add a published dataset at `data/pe/sample_deals.json` or pass records to `render_explorer(dataset)`. The explorer does not scrape sources when a visitor opens the page.")
        return

    filters = render_filter_sidebar(data)
    filtered = filter_deals(data, **filters)
    active = _active_filter_count(filters, data)
    st.caption(f"Showing **{len(filtered):,}** of {len(data):,} unique deals"
               + (f" · {active} filter{'s' if active != 1 else ''} active" if active else " · no filters applied"))
    st.sidebar.caption(f"{len(data):,} deals in the dataset" + (" (demo records)" if demo_data else "") + ".")

    metrics = summary_metrics(filtered)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching unique deals", f"{metrics['unique_deals']:,}")
    k2.metric("Disclosed transaction values", f"{metrics['disclosed_values']:,}")
    k3.metric("Records needing review", f"{metrics['needs_review']:,}")
    k4.metric("Records with evidence", f"{metrics['with_evidence']:,}")

    if filtered.empty:
        st.info("No deals match those filters. Try removing a filter or broadening the date range.")
        return

    st.markdown("#### Deal activity in the selected results")
    left, right = st.columns(2)
    with left:
        trend = deals_over_time(filtered)
        st.markdown("**Deals by publication quarter**")
        if trend.empty:
            st.caption("No publication dates are available for the selected records.")
        else:
            st.bar_chart(trend.set_index("period")["deal_count"], width="stretch")
    with right:
        industry_counts = count_by_category(filtered, "target_industry").head(10)
        st.markdown("**Target industry mix**")
        if industry_counts.empty:
            st.caption("No industry labels are available.")
        else:
            st.bar_chart(industry_counts.set_index("category")["deal_count"], width="stretch")
    st.caption("Counts use unique deal IDs in the selected dataset. Coverage reflects only the sources and dates collected; it is not a complete measure of the PE market.")

    # A per-session shortlist supports the presentation workflow without storing visitor choices globally.
    # Candidate deals respect the active filters, then narrow to the selected rationale tags.
    st.markdown("#### Rationale-based research shortlist")
    st.write("Find transactions associated with a stated or classified value-creation rationale. These are research leads, not evidence of realized returns.")
    rationale_options = filter_options(data)["rationale"]
    selected_rationales = st.multiselect(
        "Rationales to investigate", rationale_options,
        key=SHORTLIST_RATIONALE_KEY,
        help="Choose one or more available rationale categories, such as product, distribution, technology or geographic expansion.",
    )
    shortlist_candidates = filter_deals(filtered, rationales=selected_rationales) if selected_rationales else filtered.iloc[0:0].copy()
    all_ids = set(data["deal_id"].astype(str))
    saved_ids = [str(value) for value in st.session_state.get(SHORTLIST_STATE_KEY, []) if str(value) in all_ids]
    st.session_state[SHORTLIST_STATE_KEY] = saved_ids

    if selected_rationales:
        st.caption(f"{len(shortlist_candidates):,} candidate deal(s) match the selected rationale(s) and current filters.")
        if shortlist_candidates.empty:
            st.info("No deals match these rationales and filters. Try another rationale or broaden the filters.")
        else:
            candidate_view = shortlist_candidates[["deal_id", "buyer_or_investor", "target_company", "rationale_tags", "evidence_excerpt", "review_status"]].copy()
            candidate_view["rationale_tags"] = candidate_view["rationale_tags"].map(lambda x: ", ".join(x) if isinstance(x, list) and x else "Unknown")
            candidate_view = candidate_view.rename(columns={
                "deal_id": "Deal ID", "buyer_or_investor": "Buyer / investor", "target_company": "Target",
                "rationale_tags": "Rationale", "evidence_excerpt": "Supporting excerpt", "review_status": "Review status",
            })
            st.dataframe(candidate_view, width="stretch", hide_index=True)
            candidate_ids = shortlist_candidates["deal_id"].astype(str).tolist()
            already_selected = [deal_id for deal_id in saved_ids if deal_id in candidate_ids]
            to_add = st.multiselect(
                "Choose candidate deals to add", candidate_ids, default=already_selected,
                format_func=lambda deal_id: _deal_label(data, deal_id), key="pe_explorer_shortlist_add_selection",
            )
            if st.button("Add selected deals to my session shortlist", key="pe_explorer_shortlist_add"):
                st.session_state[SHORTLIST_STATE_KEY] = list(dict.fromkeys(saved_ids + [str(x) for x in to_add]))
                st.success(f"Shortlist now contains {len(st.session_state[SHORTLIST_STATE_KEY])} unique deal(s).")
                st.rerun()
    else:
        st.caption("Select at least one rationale to find candidate deals for your shortlist.")

    saved_frame = data[data["deal_id"].astype(str).isin(saved_ids)].copy()
    st.markdown("##### Your session shortlist")
    st.caption("This shortlist is stored only in your current Streamlit session; it is not shared with other visitors or saved as a global list.")
    if saved_frame.empty:
        st.info("Your shortlist is empty. Select rationale categories above and add deals to get started.")
    else:
        shortlist_view = saved_frame[["deal_id", "buyer_or_investor", "target_company", "publication_date", "rationale_tags", "evidence_excerpt", "review_status"]].copy()
        shortlist_view["publication_date"] = shortlist_view["publication_date"].map(_fmt_date)
        shortlist_view["rationale_tags"] = shortlist_view["rationale_tags"].map(lambda x: ", ".join(x) if isinstance(x, list) and x else "Unknown")
        shortlist_view = shortlist_view.rename(columns={
            "deal_id": "Deal ID", "buyer_or_investor": "Buyer / investor", "target_company": "Target",
            "publication_date": "Publication date", "rationale_tags": "Rationale",
            "evidence_excerpt": "Supporting excerpt", "review_status": "Review status",
        })
        st.dataframe(shortlist_view, width="stretch", hide_index=True)
        remove_ids = st.multiselect(
            "Remove deals from my shortlist", saved_ids,
            format_func=lambda deal_id: _deal_label(data, deal_id), key="pe_explorer_shortlist_remove_selection",
        )
        shortlist_actions = st.columns([1, 1, 3])
        with shortlist_actions[0]:
            if st.button("Remove selected", key="pe_explorer_shortlist_remove"):
                remove_set = {str(x) for x in remove_ids}
                st.session_state[SHORTLIST_STATE_KEY] = [deal_id for deal_id in saved_ids if deal_id not in remove_set]
                st.rerun()
        with shortlist_actions[1]:
            st.download_button(
                "Export shortlist CSV", data=export_filtered(saved_frame),
                file_name="lighthouse_pe_research_shortlist.csv", mime="text/csv",
                key="pe_explorer_shortlist_download",
            )
        with shortlist_actions[2]:
            st.caption("Includes evidence excerpts, source URLs, review status, unknown fields and the data dictionary field note.")

    st.markdown("#### Matching deals")
    display_cols = ["publication_date", "publisher_pe_firm", "buyer_or_investor", "target_company", "transaction_type", "target_industry", "target_geography", "rationale_tags", "price", "currency", "review_status"]
    table = filtered[display_cols].copy()
    table["publication_date"] = table["publication_date"].map(_fmt_date)
    table["rationale_tags"] = table["rationale_tags"].map(lambda x: ", ".join(x) if isinstance(x, list) and x else "Unknown")
    table["price"] = table["price"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "Unknown")
    table = table.rename(columns={
        "publication_date": "Publication date", "publisher_pe_firm": "Publisher / PE firm",
        "buyer_or_investor": "Buyer / investor", "target_company": "Target", "transaction_type": "Type",
        "target_industry": "Industry", "target_geography": "Geography", "rationale_tags": "Rationale",
        "price": "Disclosed value", "currency": "Currency", "review_status": "Review status",
    })
    st.dataframe(table, width="stretch", hide_index=True)

    # Use stable deal_id values as the selection key, and keep the selected deal in session state.
    deal_ids = filtered["deal_id"].astype(str).tolist()
    previous = st.session_state.get("pe_explorer_selected_deal")
    index = deal_ids.index(previous) if previous in deal_ids else 0
    selected_id = st.selectbox(
        "Inspect a deal", deal_ids, index=index,
        format_func=lambda deal_id: _deal_label(filtered, deal_id),
        key="pe_explorer_selected_deal",
    )
    selected = filtered.loc[filtered["deal_id"].astype(str).eq(str(selected_id))].iloc[0]
    # Stable handoff contract for Yuepheng's Deal Detail page; filter widgets stay in session state.
    st.session_state["pe_selected_deal_id"] = str(selected_id)
    with st.container(border=True):
        st.markdown(f"### {_display(selected['buyer_or_investor'], 'Unknown buyer')} → {_display(selected['target_company'], 'Unknown target')}")
        st.caption(f"Deal ID: {selected['deal_id']} · Published {_fmt_date(selected['publication_date'])} · Status: {_display(selected['transaction_status'])}")
        detail_cols = st.columns(3)
        detail_cols[0].write(f"**Type**\n\n{_display(selected['transaction_type'])}")
        detail_cols[1].write(f"**Target industry**\n\n{_display(selected['target_industry'])}")
        detail_cols[2].write(f"**Geography**\n\n{_display(selected['target_geography'])}")
        st.markdown("**Stated / classified rationale**")
        tags = selected["rationale_tags"] if isinstance(selected["rationale_tags"], list) else []
        st.write(", ".join(tags) if tags else "Unknown")
        st.markdown("**Supporting evidence**")
        excerpt = selected["evidence_excerpt"]
        st.write(f"> {excerpt}" if pd.notna(excerpt) and str(excerpt).strip() else "No supporting excerpt has been saved for this record.")
        if pd.notna(selected["is_inferred"]) and bool(selected["is_inferred"]):
            st.warning("At least one classification is inferred rather than explicitly stated.")
        st.write(f"**Review status:** {_display(selected['review_status'])}")
        st.write(f"**Disclosed transaction value:** {selected['price']:,.0f} {_display(selected['currency'], '')}" if pd.notna(selected["price"]) else "**Disclosed transaction value:** Unknown / not disclosed")
        urls = selected["source_url"] if isinstance(selected["source_url"], list) else []
        st.markdown("**Source announcements**")
        if urls:
            for url in urls:
                st.markdown(f"- [{url}]({url})")
        else:
            st.write("No source URL has been saved.")

    st.markdown("#### Export this research set")
    export_cols = st.columns([1, 1, 2])
    with export_cols[0]:
        st.download_button("Download filtered deals (CSV)", data=export_filtered(data, filters=filters), file_name="lighthouse_pe_deals.csv", mime="text/csv", key="pe_explorer_download_csv")
    with export_cols[1]:
        st.download_button("Download data dictionary", data=data_dictionary_csv(), file_name="lighthouse_pe_data_dictionary.csv", mime="text/csv", key="pe_explorer_download_dictionary")
    with export_cols[2]:
        st.caption("Exports contain the same filtered, deduplicated records used by the table and charts. Missing values stay blank/unknown, not zero.")


def _deal_label(df: pd.DataFrame, deal_id: str) -> str:
    row = df.loc[df["deal_id"].astype(str).eq(str(deal_id))].iloc[0]
    buyer = _display(row["buyer_or_investor"], "Unknown buyer")
    target = _display(row["target_company"], "Unknown target")
    return f"{buyer} → {target} ({deal_id})"
