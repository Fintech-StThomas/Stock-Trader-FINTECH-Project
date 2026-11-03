# Lighthouse PE Deal Explorer

The PE Deal Explorer is a Streamlit page under the existing Lighthouse workspace switch. In the PE workspace, the sidebar is the navigation: it switches between Deal Explorer, Sources and Coverage, and Quality and Method (linkable with `?workspace=pe&pe_page=explorer|sources|quality`), and on the Deal Explorer it also holds the search box, filters and a Clear filters button. It is designed to work with a saved dataset and does not scrape websites or call an AI model when a visitor opens the page.

## Current functionality

- Sidebar search across buyer, target, publisher, headline, industry, geography, transaction type, and evidence text.
- Sidebar filters by publisher/PE firm, publication date, transaction type, target industry, geography, and rationale tags.
- Show unique-deal counts, disclosed-value counts, records needing review, and records with evidence.
- Chart unique deals by publication quarter and target-industry mix.
- Inspect a selected deal's rationale, excerpt, source links, review status, and disclosed transaction value.
- Build a rationale-based research shortlist, add/remove deals in the current visitor session, and export the shortlist with evidence and source links.
- Export the filtered records and a field data dictionary to CSV; export filters can be applied through the shared helper so chart/table/download rules stay aligned.
- Store the selected `deal_id` as `st.session_state["pe_selected_deal_id"]` for the future Deal Detail handoff while keeping explorer filters in their existing session-state keys.
- Collapse repeated rows with the same `deal_id`, while preserving all announcement IDs and source URLs.
- Keep missing transaction values blank/unknown rather than converting them to zero; undated deals remain visible unless the visitor explicitly applies a publication-date filter.

## Dataset contract

The explorer reads `data/pe/deals.json`: **115 real deal announcements** (marked `data_status: collected`) published from January 2023 to October 2026 on the news pages of four PE firms: Tonka Bay Equity Partners, Goldner Hawn, Huron Capital and Align Capital Partners. The pages were collected on 2026-10-29.

The records come from the PE deal collector and rule-based extractor on the `claude/dashboard-functionality-yi0loj` branch (`python -m pedeals`, output `data/pe_deals/output/deals.json`, build fingerprint `d5e0683fc0f5e0b2`). On 38 held-out articles, buyer, target, transaction type and price disclosure were 100% correct, target industry 87% and target state 97%. Those reference labels were written by Claude (AI), the same model that wrote the rules, and no person has checked them yet; a later round with independent AI labellers on newer rules scored lower (industry 50%, state 69%). The **Quality and Method** page shows all of this, read from `data/pe/validation.json`. Rebuild the explorer file with:

```bash
python scripts/build_pe_explorer_dataset.py path/to/pedeals/deals.json
```

The conversion keeps short evidence quotes, not article bodies, and keeps unknown values null. `target_geography` is the target's US state, so it is unknown when the announcement names no state. A record is `needs review` when the extractor flagged it for a missing buyer, an ambiguous industry, an inferred relationship or several targets. Date-source flags appear in `review_notes` without changing the status. `is_inferred` is true for an ambiguous industry or an inferred relationship.

The expected deal fields include:

- `deal_id` (stable unique transaction ID)
- `announcement_ids` (list of source announcement IDs)
- `publisher_pe_firm`, `source_url`, `headline`, `publication_date`
- `buyer_or_investor`, `target_company`, `transaction_type`, `transaction_status`
- `target_industry`, `target_geography`, `rationale_tags`
- `price`, `currency`, `evidence_excerpt`, `review_status`, `review_notes`, `is_inferred`

Use ISO date strings (`YYYY-MM-DD`). `price` should be numeric only when a value is disclosed; otherwise use JSON `null`. The publisher/PE firm must not be assumed to be the direct buyer. Rationale labels should align with the definitions agreed by Cole and Yuepheng.

## Integration

`pe/app.py` draws the PE sidebar navigation and renders `render_explorer()` when Deal Explorer is selected; `render_filter_sidebar()` in `pe/pages/explorer.py` draws the filters into the sidebar. When a visitor switches back to Stock Research, `workspace_ui.py` drops the `pe_page` query parameter. The core operations live in:

- `pe/analytics.py` — normalization, deduplication, filters, metrics, and chart summaries
- `pe/exports.py` — stable CSV output and data dictionary
- `pe/pages/explorer.py` — Streamlit interface
- `data/pe/deals.json` — the collected deal dataset
- `scripts/build_pe_explorer_dataset.py` — converts the collector's output into that file
- `tests/test_pe_explorer.py` — logic and export tests
- `tests/test_workspaces.py` — app-level tests for workspace routing, the PE sidebar sections, deep links and sidebar filters

The root `app.py` is unchanged. Jacob can continue to own root navigation and shared data-loading integration.

## Run and test

Install the project dependencies, including Streamlit, then run:

```bash
streamlit run app.py
```

Choose **PE Deal Intelligence** in the workspace switch. The automated logic tests can be run with:

```bash
python -m pytest tests/test_pe_explorer.py tests/test_workspaces.py -q
```

These tests run the real app with Streamlit's `AppTest`, including selecting a sidebar filter and clearing it.

## Known first-version limits

- The dataset covers only four firms and the announcements they chose to publish. None disclosed a price or closing date, so those fields are unknown throughout.
- The data is a saved snapshot. Refreshing it means re-running the collector and the conversion script.
- The Deal Detail page is still owned by Yuepheng; this version displays evidence in the explorer and publishes the selected `deal_id` in session state, but page routing awaits that module and Jacob's integration.
- Source monitoring and refresh remain Nikhil's responsibility.
- Authentication and persistent personal workspaces are not implemented.
- Charts describe only the collected dataset, not total private-equity market activity.
