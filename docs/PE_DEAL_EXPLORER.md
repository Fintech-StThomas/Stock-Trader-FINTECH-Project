# Lighthouse PE Deal Explorer

The PE Deal Explorer is a Streamlit page under the existing Lighthouse workspace switch. In the PE workspace, the sidebar is the navigation: it switches between Deal Explorer, Sources and Coverage, and Quality and Method (linkable with `?workspace=pe&pe_page=explorer|sources|quality`), and on the Deal Explorer it also holds the search box, filters and a Clear filters button. It is designed to work with a saved dataset and does not scrape websites or call an AI model when a visitor opens the page.

## Current functionality

- Sidebar search across buyer, target, publisher, headline, industry, geography, transaction type, and evidence text.
- Sidebar filters by publisher/PE firm, publication date, transaction type, target industry, geography, and rationale tags.
- Show unique-deal counts, disclosed-value counts, records needing review, and records with evidence.
- Chart unique deals by publication quarter and target-industry mix.
- Inspect a selected deal's rationale, excerpt, source links, review status, and disclosed transaction value.
- Export the filtered records and a field data dictionary to CSV.
- Collapse repeated rows with the same `deal_id`, while preserving all announcement IDs and source URLs.
- Keep missing transaction values blank/unknown rather than converting them to zero.

## Dataset contract

The initial loader reads `data/pe/sample_deals.json`. The sample records are intentionally fictional and marked `data_status: demo`; they are interface fixtures, not real transactions. Replace them with the team's reviewed/published dataset once the collection and extraction modules are ready.

The expected deal fields include:

- `deal_id` (stable unique transaction ID)
- `announcement_ids` (list of source announcement IDs)
- `publisher_pe_firm`, `source_url`, `headline`, `publication_date`
- `buyer_or_investor`, `target_company`, `transaction_type`, `transaction_status`
- `target_industry`, `target_geography`, `rationale_tags`
- `price`, `currency`, `evidence_excerpt`, `review_status`, `is_inferred`

Use ISO date strings (`YYYY-MM-DD`). `price` should be numeric only when a value is disclosed; otherwise use JSON `null`. The publisher/PE firm must not be assumed to be the direct buyer. Rationale labels should align with the definitions agreed by Cole and Yuepheng.

## Integration

`pe/app.py` draws the PE sidebar navigation and renders `render_explorer()` when Deal Explorer is selected; `render_filter_sidebar()` in `pe/pages/explorer.py` draws the filters into the sidebar. When a visitor switches back to Stock Research, `workspace_ui.py` drops the `pe_page` query parameter. The core operations live in:

- `pe/analytics.py` — normalization, deduplication, filters, metrics, and chart summaries
- `pe/exports.py` — stable CSV output and data dictionary
- `pe/pages/explorer.py` — Streamlit interface
- `data/pe/sample_deals.json` — fictional demonstration records
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

- The included dataset is fictional demonstration data.
- The Deal Detail page is still owned by Yuepheng; this version displays evidence in the explorer until that page is integrated.
- Source monitoring and refresh remain Nikhil's responsibility.
- Authentication and persistent personal workspaces are not implemented.
- Charts describe only the collected dataset, not total private-equity market activity.
