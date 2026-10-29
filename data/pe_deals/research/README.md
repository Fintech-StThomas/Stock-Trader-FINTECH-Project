# PE firms by state (research, October 2026)

`pe_firms_by_state.csv`: 2-3 private equity firms headquartered in each state and DC, gathered by web research
(firm websites first, then press releases and directories) on 2026-10-29. Per state: up to two of the largest firms
by assets under management (`why_listed = largest`) and one lower-middle-market buy-and-build firm active in the
target sectors (`lmm_sector_fit`); `in_tool` marks the four firms the PE deals tab already reads.

How far to trust it:
- `confidence` is the researcher's own rating (high: firm website and a dated source agree; low: directory only).
- AUM figures mix assets under management, committed capital and capital raised; `aum_source_year` says which.
  Treat them as size bands, not rankings.
- `news_page_http_status` is a live check of the listed news page on the date in `news_page_checked_utc` (200 =
  reachable); empty means no news page was found. It does not check robots.txt or whether the page lists deals.
- Five jurisdictions have no PE firm headquartered there that could be verified (row `none found`).
- Before a firm becomes a source for the tool it needs the same vetting as the current four: robots.txt, a sitemap
  or listing of its news, and a labelled check of its pages.
