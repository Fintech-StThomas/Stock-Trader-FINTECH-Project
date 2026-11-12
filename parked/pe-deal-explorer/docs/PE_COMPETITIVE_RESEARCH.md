# Competitive research workflow

The primary user is a PE investment analyst monitoring competing firms. The explorer supports selecting firms, filtering transactions by publication date, sector, geography and rationale, reading evidence and source announcements, saving a session shortlist, and exporting the selected records.

The competitive research summary download adds a Markdown report of the current filtered set: counts by firm, sector, geography and rationale, known publication dates, unknown-date counts, review notes and source links. It uses the same filtered unique records as the explorer's table and CSV. It does not infer investment success from announcement activity.

## Published inputs

The team's existing 115 records from four firms remain unchanged at `data/pe/deals.json`. Four curated samples are stored separately at `data/pe/curated_deals.json` and combined on read through `pe/published.py`. Keeping the additions separate prevents the extractor's next rebuild from deleting them and preserves the original evaluation dataset. Duplicate IDs or source URLs are not appended twice.

Advent samples: the announced Vermeg acquisition and announced majority investment in NZCR Group. Bain samples: the Kahua minority growth investment and the Lake Como hotel exit. These are source-linked research samples, not scraper output. They were checked against their official source pages on November 4, 2026 using AI-assisted reading and still require independent team review. Classification categories are marked inferred. Full article bodies are not stored or republished.

All four samples remain `needs review`. Unknown terms remain null. Kahua's reported company valuation is not treated as the transaction value. In the hotel exit, Bain is a seller and Sansiri Capital is the buyer. Vermeg's London dateline is not treated as the target's headquarters.

Bain's pages have a shared featured-story header whose date differs from the article-body datelines. The two publication dates therefore remain null pending team review. They appear with the default full dataset but are excluded when a user explicitly sets a publication-date range. Notes retain the observed datelines so a reviewer can resolve the ambiguity. An expected closing date never establishes that a transaction has completed.

## Collection and coverage

Sources and Coverage now starts with an inventory of the six firms represented in the combined explorer dataset. It shows published record counts, curated counts and observed known dates. The separate collection section still reports actual Riverside run metadata. Riverside's six collected raw articles are not silently promoted to extracted deal records.

Advent and Bain cards link directly to the new source samples and explain that automated refresh is disabled. No permission is assumed and no collection run is fabricated. The curated records do not increment live-collection statistics. Enabling ongoing automated collection remains a separate source-access and adapter-validation task.

## Quality evaluation

The historical accuracy figures still apply only to the original extraction build. Quality and Method identifies the curated additions as outside those figures, and its existing dataset-size warning remains visible. Unknown and review-status summaries use the combined current data. These four additions do not create new measured accuracy claims.

## Checks

`python -m pytest tests/test_pe_competitive.py tests/test_pe_collection.py tests/test_pe_explorer.py tests/test_pe_quality.py tests/test_workspaces.py -q`

Checks cover the complete firm/date/sector/geography/rationale filtering path, evidence and CSV/summary consistency, source inventory totals, duplicate supplements, unknown dates, buyer-versus-seller interpretation, unchanged original data, and current app navigation.
