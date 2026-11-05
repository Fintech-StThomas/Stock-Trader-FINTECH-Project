# Downloaded PE code integration

The November 4 downloaded project was merged into the current local Lighthouse implementation by selecting new modules and resolving overlapping functions. The AI research interface and existing 115 collected plus four curated deals are preserved. Older filter screens and fictional demo records do not replace the current public experience.

## Features added

- Shared Announcement, Evidence and CollectionReport contracts.
- Offline rule-based extraction with candidate deal, non-deal, uncertain and context queues.
- Field-level evidence entries and extractor version tracking.
- A separate Deal Detail page, reachable directly through navigation with a saved-deal picker, or from an AI result's Inspect saved deal button.
- A versioned dataset repository shared by AI research, deal detail, the saved explorer loader and source inventory.
- Publishing command that rejects unreviewed records by default and can preserve the extraction evidence bundle.
- Immutable publication versions, previous-version fallback and preservation of the real bundled dataset when no publication exists.
- Validation command that compares predicted records with independently supplied reference records.
- Ten configured research publishers kept separately from actual collection coverage. Sources and Coverage and Quality and Method are removed from public navigation; the underlying collection and validation tools remain available to the team.

Newspaper source names are kept separate from publishing PE firms. Market and legal context are not counted as transactions. Duplicate rows are reported separately from transaction-group accuracy, which the imported evaluator does not measure. This merge does not create new accuracy measurements or human-reviewed labels.

## Collection modes

The original firm collector is preserved:

```
python -m pe.collectors --source riverside --cached
```

The new research-publisher collector is explicit:

```
python -m pe.collectors --research-sources --cached
```

Research configuration is in `data/pe/source_config.json`. The incoming code marks ten sources enabled, but provides no completed access review. Live collection now requires `access_reviewed: true` after the team checks access conditions and source layout. Runtime robots checks, host-scoped redirects, timeouts, delays and page-size controls use the existing polite HTTP transport. Offline mode cannot invoke the real HTTP client. The collector reads ordinary HTML and same-host article links; it is not a validated parser for all ten sites or a PDF reader.

Up to 20 candidate article links per source are configured, plus the seed pages. Landing pages are not treated as announcements. The source layout/path filters require live validation before claiming coverage. The supplied default `delay_seconds` note is enforced as two seconds per source, with longer robots requirements honored. Open Access path restrictions are enforced for Alternatives Watch. Nothing in the merged app triggers collection on a page visit.

The new collector's raw inputs and output files live under ignored `data/pe/private/research/`. Explicit collector configuration approval is separate from permission to republish article text.

## Configured sources and purposes

| Source | Role |
| --- | --- |
| Private Equity Wire | Deal-news reporting |
| Private Equity Insights | Deal-news reporting |
| ACG Insights / Middle Market Growth | Middle-market deal-news reporting |
| FinSMEs Private Equity | Deal-news reporting |
| Bain Global Private Equity Report | Market context, not transactions; Bain & Company, not Bain Capital |
| McKinsey Private Capital Insights | Market context, not transactions |
| Blackstone News | Primary firm announcements; other business lines still need classification |
| Carlyle News and Media | Primary firm announcements; other business lines still need classification |
| National Law Review | Legal/regulatory context, not transactions |
| Alternatives Watch Open Access | Open-access deal-news reporting only |

These are configured starting pages, not ten proven feeds or a list of already fetched articles.

## Extract and publish

The extractor can process either collector's saved Announcement records:

```
python -m pe.extraction --input path/to/announcements.json --output data/pe/private/research/extraction_output.json
```

The rules are a conservative baseline. They leave industry, geography, price and other unsupported values unknown; keyword and party parsing require review. Transaction status is a provisional rule output, not verified closing evidence. Context-only announcements stay outside deal counts. Source text is untrusted input, never instructions.

Review records, resolve duplicate groups and preserve evidence before publishing:

```
python -m pe.publish path/to/reviewed_extraction_bundle.json --version reviewed-v1
```

The bundle may contain both `deals` and `evidence`. Publication is refused unless all records are marked reviewed/approved/validated, or `--allow-unreviewed` is explicitly used for a labelled demo. Publishing a label is not proof that an independent human actually reviewed it. Publishing here is a local data operation, not deployment to Streamlit or GitHub.

The current data and historical validation report remain intact. New validation requires a separate independently checked reference set. `python -m pe.validation --help` shows the evaluation interface.

## Merge verification

The six existing Riverside cached announcements were processed through the imported extractor without network requests. They produced six candidate records, nine evidence entries and five review-queue items. All six remain unreviewed and were not published over the current 119 records. This exercise establishes pipeline compatibility, not extraction accuracy.

The synthetic downloaded sample deal data is retained only as a test fixture at `tests/fixtures/pe/demo_deals.json`; no fictional record is mixed into the real dataset. The downloaded release checklist claimed an empty source configuration even though ten entries were present, so it was replaced by these current notes rather than adopted as completed release evidence.

Run the full suite with `python -m pytest -q`. API behavior is tested with mocked responses; this merge does not send paid OpenAI requests, read or copy credentials, or revoke or replace keys.
