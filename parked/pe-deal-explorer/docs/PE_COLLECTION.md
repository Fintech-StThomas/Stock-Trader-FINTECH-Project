# PE announcement collection and source coverage

Nikhil's module collects source announcements and preserves evidence for the extraction team. It does not decide whether an announcement describes a deal, identify a buyer, or feed unvalidated news into the Deal Explorer. The public website reads a saved coverage file; page visits never scrape publishers or call a model.

## Current local result

The implementation supports multiple independently configured publishers. Riverside is the working pilot. Two live runs collected the same six unique announcements, dated September 23 through October 27, 2026, with stable announcement IDs and no failed pages. Both runs requested January 1 through November 3, 2026, but were deliberately capped at six articles. This is a bounded sample, not complete coverage of that period.

Advent and Bain are candidate entries, not active monitored sources. Advent's HTML layout has been inspected but its content-use conditions need resolution before activation. Bain explicitly restricts automated collection without permission; its adapter is not validated. This configuration does not imply that either firm has been collected. Teams can add other firms rather than use these candidates.

## Owned files and integration

- `pe/collectors/config.py`: validates source configuration, periods, and bounds.
- `pe/sources.json`: publisher-specific paths, selectors, limits, and access-review notes.
- `pe/collectors/http.py`: bounded transport, robots checks, redirects, delays, and retries.
- `pe/collectors/parsing.py`: listing discovery and full-article text cleaning.
- `pe/collectors/pipeline.py`: stable records, article deduplication, run reports.
- `pe/storage.py`: private run archives, last-successful pointers, public coverage export.
- `pe/pages/sources.py`: read-only Sources and Coverage screen.
- `tests/test_pe_collection.py`: collection, transport, storage, and page acceptance checks.

The only shell edit is replacing the existing Sources placeholder in `pe/app.py` with `render_sources()`. Root navigation, dependency declarations, Cole's explorer, and stock-analysis logic remain unchanged. This storage module is not a replacement for Jacob's future deal repository or shared contracts.

The collection module integrates with the existing PE workspace on main. The hosted application reads the committed coverage snapshot; live collection remains a team-run command.

## Team refresh command

Use the repository's normal Python environment and existing requirements. No added dependency or runtime model key is needed.

```bash
python -m pe.collectors --source riverside --start 2026-01-01 --end 2026-11-03 --max-articles 6
```

Omit `--source` to run all enabled firms. Repeat it to select multiple enabled sources:

```bash
python -m pe.collectors --source firm_one --source firm_two
```

Those names are placeholders: add and validate their entries first. The source date range defaults to its explicit configuration. The command does not silently advance the end date. Supported controls include `--config`, `--private-dir`, `--coverage-output`, `--max-pages`, and `--max-articles`. Read `python -m pe.collectors --help` for the full interface.

```bash
LIGHTHOUSE_OFFLINE=1 python -m pe.collectors --source riverside --cached
```

Cached mode reads the last successful run with no network requests or writes. Original collection timestamps are preserved and returned records have `is_cached=true`. Cached mode cannot be combined with date or collection-limit overrides: it describes the period actually saved. A fresh clone has no private real-article archive, so it needs an approved live refresh or the team's privately shared archive. Synthetic inputs are available at `data/pe/sample_announcements.json` for initial offline development.

Exit status 0 means the selected sources produced successful records or successful cached replay. Status 1 indicates at least one failed, partial, empty, or unavailable cached result. A successful bounded pilot can still be incomplete: inspect `truncated`, `stop_reasons`, and the requested/observed periods rather than relying on the exit status alone.

## Handoff to Yuepheng

```python
from pe.collectors import collect, load_sources
from pe.storage import CollectionStore

config = next(s for s in load_sources() if s.source_id == 'riverside')
announcements, collection_report = collect(config, cached=True)
# extract(announcements) belongs to Yuepheng's module.
```

Direct live callers must serialize writes using `with CollectionStore().lock():`; the CLI does that automatically. Pass the same store instance to `collect` if overriding its directory.

Real full text and original HTML live only under ignored `data/pe/private/`. Use `riverside/index.json` to locate `last_good_run`, then open `riverside/runs/<run_id>/announcements.json` and `manifest.json`. `raw_path` is relative to the private root. `raw_inputs` maps requested/final page URLs to saved HTML. Preserve raw inputs and notices when sharing the archive privately with the team. Do not commit that archive to the public repository.

### Announcement fields

| Field | Meaning |
| --- | --- |
| `announcement_id` | Stable source-and-URL ID, reused for known identical-content aliases. |
| `source_id`, `publisher_pe_firm` | Publishing source, not an inferred buyer. |
| `source_url`, `source_urls` | Primary provenance URL and any identical-content aliases retained. |
| `headline` | Publisher headline from its article page. |
| `publication_date` | ISO publication date from the dedicated date element; null if absent/unreadable. Never substituted with a closing date or collection date. |
| `collected_at` | UTC time of the actual successful article fetch. |
| `full_text` | Clean article body for internal extraction; no menus, scripts, or related-story cards. |
| `content_hash` | SHA-256 of whitespace-normalized body text. |
| `collection_status` | `success` for saved article records; failed pages are reported separately. |
| `is_cached` | Whether this returned record was loaded from saved storage. |
| `date_status` | `known` or `unknown`. |
| `within_requested_period` | true for confirmed in-window dates; null when unverified. Known out-of-window articles are logged and excluded. |
| `raw_path` | Internal archived HTML location; never included in public coverage. |

Synthetic fixture records add `is_synthetic=true`. Never count those as collected or validated real announcements.

## Handoff to Cole and Jacob

`data/pe/coverage.json` contains only public-safe collection metadata. It lists enabled and candidate firms, last successful version/time, requested and observed periods, counts, limits, latest attempted run, errors, and source links. It intentionally excludes full text, raw archive paths, and raw HTML.

The source page's announcement count is not the explorer's unique-deal count. Yuepheng must classify and group the inputs before Cole uses them. The existing explorer still uses its separate demonstration dataset. No claimed extraction accuracy is created by this collector.

`render_sources(coverage=None)` accepts an injected coverage dictionary for Jacob's eventual repository integration. Without one, it reads the published JSON. Missing coverage shows configuration and honest empty states; unreadable JSON shows a helpful error. There is no public refresh control.

## Collection behavior and limits

Each firm has its own scope and storage. URL normalization removes fragments and common tracking parameters; repeated listing links are fetched once. Identical clean text is grouped within a publisher while retaining source aliases and a duplicate log. This is article deduplication, not transaction deduplication. Two firms can announce one transaction and must remain separate inputs for Yuepheng.

HTML pagination follows configured next-page links with cycle and page caps. Riverside's current page embeds the visible listing in `rcCurrentsData.push` objects; the adapter reads URL strings without executing JavaScript, then fetches individual full articles. It does not scrape contact data from scripts or infer details from listing summaries. Its embedded listing is bounded by what the publisher supplies; it is not an all-history feed.

HTTP uses Lighthouse's existing per-host rate limiter and retry status policy, with a PE-specific identity and configuration. It does not reuse SEC email/header behavior. Every request and redirect must remain on the configured HTTPS source and approved path. Robots rules are checked before collection; unreachable or unexpected robots responses fail closed. Crawl-delay/request-rate requirements increase the configured delay. Timeouts, retries, redirects, page counts, article counts, and input sizes are bounded. HTTP 403 is not bypassed. An unusually long retry request stops the refresh instead of retrying early.

A missing date stays null and is counted separately. A missing article body or a listing with no recognized links is treated as a layout failure, not silently interpreted as success. Articles outside the date window may still be fetched to verify their publication dates; fetching stops at the configured cap, so a narrow older window may require a larger cap or a publisher-specific archive listing.

Every run gets its own dated archive and manifest. Successful nonempty runs atomically update the last-good pointer. Failed, partial, and empty attempts are saved for diagnosis but cannot replace it. Bounded successful samples may be published with explicit limited-coverage labels. A lock prevents overlapping CLI runs. If a process is killed, check that it has stopped before removing a stale `.refresh.lock`.

For hosting, deploy the reviewed coverage JSON with the app. Keep private inputs in team-controlled durable storage. A hosted temporary filesystem is not a permanent archive. Full-text publication rights are separate from collection access; this module deliberately publishes no article body.

## Add another firm

1. Choose a public newsroom and review its access conditions, robots rules, and content-sharing limits. Record the result and terms URL. Leave unresolved sources disabled.
2. Add a unique source ID, publisher name, listing URL, permitted article prefix, body/title/date XPath, and next-page XPath to `pe/sources.json`.
3. Keep precise article selectors: do not use the entire page as a fallback. A changed layout should fail visibly.
4. Add synthetic layout fixtures and tests for its links, dates, body boundaries, and pagination. Add a dedicated adapter if the source is not ordinary HTML; do not guess unsupported APIs or bypass access controls.
5. Enable only after review, set an explicit period and conservative limits, run a small live sample, and manually inspect the saved text and provenance.
6. Run twice and compare stable IDs, then replay offline. Check the public coverage file and the page before expanding coverage.

## Validation and AI assistance

Run `python -m pytest tests/test_pe_collection.py tests/test_pe_explorer.py tests/test_workspaces.py -q` for collection acceptance, existing explorer, and stock/PE navigation checks. Network behavior is tested using controlled responses; tests never need publisher access. Live verification is a separate small pilot, not an automated recurring test.

Coding assistance helped inspect the current code, implement parsing/storage/coverage, and create failure-case tests. Human/team decisions still include final source selection, collection period, permission/use scope, extraction labels, reference-set review, and release approval. No language model decides fields in this collector, and no paid model runs on public page visits.
