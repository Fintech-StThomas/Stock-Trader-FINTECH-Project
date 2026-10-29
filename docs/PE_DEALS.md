# PE Deal Intelligence: method, data dictionary and limitations

A separate product inside the Lighthouse dashboard (tab **🤝 PE deals**, package `pedeals/`, notebook
`notebooks/PE_Deal_Intelligence.ipynb`). It reads the deal announcements that private-equity firms publish on their own
websites and turns each one into a record a professional can filter, compare and cite, with every important field next
to the sentence it came from.

## 1. User, problem and decision

**User.** A portfolio operations manager (or the deal team) at a lower-middle-market sponsor that buys and builds
industrial, distribution and business-services companies.

**Recurring task.** Before each add-on planning cycle the user asks: *which add-ons are peer sponsors' platforms buying,
which capabilities do they say those add-ons bring (products, geography, customers, technology, capacity, people), and
how fast do the best builders close deals?* Today that means reading dozens of press releases by hand on several
sites, with no common record and no links back to the evidence.

**What the tool gives.** A searchable deal record (deal finder), a buy-and-build tracker (add-ons per platform and the
gap between them), an add-on research brief for the user's own industry (the decision scenario, §7), a "new since"
digest for monitoring, and the validation evidence that says how far each field can be trusted.

**What it does not claim.** A rationale in a press release is the sponsor's claim. Whether an add-on raised margins,
growth or returns (*operational alpha*) needs operating results the announcements do not contain, so the tool treats
every stated rationale as a hypothesis to investigate. The firms choose what to announce, so counts describe the
collected announcements, not the PE market.

## 2. Scope: sources, period, amount

| Firm | Home office | Focus (from its own site) | How its news is found |
|---|---|---|---|
| Tonka Bay Equity Partners | Minnetonka, MN | engineered manufacturing, business services, value-added distribution | news sitemap |
| Goldner Hawn | Minneapolis, MN | specialty manufacturing, value-added distribution, business and consumer services | news sitemap + news page (dates) |
| Huron Capital | Detroit, MI | infrastructure, facility and residential services (buy-and-build "ExecFactor") | news sitemap + news page (dates) |
| Align Capital Partners | Dallas, TX / Cleveland, OH | business services, technology, specialty manufacturing and distribution | post sitemap |

*Why these four.* They are peers of the user: lower-middle-market, Midwest and central US, explicit buy-and-build
strategies, and they publish their add-ons (the brief's example firm, Tonka Bay, is one of them). Each lists its
articles in an XML sitemap, so collection finds every article without crawling menus. Firms whose robots.txt blocks
automated or AI agents by name were left out even where a generic rule would allow them; other candidates were not
reachable or had no usable news sitemap.

*Period.* Announcements **published from 1 January 2023** to the collection date (about four years: enough for
platforms to show several add-ons, recent enough to matter for current planning). The start date is a parameter.

*Amount.* The sitemaps listed 518 URLs; 185 pages were fetched (the rest were last modified before the period, so they
were published before it); 148 news items fall in the period, of which **115 are deal announcements** (80 add-ons, 17
platform investments, 2 growth investments, 2 combinations, 14 exits). Per-firm counts are in the coverage table
(Data quality tab, `data/pe_deals/output/coverage.csv`).

## 3. Collection (`pedeals/collect.py`, `pedeals/fetch.py`)

* **Polite by construction.** robots.txt is read for each host and obeyed; its `Crawl-delay` (10 seconds on Tonka Bay
  and Goldner Hawn) is the minimum gap between requests, 5 seconds elsewhere; one thread per host; the User-Agent names
  the tool and links to this repository. Huron's robots.txt disallows its paginated archive, so it is not read.
* **What is kept.** Every fetched page is saved with its URL, final URL, HTTP status and UTC fetch time
  (`data/pe_deals/cache/pages.jsonl.gz`, scripts and styles stripped, text untouched); every URL the sitemaps offered is
  logged with its outcome (`discovery.csv`). Failed pages are recorded, never silently dropped, and never overwrite a
  good saved copy.
* **Cached data.** Parsing and extraction always run on the saved pages, so a run is reproducible and still works when
  a site changes or goes offline. The app and every export say **CACHED DATA** and show the collection dates. `python -m
  pedeals collect` (or the refresh button in the tab) adds new articles.
* **Period filter.** A sitemap `<lastmod>` before the start date proves the page was published before it, so those
  pages are skipped; all others are fetched and filtered by the publication date read from the page.

## 4. Processing and extraction (`pedeals/parse.py`, `pedeals/extract.py`)

1. **Parsing.** The article body is the page container holding at least 90% of the paragraph text once navigation,
   headers and footers are removed. "About <company>" paragraphs are separated as boilerplate, so the sponsor's own
   sector focus is never read as the target's industry.
2. **Publication date**, in order of trust: page metadata (`article:published_time`), the release's dateline
   ("MINNETONKA, MN (July 7, 2026) —"), the date printed next to the link on the firm's news page, the first date in the
   opening text, and finally a month-only date from a news page (stored as the 1st, flagged *date known to the month
   only*). No date found → left empty and the item is kept out of the period (14 items), never guessed. The
   publication date is never used as a closing date.
3. **Deal or not.** Headline first (non-deal signals such as *promoted, named to, podcast, fund close, job posting*
   decide unless a clear deal verb is present), then the opening text. A firm milestone ("Closes 100th Acquisition"),
   a fund close or a continuation vehicle is *Fund / firm* news, not a deal.
4. **Parties** from the release's lead sentence, split at its deal verb: the buyer is the company the release calls
   "the Company" or the last company name before the verb; the target is the first name after it; parenthesised short
   names become aliases. Headline patterns are the fallback. The publisher under any of its names (Align Capital
   Partners, ACP) is the publisher; a buyer introduced as the sponsor's portfolio / partner / platform company, or one
   that the publisher's *other* announcements show it owned at that date, is a portfolio company (the latter flagged
   *relationship inferred from another announcement*).
5. **Type, status, disclosure, rationale, target business**: see the data dictionary below.
6. **Duplicates.** URLs are normalised and stored once; the same headline twice from one firm, or the same target
   announced by two firms within 120 days, is marked `duplicate_of` (kept, not deleted); analysis uses unique deals.

Rules were chosen over a language model so that every value can be traced to a pattern and a sentence, the run is free
and deterministic, and errors can be investigated; the price is lower recall on unusual wording, which the validation
measures.

## 5. Data dictionary (`data/pe_deals/output/deals.csv`, one row per deal announcement)

| Field | Meaning | Values / notes |
|---|---|---|
| `deal_id` | stable id (hash of the URL) | |
| `source`, `publisher` | the firm whose website published the announcement | the publisher is **not necessarily the buyer** |
| `url`, `collected_at` | the evidence page and when it was fetched (UTC) | |
| `headline` | the article title | |
| `published_date` | date the firm published the announcement | ISO date; **not** the closing date |
| `published_method`, `published_evidence`, `published_precision` | how the date was found and the text it came from | page metadata / dateline / news listing page / first date in text; day or month |
| `dateline_location` | where the release was issued | the publisher's city, not the target's |
| `news_category`, `category_evidence` | Deal, Personnel, Fund / firm, Recognition / media, Other | with the words that decided it |
| `transaction_type`, `type_evidence` | Platform investment (the publisher invests), Add-on acquisition (a company it owns acquires), Growth / minority investment, Merger / combination, Exit (the publisher sells), Dividend recapitalization, Unclear | with the evidence |
| `deal_status`, `status_evidence` | Completed (the text says it happened), Announced / pending (agreement, expected to close), Unknown | from the headline, lead sentence and the sentence after it only |
| `buyer`, `buyer_short`, `buyer_role` | the acquirer or investor, its short name, its relationship to the publisher | Publisher (the PE firm itself) / <firm> portfolio company / Third party (acquirer) |
| `sponsor`, `seller` | the PE firm behind the deal; the seller in exits | |
| `target`, `n_targets`, `parties_evidence` | the company (or companies, `;`-separated) acquired or invested in | with the sentence |
| `target_description`, `description_source` | the words that describe the target's business | lead-sentence appositive, a sentence about the target, or its "About" section |
| `target_industry`, `industry_evidence`, `industry_ambiguous` | category of the **target's own business** (taxonomy in `pedeals/taxonomy.py`) | Industrial manufacturing, Value-added distribution, Aerospace & defense, IT services & cybersecurity, Software & data, Business services, Healthcare, Building & infrastructure services, Transportation & logistics, Consumer & food, Energy & environmental, Financial services, Unknown |
| `customer_markets` | industries the target **serves** (kept apart from its own industry) | e.g. Industrial / OEM, Healthcare, Government |
| `target_location`, `target_state`, `target_region` | where the target is based, if the text says | state empty when not stated or ambiguous (Kansas City) |
| `rationale`, `rationale_tags` | up to two sentences giving the stated reason, and the capabilities they name | Product / service expansion, Geographic reach, Customers / channels, Technology / capability, Capacity / scale, Talent / leadership |
| `price_disclosure`, `deal_value_usd`, `price_evidence` | Disclosed (a dollar amount tied to the deal), Not disclosed (stated), Not stated | **a missing price is empty, never 0** |
| `ownership`, `closing_date` | stake (Majority / Minority / 100% / Unknown) and a closing date only if the text gives one | otherwise Unknown / Not stated |
| `review_flags` | what a reviewer should check | e.g. several targets, industry ambiguous, date from news listing page, relationship inferred |
| `in_period`, `period_note`, `duplicate_of`, `also_reported_by`, `platform_key` | period membership, duplicates, the key that groups a platform across renames | |

`all_articles.csv` holds every parsed article (deals and other news); `coverage.csv`, `failures.csv`, `deals.json` and
`deal_brief.md` complete the export.

## 6. Validation (`pedeals/validate.py`, `data/pe_deals/validation/`)

* **Reference labels.** 109 articles were labelled field by field from the source pages
  (`reference_labels.csv`, conventions in `validation/README.md`): **71 development** articles (a stratified sample
  plus hard cases) on which the rules were improved, then the rules were **frozen in git**, then **38 held-out**
  articles were drawn at random from those not yet labelled and labelled. Only held-out accuracy is quoted as the
  tool's accuracy; development accuracy is in-sample.
* **Held-out results** (30 deals, 8 other items; 95% Wilson intervals):

| Field | Held-out accuracy | Development (in-sample) |
|---|---|---|
| Deal vs other news | 38 / 38 (91-100%) | 71 / 71 |
| News category, publication date | 38 / 38 each | 71 / 71 each |
| Transaction type, buyer, target, price disclosure | 30 / 30 each (89-100%) | 54/54, 54/54, 53/54, 54/54 |
| Target industry | 26 / 30 = 87% (70-95%) | 49 / 53 = 92% |
| Target state | 29 / 30 = 97% (83-99%) | 52 / 54 |
| Deal status | 29 / 30 = 97% (83-99%) | 54 / 54 |

* **Errors investigated.** Industry is the weakest field: a target described by its end product (a pet-wellness
  *manufacturer*, a sealing products *supplier*) or with words from two categories (a "cloud-based case management
  platform" for litigators) is a judgement call where the rules and a human can disagree. Status errors come from
  history sentences ("acquired ISPN in 2020") and were reduced by reading only the lead sentence and the one after it.
  Every mismatch is listed in `results.md` and in the Data quality tab.
* **Consistency checks** on all records (dates after collection, publisher as its own add-on buyer, zero prices,
  missing evidence, duplicate ids ...): 0 failures. **Repeatability**: two builds from the saved pages give identical
  records (fingerprint printed). `tests/test_pedeals.py` (25 tests) covers the hard cases and fails if held-out
  accuracy regresses.
* **Limits of this evidence.** The labels were produced by an AI assistant reading the pages and have not yet been
  checked by a person (`verified_by_human = no`); a group member should re-label a random subset blind and report the
  agreement. 25 of the 38 held-out items come from Align, the most active publisher, so accuracy on other sites' formats
  rests on fewer items. Validation evidence is accuracy against labels; the `review_flags` are the program's own
  warnings and are **not** evidence of accuracy, and no output was corrected by hand.

## 7. Using the outputs: the decision scenario

*"Our platform is an IT managed-services provider. Which capabilities are peer platforms adding, and where should our
next add-on search start?"* In the **Add-on research brief** the user picks *IT services & cybersecurity*: the brief
lists every such add-on with buyer, sponsor, state, stated rationale and a source link, and counts the rationale tags.
In the collected data, technology / capability and geographic reach dominate those add-ons (Goldner Hawn's Cyber
Advisors added eight managed-service and security firms in about 20 months, a median of 79 days apart; Tonka Bay's
Corporate Technologies four), which points the user to (a) security-testing and compliance capabilities and (b)
regional MSPs in states the platform does not cover. The **buy-and-build tracker** shows which platforms run a fast,
steady cadence, i.e. whose integration playbook is worth asking about.

*Supported by this evidence:* which companies were bought, by whom, when, and what the sponsors said they add.
*Needs further research:* the prices paid (never disclosed here), whether the add-ons improved margins, growth or
retention, deals the firms did not announce, and the targets' size.

## 8. Creative features (beyond the required workflow)

* **Buy-and-build tracker**: platforms grouped across renames (Marco Rubber & Plastics → Marco Sealing Solutions) by the
  short name the releases use; add-ons, companies acquired, median days between add-ons; a timeline coloured by sponsor.
* **Add-on research brief**: the decision scenario as a downloadable Markdown memo with links, generated for any
  industry and question.
* **Evidence cards and similar deals**: each record as a field → value → source-sentence table; a TF-IDF search for
  deals whose target and rationale read alike ("who else bought something like this?").
* **Monitoring**: "new since" digest and a one-firm live refresh in the tab (polite, incremental).
* **Ownership inference across announcements** and **disclosure-gap report** (what announcements leave out).

## 9. Limitations

* Four firms' self-published news: selective (firms announce what flatters them), incomplete (not every add-on is
  announced) and small. Not a market dataset.
* 14 articles have no publication date anywhere the collector may read (Huron's older pages, three Tonka Bay pages);
  they are listed, not placed in the period. Huron dates from its news page are known to the month only.
* Rule-based extraction fits the press-release formulas of these sites; a new firm needs a source entry and a check of
  its pages against labels before its records are trusted.
* Industry is inferred from description words with a small taxonomy; customers' industries are separated but the
  boundary (a "provider of services to utilities") still needs judgement.
* Prices, closing dates and stakes are almost never disclosed in these releases; the fields exist so that the gap is
  visible, not because the data can fill them.

## 10. Run it

```bash
pip install -r requirements.txt
streamlit run app.py                     # tab "🤝 PE deals"
python -m pedeals build                  # parse + extract the saved pages, export data/pe_deals/output/
python -m pedeals validate               # validation report -> data/pe_deals/validation/results.md
python -m pedeals new-since 2026-09-01   # monitoring digest
python -m pedeals collect --firm tonkabay   # polite live refresh (minutes), then build again
python -m pytest tests/test_pedeals.py
```
