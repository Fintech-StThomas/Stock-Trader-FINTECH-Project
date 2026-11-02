# PE Deal Sourcing: method, data dictionary and limitations

A separate product inside the Lighthouse dashboard (tab **🤝 PE deals**, package `pedeals/`, notebook
`notebooks/PE_Deal_Intelligence.ipynb`). It reads the deal announcements that private-equity sponsors publish on their
own websites and turns them into a sourcing record: which niches are being rolled up, which sponsor-owned companies
are due to trade, and who a deal team would bid against. Every value sits next to the sentence it came from.

## 1. User, problem and decision

**User.** A deal team (associate to principal) at a lower-middle-market sponsor sourcing in five sectors:
**building & facility services, IT services & cybersecurity, tech-enabled business services, industrial manufacturing
and value-added distribution.**

**Recurring task.** Every week the team asks: *which niches in our sectors are other sponsors consolidating, and how
fast? Which sponsor-owned companies have been held long enough to come to market? Who would we bid against for a target
in a given niche and region?* Today that means reading dozens of sponsors' press releases by hand, with no common
record and no links back to the evidence.

**What the tool gives.**
* **Sourcing radar**: add-ons per niche, the number of platforms and sponsors buying, and momentum (last twelve
  months against the twelve before).
* **Exit watch**: a portfolio ledger rebuilt from each sponsor's own announcements; companies held four or more years
  with no exit announced are candidates for a sponsor-to-sponsor deal.
* **Competing buyers**: the platforms buying in a niche (and states), their cadence and how long they have been owned.
* **Sourcing brief**: a downloadable memo per sector / niche: consolidators, what they bought (the target profile),
  exit-watch companies, what still needs research.
* **Deal finder, evidence cards, similar deals and a new-deal digest**, and the validation evidence.

**Why these five sectors.** Public evidence (2025-2026) ranks building/facility services (HVAC deals up about 13% in
2025 while industrial M&A fell; PE the main buyer), IT managed services (169 MSP deals in 2025, PE in about 69% of those
disclosed), tech-enabled business services (favoured as non-cyclical), industrial manufacturing (record 47.6% sponsor
share of industrial deals) and value-added distribution among the most active lower-middle-market buy-and-build areas;
healthcare services (state PE-ownership laws), software (2026 AI-driven repricing) and consumer (weakest activity) were
left out. Sources: Capstone Partners HVAC update (Dec 2025) and Industrials report (May 2026), Kroll (Nov 2025), Omdia
(Apr 2026), PitchBook 2025 Annual US PE Breakdown (Jan 2026). These five sectors also hold most of the deals the
sponsors below announce.

**What it does not claim.** "No exit announced" is not proof that a company is still held, an entry date found in a
release is the sponsor's own statement, and the sponsors choose what to announce: the counts describe the collected
announcements, not the PE market. Companies nobody has bought yet are outside what announcements can show.

## 2. Scope: sources, period, amount

Sixteen buy-and-build sponsors active in the five sectors: the four of the first version (Tonka Bay, Goldner Hawn,
Huron Capital, Align Capital Partners) and twelve added in November 2026 from a state-by-state research list
(`data/pe_deals/research/pe_firms_by_state.csv`): Fort Point Capital (MA), Compass Group Equity Partners (MO), ShoreView
Industries (MN), CID Capital (IN), Mason Wells (WI), Gryphon Investors (CA), Gridiron Capital (CT), Oridian Capital
Partners (DC), Azalea Capital (SC), Montage Partners (AZ), Pike Street Capital (WA) and Resilience Capital (OH).

*Vetting.* A sponsor became a source only if its robots.txt allows its news pages and does not opt out of AI crawlers by
name, its news is listed in an XML sitemap (or a listing page), and its article pages carry a date. Considered and left
out: Slate Capital, NewSpring Capital and LFM Capital (robots.txt names AI crawlers), Blue Point Capital (pages ask not
to be indexed), Trivest (pages show no date), Druid Capital (eight posts, none since 2022), Graycliff and Plexus
(AI-crawler blocks), Pfingsten (robots.txt refused).

*Period.* Announcements published from **1 January 2015**: long enough to date holdings for the exit watch; the sourcing
views default to deals since 2022.

*Amount.* 1,795 articles collected; 1,048 unique deal announcements, 605 of them placed in the five focus sectors (with the
accuracy limits in section 6); 29 articles
carry no date and are kept out of the period; one page failed (HTTP 504, recorded). Per-sponsor coverage is in the Data
quality tab and `data/pe_deals/output/coverage.csv`.

## 3. Collection (`pedeals/collect.py`, `pedeals/fetch.py`)

* **Polite by construction.** robots.txt is read for each host and obeyed, including a Crawl-delay written outside a
  user-agent group (10 seconds on several sites); at least 5 seconds between requests to a host otherwise; one thread
  per host; the User-Agent names the tool and links to this repository.
* **What is kept.** Every page with its URL, HTTP status and UTC fetch time (`data/pe_deals/cache/pages.jsonl.gz`),
  every sitemap URL with its outcome (`discovery.csv`), news listing pages used for dates (`listings.jsonl.gz`).
  Saves are atomic and checkpointed every 25 pages, so an interrupted run keeps what it fetched.
* **Cached data.** Parsing and extraction always run on the saved pages, so results are reproducible and survive a
  site going offline; the app and every export say **CACHED DATA** and show the collection dates. A local parse cache
  (keyed on each page and on the extraction code) makes rebuilds take seconds; it is never committed.

## 4. Processing and extraction (`pedeals/parse.py`, `pedeals/extract.py`, `pedeals/sourcing.py`)

1. **Parsing.** The article body is the deepest container holding 90% of the page's text blocks (paragraphs, list items
   and text-bearing divs, for themes that use no `<p>`); menus, headers and footers are removed unless they hold much of
   the text. "About <company>" sections are kept apart as boilerplate.
2. **Publication date**: page metadata, JSON-LD, `<time>`, a visible date label, the release's dateline, the date beside
   the link on the sponsor's news page, the first date in the text, a month-only listing date (flagged); else empty and
   out of the period. Never used as a closing date.
3. **Deal or not**: headline first, then the opening text (fund closes, milestones, newsletters, people and awards are
   not deals).
4. **Parties** from the lead sentence (the company the release calls "the Company", "its portfolio company X", "X,
   backed by …"), headline patterns as the fallback; names are checked for plausibility (not a state, city, month or
   headline fragment). An acquisition by a non-sponsor company is an add-on when the release says the sponsor owns it,
   when another announcement of the sponsor shows it (flagged *inferred*), or, failing both, because it is announced on
   the sponsor's own site (flagged *presumed*).
5. **Sector and niche**: the target's industry from the words describing its business (or, failing that, its name and
   the headline); the five focus sectors map from it; the niche is the sector's sub-category with the most matching
   words (`pedeals/taxonomy.py`).
6. **Holding evidence**: sentences that date an investment ("ACP acquired StenTech in May of 2022", "since partnering
   with Tonka Bay in 2023") and renames ("rebranded as CompletePet").
7. **Portfolio ledger** (`sourcing.ledger`): per sponsor, an investment announcement or a dated sentence opens a
   holding, add-ons show it is still owned, a sale closes it, a rename links the names; names merge when one contains
   the other or the releases give the same short name. Entry dates attach only to companies the sponsor's deal records
   know (people and co-investors never become holdings). A holding not mentioned for four years is marked *stale*
   (probably sold without an announcement) and kept out of the exit watch, as are holds over ten years.
8. **Duplicates**: URLs normalised and stored once; the same headline twice or the same target within 120 days marked
   `duplicate_of`.

Rules, not a language model: every value can be traced to a pattern and a sentence, runs are free and deterministic,
and errors can be investigated. The cost is lower recall on unusual wording, which the validation measures.

## 5. Data dictionary

**Deal record** (`data/pe_deals/output/deals.csv`, one row per deal announcement):

| Field | Meaning | Values / notes |
|---|---|---|
| `deal_id`, `source`, `publisher`, `url`, `collected_at` | the announcement, the sponsor that published it, when it was fetched (UTC) | the publisher is **not necessarily the buyer** |
| `headline`, `published_date`, `published_method`, `published_evidence`, `published_precision` | title; publication date and how it was found | not the closing date |
| `news_category`, `category_evidence` | Deal, Personnel, Fund / firm, Recognition / media, Other | |
| `transaction_type`, `type_evidence` | Platform investment, Add-on acquisition, Growth / minority investment, Merger / combination, Exit, Dividend recapitalization, Unclear | |
| `deal_status`, `status_evidence` | Completed, Announced / pending, Unknown | |
| `buyer`, `buyer_short`, `buyer_role`, `sponsor`, `seller` | acquirer / investor, its short name, its relationship to the publisher | Publisher; <sponsor> portfolio company (stated / inferred / presumed); Third party |
| `target`, `target_short`, `n_targets`, `parties_evidence` | company or companies acquired (`;`-separated) | |
| `target_description`, `description_source` | the words describing the target's business | |
| `target_industry`, `industry_evidence` | the target's own industry (taxonomy in `pedeals/taxonomy.py`) | |
| `focus_sector`, `niche`, `niche_evidence` | one of the five focus sectors (or Outside focus) and its niche | e.g. Building & facility services / HVAC & mechanical |
| `customer_markets` | industries the target serves, kept apart from its own | |
| `target_location`, `target_state`, `target_region` | where the target is based, if stated | empty when not stated or ambiguous |
| `rationale`, `rationale_tags` | the stated reason and the capabilities it names | |
| `price_disclosure`, `deal_value_usd`, `price_evidence`, `ownership`, `closing_date` | disclosure gaps | a missing price is empty, never 0 |
| `entry_statements`, `renames` | dated investments and renames found in the text (JSON) | feed the ledger |
| `review_flags`, `in_period`, `period_note`, `duplicate_of`, `platform_key` | reviewer warnings, period, duplicates, the key grouping a platform across renames | |

**Portfolio ledger** (`portfolio_ledger.csv`; `exit_watch.csv` is its filtered view): `sponsor`, `company`, `status`
(Exited / No exit announced), `entry_date` and `entry_basis` (investment announced / date stated in a release / first
seen), `entry_evidence`, `entry_url`, `years_held` and `years_basis` (from entry / at least), `last_seen`, `stale`,
`add_ons`, `last_add_on`, `exit_date`, `exit_buyer`, `exit_url`, `focus_sector`, `niche`, `state`, `evidence_urls`,
`signal` (strong: 5+ years from a dated entry; watch; watch (entry not dated)).

**Consolidation map** (`consolidation_since_2022.csv`): `focus_sector`, `niche`, `add_ons`, `platforms`, `sponsors`,
`last_12m`, `prior_12m`, `momentum`, `latest`, `target_states`.

## 6. Validation (`pedeals/validate.py`, `data/pe_deals/validation/`)

Accuracy is measured against articles labelled field by field from the source pages, in rounds: rules are improved on a
**development** set, frozen in a git commit, then a **held-out** set is drawn and labelled. Only held-out accuracy is the
tool's accuracy.

| Field | **Round 3 (quote this)** | Round 2 at its freeze | Round 1 | Development (in-sample) |
|---|---|---|---|---|
| Deal vs other news | 42/44 (95%) | 60/60 | 38/38 | 71/71 |
| Publication date | 44/44 (100%) | 60/60 | 38/38 | 71/71 |
| Transaction type | 30/36 (83%) | 43/48 | 30/30 | 54/54 |
| Buyer | 29/36 (81%) | 44/48 | 30/30 | 54/54 |
| Target | 32/36 (89%) | 45/48 | 30/30 | 53/54 |
| Target's state | 25/36 (69%) | 40/48 | 29/30 | 52/54 |
| Deal status | 27/36 (75%) | 42/48 | 29/30 | 54/54 |
| Price disclosure | 33/36 (92%) | 46/48 | 30/30 | 54/54 |
| **Target industry** | **18/36 (50%)** | 26/47 | 27/30 | 50/53 |
| **Focus sector** | **18/36 (50%)** | 30/47 | — | — |
| **Niche** | **8/27 (30%)** | 13/31 | — | — |

Round 1: 38 articles from the first four sponsors. Round 2: 60 articles from the twelve added sponsors and 2015-2022,
scored at the freeze (commit `0bf8a14`). The industry rules were then fixed on round 2's mismatches (industry rose to
39/47 on round 2, which is now in-sample), frozen again (commit `1835b7f`) and tested on round 3: 44 new articles from
every sponsor. Per-round tables, 95% intervals and every mismatch: `data/pe_deals/validation/results.md`.

**What this says.** Finding deals, dates, buyers and targets holds up on sponsors the rules were not written on (83-100%).
**The sector and niche do not: about half of the targets get the sector a reader would give them**, and each fresh
round scored lower than the last, so the in-sample numbers overstate accuracy for new sponsors. The misses on round 3
are taxonomy boundaries (a manufacturer of baked goods or beauty products: Consumer or Industrial manufacturing?), the
wrong sentence taken as the description, and vocabulary the keyword lists do not cover. Two rounds of keyword tuning
moved the out-of-sample number little; a model-based classifier (section 9) is the likelier fix. Until then the app
says so beside the sector filter, and the sourcing views should be read as a first cut to be checked against each
deal's description.

* **Consistency checks** on every record (dates after collection, sponsor as its own add-on buyer, zero prices, buyer
  equal to target, platform or add-on without buyer and target, duplicate ids): none fails. **Repeatability**:
  two builds from the saved pages give identical records (fingerprint printed in `results.md`).
* **Limits of this evidence.** Labels were produced by AI models reading the pages (round 1 by the model that wrote the
  rules; round 2 by a different model, blind to the program's output); none has been checked by a person
  (`verified_by_human = no`). Before quoting the numbers, a group member should re-label a random subset blind and
  report the agreement. The exit watch's entry dates are scored indirectly (through the fields above and the ledger
  tests), not as a labelled field. `review_flags` are the program's own warnings, not evidence of accuracy, and no
  output was corrected by hand.

## 7. Using the outputs: a sourcing week

*"We are building an IT managed-services platform. Who else is buying MSPs, what did they pay for (what kind of
company), and which sponsor-owned MSP or adjacent business may come to market?"* In **Competing buyers** the team picks
IT services & cybersecurity / Managed IT services (MSP): Goldner Hawn's Cyber Advisors (five MSP add-ons; sold in 2026)
and Tonka Bay's Corporate Technologies (four, still owned) were the most active buyers, in the Upper Midwest and Iowa /
Indiana. The **Sourcing brief** lists those targets with their descriptions and states (the profile to look for) and the
sponsor-owned IT companies past four years. The **Exit watch** shows the broader pipeline across the five sectors: for
example E Source and Marco Sealing Solutions (Align, since 2019), Mechanix Wear and Ncontracts (Gryphon), KGM
(Compass), PWSG (CID) and Right Time Group (Gryphon, twelve add-ons): the companies whose sponsors are most likely to
start a sale process.

The ledger's 85 exits with a dated entry were held a median of 5.6 years (middle half 3.7-7.4), which is why the watch
starts at four years. *Supported by this evidence:* who bought what, where and when, and how long each sponsor says it
has owned its platforms. *Needs further research:* whether the companies are still held and for sale, their size and
financials, prices, the sponsors' fund vintage, and companies no sponsor has bought.

## 8. Creative features

Exit watch from a portfolio ledger rebuilt out of press releases (entry dates read from passing sentences, renames
linked); consolidation map with momentum; competing buyers per niche and region; sourcing brief generator; evidence
cards and similar-deal search; new-deal digest and polite live refresh; ownership inferred across announcements;
disclosure-gap report.

## 9. Limitations

* Sixteen sponsors' self-published news: selective, incomplete, and skewed to sponsors that announce add-ons; not a
  market dataset.
* "No exit announced" can mean a quiet sale: stale and very long holds are excluded by default, but a candidate still
  needs checking before outreach.
* Entry dates are read from text; holdings with no dated entry count "at least" from their first mention.
* **Sector and niche are the weakest fields** (about 50% and 30% on the latest held-out round): niches come from a
  keyword taxonomy, "Other: <sector>" marks deals it does not place, and a deal can be filed under the wrong sector.
  The next step is a language-model classifier for these two fields only, scored on the same held-out rounds, with
  the keyword rules kept as the fallback.
* Rule-based extraction fits these sites' release formulas; a new sponsor needs vetting and a labelled check.
* Prices, closing dates and stakes are almost never disclosed.

## 10. Run it

```bash
pip install -r requirements.txt
streamlit run app.py                       # tab "🤝 PE deals"
python -m pedeals build                    # parse + extract the saved pages; export data/pe_deals/output/
python -m pedeals validate                 # validation report -> data/pe_deals/validation/results.md
python -m pedeals new-since 2026-09-01     # monitoring digest
python -m pedeals collect --firm gryphon   # polite live refresh of one sponsor (minutes), then build again
python -m pytest tests/test_pedeals.py
```
