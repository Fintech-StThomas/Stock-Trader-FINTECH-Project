# Handoff: where the Lighthouse work stopped (15 October 2026, second session of the day)

Branch `claude/ecstatic-wright-3w8w59`, started from `main` at 9c98e9a after pull request #6 (Phases 4-8) was merged.
It carries open items 1-3 of the previous handoff; no pull request is open for it yet (ask the user before opening
one, and again before merging once CI passes). **The watch workflow only runs once this branch is on `main`**: GitHub
runs scheduled workflows from the default branch only, and the first forward snapshot is due after the close of
30 October 2026.

## What this session did (details: METHODOLOGY §8c and §11 "Seventh October 2026 session", DATA_AND_ML §5-6)

* **Item 1, forward evidence: the watch list is built** (`lighthouse/research/watch.py`, `python -m lighthouse journal
  watch`, `.github/workflows/watch.yml`, the Journal tab). The three near misses (`fcf_yield`, `short_volume`,
  `risk_similarity`) are recorded at each month end for every point-in-time member under their logged configuration
  hashes (the same trials, not new ones), in two new append-only journal tables (schema 2; older files upgrade in
  place). A snapshot counts only if written before the next open; the recorder takes no date. The forward rule is
  written down before any data exists (METHODOLOGY §8c): an annual look, at least 36 months, Newey-West t >= 2, positive
  net return for the tuning use; meeting it only makes the case for a new pre-registered test. Honest power: at their
  hold-out ICs a t of 2 needs about 7 years (risk similarity), 10 (short volume) and decades (FCF yield). Watch-only:
  a test proves no verdict or ticket reads it.
* **Item 2, the user's decisions, implemented**: the DCF's equity premium is the geometric mean of the Ken French
  market factor since 1963 (6.2% in data to August 2026; the arithmetic 7.2% is shown as a sensitivity on the base
  case); RONIC = WACC in the bear case; REIT FFO per NAREIT (impairments added back); a bank that never tagged preferred
  stock counts it as 0, flagged. The MSFT live memo was not re-run (no SEC contact e-mail approved this session), so
  the $199 base case in the docs is from before these changes.
* **Item 3, the five data limits, fixed** (METHODOLOGY §11, seventh session): class shares take their issuer's Form 4s
  by CIK (BRK-B, BF-B; both classes of GOOGL/GOOG, FOXA/FOX, NWSA/NWS share them); the NYSE calendar knows the one-off
  closures since 1990; a pre-2018 FINRA day is cached only when every facility answered; `edgar.bulk_quarter` records
  its tags and refetches for more; a successor's filing list merges its predecessors' (XOM), and documents are fetched
  under the registrant that filed them. The §8b results predate these fixes and were not re-run (a re-run is a new
  trial); DATA_AND_ML §5 counts the member-months they touch (a few names a month).
* **Item 4** (local LLM red team) skipped: no local model in this environment.
* **Found on the way and fixed: the memo's market value after a split.** The price is as traded on the memo's date,
  the cover-page share count as of its filing; a split in between (AAPL on 15 September 2020) put market value, EV
  and every per-share DCF value off by the split ratio ($0.49tn instead of $1.97tn). `financials_as_of` now adjusts the
  count (METHODOLOGY §11). The research signals have the same flaw and keep it (open item 2).
* Also: both journal workflows now stop when the `journal-data` branch exists but its journal cannot be read (before,
  `|| true` would have started an empty journal and saved it over the real one); the signal-library table in §8b has
  its header row back.

## Kickoff prompt for the next session

> Continue Lighthouse in lotainc0/Stock-Trader-FINTECH-Project. Read `docs/NEXT_SESSION.md`, `docs/ROADMAP_STATUS.md`
> and METHODOLOGY §8b, §8c and §11 first. If the pull request for `claude/ecstatic-wright-3w8w59` was merged, start a
> new branch from `main`. No signal survived; do not add weight to any signal in `decision.py` without a new
> pre-registered test, and never let the watch list reach a verdict or ticket. Pick from "Open items" below in the
> order I ask. Keep every rule in "Rules that stay in force". Ask me before deleting anything, and before opening or
> merging a PR.

## Open items (the user's choice; none started)

1. **Get the watch list running before 30 October 2026**: open and merge the pull request for this branch (the user
   decides), check that the repository secret `LIGHTHOUSE_SEC_EMAIL` exists (without it only short volume is
   recorded), and look at the first run's summary (Actions -> watch). Nothing in `watch.yml` has run yet; the first run
   with an empty cache should take about an hour (an estimate).
2. **A split flaw in the research signals' market value, for the user**: `fundamental_signals.daily_values` multiplies
   the as-traded price by the latest filing's cover-page share count, so between a split's ex-date and the next filing
   it uses the pre-split count (AAPL, 31 August to 30 October 2020: $550bn instead of $2.1tn; FCF yield four times too
   high). It touches `size`, `fcf_yield`, `fcf_sbc_yield`, `earnings_yield`, `book_to_market`, `ebitda_ev`,
   `shareholder_yield`, `altman_z` and `insider_net` in the months after a split. The watch list keeps the flaw
   because it records the logged trial (METHODOLOGY §8c). Fixing it (as `analyst.financials_as_of` now does for the
   memo) changes those signals: their results then need a new trial, and the watch list a new configuration.
3. **The forward rule's thresholds** (36 months, t >= 2, one annual look, at most one month end in six missing or
   late) were proposed by this session, before any forward data: the user may change them now, never after seeing
   forward results.
4. **Live checks not run here**: the MSFT memo after the item 2 changes (how far the geometric premium and the bear
   case's RONIC move the $199 base case); XOM's merged filing list against EDGAR; re-recording the sample pack
   (`scripts/build_edgar_sample.py`) so its filing lists carry the new `cik` column (the old pack still reads).
   Each needs the user's approval of an SEC contact e-mail for that session.
5. **A quarterly short-volume variant** (to cut its 60-66% monthly turnover) would be a new trial: log it in
   `reports/research/experiments.csv` first, then add it to the watch list under its own name.
6. **Local LLM red team** (`memo --red-team`, Ollama): untested live; only if the user runs a local model.

## Rules that stay in force

* Never delete a file, folder or anything else on the user's device without asking first and getting explicit consent;
  avoid `rm`, even in scratch space.
* Commit and push work in progress after each step (sessions get interrupted). Work on the branch the session is given;
  never push to `main`. Ask the user before opening the PR and again before merging once CI passes.
* Before every push: `LIGHTHOUSE_OFFLINE=1 MPLBACKEND=Agg python -m pytest` (CI runs Python 3.10 and 3.12; 331 tests),
  with an empty `LIGHTHOUSE_CACHE_DIR` so a live cache cannot leak into the offline tests. CI's 3.10 job runs
  pandas 2 (pandas 3 needs Python 3.11): after re-recording data or touching dates, run the suite once under pandas 2
  as well (a virtualenv with `pandas>=2.3,<3`), which is how the microsecond-date failure was found.
* After any scoring change: `python scripts/cross_ticker_evaluation.py --check`; if numbers change on purpose,
  regenerate and explain them in the docs (METHODOLOGY §8 and §11).
* $0 budget: free sources and free keys only (FRED, Alpaca paper); paid providers only as stubs. Read terms, respect
  rate limits, stream or filter bulk downloads, keep large data in the gitignored `data/cache/` as Parquet, never
  commit large data; research universe about 500 names.
* SEC EDGAR: User-Agent with the contact e-mail from `LIGHTHOUSE_SEC_EMAIL` (sent to sec.gov only, never written into
  any committed file), under 10 requests a second (`lighthouse/http.py` enforces 8; `LIGHTHOUSE_SEC_RATE` splits it
  across processes, and the 10-K text workers split their process's share), retries with backoff. On 15 October 2026
  the user approved their account e-mail as the contact for that session; ask again in a new one.
* No look-ahead: every new signal gets a truncation test; anything filed or published on a date is used from the next
  session. Never invent data. Tests run offline only. Never skip, weaken or quarantine a test.
* The hold-out of every logged signal has been used (`reports/research/holdout_log.csv`); looking again needs
  `--allow-repeat` and counts as a new trial.
* Do not route around network-policy denials or a site's own bot checks (Stooq).
* The user wants automated buy/sell decisions and low-lag indicators; keep the listed defaults unless evidence says
  otherwise, and say so when it does.

## Decisions the user made

* 14 October 2026: **short-horizon longs keep the R-target exit** (the trend exit's edge came from a survivor-only
  cohort; reopen only with data that includes the delisted).
* 14 October 2026: PR #5 (Phases 1-3) opened and merged.
* 15 October 2026: the SEC contact e-mail for that session's runs; up to 2-3 helper agents allowed for parallel work;
  the executive summary and pitch brought up to date with the current results (done); pull request #6 for
  `claude/focused-bohr-klq8zc` opened (merge only when the user says so).
* 15 October 2026 (next session, branch `claude/ecstatic-wright-3w8w59`, from `main` after PR #6 was merged): work
  open items 1 and 3 with helper agents; item 4 skipped (no local model here). The item 2 decisions: **the DCF's
  equity premium is the geometric mean** of the Ken French market factor since 1963 (the arithmetic 7.2% shown as a
  sensitivity); **RONIC = WACC in the bear case** (base and bull keep today's ROIC floored at the WACC); **REIT FFO
  per NAREIT** (real-estate impairments added back, the impairment line kept beside it); **a bank that never tagged
  preferred stock counts it as 0**, flagged (preferred tagged in other periods but not this one: still no tangible
  book).

## Running the research again (a fresh container re-downloads everything)

Point `LIGHTHOUSE_CACHE_DIR` at a scratch folder. Run long jobs from a pinned copy of the code
(`git worktree add --detach <scratch>/runtree <commit>`) so merging new code cannot change a run halfway (Python
imports lazily), and copy `reports/research/*.csv` back afterwards. Times on this environment's network, at 6-7 SEC
requests a second: Yahoo prices for ~880 tickers ~15 minutes (throttled ones retried with 2 threads); SEC facts,
filing lists and profiles for ~640 companies ~40 minutes; 84 insider data sets and ~2,000 FINRA daily files ~20
minutes; 13F breadth ~8 minutes; every 10-K since 2006 (~11,000) ~75 minutes; the earnings releases of half the
members (~17,000 EX-99s of ~900 KB each) about 2 hours.

```bash
python -m lighthouse memo MSFT --source sample --no-peers          # offline memo from the sample pack
python -m lighthouse memo MSFT --open-thesis --peer-norms          # live: peers, peer-median beacons, thesis saved
python -m lighthouse thesis check | list | close ID --reason "..."
python -m lighthouse journal run --universe "Sample pack (offline)" --allow-stale-data
python -m lighthouse journal watch [--due]                         # record the month-end watch snapshot (only then)
python -m lighthouse journal score                                 # verdicts, tickets and the watch-only section
streamlit run app.py                                               # tabs: Analyst memo, Coverage, Journal, Research
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py --no-text --no-8k   # tuning (cached signals)
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py --signals NAME --holdout --allow-repeat  # a new trial
python scripts/research_signals.py --blend SIGNAL [SIGNAL ...]     # the neutral blend of survivors (none today)
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/build_edgar_sample.py   # re-record the sample pack
```
