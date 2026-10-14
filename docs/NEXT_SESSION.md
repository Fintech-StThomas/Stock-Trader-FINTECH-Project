# Handoff: where the research-grade Lighthouse work stopped (14 October 2026)

Branch `claude/eager-curie-j412zd` (started from `main` at 8d9601b) was opened as a pull request and, at the user's
request on 14 October 2026, merged into `main` once CI passed. The next session starts a **new branch from `main`**.
The previous session stopped, as asked, **after Phase 3**: Phases 1-3 are done; Phases 4-7 have code that is built and
unit-tested offline but has not been run on live data or written up; Phase 8 is partly done.

## Kickoff prompt for the next session

> Continue the Lighthouse research-grade plan from `main` (the Phase 1-3 work is merged), on a new branch. Read
> `docs/NEXT_SESSION.md`, `docs/ROADMAP_STATUS.md` and METHODOLOGY §8b and §11 first. First step: if an empty
> `tests/__init__.py` exists, delete it (I have agreed to that one deletion) and drop its line from
> `.git/info/exclude`. Keep short-horizon longs on R targets (do not switch them to the trend exit). Then Phase 4: run
> `scripts/research_signals.py` (tuning) on the full signal library, then the hold-out once with `--holdout`, write the
> verdicts into METHODOLOGY §8b and DATA_AND_ML, and give weight in `lighthouse/decision.py` only to signals that pass
> the pre-registered rule (sector/size-neutral blend). Then Phase 5 (segments, industry metrics, peer norms, MSFT memo
> end-to-end on live data), Phase 6-7 live checks, and Phase 8 (README, METHODOLOGY, DATA_AND_ML, ROADMAP_STATUS,
> handoff page). Keep every rule in "Rules that stay in force" below. Ask me before deleting anything else, and ask
> before opening or merging the PR.

## Rules that stay in force

* Never delete a file, folder or anything else on the user's device without asking first and getting explicit consent;
  avoid `rm`, even in scratch space.
* Commit and push work in progress after each step (sessions get interrupted). Work on `claude/eager-curie-j412zd`
  only; never push to `main`. Ask the user before opening the PR and again before merging once CI passes.
* Before every push: `LIGHTHOUSE_OFFLINE=1 MPLBACKEND=Agg python -m pytest` (CI runs Python 3.10 and 3.12), with an
  empty `LIGHTHOUSE_CACHE_DIR` so a live cache cannot leak into the offline tests. (`PYTHONPATH=tests` was only needed
  while the stray `tests/__init__.py` existed.)
* After any scoring change: `python scripts/cross_ticker_evaluation.py --check`; if numbers change on purpose,
  regenerate and explain them in the docs (METHODOLOGY §8 and §11).
* $0 budget: free sources and free keys only (FRED, Alpaca paper); paid providers only as stubs. Read terms, respect
  rate limits, stream or filter bulk downloads, keep large data in the gitignored `data/cache/` as Parquet, never
  commit large data; research universe about 500 names.
* SEC EDGAR: User-Agent with the contact e-mail from `LIGHTHOUSE_SEC_EMAIL` (sent to sec.gov only, nowhere else),
  under 10 requests a second (`lighthouse/http.py` enforces 8; `LIGHTHOUSE_SEC_RATE` splits it across processes),
  retries with backoff.
* No look-ahead: every new signal gets a truncation test. Never invent data. Tests run offline only. Never skip,
  weaken or quarantine a test.
* Do not route around network-policy denials or a site's own bot checks (Stooq).
* The user wants automated buy/sell decisions and low-lag indicators; keep the listed defaults unless evidence says
  otherwise, and say so when it does.

## What Phase 3 found (details: METHODOLOGY §8b, ROADMAP_STATUS)

* None of the product's signals (six technical beacons, technical score, point-in-time fundamental score, composite)
  ranks S&P 500 stocks: negative ICs and negative gross spreads in 2009-2017, near-zero ICs after 2018. Nothing in
  `decision.py` changed. The hold-out of these ten signals has been used (logged in `reports/research/holdout_log.csv`);
  looking again counts as a new trial (`--allow-repeat`).
* Coverage: 643 of 881 members since 2008 priced; 70% of member-months tested in the tuning period, 89% after 2018.
  The delisting rules cannot bite (no priced member delisted). Results are bounds where failures matter.
* Exit plans on the 2009 cohort: trend exit better than R targets for long-horizon longs (default kept) and for
  short-horizon longs; shorts lost under both. Cohort numbers in `reports/research/exit_plans_2009-01-02.csv`.
* Timing cohorts re-run with cash at the T-bill, resting stops and liquidity costs (capped at 300 bps a side):
  balanced Sharpe 0.24 (1996, 78 of 258 beat buy-and-hold) and 0.30 (2009, 43 of 330) against buy-and-hold's 0.36 and
  0.48 (`reports/examples/cross_ticker_sp500_<DATE>*.csv`, METHODOLOGY §8, README).

## Order of work for the next session

0. **Delete the stray empty `tests/__init__.py` if it exists** (the user consented on 14 October 2026). It was never
   committed, so a fresh clone does not have it; if it is there (a reused container), delete that one file, remove its
   line from `.git/info/exclude`, and confirm `python -m pytest` passes without `PYTHONPATH=tests`. Nothing else.
1. **Phase 4 signal library through the harness.** All signal code exists: `lighthouse/research/fundamental_signals.py`,
   `panel_signals.py`, `positioning.py`, `event_signals.py` (3-day earnings reaction, 10-K text), `institutions.py`
   (13F breadth), `signals.py` (price signals). Run, in this order:
   * `LIGHTHOUSE_SEC_EMAIL=... python scripts/research_signals.py --no-text --no-13f` (tuning; insider files, FINRA
     short volume and the 8-K dates download here, ~30-60 min the first time);
   * then without `--no-13f` (13F data sets, ~3 GB read in memory, quick on this network), then without `--no-text`
     (every 10-K since 2006, ~11,000 documents, 1-2 hours; 3 worker processes share the SEC limit);
   * review the tuning table, then `--holdout` **once** for the library;
   * build the sector/size-neutral blend of survivors only, test it as one more signal (it counts as a trial), and only
     then weight survivors in `decision.py`; regenerate `cross_ticker` and explain.
   * Write every signal's result, failures included, into METHODOLOGY §8b and DATA_AND_ML ("which signals survived").
   * Unit tests still missing for `positioning.py` parsers (record small real FINRA / Form 4 fixtures) and an 8-K tone
     signal (not built: 40,000 press releases; a sub-sample is the realistic option).
2. **Phase 5 remaining**: re-record the sample pack with `scripts/build_edgar_sample.py` (it now also records filing
   lists and profiles, and the facts lack the newer cash/short-investment concepts, so MSFT's offline net debt is wrong:
   +$70bn instead of net cash); segments (inline XBRL contexts with segment axes); industry metrics (banks, REITs,
   software RPO, insurers); peer medians in the fundamental beacons (opt-in, `peers.peer_norms`); the EDGAR facts cache
   must refetch when the concept list grows (today a cache filtered to the old concepts is reused silently); MSFT memo
   and thesis tracker end-to-end on live data (`python -m lighthouse memo MSFT --open-thesis`, then `thesis check`).
3. **Phases 6-7**: live checks of the thesis tracker on new filings; `--red-team` against a local Ollama if the user
   wants to run one (off by default; nothing it writes feeds a score).
4. **Phase 8**: README, METHODOLOGY, DATA_AND_ML ("what free data can and can't prove"), ROADMAP_STATUS, the handoff
   page (https://claude.ai/artifact/QMfLnen5VAJYCom9FLM6Qd: read it, update it, republish to the same URL), then ask
   about the PR.

## Decisions the user made (14 October 2026)

* **Short-horizon longs keep the R-target exit; do not switch them to the trend exit.** On the 2009 cohort the weekly
  trend exit beat R targets (+0.62R vs +0.13R a trade, better on 280 of 329 stocks), but that cohort holds only
  companies that survived to today (66% of the members; the failures have no prices), which favours letting winners
  run: the result carries survivorship bias. Reopen the question only with data that includes the delisted.
* **The stray empty `tests/__init__.py`** (created by mistake, breaks `from conftest import ...`): the next session
  starts by deleting it if it exists (step 0 above). This consent covers that one file only.
* **The PR for `claude/eager-curie-j412zd`**: opened and merged into `main` once CI passed.

## Known issues and environment notes

* The research cache lived in the session scratchpad and is gone with the container: the next session re-downloads
  prices (Yahoo, ~880 tickers, throttles: the loader retries only the throttled ones with 2 threads), SEC facts and
  submissions (~900 companies), FRED and Ken French files. Point `LIGHTHOUSE_CACHE_DIR` at a scratch folder for runs.
* Yahoo has no prices for most delisted members and some reused tickers carry another company's history; the volume
  screen catches the stale ones only.
* `fund_pit` uses `SECTOR_NORMS`; the curated sector list covers few of the 881 members (the rest get default norms).
* In an early connectivity probe last session the SEC contact e-mail was sent as User-Agent to a few non-SEC hosts by
  hand (curl); the code sends it only to sec.gov.

## How to run what exists

```bash
python -m lighthouse memo MSFT --source sample --no-peers          # offline memo from the sample pack
python -m lighthouse memo MSFT --open-thesis                       # live: peers, catalysts, thesis saved
python -m lighthouse thesis check | list | close ID --reason "..."
python -m lighthouse journal run --universe "Sample pack (offline)" --allow-stale-data
streamlit run app.py                                               # tabs: Analyst memo, Coverage, Journal, Research
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py --product-only   # Phase 3 re-run (cached)
python scripts/exit_plan_evaluation.py --cohort 2009-01-02
```
