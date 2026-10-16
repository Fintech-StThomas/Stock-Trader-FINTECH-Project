# Handoff: where the Lighthouse work stopped (16 October 2026)

Branch `claude/ecstatic-wright-3w8w59`, started from `main` at 9c98e9a after pull request #6 (Phases 4-8) was merged.
No pull request is open for it yet: ask the user before opening one, and again before merging once CI passes. **The
watch workflow only runs once this branch is on `main`**: GitHub runs scheduled workflows from the default branch
only, and the first forward snapshot is due after the close of 30 October 2026.

## What the last sessions did (details: METHODOLOGY §8-8d and §11, seventh and eighth sessions)

* **Forward evidence, valuation decisions, data limits** (seventh session): the watch list of three near-miss signals
  (`lighthouse/research/watch.py`, `journal watch`, `.github/workflows/watch.yml`, journal schema 2); the DCF's
  geometric equity premium, RONIC = WACC in the bear case, NAREIT FFO, banks without tagged preferred at 0; five data
  limits fixed; the memo's market value after a split fixed.
* **Simplification** (eighth session, the user's request: "simplify it ... research more on the back end than an
  extra feature"):
  * Tier 1: one chart stack (Plotly; saved reports write `TICKER_charts.html`), no local-LLM layer, the research code of
    the failed and unwatched signals removed (their logged rows stay), three one-off scripts removed, `memo.py` renamed
    `cache.py`, this file now holds the roadmap.
  * Tier 2: the verdict reads prices only (score = technical − weather; no fundamental gate); no short tickets;
    portfolio methods equal weight / risk parity / minimum variance; SEC EDGAR the only fundamentals source.
  * Tier 3: one cover-page reader (`pit.cover_shares`) and one split restatement (`total_return.restate_shares`) behind
    every market value (memo, research signals, membership); the `run_backtest(start=...)` pandas-3 bug fixed.
  * App: five tabs. *Analyse a stock* builds the analyst memo with every analysis (peers opt-in in the sidebar) and
    shows it next to the verdict in sub-tabs (plan & ticket, memo, technicals, backtest, data); *Find trades* merges the
    trade desk and the watchlist; *Journal & theses* holds coverage, the decision journal and the watch list; the
    Research tab is gone and *How it works* summarises the research from its result files.
* **Research** (eighth session, live data, SEC contact e-mail approved for that session):
  * Pre-registered timing test (METHODOLOGY §8d, committed and pushed as 3ee8756 before the run): on the 1996 and 2009
    S&P 500 cohorts the rule did **not** earn more than its own average exposure held without trading (H1 rejected),
    its drawdown advantage over that exposure held in 1996 and reversed in 2009 (H2 not accepted), and it did not beat
    a plain 200-day trend rule on Sharpe in both cohorts (H3 not accepted). Robust to dropping recycled tickers.
  * The nine share-count signals re-run as new trials after the split fix: no verdict changed (56 trials logged);
    the watch list's `fcf_yield` now records the corrected trial `dd05ff443270` (`watch.SUPERSEDED` keeps the first).
  * The MSFT example (`reports/examples/MSFT/`) regenerated with HTML charts and its memo; its six PNGs deleted with
    the user's consent.

## Open items (the user's choice)

1. **H3's pre-registered consequence, for the user to decide**: a plain 200-day trend rule did about as well as the
   six-beacon rule on Sharpe (median 0.27 vs 0.28 in 1996, 0.32 vs 0.33 in 2009), with more return when fully invested
   and much deeper drawdowns. Simplify the verdict's trading rule to the trend rule (the beacons would stay as the
   explanation), keep the current rule, or leave it open. Nothing changed in the product.
2. **Get the watch list running before 30 October 2026**: open and merge the pull request (the user decides), check that
   the repository secret `LIGHTHOUSE_SEC_EMAIL` exists (without it only short volume is recorded), and look at the
   first run's summary (Actions -> watch).
3. **The forward rule's thresholds** (36 months, t >= 2, one annual look, at most one month end in six missing or late)
   were proposed before any forward data: the user may change them now, never after seeing forward results.
4. **Live checks not run**: the MSFT live memo after the valuation decisions (how far they move the $199 base case);
   XOM's merged filing list against EDGAR; re-recording the sample pack (`scripts/build_edgar_sample.py`) so its filing
   lists carry the `cik` column. Each needs the user's approval of an SEC contact e-mail for that session.
5. **A quarterly short-volume variant** (to cut its 60-66% monthly turnover) would be a new trial: log it in
   `reports/research/experiments.csv` first, then add it to the watch list under its own name.
6. **Clean-up waiting for consent**: `/all_pd2.log` at the container root (80 bytes, written by a broken test pipeline),
   the helper agents' worktrees under `.claude/worktrees/`, and the scratch caches. Nothing was deleted without asking.

## Roadmap status (research-grade Lighthouse on free data, budget $0)

| Phase | Status | What exists |
|---|---|---|
| 1. Free data foundations | done | SEC EDGAR client and point-in-time statements by filing date; historical S&P 500 membership with coverage; dividends, splits, total returns and delisting rules; live-mode rule for tickets; survivorship-aware cohort evaluation |
| 2. Forward journal and paper trading | done (paper fills untested against a live account) | Append-only SQLite journal with inputs, data version and code commit (schema 2: the watch tables); virtual positions; scorecard (hit rate, calibration, Brier skill); Alpaca paper adapter; `.github/workflows/journal.yml` after every close, journal on the `journal-data` branch |
| 3. Research harness | done | Month-end point-in-time panel, rank IC with Newey-West t, quintiles after costs, Fama-French 5 + momentum alphas, both delisting rules, tuning 2009-2017 / hold-out 2018+ evaluated once, deflated Sharpe, append-only experiment log; realistic backtest costs. Product signals: none survives |
| 4. Signal library | done; failed signals' code removed in October 2026 | 37 signals tested; none survives; three near misses on the watch list |
| 5. Single-stock analyst toolkit | done | Point-in-time financials, linked model, WACC, bull/base/bear DCF, reverse DCF, valuation bands, SIC peers, industry metrics, segments, red flags, capital allocation, catalysts, memo; now part of every analysis in the app |
| 6. Senior-analyst layer | done | Variant view, thesis tracker with kill criteria checked on new filings, coverage, conviction scale, post-mortems |
| 7. Text without a paid LLM | done; the optional local LLM was removed in October 2026 | Deterministic text features (`lighthouse/text.py`) |
| 8. App and docs | done; simplified in October 2026 | Five tabs; README, METHODOLOGY, DATA_AND_ML, USER_GUIDE, EXECUTIVE_SUMMARY, PITCH, this handoff |

Findings that still stand: the rule has no edge as a stock-timing tool (§8, §8d) and no signal from free data ranks
the S&P 500 well enough to survive a hold-out (§8b); coverage of the delisted is the binding data gap (Yahoo prices 53%
of the 1996 cohort and 66% of 2009; every priced member still trades); EDGAR is good enough for fundamentals from 2009.

| Data gap (free data) | Cost | Status |
|---|---|---|
| Prices of delisted stocks | The untested part of every cohort is mostly its failures; results are bounds | Reported as coverage; both delisting bounds published |
| Ticker reuse in Yahoo | A recycled ticker's history can belong to another company (ETS, STI, SNDK in the cohorts) | The research panel drops months with volume on fewer than 15 of 21 sessions; the cohort test reports a robustness check without late-starting series; a reused ticker that trades normally is not detected |
| Fundamentals before 2009 | Unknown | By design |
| Ticker → CIK for dead companies | No EDGAR fundamentals for most delisted members | Name matching against Form 25/15 filers exists (`membership.delisting_dates`), not used in the cohort runs |
| Successor issuers | History split across CIKs | Curated map (`edgar.PREDECESSOR_CIKS`), one entry (Exxon), facts and filing lists merged |

## Kickoff prompt for the next session

> Continue Lighthouse in lotainc0/Stock-Trader-FINTECH-Project. Read `docs/NEXT_SESSION.md` and METHODOLOGY §8-8d and
> §11 first. If the pull request for `claude/ecstatic-wright-3w8w59` was merged, start a new branch from `main`. No
> signal survived and the timing test failed; do not add weight to any signal in `decision.py`, and do not change the
> trading rule, without the user's decision and, for a new signal, a new pre-registered test; never let the watch list
> reach a verdict or ticket. Pick from "Open items" in the order I ask. Keep every rule in "Rules that stay in force".
> Ask me before deleting anything, and before opening or merging a PR.

## Rules that stay in force

* Never delete a file, folder or anything else on the user's device without asking first and getting explicit consent;
  avoid `rm`, even in scratch space.
* Commit and push work in progress after each step (sessions get interrupted). Work on the branch the session is given;
  never push to `main`. Ask the user before opening the PR and again before merging once CI passes.
* Before every push: `LIGHTHOUSE_OFFLINE=1 python -m pytest` (CI runs Python 3.10 and 3.12; 292 tests), with an empty
  `LIGHTHOUSE_CACHE_DIR` so a live cache cannot leak into the offline tests. CI's 3.10 job runs pandas 2 (pandas 3 needs
  Python 3.11): after re-recording data or touching dates, run the suite once under pandas 2 as well (a virtualenv with
  `pandas>=2.3,<3`).
* After any scoring change: `python scripts/cross_ticker_evaluation.py --check`; if numbers change on purpose,
  regenerate and explain them in the docs (METHODOLOGY §8 and §11).
* A test of the product's rules is written down (rule, data, what follows from each outcome) and committed before it
  is run, as §8d was; results are reported whatever they show, and nothing is chosen after the run.
* $0 budget: free sources and free keys only (FRED, Alpaca paper); read terms, respect rate limits, keep large data in
  the gitignored `data/cache/` as Parquet, never commit large data; research universe about 500 names.
* SEC EDGAR: User-Agent with the contact e-mail from `LIGHTHOUSE_SEC_EMAIL` (sent to sec.gov only, never written into
  any file), under 10 requests a second (`lighthouse/http.py` enforces 8). The user approves the e-mail per session.
* No look-ahead: every new signal gets a truncation test; anything filed or published on a date is used from the next
  session. Never invent data. Tests run offline only. Never skip, weaken or quarantine a test.
* Every logged signal's hold-out has been used (`reports/research/holdout_log.csv`); looking again needs
  `--allow-repeat` and counts as a new trial; a changed computation is a new configuration (see
  `fundamental_signals.config_extra`).
* Do not route around network-policy denials or a site's own bot checks (Stooq).
* The user wants automated buy/sell decisions and low-lag indicators; keep the listed defaults unless evidence says
  otherwise, and say so when it does.

## Decisions the user made

* 14 October 2026: short-horizon longs keep the R-target exit; PR #5 (Phases 1-3) opened and merged.
* 15 October 2026: an SEC contact e-mail for that session's runs; up to 2-3 helper agents; the executive summary and
  pitch kept current; PR #6 opened and later merged.
* 15 October 2026 (this branch): open items 1 and 3 with helper agents, item 4 skipped (no local model); the DCF's
  equity premium is the geometric mean since 1963 (arithmetic 7.2% shown as a sensitivity); RONIC = WACC in the bear
  case; REIT FFO per NAREIT; a bank that never tagged preferred stock counts it as 0, flagged.
* 15 October 2026 (same branch, later): **simplify** in three tiers (above); the verdict and the ticket are the rule's
  alone, the fundamentals only in the analyst memo; the analyst memo is part of "Analyse a stock"; the six PNGs of the
  MSFT example deleted and the example rebuilt with HTML charts; the account e-mail approved as the SEC contact for
  that session's live runs (environment only, never in a file).

## Running the research again (a fresh container re-downloads everything)

Point `LIGHTHOUSE_CACHE_DIR` at a scratch folder. Run long jobs from a pinned copy of the code
(`git worktree add --detach <scratch>/runtree <commit>`) so new code cannot change a run halfway, and copy
`reports/research/*.csv` and `reports/examples/cross_ticker_sp500_*.csv` back afterwards. On this environment's network:
Yahoo prices for ~880 tickers ~15 minutes; SEC facts, filing lists and profiles for ~640 companies ~40 minutes; Yahoo
dividend and split histories ~10 a minute (about an hour for the panel); every 10-K since 2006 ~75 minutes.

```bash
python -m lighthouse memo MSFT --source sample --no-peers          # offline memo from the sample pack
python -m lighthouse memo MSFT --open-thesis --peer-norms          # live: peers, peer-median beacons, thesis saved
python -m lighthouse thesis check | list | close ID --reason "..."
python -m lighthouse journal run --universe "Sample pack (offline)" --allow-stale-data
python -m lighthouse journal watch [--due]                         # record the month-end watch snapshot (only then)
python -m lighthouse journal score                                 # verdicts, tickets and the watch-only section
python scripts/cross_ticker_evaluation.py --cohort 1996-01-02      # the cohort test and its pre-registered timing test
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py --no-text            # tuning (cached signals)
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py --signals NAME --holdout   # a new trial
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/build_edgar_sample.py   # re-record the sample pack
```
