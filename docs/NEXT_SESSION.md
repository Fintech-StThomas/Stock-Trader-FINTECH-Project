# Handoff: where the research-grade Lighthouse work stopped (15 October 2026)

Branch `claude/focused-bohr-klq8zc` (started from `main` at 2ec89f2, after the Phase 1-3 work was merged as pull
request #5) carries Phases 4-8. **The pull request for it has not been opened**: the user decides (ask first, and ask
again before merging once CI passes). The eight-phase plan in `docs/ROADMAP_STATUS.md` is done; what follows is
optional next work.

## What this session found (details: METHODOLOGY §8b and §11, DATA_AND_ML §5-6, ROADMAP_STATUS)

* **No signal survives the pre-registered rule.** The 37 library signals went through tuning (2009-2017) and the
  hold-out (2018-2026, evaluated once; 47 trials with the product's own ten). Three passed tuning (FCF yield, FCF yield
  after SBC, size); in the hold-out the FCF yields kept a positive, insignificant IC (t 0.7) and size reversed. The
  sector/size-neutral blend is built (`lighthouse/research/blend.py`, `research_signals.py --blend`) but there is
  nothing to blend, and `lighthouse/decision.py` is unchanged (`cross_ticker_evaluation.py --check` unchanged).
  The earnings-release tone (`eightk_tone`, half the members) was evaluated last and failed both tests (hold-out IC
  0.000).
* **Near misses worth forward-testing, not weighting**: short-sale volume (IC t 3.5 / 1.9, but 60-66% monthly turnover
  makes it lose after costs) and risk-factor similarity (significant only in the hold-out, t 2.3).
* **Analyst toolkit on live data** (MSFT): peers, peer-median beacons, industry metrics, XBRL segments, thesis tracker
  on a real new 10-K, post-mortem. Two fixes came from it: the DCF's terminal value (value-driver formula; MSFT base
  case $94 -> $199) and the post-mortem's growth rate (annualised over quarter ends).
* **Fixed before the hold-out**: positioning parser bugs (class shares, preferred-stock rows, free-text symbols, FINRA's
  floored days to cover, cached empty answers), same-session look-ahead in the Form 4 and short-volume timing, repeated
  4/A amendments and a typing error, headerless 2010 FINRA files, the text workers' SEC rate (METHODOLOGY §11).

## Kickoff prompt for the next session

> Continue Lighthouse in lotainc0/Stock-Trader-FINTECH-Project. Read `docs/NEXT_SESSION.md`, `docs/ROADMAP_STATUS.md`
> and METHODOLOGY §8b and §11 first. If the pull request for `claude/focused-bohr-klq8zc` was merged, start a new branch
> from `main`. The research plan is done and no signal survived; do not add weight to any signal in `decision.py`
> without a new pre-registered test. Pick from "Open items" below in the order I ask. Keep every rule in "Rules that
> stay in force". Ask me before deleting anything, and before opening or merging a PR.

## Open items (the user's choice; none started)

1. **Forward evidence instead of more backtests.** Record the near misses (risk-factor similarity, FCF yield, short
   volume) in the forward journal as *watch-only* signals, scored by `lighthouse/scorecard.py` on new data. Any
   variant (for example short volume rebalanced quarterly to cut turnover) is a new configuration and counts as a new
   trial in `reports/research/experiments.csv`.
2. **Decisions the code leaves strict, for the user**: the DCF's equity premium (the Ken French arithmetic mean since
   1963, 7.2%; a 5-5.5% premium is common in practice) and RONIC (today's ROIC floored at the WACC); FFO without adding
   back real-estate impairments (NAREIT adds them back); a bank that never tags preferred stock gets no tangible book.
3. **Stale pitch documents**: `docs/EXECUTIVE_SUMMARY.md` (and possibly `docs/PITCH.md`) still quote the timing results
   from before cash earned the T-bill (balanced Sharpe 0.40, "36 automated tests", "statements lagged 90 days"). Ask the
   user whether they are frozen deliverables before editing.
4. **Known data limits** (documented, not fixed): Berkshire's and Brown-Forman's Form 4 symbols (`BRK.A`, `BFA, BFB`)
   do not map to the class in the index; one-off market closures shorten the 5-session short-volume window; a
   transient error on one pre-2018 FINRA facility caches a partial day; `edgar.bulk_quarter` filters concepts at
   download like the facts cache once did; XOM's filing list holds only the new holding company's filings (the facts
   merge the predecessor CIK, the submissions do not).
5. **Local LLM red team** (`memo --red-team`, Ollama): untested live; only if the user runs a local model.

## Rules that stay in force

* Never delete a file, folder or anything else on the user's device without asking first and getting explicit consent;
  avoid `rm`, even in scratch space.
* Commit and push work in progress after each step (sessions get interrupted). Work on the branch the session is given;
  never push to `main`. Ask the user before opening the PR and again before merging once CI passes.
* Before every push: `LIGHTHOUSE_OFFLINE=1 MPLBACKEND=Agg python -m pytest` (CI runs Python 3.10 and 3.12; 285 tests),
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
* 15 October 2026: the SEC contact e-mail for that session's runs; up to 2-3 helper agents allowed for parallel work.

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
streamlit run app.py                                               # tabs: Analyst memo, Coverage, Journal, Research
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py --no-text --no-8k   # tuning (cached signals)
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py --signals NAME --holdout --allow-repeat  # a new trial
python scripts/research_signals.py --blend SIGNAL [SIGNAL ...]     # the neutral blend of survivors (none today)
LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/build_edgar_sample.py   # re-record the sample pack
```
