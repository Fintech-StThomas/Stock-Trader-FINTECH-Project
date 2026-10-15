# Roadmap status: research-grade Lighthouse on free data

Updated as each phase lands, so an interrupted session can pick up where the last one stopped. The plan has eight
phases; the budget is $0 (free data, free libraries, free hosting).

| Phase | Status | What exists |
|---|---|---|
| 1. Free data foundations | **done** | SEC EDGAR client and point-in-time statements by filing date; historical S&P 500 membership with coverage; dividends, splits, total returns and delisting rules; live-mode rule for tickets; survivorship-aware cohort evaluation |
| 2. Forward journal and paper trading | **done** (paper fills untested against a live account: no keys here) | Append-only SQLite journal (`lighthouse/journal.py`, triggers refuse UPDATE/DELETE; schema 2 adds the watch-only research signals and their outcomes, METHODOLOGY §8c) with inputs, data version and code commit; virtual positions replayed with each ticket's rules; scorecard (`lighthouse/scorecard.py`: hit rate vs SPY, follow-the-verdict vs buy-and-hold vs SPY, calibration, Brier and Brier skill); Alpaca paper adapter (`lighthouse/broker.py`: OTO / bracket orders, review exits at the open, reconciliation of fills vs the backtest's open fills); `python -m lighthouse journal run / score / export`; `.github/workflows/journal.yml` after every close, journal kept on the `journal-data` branch, an issue opened when an exit fires |
| 3. Research harness | **done** | `lighthouse/research/`: month-end point-in-time S&P 500 panel (volume screen, coverage against every member), rank IC with Newey-West t, quintiles, long-short and long-only returns after per-stock costs and borrow, turnover, Fama-French 5 + momentum alphas, both delisting rules, tuning 2009-2017 / hold-out 2018+ evaluated once (guarded and logged), deflated Sharpe from the trial count, append-only experiment log (`reports/research/`). Realistic costs in the strategy backtest (liquidity spread and impact, resting stops with gap fills, borrow fee, cash at the T-bill). Product signals tested: **none survives** (METHODOLOGY §8b). Exit plans re-run on the 2009 cohort |
| 4. Signal library | **done** | Fundamental (SUE, GP/A, ROIC and trend, accruals, Piotroski, Altman, Beneish, cash conversion, value vs peers and own history, net issuance, shareholder yield, FCF after SBC), panel (industry and residual momentum), price (12-1 momentum, low volatility, 1-month reversal, 52-week high, size), positioning (insider Form 4, FINRA short volume, days to cover), events (3-day earnings reaction, earnings-release tone on half the members), 10-K text (Lazy Prices, risk-factor similarity, Loughran-McDonald tone, fog), 13F breadth; parsers tested on recorded FINRA and SEC excerpts. Tuning and the hold-out run once (15 October 2026): **none survives**; the sector/size-neutral blend (`research/blend.py`) is built and tested but has no survivors to blend; `decision.py` unchanged (METHODOLOGY §8b) |
| 5. Single-stock analyst toolkit | **done** | `lighthouse/analyst/`: point-in-time financials, linked 3-statement model, WACC (FRED + Ken French), bull/base/bear DCF after SBC with a value-driver terminal value, reverse DCF, own-history valuation bands, SIC + market-value peers, peer medians in the beacons (opt-in, `memo --peer-norms`), industry metrics (banks, REITs, software, insurers), business segments from the 10-K's XBRL, red flags, balance-sheet risk incl. debt maturities, capital allocation, catalyst calendar, investment memo. Run end to end on MSFT with live data; the facts cache refetches when the concept list grows; sample pack re-recorded |
| 6. Senior-analyst layer | **done** (live check on one new 10-K) | Variant view (price-implied vs trend growth), thesis tracker with kill criteria checked on new filings, coverage monitor with journal calibration, conviction scale (never above the risk-rule size), post-mortems (`lighthouse/analyst/thesis.py`, `python -m lighthouse thesis`) |
| 7. Text reading without a paid LLM | **done** (the local LLM is untested live: no Ollama here) | `lighthouse/text.py` deterministic features (used by the 10-K and 8-K signals); `lighthouse/llm.py` optional local Ollama (off by default, citations and number checks, red team); Anthropic adapter a refusing stub |
| 8. App and docs | **done** | App tabs Analyst memo, Coverage, Journal, Research; README, METHODOLOGY, DATA_AND_ML ("which signals survived", "what free data can and cannot prove"), this page and the handoff |

## Phase 1 findings

* **The rule has shown no edge in stock timing.** On every S&P 500 member of 2 January 1996 (258 of 487 with prices)
  the balanced rule's Sharpe averaged 0.13 against buy-and-hold's 0.36 and beat it on 29 stocks; on the members of
  2 January 2009 (330 of 498) 0.23 against 0.48, again 29 stocks. It lowered drawdown on 98% and 95% of them.
  Re-run in Phase 3 with cash at the T-bill, resting stops and liquidity costs: 0.24 (78 of 258 beat buy-and-hold)
  and 0.30 (43 of 330), still well below buy-and-hold.
* **Coverage is the binding gap.** Yahoo prices 53% of the 1996 cohort and 66% of the 2009 cohort; every priced member
  still trades today. Member-day coverage of the index: 36% in 1996, 60% in 2009, 95% in 2026. The delisting rules
  therefore act only at the cohort level (unpriced members excluded, or worth nothing).
* **The fundamental gate with hard-coded sector norms costs return** on the 20 sample stocks once real filing-dated
  fundamentals feed it (Sharpe 0.401 → 0.383).
* **EDGAR is good enough for fundamentals from 2009**: the core line items (revenue, net income, EPS, equity, assets,
  operating cash flow) are reported for 84-100% of the sample stocks' fiscal years; banks report no capex or operating
  income (an EBIT proxy fills the latter, flagged).

## Phase 2 notes

* The first live run (13 October 2026 close) journaled AAPL Buy, MSFT Strong Buy, NVDA Buy (virtual positions opened)
  and JPM Hold. The forecast `p_win` is each stock's setup-test win rate shrunk toward the pooled rate (trend plan 45%,
  R targets 63%) as if the pool were 10 trades: NVDA's 3 wins in 3 trades is a 62% forecast, not 100%.
* A bug found by the live run: Yahoo can list today's date with an empty close shortly after the close; the cache then
  looked current while its last real close was a day old. The cache must now hold the last finished session's close.
* To switch on paper trading, add the repository secrets ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (free paper keys);
  for filing-dated fundamentals, LIGHTHOUSE_SEC_EMAIL. Until then the workflow journals and alerts, and the paper step
  is a dry run that lists the orders it would send.
* Alpaca's paper adapter is tested against the request and response formats in Alpaca's documentation, not a recorded
  session (no keys in this environment).

## Data gaps that remain (free data)

| Gap | Cost | Status |
|---|---|---|
| Prices of delisted stocks | The untested part of every cohort is mostly its failures; results are bounds, not estimates | Reported as coverage; both delisting bounds published |
| Ticker reuse in Yahoo | A recycled ticker's history can belong to another company | Partly caught: the research panel drops months with volume on fewer than 15 of 21 sessions (CPWR, EP, PARA, SII, GENZ: 115 member-months); a reused ticker that trades normally is not detected |
| Fundamentals before 2009 | Unknown, gate off | By design |
| Ticker → CIK for dead companies | No EDGAR fundamentals for most delisted members | Name matching against Form 25/15 filers exists (`membership.delisting_dates`) but is not used in the cohort runs |
| Successor issuers (holding-company reorganisations) | History split across CIKs | Curated map, one entry (Exxon) |

## Phase 3 findings (14 October 2026)

* **No signal the product uses ranks S&P 500 stocks.** Tuning 2009-2017: every technical beacon, the technical and
  composite scores had negative ICs (t -1.0 to -2.3) and negative gross long-short spreads (-1% to -10% a year), so
  the verdict does not depend on costs. Hold-out 2018 on: ICs near zero, all long-only returns after costs negative
  except the point-in-time fundamental score (+0.4% a year, t 0.25). None passes the pre-registered rule; nothing in
  `decision.py` changes. Full table: METHODOLOGY §8b; files: `reports/research/`.
* **Coverage**: 643 of 881 members since 2008 have Yahoo prices; tested share 70% of member-months in 2009-2017, 89%
  after 2018. No priced member stopped trading before today, so the delisting rules give identical results: the dead
  are simply missing. Bounds, not estimates, for anything that depends on failures.
* **Costs are conservative for large caps**: Corwin-Schultz from daily ranges gives ~35 bps full spread for AAPL/MSFT;
  results also show a flat 10 bps a side. A zero-volume series (CPWR) had made costs explode until the volume screen.
* **Exit plans, 2009 cohort (329 of 498 priced)**: the trend exit beat R-targets for long-horizon longs (avg +1.62R vs
  +0.48R, better on 268 of 326 stocks: the default stays) and for short-horizon longs (+0.62R vs +0.13R, 280 of 329:
  the user decided to keep targets for the short horizon, because a survivor-only cohort favours letting winners run;
  docs/NEXT_SESSION.md). Shorts lost
  under both (-0.29R targets, -0.52R trend): a survivor-only, mostly rising cohort is the worst case for shorts.

## Phase 4 findings (15 October 2026)

* **No library signal survives the pre-registered rule.** 37 signals, 47 trials with the product's own. Three passed
  tuning (FCF yield, FCF yield after SBC, size); in the hold-out the FCF yields kept a positive but insignificant IC
  (+0.009, t 0.7) and size reversed. The earnings-release tone (8-K, half the members) failed both tests. Deflated Sharpe ratios about 0. No blend, no weights in `decision.py`. Full table:
  METHODOLOGY §8b; files: `reports/research/`.
* **Short-sale volume ranks stocks but cannot pay its costs** at monthly rebalancing (IC t 3.5 / 1.9; turnover 60-66% a
  month; -6.5% a year net). **Risk-factor similarity** was significant only in the hold-out (t 2.3): a forward-journal
  candidate, not a survivor.
* **Fixed before the hold-out**: five positioning parser bugs found by tests on real FINRA and SEC excerpts, a few
  hours of same-session look-ahead in the Form 4 and short-volume timing, repeated 4/A amendments, a $1.56 trillion
  typing error, headerless 2010 FINRA files, and the text workers' share of the SEC rate (METHODOLOGY §11). After it: the
  deflated Sharpe's variance of trials now comes from the whole log (a single-signal run had used its own).
* **Coverage**: 645 of 883 members since 2008 priced; 80-90% of member-months tested in the hold-out (Altman Z 46%,
  GP/A 62%), 23-71% in tuning (fundamental signals lowest).

## Phase 5-7 notes (15 October 2026)

* MSFT memo end to end on live data: peers (SIC 7372), peer-median beacons, software metrics (RPO $684bn, 2.1x revenue,
  +82% a year), segments from the FY2026 10-K (Intelligent Cloud $137.8bn, +29.7%), thesis saved and checked. It found a
  DCF flaw: the terminal value carried the build-out capex (35% of revenue) forever; now a value-driver terminal value
  (base case $94 -> $199 a share at an 11.5% WACC; the price implies 36.6% growth a year for five years).
* Thesis tracker on a real new filing: a thesis opened as of 1 May 2026 saw the 29 July 10-K, no kill criterion broke,
  and the post-mortem's revenue growth is now annualised over quarter ends (it read +9.6% instead of about +18%).
* Industry metrics depend on standard us-gaap tags: WFC, BAC (2020-22) and JPM (2018-23) lack some bank lines, PLD and
  SPG have no gains-on-sale tag (no FFO), ALL stopped tagging losses in 2024.
* The local LLM red team (`memo --red-team`, Ollama) was not run: no local model in this environment; nothing it writes
  feeds a score.

## Seventh session notes (15 October 2026, branch `claude/ecstatic-wright-3w8w59`)

* **Forward watch list** (open item 1; METHODOLOGY §8c): `fcf_yield`, `short_volume` and `risk_similarity` are recorded
  at each month end for every point-in-time member under their logged configuration hashes, in two append-only journal
  tables (schema 2), scored by the harness's own code on snapshots written before the next open only; `journal watch`,
  `.github/workflows/watch.yml` (needs `main`; first snapshot after the close of 30 October 2026) and the Journal tab.
  Watch-only: no verdict or ticket reads it (tested). At their hold-out ICs a Newey-West t of 2 is about 7 years away
  for risk-factor similarity, 10 for short volume and decades for FCF yield.
* **The user's valuation decisions** (open item 2): equity premium = the geometric mean of the Ken French market factor
  since 1963 (6.2%; arithmetic 7.2% shown as a sensitivity); RONIC = WACC in the bear case; REIT FFO per NAREIT; a bank
  that never tagged preferred stock counts it as 0, flagged. The live MSFT memo was not re-run.
* **Five data limits fixed** (open item 3; METHODOLOGY §11, seventh session): class-share Form 4s by issuer CIK, the
  NYSE one-off closures, FINRA days cached only when complete, the bulk data-set cache's tag record, predecessor filing
  lists (XOM). The §8b results predate them and were not re-run (a re-run is a new trial).
* **The memo's market value after a split** fixed (the cover-page count is split-adjusted to the memo's date); the
  research signals keep the same flaw as part of their logged trials, a decision left to the user.

## Next

See `docs/NEXT_SESSION.md` for the kickoff prompt, the order of work and the open decisions.
