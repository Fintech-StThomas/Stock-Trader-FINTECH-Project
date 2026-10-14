# Roadmap status: research-grade Lighthouse on free data

Updated as each phase lands, so an interrupted session can pick up where the last one stopped. The plan has eight
phases; the budget is $0 (free data, free libraries, free hosting).

| Phase | Status | What exists |
|---|---|---|
| 1. Free data foundations | **done** | SEC EDGAR client and point-in-time statements by filing date; historical S&P 500 membership with coverage; dividends, splits, total returns and delisting rules; live-mode rule for tickets; survivorship-aware cohort evaluation |
| 2. Forward journal and paper trading | **done** (paper fills untested against a live account: no keys here) | Append-only SQLite journal (`lighthouse/journal.py`, triggers refuse UPDATE/DELETE) with inputs, data version and code commit; virtual positions replayed with each ticket's rules; scorecard (`lighthouse/scorecard.py`: hit rate vs SPY, follow-the-verdict vs buy-and-hold vs SPY, calibration, Brier and Brier skill); Alpaca paper adapter (`lighthouse/broker.py`: OTO / bracket orders, review exits at the open, reconciliation of fills vs the backtest's open fills); `python -m lighthouse journal run / score / export`; `.github/workflows/journal.yml` after every close, journal kept on the `journal-data` branch, an issue opened when an exit fires |
| 3. Research harness | **done** | `lighthouse/research/`: month-end point-in-time S&P 500 panel (volume screen, coverage against every member), rank IC with Newey-West t, quintiles, long-short and long-only returns after per-stock costs and borrow, turnover, Fama-French 5 + momentum alphas, both delisting rules, tuning 2009-2017 / hold-out 2018+ evaluated once (guarded and logged), deflated Sharpe from the trial count, append-only experiment log (`reports/research/`). Realistic costs in the strategy backtest (liquidity spread and impact, resting stops with gap fills, borrow fee, cash at the T-bill). Product signals tested: **none survives** (METHODOLOGY §8b). Exit plans re-run on the 2009 cohort |
| 4. Signal library | **code built and unit-tested, not yet evaluated** | Fundamental (SUE, GP/A, ROIC and trend, accruals, Piotroski, Altman, Beneish, cash conversion, value vs peers and own history, net issuance, shareholder yield, FCF after SBC), panel (industry and residual momentum), price (12-1 momentum, low volatility, 1-month reversal, 52-week high, size), positioning (insider Form 4, FINRA short volume, short interest / days to cover), events (3-day earnings reaction), 10-K text (Lazy Prices similarity, risk-factor similarity, Loughran-McDonald tone, fog), 13F breadth. Signals cached; the tuning and hold-out runs, the sector/size-neutral blend and any decision.py weights are next session's work |
| 5. Single-stock analyst toolkit | **built, tested offline** | `lighthouse/analyst/`: point-in-time financials, linked 3-statement model, WACC (FRED + Ken French), bull/base/bear DCF after SBC, reverse DCF, own-history valuation bands, SIC + market-value peers, red flags, balance-sheet risk incl. debt maturities, capital allocation, catalyst calendar, investment memo (`python -m lighthouse memo`). Not yet: segments, industry metrics, peer norms in the beacons, MSFT end-to-end on live data |
| 6. Senior-analyst layer | **built, tested offline** | Variant view (price-implied vs trend growth), thesis tracker with kill criteria checked on new filings, coverage monitor with journal calibration, conviction scale (never above the risk-rule size), post-mortems (`lighthouse/analyst/thesis.py`, `python -m lighthouse thesis`) |
| 7. Text reading without a paid LLM | **built, tested offline** | `lighthouse/text.py` deterministic features; `lighthouse/llm.py` optional local Ollama (off by default, citations and number checks, red team); Anthropic adapter a refusing stub |
| 8. App and docs | **partly** | App tabs Analyst memo, Coverage, Journal, Research (st.fragment). README / METHODOLOGY / DATA_AND_ML updates for Phases 4-7, the handoff page and the PR are pending |

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
  the short-horizon default is still targets; see "Decisions for the user" in docs/NEXT_SESSION.md). Shorts lost
  under both (-0.29R targets, -0.52R trend): a survivor-only, mostly rising cohort is the worst case for shorts.

## Next

See `docs/NEXT_SESSION.md` for the kickoff prompt, the order of work and the open decisions.
