# Roadmap status: research-grade Lighthouse on free data

Updated as each phase lands, so an interrupted session can pick up where the last one stopped. The plan has eight
phases; the budget is $0 (free data, free libraries, free hosting).

| Phase | Status | What exists |
|---|---|---|
| 1. Free data foundations | **done** | SEC EDGAR client and point-in-time statements by filing date; historical S&P 500 membership with coverage; dividends, splits, total returns and delisting rules; live-mode rule for tickets; survivorship-aware cohort evaluation |
| 2. Forward journal and paper trading | **done** (paper fills untested against a live account: no keys here) | Append-only SQLite journal (`lighthouse/journal.py`, triggers refuse UPDATE/DELETE) with inputs, data version and code commit; virtual positions replayed with each ticket's rules; scorecard (`lighthouse/scorecard.py`: hit rate vs SPY, follow-the-verdict vs buy-and-hold vs SPY, calibration, Brier and Brier skill); Alpaca paper adapter (`lighthouse/broker.py`: OTO / bracket orders, review exits at the open, reconciliation of fills vs the backtest's open fills); `python -m lighthouse journal run / score / export`; `.github/workflows/journal.yml` after every close, journal kept on the `journal-data` branch, an issue opened when an exit fires |
| 3. Research harness | not started | |
| 4. Signal library | not started | |
| 5. Single-stock analyst toolkit | not started | |
| 6. Senior-analyst layer | not started | |
| 7. Text reading without a paid LLM | not started | |
| 8. App and docs | not started | |

## Phase 1 findings

* **The rule has shown no edge in stock timing.** On every S&P 500 member of 2 January 1996 (258 of 487 with prices)
  the balanced rule's Sharpe averaged 0.13 against buy-and-hold's 0.36 and beat it on 29 stocks; on the members of
  2 January 2009 (330 of 498) 0.23 against 0.48, again 29 stocks. It lowered drawdown on 98% and 95% of them.
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
| Ticker reuse in Yahoo | A recycled ticker's history can belong to another company | Not detected; documented |
| Fundamentals before 2009 | Unknown, gate off | By design |
| Ticker → CIK for dead companies | No EDGAR fundamentals for most delisted members | Name matching against Form 25/15 filers exists (`membership.delisting_dates`) but is not used in the cohort runs |
| Successor issuers (holding-company reorganisations) | History split across CIKs | Curated map, one entry (Exxon) |

## Next

Phase 2: the forward decision journal and paper trading, then the research harness (Phase 3) with costs and cash at the
T-bill rate, which changes the cohort numbers above (cash earned 0% in them).
