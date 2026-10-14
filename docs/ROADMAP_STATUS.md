# Roadmap status: research-grade Lighthouse on free data

Updated as each phase lands, so an interrupted session can pick up where the last one stopped. The plan has eight
phases; the budget is $0 (free data, free libraries, free hosting).

| Phase | Status | What exists |
|---|---|---|
| 1. Free data foundations | **done** | SEC EDGAR client and point-in-time statements by filing date; historical S&P 500 membership with coverage; dividends, splits, total returns and delisting rules; live-mode rule for tickets; survivorship-aware cohort evaluation |
| 2. Forward journal and paper trading | not started | |
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
