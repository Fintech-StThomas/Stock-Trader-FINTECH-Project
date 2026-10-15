# Better data, and where machine learning fits

Lighthouse runs on free data (Yahoo Finance through `yfinance`, Stooq as a fallback, and a bundled offline pack). That is
enough to prototype, but not enough to trust real money to. This page lists what to connect, in priority order, and how
machine learning could be added without fooling ourselves.

Vendor names, plans and prices change often. Check each provider's own pricing page before you subscribe; the notes below
were checked in October 2026.

## 1. What is wrong with the current data

| Gap | Why it matters | Fix |
|---|---|---|
| Yahoo Finance is unofficial and rate-limited | Requests fail without warning; fields change meaning (e.g. `dividendYield` moved from a fraction to a percent) | A paid, documented price API |
| Daily bars only, and Yahoo's quote | Signals use finished sessions and "buy now" is priced from Yahoo's latest quote with its timestamp, which can lag by up to 15 minutes on some exchanges and is not your broker's market | A real-time quote from a broker API or a consolidated (SIP) feed |
| ~~Only ~4–5 years of fundamentals, dated by fiscal period end~~ (done October 2026, free) | The backtest's fundamental gate only acted in recent years, and period end ≠ the date the numbers became public | SEC EDGAR XBRL facts keyed by **filing date** (`lighthouse/edgar.py`, `lighthouse/pit.py`), from 2009 |
| Survivorship bias (partly addressed, free) | Testing only on companies that still exist (and won) flatters every strategy | Historical S&P 500 membership from free sources (`lighthouse/membership.py`) with coverage reported; prices for delisted stocks still need a paid source (Norgate, Sharadar, CRSP), so results are published as bounds |
| ~~Fixed 2% risk-free rate~~ (done October 2026) | Sharpe ratios were overstated when T-bills paid ~4%, understated when they paid ~0% | Live analyses use the 3-month T-bill (FRED `DTB3`) day by day, and uninvested backtest cash earns it; offline and sample data keep 2% |
| No earnings calendar or borrow cost (short interest: partly free) | Stops can be gapped through on earnings; shorts can be expensive or recalled | Earnings dates come from 8-K Item 2.02 filing dates (free, after the fact: no forward calendar); FINRA short interest is free from 2018 and short-sale volume from 2009 (`lighthouse/research/positioning.py`); borrow fees and availability still need a broker or a paid source |

## 2. Recommended sources

### Prices and real-time quotes
* **A broker API you can also trade through.** The quote you decide on is the market you trade in, and orders can be
  placed from the same code. Options: **Alpaca** (commission-free stock API with paper trading; its free real-time feed is
  IEX only, a single exchange with a few percent of volume, and consolidated SIP data needs a paid plan),
  **Interactive Brokers** (TWS / Client Portal API; real-time data needs market-data subscriptions), and the
  **Schwab Trader API – Individual** (market data and order placement for your own account; it uses OAuth tokens that
  must be renewed regularly, which complicates unattended use).
* **Massive (formerly Polygon.io, renamed in late 2025)**: consolidated real-time and historical US stock data over REST
  and WebSocket. It has a free tier (end-of-day and delayed data, a few calls per minute); real-time needs a paid plan.
  The old `api.polygon.io` endpoints keep working.
* **Tiingo**: clean, adjusted end-of-day history back decades, plus intraday data, at individual-friendly prices.
* **Databento**: institutional-grade historical and live data taken directly from exchange feeds, pay as you go. Use
  it for tick-level or order-book research.

### History without survivorship bias, for honest backtests
* **Norgate Data**: US stocks including delisted ones, with historical index constituents (for example, who was in the
  S&P 500 in 2008). Widely used by systematic retail and professional traders.
* **Sharadar (via Nasdaq Data Link)**: survivorship-bias-free US prices plus point-in-time fundamentals, including
  delisted companies.
* **CRSP and Compustat (via WRDS)**: the academic gold standard, usually accessed through a university licence.

### Fundamentals that are point-in-time
* **SEC EDGAR XBRL APIs** (`data.sec.gov`, the `companyfacts` and `frames` endpoints): free and official. Every 10-K and
  10-Q figure comes with its **filing date**, which is exactly what a look-ahead-free backtest needs. The SEC asks for a
  descriptive `User-Agent` header (name and e-mail) and at most 10 requests per second.
* Sharadar SF1 (above); **Financial Modeling Prep** (inexpensive and broad, good for screening; check how it dates figures).

### Macro data, the risk-free rate and factors
* **FRED** (St. Louis Fed, free API key; single series also download as CSV without one, which is how Lighthouse
  reads the 3-month T-bill `DTB3` for its risk-free rate): Treasury yields, credit
  spreads (`BAA10Y`) and the VIX (`VIXCLS`). Useful as regime features too.
* **Kenneth French Data Library**: Fama-French factors and momentum, to check whether a result is genuine or just
  exposure to value, size or momentum.

### Events and short-side data
* Earnings dates (Nasdaq, Zacks, Financial Modeling Prep, Finnhub; `yfinance`'s `Ticker.calendar` is unofficial). The
  trade tickets tell you to check this date because earnings gaps jump straight through stops.
* Short interest from **FINRA** (published twice a month). For borrow availability and fees, ask your broker (Interactive
  Brokers shows both) or use paid sources such as S3 Partners or Ortex.

## 3. How to plug a source in

`lighthouse/data.py` tries sources in order (`cache → yfinance → stooq → stale cache → sample`). Each source is one
function that returns a daily OHLCV `DataFrame`:

1. Write `_from_<vendor>(ticker, start, end)`, reading the API key from an environment variable such as `TIINGO_API_KEY`
   (never hard-code it, and never commit it).
2. Add its name to the `order` list in `load_prices`.
3. Everything downstream (cleaning report, cache, indicators, backtests, trade tickets, portfolios) works unchanged.

Fundamentals follow the same pattern in `load_fundamentals`. Map the vendor's fields onto `FundamentalSnapshot` and the
statements onto `StatementHistory`, with the **filing date** as the index (or move `reporting_lag_days` to 0 once the dates
are true filing dates).

I did not ship adapters for keyed vendors: they cannot be tested without your API keys, and this build's sandbox cannot
reach those hosts. They are about a day's work each once a key is available.

## 4. Machine learning: where it helps and how to do it honestly

Daily data on a few hundred stocks is a small, noisy and shifting dataset. Most "AI stock picker" results are look-ahead
or overfitting. Used carefully, ML is most valuable around the existing rules rather than replacing them.

### The highest-value use: meta-labeling
Keep the Lighthouse rules as the primary signal (they decide *whether* there is a setup and *which side*). Train a model that
predicts the **probability that a setup reaches target 1 before its stop**, and use that probability to size or skip the trade
(López de Prado, *Advances in Financial Machine Learning*, 2018).

The labels already exist. `tradeplan.setup_statistics()` replays every historical setup with the plan's stop, target and
time stop: this is *triple-barrier labelling*. Each row is one trade with its outcome.

* **Features at signal time**: the six technical beacon votes, fundamental group scores, regime flags, realised volatility
  and its trend, distance to the 200-day SMA, days to the next earnings report, sector, market breadth and VIX (from FRED).
* **Models**: start with regularised logistic regression and gradient-boosted trees (LightGBM or XGBoost). Explain them with
  permutation importance or SHAP. Deep networks need far more data than this.
* **Validation**: never shuffle time series. Use *purged k-fold cross-validation with an embargo*, because trades overlap
  in time and a random split leaks the future. Then confirm on a final walk-forward period the model never saw.
* **Judge it by money, not accuracy**: out-of-sample expectancy in R after costs, compared with the rules alone. Also check
  calibration (does a 60% prediction win about 60% of the time?) and the Brier score. Many variants get tried, so deflate
  the Sharpe ratio for multiple testing (Bailey & López de Prado 2014).
* **Prerequisite**: a survivorship-free universe and point-in-time fundamentals (section 2). Without them the model
  mostly learns which companies survived.

### Other good uses
* **Volatility forecasting** (GARCH or HAR-RV models) for sizing and stop distances, instead of trailing realised volatility.
* **Regime detection** (hidden Markov models on index returns and volatility) as an alternative to the 200-day rule.
* **Text features**: sentiment and surprise measures from earnings calls, 8-K filings and news, used as inputs to the
  meta-model rather than as trading signals on their own.
* **Data-quality checks**: anomaly detection on incoming prices and fundamentals (unadjusted splits, bad ticks).

### What a first version would look like
A `lighthouse/ml.py` module that (1) builds the dataset from `setup_statistics` across a universe, (2) implements purged
k-fold CV with an embargo, (3) trains and calibrates a gradient-boosted classifier, and (4) adds an "ML confidence" line to
each trade ticket next to its out-of-sample record. It should ship only if it beats the rules alone out of sample after
costs. It needs `scikit-learn` (and optionally `lightgbm`) as dependencies.

## 5. Which signals survived (October 2026)

Every signal Lighthouse could build from free data went through the same test (`docs/METHODOLOGY.md` §8b): does it rank
the point-in-time S&P 500 by next month's return, after costs, in 2009-2017 (tuning) and again in 2018-2026 (a hold-out
evaluated once), under both delisting rules, with the deflated Sharpe ratio counting every configuration tried?

| Group | Signals | Result |
|---|---|---|
| The product's own scores | six technical beacons, technical score, point-in-time fundamental score, composite | None survives; the technical beacons had negative ICs in 2009-2017 |
| Fundamental (SEC XBRL, dated by filing) | SUE, GP/A, ROIC and its trend, accruals, Piotroski, Altman Z, Beneish, cash conversion, earnings / book / FCF / FCF-after-SBC / EBITDA yields, value vs own history, net issuance, shareholder yield | FCF yield and FCF yield after SBC passed tuning (IC t 2.3, 2.1) and kept their sign out of sample, too weakly to count (t 0.7) |
| Price and size | 12-1 momentum, low volatility, 1-month reversal, 52-week high, size | Size passed tuning (t 2.0) and reversed out of sample |
| Panel | industry momentum, residual momentum | None |
| Positioning (Form 4, FINRA) | insider buyers, insider net purchases, short-sale volume, days to cover | Short-sale volume ranks stocks in both periods but turns over too fast to trade monthly after costs |
| Events and text (8-K, 10-K) | 3-day earnings reaction, earnings-release tone (half the members), 10-K similarity, risk-factor similarity, Loughran-McDonald tone, fog | Risk-factor similarity was significant only in the hold-out; the 10-K tone hypotheses had the wrong sign; the earnings-release tone had no IC out of sample |
| Institutions (13F) | breadth of ownership | None |

So **no signal carries weight in `lighthouse/decision.py`**, and the verdicts stay what they were: a risk-control rule,
not a stock picker. The research harness, its experiment log and the hold-out guard stay in place for the next idea:
a new signal is one more trial, and the forward journal (Phase 2) collects the out-of-sample data no backtest can.

## 6. What free data can and cannot prove

What it can prove:

* **That a rule does not work.** A signal with negative or zero rank correlation over 17 years of the point-in-time
  S&P 500, before any cost, does not work there; free data is enough to say no, and the harness said it 47 times.
* **What a company reported, and when.** SEC XBRL facts carry the filing date of every number, so fundamentals are
  point in time from 2009 (large filers), and the analyst toolkit's memo uses only what was public on its date.
* **Risk.** Volatility, drawdowns and the effect of a regime filter or a stop are measured on prices, which are good
  for the companies that still trade.

What it cannot prove:

* **That a rule works on the whole market.** Yahoo has no prices for most delisted members (238 of 883 S&P 500
  members since 2008), and none of the priced ones stopped trading before today, so the dead are simply missing: every
  cross-sectional and cohort result is a bound. Delisting returns (CRSP) or survivorship-free prices (Norgate,
  Sharadar) would turn bounds into estimates.
* **Small effects after costs in large caps.** Spreads come from daily highs and lows (Corwin-Schultz), which read
  part of the day's volatility as spread (~35 bps for AAPL against a quoted 1-3 bps): a signal worth 1-2% a year after
  realistic costs can look negative here. Every result also shows a flat 10 bps a side; quote or TAQ data would settle it.
* **Anything before 2009 in fundamentals,** or for companies whose filings the ticker table no longer maps (most of the
  dead), or for concepts a company tags under its own taxonomy (some banks' loans and intangibles, some REITs' gains).
* **Borrow costs and short availability.** Assumed (0.5% a year), not measured.
* **Forward-looking events.** Earnings dates are known from 8-K filings after the fact; there is no free forward
  calendar or consensus estimate, so the reverse DCF stands in for expectations.

The honest use of free data is therefore to reject ideas cheaply, to keep every number point in time, and to collect
forward evidence in the journal. Paid data would be worth buying for one thing first: delisted prices and delisting
returns, because the missing dead are the one bias no amount of free data can bound tightly.
