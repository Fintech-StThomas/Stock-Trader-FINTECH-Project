# Lighthouse methodology — every choice, and why

This document exists so that the team can defend every number in the product. It follows the order of the pipeline.

## 1. Data

| Item | Choice | Why |
|---|---|---|
| Prices | Daily OHLCV, dividend- and split-adjusted, from Yahoo Finance (`yfinance`), fallback Stooq CSV | Free, no API key, adjusted series avoid false "crashes" at ex-dividend/split dates |
| History | 10 years by default (15 in the notebook), plus a 1.2-year warm-up so the 200-day SMA and 52-week high are defined on day one | Covers at least one bear market; indicators need 252 days |
| Benchmark | S&P 500 index (`^GSPC`) | The market the customer compares against; used for regime detection |
| Fundamentals | Yahoo snapshot (trailing/forward P/E, PEG, P/B, EV/EBITDA, ROE, margins, D/E, current ratio, FCF, growth) plus annual and quarterly statements | One valuation ratio and one quality/growth measure are required; we use several so a missing field never breaks the score |
| Fundamentals, point in time | SEC EDGAR XBRL (`companyfacts` API, the quarterly form indexes and the bulk Financial Statement Data Sets; `lighthouse/edgar.py`, `lighthouse/pit.py`). Every value keeps the filing that first reported it and that filing's date; quarters come from 3-month values or year-to-date differences; a value first public in a later filing (a restatement, a comparative column) is not used for the earlier date. The SEC's User-Agent rule (a contact e-mail, from `LIGHTHOUSE_SEC_EMAIL`) and its 10 requests/second limit are enforced in `lighthouse/http.py` | Free and official; dates every number by when investors could read it. XBRL starts in 2009 (large filers), so earlier fundamentals are unknown and the backtest's gate is off there |
| Corporate actions | Dividends and splits by ex-date (Yahoo `actions`, cached; recorded in the sample pack). `lighthouse/total_return.py` turns a total-return close back into the price-only close (the exact inverse of Yahoo's adjustment, checked to 1e-6) and restates per-share EDGAR numbers in the price series' share units | A P/E needs price and EPS in the same units: AAPL's FY2012 EPS of $44.15 is $1.58 in today's shares |
| Universe, point in time | Historical S&P 500 membership since 1996 (`lighthouse/membership.py`): fja05680/sp500 on GitHub, brought up to date with Wikipedia's change log and checked against Wikipedia's current list; every universe reports the share of member-days that have prices (coverage) | Testing today's members tests the survivors. Yahoo has no history for most delisted stocks: coverage of the members is 36% in 1996, 60% in 2009, 95% in 2026 |
| Live-mode rule | Trade tickets are refused from bundled sample data, a stale cache, a past date or data missing a finished NYSE session, unless the investor overrides it explicitly (CLI `--allow-stale-data`, the app's checkbox, `LIGHTHOUSE_ALLOW_STALE_DATA=1`); an overridden ticket says "Not for trading" | A ticket is an order; the market clock knows the NYSE holidays and the one-off closures since 1990 (`data.NYSE_SPECIAL_CLOSURES`; a future one has to be added when it is announced) |
| Cleaning | Sort, de-duplicate dates, drop rows without a positive close, fill missing O/H/L from the close, enforce High ≥ max(O,C) and Low ≤ min(O,C), disable volume if >50% missing, flag >50% one-day moves | Each step is counted and shown to the user (`CleaningReport`) |
| Offline pack | Real data shipped in open-source packages (Microsoft OHLCV 1986-2017 from `pmdarima`, 19 S&P stocks and the index 1990-2022 from `skfolio`) | The notebook and tests must run end-to-end without internet; provenance in `scripts/build_sample_pack.py` |
| Caching | Last live download cached in `data/cache/` for at most 20 hours, and (in `auto`) only until the next session close (16:00 New York) after the download; files are written atomically and each ticker is downloaded by one thread at a time | Yahoo rate-limits; the product stays usable; a finished session is never missing from the cache; parallel loads never read a half-written file |
| Unfinished sessions | During market hours today's daily bar is left out of every indicator, and never cached; a note says so | Its close, range and volume are not final (a partial day's volume reads as a collapse), and the rules were tested on finished days |
| Risk-free rate | The 3-month Treasury bill (FRED `DTB3`, secondary market, daily) from FRED's public CSV endpoint (no API key), cached like prices; a day without a value takes the last published one. Used for the Sharpe and Sortino ratios of every backtest and walk-forward test on current prices; the portfolio builder's rate input defaults to the latest value. Offline and on the sample pack: a fixed 2% | T-bills paid ~0% in 2020-21 and ~4-5% in 2023-26: a fixed 2% overstated Sharpe ratios in high-rate years and understated them near zero. Over the ten years to October 2026 it averaged 2.4%, so live Sharpe ratios fall by ~0.02-0.03 (MSFT's rule 0.74 → 0.71). Cash in the backtests still earns 0% |
| Order price | "Buy now" tickets and the position plan are priced from the feed's latest quote with its timestamp and market state (Yahoo chart metadata: `regularMarketPrice`, `regularMarketTime`, trading periods); the last close when there is none, when it is older than the data, or when it is more than 25% from the close (a split or a bad tick) | The decision is made on finished sessions, the order on the price you can trade at. Yahoo quotes can lag by up to 15 minutes on some exchanges |

## 2. Fundamental beacons

Each beacon votes in [−1, +1]. Group score = mean of available beacons; fundamental score = 50 + 50 × (0.40 valuation + 0.35 quality + 0.25 growth), re-normalised when a group is empty.

| Beacon | Rule | Reasoning |
|---|---|---|
| P/E (trailing) | +1 at half the sector-norm P/E, −1 at 1.5×, −0.75 if earnings are negative | Relative valuation: a tech stock at 25× is normal, an energy stock at 25× is expensive |
| P/E (forward) | same vs 0.9× the sector norm | Forward multiples run lower than trailing |
| PEG | +1 at 0.5, 0 at 1.5, −1 at 2.5 | Growth-adjusted price; the classic "PEG ≈ 1 is fair" |
| P/B | +1 at 0.4× norm, −1 at 1.6× norm | Matters most for financials and asset-heavy firms; secondary for asset-light |
| EV/EBITDA | +1 at half norm, −1 at 1.5× norm; skipped for financials | Capital-structure-neutral valuation |
| Earnings stability | coefficient of variation of annual EPS: +0.5 at 0, 0 at 35%, −1 at 70% | A P/E on erratic earnings is a weak anchor |
| ROE | 0 at 10%, +1 at 25%, −1 at −5% | 10% ≈ cost of equity; >20% is a compounding business |
| Operating margin | 0 at the sector norm, ±1 at ±15 pts | Pricing power / cost advantage, sector-relative |
| Debt/Equity | +1 at 0, 0 at 1, −1 at 2; skipped for financials | Leverage amplifies losses and refinancing risk |
| FCF yield | 0 at 3%, +1 at 6%, −1 at 0% | Cash actually generated relative to price |
| Current ratio | −1 at 0, 0 at 1, +0.5 at 1.5 | Liquidity check, capped so it cannot dominate |
| Revenue growth | 0 at 5%, +1 at 20%, −1 at −10% | 5% ≈ nominal GDP growth |
| EPS growth | 0 at 8%, +1 at 28%, −1 at −12% | 8% ≈ long-run market EPS growth |

Sector norms (`SECTOR_NORMS`) are approximate long-run medians for the 11 GICS sectors and are explicitly assumptions the user can edit. Growth uses annual statements first (stable) and Yahoo's quarterly year-on-year figures as a fallback.

**Point-in-time history.** For the backtest, each annual statement becomes usable on the session after its SEC filing
date (EDGAR XBRL, from 2009); a fiscal year applies until a later one has been filed. Statements without filing dates
(Yahoo's, when EDGAR is not used) become usable 90 days after their period end (the 10-K deadline for large filers is
60 days). Quality and growth are fixed for the year; the P/E is recomputed daily from the price-only close and the
filed EPS restated in the price series' share units (a filing-dated EPS without split history is not compared with the
price at all). Before the first statement the score is "unknown" and no gate is applied. Live analyses use EDGAR when
`LIGHTHOUSE_SEC_EMAIL` is set; the sample pack carries the 20 stocks' recorded filings, so the offline backtests use
them too (they used to have fundamentals for MSFT only, from 2013).

## 3. Technical beacons

All indicators are causal (rolling windows, `ewm(adjust=False)`): the value on day *t* uses only data up to *t*. A unit test truncates the series and checks that nothing in the past changes.

| Beacon | Formula | Reasoning | Weight long / short |
|---|---|---|---|
| Trend | 0.5·clip((Close/SMA200 − 1)/10%) + 0.5·clip((SMA50/SMA200 − 1)/5%) | Continuous Golden/Death Cross: fewer whipsaws than a binary cross | 35% / 20% |
| RSI(14) | linear from −1 at 30 to +1 at 70, then fading (+1 → −1 from 70 to 100) | Confirms momentum; extremes signal exhaustion | 15% / 20% |
| MACD(12/26/9) | histogram / ATR, clipped | Momentum acceleration, scale-free | 10% / 20% |
| Price bands %B (trend filter, 2σ over 20 sessions) | 2·(%B − 0.5) inside the bands, fading above 1 / below 0 | Where price sits around its low-lag centre line; above the band = over-extended | 10% / 15% |
| Volume | 0.5·sign(20-day return)·clip((vol20/vol60 − 1)/0.5, 0, 1) + 0.5·sign(OBV − SMA20(OBV)) | A move on expanding volume is more trustworthy; excluded (grey) when volume is unavailable | 10% / 15% |
| 52-week high | +1 at the high, 0 at 20% below, −1 at 40% below | Breakout proximity; momentum literature | 20% / 10% |

Technical score = 50 + 50 × weighted mean, with weights re-normalised over available beacons. Long-term weights lean on slow beacons (trend, breakout); short-term on oscillators.

**Price bands: low lag instead of Bollinger's.** Classic Bollinger Bands are a 20-day simple moving average ± 2σ of the
same closes. A trailing 20-day average lags price by (20 − 1) / 2 = 9.5 sessions, so the bands trail every move. Lighthouse
centres its bands on **Holt's linear trend filter** (Holt 1957; double exponential smoothing, which is also the steady
state of a Kalman filter for a local linear trend). Each session it projects yesterday's level along yesterday's slope,
then moves the level 40% of the way to today's close and the slope 10% of the way to the level's latest change
(`band_alpha` 0.4, `band_beta` 0.1). Because it carries the slope forward it follows a straight-line trend with no lag at
all, where any moving average falls behind. The bands sit ± 2 standard deviations of the closes around that centre over
the last 20 sessions. They are causal (a recursion over past closes; nothing is shifted forward; the truncation test
covers them), and the *price bands* beacon scores %B on them exactly as it scored Bollinger's %B.

How far behind each candidate centre runs, on the 20 sample stocks over their last 15 years
(`scripts/band_lag_evaluation.py` rebuilds this table):

| Centre | Best-matching lag | Turns after price turns | Half of a jump covered after | Overshoot after a jump | Lag in a steady trend | Roughness |
|---|---|---|---|---|---|---|
| SMA 20 (Bollinger) | 9.5 | 5.7 | 9 | 0% | 9.5 | 0.20 |
| Hull MA 20 (used until October 2026) | 2.6 | 3.2 | 3 | 26% | 0.7 | 0.42 |
| Hull MA 10 | 1.2 | 1.5 | 1 | 24% | 0.3 | 0.57 |
| Linear-regression end point, 20 | 2.4 | 3.0 | 2 | 30% | 0.0 | 0.40 |
| EMA 5 | 1.2 | 1.1 | 1 | 0% | 2.0 | 0.42 |
| Holt 0.3 / 0.1 | 1.3 | 1.5 | 1 | 12% | 0.0 | 0.43 |
| **Holt 0.4 / 0.1 (Lighthouse)** | **0.9** | **1.0** | **1** | **9%** | **0.0** | **0.50** |
| Holt 0.5 / 0.1 | 0.1 | 0.7 | 0 | 6% | 0.0 | 0.57 |

All lags are in sessions. *Best-matching lag*: the shift k for which the centre on day t sits closest to the close of day
t − k. *Roughness*: the standard deviation of the centre's daily change over the close's (1 = as jumpy as the price).
The Holt filter at 0.4 / 0.1 is the only candidate that is about one session behind on every measure, with no lag in a
trend and less overshoot than the Hull MA (which swings 26% past a jump before settling). The cost is a line about 20%
rougher than the Hull MA's. Going further (0.5) puts the centre almost on the price (half of it is today's close), so
the bands measure little more than today's move, and the short-horizon backtest got worse (below).

Each centre change was tested on the score before it was made. Hull MA 20 → Holt 0.4 / 0.1, same rules otherwise:

| Test | Hull MA 20 | Holt 0.4 / 0.1 |
|---|---|---|
| Cross-ticker test, balanced, 1990-2022 (20 stocks), average Sharpe | 0.399 | 0.401 (7 better, 4 worse, 9 unchanged) |
| Same, stocks with a higher Sharpe than buy-and-hold | 3 of 20 | 4 of 20 (AAPL) |
| Long-horizon rule, last 15 years, average Sharpe | 0.322 | 0.322 (1 better, 1 worse) |
| Short-horizon rule, last 15 years, average Sharpe | −0.127 | −0.137 (9 better, 11 worse) |
| Holt 0.5 / 0.1 instead (for comparison) | | long 15 y 0.321, short −0.147 |

On returns the change is again a wash; the gain is a chart and a beacon that describe where the price is today. The MSFT
example stays a Buy (71.1 → 71.3) with the same trades. The step before (October 2026): Bollinger's SMA centre → Hull MA
20 moved the cross-ticker Sharpe 0.394 → 0.399 and the 15-year rules 0.323 → 0.322 (long) and −0.134 → −0.127 (short).
The classic bands were removed in October 2026; `indicators.hull_ma` stays for comparisons.

## 4. Market weather (regime)

* Trend: S&P 500 above / below its 200-day SMA → Bull / Bear.
* Volatility: 21-day realised vol above 1.25× its trailing 252-day median → Turbulent.
* Effects: Bear −8 score points, entry threshold +10, new positions ×0.5. Turbulent −3 points. Bear & Turbulent: new entries blocked.

Reasoning: the 200-day rule is the simplest widely studied market-timing filter (Faber 2007); volatility clustering means turbulence persists, and most of the worst days happen below the 200-day line. On Microsoft 2002-2017 the regime filter raised the signal rule's CAGR from 4.4% to 5.8% and its Sharpe from 0.22 to 0.29 with fewer trades.

## 5. Decision and position plan

* Composite = 0.60·F + 0.40·T (long) or 0.30·F + 0.70·T (short) − weather penalty. Missing fundamentals → technical only, with a warning.
* Verdict bands: ≥72 Strong Buy, ≥60 Buy, ≥45 Hold, ≥32 Reduce, else Sell. Bands are symmetric around 50 with a wider Hold zone so small changes do not flip the verdict.
* Exit level: long horizon = 200-day SMA × (1 − 8%); short horizon = 2.5× ATR(14) trailing (2.0 aggressive, 3.0 conservative). The live plan and the backtest use the same exit, so what the user sees is what was tested.
* The investor gives **cash available to invest now** and the value of stocks already held. Risk limits (max loss, concentration
  cap) are a % of the whole account (cash + holdings), the professional convention; a new position is paid from cash only,
  so the cash is a hard cap ("cash available" becomes the binding rule when it is the smallest).
* Size = min(max-loss rule, volatility rule, concentration cap) × conviction × regime multiplier, capped by the cash:
  * Max-loss rule: position % = max loss % / distance to the exit level. The investor's number, not ours.
  * Volatility rule: target vol (10 / 15 / 25%) / 60-day realised vol.
  * Concentration cap: 10 / 20 / 35%.
  * Conviction: 0 at score 45, 1 at 80.

## 6. Backtest protocol

* Long-only, one stock, capital = the "sleeve" allocated to it.
* Review day: last trading day of the month (long) or week (short). Daily reviews were tested and rejected (more trades, lower Sharpe).
* Enter when trend beacon > 0, technical score ≥ 60 (+10 in a bear market), not Bear/Turbulent, point-in-time fundamental score ≥ 35 if known.
* Exit when score ≤ 35 (long) / 40 (short), close < 200-SMA × 0.92 (long) / 0.97 (short), or the ATR stop (short, checked daily).
* Size: volatility target / 60-day realised vol × regime multiplier, re-balanced on review days when the target moves by more than 10 points.
* Fills at the next day's open (close when no open data). Costs per side (`cost_model = "liquidity"`, `lighthouse/costs.py`):
  5 bps commission + the Corwin-Schultz half spread estimated from the last 21 sessions' highs and lows (floored at
  5 bps, which is what close-only data such as most of the sample pack gets) + square-root market impact
  (daily volatility × √(order / average daily dollar volume)), the order being the position's share of the account.
  `cost_model = "fixed"` restores a flat 10 bps.
* Stops are resting orders (`resting_stops`): filled during the day at the stop price, or at the open when the stock
  gaps through it (the setup test already worked this way). Long horizon: the ticket's stop order, the 200-day SMA
  minus 8% at entry, never moved; short horizon: the ATR trailing stop, raised at each close.
* Uninvested cash earns the risk-free rate (`cash_yield`): the 3-month T-bill day by day on live data, the fixed 2%
  offline. Sharpe ratios subtract the same rate, so a rule is no longer charged for cash that earned nothing.
* Metrics: total return, CAGR, volatility, Sharpe and Sortino (risk-free rate: the 3-month T-bill day by day, FRED `DTB3`, when the prices are current; a fixed 2% for offline and sample data, so the published tables stay reproducible), max drawdown, Calmar, trades, win rate, average win/loss, profit factor, exposure; plus layer attribution, sensitivity grid (entry 50-70 × exit 25-45), walk-forward (tune on 60%, test on 40%) and regime breakdown.
* No look-ahead: unit-tested by re-running on truncated data and comparing equity curves.

## 7. What we tested and rejected (design-phase experiments, 12-16 stocks, 1990-2022)

| Variant | Result | Decision |
|---|---|---|
| 2.5× ATR trailing stop on the long-term rule | 380 trades per stock, Sharpe 0.17 vs 0.41 without | Rejected for long horizon; kept (wider, 2× factor) for short horizon as the max-loss anchor |
| Daily review | more trades, lower Sharpe than weekly; monthly best | Monthly (long) / weekly (short) |
| Exit buffer 3% vs 8% below the 200-SMA | 8% fewer whipsaws, higher win rate | 8% long / 3% short |
| Exit score 45 vs 35 | 35 better on every metric | 35 |
| "Don't chase" filter (skip entries >15% above the SMA) | lower Sharpe | Rejected |
| Death-cross exit (trend beacon < −0.5) | no improvement over the SMA buffer | Rejected |
| Position size frozen at entry with 20-day vol | large size errors after volatile entries | 60-day vol, monthly re-balance |

## 8. Results and interpretation

See `reports/examples/cross_ticker_evaluation.csv` (20 stocks, 1990-2022). Rebuilt by `scripts/cross_ticker_evaluation.py`. Averages: buy-and-hold CAGR 13.3%, vol 33.5%, Sharpe 0.50, max DD −69%; Lighthouse balanced CAGR 7.3%, vol 12.2%, Sharpe 0.47, max DD −27%; conservative vol 8.5%, max DD −18%. Lower drawdown and volatility in 20/20 stocks; higher Sharpe in 9/20.

What changed the published numbers this session, one step at a time (20 sample stocks, balanced, averages):

| Step | CAGR | Sharpe | Max drawdown | Beat buy-and-hold (Sharpe) |
|---|---|---|---|---|
| Published before this session (fundamentals for MSFT only) | 6.5% | 0.401 | −32% | 4 of 20 |
| Fundamental gate from SEC filings on all 20 stocks | 6.2% | 0.383 | −31% | 3 |
| + uninvested cash earns the risk-free rate (2% offline) | 7.4% | 0.476 | −27% | 9 |
| + stops as resting orders (71 stop-order exits) | 7.3% | 0.471 | −27% | 9 |
| + liquidity-scaled costs (now) | 7.3% | 0.470 | −27% | 9 |

The largest change is the cash: the rule spends much of its time partly or wholly in cash, which used to earn nothing
while the Sharpe ratio subtracted 2% a year for it. Liquidity costs barely move these numbers because 19 of the 20
sample series are close-only (no range, so the spread estimate falls to its 5 bps floor); on live data with highs, lows
and volume they matter more.

Since the fourth October 2026 session the backtest dates every fundamental by its SEC filing, so the point-in-time
fundamental gate (score ≥ 35 to enter) now acts from 2009 on all 20 stocks instead of MSFT alone. It cost return:
average Sharpe 0.401 → 0.383 (4 stocks better, 11 worse, 5 unchanged), 4 → 3 stocks ahead of buy-and-hold. The gate
scores valuation against the hard-coded `SECTOR_NORMS`; whether any fundamental signal helps is a research question for
the cross-sectional harness (Phase 3 of `docs/ROADMAP_STATUS.md`), not for this gate.

**Survivorship-aware cohorts.** The 20 sample stocks are the survivors of 1990-2022. `--cohort DATE` tests every
S&P 500 member on DATE, from DATE until today or until its prices stop, with live Yahoo prices, the S&P 500 regime,
the FRED T-bill for Sharpe ratios and for uninvested cash, SEC filing-dated fundamentals, resting stop orders and
liquidity-scaled costs (`reports/examples/cross_ticker_sp500_<DATE>.csv` and `_summary.csv`; re-run 14 October 2026):

| Cohort | Members | With prices (coverage) | Buy & hold: CAGR / Sharpe / max DD | Signals + regime Sharpe | Lighthouse balanced: CAGR / vol / Sharpe / max DD | Beat buy & hold (Sharpe / CAGR) | Lower drawdown |
|---|---|---|---|---|---|---|---|
| 1996-01-02 → 2026-10 | 487 | 258 (53%) | 7.6% / 0.36 / −72% | 0.25 | 5.1% / 11.4% / 0.24 / −28% | 78 / 55 of 258 | 254 |
| 2009-01-02 → 2026-10 | 498 | 330 (66%) | 12.0% / 0.48 / −59% | 0.36 | 4.9% / 11.9% / 0.30 / −28% | 43 / 31 of 330 | 314 |

Per-stock averages. As one equal-weight cohort (one sleeve per member, never rebalanced), buy-and-hold earned 11.1% a
year from 1996 and 17.0% from 2009; the Lighthouse rule 5.4% and 5.5%; the signals alone, fully invested when in, 8.5%
and 10.8%. Those figures leave out the members Yahoo cannot price. If every one of them was worth nothing (the
pessimistic bound), the cohort earned 8.8% / 3.2% / 6.3% from 1996 and 14.3% / 3.1% / 8.3% from 2009. Before cash
earned the T-bill (and with a fixed 10 bps per side and close-checked stops) the balanced rule's Sharpe was 0.13 and
0.23 and it beat buy-and-hold on 29 stocks in each cohort: the rule holds cash for much of the time, and that cash had
earned nothing. Every member
Yahoo could price still trades today: free data has no price history for the delisted, so the delisting rules change
nothing at the stock level and the whole bias sits in the untested 47% (1996) and 34% (2009). Ticker reuse is a second
risk: Yahoo's history for a recycled ticker can belong to another company.

Interpretation: on these data the rule has no edge as a stock-timing tool. It lowers volatility and drawdown on nearly
every stock, but it gives up most of the return, beats buy-and-hold on risk-adjusted terms on 1 stock in 8 (2009) to
1 in 3 (1996), and the regime filter plus sizing sit in cash (earning the T-bill) for much of a long bull market. The product
still delivers *the risk profile the investor asked for*; it does not deliver extra return, and it should not be
presented as if it did. What comes next is cross-stock research (which stocks to own, not when to own one), forward
testing with a journal that cannot contain look-ahead, and honest costs (`docs/ROADMAP_STATUS.md`).

### 8b. Cross-sectional research on the point-in-time S&P 500 (`lighthouse/research/`)

The timing tests above ask whether the rule beats holding one stock. The research harness asks the question the
verdict implies: **do stocks with higher scores do better next month than stocks with lower scores?**

Protocol (`lighthouse/research/harness.py`, run by `scripts/research_signals.py`):

* Universe: every S&P 500 member at each month end, as it was (historical membership, §1), 2009-2026. Members
  without a Yahoo price cannot be tested and count against coverage; a member is tested only in months it traded on
  at least 15 of the last 21 sessions (a series without volume is stale, or another company's data under a reused
  ticker: CPWR's cached series had zero volume and a $2 price for the four years it was a member).
* Statistics: monthly Spearman rank IC with Newey-West t (3 lags); quintile returns; top minus bottom quintile
  (long-short, with a 0.5% a year borrow fee on the short leg) and top quintile minus the equal-weight universe
  (long-only); top-quintile turnover; Fama-French five-factor + momentum alphas with Newey-West t, factors aligned to
  the month being predicted.
* Costs, per side: 5 bps commission + the Corwin-Schultz half spread (floor 5 bps) + square-root impact of a $10,000
  order (a $1m portfolio, 100 names a leg), capped at 300 bps. From daily highs and lows the estimator reads part of
  the day's volatility as spread (about 35 bps full spread for AAPL or MSFT in 2009-2017 against a quoted 1-3 bps), so
  these costs are conservative for large companies; every result also shows the long-only return at a flat 10 bps a
  side (a sensitivity, not part of the rule).
* Both delisting rules (a member's last month counted at its last price, or at -100%), with coverage.
* Tuning 2009-2017; hold-out 2018 onward, evaluated once per signal and logged (`reports/research/holdout_log.csv`); a
  second look counts as a new trial. Every evaluation is appended to `reports/research/experiments.csv`; the number of
  distinct signal configurations tried is the trial count behind the deflated Sharpe ratio (Bailey & López de Prado
  2014).
* Survival rule, written before any hold-out was run: under both delisting rules, tuning IC Newey-West t ≥ 2 and a
  positive net return for the signal's better use (long-only or long-short); hold-out IC > 0, the same use's net return
  > 0 and its deflated Sharpe ≥ 0.90. Only a survivor may carry weight in `lighthouse/decision.py`.

**Results for the signals the product uses** (October 2026; 643 of 881 ever-members priced; tuning coverage 70% of
member-months, hold-out 89%; both delisting rules give the same numbers because no priced member stopped trading
before the data end: Yahoo has no prices for the dead, who are most of the untested share):

| Signal | Tuning IC (NW t) | Gross L/S | Long-only net (10 bps) | Hold-out IC (NW t) | Hold-out long-only net | Deflated Sharpe | Survives |
|---|---|---|---|---|---|---|---|
| `tech_long` | -0.026 (-1.4) | -9.4% | -10.2% (-7.3%) | +0.003 (+0.2) | -4.5% | 0.00 | no |
| `tech_short` | -0.030 (-1.9) | -9.0% | -10.8% (-7.3%) | +0.001 (+0.1) | -5.0% | 0.00 | no |
| `beacon_trend` | -0.019 (-1.1) | -6.6% | -6.1% (-4.4%) | +0.006 (+0.4) | -0.8% | 0.00 | no |
| `beacon_rsi` | -0.033 (-2.3) | -9.5% | -11.6% (-7.8%) | -0.006 (-0.4) | -8.3% | 0.00 | no |
| `beacon_macd` | -0.023 (-1.9) | -5.2% | -10.1% (-5.3%) | +0.013 (+0.9) | -4.7% | 0.00 | no |
| `beacon_bands` | -0.025 (-2.2) | -7.4% | -10.9% (-6.4%) | -0.004 (-0.3) | -8.3% | 0.00 | no |
| `beacon_volume` | -0.011 (-1.1) | -1.2% | -6.0% (-1.6%) | -0.001 (-0.1) | -7.1% | 0.00 | no |
| `beacon_breakout` | -0.022 (-1.0) | -9.5% | -8.8% (-6.5%) | +0.000 (-0.0) | -4.8% | 0.00 | no |
| `fund_pit` | -0.005 (-0.6) | -3.0% | -3.3% (-2.7%) | +0.011 (+1.0) | +0.4% | 0.01 | no |
| `composite_long` | -0.020 (-1.4) | -9.3% | -8.0% (-6.2%) | +0.009 (+0.9) | -1.4% | 0.00 | no |

Returns are annualised; "gross L/S" is top minus bottom quintile before costs; the long-only columns are the top
quintile minus the equal-weight universe after costs (in brackets at 10 bps a side). Ten configurations were tried.

Interpretation: **none of the product's signals ranks large US stocks usefully.** In 2009-2017 every technical beacon
had a negative IC (stocks the beacons liked did worse the next month: in large caps a month's strength tends to
reverse), and the gross spreads were negative before any cost, so the verdict does not depend on the cost model. After
2018 the ICs are near zero. The point-in-time fundamental score is the only one with a positive hold-out (IC +0.011,
t 1.0; long-only +0.4% a year after costs, t 0.25): too weak to call. The rules still do what §8 says (they cut
volatility and drawdown); they are not a stock-selection tool, and nothing here justifies more weight on any of them.
The broader signal library (value, quality, momentum, earnings, positioning, text; Phase 4) is the place to look for
one that is.

**Results for the signal library** (Phase 4, run 15 October 2026 at commit `a1c64bb`, the earnings-release tone last at
`6f7ecc1`; the same protocol, data and costs; 645 of 883 ever-members priced). 37 signals, each one configuration fixed
before its tuning run: with the product's ten, 47 trials behind the deflated Sharpe ratio (cross-trial Sharpe dispersion
0.185 a month). Coverage in the hold-out is
80-90% of member-months for every signal (24% for the earnings-release tone, run on half the members; 46% for the Altman Z, which needs current assets, and 62% for gross
profitability, which banks and insurers do not report); in the tuning period 23-71%, lowest for the fundamental
signals (XBRL from 2009, trends and changes need a year or more of filings). Both delisting rules again give identical
numbers. Returns are annualised; "better use" is the long-only or long-short version the rule picks from the tuning
results; net returns after the cost model (in brackets at a flat 10 bps a side).

|---|---|---|---|---|---|---|---|
| *Fundamental (SEC filings)* | | | | | | | |
| `sue` | +0.008 (+1.0) | +0.4% | L/O -1.5% (-0.4%) | +0.007 (+0.8) | -0.5% | 0.00 | no |
| `gp_assets` | +0.002 (+0.1) | -0.1% | L/O -1.3% (-1.0%) | +0.009 (+0.7) | -1.1% | 0.00 | no |
| `roic` | +0.008 (+0.6) | +1.6% | L/O +0.3% (+0.6%) | +0.019 (+1.6) | +3.0% | 0.00 | no |
| `roic_trend` | -0.012 (-1.5) | -2.2% | L/O -1.1% (-0.5%) | +0.002 (+0.2) | +2.3% | 0.00 | no |
| `accruals` | +0.006 (+0.6) | +2.1% | L/O +2.4% (+2.9%) | -0.002 (-0.2) | +1.5% | 0.00 | no |
| `piotroski` | +0.005 (+0.6) | +0.4% | L/O -0.8% (+0.0%) | -0.007 (-0.8) | -2.3% | 0.00 | no |
| `altman_z` | +0.001 (+0.0) | -1.5% | L/O -0.8% (-0.4%) | +0.004 (+0.3) | -0.2% | 0.00 | no |
| `beneish` | +0.005 (+0.5) | +2.0% | L/S -0.5% (+0.9%) | +0.000 (+0.0) | -2.2% | 0.00 | no |
| `cash_conversion` | -0.003 (-0.3) | +0.1% | L/O -0.7% (-0.3%) | -0.000 (-0.0) | +0.6% | 0.00 | no |
| `earnings_yield` | +0.015 (+1.4) | +0.7% | L/O -0.2% (+0.4%) | +0.010 (+0.8) | -1.1% | 0.00 | no |
| `book_to_market` | -0.001 (-0.1) | -0.1% | L/O +0.4% (+0.8%) | -0.003 (-0.2) | +0.5% | 0.00 | no |
| `fcf_yield` | +0.024 (+2.3) | +5.4% | L/S +3.5% (+4.5%) | +0.009 (+0.7) | +0.9% | 0.00 | **tuning only** |
| `fcf_sbc_yield` | +0.022 (+2.1) | +4.2% | L/S +2.3% (+3.2%) | +0.009 (+0.7) | +0.3% | 0.00 | **tuning only** |
| `ebitda_ev` | +0.017 (+1.4) | +2.0% | L/O +0.7% (+1.2%) | +0.005 (+0.3) | +0.8% | 0.00 | no |
| `value_own_history` | -0.003 (-0.3) | -3.7% | L/O -2.4% (-1.6%) | -0.001 (-0.2) | -3.2% | 0.00 | no |
| `net_issuance` | +0.022 (+2.0) | +3.7% | L/S +1.5% (+2.6%) | +0.011 (+1.0) | -1.3% | 0.00 | no |
| `shareholder_yield` | +0.014 (+1.3) | +0.2% | L/O +0.2% (+0.7%) | +0.019 (+1.5) | +1.0% | 0.00 | no |
| *Price and size* | | | | | | | |
| `mom_12_1` | -0.008 (-0.4) | -7.8% | L/O -5.3% (-3.8%) | +0.002 (+0.1) | -0.4% | 0.00 | no |
| `low_vol` | -0.015 (-0.6) | -11.0% | L/O -5.0% (-4.7%) | -0.011 (-0.5) | -3.2% | 0.00 | no |
| `reversal_1m` | +0.017 (+1.2) | +2.8% | L/O -4.5% (+0.2%) | +0.001 (+0.1) | -6.8% | 0.00 | no |
| `high_52w` | -0.023 (-1.0) | -12.8% | L/O -8.8% (-6.5%) | +0.000 (+0.0) | -4.8% | 0.00 | no |
| `size` | +0.022 (+2.0) | +4.9% | L/S +3.3% (+4.1%) | -0.004 (-0.3) | -1.8% | 0.00 | **tuning only** |
| *Panel* | | | | | | | |
| `industry_mom` | -0.018 (-1.4) | -7.5% | L/O -6.7% (-5.4%) | +0.012 (+1.0) | +1.9% | 0.00 | no |
| `residual_mom` | -0.002 (-0.2) | +0.8% | L/O -1.0% (+0.2%) | +0.002 (+0.2) | +0.3% | 0.00 | no |
| *Positioning (Form 4, FINRA)* | | | | | | | |
| `insider_buyers` | +0.004 (+0.7) | +0.6% | L/O -0.8% (+0.6%) | +0.002 (+0.5) | -2.2% | 0.00 | no |
| `insider_net` | +0.000 (+0.0) | +1.8% | L/O +0.1% (+1.5%) | +0.001 (+0.1) | -0.9% | 0.00 | no |
| `short_volume` | +0.019 (+3.5) | +3.2% | L/O -2.7% (+0.4%) | +0.013 (+1.9) | -3.3% | 0.00 | no |
| `days_to_cover` | n/a (data from 2018) | n/a | n/a | +0.013 (+1.7) | -1.9% | 0.00 | no |
| *Events and text (8-K, 10-K)* | | | | | | | |
| `ear_3d` | +0.003 (+0.3) | -1.4% | L/O -1.9% (-0.0%) | +0.001 (+0.2) | -3.0% | 0.00 | no |
| `text_similarity` | -0.000 (-0.1) | -0.3% | L/O -1.6% (-1.0%) | +0.010 (+1.9) | -0.1% | 0.00 | no |
| `risk_similarity` | -0.000 (-0.1) | +0.3% | L/O -0.7% (-0.2%) | +0.011 (+2.3) | +0.6% | 0.00 | no |
| `lm_uncertainty` | -0.022 (-2.4) | -5.3% | L/O -3.3% (-3.1%) | -0.012 (-1.1) | -2.3% | 0.00 | no |
| `lm_litigious` | -0.000 (-0.0) | +1.6% | L/S +0.6% (+1.0%) | -0.014 (-1.4) | -5.3% | 0.00 | no |
| `lm_negative` | -0.021 (-1.6) | -6.6% | L/O -4.9% (-4.6%) | -0.012 (-1.4) | -1.4% | 0.00 | no |
| `fog` | -0.001 (-0.2) | -0.6% | L/O -1.0% (-0.7%) | +0.001 (+0.1) | +0.2% | 0.00 | no |
| `eightk_tone` (half the members) | +0.007 (+0.8) | +2.8% | L/O +0.2% (+2.2%) | +0.000 (+0.0) | -2.2% | 0.01 | no |
| *Institutions (13F)* | | | | | | | |
| `inst_breadth` | +0.006 (+0.4) | +1.8% | L/O -0.4% (+0.6%) | +0.001 (+0.1) | +0.6% | 0.00 | no |

Interpretation:

* **Nothing survives, so nothing is weighted.** Three signals passed the tuning test: the free-cash-flow yield (IC
  +0.024, t 2.3; long-short +3.5% a year after costs), the same after stock-based pay (+2.3%), and small size within the
  index (t 2.00, +3.3%). In the hold-out the two FCF yields kept their sign (IC +0.009, t 0.7; long-short +0.9% and
  +0.3% a year) but not their strength, and size reversed (the largest companies led after 2018). Every deflated Sharpe
  ratio is about 0: with 47 trials the bar for a single hold-out Sharpe is high, and none came near it. The sector- and
  size-neutral blend (`lighthouse/research/blend.py`) is built for survivors only, so it was not run: blending the three
  tuning passers after seeing their hold-out would be choosing on the hold-out. `lighthouse/decision.py` is unchanged.
* **Short-sale volume predicts and cannot be traded monthly.** The strongest IC in both periods (t 3.5 in tuning, 1.9 in
  the hold-out), but its top quintile turns over 60-66% a month, so a gross long-short spread of +3.2% a year becomes
  -6.5% under the cost model (about break-even at 10 bps a side). A real signal in rank terms that the costs eat.
* **Risk-factor similarity** (Lazy Prices on Item 1A) had a flat tuning period and a hold-out IC of +0.011 (t 2.3): the
  rule does not allow promoting it on hold-out evidence. It is a candidate for the forward journal, which gives new
  out-of-sample data, not for weight.
* **The earnings-release tone** (the change in net tone from the company's previous release, half the members, a
  median of 170 names a month) had a small positive tuning IC (+0.007, t 0.8) and none in the hold-out (t 0.0); its
  35% monthly turnover costs more than its +2.8% gross spread.
* **The 10-K tone hypotheses had the wrong sign.** More uncertainty and negative words came before *higher* returns
  (tuning t -2.4 and -1.6, hold-out -1.1 and -1.4), which reads as a risk premium rather than mispricing. Signs were
  registered before testing and are not flipped after.
* **Momentum, low volatility and the 52-week high** were negative in 2009-2017 (the 2009 momentum crash and the rally
  in the riskiest stocks) and near zero after: in large caps these classic anomalies have been weak since 2009.
* Several parser and timing errors were fixed before the hold-out was run (§11, sixth session): the tuning rows
  logged before those fixes (insider and short-volume signals) stay in `reports/research/experiments.csv`; the results
  above are the corrected ones.

## 9. Trade tickets: long and short (`lighthouse/tradeplan.py`)

| Item | Rule | Reasoning |
|---|---|---|
| Long setup | Actionable when the verdict is Buy / Strong Buy **and** the trend beacon is positive (the backtested entry gate) | The ticket only fires on what was tested |
| Short setup | Actionable when the composite ≤ 40, the trend beacon is negative, the technical score ≤ 40, and the market is not Bull / Calm | Mirror of the long rule around 50 (Buy ≥ 60 ↔ short ≤ 40; Bear / Turbulent blocks longs ↔ Bull / Calm blocks shorts); bull markets halve short size |
| Watch list | Positive (negative) evidence without the trend confirmation, or a Hold with a strong technical score | Levels are shown; no order |
| Stop-loss | Longs: the horizon's stop (200-day SMA − 8% long term, 2.5 × ATR short term). **Shorts always use the short-horizon ATR stop** | A short's loss is open-ended and borrow fees accrue daily; a wide SMA stop would make its targets meaningless |
| Size | The three risk rules as % of the account × conviction × regime, capped by the cash; shorts are capped at 1:1 collateral (Regulation T would allow 2:1) | Same engine as the Position plan, so the long ticket and the plan agree |
| Entry | Limit order at the last price; skip if price has run 0.5 × ATR beyond it | Do not chase |
| Long-horizon buys: trend plan (default) | No fixed target. The stop order stays where it was placed; at the last close of each month, sell at the next open if the close is below the 200-day SMA − 8% (the *trend exit*, a level that rises with the average) or the technical score is ≤ 35 (the *signal exit*). No time stop | The long-horizon backtest's own exits, so the ticket and the backtest agree; tested against R targets below |
| Short horizon and shorts: targets | Target 1 = entry ± 2R, target 2 = entry ± 3R (R = entry-to-stop distance; editable). Long-horizon buys can use them too (`TradeSettings.long_exit = "targets"`) | R-multiples express reward in units of risk; a 2R target needs a win rate above 1/3 to pay |
| Management (targets) | Sell 50% at target 1, move the stop on the rest to breakeven, then trail it at the best close ∓ k × ATR (chandelier exit); time stop after 40 (short horizon) / 252 (long) sessions | Locks in gains, caps dead money |
| Portfolio heat | Ideas are funded best-first until the cash, the idea count (5) or the total planned risk (6% of the account) runs out; leftovers smaller than 25% of an idea's full size are not funded | Limits correlated losses across many open trades |

**Historical setup test.** For every past day the stock met the (point-in-time) setup rule, the trade is entered at the next
session's open (the close for close-only data) and managed with exactly the rules above, one trade at a time, after costs.
A trend plan's review-day exits fill at the next session's open, as in the backtest.
Gaps fill at the open; when a stop and a target fall in the same session the stop is assumed first. Results are in R
(expectancy, win rate, profit factor, share of trades reaching target 1 / stopped / timed out). A ticket whose rules lost
money on that stock is flagged in red. Shorts are simulated on the dividend-adjusted series, so the dividends a short seller
owes are reflected; borrow fees are not.

**Long-horizon exits: trend plan instead of far targets.** The long-horizon stop sits 8% below the 200-day SMA, typically
15-30% under the price, so R is large and 2R / 3R targets sat 30-80% away (MSFT in October 2026: stop 26% away, targets
+53% and +79%). Candidate plans were replayed with the setup test on every long-horizon setup of the 20 sample stocks,
balanced profile, after costs, all with the same entries and the same initial stop (so R is comparable):

| Plan (1990-2022) | Trades | Avg R | Win rate | Avg sessions | Return a year held |
|---|---|---|---|---|---|
| 2R / 3R targets, 5 × ATR trail, 252-session time stop (until October 2026) | 707 | +0.56 | 63% | 170 | 16.8% |
| Volatility targets: 1σ / 2σ of the 1-year move, 5 × ATR trail | 767 | +0.51 | 66% | 158 | 16.1% |
| Stop raised monthly to the 200-day SMA − 8%, tightened to a 5 × ATR trail after a 1σ (1-year) gain | 745 | +0.53 | 56% | 153 | 16.7% |
| Stop order raised to the 200-day SMA − 8% each month | 361 | +2.17 | 44% | 310 | 26.4% |
| Same, plus the signal exit | 369 | +2.26 | 45% | 302 | 26.3% |
| Same, but half sold at +1σ | 361 | +1.32 | 46% | 310 | 16.2% |
| Chandelier 5 × ATR trail from entry (with the SMA trail) | 3,566 | +0.09 | 43% | 27 | 12.9% |
| **Trend plan: stop order kept; month-end trend and signal exits (the backtest's)** | **296** | **+3.56** | **45%** | **392** | **33.9%** |

The trend plan had the higher average R on all 20 stocks, in each decade (1990s +6.3R vs +0.75R, 2000s +1.5R vs +0.26R,
2010-22 +3.4R vs +0.64R), and over the app's shorter windows (`scripts/exit_plan_evaluation.py`): last 15 years +2.53R vs
+0.57R (20.1% vs 13.2% a year held; better on 18 of 20 stocks), last 10 years +2.09R vs +0.62R (19.5% vs 15.3%; 16 of 20).
Loss tails were similar: 5% of trend-plan trades lost more than 1.3R (gaps through the stop) against 1.2R for the targets.
What it costs: a lower median trade (−0.22R; most trades lose a little, a few win a lot), fewer trades, long holding periods,
and a large give-back from the peak before a trend exit fires. Faster profit-taking (volatility targets, partial sales,
tight ATR trails) cut the winners that pay for everything else. Checking the trend rules at month-end closes rather than
resting the raised stop as an order avoided intra-month shake-outs (+3.56R vs +2.26R). Caveat: the sample stocks are 2022
survivors, which flatters any plan that lets winners run; the trend plan still beat the targets on GE, BBY and RRC, the
pack's weakest stocks. Long-term investors who prefer fixed targets can switch them back on.

**When to get out.** Every ticket lists its exits with a trigger, the result for the *whole* position if the trade ends
that way, and how often (and how fast) the setup test's trades ended that way on that stock. A trend plan lists the stop
order (−1R), the trend exit (today's level; its result depends on how far the trend ran, so the row shows the average
result of past trend exits) and the signal exit (today's technical score against 35). A targets plan lists the stop (−1R); target 1 (the
partial sold there, which is also the result if the rest is then stopped at breakeven); the trailing stop (between target 1
and target 2); target 2 (the best planned outcome); the time stop (between the stop and target 1); and a verdict turning
against the trade at the weekly (short horizon) or monthly (long horizon) review. The exits are decided, and the stop
order placed, when the trade is entered.

**Position replay (CLI `monitor`).** An open position is replayed from the session after its entry with the same rules,
assuming the plan was followed. A trend plan reports a stop hit, a trend or signal exit already taken, or a review at the
last close that says sell at the next open. On a targets plan: the shares sold at target 1 are the ticket's split (half by default, all of a single
share), every later exit closes what is left. P&L is realised on those fills (the fill price, so a gap through a target
or a stop counts at the open) and unrealised only on the shares still held, at the last close; R is that total over the
initial risk of the whole position. It also reports how far price is from the current (breakeven or trailing) stop, a
due time stop and whether the verdict has turned against the trade. The app has no position-entry screen: the tickets
carry the exits instead.

**Short-sale facts shown on short tickets.** Regulation T initial margin is 150% of the short value (the sale proceeds plus a
50% deposit). FINRA Rule 4210 maintenance for a short stock at $5 or more is the greater of $5 per share or 30% of market
value (below $5: the greater of $2.50 per share or 100%). SEC Rule 201: after a 10% fall from the prior close, shorts may only
be sold above the national best bid for the rest of that day and the next. Shorting needs a margin account (not a cash
account or an IRA), borrowable shares, and the payment of dividends in lieu.

## 10. Portfolio construction and risk (`lighthouse/portfolio.py`)

| Step | Choice | Reasoning |
|---|---|---|
| Returns | Daily simple returns on the common dates of the basket; series ending > 10 days before the others are left out (and reported) | One stale series must not cut everyone's history |
| Covariance | Ledoit-Wolf (2004) shrinkage toward a scaled identity, estimated on the last 252 sessions (editable) | The sample covariance is noisy and ill-conditioned; verified against scikit-learn |
| Expected returns | Only Maximum Sharpe uses them: Bayes-Stein (Jorion 1986) shrinkage toward the minimum-variance portfolio's mean | Historical means are the noisiest input in finance |
| Methods | Equal weight; inverse volatility; equal risk contribution (Spinu 2013 convex form); minimum variance; maximum Sharpe (convex reformulation); hierarchical risk parity (López de Prado 2016: single-linkage on √((1−ρ)/2), quasi-diagonal order, recursive bisection) | From assumption-free (1/N) to estimation-heavy (max Sharpe), so the walk-forward table shows what estimation buys |
| Constraints | Long-only, fully invested, per-stock min/max, optional per-sector cap, solved inside the optimiser (heuristic methods are projected onto the caps by least squares) | Feasibility is checked first, with a plain-English error |
| Volatility target | If expected volatility exceeds the target, the excess is held in cash | Delivers the risk the investor asked for |
| Orders | Whole shares by flooring, then the leftover cash buys one share at a time for the position furthest below target | Never overspends; a cash buffer is never spent |
| Risk report | Expected volatility (shrunk covariance); historical-simulation VaR and expected shortfall at 95% for 1 day and 1 month (overlapping 21-session windows); beta, correlation and tracking error vs the S&P 500; back-cast max drawdown; risk contributions; diversification ratio; effective number of stocks (1/Σw²); stress tests through the dot-com bust, the GFC, 2011, Q4 2018, the COVID crash and 2022 (S&P 500 closing peak to trough) when covered by the data | Each answers "how bad can it get?" in a different way |
| Walk-forward test | Monthly (or quarterly) rebalance; weights from the trailing window only; trades executed at the **next** day's close; 10 bps per dollar traded (turnover = buys + sells as a share of the portfolio); weights drift between rebalances; cash earns 0% | No look-ahead (unit-tested by truncation) |

**Goals.** *Capital preservation* picks the 10 calmest stocks and uses minimum variance with a 12% volatility target. *Balanced*
picks the 10 best technical scores (≥ 45) and uses risk parity. *Growth* picks the 8 strongest 12-1 month momentum names in an
up-trend and uses maximum Sharpe. *Maximum diversification* picks 15 names (technical score ≥ 45) and uses HRP. At most 3 stocks
per sector and sector weight caps apply. A ticker outside the curated sector list gets the sector its fundamentals
snapshot reports (Yahoo's `info["sector"]`, cached); one with no sector at all (ETFs, unknown symbols) is labelled
"Unknown" and is exempt from both the 3-per-sector limit and the sector weight caps, because "Unknown" is not a sector.
Before this rule, every unlabelled ticker shared one "Unknown" sector, so a custom universe produced at most 3 picks.
When a universe spans too few sectors to fill the goal (a sector list such as "Technology"), the per-sector limit is raised
to ⌈stocks wanted / sectors⌉, and a sector weight cap that the sectors cannot satisfy is dropped; both are noted on the
result.
All 122 curated labels were checked against Yahoo's live `info["sector"]` in October 2026 and match. The walk-forward test **re-selects** the stocks every rebalance with only the data
available then, so it does not benefit from knowing today's winners. That matters: the back-cast of today's picks over the
last year is inflated by how they were chosen, so the app shows it for risk only and uses the walk-forward test for performance.

**Holdings are kept while they qualify (October 2026).** Re-picking the top N from scratch every month made the
technical-score and momentum ranks, which move week to week, replace most of the portfolio each month: the balanced goal
on the live large-cap universe turned over 18.5× a year (buys + sells). *Balanced*, *growth* and *diversified* now keep a
stock they hold for as long as it still passes the goal's filter (technical score ≥ 45; for growth, a close above the
200-day SMA) and re-fill only the freed places by rank (`Goal.keep_rank`, a rank buffer; infinite for these three). The
walk-forward test applies it to its own simulated holdings; the app and the CLI (`--held`) apply it to the stocks the
investor says they hold. Walk-forward, monthly, 10 bps per dollar traded (`scripts/turnover_evaluation.py`):

| Goal | Universe | Turnover / yr, from scratch → kept | Sharpe | CAGR | Max drawdown |
|---|---|---|---|---|---|
| Balanced | 86 large caps, Aug 2016 – Oct 2026 (live) | 18.5× → 2.4× | 0.48 → 0.79 | 9.1% → 15.4% | −29% → −31% |
| Growth | same | 10.0× → 4.9× | 0.78 → 0.94 | 18.7% → 22.5% | −35% → −34% |
| Maximum diversification | same | 16.4× → 3.3× | 0.61 → 0.86 | 11.0% → 15.4% | −30% → −27% |
| Balanced | 19 sample stocks, 2012-2022 | 6.8× → 2.4× | 0.90 → 0.83 | 16.8% → 15.4% | −32% → −31% |
| Growth | same | 6.6× → 4.9× | 0.73 → 0.71 | 14.2% → 13.8% | −27% → −27% |
| Maximum diversification | same | 5.6× → 3.8× | 0.84 → 0.85 | 15.2% → 15.5% | −31% → −31% |

Also tested on the large caps (balanced): a 3-percentage-point no-trade band on the weights (18.5× → 18.2×: the turnover
came from swapping stocks, not from re-weighting them); keeping holdings only while they rank in the top 2N or 3N (13.5×
and 9.5×); quarterly reviews (6.5×, Sharpe 0.59). On the 1990-2022 sample pack the kept version turned over 2.7× instead
of 6.7× with the same Sharpe (0.69). *Capital preservation* keeps re-selecting from scratch: it ranks by 1-year
volatility, which moves slowly (3.3× a year on the large caps), and a buffer lowered its Sharpe on every universe tested
(large caps 0.55 → 0.51, sample 0.69 → 0.62). The sample-pack results are close to a wash; the large-cap gains are large
but come from one 10-year window, so read them as "much less trading at no cost", not as a promised improvement.

**What the evidence says.** On the app's default basket (AAPL, JNJ, XOM, JPM, KO, PG, UNH, WMT; offline sample data;
walk-forward from late 2012 to December 2022; 25% cap; 10 bps costs) plain equal weight had the best Sharpe (0.84). Risk
parity was close (0.82) with lower volatility (15.1% vs 15.7%). Minimum variance had the smallest drawdown (−25.8% vs
−30.8%). Maximum Sharpe turned over 3.6× a year without beating 1/N (Sharpe 0.73). All six beat the S&P 500's 0.53 over
the window, a result that owes much to survivorship (these are 2022's survivors). This matches DeMiguel, Garlappi & Uppal (2009): estimation error
eats most of the theoretical gain from optimisation, which is why risk parity is the default and maximum Sharpe is labelled
the most estimation-sensitive.

## 11. Corrections made in the October 2026 code review

| Issue | Effect | Fix |
|---|---|---|
| Bollinger middle band computed but never drawn | The bands appeared to float around nothing | Dashed 20-day SMA on both chart sets (the classic bands were later replaced by low-lag price bands, §3) |
| Market-weather shading never appeared in the interactive charts | Plotly silently drops shapes aimed at a subplot that has no trace yet; the legend listed regimes that were not drawn | Shapes are added with `exclude_empty_subplots=False` |
| Sortino ratio used the standard deviation of the negative returns only | Wrong formula (overstated or understated depending on the series) | Downside deviation below the risk-free rate over **all** periods (Sortino & Price 1994) |
| Profit factor summed unweighted trade returns | A small trade counted as much as a large one | Gross profit / gross loss of size-weighted P&L |
| Stop distance clamped at 60% when sizing | For a stock far above its 200-day SMA, the real loss at the stop could exceed the investor's limit | No upper clamp (the 1% floor stays; it only makes positions smaller) |
| CLI `--horizon short` kept the long-horizon rule | Weekly review and ATR stop were never applied from the command line | The CLI starts from `StrategyParams.for_horizon` |
| A fresh price cache was reused even when it covered a shorter window | A 6-year watchlist download could silently shorten a later 10-year analysis | The cache records the requested start and must cover the new request |
| Dividend yield guessed percent vs fraction from the value | Yields under 0.3% read as percent were 100× too high | Uses annual dividend per share ÷ price first |
| Position plan sized from "portfolio value" | Implied the whole portfolio was cash to deploy | Sized from cash available; risk limits on the whole account |

The changes do not alter the strategy backtest itself. Regenerating `reports/examples/MSFT` reproduced every return,
drawdown, Sharpe and trade figure exactly; only the Sortino and profit-factor columns changed.

### Second October 2026 session

| Issue | Effect | Fix |
|---|---|---|
| Tickers outside the curated sector list all became "Unknown" | Goal portfolios on custom universes stopped at 3 stocks; sector caps were infeasible or forced concentration | Sector from the fundamentals snapshot; "Unknown" exempt from per-sector limits |
| Position replay valued every share at the last price | After target 1 (half sold) or a stop, P&L and R ignored the fills already taken (one test case: $299 reported, $231 realised) | Realised P&L on the exits, unrealised on the open shares |
| A bar for a session still trading was used as a finished day | Signals, especially volume, moved during the day; a partial bar could sit in the cache for 20 hours | Unfinished bar left out and never cached; orders priced from a timestamped quote |
| A cache written before a session closed was reused after it | A finished session could be missing for most of a day | The cache expires at the first session close after the download |
| The app's "Position monitor" asked the investor to type in positions | Not what the tickets are for: the sell decision belongs to the plan | Removed; every ticket shows *When to get out* |
| Bollinger Bands trailed the price by ~10 sessions | The chart and the beacon described where price was two weeks earlier | Hull-MA price bands (~3 sessions); the classic bands are gone from the score and every chart (§3) |
| The README's cross-ticker table had no generating script | Its provenance could not be checked | `scripts/cross_ticker_evaluation.py` rebuilds it (it reproduced the old file byte for byte before the band change) |
| The CLI `monitor` dropped today's unfinished bar | A position entered today could not be replayed, and a stop hit today was not reported | `analyze(keep_partial=True)` keeps today's bar (stock and benchmark) for the replay only |
| A benchmark with no rows in the stock's window crashed the analysis | E.g. live prices with the S&P 500 falling back to the 2022 sample file | `load_prices` raises when the window is empty and `align` when the frames do not overlap; the regime rules are then switched off with a warning |
| A quote was requested even after the prices had fallen back to old data | An extra network timeout per ticker, and today's price paired with stale signals | A separate quote is fetched only for current prices |

### Third October 2026 session

| Issue | Effect | Fix |
|---|---|---|
| Long-horizon targets sat 30-80% away (2R / 3R on a stop 8% below the 200-day SMA), and the ticket's fixed stop differed from the backtest's monthly SMA exit | Profit targets that a long-term trade rarely reached; two different exit models | Long-horizon buys follow the backtest's month-end trend and signal exits, no fixed target (+3.56R vs +0.56R per trade on the sample setups, §9); `scripts/exit_plan_evaluation.py` |
| Goal portfolios re-picked their stocks from scratch every month | The balanced goal turned over 18.5× a year on the live large caps, paying costs to swap stocks whose ranks moved a little | Holdings are kept while they still qualify (balanced 18.5× → 2.4×; §10); the walk-forward table shows both versions; `scripts/turnover_evaluation.py` |
| Sharpe and Sortino ratios used a fixed 2% risk-free rate | Overstated in high-rate years (T-bills ~4-5% in 2023-26), understated near zero (2020-21) | The 3-month T-bill (FRED `DTB3`) day by day for live analyses and portfolio walk-forward tests; 2% offline (§1) |
| The Hull-MA band centre still trailed the price by ~3 sessions and swung 26% past a jump | The chart's centre line and the *price bands* beacon described the price of three days earlier | Holt's trend filter (0.4 / 0.1): ~1 session behind, no lag in a trend, 9% overshoot (§3); `scripts/band_lag_evaluation.py` measures every candidate |

### Fourth October 2026 session (research-grade data)

| Issue | Effect | Fix |
|---|---|---|
| Statements became usable a fixed 90 days after their period end | The gate could use a 10-K before it was filed (late filers), or wait a month after it was (most large filers file in ~30-60 days) | Filing dates from SEC EDGAR XBRL; usable the session after filing. The 90-day lag remains only for Yahoo's undated statements |
| Only MSFT had fundamentals offline (5 annual rows, 2013-2017, typed in by hand and labelled approximate) | The fundamental gate was never tested on 19 of the 20 sample stocks | The sample pack records each stock's SEC XBRL facts (filed 2009-2026) and its dividends and splits (`scripts/build_edgar_sample.py`); the cross-ticker table regenerates with the gate active from 2009 (Sharpe 0.401 → 0.383) |
| P/E compared a split-adjusted price with as-reported EPS | Wrong by the later split factor (AAPL: 28×) once EPS comes from filings | EPS restated in the price series' share units; price-only close recovered from the total-return close |
| The test universe was today's survivors only | Every result was flattered, especially "let winners run" exits | Historical S&P 500 membership; `--cohort DATE` publishes the survivorship-aware results with coverage (§8) |
| Today's ticker map pointed XOM at a new holding company (2026) with two quarters of filings | Exxon's history would have disappeared | `edgar.PREDECESSOR_CIKS` merges a successor issuer's predecessors |
| A ticket could be produced from sample, stale or past data | An order priced from data that is not current | The live-mode rule (§1) |
| The market clock ignored exchange holidays | A cache was treated as stale one close too early on holidays | NYSE holiday calendar (`data.nyse_holidays`) |
| Uninvested backtest cash earned 0% while Sharpe ratios subtracted the risk-free rate | A rule that holds cash was charged for it twice (Sharpe 0.38 → 0.48 on the sample once fixed) | Cash earns the T-bill (live) or 2% (offline) in the strategy backtest and the portfolio walk-forward |
| The strategy backtest checked stops at the close and sold at the next open; the long horizon had no stop at all | Gaps and intraday falls through a stop were filled at the wrong price; the backtest and the ticket's stop order disagreed | Resting stop orders, filled at the stop or at the open on a gap |
| A fixed 10 bps per side regardless of liquidity | Small, illiquid or volatile stocks looked cheaper to trade than they are | Corwin-Schultz half spread + square-root impact (`lighthouse/costs.py`) |
| Shorts carried no borrow cost | The short setup test overstated shorts | A configurable borrow fee (default 0.5% a year, an assumption) in the short setup test, shown on every short ticket |

### Fifth October 2026 session (research harness)

| Issue | Effect | Fix |
|---|---|---|
| A cached Yahoo series with no volume (CPWR, 2011-2014: zero volume, ~$2 price) entered the panel | Its square-root impact cost was effectively infinite (up to 4,000,000 bps), and one such stock in a quintile swamped the leg's cost: net returns looked 30-40 points a year worse than they were | Members are tested only in months with volume on ≥ 15 of the last 21 sessions (115 member-months dropped); per-side cost capped at 300 bps |
| Experiment-log rows written by the hold-out had three more fields than the header | experiments.csv could not be read back, so the hold-out guard would have failed on the next run | Fixed log columns (`harness.LOG_COLUMNS`) with an `extra` JSON column for anything new; the day's 80 rows repaired in place (values unchanged) |
| The Ken French 49-industry file's first table was found only as "main" | Industry momentum could not load (the research run stopped) | The first table of a library file is also stored under its own title |
| The strategy backtest computed liquidity costs itself, with no cap | On a series that barely trades (EP's cached 2014 data: $0.80, a few hundred shares a day) a sale filled at a negative price and the equity curve went below zero (5 cohort stocks: CPWR, EP, COMS, DIGI, MCIC) | Per-side cost capped at 300 bps in the backtest and in `costs.CostModel` (`costs.MAX_COST_BPS`); sample-pack results unchanged |
| The DCF's free cash flow subtracted stock-based pay twice (once in the operating margin, once again) | Equity values understated by roughly the SBC share of cash flow (~15% for MSFT) | FCFF after SBC = EBIT x (1 - t) + D&A - capex - change in working capital; the cash view adds SBC back |

### Sixth October 2026 session (signal library, analyst toolkit on live data)

Every one of these was found before the library's hold-out was evaluated: by the parser tests recorded from real
FINRA and SEC files, by the first live research runs, or by the first live MSFT memo.

| Issue | Effect | Fix |
|---|---|---|
| FINRA writes class shares `BRK/B`; symbols were renamed after the daily sum | BRK-B and BF-B had no short volume; two raw symbols could share a label and the research run stopped ("duplicate labels") | Symbols normalised before summing; a session cached by the old code is normalised when read |
| Form 4 purchases of preferred stock, notes and warrants counted as insider buys | A $5.1m purchase of JPMorgan Series HH preferred counted as a buy of the common (about 200 of 30,895 rows in 2020 Q1) | Non-common securities left out |
| The Form 4 issuer symbol is free text (`LEN, LEN-B`, `(CALX)`, `NYSE: SCS`) | Those filings matched no ticker | Quotes, brackets and US exchange prefixes removed; lists split on commas and semicolons, never on spaces |
| Days to cover used FINRA's published figure | It is floored at 1.00 (41% of rows in December 2023, 7% of liquid stocks tied) and is 999.99 at zero volume | Short interest / average daily volume from the two quantities; zero volume is missing |
| An empty short-interest answer for a recent date was cached | A month still running asked for a future settlement date and stayed missing for good | Empty answers under 30 days old are not cached |
| A Form 4 dated the month-end session, and that session's FINRA short volume, were used at its close | A few hours of look-ahead: EDGAR dates Forms 4 accepted until 10 pm, FINRA posts the file in the evening | A Form 4 counts from the day after its filing date (as filings do everywhere else); short volume reads the five sessions before the month's last |
| 4/A amendments repeat whole forms; one 2020 Q1 row had the total typed into the price field ($1.56 trillion) | Trades double counted in `insider_net`; one row could dominate a stock's net purchases | 4/A rows that repeat a Form 4 transaction are dropped; a row above $5bn keeps its buyer, not its value |
| Some 2010 FINRA files have no header line | The parser stopped the run | Such files are read with the columns of their field count |
| The 10-K text workers took 8 SEC requests a second whatever share the run was given | Several processes together could exceed the SEC's 10 a second | The workers split `LIGHTHOUSE_SEC_RATE` |
| A company's XBRL facts were cached filtered to the concepts of the day | When the concept list grew, the new concepts read as never reported | The cache records its concepts and is downloaded again when asked for more (offline it is used as it is) |
| The sample pack predated the newer cash and short-term investment concepts | MSFT's offline net debt was +$70bn instead of net cash | Sample pack re-recorded (122 concepts, filing lists, profiles); MSFT offline net debt -$61bn; `cross_ticker --check` unchanged |
| The DCF's terminal value capitalised the last projected year's free cash flow | A company in a build-out carried that year's capex forever: MSFT (capex 35% of revenue, D&A 10%) was worth $94 a share and no growth rate up to 60% a year reproduced its price | Value-driver terminal value, NOPAT x (1 - g / RONIC) / (WACC - g), RONIC = today's ROIC floored at the WACC (Koller, Goedhart & Wessels, *Valuation*): MSFT base case $199 at the same 11.5% WACC; the price implies 36.6% growth a year for five years |
| The re-recorded sample pack was written by pandas 3, which stores dates in microseconds | pandas 2 (the newest on Python 3.10, which CI runs) refused to merge them with nanosecond dates: 40+ offline tests failed on CI's 3.10 job | Facts and filing lists are cast to nanoseconds when read; a test pins the unit |
| The deflated Sharpe ratio's variance of trials came from the signals evaluated in the same run | A run with `--signals NAME` judged the signal against itself (the earnings-release tone's hold-out used 0.055 a month instead of the log's 0.185); its verdict does not change (it failed tuning; deflated Sharpe 0.01) | `ExperimentLog.trial_sharpe_std`: the latest tuning row of every logged configuration |
| The post-mortem annualised trailing revenue growth over calendar days | One quarter's change over 167 days read as +9.6% a year (MSFT, May to October 2026) instead of about +18% | Annualised over the quarter ends behind the two trailing sums |

### Seventh October 2026 session

The data limits the sixth session documented and left open. The §8b results were computed before these fixes, on the
data as it was, and they stay as logged: every hold-out has been used, so running the research again would be a new
trial. Each row says which logged signals the fix touches and how much of them, as far as that can be counted offline.

| Issue | Effect | Fix |
|---|---|---|
| A Form 4 names the share class traded, in the filer's spelling: Berkshire's insiders file as `BRK.A`, Brown-Forman's as `BFA, BFB`, while the index holds BRK-B and BF-B | Their trades matched no member, so `insider_buyers` and `insider_net` read no insider activity for BRK-B (a member since February 2010: about 200 month ends) and BF-B (about 213, 2009-2026) in every month, whatever was filed. Each class of a two-class issuer (GOOGL and GOOG, FOXA and FOX, NWSA and NWS) saw only the filings that named it. How many of those member-months held a trade cannot be counted offline (the insider data sets are not in the repository); they are two to eight of the 350-450 names ranked each month | `positioning.member_trades`: a class share, and every class of an issuer with two classes in the index, takes every trade filed under the issuer's CIK (from EDGAR's ticker table, as the fundamental signals use it; every class files under one CIK, and a CIK is never reused, so look-alike symbols such as Brooks Automation's `BRKS` or Saul Centers' `BFS` stay with their own issuers). A Form 4 line listed under several class symbols counts once. Chosen for two-class issuers: both classes carry the company's insider trades, as they carry its fundamentals. Every other member is matched by its symbol as filed, as before (a predecessor registrant, such as Google Inc. before October 2015, too). Without the ticker table, matching is by symbol only |
| The NYSE calendar had the regular holidays only, not the one-off closures (11-14 September 2001, Hurricane Sandy, the national days of mourning) | The short-volume window (the five sessions before the month's last) asked FINRA for closed days, which have no file, and summed fewer sessions. In the research period (FINRA's files start in August 2009) one month end is touched: October 2012, whose window held 29 and 30 October (Hurricane Sandy) and summed three sessions instead of five, one of the 101 tuning months of `short_volume`; no hold-out month, month end or short-interest settlement date (`days_to_cover`) changes (computed from both calendars). The market clock differed only around a closure: from the close of 9 January 2025 to the next day's close it named 9 January the last finished session, and a bar from before a closure counted one session too many behind | `data.NYSE_SPECIAL_CLOSURES` (27 April 1994; 11-14 September 2001; 11 June 2004; 2 January 2007; 29-30 October 2012; 5 December 2018; 9 January 2025) are part of `nyse_holidays`, so every user of the calendar skips them: the short-volume window, the settlement dates, the cache's last session close, the live-mode rule and the segment tables' next session. Cache expiry today is unchanged. An unplanned closure cannot be known in advance: one announced later has to be added to the list |
| `edgar.bulk_quarter` cached a quarterly Financial Statement Data Set filtered to the tags asked for the first time | A tag asked for later read as never reported in that quarter, as the company-facts cache did before the sixth session. Nothing in the logged research reads the bulk data sets (the signals use the company facts), so no result is touched | As for the facts: the cache records the tags it holds (`edgar_fsds/<quarter>.json`) and is downloaded again when asked for more, keeping the union of what is asked, what it held and `pit.CONCEPTS`; a table cached before the record covers the tags it has rows for; offline or with EDGAR unreachable it is used as it is |

## 12. Speed

Measured offline with Streamlit's test harness on the sample pack (before → after this session's work): first page load
1.75 → 0.70 s; a click with an analysis on screen 2.88 → 0.43 s; with desk and portfolio results on screen 5.69 → 0.58 s;
a trade-desk scan of 8 tickers 6.66 → 1.53 s; a deep evaluation (layers, sensitivity grid, walk-forward) 7.9 → 1.4 s.
A cold live screen of the 12 mega caps took 41 s one ticker at a time and 8 s with 8 threads. What made the difference:

* **pandas copies `attrs` into every derived object** (a deep copy, every operation). The price frame carried the cleaning
  report and the quote, so indicator maths ran several times slower than it should. `analyze` now computes on attr-free
  frames and puts the metadata back on the result.
* **`lighthouse/memo.py`** memoises indicators, beacon scores, technical scores and the regime table on a content
  fingerprint of the input and on the parameters that can change them (trading thresholds and costs are excluded), and
  returns copies. A deep evaluation's ~80 backtests differ only in thresholds and now share one computation.
* **Plotly**: `add_vrect` / `add_hline` re-validate the whole layout on every call; the regime shading of a long history
  took 2.6 s. All shapes are now set in one layout update. The app keeps each figure on the result it was drawn from,
  so a click re-uses it, and the desk funds its ideas once per scan and settings.
* **Streamlit caches**: large read-only results use `st.cache_resource` (`st.cache_data` pickles megabytes on every hit).
* **Threads for the network** (`lighthouse/parallel.py`): live multi-ticker work runs in up to 8 threads; offline work
  stays serial because Python threads only add contention to pure CPU work.
* **Imports**: the app no longer loads matplotlib (`palette.py` holds the shared colours), and scipy loads only when an
  optimiser runs.

* **Fragments** (third October 2026 session): each interactive section is an `st.fragment`, so a click re-runs only its
  section: on the Analyse tab the trade ticket's side, the beacon filter, the chart lookback, the what-if sizer and the
  sensitivity metric; and each of the trade desk, portfolio builder, watchlist and data tabs as a whole. Measured in
  headless Chromium (click until Streamlit reports idle, median of 8, MSFT analysed): chart lookback 1.34 → 1.11 s, beacon
  filter 1.12 → 0.73 s, ticket side 1.20 → 0.79 s. What remains is mostly the browser redrawing that section's charts.
  The watchlist's "open the full analysis" button asks for a full re-run explicitly, because the Analyse tab lives
  outside its fragment.
* **Weekly points for long charts**: a price, backtest, walk-forward, comparison or portfolio chart longer than ~3 years
  (800 sessions) is drawn from each week's last session (volume summed over the week, drawdowns from the week's low, so
  the deepest drawdown survives; trade markers and every table keep daily values). The full-history price chart of a
  25-year analysis went from 3.2 MB to 0.7 MB of chart data and the backtest chart from 1.8 MB to 0.4 MB.

Every published number is unchanged by these changes: the cross-ticker table and the MSFT example report regenerate
byte for byte. What is still slow: a portfolio walk-forward test (6 methods × ~120 monthly optimisations, about 1.5 s
for 8 stocks), and a cold live scan, which is bound by Yahoo's response time (fundamentals are 7 requests per ticker).

## 13. Q&A preparation

* **Why these indicators?** Trend (200-day) is the most robust timing signal in the literature; RSI/MACD/price bands cover momentum, acceleration and range; volume validates moves; 52-week high is the simplest breakout measure. Fundamentals cover the three questions an analyst asks: is it cheap, is it good, is it growing.
* **Why should they predict returns?** Trend and momentum persistence are documented anomalies; quality and value premia are documented factors. We do not claim short-term prediction; we claim better risk-adjusted participation.
* **Why these thresholds?** Chosen *before* tuning for interpretability (60 = clearly positive evidence; 35 = clearly negative), then checked with a sensitivity grid: results form a plateau and in-sample tuning did not improve out-of-sample results.
* **Different market conditions?** The regime table shows the rule loses roughly half as much as buy-and-hold in Bear/Turbulent markets and keeps 60-65% exposure in Bull/Calm markets.
* **Sensitivity?** Sharpe across 25 threshold pairs on Microsoft ranged 0.08-0.20 (median 0.16, default 0.18); the biggest sensitivity is the review frequency and the stop design, which is why both are documented above.
* **Biggest weaknesses?** Whipsaw losses in long sideways markets; sector norms are assumptions; short fundamental history; survivorship in our test universe; no transaction-tax modelling.
* **What did GenAI contribute?** Code scaffolding, indicator implementations and the experiment harness. The team decided the customer, the signals, the weights, the exit design, what to reject, and how to present results honestly.
