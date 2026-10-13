# Lighthouse methodology — every choice, and why

This document exists so that the team can defend every number in the product. It follows the order of the pipeline.

## 1. Data

| Item | Choice | Why |
|---|---|---|
| Prices | Daily OHLCV, dividend- and split-adjusted, from Yahoo Finance (`yfinance`), fallback Stooq CSV | Free, no API key, adjusted series avoid false "crashes" at ex-dividend/split dates |
| History | 10 years by default (15 in the notebook), plus a 1.2-year warm-up so the 200-day SMA and 52-week high are defined on day one | Covers at least one bear market; indicators need 252 days |
| Benchmark | S&P 500 index (`^GSPC`) | The market the customer compares against; used for regime detection |
| Fundamentals | Yahoo snapshot (trailing/forward P/E, PEG, P/B, EV/EBITDA, ROE, margins, D/E, current ratio, FCF, growth) plus annual and quarterly statements | One valuation ratio and one quality/growth measure are required; we use several so a missing field never breaks the score |
| Cleaning | Sort, de-duplicate dates, drop rows without a positive close, fill missing O/H/L from the close, enforce High ≥ max(O,C) and Low ≤ min(O,C), disable volume if >50% missing, flag >50% one-day moves | Each step is counted and shown to the user (`CleaningReport`) |
| Offline pack | Real data shipped in open-source packages (Microsoft OHLCV 1986-2017 from `pmdarima`, 19 S&P stocks and the index 1990-2022 from `skfolio`) | The notebook and tests must run end-to-end without internet; provenance in `scripts/build_sample_pack.py` |
| Caching | Last live download cached in `data/cache/` for 20 hours | Yahoo rate-limits; the product stays usable |

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

**Point-in-time history.** For the backtest, each annual statement becomes usable 90 days after its period end (the 10-K filing deadline for large filers is 60 days; 90 is conservative). Quality and growth are fixed for the year; the P/E is recomputed daily from price / lagged EPS. Before the first statement the score is "unknown" and no gate is applied.

## 3. Technical beacons

All indicators are causal (rolling windows, `ewm(adjust=False)`): the value on day *t* uses only data up to *t*. A unit test truncates the series and checks that nothing in the past changes.

| Beacon | Formula | Reasoning | Weight long / short |
|---|---|---|---|
| Trend | 0.5·clip((Close/SMA200 − 1)/10%) + 0.5·clip((SMA50/SMA200 − 1)/5%) | Continuous Golden/Death Cross: fewer whipsaws than a binary cross | 35% / 20% |
| RSI(14) | linear from −1 at 30 to +1 at 70, then fading (+1 → −1 from 70 to 100) | Confirms momentum; extremes signal exhaustion | 15% / 20% |
| MACD(12/26/9) | histogram / ATR, clipped | Momentum acceleration, scale-free | 10% / 20% |
| Bollinger %B (20, 2σ) | 2·(%B − 0.5) inside the bands, fading above 1 / below 0 | Where price sits in its recent range; above the band = over-extended | 10% / 15% |
| Volume | 0.5·sign(20-day return)·clip((vol20/vol60 − 1)/0.5, 0, 1) + 0.5·sign(OBV − SMA20(OBV)) | A move on expanding volume is more trustworthy; excluded (grey) when volume is unavailable | 10% / 15% |
| 52-week high | +1 at the high, 0 at 20% below, −1 at 40% below | Breakout proximity; momentum literature | 20% / 10% |

Technical score = 50 + 50 × weighted mean, with weights re-normalised over available beacons. Long-term weights lean on slow beacons (trend, breakout); short-term on oscillators.

**Why the Bollinger Bands trail the price.** The bands are the 20-day simple moving average (the *middle band*, drawn dashed on
every chart) ± 2 population standard deviations of the same 20 closes, as John Bollinger defines them. A trailing 20-day average
lags price by (20 − 1) / 2 = 9.5 sessions, and after a sharp move the band width keeps growing until the move leaves the
20-day window. On the sample data the middle band lines up best with the close from 9–10 sessions earlier, exactly as
theory says. Centring the bands on the price would use the next ~10 days of prices, which is look-ahead, so they are not
shifted.

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
* Fills at the next day's open (close when no open data), 5 bps commission + 5 bps slippage per side.
* Metrics: total return, CAGR, volatility, Sharpe and Sortino (2% risk-free), max drawdown, Calmar, trades, win rate, average win/loss, profit factor, exposure; plus layer attribution, sensitivity grid (entry 50-70 × exit 25-45), walk-forward (tune on 60%, test on 40%) and regime breakdown.
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

See `reports/examples/cross_ticker_evaluation.csv` (20 stocks, 1990-2022). Averages: buy-and-hold CAGR 13.3%, vol 33.5%, Sharpe 0.50, max DD −69%; Lighthouse balanced CAGR 6.4%, vol 12.7%, Sharpe 0.39, max DD −32%; conservative vol 8.8%, max DD −23%. Lower drawdown and volatility in 20/20 stocks; higher Sharpe in 4/20.

Interpretation: on a universe of survivors that compounded at 13% a year, a rule that is sometimes in cash cannot win on return. The product's value is that it delivers *the risk profile the investor asked for* — a balanced investor who said "15% volatility, lose at most 2% per idea" actually gets ~13% volatility and a defined worst case — while keeping most of the up-trend exposure. We state this plainly rather than hiding it behind a cherry-picked ticker.

## 9. Trade tickets: long and short (`lighthouse/tradeplan.py`)

| Item | Rule | Reasoning |
|---|---|---|
| Long setup | Actionable when the verdict is Buy / Strong Buy **and** the trend beacon is positive (the backtested entry gate) | The ticket only fires on what was tested |
| Short setup | Actionable when the composite ≤ 40, the trend beacon is negative, the technical score ≤ 40, and the market is not Bull / Calm | Mirror of the long rule around 50 (Buy ≥ 60 ↔ short ≤ 40; Bear / Turbulent blocks longs ↔ Bull / Calm blocks shorts); bull markets halve short size |
| Watch list | Positive (negative) evidence without the trend confirmation, or a Hold with a strong technical score | Levels are shown; no order |
| Stop-loss | Longs: the horizon's stop (200-day SMA − 8% long term, 2.5 × ATR short term). **Shorts always use the short-horizon ATR stop** | A short's loss is open-ended and borrow fees accrue daily; a wide SMA stop would make its targets meaningless |
| Size | The three risk rules as % of the account × conviction × regime, capped by the cash; shorts are capped at 1:1 collateral (Regulation T would allow 2:1) | Same engine as the Position plan, so the long ticket and the plan agree |
| Entry | Limit order at the last price; skip if price has run 0.5 × ATR beyond it | Do not chase |
| Targets | Target 1 = entry ± 2R, target 2 = entry ± 3R (R = entry-to-stop distance; editable) | R-multiples express reward in units of risk; a 2R target needs a win rate above 1/3 to pay |
| Management | Sell 50% at target 1, move the stop on the rest to breakeven, then trail it at the best close ∓ k × ATR (chandelier exit); time stop after 40 (short horizon) / 252 (long) sessions | Locks in gains, caps dead money |
| Portfolio heat | Ideas are funded best-first until the cash, the idea count (5) or the total planned risk (6% of the account) runs out; leftovers smaller than 25% of an idea's full size are not funded | Limits correlated losses across many open trades |

**Historical setup test.** For every past day the stock met the (point-in-time) setup rule, the trade is entered at the next
session's open (the close for close-only data) and managed with exactly the rules above, one trade at a time, after costs.
Gaps fill at the open; when a stop and a target fall in the same session the stop is assumed first. Results are in R
(expectancy, win rate, profit factor, share of trades reaching target 1 / stopped / timed out). A ticket whose rules lost
money on that stock is flagged in red. Shorts are simulated on the dividend-adjusted series, so the dividends a short seller
owes are reflected; borrow fees are not.

**Position monitor.** An open position is replayed from the session after its entry with the same rules. The monitor reports a
stop or target hit (with the date), how far price is from the stop (in % and ATR; "near stop" within 0.5 ATR), the current
breakeven or trailing stop, a due time stop, and whether the latest verdict has turned against the trade.

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
| Walk-forward test | Monthly (or quarterly) rebalance; weights from the trailing window only; trades executed at the **next** day's close; 10 bps per unit of turnover; weights drift between rebalances; cash earns 0% | No look-ahead (unit-tested by truncation) |

**Goals.** *Capital preservation* picks the 10 calmest stocks and uses minimum variance with a 12% volatility target. *Balanced*
picks the 10 best technical scores (≥ 45) and uses risk parity. *Growth* picks the 8 strongest 12-1 month momentum names in an
up-trend and uses maximum Sharpe. *Maximum diversification* picks 15 names (technical score ≥ 45) and uses HRP. At most 3 stocks
per sector and sector weight caps apply. The walk-forward test **re-selects** the stocks every rebalance with only the data
available then, so it does not benefit from knowing today's winners. That matters: the back-cast of today's picks over the
last year is inflated by how they were chosen, so the app shows it for risk only and uses the walk-forward test for performance.

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
| Bollinger middle band computed but never drawn | The bands appeared to float around nothing | Dashed 20-day SMA on both chart sets, plus the lag explanation above |
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

## 12. Q&A preparation

* **Why these indicators?** Trend (200-day) is the most robust timing signal in the literature; RSI/MACD/Bollinger cover momentum, acceleration and range; volume validates moves; 52-week high is the simplest breakout measure. Fundamentals cover the three questions an analyst asks: is it cheap, is it good, is it growing.
* **Why should they predict returns?** Trend and momentum persistence are documented anomalies; quality and value premia are documented factors. We do not claim short-term prediction; we claim better risk-adjusted participation.
* **Why these thresholds?** Chosen *before* tuning for interpretability (60 = clearly positive evidence; 35 = clearly negative), then checked with a sensitivity grid: results form a plateau and in-sample tuning did not improve out-of-sample results.
* **Different market conditions?** The regime table shows the rule loses roughly half as much as buy-and-hold in Bear/Turbulent markets and keeps 60-65% exposure in Bull/Calm markets.
* **Sensitivity?** Sharpe across 20 threshold pairs on Microsoft ranged 0.05-0.20 (default 0.18); the biggest sensitivity is the review frequency and the stop design, which is why both are documented above.
* **Biggest weaknesses?** Whipsaw losses in long sideways markets; sector norms are assumptions; short fundamental history; survivorship in our test universe; no transaction-tax modelling.
* **What did GenAI contribute?** Code scaffolding, indicator implementations and the experiment harness. The team decided the customer, the signals, the weights, the exit design, what to reject, and how to present results honestly.
