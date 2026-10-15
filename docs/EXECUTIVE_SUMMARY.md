# Lighthouse — Executive Summary

*An explainable stock decision engine for self-directed investors. "A clear signal in a noisy market."*

## The customer and the problem

Our customer is the **self-directed investor**: someone with a brokerage account, a long-term portfolio of five to
thirty stocks, and a day job. They are not day traders. Before they buy or sell they want a disciplined second
opinion — and today they cannot get one that is both *explainable* and *personal*. Brokerage apps show data, not
decisions. Finance sites and social media give opinions without the evidence. Generic AI chat gives confident
answers with no backtest, no risk plan and no idea how much the user can afford to lose.

Lighthouse answers the three questions that investor actually has: **Should I buy this? How much? Where would I be
wrong?** — and shows its work.

## What the product does

The investor enters a ticker and a few facts about themselves: horizon (long or short), risk tolerance
(conservative / balanced / aggressive), the most they accept losing on one position (as % of the whole account), the
cash they can invest now and, optionally, the value of the stocks they already hold. Lighthouse then runs automatically:

1. **Collects and cleans the data** — ten-plus years of daily prices and volume, the S&P 500, and fundamentals
   from the company's SEC filings since 2009, each number dated by the filing that first reported it. Every cleaning
   step and every data source is reported. If the internet is unavailable it falls back to a bundled historical
   sample pack (real prices and recorded SEC filings) so it always runs.
2. **Fundamental analysis** — thirteen *beacons* (P/E, forward P/E, PEG, P/B, EV/EBITDA, earnings stability, ROE,
   operating margin, debt/equity, free-cash-flow yield, current ratio, revenue growth, EPS growth), each compared with
   a sector norm or an absolute band and voting between −1 and +1. Valuation weighs 40%, quality 35%, growth 25%.
   The investment memo can also score them against the median of real peers (same industry, closest in size).
3. **Technical analysis** — six beacons: trend (50/200-day SMA, Golden/Death Cross), RSI(14), MACD(12/26/9),
   position in low-lag price bands (%B), volume confirmation (OBV and 20/60-day volume ratio) and distance from the 52-week high.
4. **Market weather** — the S&P 500 is classified Bull/Bear (above/below its 200-day average) and Calm/Turbulent
   (realised volatility vs. its one-year norm). Bear markets cost points and halve new positions; Bear/Turbulent
   blocks new entries.
5. **The decision** — Composite = 60% fundamental + 40% technical for long-term investors (30/70 for short-term),
   minus the weather penalty, mapped to **Strong Buy / Buy / Hold / Reduce / Sell**. The position plan takes the
   *smallest* size allowed by three independent rules — the investor's max-loss limit, a volatility target and a
   concentration cap — and scales it by conviction. It states the shares to buy, the exit level and the worst-case
   dollar loss.
6. **Honest evidence** — a point-in-time backtest of the trading rule versus buy-and-hold and the S&P 500, with
   layer attribution, a threshold-sensitivity grid, an out-of-sample test and a breakdown by market regime.
7. **Watchlist ranking** — the same engine applied to a list of tickers.
8. **Trade desk** — long and short tickets sized from the investor's cash: limit entry, stop-loss and the exits
   (long-term buys ride the trend until a month-end close below the 200-day SMA − 8% or a faded technical score, the
   backtest's own exits; short-term trades and shorts take profit at two levels in multiples of the risk, with
   breakeven, trailing and time stops), a cap on total open risk, and how those exact rules did on each stock's history. Each ticket lists every way out, for a profit or a loss, with the price,
   the result and how often it happened before; orders are priced from a timestamped live quote.
9. **Portfolio builder** — a basket (the investor's or one picked for a goal) allocated by risk parity, minimum
   variance, hierarchical risk parity, maximum Sharpe or simpler rules on a shrunk covariance matrix, with VaR,
   expected shortfall, beta, risk contributions, stress tests and a walk-forward test of every method.

10. **Analyst memo and thesis tracker** — for one company: a linked three-statement model, bull / base / bear
   DCFs and the growth the price implies, peers, industry metrics (banks, REITs, software, insurers), business
   segments, red flags, a catalyst calendar, and a thesis whose kill criteria are checked on every new filing.

It ships as a Colab/Jupyter notebook, a command-line tool and a point-and-click Streamlit app. A non-programmer
changes nothing but the inputs cell.

## Investment thesis and decision process

We buy quality businesses when the market agrees with us. Fundamentals decide *what* deserves capital (profitable,
reasonably valued, growing companies); technicals decide *when* (price above its 200-day trend, momentum
confirming, not blocked by a hostile market). Neither alone is enough: a cheap stock in a downtrend keeps getting
cheaper, and a strong chart on a loss-making company is speculation. Risk management decides *how much*: no single
idea may cost more than the investor's stated maximum loss.

The trading rule behind the backtest is deliberately slow for long-term investors: reviewed **monthly**, enter when
the trend is up and the technical score is at least 60, exit when the score falls to 35 or price closes 8% below
its 200-day average. Short-term investors get a weekly review and an ATR trailing stop.

## Key results (what we found, honestly)

Example: Microsoft, long-term balanced investor, $25,000 portfolio, 2% max loss (offline sample, as of Nov 2017):
**Buy, score 71** (fundamental 63, technical 83, market Bull/Calm). Plan: 19 shares ≈ $1,594 (6.6% of the
portfolio), exit level $64.73, worst case $364 — the max-loss rule, not the risk preset, limited the size because
the exit sits 23% below the price.

Rule tested on 20 S&P 500 stocks, December 1990 – December 2022 (costs scaled to each stock's liquidity, stops as
resting orders, next-day fills, uninvested cash earning 2%; `reports/examples/cross_ticker_evaluation.csv`):

| | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & hold | 13.3% | 33.5% | 0.50 | −69% |
| Lighthouse signals + regime, fully invested when in | 10.9% | — | 0.47 | −55% |
| **Lighthouse, balanced profile (vol-targeted)** | 7.3% | 12.2% | 0.47 | **−27%** |
| Lighthouse, conservative profile | 5.8% | 8.5% | 0.47 | −18% |

Lighthouse cut the maximum drawdown and the volatility of **every one of the 20 stocks**, kept the balanced
profile's realised volatility below its 15% target (12.2% on average), and averaged 12 trades per stock in 32 years
with a 53% win rate. It did **not** beat buy-and-hold on raw return except on one name (GE) and beat it on Sharpe ratio for
9 of 20: these are survivors that compounded relentlessly, and any rule that is sometimes in cash trails them. Its value
is risk-managed exposure: in Bear/Turbulent markets the Microsoft rule lost 18% annualised while buy-and-hold lost 31%.
On Microsoft, thresholds tuned on the first 60% of the history did a little better on the last 40% than the defaults
(Sharpe 0.47 against 0.42), and across 25 threshold pairs the Sharpe ratio moved within 0.06-0.21: the defaults sit in
a broad, flat region, so tuning buys little.

Survivorship-aware tests are harsher. On every S&P 500 member of 2 January 1996 (258 of 487 with free prices) the
balanced rule's Sharpe averaged 0.24 against buy-and-hold's 0.36; on the members of 2 January 2009 (330 of 498), 0.30
against 0.48. And the scores do not pick stocks: on the point-in-time S&P 500, 2009-2026, neither the product's own
scores nor any of 37 further signals built from free data (value, quality, earnings, momentum, insider and short-sale
activity, institutional breadth, filing text) ranked next month's returns well enough to survive a hold-out tested
once. The product's value is the risk discipline, not stock selection (`docs/METHODOLOGY.md` §8, §8b).

## What makes Lighthouse different

* **Explainable by construction** — every point of the score is traceable to a named beacon with a one-line reason.
* **Personal** — the same stock gets a different size and exit level for different investors; the max-loss rule is
  the investor's, not ours.
* **Market-aware** — the rule changes with the market's weather instead of pretending every year is 2017.
* **Honest testing** — no look-ahead (every SEC figure is used from the session after its filing), costs scaled to
  liquidity, survivorship-aware cohorts, a research hold-out evaluated once with the trial count behind every
  verdict, and a plain list of what we tried and rejected (ATR stops on long-term positions, daily reviews, "don't
  chase" filters, 47 cross-sectional signal configurations).
* **Runs anywhere** — live data when online, bundled real data when offline; 330 automated tests.

## Major limitations and risks

Single-stock timing is hard: in long sideways markets (Microsoft 2002–2012) the rule bleeds small losses, and in
relentless bull runs it holds too little. Sector norms are long-run approximations (peer medians are an option in the
memo). SEC fundamentals start in 2009, so the point-in-time fundamental gate is off before then. Free data has no
prices for most delisted companies, so every survivorship-aware result is a bound, not an estimate. The backtest
ignores taxes; borrow fees for shorts are an assumption. Lighthouse is decision support, not advice: it shows the
evidence and its reasoning; the decision stays with the investor.
