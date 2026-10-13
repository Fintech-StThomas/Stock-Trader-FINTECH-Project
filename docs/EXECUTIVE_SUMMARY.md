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
   (valuation ratios, profitability, leverage, cash flow, growth, up to five years of annual statements). Every
   cleaning step and every data source is reported. If the internet is unavailable it falls back to a bundled
   historical sample pack so it always runs.
2. **Fundamental analysis** — thirteen *beacons* (P/E, forward P/E, PEG, P/B, EV/EBITDA, earnings stability, ROE,
   operating margin, debt/equity, free-cash-flow yield, current ratio, revenue growth, EPS growth), each compared with
   a sector norm or an absolute band and voting between −1 and +1. Valuation weighs 40%, quality 35%, growth 25%.
3. **Technical analysis** — six beacons: trend (50/200-day SMA, Golden/Death Cross), RSI(14), MACD(12/26/9),
   Bollinger %B, volume confirmation (OBV and 20/60-day volume ratio) and distance from the 52-week high.
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
8. **Trade desk** — long and short tickets sized from the investor's cash: limit entry, stop-loss, two take-profit
   levels in multiples of the risk, breakeven and trailing stops, a time stop, a cap on total open risk, and how those
   exact rules did on each stock's history. A position monitor flags stop hits and positions close to their stop.
9. **Portfolio builder** — a basket (the investor's or one picked for a goal) allocated by risk parity, minimum
   variance, hierarchical risk parity, maximum Sharpe or simpler rules on a shrunk covariance matrix, with VaR,
   expected shortfall, beta, risk contributions, stress tests and a walk-forward test of every method.

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
**Strong Buy, score 73** (fundamental 63, technical 88, market Bull/Calm). Plan: 21 shares ≈ $1,768 (7.1% of the
portfolio), exit level $64.73, worst case $403 — the max-loss rule, not the risk preset, limited the size because
the exit sits 23% below the price.

Rule tested on 20 S&P 500 stocks, December 1990 – December 2022 (10 bps costs per side, next-day fills):

| | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & hold | 13.3% | 33.5% | 0.50 | −69% |
| Lighthouse signals + regime, fully invested when in | 10.1% | — | 0.44 | −58% |
| **Lighthouse, balanced profile (vol-targeted)** | 6.4% | 12.7% | 0.39 | **−32%** |
| Lighthouse, conservative profile | 4.6% | 8.8% | 0.32 | −23% |

Lighthouse cut the maximum drawdown and the volatility of **every one of the 20 stocks**, kept the balanced
profile's realised volatility close to its 15% target, and averaged 13 trades per stock in 32 years with a 55% win
rate. It did **not** beat buy-and-hold on raw return or Sharpe ratio for most names (4 of 20 on Sharpe): these are
survivors that compounded relentlessly, and any rule that is sometimes in cash trails them. Its value is
risk-managed exposure: in Bear/Turbulent markets the Microsoft rule lost 17% annualised while buy-and-hold lost 31%.
The out-of-sample test shows that tuning thresholds on the first 60% of history adds nothing on the last 40% — the
default rule is a plateau, not a lucky spike.

## What makes Lighthouse different

* **Explainable by construction** — every point of the score is traceable to a named beacon with a one-line reason.
* **Personal** — the same stock gets a different size and exit level for different investors; the max-loss rule is
  the investor's, not ours.
* **Market-aware** — the rule changes with the market's weather instead of pretending every year is 2017.
* **Honest testing** — no look-ahead, statements lagged 90 days, costs, out-of-sample and sensitivity tests, and a
  plain list of what we tried and rejected (ATR stops on long-term positions, daily reviews, "don't chase" filters).
* **Runs anywhere** — live data when online, bundled real data when offline; 36 automated tests.

## Major limitations and risks

Single-stock timing is hard: in long sideways markets (Microsoft 2002–2012) the rule bleeds small losses, and in
relentless bull runs it holds too little. Sector norms are long-run approximations, not live peer data. Yahoo
Finance provides only four to five years of statements, so the point-in-time fundamental gate acts only in recent
years. The backtest ignores taxes and checks stops at the close. The evaluation universe is survivorship-biased in
favour of buy-and-hold. Lighthouse is decision support, not advice: it shows the evidence and its reasoning; the
decision stays with the investor.
