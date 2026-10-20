# Lighthouse — Executive Summary

*An explainable stock decision engine for self-directed investors. "A clear signal in a noisy market."*

## The customer and the problem

Our customer is the **self-directed investor**: someone with a brokerage account, a long-term portfolio of five to
thirty stocks, and a day job. They are not day traders. Before they buy or sell they want a disciplined second
opinion — and today they cannot get one that is both *explainable* and *personal*. Brokerage apps show data, not
decisions. Finance sites and social media give opinions without the evidence. Generic AI chat gives confident
answers with no backtest, no risk plan and no idea how much the user can afford to lose.

Lighthouse answers the questions that investor actually has: **Should I buy this now? How much? Where would I be
wrong? And what does the business look like in its own filings?** — and shows its work.

## What the product does

The investor enters a ticker and a few facts about themselves: horizon (long or short), risk tolerance
(conservative / balanced / aggressive), the most they accept losing on one position (as % of the whole account), the
cash they can invest now and, optionally, the value of the stocks they already hold. Lighthouse then runs automatically:

1. **Collects and cleans the data** — ten-plus years of daily prices and volume, the S&P 500, and the company's SEC
   filings since 2009, each number dated by the filing that first reported it. Every cleaning step and every data
   source is reported. Offline it falls back to a bundled historical sample pack (real prices and recorded SEC
   filings), so it always runs.
2. **Technical analysis** — six *beacons*, each voting between −1 and +1 with a one-line reason: trend (50/200-day
   SMA, Golden/Death Cross), RSI(14), MACD(12/26/9), position in low-lag price bands (%B), volume confirmation (OBV and
   20/60-day volume ratio) and distance from the 52-week high.
3. **Market weather** — the S&P 500 is classified Bull/Bear (above/below its 200-day average) and Calm/Turbulent
   (realised volatility vs. its one-year norm). Bear markets cost points and halve new positions; Bear/Turbulent
   blocks new entries.
4. **The decision** — Composite = 60% fundamental + 40% technical for long-term investors (30/70 for short-term),
   minus the weather penalty, mapped to **Strong Buy / Buy / Hold / Reduce / Sell**; the fundamental score comes from
   ten beacons on the SEC filings against sector norms. The position plan takes the *smallest* size allowed by three independent rules — the investor's max-loss
   limit, a volatility target and a concentration cap — scales it by conviction and pays it from the cash available.
   It states the shares to buy, the exit level and the worst-case dollar loss.
5. **Trade ticket** — limit entry, stop-loss and the exits (long-term buys ride the trend until a month-end close
   below the 200-day SMA − 8% or a faded technical score, the backtest's own exits; short-term trades take profit at
   two levels in multiples of the risk, with breakeven, trailing and time stops), how those exact rules did on the
   stock's own history, and every way out listed before the investor enters. Orders are priced from a timestamped
   live quote; a scan of a whole universe funds the best setups within a total-risk cap.
6. **Analyst memo** — the fundamental case, read from the SEC filings as they were filed: a linked three-statement
   model, bull / base / bear DCFs and the growth the price implies, the stock's own valuation bands, fundamental
   beacons against industry peers or sector norms, earnings-quality and balance-sheet checks, industry metrics (banks,
   REITs, software, insurers), business segments, a catalyst calendar, the variant view, and a thesis whose kill
   criteria are checked on every new filing. Its stance sits next to the verdict and does not change the score.
7. **Honest evidence** — a point-in-time backtest of the trading rule against buy-and-hold, the S&P 500 and two
   yardsticks for timing (the rule's own average exposure held without trading, and a plain 200-day trend rule), with
   layer attribution, a threshold-sensitivity grid, an out-of-sample test and a breakdown by market regime.
8. **Portfolio builder** — a basket (the investor's or one picked for a goal) allocated by equal weight, risk parity
   or minimum variance on a shrunk covariance matrix, with VaR, expected shortfall, beta, risk contributions, stress
   tests and a walk-forward test of every method.

It ships as a Colab/Jupyter notebook, a command-line tool and a point-and-click Streamlit app with four tabs. A
non-programmer changes nothing but the inputs cell.

## Investment thesis and decision process

**We buy quality businesses when the market agrees with us.** Fundamentals decide *what* deserves capital
(profitable, reasonably valued, growing companies); technicals decide *when* (price above its 200-day trend, momentum
confirming, not blocked by a hostile market). Risk management decides *how much*: no single idea may cost more than
the investor's stated maximum loss. The memo sets the price against the business: what it is worth on
three explicit scenarios, what growth the price already assumes, and the measurable facts that would prove a thesis
wrong.

What the evidence says about the fundamentals, honestly: on the point-in-time S&P 500, 2009-2026, no fundamental,
quality, earnings, short-interest or text signal survived a hold-out tested once (`docs/METHODOLOGY.md` §8b), and the
backtest's fundamental gate costs a little return on the sample stocks (Sharpe 0.470 with it, 0.483 without). We keep
them in the score because a verdict on a business should look at the business; the numbers are published either way.

The trading rule is deliberately slow for long-term investors: reviewed **monthly**, enter when the trend is up and
the technical score is at least 60, exit when the score falls to 35 or the price closes 8% below its 200-day average.
Short-term investors get a weekly review and an ATR trailing stop.

## Key results (what we found, honestly)

Example: Microsoft, long-term balanced investor, $25,000 of cash, 2% max loss (offline sample, as of November 2017):
**Strong Buy, score 72** (fundamental 65, technical 83, market Bull/Calm). Plan: 20 shares ≈ $1,677 (6.8% of the
account), exit level $64.73, worst case $383 — the max-loss rule limited the size because the exit sits 23% below the price. The memo,
from the filings public that day, says **hold**: a probability-weighted value of $77.88 (−7%), and the price implies
7.8% revenue growth a year for five years, close to the company's own trend.

Rule tested on 20 S&P 500 stocks, December 1990 – December 2022 (costs scaled to each stock's liquidity, stops as
resting orders, next-day fills, uninvested cash earning 2%; `reports/examples/cross_ticker_evaluation.csv`):

| | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & hold | 13.3% | 33.5% | 0.50 | −69% |
| Lighthouse signals + regime, fully invested when in | 10.9% | — | 0.47 | −55% |
| **Lighthouse, balanced profile (vol-targeted)** | 7.3% | 12.2% | 0.47 | **−27%** |
| Lighthouse, conservative profile | 5.8% | 8.5% | 0.47 | −18% |
| The balanced rule's average exposure (43%), never traded | 8.0% | 13.0% | 0.50 | −33% |
| Plain 200-day trend rule | 9.3% | — | 0.41 | −59% |

Lighthouse cut the maximum drawdown and the volatility of **every one of the 20 stocks** and kept the balanced
profile's realised volatility below its 15% target. It did **not** beat buy-and-hold on raw return, and the
honest yardstick shows why: holding the stock and cash in the rule's own average proportion, never trading, earned a
little more (8.0% against 7.3% a year) with a deeper drawdown (−33% against −27%). The timing buys a smaller drawdown,
not extra return. It does beat a
plain 200-day trend rule on Sharpe on 14 of the 20 stocks.

Survivorship-aware tests are harsher (run on the price-only rule of 15 October, before the fundamental gate
came back). On every S&P 500 member of 2 January 1996 (260 of 487 with free prices) the
balanced rule's median Sharpe was 0.28 against buy-and-hold's 0.37; on the members of 2 January 2009 (331 of 498), 0.33
against 0.48. A test we pre-registered (its rule committed before the run) asked whether the timing adds anything, and
it does not: holding the rule's own average exposure without trading earned more on 73% (1996) and 88% (2009) of the
stocks, the drawdown saved against that exposure in 1996 turned into a loss in 2009, and a plain 200-day trend rule did
about as well on Sharpe. Nor do the scores pick stocks: on the point-in-time S&P 500, 2009-2026, neither the product's
own scores nor any of 37 further signals built from free data (value, quality, earnings, momentum, insider and
short-sale activity, institutional breadth, filing text) ranked next month's returns well enough to survive a hold-out
tested once (56 trials logged). The product's value is the risk discipline, the explanation and the memo, not stock
selection or timing (`docs/METHODOLOGY.md` §8-8d).

## What makes Lighthouse different

* **Explainable by construction** — every point of the score is traceable to a named beacon with a one-line reason;
  every number in the memo is traceable to a filing.
* **Personal** — the same stock gets a different size and exit level for different investors; the max-loss rule is
  the investor's, not ours.
* **Market-aware** — the rule changes with the market's weather instead of pretending every year is 2017.
* **Honest testing** — no look-ahead (every SEC figure is used from the session after its filing), costs scaled to
  liquidity, survivorship-aware cohorts, yardsticks that ask what the timing adds, a research hold-out evaluated once
  with the trial count behind every verdict, tests whose rule was written before the run, and a plain list of what
  we tried and rejected.
* **Runs anywhere** — live data when online, bundled real data when offline; 293 automated tests.

## Major limitations and risks

Single-stock timing is hard: in long sideways markets (Microsoft 2002–2012) the rule bleeds small losses, and in
relentless bull runs it holds too little. The memo's sector norms and DCF inputs are assumptions it shows. SEC
fundamentals start in 2009. Free data has no prices for most delisted companies, so every survivorship-aware result is
a bound, not an estimate. The backtest ignores taxes. Lighthouse is decision support, not advice: it shows the
evidence and its reasoning; the decision stays with the investor.
