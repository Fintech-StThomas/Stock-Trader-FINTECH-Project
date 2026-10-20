# Lighthouse — 5-minute pitch script and demo plan

**Format:** 5-minute product launch + 3-minute Q&A. Present the product and the investment logic, not the code.

## Slide outline (8 slides, ~35 seconds each)

1. **Hook (0:00-0:30)** — "Every investor has been here: you *like* a stock, your app shows you forty numbers and a green chart, and still nobody will answer the questions that matter: *Should I buy it now? How much? Where am I wrong? And what am I actually buying?*"
2. **The customer (0:30-1:00)** — The self-directed investor: brokerage account, 5-30 stocks, a day job. Not a day trader. Wants a disciplined second opinion they can understand and a risk plan sized to *their* portfolio.
3. **The product (1:00-1:45)** — Lighthouse. Type a ticker, tell it who you are (horizon, risk tolerance, the most you'll lose on one idea, the cash you have). Seconds later: a **verdict**, the **why** (every signal is a beacon — green, amber, red — with a one-line reason), a **trade ticket** (shares, entry, stop, exits, worst-case dollars) and, beside it, an **analyst memo** from the company's own SEC filings: what it's worth on three scenarios, what growth the price assumes, the red flags, and what would prove the thesis wrong.
4. **The investment logic (1:45-2:30)** — "We buy quality businesses when the market agrees with us." Fundamentals decide *what* (valuation, quality, growth — 10 beacons from the company's SEC filings vs sector norms). Technicals decide *when* (trend above the 200-day line, momentum confirming — 6 beacons). Market weather decides *whether now* (Bull/Bear × Calm/Turbulent). Your max-loss rule decides *how much*. The memo goes deeper into the filings: the DCF, the red flags, what would prove you wrong.
5. **Live demo (2:30-3:30)** — see below.
6. **The evidence (3:30-4:15)** — 20 S&P 500 stocks, 32 years, liquidity-scaled costs, no look-ahead: Lighthouse cut the maximum drawdown and volatility of **every single stock** (average −27% vs −69%) and kept a balanced investor near the 15% volatility target. Then the honest part, on every S&P 500 member of 1996 and of 2009, not just today's winners: it trails buy-and-hold, and a test we wrote down *before* running it shows the timing adds nothing over simply holding the same average exposure. "So we don't sell you a timing edge. We sell you the risk you asked for, the reasons, and the memo — and we publish the test that went against us."
7. **Why us, not a finance site or a chatbot (4:15-4:45)** — explainable by construction, personal (same stock, different plan for different people), market-aware, honestly tested (pre-registered tests, a hold-out used once, every trial logged), runs anywhere (even offline), 292 automated tests.
8. **Ask / close (4:45-5:00)** — "Lighthouse doesn't promise to beat the market. It promises you'll never again buy a stock without knowing why, how much, where you're wrong, and what the business looks like." Vote for the product you'd actually use.

## Live demo (60 seconds, pre-loaded)

1. Streamlit app, sidebar already filled: MSFT, long-term, balanced, max loss 2%, $25,000 cash, data source `sample`. Press **Analyse**.
2. Point at the cards: *Strong Buy, 72/100 — fundamental 65, technical 83, weather Bull/Calm; the memo says HOLD.* "The business scores well, the chart says go, and the DCF says the price is about fair."
3. Plan & ticket tab: *20 shares ≈ $1,677 (6.8%), exit $64.73, worst case $383. Size limited by the max-loss rule.* "The exit sits 23% below the price, so your 2% rule sizes it small — that is the product protecting you from yourself."
4. Analyst memo tab: the DCF's weighted value $77.88 against $83.87, and "the price implies 7.8% revenue growth a year for five years, close to Microsoft's own trend: no variant view." Scroll to *What would prove this wrong*.
5. Switch risk to *aggressive* and horizon to *short*: Buy at 71 (the score now leans on technicals, 70%), size jumps to 26% of the account (76 shares, held back by the concentration cap), the exit tightens to an ATR stop at $81.56. "Same stock, different investor, different plan."
6. Backtest tab: the rule's drawdown was −33% against buy-and-hold's −59% over 2007-2017 — and the *same exposure, never traded* line next to it shows −30% with more return. "That line is why we don't claim a timing edge."

Fallback if the network is down: the demo already runs on the bundled sample data (`sample`), identical offline.

## Q&A — prepared answers

* **Why these indicators / thresholds / weights?** → `docs/METHODOLOGY.md` sections 3-6. Short version: the most robust, widely studied price signals; thresholds set for interpretability first and checked with a sensitivity grid; weights follow the horizon (long-term leans on trend and breakouts, short-term on momentum oscillators).
* **Do the fundamentals help?** → §2 and §8b, honestly: on the point-in-time S&P 500, 2009-2026, none of 37 signals from free data (value, quality, earnings, insider, short interest, filing text) survived a hold-out tested once, and the backtest's fundamental gate costs a little return on the sample stocks (Sharpe 0.470 vs 0.483 without). We keep them because a verdict on a business should look at the business, and we publish both numbers.
* **Does the timing work?** → §8d, pre-registered: no. Holding the rule's average exposure without trading earned more on most stocks of both cohorts; the drawdown saving appeared in 1996 and reversed in 2009; a plain 200-day trend rule did about as well. What the product delivers is sizing to your risk.
* **How does it behave in different markets?** → regime table (MSFT, 2007-2017): −18% a year against buy-and-hold's −45% in Bear/Turbulent markets, ~59% exposure in Bull/Calm.
* **Biggest weaknesses?** → no demonstrated timing or stock-selection edge; whipsaw losses in long sideways markets; the memo's sector norms and DCF inputs are assumptions; SEC fundamentals start in 2009; free data has no prices for most delisted companies, so survivorship-aware results are bounds; no taxes.
* **What did GenAI do and what did you decide?** → GenAI wrote scaffolding and indicator code and ran our experiment harness; we chose the customer, the signals, the monthly review, the SMA-based exit, what to cut (shorts, estimation-heavy portfolios), what to pre-register, and insisted on reporting the tests that went against us.
