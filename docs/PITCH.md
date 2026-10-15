# Lighthouse — 5-minute pitch script and demo plan

**Format:** 5-minute product launch + 3-minute Q&A. Present the product and the investment logic, not the code.

## Slide outline (8 slides, ~35 seconds each)

1. **Hook (0:00-0:30)** — "Every investor has been here: you *like* a stock, your app shows you forty numbers and a green chart, and still nobody will answer the only three questions that matter: *Should I buy it? How much? Where am I wrong?*"
2. **The customer (0:30-1:00)** — The self-directed investor: brokerage account, 5-30 stocks, a day job. Not a day trader. Wants a disciplined second opinion they can understand and a risk plan sized to *their* portfolio.
3. **The product (1:00-1:45)** — Lighthouse. Type a ticker, tell it who you are (horizon, risk tolerance, the most you'll lose on one idea, portfolio size). Thirty seconds later: a **verdict**, the **why** (every signal is a beacon — green, amber, red — with a one-line reason), and a **position plan**: shares, exit level, worst-case dollars.
4. **The investment logic (1:45-2:30)** — "We buy quality businesses when the market agrees with us." Fundamentals decide *what* (valuation, quality, growth — 13 beacons vs sector norms). Technicals decide *when* (trend above the 200-day line, momentum confirming — 6 beacons). Market weather decides *whether now* (Bull/Bear × Calm/Turbulent). Your max-loss rule decides *how much*.
5. **Live demo (2:30-3:30)** — see below.
6. **The evidence (3:30-4:15)** — 20 S&P 500 stocks, 32 years, liquidity-scaled costs, no look-ahead. Lighthouse cut the maximum drawdown and volatility of **every single stock** (average max drawdown −27% vs −69%), kept a balanced investor's volatility at ~12% against a 15% target, and traded only ~12 times per stock in 32 years. It trails buy-and-hold on raw return — we say so on the slide — because its job is to deliver the risk you asked for, not to out-gamble the S&P. On every S&P 500 member of 2009, not just today's winners, it still trails (Sharpe 0.30 vs 0.48), and we tested 47 stock-picking signals: none survived an honest hold-out. Sensitivity test: the default thresholds sit in a broad, flat region; tuning buys little.
7. **Why us, not a finance site or a chatbot (4:15-4:45)** — explainable by construction, personal (same stock, different plan for different people), market-aware, honestly tested (including the results that went against us), runs anywhere (even offline), 330 automated tests.
8. **Ask / close (4:45-5:00)** — "Lighthouse doesn't promise to beat the market. It promises you'll never again buy a stock without knowing why, how much, and where you're wrong." Vote for the product you'd actually use.

## Live demo (60 seconds, pre-loaded)

1. Streamlit app, sidebar already filled: MSFT, long-term, balanced, max loss 2%, $25,000. Press **Analyse**.
2. Point at the verdict card: *Buy, 71/100 — fundamental 63, technical 83, weather Bull/Calm.*
3. Point at the position plan: *19 shares ≈ $1,594 (6.6%), exit $64.73, worst case $364. Size limited by the max-loss rule.* "The exit sits 23% below the price, so your 2% rule sizes it small — that is the product protecting you from yourself."
4. Scroll to the beacon chart: "Every point of the 71 is right here. PEG is red — you're paying for growth. ROE and EPS growth are green. Trend and 52-week high carry the technical score."
5. Switch risk to *aggressive* and horizon to *short*: verdict stays Buy at 70, size jumps to 25% (75 shares, held back by the concentration cap), exit tightens to an ATR stop at $81.56. "Same stock, different investor, different plan."
6. Flash the backtest chart: strategy sat out most of 2008 (drawdown −33% vs −59%).

Fallback if the network is down: the app's data source dropdown → `sample` (bundled real data) — the demo is identical.

## Q&A — prepared answers

* **Why these indicators / thresholds / weights?** → `docs/METHODOLOGY.md` sections 2-5. Short version: the most robust, widely studied signals; thresholds set for interpretability first and checked with a sensitivity grid; weights follow the horizon (long-term = buying the business, short-term = trading price).
* **How does it behave in different markets?** → regime table (MSFT, 2002-2017): −18% a year against buy-and-hold's −31% in Bear/Turbulent markets, ~64% exposure in Bull/Calm.
* **How sensitive are the results?** → sensitivity heat-map and walk-forward slide; biggest sensitivities are the review frequency and stop design, both documented with the experiments that rejected the alternatives.
* **Biggest weaknesses?** → whipsaw losses in long sideways markets; the scores do not pick stocks (none of 47 signal configurations survived the hold-out); sector norms are assumptions; SEC fundamentals start in 2009; free data has no prices for most delisted companies, so survivorship-aware results are bounds; no taxes.
* **What did GenAI do and what did you decide?** → GenAI wrote scaffolding and indicator code and ran our experiment harness; we chose the customer, the signals, the weights, the monthly review, the SMA-based exit, rejected ATR stops for long-term positions, and insisted on reporting underperformance versus buy-and-hold.
