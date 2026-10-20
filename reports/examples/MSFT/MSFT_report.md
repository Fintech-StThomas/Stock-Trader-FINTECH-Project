# Lighthouse report: MSFT
*Signals as of 2017-11-10 · price $83.87 (close) · investor: long-term, balanced risk, $25,000 cash to invest, max loss 2.0% of the account per position*

## Verdict: **Strong Buy** — Lighthouse score 72 / 100
Open or add to a position - the evidence lines up across fundamentals, technicals and the market backdrop.

### How the score is built
- Fundamental score **65** × weight 60% = 38.8
- Technical score **83** × weight 40% = 33.3
- Market-weather adjustment: **0** (Bull / Calm)
- **Lighthouse score = 72.2** → Strong Buy (bands: ≥72 Strong Buy, ≥60 Buy, ≥45 Hold, ≥32 Reduce, else Sell)
- The analyst memo (`python -m lighthouse memo` or the app's Analyse tab) goes further into the business: the DCF, peers, red flags and what would prove the thesis wrong.

### Fundamental beacons (score 65)
*Technology · data: SEC EDGAR XBRL, point in time (2017-11-10)*

| signal | reading | light | vote | weight | points | why |
|---|---|---|---|---|---|---|
| Price / Earnings (trailing) | P/E 28.0 vs Technology ~28 | 🟡 | -0.00 | 13% | -0.0 | Trailing P/E of 28.0 is above the typical Technology multiple (~28): you pay more per dollar of earnings than for a typical peer. |
| Price / Book | P/B 7.22 vs ~7.0 | 🟡 | -0.05 | 13% | -0.3 | Price-to-book of 7.22 is above the Technology norm; for asset-light companies this is a secondary check. |
| EV / EBITDA | EV/EBITDA 23.6 vs ~20 | 🔴 | -0.36 | 13% | -2.4 | Enterprise value per dollar of operating cash earnings, which is capital-structure neutral, is above the sector norm. |
| Return on equity | ROE 25.8% | 🟢 | +1.00 | 7% | +3.5 | Return on equity above 20% signals a business that compounds shareholder capital efficiently. |
| Operating margin | op. margin 26.4% vs ~22.0% | 🟢 | +0.29 | 7% | +1.0 | Operating margin is above the Technology norm, which suggests pricing power or cost advantages. |
| Debt / Equity | D/E 0.86 | 🟡 | +0.14 | 7% | +0.5 | Debt below equity: a balance sheet that can absorb a downturn. |
| Free-cash-flow yield | FCF yield 5.0% | 🟢 | +0.66 | 7% | +2.3 | Free cash flow relative to market value: above 5% is a genuine cash return; below 1% means the price relies on future growth rather than cash generated today. |
| Current ratio | current ratio 3.12 | 🟢 | +0.50 | 7% | +1.7 | Short-term assets cover short-term liabilities. |
| Revenue growth (YoY) | revenue 10.1% | 🟢 | +0.34 | 12% | +2.1 | Double-digit revenue growth: demand for the product is expanding. |
| EPS growth (YoY) | EPS 39.4% | 🟢 | +1.00 | 12% | +6.2 | Earnings per share are growing faster than the market's long-run ~8%. |

### Technical beacons (score 83)
| signal | reading | light | vote | weight | points | why |
|---|---|---|---|---|---|---|
| Trend (50/200-day SMA) | price +19.2% vs 200-SMA; Golden Cross (50 > 200) | 🟢 | +1.00 | 35% | +17.5 | Price is above its 200-day average and the 50-day sits above the 200-day: the long-term trend is up. |
| Momentum (RSI 14) | RSI 74.9 | 🟢 | +0.67 | 15% | +5.1 | RSI above 70 is overbought: momentum is strong but stretched, so the signal fades. |
| MACD (12/26/9) | histogram +0.06 (+0.05 ATR) | 🟡 | +0.05 | 10% | +0.3 | MACD line is above its signal line: short-term momentum is accelerating upward. |
| Price bands (trend filter, 2σ) | %B 0.24 | 🔴 | -0.52 | 10% | -2.6 | Price sits in the lower half of its bands: sellers are in control. |
| Volume confirmation (OBV, 20/60-day) | volume 1.20x avg; OBV up | 🟢 | +0.70 | 10% | +3.5 | Recent volume is 1.20x its 3-month average and on-balance volume is rising: the move is confirmed by participation. |
| 52-week high proximity | -0.8% from 52-week high | 🟢 | +0.96 | 20% | +9.6 | Trading at or near its 52-week high: a breakout zone where trend-followers add exposure. |

### Market weather
Market weather is Bull / Calm: the S&P 500 is above its 200-day average (up-trend) and realised volatility (5%) is near or below its one-year norm (7%).

### Position plan
- Buy **20 shares** near $83.87 = **$1,677** of your $25,000 cash (6.8% of the $25,000 account); $23,323 stays in cash.
- Exit level **$64.73** (22.8% below): worst case ≈ $383 (1.53% of the account, limit 2%) if the stop fills at its price; an overnight gap through the stop can lose more.
- Size was limited by the **max-loss rule**.
  - Exit level 22.8% below the current price at $64.73 (200-day SMA minus 8% buffer).
  - Max-loss rule: losing 2% of the account ($500) at the stop caps the position at 8.8% of the account.
  - Volatility rule: 16% realised vol vs a 15% target caps the position at 93.9% of the account.
  - Scaled by conviction 0.78 (score 72) and market-regime multiplier 1.

### How this rule behaved in the past (2007-08-30 → 2017-11-10)
Long-only, reviewed monthly at the close, filled at the next open; costs: commission + Corwin-Schultz half spread (floor 5 bps) + square-root impact; stops: resting orders (gaps fill at the open); cash earns the risk-free rate. Returns are for the capital allocated to this stock. Sharpe and Sortino use a fixed 2% a year (no T-bill series: offline or sample data).

|  | Total return | CAGR | Volatility | Sharpe | Sortino | Max drawdown | Calmar | Trades | Win rate | Avg win | Avg loss | Profit Factor | Time invested | Years |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lighthouse strategy | 22.6% | 2.0% | 11.8% | 0.06 | 0.08 | -32.7% | 0.06 | 8 | 37.5% | 33.3% | -14.2% | 1.93 | 45.1% | 10.2 |
| Same exposure, never traded | 117.5% | 7.9% | 12.6% | 0.51 | 0.76 | -29.7% | 0.27 | n/a | n/a | n/a | n/a | n/a | n/a | 10.2 |
| Buy & hold | 251.5% | 13.1% | 27.9% | 0.51 | 0.76 | -59.1% | 0.22 | n/a | n/a | n/a | n/a | n/a | n/a | 10.2 |
| S&P 500 | 77.2% | 5.8% | 20.5% | 0.28 | 0.39 | -56.8% | 0.10 | n/a | n/a | n/a | n/a | n/a | n/a | 10.2 |

*Same exposure, never traded*: the stock and cash in a fixed mix at the rule's average exposure (45%), the honest yardstick for a rule that is often in cash. Here the rule's Sharpe is 0.06 against 0.51 and its worst drawdown -33% against -30%.

**What each layer contributes**

|  | CAGR | Volatility | Sharpe | Max drawdown | Trades | Win rate | Time invested |
|---|---|---|---|---|---|---|---|
| Technical signals only (fully invested) | 5.9% | 19.6% | 0.29 | -35.9% | 6 | 50.0% | 73.9% |
| + market regime filter | 5.9% | 19.6% | 0.29 | -35.9% | 6 | 50.0% | 73.9% |
| + fundamental gate (point-in-time) | 5.9% | 19.6% | 0.29 | -35.9% | 6 | 50.0% | 73.9% |
| + volatility-targeted sizing (full Lighthouse) | 2.0% | 11.8% | 0.06 | -32.7% | 8 | 37.5% | 45.1% |
| Same exposure, never traded | 7.9% | 12.6% | 0.51 | -29.7% | n/a | n/a | 45.1% |
| Plain 200-day trend rule | 7.2% | 18.9% | 0.36 | -40.1% | 11 | 36.4% | 66.6% |
| Buy & hold | 13.1% | 27.9% | 0.51 | -59.1% | n/a | n/a | n/a |
| S&P 500 | 5.8% | 20.5% | 0.28 | -56.8% | n/a | n/a | n/a |

**Behaviour by market regime** (annualised daily returns)

| regime | Days | Share of time | Strategy ann. return | Buy & hold ann. return | Avg exposure |
|---|---|---|---|---|---|
| Bull / Calm | 1647 | 64.1% | 9.3% | 30.8% | 59.0% |
| Bear / Turbulent | 469 | 18.2% | -17.8% | -44.6% | 10.8% |
| Bull / Turbulent | 235 | 9.1% | 13.8% | 66.2% | 50.2% |
| Bear / Calm | 219 | 8.5% | -15.4% | -17.0% | 8.5% |

**Out-of-sample check** (thresholds tuned before 2013-10-15, tested after it)

|  | Total return | CAGR | Volatility | Sharpe | Max drawdown | Trades |
|---|---|---|---|---|---|---|
| Default rule (60/35) | 55.0% | 11.4% | 13.8% | 0.70 | -29.5% | 3 |
| Tuned in-sample (70/45) | 61.2% | 12.4% | 13.7% | 0.78 | -26.7% | 2 |
| Buy & hold | 170.2% | 27.7% | 21.9% | 1.13 | -18.0% | n/a |

**Sensitivity**: across 25 entry/exit threshold pairs the Sharpe ratio ranges from 0.06 (entry 55/exit 25) to 0.29 (entry 70/exit 45); median 0.09.

### Data notes and warnings
- OFFLINE DATA (sample): series ends 2017-11-10; connect to the internet for live prices
- Prices: sample; 2,821 trading days 2006-08-30 → 2017-11-10.

### Limitations you should know
- Sector norms are long-run approximations, not live peer data; a cheap stock in an expensive sector can still be expensive. The backtest's fundamental gate uses SEC filings as first reported, from the day after each was filed.
- On the point-in-time S&P 500 neither the technical nor the fundamental scores ranked stocks usefully (docs/METHODOLOGY.md section 8b): it is a risk-control rule, not a stock picker.
- The backtest is long-only; costs are estimated from the stock's own spread and volume, stops rest as orders (an overnight gap fills at the open), and it ignores taxes. Run `python -m lighthouse memo` for the analyst view.
- Past behaviour of a rule is evidence about its *design*, not a promise about its future returns.
- Lighthouse is a decision-support tool, not investment advice.