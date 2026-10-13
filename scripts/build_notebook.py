"""Builds notebooks/Lighthouse.ipynb (kept as a script so the notebook is reproducible and diff-friendly)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))     # noqa: E731

md("""# 🔦 Lighthouse — explainable stock decisions for self-directed investors

*A clear signal in a noisy market.*

Lighthouse turns a ticker and a few facts about **you** (horizon, risk tolerance, the most you are willing to lose) into a
**Buy / Hold / Sell verdict, a position plan and the reasons behind it** — then shows you honestly how that rule would have
behaved in the past.

This notebook runs the full product end-to-end:

1. **Data collection & cleaning** — prices, volume, fundamentals, the S&P 500 benchmark
2. **Fundamental analysis** — valuation, quality and growth *beacons*
3. **Technical analysis** — six beacons (trend, RSI, MACD, price bands, volume, 52-week breakout)
4. **Market weather** — bull/bear × calm/turbulent regime detection
5. **Decision** — explainable score → verdict → position size, exit level and max loss
6. **Strategy testing** — point-in-time backtest vs. buy-and-hold and the S&P 500, layer attribution, sensitivity, out-of-sample test
7. **Watchlist ranking**
8. **Trade ticket** — what to buy (or short) right now with your cash: entry, stop-loss, take-profit levels, and how those exact rules did historically
9. **Portfolio builder** — allocate a basket (risk parity, minimum variance, HRP, ...), measure its risk, and test the method out-of-sample

> Runs on Google Colab or any Jupyter. With internet access it downloads live data; without it, it falls back to the bundled
> historical sample pack (Microsoft 1986–2017 with volume, 19 other S&P stocks 1990–2022, S&P 500 index) so it always runs end to end.
""")
md("## 0 · Setup")
code("""# On Google Colab this clones the project; inside the repo (or its notebooks/ folder) it just finds the package.
import os, sys, subprocess, pathlib
REPO_URL = "https://github.com/lotainc0/Stock-Trader-FINTECH-Project.git"
here = pathlib.Path.cwd()
root = next((p for p in [here, *here.parents] if (p / "lighthouse" / "__init__.py").exists()), None)
if root is None:
    subprocess.run(["git", "clone", "-q", REPO_URL, "lighthouse_repo"], check=True)
    root = here / "lighthouse_repo"
os.chdir(root); sys.path.insert(0, str(root))
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True, timeout=900)
except Exception as exc:  # offline or already installed - keep going
    print("pip install skipped:", exc)
import lighthouse, pandas as pd, numpy as np, matplotlib.pyplot as plt
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
print("Lighthouse", lighthouse.__version__, "ready in", root)""")
md("""## 1 · Tell Lighthouse about yourself

Everything a non-programmer needs to change lives in this one cell.""")
code("""TICKER          = "MSFT"        # any US-listed ticker, e.g. "AAPL", "NVDA", "JPM"
HORIZON         = "long"        # "long" (6-24+ months, fundamentals weigh 60%) or "short" (weeks-months, technicals weigh 70%)
RISK            = "balanced"    # "conservative" | "balanced" | "aggressive"
MAX_LOSS_PCT    = 2.0           # the most you accept losing on THIS position, as % of your whole account
CASH_AVAILABLE  = 10_000        # $ you can invest right now: new positions are paid from this cash only
HOLDINGS_VALUE  = 15_000        # $ value of stocks you already hold (0 if none); risk limits are a % of cash + holdings
YEARS           = 15            # years of history to analyse and backtest (15 covers at least one full bear market)
WATCHLIST       = ["MSFT", "AAPL", "JNJ", "XOM", "JPM", "UNH"]   # for the ranking section
DATA_SOURCE     = "auto"        # "auto" = live (Yahoo Finance, then Stooq) with offline fallback; "sample" = bundled pack

from lighthouse import InvestorProfile, StrategyParams
profile = InvestorProfile.from_cash(CASH_AVAILABLE, HOLDINGS_VALUE, horizon=HORIZON, risk=RISK, max_loss_pct=MAX_LOSS_PCT)
params  = StrategyParams.for_horizon(HORIZON)      # the trading rule preset for this horizon (editable: params.copy(entry_score=65))
print(profile.describe()); print(params.to_dict())""")
md("""## 2 · Data collection and cleaning

Lighthouse pulls **daily prices and volume**, the **S&P 500** (for market weather and benchmarking) and **fundamentals**
(valuation ratios, profitability, leverage, cash flow, growth, plus up to five years of annual statements for point-in-time testing).
Every change made during cleaning is reported, and the product always tells you which source the data came from.""")
code("""from lighthouse.data import load_prices, load_benchmark, load_fundamentals, align
prices = load_prices(TICKER, years=YEARS, source=DATA_SOURCE)
bench  = load_benchmark("^GSPC", years=YEARS, source=DATA_SOURCE)
prices, bench = align(prices, bench)
print(prices.attrs["cleaning"].summary()); print(bench.attrs["cleaning"].summary())
prices.tail()""")
code("""snapshot, statements = load_fundamentals(TICKER, source=DATA_SOURCE, last_price=float(prices["Close"].iloc[-1]))
print(f"{snapshot.name} · {snapshot.sector} · {snapshot.industry} · source: {snapshot.source} (as of {snapshot.as_of})")
pd.Series(snapshot.to_dict()).drop(["notes"]).to_frame("value")""")
code("""print(f"{statements.years} annual statements available for point-in-time testing")
statements.annual""")
md("""## 3 · Fundamental analysis

Each ratio becomes a **beacon** that votes between −1 (expensive / weak) and +1 (cheap / strong) against a sector norm or an
absolute band. Valuation weighs 40%, quality 35%, growth 25%. Missing data never silently biases the score — unavailable
beacons are dropped and the weights re-normalised.""")
code("""from lighthouse.fundamentals import fundamental_beacons, fundamental_score
from lighthouse.report import beacon_frame_df
f_beacons = fundamental_beacons(snapshot, statements, params)
F = fundamental_score(f_beacons)
print("Fundamental score:", None if F is None else round(F, 1))
beacon_frame_df(f_beacons)""")
md("""## 4 · Technical analysis

Six beacons, all computed with **causal** indicators (no future data leaks into any value): trend (50/200-day SMA, Golden/Death
Cross), RSI(14), MACD(12/26/9), position in the low-lag price bands (%B around Holt's trend filter, about 1 session behind the price), volume confirmation (OBV + 20/60-day volume ratio) and distance from the 52-week high.
The weights depend on the horizon: long-term investors lean on trend and breakouts; short-term on momentum oscillators.""")
code("""from lighthouse.technicals import compute_indicators, technical_score, latest_beacons
ind = compute_indicators(prices, params)
T_series = technical_score(ind, HORIZON, params)
t_beacons = latest_beacons(ind, HORIZON, params)
print("Technical score today:", round(float(T_series.iloc[-1]), 1))
beacon_frame_df(t_beacons)""")
md("""## 5 · Market weather

Is the broad market in an up-trend (S&P 500 above its 200-day average) and is volatility calm or turbulent (21-day realised
volatility vs. 1.25× its one-year median)? In a bear market Lighthouse deducts points, raises the entry bar and halves new
positions; in a Bear/Turbulent market it blocks new entries altogether.""")
code("""from lighthouse.regime import regime_table, latest_regime
regimes = regime_table(bench, params)
weather = latest_regime(regimes)
print(weather.describe())
regimes["regime"].value_counts(normalize=True).round(3)""")
md("""## 6 · The decision

**Composite = w_F × Fundamental + w_T × Technical − market-weather penalty**, mapped to a verdict
(≥72 Strong Buy · ≥60 Buy · ≥45 Hold · ≥32 Reduce · else Sell). The position plan takes the *smallest* size allowed by three
independent rules — your max-loss limit, a volatility target and a concentration cap — then scales it by conviction.""")
code("""from lighthouse import analyze
result = analyze(TICKER, profile, params, years=YEARS, source=DATA_SOURCE)
d = result.decision
print(d.headline()); print(d.plan.action)
print(f"Position: {d.plan.position_pct:.1f}% (${d.plan.position_value:,.0f}, {d.plan.shares} shares) · exit level "
      f"{'n/a' if d.plan.stop_price is None else f'${d.plan.stop_price:,.2f}'} · worst case ${d.plan.max_loss_value:,.0f}")
for w in result.warnings: print("⚠", w)""")
code("""from lighthouse import charts as ch
fig = ch.plot_score_breakdown(d); plt.show()""")
code("""fig = ch.plot_dashboard(result); plt.show()""")
md("""### The same stock, three different investors
The verdict and the position size depend on who is asking — that is the point of a personalised product.""")
code("""rows = []
for h in ["long", "short"]:
    for rk in ["conservative", "balanced", "aggressive"]:
        pr = InvestorProfile.from_cash(CASH_AVAILABLE, HOLDINGS_VALUE, horizon=h, risk=rk, max_loss_pct=MAX_LOSS_PCT)
        rr = analyze(TICKER, pr, years=YEARS, source=DATA_SOURCE, run_backtest_=False)
        rows.append({"horizon": h, "risk": rk, "verdict": rr.decision.verdict, "score": round(rr.decision.composite_score, 1),
                     "position %": round(rr.decision.plan.position_pct, 1), "exit level": rr.decision.plan.stop_price and round(rr.decision.plan.stop_price, 2),
                     "binding rule": rr.decision.plan.binding_constraint})
pd.DataFrame(rows)""")
md("""## 7 · Strategy testing and performance evaluation

The rule is tested the way an investor would have lived it: decided at the close on the review day, filled the **next** day,
costs on both sides, statements usable only after a 90-day reporting lag, the regime read from the benchmark up to that day.
We report strategy return, buy-and-hold, the S&P 500, volatility, Sharpe, max drawdown, number of trades and win rate.""")
code("""from lighthouse.report import format_metrics
bt = result.backtest
print(f"{bt.start.date()} → {bt.end.date()} · fills at the {bt.options['fills']} · reviewed {params.decision_frequency}")
format_metrics(bt.summary())""")
code("""fig = ch.plot_backtest(bt, result.regimes); plt.show()""")
code("""bt.trades_frame().tail(10)""")
md("""### What each layer contributes, and how the rule behaves in each market regime""")
code("""extras = result.evaluate(layers=True, sensitivity=True, walk_forward_test=True)
display(format_metrics(extras["layers"]))
fig = ch.plot_layers(extras["layers"]); plt.show()
format_metrics(extras["regime_breakdown"])""")
md("""### Sensitivity: is the rule robust or a lucky spike?
A trustworthy rule shows a *plateau* of similar results around its default thresholds.""")
code("""fig = ch.plot_sensitivity(extras["sensitivity"], "sharpe"); plt.show()
extras["sensitivity"].describe().round(2)""")
md("""### Out-of-sample test
Thresholds are tuned on the first 60% of history and then applied — untouched — to the last 40%. If tuning only fitted noise,
the tuned rule will not beat the default rule out of sample.""")
code("""wf = extras["walk_forward"]
print("split date:", wf["split_date"].date())
display(format_metrics(wf["out_of_sample"]))
fig = ch.plot_walk_forward(wf); plt.show()""")
md("""## 8 · Rank a watchlist""")
code("""from lighthouse import screen
ranking = screen(WATCHLIST, profile, params, years=min(YEARS, 6), source=DATA_SOURCE, progress=False)
display(ranking[["ticker", "name", "verdict", "score", "fundamental", "technical", "regime", "position_pct", "price", "stop", "error"]])
fig = ch.plot_screener(ranking); plt.show()""")
md("""## 9 · Trade ticket: buy or short it right now

The ticket sizes the trade from your **cash**, sets the stop-loss, two take-profit levels (in multiples of the risk, *R*),
a trailing stop and a time stop, and replays those exact rules on the stock's own history. Shorts mirror the long rules and
always use the tactical short-horizon stop.""")
code("""from lighthouse.tradeplan import build_trade_plan, TradeSettings
settings = TradeSettings(reward_multiples=(2, 3), partial_exit=0.5)
for side in ("long", "short"):
    ticket = build_trade_plan(result, side, settings)
    print(f"--- {side.upper()}: {ticket.status} ---")
    print("\\n".join(f"{i}. {step}" for i, step in enumerate(ticket.management_steps(), 1)))
    if ticket.stats:
        st_ = ticket.stats
        print(f"history: {st_.n_trades} trades, win rate {st_.win_rate:.0%}, {st_.avg_r:+.2f}R per trade, profit factor {st_.profit_factor:.2f}")
    print()""")
md("""## 10 · Portfolio builder: let Lighthouse work out the percentages

Give it a basket; it estimates the risk (Ledoit-Wolf shrunk covariance), allocates with the method you choose, turns the
weights into whole shares for your cash, reports the risk (volatility, VaR, expected shortfall, beta, risk contributions,
stress tests) and walk-forward tests **every** allocation method on the same basket so you can see which one actually held up.""")
code("""from lighthouse import portfolio as pf
basket = ["AAPL", "JNJ", "XOM", "JPM", "KO", "PG", "UNH", "WMT"]
port = pf.build_portfolio(basket, "risk_parity", pf.PortfolioConstraints(max_weight=0.25, target_vol=0.12),
                          capital=CASH_AVAILABLE, years=10, source=DATA_SOURCE)
print("\\n".join(port.notes))
display(port.table().round(3))
print({k: round(v, 4) for k, v in port.risk.items() if isinstance(v, float)})
format_metrics(port.comparison)""")
code("""# Or let Lighthouse pick the stocks for a goal: "preservation", "balanced", "growth" or "diversified"
goal = pf.goal_portfolio("balanced", capital=CASH_AVAILABLE, years=10, source=DATA_SOURCE)
display(goal.selection.round(3)); display(goal.table().round(3))
format_metrics(goal.comparison)""")
md("""## 11 · The full written report
Everything above, in plain English, ready to save or share.""")
code("""from IPython.display import Markdown
Markdown(result.report())""")
code("""# Save the report, tables and charts to a folder
written = result.save(f"reports/{TICKER}")
print("\\n".join(written))""")
md("""## Limitations and risks (read before acting)

* **Single-stock timing is hard.** On relentlessly compounding mega-caps, any rule that is sometimes in cash trails buy-and-hold on
  raw return. Lighthouse's value is *risk-managed* exposure: lower drawdowns, smaller positions in turbulent markets, and an
  explicit worst-case loss — judge it on Sharpe, drawdown and the regime table, not on total return alone.
* **Sector norms are assumptions**, not live peer data. Edit them in `lighthouse/fundamentals.py` if you disagree.
* **Fundamental history is short** (Yahoo provides ~4-5 annual statements), so the point-in-time gate only acts in recent years.
* **Costs are simple** (5 bps commission + 5 bps slippage per side), taxes are ignored, the strategy backtest checks stops at the close.
* **Not investment advice.** Lighthouse is a decision-support tool: it shows you the evidence and its reasoning; the decision is yours.
""")
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"      # stable ids: rebuilding an unchanged notebook gives an identical file
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "colab": {"name": "Lighthouse.ipynb", "toc_visible": True}}
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/Lighthouse.ipynb")
print("wrote notebooks/Lighthouse.ipynb with", len(cells), "cells")
