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
**Buy / Hold / Sell verdict, a position plan and a trade ticket**, from a price rule whose past behaviour it shows you
honestly, and puts an **analyst memo** read from the company's SEC filings next to it.

This notebook runs the full product end-to-end:

1. **Data collection & cleaning** — prices, volume and the S&P 500 benchmark
2. **Technical analysis** — six beacons (trend, RSI, MACD, price bands, volume, 52-week breakout)
3. **Market weather** — bull/bear × calm/turbulent regime detection
4. **Decision** — explainable score → verdict → position size, exit level and max loss
5. **Analyst memo** — the fundamental case from the SEC filings: DCF, reverse DCF, quality, balance sheet, what would prove it wrong
6. **Strategy testing** — point-in-time backtest vs buy-and-hold, the same exposure held without trading, and the S&P 500
7. **Watchlist ranking**
8. **Trade ticket** — what to buy right now with your cash: entry, stop-loss, the exits, and how those exact rules did historically
9. **Portfolio builder** — allocate a basket (equal weight, risk parity, minimum variance), measure its risk, test it out-of-sample

> Runs on Google Colab or any Jupyter. With internet access it downloads live data (the memo needs a contact e-mail for SEC
> EDGAR in the environment variable `LIGHTHOUSE_SEC_EMAIL`); without it, it falls back to the bundled historical sample pack
> (Microsoft 1986–2017 with volume, 19 other S&P stocks 1990–2022 with their SEC filings, the S&P 500 index), so it always
> runs end to end.
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
import lighthouse, pandas as pd, numpy as np
from lighthouse import interactive as ix      # every chart is an interactive Plotly figure: hover, zoom, PNG download
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
print("Lighthouse", lighthouse.__version__, "ready in", root)""")
md("""## 1 · Tell Lighthouse about yourself

Everything a non-programmer needs to change lives in this one cell.""")
code("""TICKER          = "MSFT"        # any US-listed ticker, e.g. "AAPL", "NVDA", "JPM"
HORIZON         = "long"        # "long" (6-24+ months, monthly review, trend exits) or "short" (weeks-months, weekly review)
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

Lighthouse pulls **daily prices and volume** and the **S&P 500** (for market weather and benchmarking). Every change made
during cleaning is reported, and the product always tells you which source the data came from. The company's SEC filings
are read later, for the analyst memo.""")
code("""from lighthouse.data import load_prices, load_benchmark, align
prices = load_prices(TICKER, years=YEARS, source=DATA_SOURCE)
bench  = load_benchmark("^GSPC", years=YEARS, source=DATA_SOURCE)
prices, bench = align(prices, bench)
print(prices.attrs["cleaning"].summary()); print(bench.attrs["cleaning"].summary())
prices.tail()""")
md("""## 3 · Technical analysis

Six beacons, all computed with **causal** indicators (no future data leaks into any value): trend (50/200-day SMA, Golden/Death
Cross), RSI(14), MACD(12/26/9), position in the low-lag price bands (%B around Holt's trend filter, about 1 session behind the
price), volume confirmation (OBV + 20/60-day volume ratio) and distance from the 52-week high. Each votes between −1 and +1;
the weights depend on the horizon: long-term investors lean on trend and breakouts, short-term on momentum oscillators.""")
code("""from lighthouse.technicals import compute_indicators, technical_score, latest_beacons
from lighthouse.report import beacon_frame_df
ind = compute_indicators(prices, params)
T_series = technical_score(ind, HORIZON, params)
t_beacons = latest_beacons(ind, HORIZON, params)
print("Technical score today:", round(float(T_series.iloc[-1]), 1))
beacon_frame_df(t_beacons)""")
md("""## 4 · Market weather

Is the broad market in an up-trend (S&P 500 above its 200-day average) and is volatility calm or turbulent (21-day realised
volatility vs. 1.25× its one-year median)? In a bear market Lighthouse deducts points, raises the entry bar and halves new
positions; in a Bear/Turbulent market it blocks new entries altogether.""")
code("""from lighthouse.regime import regime_table, latest_regime
regimes = regime_table(bench, params)
weather = latest_regime(regimes)
print(weather.describe())
regimes["regime"].value_counts(normalize=True).round(3)""")
md("""## 5 · The decision

**Lighthouse score = technical score − market-weather penalty**, mapped to a verdict
(≥72 Strong Buy · ≥60 Buy · ≥45 Hold · ≥32 Reduce · else Sell). The rule reads prices only: every fundamental and
alternative signal Lighthouse tested on the point-in-time S&P 500 failed its pre-registered hold-out (docs/METHODOLOGY.md §8),
so none carries weight. The position plan takes the *smallest* size allowed by three independent rules — your max-loss
limit, a volatility target and a concentration cap — then scales it by conviction.""")
code("""from lighthouse import analyze
result = analyze(TICKER, profile, params, years=YEARS, source=DATA_SOURCE)
d = result.decision
print(d.headline()); print(d.plan.action)
print(f"Position: {d.plan.position_pct:.1f}% (${d.plan.position_value:,.0f}, {d.plan.shares} shares) · exit level "
      f"{'n/a' if d.plan.stop_price is None else f'${d.plan.stop_price:,.2f}'} · worst case ${d.plan.max_loss_value:,.0f}")
for w in result.warnings: print("⚠", w)""")
code("""ix.score_breakdown(d).show()""")
code("""ix.dashboard(result).show()""")
md("""### The same stock, different investors
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
md("""## 6 · The analyst memo: the fundamental case

The memo reads the company's **SEC filings as they were filed** (nothing filed after the analysis date) and builds the
case a human analyst would write: a bull / base / bear DCF and the growth the price implies, the fundamental beacons
against sector norms (or industry peers by SIC code with `peers=True`, slower), earnings-quality and balance-sheet checks,
capital allocation, catalysts, the variant view and the measurable criteria that would prove the thesis wrong. It sits
next to the verdict and never changes it.""")
code("""from IPython.display import Markdown
from lighthouse.analyst import memo as analyst_memo
memo = analyst_memo.build(TICKER, result=result, source=DATA_SOURCE, peers=False, peer_norms=True)
print(f"stance {memo.stance.upper()} · weighted value "
      f"{'n/a' if memo.valuation is None else f'${memo.valuation.target:,.2f}'} vs ${memo.financials.price:,.2f}")
Markdown(memo.markdown())""")
md("""## 7 · Strategy testing and performance evaluation

The rule is tested the way an investor would have lived it: decided at the close on the review day, filled the **next** day,
costs on both sides, the regime read from the benchmark up to that day. Next to buy-and-hold and the S&P 500 the table shows
**the same average exposure held without trading**: the stock and cash in the proportion the rule held on average. It is the
honest yardstick for timing: what the rule earns above it is what the timing added.""")
code("""from lighthouse.report import format_metrics
bt = result.backtest
print(f"{bt.start.date()} → {bt.end.date()} · fills at the {bt.options['fills']} · reviewed {params.decision_frequency}")
format_metrics(bt.summary())""")
code("""ix.backtest(bt, result.regimes).show()""")
code("""bt.trades_frame().tail(10)""")
md("""### What each layer contributes, and how the rule behaves in each market regime""")
code("""extras = result.evaluate(layers=True, sensitivity=True, walk_forward_test=True)
display(format_metrics(extras["layers"]))
ix.layers(extras["layers"]).show()
format_metrics(extras["regime_breakdown"])""")
md("""### Sensitivity: is the rule robust or a lucky spike?
A trustworthy rule shows a *plateau* of similar results around its default thresholds.""")
code("""ix.sensitivity(extras["sensitivity"], "sharpe").show()
extras["sensitivity"].describe().round(2)""")
md("""### Out-of-sample test
Thresholds are tuned on the first 60% of history and then applied — untouched — to the last 40%. If tuning only fitted noise,
the tuned rule will not beat the default rule out of sample.""")
code("""wf = extras["walk_forward"]
print("split date:", wf["split_date"].date())
display(format_metrics(wf["out_of_sample"]))
ix.walk_forward(wf).show()""")
md("""## 8 · Rank a watchlist""")
code("""from lighthouse import screen
ranking = screen(WATCHLIST, profile, params, years=min(YEARS, 6), source=DATA_SOURCE, progress=False)
display(ranking[["ticker", "name", "verdict", "score", "technical", "regime", "position_pct", "price", "stop", "error"]])
ix.screener(ranking).show()""")
md("""## 9 · Trade ticket: buy it right now

The ticket sizes the trade from your **cash**, sets the stop-loss and the exits, and replays those exact rules on the
stock's own history. A long-term buy has no fixed profit target: it is sold after a month-end close below the 200-day SMA
− 8% (a level that rises with the trend) or when the technical score fades to 35, the backtest's own exits. Short-term
trades take profit at two levels in multiples of the risk (*R*) with a trailing and a time stop. Tickets need current
prices: on the bundled sample they are refused unless `allow_stale=True`.""")
code("""from lighthouse.tradeplan import build_trade_plan, TradeSettings, StaleDataError
settings = TradeSettings(reward_multiples=(2, 3), partial_exit=0.5)
try:
    ticket = build_trade_plan(result, "long", settings, allow_stale=DATA_SOURCE == "sample")
    print(f"--- {ticket.status} ---")
    print("\\n".join(f"{i}. {step}" for i, step in enumerate(ticket.management_steps(), 1)))
    if ticket.stats:
        st_ = ticket.stats
        print(f"history: {st_.n_trades} trades, win rate {st_.win_rate:.0%}, {st_.avg_r:+.2f}R per trade, profit factor {st_.profit_factor:.2f}")
except StaleDataError as exc:
    print("No ticket:", exc)""")
md("""## 10 · Portfolio builder: let Lighthouse work out the percentages

Give it a basket; it estimates the risk (Ledoit-Wolf shrunk covariance), allocates with the method you choose (equal weight,
risk parity or minimum variance), turns the weights into whole shares for your cash, reports the risk (volatility, VaR,
expected shortfall, beta, risk contributions, stress tests) and walk-forward tests **every** method on the same basket so
you can see which one actually held up.""")
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
code("""Markdown(result.report())""")
code("""# Save the report, the tables and the interactive charts (one HTML file) to a folder
written = result.save(f"reports/{TICKER}")
print("\\n".join(written))""")
md("""## Limitations and risks (read before acting)

* **The timing has no demonstrated edge.** A test pre-registered before it was run (docs/METHODOLOGY.md §8d) found, on
  every S&P 500 member of 1996 and of 2009, that holding the rule's own average exposure without trading earned more
  than the rule on most stocks, and that its drawdown advantage over that exposure did not hold in both cohorts. What
  the rule delivers is a position sized to your risk; judge it on that.
* **No other signal survived.** Fundamental, quality, short-interest and text signals were tested on the point-in-time
  S&P 500 and failed their hold-out, so the verdict reads prices only and the memo is context, not a signal.
* **The memo's inputs are assumptions**: sector norms, the DCF's growth fade and discount rate. It shows each one.
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
