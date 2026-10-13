"""
Compare the long-horizon exit plans with the historical setup test on the offline sample pack.

For every stock in the pack (20 S&P 500 names, data to 2022, or 2017 for MSFT) the setup test replays every past long
setup of the long-horizon, balanced investor, one trade at a time, after costs, under two plans:

  targets   the R-multiple plan: target 1 at 2R (half sold, stop to breakeven), a 5 x ATR trailing stop on the rest,
            target 2 at 3R, a 252-session time stop. R is the distance to the stop: 200-day SMA minus 8%.
  trend     the trend plan (the default since October 2026): the same stop order, no fixed target; at each month's
            last close, sell at the next open if the close is below the 200-day SMA minus 8% or the technical score is
            35 or lower. These are the long-horizon backtest's own exits.

Reported over the whole history (from 1990) and over the last 15 and 10 years (the app's window is 10 by default).

Run:  LIGHTHOUSE_OFFLINE=1 python scripts/exit_plan_evaluation.py
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("LIGHTHOUSE_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lighthouse import analyze, InvestorProfile  # noqa: E402
from lighthouse.data import sample_pack_tickers  # noqa: E402
from lighthouse.tradeplan import TradeSettings, setup_statistics  # noqa: E402

PLANS = {"targets": TradeSettings(long_exit="targets"), "trend": TradeSettings(long_exit="trend")}
WINDOWS = {"1990-2022": 40, "last 15 y": 15, "last 10 y": 10}


def evaluate(ticker: str) -> list:
    rows = []
    for window, years in WINDOWS.items():
        r = analyze(ticker, InvestorProfile(horizon="long", risk="balanced"), years=years, source="sample", run_backtest_=False)
        for plan, s in PLANS.items():
            try:
                tr = setup_statistics(r, "long", s).trades
            except ValueError:
                continue
            rows += [dict(window=window, plan=plan, ticker=ticker, r=t.r, ret=t.return_pct / 100.0, days=t.days,
                          first=t.outcome.split(" + ")[0]) for t in tr.itertuples()]
    return rows


def table(df: pd.DataFrame) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        per_stock = g.groupby("ticker")["r"].mean()
        return pd.Series({
            "trades": len(g), "avg R": g.r.mean(), "median R": g.r.median(), "win rate": (g.r > 0).mean(),
            "stopped": (g["first"] == "stop").mean(), "avg sessions": g.days.mean(),
            "return a year held": g.ret.sum() / g.days.sum() * 252, "stocks with avg R > 0": int((per_stock > 0).sum()),
        })
    return df.groupby(["window", "plan"], sort=False)[["ticker", "r", "ret", "days", "first"]].apply(one)


def main() -> int:
    tickers = [t for t in sample_pack_tickers() if not t.startswith("^")]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        df = pd.DataFrame([row for part in pool.map(evaluate, tickers) for row in part])
    out = table(df)
    print(f"Long-horizon setups, balanced profile, {len(tickers)} sample stocks, after costs\n")
    print(out.to_string(formatters={"trades": "{:.0f}".format, "avg R": "{:+.2f}".format, "median R": "{:+.2f}".format,
                                    "win rate": "{:.0%}".format, "stopped": "{:.0%}".format, "avg sessions": "{:.0f}".format,
                                    "return a year held": "{:.1%}".format, "stocks with avg R > 0": "{:.0f}".format}))
    for window in WINDOWS:
        w = df[df.window == window].groupby(["ticker", "plan"])["r"].mean().unstack()
        if {"trend", "targets"} <= set(w.columns):
            w = w.dropna()
            print(f"{window}: the trend plan has the higher average R on {int((w.trend > w.targets).sum())} of {len(w)} stocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
