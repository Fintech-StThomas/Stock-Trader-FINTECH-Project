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

With `--cohort DATE` the same comparison runs on every S&P 500 member on DATE that has prices (live data, from DATE
to today, survivorship-aware), and for three kinds of trade:

  long horizon, longs   targets vs the trend plan (as above)
  short horizon, longs  targets (2R / 3R, 2.5 x ATR trail, 40-session time stop) vs a weekly trend exit: the ATR stop
                        order, and at each week's last close a sale at the next open if the close is below the 200-day
                        SMA minus 3% or the technical score is 40 or lower (the short-horizon backtest's own exits)
  shorts                targets vs the mirror of that weekly exit (cover if the close is above the 200-day SMA plus 3%
                        or the technical score is 60 or higher); borrow fee 0.5% a year

Run:  LIGHTHOUSE_OFFLINE=1 python scripts/exit_plan_evaluation.py
      python scripts/exit_plan_evaluation.py --cohort 2009-01-02        # writes reports/research/exit_plans_<DATE>.csv
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if "--cohort" not in sys.argv:
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


KINDS = {"long horizon, longs": ("long", "long"), "short horizon, longs": ("short", "long"), "shorts": ("short", "short")}
_COHORT = None


def _init(c):
    global _COHORT
    _COHORT = c


def _cohort_one(ticker: str) -> list:
    from lighthouse.research.cohort import replay_inputs
    frames, bench, start = _COHORT
    rows = []
    for kind, (horizon, side) in KINDS.items():
        try:
            res = replay_inputs(ticker, frames[ticker], bench, InvestorProfile(horizon=horizon, risk="balanced"))
        except Exception:  # noqa: BLE001 - a series that does not overlap the benchmark
            continue
        for plan in ("targets", "trend"):
            try:
                tr = setup_statistics(res, side, TradeSettings(long_exit=plan), exit_style=plan).trades
            except ValueError:
                continue
            tr = tr[tr["signal_date"] >= pd.Timestamp(start)]
            rows += [dict(window=kind, plan=plan, ticker=ticker, r=t.r, ret=t.return_pct / 100.0, days=t.days,
                          first=t.outcome.split(" + ")[0]) for t in tr.itertuples()]
    return rows


def cohort(date: str, workers: int) -> int:
    import multiprocessing as mp
    from lighthouse import membership
    from lighthouse.data import load_benchmark
    from lighthouse.research import panel
    start = pd.Timestamp(date)
    uni = membership.sp500()
    members = uni.members_on(start)
    frames, missing = panel.load_frames(members, start=start - pd.Timedelta(days=int(365.25 * 1.3)))
    frames = {t: df for t, df in frames.items() if df.index[-1] >= start + pd.Timedelta(days=30) and len(df) > 300}
    bench = load_benchmark("^GSPC", years=None, start=start - pd.Timedelta(days=int(365.25 * 1.3)))
    bench.attrs = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=_init,
                             initargs=((frames, bench, start),)) as pool:
        df = pd.DataFrame([row for part in pool.map(_cohort_one, sorted(frames)) for row in part])
    out = table(df)
    print(f"S&P 500 members on {start.date()}: {len(members)}; {len(frames)} with prices ({len(frames) / len(members):.0%} "
          f"coverage, all still trading today, so both delisting rules give the same trades); setups from {start.date()}, "
          f"balanced profile, after costs\n")
    print(out.to_string(formatters={"trades": "{:.0f}".format, "avg R": "{:+.2f}".format, "median R": "{:+.2f}".format,
                                    "win rate": "{:.0%}".format, "stopped": "{:.0%}".format, "avg sessions": "{:.0f}".format,
                                    "return a year held": "{:.1%}".format, "stocks with avg R > 0": "{:.0f}".format}))
    for kind in KINDS:
        w = df[df.window == kind].groupby(["ticker", "plan"])["r"].mean().unstack()
        if {"trend", "targets"} <= set(w.columns):
            w = w.dropna()
            print(f"{kind}: the trend exit has the higher average R on {int((w.trend > w.targets).sum())} of {len(w)} stocks")
    dest = ROOT / "reports" / "research" / f"exit_plans_{start.date()}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.assign(coverage=len(frames) / len(members), members=len(members)).to_csv(dest)
    print(f"\nwrote {dest.relative_to(ROOT)}")
    return 0


def main() -> int:
    if "--cohort" in sys.argv:
        k = sys.argv.index("--cohort")
        return cohort(sys.argv[k + 1], min(8, os.cpu_count() or 1))
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
