"""
Goal portfolios before and after the rank buffer: walk-forward turnover, return and risk for every goal.

"Before" re-selects each goal's stocks from scratch at every monthly review; "after" keeps a stock already held while it
still passes the goal's filter (Goal.keep_rank) and re-fills only the freed places. Both are re-run month by month with
the data available then, trade at the next day's close and pay 10 bps per dollar traded. Turnover is buys + sells as a
share of the portfolio, per year.

Run:  LIGHTHOUSE_OFFLINE=1 python scripts/turnover_evaluation.py                                  # sample pack, 10 years
      python scripts/turnover_evaluation.py --universe "Large caps (all sectors)" --source auto    # live data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from lighthouse import portfolio as pf  # noqa: E402
from lighthouse.universe import UNIVERSES, available_universe  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--universe", default=None, help="a pre-built universe name (default: every ticker available)")
    ap.add_argument("--source", default="sample" if os.environ.get("LIGHTHOUSE_OFFLINE") else "auto")
    ap.add_argument("--years", type=float, default=10)
    a = ap.parse_args(argv)
    tickers = UNIVERSES[a.universe] if a.universe else available_universe(a.source)
    uni = pf.load_universe(tickers, years=a.years, source=a.source)
    print(f"{a.universe or 'All available tickers'}: {len(uni.tickers)} stocks, {uni.closes.index[0].date()} to "
          f"{uni.closes.index[-1].date()}, source {a.source}\n")
    rows = []
    for key, g in pf.GOALS.items():
        r = pf.goal_portfolio(key, capital=10_000, universe=uni)
        for label, m in r.comparison.iterrows():
            if label.startswith(g.label):
                rows.append({"goal": g.label, "selection": "kept while they qualify" if "kept" in label else "from scratch",
                             "turnover / yr": m.turnover_annual, "CAGR": m.cagr, "volatility": m.volatility,
                             "Sharpe": m.sharpe, "max drawdown": m.max_drawdown})
    df = pd.DataFrame(rows).set_index(["goal", "selection"])
    print(df.to_string(formatters={"turnover / yr": "{:.1f}x".format, "CAGR": "{:.1%}".format, "volatility": "{:.1%}".format,
                                   "Sharpe": "{:.2f}".format, "max drawdown": "{:.0%}".format}))
    print("\n(preservation re-selects from scratch: its ranking, 1-year volatility, moves slowly)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
