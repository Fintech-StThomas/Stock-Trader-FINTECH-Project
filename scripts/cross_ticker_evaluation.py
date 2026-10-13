"""
Regenerate reports/examples/cross_ticker_evaluation.csv and the README's "Results in one table".

Every stock in the offline sample pack (20 S&P 500 names, data to 2022, or 2017 for MSFT) is backtested over its whole
history with the long-horizon rule:

  signals_*       technical signals + market-regime filter, fully invested when in (no stop, no sizing)
  lighthouse_*    the full product, balanced profile (regime, point-in-time fundamental gate, exit, vol-targeted size)
  conservative_*  the full product, conservative profile
  bh_*            buy and hold

Costs: 5 bps commission + 5 bps slippage per side; fills at the next session's open (close for close-only files).

Run:  LIGHTHOUSE_OFFLINE=1 python scripts/cross_ticker_evaluation.py            # writes the CSV, prints the summary
      LIGHTHOUSE_OFFLINE=1 python scripts/cross_ticker_evaluation.py --check    # fails if the CSV is out of date
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("LIGHTHOUSE_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lighthouse import analyze, InvestorProfile, StrategyParams  # noqa: E402
from lighthouse.backtest import compute_metrics, run_backtest  # noqa: E402
from lighthouse.data import sample_pack_tickers  # noqa: E402

OUT = ROOT / "reports" / "examples" / "cross_ticker_evaluation.csv"


def evaluate(ticker: str) -> dict:
    p = StrategyParams()
    r = analyze(ticker, InvestorProfile(horizon="long", risk="balanced"), p, years=40, source="sample")
    bt = r.backtest
    sig = run_backtest(r.prices, r.benchmark, p, r.profile, r.pit_scores, use_regime=True, use_fundamental_gate=False,
                       use_stops=False, sizing="full").metrics
    cons = analyze(ticker, InvestorProfile(horizon="long", risk="conservative"), p, years=40, source="sample").backtest.metrics
    m, bh = bt.metrics, compute_metrics(bt.buy_hold, p.risk_free_rate, p.trading_days)
    return {"ticker": ticker, "start": str(bt.start.date()), "end": str(bt.end.date()),
            "signals_cagr": sig["cagr"], "signals_sharpe": sig["sharpe"], "signals_mdd": sig["max_drawdown"],
            "lighthouse_cagr": m["cagr"], "lighthouse_vol": m["volatility"], "lighthouse_sharpe": m["sharpe"],
            "lighthouse_mdd": m["max_drawdown"], "trades": int(m["n_trades"]), "win_rate": m["win_rate"],
            "conservative_cagr": cons["cagr"], "conservative_vol": cons["volatility"], "conservative_sharpe": cons["sharpe"],
            "conservative_mdd": cons["max_drawdown"],
            "bh_cagr": bh["cagr"], "bh_vol": bh["volatility"], "bh_sharpe": bh["sharpe"], "bh_mdd": bh["max_drawdown"]}


# row order of the published table
ORDER = ["MSFT", "AAPL", "JNJ", "XOM", "GE", "KO", "JPM", "HD", "PFE", "WMT", "UNH", "PG", "CVX", "LLY", "BAC", "AMD", "MRK",
         "PEP", "BBY", "RRC"]


def build() -> pd.DataFrame:
    tickers = [t for t in sample_pack_tickers() if not t.startswith("^")]
    order = [t for t in ORDER if t in tickers] + sorted(set(tickers) - set(ORDER))
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        rows = list(pool.map(evaluate, order))
    return pd.DataFrame(rows).set_index("ticker")


def summary(df: pd.DataFrame) -> str:
    pct = lambda v: f"{v * 100:.1f}%"   # noqa: E731
    lines = ["| | CAGR | Volatility | Sharpe | Max drawdown |", "|---|---|---|---|---|",
             f"| Buy & hold | {pct(df.bh_cagr.mean())} | {pct(df.bh_vol.mean())} | {df.bh_sharpe.mean():.2f} | "
             f"−{abs(df.bh_mdd.mean()) * 100:.0f}% |",
             f"| Lighthouse signals + regime (fully invested when in) | {pct(df.signals_cagr.mean())} | — | "
             f"{df.signals_sharpe.mean():.2f} | −{abs(df.signals_mdd.mean()) * 100:.0f}% |",
             f"| **Lighthouse balanced (vol-targeted)** | {pct(df.lighthouse_cagr.mean())} | {pct(df.lighthouse_vol.mean())} | "
             f"{df.lighthouse_sharpe.mean():.2f} | **−{abs(df.lighthouse_mdd.mean()) * 100:.0f}%** |",
             f"| Lighthouse conservative | {pct(df.conservative_cagr.mean())} | {pct(df.conservative_vol.mean())} | "
             f"{df.conservative_sharpe.mean():.2f} | −{abs(df.conservative_mdd.mean()) * 100:.0f}% |"]
    n = len(df)
    lower_risk = int(((df.lighthouse_mdd > df.bh_mdd) & (df.lighthouse_vol < df.bh_vol)).sum())
    better_sharpe = int((df.lighthouse_sharpe > df.bh_sharpe).sum())
    lines.append(f"\nLower drawdown and volatility in {lower_risk} of {n} stocks; higher Sharpe than buy-and-hold in "
                 f"{better_sharpe} of {n}.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed CSV instead of writing it")
    a = ap.parse_args(argv)
    df = build()
    if a.check:
        old = pd.read_csv(OUT, index_col="ticker")
        num = old.select_dtypes("number").columns
        same = (sorted(old.index) == sorted(df.index) and list(old.columns) == list(df.columns)
                and np.allclose(old[num].to_numpy(float), df.loc[old.index, num].to_numpy(float), atol=1e-10))
        print("cross_ticker_evaluation.csv is " + ("up to date" if same else "OUT OF DATE: re-run without --check"))
        return 0 if same else 1
    df.to_csv(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    print(summary(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
