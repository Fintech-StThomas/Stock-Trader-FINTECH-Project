"""
Regenerate reports/examples/cross_ticker_evaluation.csv and the README's "Results in one table".

Every stock in the offline sample pack (20 S&P 500 names, data to 2022, or 2017 for MSFT) is backtested over its whole
history with the long-horizon rule:

  signals_*       technical signals + market-regime filter, fully invested when in (no stop, no sizing)
  lighthouse_*    the full product, balanced profile (regime, exits, vol-targeted size)
  conservative_*  the full product, conservative profile
  same_*          the balanced product's average exposure held without timing (stock and cash in a fixed mix)
  bh_*            buy and hold

Costs: 5 bps commission + 5 bps slippage per side; fills at the next session's open (close for close-only files).

The sample pack holds today's survivors, which flatters every rule. `--cohort DATE` runs the same test on every S&P 500
member on DATE (historical membership, lighthouse.membership), followed until today or until its prices stop, with live
data (Yahoo prices, the FRED T-bill), under both delisting rules, and reports the share of members that had prices
(coverage):

  reports/examples/cross_ticker_sp500_<DATE>.csv           one row per member and delisting rule
  reports/examples/cross_ticker_sp500_<DATE>_summary.csv   averages per rule, coverage, how many beat buy-and-hold

Run:  LIGHTHOUSE_OFFLINE=1 python scripts/cross_ticker_evaluation.py            # writes the CSV, prints the summary
      LIGHTHOUSE_OFFLINE=1 python scripts/cross_ticker_evaluation.py --check    # fails if the CSV is out of date
      python scripts/cross_ticker_evaluation.py --cohort 1996-01-02
"""
from __future__ import annotations

import argparse
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

from lighthouse import analyze, InvestorProfile, StrategyParams  # noqa: E402
from lighthouse.backtest import compute_metrics, run_backtest  # noqa: E402
from lighthouse.data import sample_pack_tickers  # noqa: E402

OUT = ROOT / "reports" / "examples" / "cross_ticker_evaluation.csv"


def evaluate(ticker: str) -> dict:
    p = StrategyParams()
    r = analyze(ticker, InvestorProfile(horizon="long", risk="balanced"), p, years=40, source="sample")
    bt = r.backtest
    sig = run_backtest(r.prices, r.benchmark, p, r.profile, use_regime=True, use_stops=False, sizing="full").metrics
    cons = analyze(ticker, InvestorProfile(horizon="long", risk="conservative"), p, years=40, source="sample").backtest.metrics
    m, bh = bt.metrics, compute_metrics(bt.buy_hold, bt.rf, p.trading_days)
    same = compute_metrics(bt.same_exposure, bt.rf, p.trading_days)
    return {"ticker": ticker, "start": str(bt.start.date()), "end": str(bt.end.date()),
            "signals_cagr": sig["cagr"], "signals_sharpe": sig["sharpe"], "signals_mdd": sig["max_drawdown"],
            "lighthouse_cagr": m["cagr"], "lighthouse_vol": m["volatility"], "lighthouse_sharpe": m["sharpe"],
            "lighthouse_mdd": m["max_drawdown"], "trades": int(m["n_trades"]), "win_rate": m["win_rate"],
            "conservative_cagr": cons["cagr"], "conservative_vol": cons["volatility"], "conservative_sharpe": cons["sharpe"],
            "conservative_mdd": cons["max_drawdown"],
            "same_cagr": same["cagr"], "same_vol": same["volatility"], "same_sharpe": same["sharpe"],
            "same_mdd": same["max_drawdown"], "exposure": m["exposure"],
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
             f"{df.conservative_sharpe.mean():.2f} | −{abs(df.conservative_mdd.mean()) * 100:.0f}% |",
             f"| Same exposure as balanced, never traded ({df.exposure.mean():.0%} in the stock) | "
             f"{pct(df.same_cagr.mean())} | {pct(df.same_vol.mean())} | {df.same_sharpe.mean():.2f} | "
             f"−{abs(df.same_mdd.mean()) * 100:.0f}% |"]
    n = len(df)
    lower_risk = int(((df.lighthouse_mdd > df.bh_mdd) & (df.lighthouse_vol < df.bh_vol)).sum())
    better_sharpe = int((df.lighthouse_sharpe > df.bh_sharpe).sum())
    beat_same = int((df.lighthouse_sharpe > df.same_sharpe).sum())
    lines.append(f"\nLower drawdown and volatility in {lower_risk} of {n} stocks; higher Sharpe than buy-and-hold in "
                 f"{better_sharpe} of {n}, than the same exposure without timing in {beat_same} of {n}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- cohort (survivorship-aware) mode
_INPUTS = None


def _init(inputs) -> None:
    global _INPUTS
    _INPUTS = inputs


def _eval_member(job):
    from lighthouse.research.cohort import evaluate_member
    ticker, start, rule = job
    return evaluate_member(ticker, _INPUTS, start, rule)


def cohort(date: str, workers: int) -> int:
    import multiprocessing as mp
    from lighthouse import membership
    from lighthouse.data import load_benchmark, load_risk_free
    from lighthouse.research import panel
    from lighthouse.research.cohort import CohortInputs, DELISTING_RULES

    start = pd.Timestamp(date)
    uni = membership.sp500()
    members = uni.members_on(start)
    print(f"{uni.summary()}\n{len(members)} members on {start.date()} (sources: {'; '.join(uni.sources)})")
    frames, missing = panel.load_frames(members, start=start - pd.Timedelta(days=int(365.25 * 1.3)))
    frames = {t: df for t, df in frames.items() if df.index[-1] >= start + pd.Timedelta(days=30)}
    bench = load_benchmark("^GSPC", years=None, start=start - pd.Timedelta(days=int(365.25 * 1.3)))
    bench.attrs = {}
    rf = load_risk_free()
    data_end = bench.index[-1]
    inputs = CohortInputs(frames=frames, benchmark=bench, risk_free=rf, data_end=data_end)
    jobs = [(t, start, rule) for t in sorted(frames) for rule in DELISTING_RULES]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init, initargs=(inputs,)) as pool:
        rows = [r for r in pool.map(_eval_member, jobs, chunksize=4) if r is not None]
    return report_cohort(pd.DataFrame(rows), members, missing, uni, start, data_end)


def report_cohort(df, members, missing, uni, start, data_end) -> int:
    from lighthouse.research.cohort import summarize
    tested = df["ticker"].nunique()
    summ = summarize(df, len(members))
    summ["start"], summ["end"] = str(start.date()), str(data_end.date())
    summ["sources"] = "; ".join(uni.sources)
    out = ROOT / "reports" / "examples" / f"cross_ticker_sp500_{start.date()}.csv"
    df.to_csv(out, index=False)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"))
    print(f"\n{tested} of {len(members)} members tested ({tested / len(members):.0%} coverage); "
          f"{len(missing)} without prices")
    pct = lambda v: f"{v * 100:.1f}%"   # noqa: E731
    for rule, r in summ.iterrows():
        print(f"\n[{rule}] buy & hold CAGR {pct(r.bh_cagr)}, Sharpe {r.bh_sharpe:.2f}, max DD {pct(r.bh_mdd)} | "
              f"signals + regime Sharpe {r.signals_sharpe:.2f} | Lighthouse balanced CAGR {pct(r.lighthouse_cagr)}, "
              f"vol {pct(r.lighthouse_vol)}, Sharpe {r.lighthouse_sharpe:.2f}, max DD {pct(r.lighthouse_mdd)} | "
              f"beats buy & hold on Sharpe {int(r.beat_bh_sharpe)} / {int(r.tested)}, on CAGR {int(r.beat_bh_cagr)}, "
              f"lower drawdown {int(r.lower_mdd)}; {r.trades_per_stock:.1f} trades per stock | as one equal-weight cohort "
              f"over {r.cohort_years:.1f} years: buy & hold {pct(r.cohort_bh_cagr)} a year, Lighthouse {pct(r.cohort_lighthouse_cagr)}, "
              f"signals {pct(r.cohort_signals_cagr)}, the same exposure without timing {pct(r.cohort_same_cagr)} | "
              f"vs the same exposure: higher Sharpe on {int(r.beat_same_sharpe)}, lower drawdown on "
              f"{int(r.lower_mdd_than_same)}; average Sharpe {r.same_sharpe:.2f}, max DD {pct(r.same_mdd)}")
    print(f"\nwrote {out.relative_to(ROOT)} and its _summary.csv")
    return 0


def resummarize(date: str) -> int:
    """Rebuild the summary of a cohort run from its saved per-member CSV (no downloads, no backtests)."""
    from lighthouse import membership
    out = ROOT / "reports" / "examples" / f"cross_ticker_sp500_{pd.Timestamp(date).date()}.csv"
    df = pd.read_csv(out)
    uni = membership.sp500()
    members = uni.members_on(pd.Timestamp(date))
    old = pd.read_csv(out.with_name(out.stem + "_summary.csv"), index_col=0)
    missing = range(len(members) - df["ticker"].nunique())
    return report_cohort(df, members, missing, uni, pd.Timestamp(date), pd.Timestamp(old["end"].iloc[0]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed CSV instead of writing it")
    ap.add_argument("--cohort", default=None, help="test every S&P 500 member on this date (live data) instead")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--resummarize", action="store_true", help="with --cohort: rebuild the summary from the saved CSV")
    a = ap.parse_args(argv)
    if a.cohort:
        return resummarize(a.cohort) if a.resummarize else cohort(a.cohort, a.workers)
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
