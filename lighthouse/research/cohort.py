"""
The per-stock rule test on a fixed cohort: every member of an index on one date, followed until today or until it
stopped trading, which is what an investor choosing from that index on that date would have faced.

Each member with prices is backtested exactly as `scripts/cross_ticker_evaluation.py` tests the sample pack (signals
+ regime fully invested, the balanced and conservative products, buy-and-hold), under both delisting rules:

  exclude      the last price stands (as if the investor sold at it)
  pessimistic  a stock whose prices stop before the end of the data is worth nothing the next session (-100% for
               buy-and-hold and for an open position)

Members without any price in the window cannot be tested: they count against coverage, and the summary says how many.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest import compute_metrics, run_backtest
from ..config import InvestorProfile, StrategyParams
from ..data import align
from ..fundamentals import point_in_time_scores
from ..total_return import DELISTING_RULES, apply_delisting, is_delisted


@dataclass
class CohortInputs:
    frames: Dict[str, pd.DataFrame]            # clean OHLCV per member
    benchmark: pd.DataFrame                    # the market index (regime)
    risk_free: Optional[pd.Series] = None      # daily T-bill (None: params.risk_free_rate)
    pit: Dict[str, pd.Series] = field(default_factory=dict)   # point-in-time fundamental scores per member
    data_end: Optional[pd.Timestamp] = None    # the end of the data (a series ending well before it is delisted)


def evaluate_member(ticker: str, inp: CohortInputs, start, rule: str, params: Optional[StrategyParams] = None) -> Optional[dict]:
    p = params or StrategyParams()
    df = inp.frames[ticker]
    df = df[df.index >= pd.Timestamp(start) - pd.Timedelta(days=int(365.25 * 1.3))]   # warm-up for the 200-day SMA
    if len(df) < p.min_history + 30:
        return None
    end = inp.data_end or inp.benchmark.index[-1]
    delisted = is_delisted(df, end)
    df = apply_delisting(df, rule, end)
    try:
        px, bench = align(df, inp.benchmark)
    except Exception:  # noqa: BLE001 - no overlap with the benchmark
        return None
    px = px.copy()
    px.attrs = {"ticker": ticker}
    bench = bench.copy()
    bench.attrs = {}
    pit = inp.pit.get(ticker)
    pit = pit.reindex(px.index) if pit is not None else None
    out: Dict[str, object] = {"ticker": ticker, "rule": rule, "delisted": delisted,
                              "last_price": str(inp.frames[ticker].index[-1].date())}
    try:
        bal = run_backtest(px, bench, p, InvestorProfile(horizon="long", risk="balanced"), pit, start=start,
                           risk_free=inp.risk_free)
        sig = run_backtest(px, bench, p, InvestorProfile(horizon="long", risk="balanced"), pit, use_regime=True,
                           use_fundamental_gate=False, use_stops=False, sizing="full", start=start, risk_free=inp.risk_free)
        cons = run_backtest(px, bench, p, InvestorProfile(horizon="long", risk="conservative"), pit, start=start,
                            risk_free=inp.risk_free)
    except ValueError:
        return None
    m, s, c = bal.metrics, sig.metrics, cons.metrics
    bh = compute_metrics(bal.buy_hold, bal.rf, p.trading_days)
    out.update({"start": str(bal.start.date()), "end": str(bal.end.date()), "years": bh["years"],
                "signals_cagr": s["cagr"], "signals_sharpe": s["sharpe"], "signals_mdd": s["max_drawdown"],
                "lighthouse_cagr": m["cagr"], "lighthouse_vol": m["volatility"], "lighthouse_sharpe": m["sharpe"],
                "lighthouse_mdd": m["max_drawdown"], "trades": int(m["n_trades"]), "win_rate": m["win_rate"],
                "exposure": m["exposure"],
                "conservative_cagr": c["cagr"], "conservative_vol": c["volatility"], "conservative_sharpe": c["sharpe"],
                "conservative_mdd": c["max_drawdown"],
                "bh_cagr": bh["cagr"], "bh_vol": bh["volatility"], "bh_sharpe": bh["sharpe"], "bh_mdd": bh["max_drawdown"],
                "fundamentals": bool(pit is not None and pit.notna().any())})
    return out


def summarize(df: pd.DataFrame, members: int) -> pd.DataFrame:
    """One row per delisting rule.

    Per-stock columns average the tested members (those with prices). Cohort columns treat the cohort as one portfolio,
    equal weight at the start (one sleeve per member, never rebalanced): its final wealth is the average of the
    members' final wealth multiples. Under "exclude" the untested members are left out; under "pessimistic" each of
    them is worth nothing (-100%), for buy-and-hold and for the rule alike, because without prices nothing is known
    about when the rule would have held them. Free price feeds drop delisted histories, so the untested members are
    mostly the ones that delisted."""
    rows = {}
    for rule, g in df.groupby("rule"):
        years = float(g["years"].max())
        n_tested = len(g)
        n_cohort = members if rule == "pessimistic" else n_tested
        w_bh = ((1.0 + g.bh_cagr) ** g.years).sum() / n_cohort
        w_lh = ((1.0 + g.lighthouse_cagr) ** g.years).sum() / n_cohort
        w_sig = ((1.0 + g.signals_cagr) ** g.years).sum() / n_cohort
        cagr = lambda w: float(w ** (1.0 / years) - 1.0) if w > 0 and years > 0 else -1.0   # noqa: E731
        rows[rule] = {
            "members": members, "tested": n_tested, "coverage": n_tested / members if members else np.nan,
            "delisted_tested": int(g["delisted"].sum()),
            "bh_cagr": g.bh_cagr.mean(), "bh_sharpe": g.bh_sharpe.mean(), "bh_mdd": g.bh_mdd.mean(),
            "signals_cagr": g.signals_cagr.mean(), "signals_sharpe": g.signals_sharpe.mean(),
            "lighthouse_cagr": g.lighthouse_cagr.mean(), "lighthouse_vol": g.lighthouse_vol.mean(),
            "lighthouse_sharpe": g.lighthouse_sharpe.mean(), "lighthouse_mdd": g.lighthouse_mdd.mean(),
            "conservative_sharpe": g.conservative_sharpe.mean(),
            "beat_bh_sharpe": int((g.lighthouse_sharpe > g.bh_sharpe).sum()),
            "beat_bh_cagr": int((g.lighthouse_cagr > g.bh_cagr).sum()),
            "lower_mdd": int((g.lighthouse_mdd > g.bh_mdd).sum()),
            "trades_per_stock": g.trades.mean(),
            "cohort_years": years, "cohort_bh_cagr": cagr(w_bh), "cohort_signals_cagr": cagr(w_sig),
            "cohort_lighthouse_cagr": cagr(w_lh),
        }
    return pd.DataFrame(rows).T


__all__ = ["CohortInputs", "evaluate_member", "summarize", "DELISTING_RULES"]
