"""Watchlist screener: analyse several tickers and rank the opportunities."""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from .config import InvestorProfile, StrategyParams
from .engine import analyze
from .parallel import map_tickers


def screen(tickers: Iterable[str], profile: Optional[InvestorProfile] = None, params: Optional[StrategyParams] = None,
           years: float = 5, source: str = "auto", with_backtest: bool = False, benchmark: str = "^GSPC",
           progress: bool = True, max_workers: Optional[int] = None) -> pd.DataFrame:
    """Rank a watchlist by Lighthouse score. Tickers that fail to load are reported, not dropped silently.
    Live tickers load in parallel threads (`max_workers`, default up to 8)."""
    done = (lambda t, k, n: print(f"  analysed {t} ({k}/{n})", flush=True)) if progress else None
    results, errors = map_tickers(lambda t: analyze(t, profile, params, years=years, source=source, run_backtest_=with_backtest,
                                                    benchmark=benchmark), tickers, max_workers, source, done)
    rows: List[dict] = []
    for t in dict.fromkeys(x.upper().strip() for x in tickers if x and x.strip()):
        try:
            if t in errors:
                raise RuntimeError(errors[t])
            r = results[t]
            d, f = r.decision, r.fundamentals
            row = {
                "ticker": t, "name": f.name or t, "verdict": d.verdict, "score": round(d.composite_score, 1),
                "fundamental": None if d.fundamental_score is None else round(d.fundamental_score, 1),
                "technical": round(d.technical_score, 1), "regime": d.regime.label if d.regime else "n/a",
                "position_pct": round(d.plan.position_pct, 1), "shares": d.plan.shares, "cost": round(d.plan.cost, 2),
                "price": round(d.price, 2),
                "stop": None if d.plan.stop_price is None else round(d.plan.stop_price, 2),
                "pe": f.pe_trailing, "roe": f.roe, "rev_growth": f.revenue_growth, "sector": f.sector,
                "data_source": r.prices.attrs.get("source"), "as_of": d.as_of, "error": "",
            }
            if with_backtest and r.backtest is not None:
                m = r.backtest.metrics
                row.update({"bt_cagr": m["cagr"], "bt_sharpe": m["sharpe"], "bt_max_dd": m["max_drawdown"], "bt_trades": m["n_trades"]})
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the screen
            msg = errors.get(t) or f"{type(exc).__name__}: {str(exc)[:160]}"
            rows.append({"ticker": t, "name": t, "verdict": "n/a", "score": None, "error": msg})
    df = pd.DataFrame(rows)
    if "score" in df:
        df = df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
    return df
