"""
The signal library for the research harness. Every signal is a number per stock and date, higher meaning "expected to
do better", computed only from data public on that date (each daily series is causal; tests truncate the data and
check that nothing in the past moves).

Families
  product      Lighthouse's own: the technical score (long and short weights) and each of the six technical beacons
  fundamental  from SEC filings by filing date (lighthouse.research.fundamental_signals)
  positioning  FINRA short-sale volume (lighthouse.research.positioning)
  events       the 3-day earnings reaction and the 10-K text features (lighthouse.research.event_signals)

The price anomalies (momentum, low volatility, reversal, 52-week high, size), industry and residual momentum, insider
trades, days to cover, 13F breadth and the earnings-release tone were tested once (docs/METHODOLOGY.md section 8b),
failed, and were removed in October 2026; their results stay in reports/research/experiments.csv with the commit that
computed them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..config import StrategyParams
from ..technicals import _beacon_scores, _compute_indicators, _technical_score
from ..config import TECHNICAL_WEIGHTS


@dataclass(frozen=True)
class SignalSpec:
    name: str
    family: str
    description: str
    fn: Callable[[pd.DataFrame], pd.Series]          # daily OHLCV frame -> daily signal


def _ind(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.attrs = {}
    return _compute_indicators(d, StrategyParams())


def product_signals(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """All of Lighthouse's technical signals for one stock in one pass (indicators are the expensive part)."""
    p = StrategyParams()
    ind = _ind(df)
    b = _beacon_scores(ind, p)
    out = {f"beacon_{k}": b[k] for k in b.columns}
    out["tech_long"] = _technical_score(ind, TECHNICAL_WEIGHTS["long"], p)
    out["tech_short"] = _technical_score(ind, TECHNICAL_WEIGHTS["short"], p)
    return out


PRODUCT_SIGNALS = ["tech_long", "tech_short", "beacon_trend", "beacon_rsi", "beacon_macd", "beacon_bands",
                   "beacon_volume", "beacon_breakout"]
DESCRIPTIONS = {
    "tech_long": "Lighthouse technical score, long-horizon weights", "tech_short": "technical score, short-horizon weights",
    "beacon_trend": "trend beacon (price and 50-day vs 200-day SMA)", "beacon_rsi": "RSI(14) beacon",
    "beacon_macd": "MACD histogram / ATR beacon", "beacon_bands": "low-lag price-band %B beacon",
    "beacon_volume": "volume confirmation beacon", "beacon_breakout": "52-week-high proximity beacon",
    "fund_pit": "Lighthouse point-in-time fundamental score (SECTOR_NORMS)",
    "composite_long": "Lighthouse long-horizon composite (0.6 fundamental + 0.4 technical)",
}


def daily_signals(frames: Dict[str, pd.DataFrame], names: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.Series]]:
    """{signal: {ticker: daily series}} for the product's signals."""
    want = set(names) if names is not None else set(PRODUCT_SIGNALS)
    out: Dict[str, Dict[str, pd.Series]] = {k: {} for k in want}
    need_product = bool(want & set(PRODUCT_SIGNALS))
    for t, df in frames.items():
        if len(df) < 260:
            continue
        if need_product:
            ps = product_signals(df)
            for k, v in ps.items():
                if k in want:
                    out[k][t] = v
    return out


__all__ = ["SignalSpec", "PRODUCT_SIGNALS", "DESCRIPTIONS", "daily_signals", "product_signals"]
