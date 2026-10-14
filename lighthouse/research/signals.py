"""
The signal library for the research harness. Every signal is a number per stock and date, higher meaning "expected to
do better", computed only from data public on that date (each daily series is causal; tests truncate the data and
check that nothing in the past moves).

Families
  product      Lighthouse's own: the technical score (long and short weights), each of the six technical beacons,
               the point-in-time fundamental score and the long-horizon composite
  price        12-1 month momentum, low volatility, short-term reversal (and, at the panel level, industry and
               residual momentum: lighthouse.research.panel_signals)
  fundamental  from SEC filings by filing date (lighthouse.research.fundamental_signals)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
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


def momentum_12_1(df: pd.DataFrame) -> pd.Series:
    """Return from 12 months to 1 month ago (Jegadeesh & Titman 1993): the most recent month is skipped because it
    tends to reverse."""
    c = df["Close"]
    return c.shift(21) / c.shift(252) - 1.0


def low_volatility(df: pd.DataFrame) -> pd.Series:
    """Minus the annualised volatility of daily returns over the last year (low-volatility anomaly, Ang et al. 2006;
    Baker, Bradley & Wurgler 2011)."""
    r = np.log(df["Close"] / df["Close"].shift(1))
    return -(r.rolling(252, min_periods=200).std() * np.sqrt(252))


def reversal_1m(df: pd.DataFrame) -> pd.Series:
    """Minus last month's return (short-term reversal, Jegadeesh 1990; Lehmann 1990)."""
    c = df["Close"]
    return -(c / c.shift(21) - 1.0)


def high_52w(df: pd.DataFrame) -> pd.Series:
    """Price relative to its 52-week high (George & Hwang 2004)."""
    c = df["Close"]
    return c / c.rolling(252, min_periods=200).max()


PRICE_SIGNALS: Dict[str, SignalSpec] = {
    "mom_12_1": SignalSpec("mom_12_1", "price", "12-1 month momentum", momentum_12_1),
    "low_vol": SignalSpec("low_vol", "price", "low 1-year volatility", low_volatility),
    "reversal_1m": SignalSpec("reversal_1m", "price", "1-month reversal (minus last month's return)", reversal_1m),
    "high_52w": SignalSpec("high_52w", "price", "closeness to the 52-week high", high_52w),
}
PRODUCT_SIGNALS = ["tech_long", "tech_short", "beacon_trend", "beacon_rsi", "beacon_macd", "beacon_bands",
                   "beacon_volume", "beacon_breakout"]
DESCRIPTIONS = {
    "tech_long": "Lighthouse technical score, long-horizon weights", "tech_short": "technical score, short-horizon weights",
    "beacon_trend": "trend beacon (price and 50-day vs 200-day SMA)", "beacon_rsi": "RSI(14) beacon",
    "beacon_macd": "MACD histogram / ATR beacon", "beacon_bands": "low-lag price-band %B beacon",
    "beacon_volume": "volume confirmation beacon", "beacon_breakout": "52-week-high proximity beacon",
    "fund_pit": "Lighthouse point-in-time fundamental score (SECTOR_NORMS)",
    "composite_long": "Lighthouse long-horizon composite (0.6 fundamental + 0.4 technical)",
    **{k: v.description for k, v in PRICE_SIGNALS.items()},
}


def daily_signals(frames: Dict[str, pd.DataFrame], names: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.Series]]:
    """{signal: {ticker: daily series}} for the product and price signals."""
    want = set(names) if names is not None else set(PRODUCT_SIGNALS) | set(PRICE_SIGNALS)
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
        for k, spec in PRICE_SIGNALS.items():
            if k in want:
                out[k][t] = spec.fn(df)
    return out


__all__ = ["SignalSpec", "PRICE_SIGNALS", "PRODUCT_SIGNALS", "DESCRIPTIONS", "daily_signals", "product_signals",
           "momentum_12_1", "low_volatility", "reversal_1m", "high_52w"]
