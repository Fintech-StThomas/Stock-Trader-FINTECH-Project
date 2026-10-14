"""
Trading costs that scale with liquidity, estimated from free daily data (no quotes, no order book).

Per side, in basis points of the traded value:
  commission    a fixed fee (default 5 bps, StrategyParams.cost_bps; many US brokers now charge 0)
  half spread   Corwin & Schultz (2012): the bid-ask spread implied by daily highs and lows (a high is usually a buy at
                the ask, a low a sell at the bid; over two days volatility grows but the spread does not). Computed
                from the day before and the day itself, so the estimate on day t uses nothing after t; averaged over
                21 sessions; floored at `min_half_spread_bps` (it reads zero on close-only data such as most of the
                sample pack, and is noisy day to day).
  impact        the square-root law: k x daily volatility x sqrt(order value / average daily dollar volume), with
                k = 1 (Toth et al. 2011; Almgren et al. 2005). Zero when volume is unknown.

Shorts also pay a borrow fee: StrategyParams.short_borrow_fee, an annual rate charged on the short's value for every
day it is held. Free data has no borrow rates; the default (0.5% a year, a typical general-collateral rate for large
caps) is an assumption, shown as one; hard-to-borrow stocks cost several percent to over 100% a year.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_spread(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Proportional bid-ask spread (full spread, e.g. 0.001 = 10 bps) from daily highs and lows, causal: the two-day
    estimate dated t uses days t-1 and t; negative two-day estimates count as zero; mean over `window` days."""
    h, l = high.astype(float), low.astype(float)
    ok = (h > 0) & (l > 0) & (h >= l)
    lhl = np.log(h.where(ok) / l.where(ok)) ** 2
    beta = lhl + lhl.shift(1)
    hmax = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    lmin = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(hmax / lmin) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s.clip(lower=0.0)
    return s.rolling(window, min_periods=max(5, window // 3)).mean()


def dollar_volume(close: pd.Series, volume: Optional[pd.Series], window: int = 21) -> pd.Series:
    if volume is None or volume.isna().all():
        return pd.Series(np.nan, index=close.index)
    return (close * volume).rolling(window, min_periods=max(5, window // 3)).mean()


def impact_bps(order_value: float, adv_value, daily_vol, k: float = 1.0):
    """Square-root market impact in bps (vectorised over adv_value and daily_vol)."""
    adv = np.asarray(adv_value, dtype=float)
    vol = np.asarray(daily_vol, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = k * vol * np.sqrt(np.maximum(order_value, 0.0) / adv) * 1e4
    return np.where(np.isfinite(out), out, 0.0)


@dataclass
class CostModel:
    commission_bps: float = 5.0
    min_half_spread_bps: float = 5.0     # floor: the old fixed "slippage" (and the value on close-only data)
    impact_k: float = 1.0
    spread_window: int = 21

    def per_side_bps(self, df: pd.DataFrame, order_value: float) -> pd.Series:
        """Cost per side (bps) for every day of an OHLCV frame, for an order of `order_value` dollars."""
        cs = corwin_schultz_spread(df["High"], df["Low"], self.spread_window) * 1e4 / 2.0
        half = cs.fillna(0.0).clip(lower=self.min_half_spread_bps)
        r = np.log(df["Close"] / df["Close"].shift(1))
        dvol = r.rolling(self.spread_window, min_periods=5).std()
        adv = dollar_volume(df["Close"], df["Volume"] if "Volume" in df else None, self.spread_window)
        imp = pd.Series(impact_bps(order_value, adv.to_numpy(), dvol.to_numpy(), self.impact_k), index=df.index)
        return (self.commission_bps + half + imp).rename("cost_bps")


def borrow_cost(value_held: float, days: int, annual_fee: float, trading_days: int = 252) -> float:
    """Borrow fee in dollars for holding a short of `value_held` for `days` sessions."""
    return float(value_held) * float(annual_fee) * float(days) / float(trading_days)


__all__ = ["CostModel", "corwin_schultz_spread", "dollar_volume", "impact_bps", "borrow_cost"]
