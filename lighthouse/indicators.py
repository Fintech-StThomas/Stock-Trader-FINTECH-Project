"""
Pure technical-indicator functions.

Every function is *causal*: the value at date t uses only data up to and
including t (rolling windows and exponential averages never look forward).
This is what makes the backtest free of look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # when there were no losses at all RS is infinite -> RSI 100
    out = out.where(avg_loss != 0.0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower)
    pct_b = (close - lower) / width.replace(0.0, np.nan)
    bandwidth = width / mid.replace(0.0, np.nan)
    return mid, upper, lower, pct_b, bandwidth


def wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly weighted moving average (weights 1..window, the newest bar weighs most)."""
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if window >= 1 and len(x) >= window:
        w = np.arange(1, window + 1, dtype=float)
        out[window - 1:] = np.convolve(x, w[::-1], mode="valid") / w.sum()   # NaN inside a window propagates
    return pd.Series(out, index=series.index)


def hull_ma(series: pd.Series, window: int) -> pd.Series:
    """Hull moving average (Hull 2005): WMA(2 x WMA(n/2) - WMA(n), sqrt(n)). Doubling the fast average and subtracting
    the slow one cancels most of the lag; the final short WMA smooths the result. Causal like every input."""
    half, root = max(int(window // 2), 1), max(int(round(np.sqrt(window))), 1)
    return wma(2.0 * wma(series, half) - wma(series, window), root)


def lowlag_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bands that follow the price: a Hull moving average as the centre, +/- `num_std` standard deviations of the
    closes around it over the last `window` sessions. They answer the same question as Bollinger Bands (is price
    stretched?) relative to where price is now rather than where it was ~10 sessions ago. Causal: no shift forward."""
    mid = hull_ma(close, window)
    std = (close - mid).rolling(window, min_periods=window).std(ddof=0)
    upper, lower = mid + num_std * std, mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return mid, upper, lower, pct_b


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume. Returns NaN series if volume is unavailable."""
    if volume is None or volume.isna().all():
        return pd.Series(np.nan, index=close.index)
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Simple price momentum: % change over `window` trading days."""
    return close / close.shift(window) - 1.0


def realized_vol(close: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    """Annualised realised volatility of daily log returns."""
    r = np.log(close / close.shift(1))
    return r.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(trading_days)


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown(equity)
    return float(dd.min()) if len(dd) else float("nan")
