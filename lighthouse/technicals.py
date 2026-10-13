"""
Technical analysis: six "beacons" that each vote between -1 (bearish) and +1
(bullish). The weighted average becomes the Technical Score (0-100).

The same vectorised functions drive both the live recommendation and the
backtest, so what the investor sees today is exactly what was tested.

Beacons
-------
trend      Price vs 200-day SMA and 50-day vs 200-day SMA (Golden / Death Cross)
rsi        14-day RSI: confirms momentum between 30-70, flags exhaustion beyond
macd       MACD histogram normalised by ATR (momentum acceleration)
bands      Position inside the low-lag price bands (%B, Hull MA centre): trend confirmation vs. over-extension
volume     Volume confirmation: expanding volume behind the move + OBV trend
breakout   Distance from the 52-week high (breakout proximity)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import StrategyParams, TECHNICAL_WEIGHTS


@dataclass
class Beacon:
    """One explainable signal."""
    key: str
    name: str
    group: str            # "technical" | "fundamental"
    value: str            # human-readable reading, e.g. "RSI 61.2"
    score: float          # -1 .. +1
    weight: float         # within its group
    rationale: str

    @property
    def contribution(self) -> float:
        """Points added to (or removed from) the group score (0-100 scale)."""
        return 50.0 * self.weight * self.score

    @property
    def light(self) -> str:
        if np.isnan(self.score):
            return "grey"
        if self.score >= 0.25:
            return "green"
        if self.score <= -0.25:
            return "red"
        return "amber"


# --------------------------------------------------------------------------- #
# Indicator table
# --------------------------------------------------------------------------- #
def compute_indicators(prices: pd.DataFrame, params: Optional[StrategyParams] = None) -> pd.DataFrame:
    """Return a copy of `prices` with all indicator columns added (all causal)."""
    p = params or StrategyParams()
    df = prices.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    volume = df["Volume"] if "Volume" in df else pd.Series(np.nan, index=df.index)

    df["sma_fast"] = ind.sma(close, p.sma_fast)
    df["sma_slow"] = ind.sma(close, p.sma_slow)
    df["rsi"] = ind.rsi(close, p.rsi_window)
    df["macd"], df["macd_signal"], df["macd_hist"] = ind.macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    df["band_mid"], df["band_upper"], df["band_lower"], df["pct_b"] = ind.lowlag_bands(close, p.band_window, p.band_std)
    df["atr"] = ind.atr(high, low, close, p.atr_window)
    df["atr_pct"] = df["atr"] / close
    df["realized_vol"] = ind.realized_vol(close, p.vol_window, p.trading_days)
    df["sizing_vol"] = ind.realized_vol(close, p.sizing_vol_window, p.trading_days)
    df["high_52w"] = close.rolling(p.high_window, min_periods=p.high_window).max()
    df["dist_high"] = close / df["high_52w"] - 1.0
    df["ret_20d"] = ind.momentum(close, p.volume_window)

    has_volume = volume.notna().sum() > p.volume_baseline
    if has_volume:
        vol = volume.astype(float)
        df["vol_ratio"] = vol.rolling(p.volume_window, min_periods=p.volume_window).mean() / \
            vol.rolling(p.volume_baseline, min_periods=p.volume_baseline).mean()
        df["obv"] = ind.obv(close, vol)
        df["obv_sma"] = ind.sma(df["obv"], p.volume_window)
    else:
        df["vol_ratio"] = np.nan
        df["obv"] = np.nan
        df["obv_sma"] = np.nan
    return df


# --------------------------------------------------------------------------- #
# Beacon scores (vectorised, each in [-1, 1])
# --------------------------------------------------------------------------- #
def _clip(x, lo=-1.0, hi=1.0):
    return np.clip(x, lo, hi)


def beacon_scores(ind_df: pd.DataFrame, params: Optional[StrategyParams] = None) -> pd.DataFrame:
    """Per-date score of every technical beacon. NaN where an indicator is undefined."""
    p = params or StrategyParams()
    close = ind_df["Close"]
    out = pd.DataFrame(index=ind_df.index)

    # 1) Trend: continuous version of "price above 200-SMA" and "Golden/Death Cross".
    #    +/-10% from the 200-day SMA saturates the first term; +/-5% gap between
    #    the 50 and 200 SMA saturates the second.
    t1 = _clip((close / ind_df["sma_slow"] - 1.0) / 0.10)
    t2 = _clip((ind_df["sma_fast"] / ind_df["sma_slow"] - 1.0) / 0.05)
    out["trend"] = 0.5 * t1 + 0.5 * t2

    # 2) RSI: linear between oversold/overbought, fading beyond (exhaustion).
    r = ind_df["rsi"]
    mid = 50.0
    half = (p.rsi_overbought - p.rsi_oversold) / 2.0
    core = _clip((r - mid) / half)
    over = _clip(1.0 - (r - p.rsi_overbought) / 15.0)      # 70 -> +1, 85 -> 0, 100 -> -1
    under = _clip(-1.0 + (p.rsi_oversold - r) / 15.0)      # 30 -> -1, 15 -> 0,   0 -> +1
    rsi_score = core.where(r <= p.rsi_overbought, over).where(r >= p.rsi_oversold, under)
    out["rsi"] = rsi_score.where(r.notna())

    # 3) MACD histogram scaled by ATR (one ATR of histogram = full conviction).
    out["macd"] = _clip(ind_df["macd_hist"] / ind_df["atr"].replace(0.0, np.nan))

    # 4) Price-band %B: 0.5 = on the centre line; above the upper band = over-extended.
    b = ind_df["pct_b"]
    core_b = _clip((b - 0.5) * 2.0)
    over_b = _clip(0.5 - (b - 1.0) * 2.0)      # 1.0 -> +0.5, 1.25 -> 0, 1.75 -> -1
    under_b = _clip(-0.5 + (0.0 - b) * 2.0)    # 0.0 -> -0.5, 0.25 below -> 0
    band_score = core_b.where(b <= 1.0, over_b).where(b >= 0.0, under_b)
    out["bands"] = band_score.where(b.notna())

    # 5) Volume confirmation: expanding volume behind the 20-day move (+/-) and OBV trend.
    if ind_df["vol_ratio"].notna().any():
        expansion = _clip((ind_df["vol_ratio"] - 1.0) / 0.5, 0.0, 1.0)      # 1.5x baseline = full
        direction = np.sign(ind_df["ret_20d"])
        confirm = 0.5 * direction * expansion
        obv_trend = 0.5 * np.sign(ind_df["obv"] - ind_df["obv_sma"])
        v = confirm + obv_trend
        out["volume"] = v.where(ind_df["vol_ratio"].notna() & ind_df["obv_sma"].notna())
    else:
        out["volume"] = np.nan   # neutral + flagged as unavailable

    # 6) Breakout proximity: at the 52-week high = +1, 20% below = 0, 40% below = -1.
    out["breakout"] = _clip(1.0 + ind_df["dist_high"] / 0.20)
    return out


def technical_score(ind_df: pd.DataFrame, horizon: str = "long",
                    params: Optional[StrategyParams] = None,
                    weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """Weighted technical score 0-100 for every date (NaN until indicators warm up).

    Beacons that are unavailable (NaN) are dropped and the remaining weights are
    renormalised, so a missing volume feed never silently biases the score.
    """
    w = weights or TECHNICAL_WEIGHTS[horizon]
    scores = beacon_scores(ind_df, params)
    wser = pd.Series(w)
    cols = [c for c in wser.index if c in scores.columns]
    s = scores[cols]
    wmat = pd.DataFrame(np.tile(wser[cols].values, (len(s), 1)), index=s.index, columns=cols)
    wmat = wmat.where(s.notna(), 0.0)
    total_w = wmat.sum(axis=1)
    weighted = (s.fillna(0.0) * wmat).sum(axis=1) / total_w.replace(0.0, np.nan)
    # require the core trend/breakout beacons (slow SMA, 52w high) before producing a score
    ready = scores["trend"].notna() & scores["breakout"].notna()
    return (50.0 + 50.0 * weighted).where(ready)


# --------------------------------------------------------------------------- #
# Explainable beacons for the latest date
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def latest_beacons(ind_df: pd.DataFrame, horizon: str = "long",
                   params: Optional[StrategyParams] = None) -> List[Beacon]:
    p = params or StrategyParams()
    w = TECHNICAL_WEIGHTS[horizon]
    scores = beacon_scores(ind_df, p)
    row, sc = ind_df.iloc[-1], scores.iloc[-1]
    beacons: List[Beacon] = []

    # trend
    above = row["Close"] > row["sma_slow"]
    golden = row["sma_fast"] > row["sma_slow"]
    cross = "Golden Cross (50 > 200)" if golden else "Death Cross (50 < 200)"
    beacons.append(Beacon(
        "trend", "Trend (50/200-day SMA)", "technical",
        f"price {_fmt_pct(row['Close'] / row['sma_slow'] - 1)} vs 200-SMA; {cross}",
        float(sc["trend"]), w["trend"],
        ("Price is above its 200-day average and the 50-day sits above the 200-day: the "
         "long-term trend is up." if above and golden else
         "Price is below its 200-day average and the 50-day sits below the 200-day: the "
         "long-term trend is down." if (not above and not golden) else
         "Mixed trend signals: price and the moving averages disagree, which often happens near a turning point.")))

    # rsi
    r = row["rsi"]
    if r > p.rsi_overbought:
        why = f"RSI above {p.rsi_overbought:.0f} is overbought: momentum is strong but stretched, so the signal fades."
    elif r < p.rsi_oversold:
        why = f"RSI below {p.rsi_oversold:.0f} is oversold: selling pressure dominates, though a bounce becomes more likely."
    elif r >= 50:
        why = "RSI between 50 and 70 confirms healthy, non-extreme upward momentum."
    else:
        why = "RSI between 30 and 50 shows momentum tilted downward without being washed out."
    beacons.append(Beacon("rsi", f"Momentum (RSI {p.rsi_window})", "technical", f"RSI {r:.1f}", float(sc["rsi"]), w["rsi"], why))

    # macd
    h = row["macd_hist"]
    beacons.append(Beacon(
        "macd", "MACD (12/26/9)", "technical",
        f"histogram {h:+.2f} ({h / row['atr']:+.2f} ATR)" if row["atr"] else f"histogram {h:+.2f}",
        float(sc["macd"]), w["macd"],
        "MACD line is above its signal line: short-term momentum is accelerating upward."
        if h > 0 else "MACD line is below its signal line: short-term momentum is fading or negative."))

    # price bands
    b = row["pct_b"]
    if b > 1:
        why = "Price closed above the upper band: extended, with higher odds of a pause or pull-back."
    elif b < 0:
        why = "Price closed below the lower band: washed out, which can precede a rebound but confirms weakness."
    elif b >= 0.5:
        why = "Price sits in the upper half of its bands: buyers are in control without over-extension."
    else:
        why = "Price sits in the lower half of its bands: sellers are in control."
    beacons.append(Beacon("bands", f"Price bands (Hull {p.band_window}, {p.band_std:g}σ)", "technical",
                          f"%B {b:.2f}", float(sc["bands"]), w["bands"], why))

    # volume
    if pd.notna(sc["volume"]):
        vr = row["vol_ratio"]
        obv_up = row["obv"] > row["obv_sma"]
        why = (f"Recent volume is {vr:.2f}x its 3-month average and on-balance volume is "
               f"{'rising' if obv_up else 'falling'}: the move is {'confirmed' if (sc['volume'] > 0) else 'not confirmed'} by participation.")
        beacons.append(Beacon("volume", "Volume confirmation (OBV, 20/60-day)", "technical",
                              f"volume {vr:.2f}x avg; OBV {'up' if obv_up else 'down'}", float(sc["volume"]), w["volume"], why))
    else:
        beacons.append(Beacon("volume", "Volume confirmation (OBV, 20/60-day)", "technical",
                              "n/a", float("nan"), w["volume"],
                              "Volume data is unavailable for this series, so this beacon is excluded and the others re-weighted."))

    # breakout
    d = row["dist_high"]
    beacons.append(Beacon(
        "breakout", "52-week high proximity", "technical", f"{_fmt_pct(d)} from 52-week high",
        float(sc["breakout"]), w["breakout"],
        "Trading at or near its 52-week high: a breakout zone where trend-followers add exposure." if d > -0.03 else
        "Within 15% of its 52-week high: the up-trend is intact." if d > -0.15 else
        "Well below its 52-week high: the stock would need a sustained recovery before trend investors re-enter."))
    return beacons
