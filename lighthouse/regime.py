"""
Market regime ("market weather"): is the broad market in an up-trend or a
down-trend, and is volatility calm or elevated?

Trend  : benchmark close above / below its 200-day SMA  -> Bull / Bear
Vol    : 21-day realised vol vs. 1.25x its trailing 1-year median -> Calm / Turbulent

Both tests are causal, so the regime series can be used inside the backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import indicators as ind
from . import memo
from .config import StrategyParams


@dataclass
class RegimeSnapshot:
    trend: str                 # "bull" | "bear"
    volatility: str            # "calm" | "turbulent"
    benchmark_vs_sma: float    # % distance of benchmark from its 200-day SMA
    vol: float                 # annualised realised vol of the benchmark
    vol_median: float          # its trailing 1-year median
    as_of: pd.Timestamp
    benchmark: str = "^GSPC"

    @property
    def label(self) -> str:
        return f"{self.trend.title()} / {self.volatility.title()}"

    @property
    def is_bear(self) -> bool:
        return self.trend == "bear"

    @property
    def is_turbulent(self) -> bool:
        return self.volatility == "turbulent"

    def describe(self) -> str:
        t = ("the S&P 500 is above its 200-day average (up-trend)" if not self.is_bear
             else "the S&P 500 is below its 200-day average (down-trend)")
        v = (f"realised volatility ({self.vol * 100:.0f}%) is {'well above' if self.is_turbulent else 'near or below'} "
             f"its one-year norm ({self.vol_median * 100:.0f}%)")
        return f"Market weather is {self.label}: {t} and {v}."


def regime_table(benchmark: pd.DataFrame, params: Optional[StrategyParams] = None) -> pd.DataFrame:
    """Daily regime classification of the benchmark (all columns causal). Memoised on the data and parameters."""
    p = params or StrategyParams()
    return memo.memoize(("regime", memo.frame_fingerprint(benchmark), memo.params_key(p)), lambda: _regime_table(benchmark, p))


def _regime_table(benchmark: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    close = benchmark["Close"].copy()
    close.attrs = {}      # see technicals.compute_indicators: attrs are deep-copied by every pandas operation
    df = pd.DataFrame(index=benchmark.index)
    df["close"] = close
    df["sma"] = ind.sma(close, p.regime_sma)
    df["vol"] = ind.realized_vol(close, p.regime_vol_window, p.trading_days)
    df["vol_median"] = df["vol"].rolling(p.regime_vol_lookback, min_periods=p.regime_vol_window * 3).median()
    ready = df["sma"].notna() & df["vol_median"].notna()
    bull = (close > df["sma"]).astype(float).where(ready)
    turbulent = (df["vol"] > p.regime_vol_multiple * df["vol_median"]).astype(float).where(ready)
    df["bull"] = bull
    df["turbulent"] = turbulent
    labels = [f"{'Bull' if b == 1.0 else 'Bear'} / {'Turbulent' if t == 1.0 else 'Calm'}" for b, t in zip(bull, turbulent)]
    df["regime"] = pd.Series(labels, index=df.index, dtype="object").where(ready)
    return df


def latest_regime(table: pd.DataFrame, benchmark_name: str = "^GSPC") -> Optional[RegimeSnapshot]:
    valid = table.dropna(subset=["regime"])
    if valid.empty:
        return None
    row = valid.iloc[-1]
    return RegimeSnapshot(
        trend="bull" if row["bull"] == 1.0 else "bear",
        volatility="turbulent" if row["turbulent"] == 1.0 else "calm",
        benchmark_vs_sma=float(row["close"] / row["sma"] - 1.0),
        vol=float(row["vol"]), vol_median=float(row["vol_median"]),
        as_of=valid.index[-1], benchmark=benchmark_name,
    )


def regime_adjustments(bear: bool, turbulent: bool, params: Optional[StrategyParams] = None) -> dict:
    """How the regime changes the decision rules."""
    p = params or StrategyParams()
    return {
        "score_penalty": (p.bear_score_penalty if bear else 0.0) + (p.turbulent_score_penalty if turbulent else 0.0),
        "entry_penalty": p.bear_entry_penalty if bear else 0.0,
        "size_multiplier": p.bear_size_multiplier if bear else 1.0,
        "block_entries": bool(p.block_entries_in_bear_turbulent and bear and turbulent),
    }
