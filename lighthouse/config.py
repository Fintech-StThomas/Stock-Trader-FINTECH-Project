"""
Configuration: investor profiles and strategy parameters.

Every number in this file is a *decision* that the product makes on behalf of
the investor. They are deliberately kept in one place so a non-programmer can
change them without touching the analysis code, and so the team can defend
each one (see docs/METHODOLOGY.md for the reasoning behind each default).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
# Investor profile
# --------------------------------------------------------------------------- #
# Risk presets: how much of the portfolio a single position may take, what
# annualised volatility we target for that position, and how far the
# protective stop sits (in Average True Range units).
RISK_PRESETS: Dict[str, Dict[str, float]] = {
    #                 max single position   target volatility   stop distance
    "conservative": {"max_position_pct": 10.0, "target_vol": 0.10, "stop_atr_multiple": 3.0},
    "balanced":     {"max_position_pct": 20.0, "target_vol": 0.15, "stop_atr_multiple": 2.5},
    "aggressive":   {"max_position_pct": 35.0, "target_vol": 0.25, "stop_atr_multiple": 2.0},
}

# Horizon presets: how the composite score blends fundamentals and technicals.
# Long-term investors are buying the business (fundamentals dominate); short-
# term investors are trading price behaviour (technicals dominate).
HORIZON_WEIGHTS: Dict[str, Tuple[float, float]] = {
    #          (fundamental, technical)
    "long":  (0.60, 0.40),
    "short": (0.30, 0.70),
}

# Stops: a long-term investor needs room for normal pull-backs, so the ATR stop multiple
# from the risk preset is scaled up for the long horizon.
HORIZON_STOP_FACTOR: Dict[str, float] = {"long": 2.0, "short": 1.0}

# Technical beacon weights per horizon (must sum to 1).
TECHNICAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "long":  {"trend": 0.35, "rsi": 0.15, "macd": 0.10, "bands": 0.10, "volume": 0.10, "breakout": 0.20},
    "short": {"trend": 0.20, "rsi": 0.20, "macd": 0.20, "bands": 0.15, "volume": 0.15, "breakout": 0.10},
}

# Fundamental group weights (must sum to 1).
FUNDAMENTAL_WEIGHTS: Dict[str, float] = {"valuation": 0.40, "quality": 0.35, "growth": 0.25}

# Composite score -> verdict. Scores are 0-100.
VERDICT_BANDS = [
    (72, "Strong Buy"),
    (60, "Buy"),
    (45, "Hold"),
    (32, "Reduce"),
    (0,  "Sell"),
]


@dataclass
class InvestorProfile:
    """What the investor tells Lighthouse about themselves."""
    horizon: str = "long"            # "long" (6-24 months+) or "short" (weeks to a few months)
    risk: str = "balanced"           # "conservative" | "balanced" | "aggressive"
    max_loss_pct: float = 2.0        # max acceptable loss on ONE position, as % of the total account
    portfolio_value: float = 10_000  # TOTAL account value = cash + stocks already held; the risk limits are % of this
    cash_available: Optional[float] = None   # cash you can invest right now; None = the whole account is cash
    # optional expert overrides of the risk preset (None = use the preset)
    max_position_override: Optional[float] = None
    target_vol_override: Optional[float] = None
    stop_atr_override: Optional[float] = None

    def __post_init__(self) -> None:
        self.horizon = str(self.horizon).lower().strip()
        self.risk = str(self.risk).lower().strip()
        if self.horizon not in HORIZON_WEIGHTS:
            raise ValueError(f"horizon must be one of {list(HORIZON_WEIGHTS)}, got {self.horizon!r}")
        if self.risk not in RISK_PRESETS:
            raise ValueError(f"risk must be one of {list(RISK_PRESETS)}, got {self.risk!r}")
        if not (0 < self.max_loss_pct <= 100):
            raise ValueError("max_loss_pct must be in (0, 100]")
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")
        if self.cash_available is not None:
            if self.cash_available < 0:
                raise ValueError("cash_available cannot be negative")
            if self.cash_available > self.portfolio_value + 1e-9:
                raise ValueError("cash_available cannot exceed portfolio_value (the account = cash + holdings)")

    @classmethod
    def from_cash(cls, cash: float, holdings_value: float = 0.0, **kw) -> "InvestorProfile":
        """Profile from what the investor actually has: cash to invest now plus the value of stocks already held."""
        if cash < 0 or holdings_value < 0:
            raise ValueError("cash and holdings_value cannot be negative")
        return cls(portfolio_value=float(cash) + float(holdings_value), cash_available=float(cash), **kw)

    # convenience accessors --------------------------------------------------
    @property
    def account_value(self) -> float:
        return float(self.portfolio_value)

    @property
    def cash(self) -> float:
        """Spendable cash: caps what a new position may cost."""
        return float(self.portfolio_value if self.cash_available is None else self.cash_available)

    @property
    def holdings_value(self) -> float:
        return self.account_value - self.cash

    @property
    def max_loss_value(self) -> float:
        return self.account_value * self.max_loss_pct / 100.0

    @property
    def fundamental_weight(self) -> float:
        return HORIZON_WEIGHTS[self.horizon][0]

    @property
    def technical_weight(self) -> float:
        return HORIZON_WEIGHTS[self.horizon][1]

    @property
    def technical_weights(self) -> Dict[str, float]:
        return TECHNICAL_WEIGHTS[self.horizon]

    @property
    def max_position_pct(self) -> float:
        return float(self.max_position_override if self.max_position_override is not None else RISK_PRESETS[self.risk]["max_position_pct"])

    @property
    def target_vol(self) -> float:
        return float(self.target_vol_override if self.target_vol_override is not None else RISK_PRESETS[self.risk]["target_vol"])

    @property
    def stop_atr_multiple(self) -> float:
        if self.stop_atr_override is not None:
            return float(self.stop_atr_override)
        return float(RISK_PRESETS[self.risk]["stop_atr_multiple"] * HORIZON_STOP_FACTOR[self.horizon])

    def with_overrides(self, **kw) -> "InvestorProfile":
        d = self.to_dict(); d.update(kw)
        return InvestorProfile(**d)

    def describe(self) -> str:
        if self.cash_available is None or abs(self.cash - self.account_value) < 0.5:
            money = f"${self.cash:,.0f} cash to invest"
        else:
            money = f"${self.cash:,.0f} cash to invest in a ${self.account_value:,.0f} account"
        return f"{self.horizon}-term, {self.risk} risk, {money}, max loss {self.max_loss_pct:.1f}% of the account per position"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategyParams:
    """Indicator windows, thresholds and trading-rule settings."""
    # --- trend ---------------------------------------------------------------
    sma_fast: int = 50
    sma_slow: int = 200
    # --- momentum ------------------------------------------------------------
    rsi_window: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # --- volatility / bands --------------------------------------------------
    band_alpha: float = 0.4       # price bands: centre = Holt's trend filter; share of each close taken into the level ...
    band_beta: float = 0.1        # ... and of each level change taken into the slope (larger = less lag, rougher line)
    band_window: int = 20         # bands at +/- band_std standard deviations of the closes around the centre over this
    band_std: float = 2.0         # many sessions
    atr_window: int = 14
    vol_window: int = 20          # realised-volatility window for the regime/beacons (trading days)
    sizing_vol_window: int = 60   # realised-volatility window used for position sizing (smoother)
    rebalance_threshold: float = 0.10   # re-size an open position on review days when the target moves by more than this
    # --- volume ----------------------------------------------------------------
    volume_window: int = 20
    volume_baseline: int = 60
    # --- breakout --------------------------------------------------------------
    high_window: int = 252        # 52-week high
    # --- trading rules (used by the backtest) ----------------------------------
    # Defaults below are the LONG-horizon rule; StrategyParams.for_horizon("short") switches to the faster rule.
    entry_score: float = 60.0     # technical score needed to open a position
    exit_score: float = 35.0      # technical score at which an open position is closed
    trend_gate: bool = True       # only enter when the trend beacon is positive (price & 50-SMA above the 200-SMA)
    exit_below_sma: Optional[float] = 0.08  # exit when close < 200-SMA x (1 - buffer); None = off
    atr_stop: bool = False        # trailing ATR stop (short-horizon rule); long horizon uses the 200-SMA exit as its stop
    decision_frequency: str = "monthly"   # "daily" | "weekly" | "monthly": how often the entry/exit rule is evaluated
    score_smoothing: int = 1      # days of averaging applied to the technical score before testing thresholds
    # tested and rejected in the design phase (kept so the sensitivity tools can show why):
    trend_exit: Optional[float] = None          # exit when the trend beacon falls below this (None = off)
    max_entry_distance: Optional[float] = None  # "don't chase": skip entries when close is this far above the 200-SMA
    bear_entry_penalty: float = 10.0        # extra points required to enter in a bear market
    bear_size_multiplier: float = 0.5       # position scaling in a bear market
    block_entries_in_bear_turbulent: bool = True
    fundamental_gate_min: float = 35.0      # point-in-time fundamental score below this blocks new entries
    reporting_lag_days: int = 90            # financial statements become usable this long after period end
    # --- regime ------------------------------------------------------------------
    regime_sma: int = 200
    regime_vol_window: int = 21
    regime_vol_lookback: int = 252
    regime_vol_multiple: float = 1.25       # vol above 1.25x its 1-year median = turbulent
    bear_score_penalty: float = 8.0         # composite-score points removed in a bear market ("headwind")
    turbulent_score_penalty: float = 3.0    # composite-score points removed when volatility is elevated
    # --- costs & evaluation ------------------------------------------------------
    cost_bps: float = 5.0          # commission per side, basis points
    slippage_bps: float = 5.0      # slippage per side, basis points
    risk_free_rate: float = 0.02   # annual, for Sharpe/Sortino
    trading_days: int = 252

    @classmethod
    def for_horizon(cls, horizon: str = "long", **overrides) -> "StrategyParams":
        """Rule presets: long-term investors review monthly with wide buffers; short-term weekly with an ATR stop."""
        base = {}
        if str(horizon).lower() == "short":
            base = dict(exit_score=40.0, exit_below_sma=0.03, atr_stop=True, decision_frequency="weekly")
        base.update(overrides)
        return cls(**base)

    def copy(self, **overrides) -> "StrategyParams":
        d = asdict(self)
        d.update(overrides)
        return StrategyParams(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def min_history(self) -> int:
        """Rows needed before every indicator is defined."""
        return max(self.sma_slow, self.high_window, self.regime_vol_lookback) + 5


def verdict_from_score(score: float) -> str:
    for threshold, label in VERDICT_BANDS:
        if score >= threshold:
            return label
    return VERDICT_BANDS[-1][1]
