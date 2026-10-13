"""
Decision engine: blend the Fundamental and Technical scores for the investor's
horizon, apply the market-regime headwind, map the result to a verdict, and
turn the verdict into a concrete position plan (size, shares, stop, max loss).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import InvestorProfile, StrategyParams, verdict_from_score
from .regime import RegimeSnapshot, regime_adjustments
from .technicals import Beacon

ACTION_TEXT = {
    "Strong Buy": "Open or add to a position - the evidence lines up across fundamentals, technicals and the market backdrop.",
    "Buy":        "Open a position at the suggested size; the balance of evidence is positive.",
    "Hold":       "Keep an existing position but do not add new money; the evidence is mixed.",
    "Reduce":     "Trim an existing position (about half) and do not buy; the evidence leans negative.",
    "Sell":       "Exit the position; the evidence is clearly negative.",
}


@dataclass
class PositionPlan:
    verdict: str
    position_pct: float          # % of the total account to put in this stock (after the cash cap)
    position_value: float        # $ target for the position (<= cash available)
    shares: int                  # whole shares that fit in position_value at entry_price
    entry_price: float
    stop_price: Optional[float]
    stop_pct: Optional[float]    # distance to stop, as a fraction of price
    max_loss_value: float        # $ lost on `shares` if the stop is hit at the stop price (gaps can make it worse)
    conviction: float            # 0-1 scaling from the composite score
    binding_constraint: str      # which rule limited the size
    rationale: List[str] = field(default_factory=list)
    side: str = "long"           # "long" (buy) or "short" (sell short)
    cash_available: float = 0.0  # spendable cash the plan was sized against

    @property
    def action(self) -> str:
        return ACTION_TEXT.get(self.verdict, "")

    @property
    def cost(self) -> float:
        """Cash needed for the whole shares (a short ties up this much as collateral instead)."""
        return float(self.shares * self.entry_price)

    @property
    def cash_left(self) -> float:
        return float(max(self.cash_available - self.cost, 0.0))


@dataclass
class Decision:
    ticker: str
    as_of: str
    price: float
    profile: InvestorProfile
    fundamental_score: Optional[float]
    technical_score: float
    regime: Optional[RegimeSnapshot]
    regime_penalty: float
    composite_score: float
    verdict: str
    beacons: List[Beacon]
    plan: PositionPlan
    warnings: List[str] = field(default_factory=list)

    @property
    def fundamental_beacons(self) -> List[Beacon]:
        return [b for b in self.beacons if b.group == "fundamental"]

    @property
    def technical_beacons(self) -> List[Beacon]:
        return [b for b in self.beacons if b.group == "technical"]

    def headline(self) -> str:
        return (f"{self.ticker}: {self.verdict.upper()} (Lighthouse score {self.composite_score:.0f}/100) "
                f"for a {self.profile.horizon}-term, {self.profile.risk} investor")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def composite_score(fundamental: Optional[float], technical: float, profile: InvestorProfile,
                    regime_penalty: float = 0.0) -> float:
    """Horizon-weighted blend, minus the regime headwind. Missing fundamentals -> technical only."""
    if fundamental is None or (isinstance(fundamental, float) and np.isnan(fundamental)):
        blended = technical
    else:
        blended = profile.fundamental_weight * fundamental + profile.technical_weight * technical
    return float(np.clip(blended - regime_penalty, 0.0, 100.0))


def conviction_from_score(score: float) -> float:
    """0 at 45 (bottom of Hold), 1 at 80+. Scales the position size with the strength of the evidence."""
    return float(np.clip((score - 45.0) / 35.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Position sizing
# --------------------------------------------------------------------------- #
def stop_level(price: float, atr: float, sma_slow: Optional[float], profile: InvestorProfile,
               params: Optional[StrategyParams] = None, side: str = "long") -> tuple:
    """Where the position is abandoned, and why. Returns (level, label, kind).

    Long horizon : the 200-day SMA minus the exit buffer (the same rule the backtest uses);
                   for a short, the 200-day SMA plus the buffer.
    Short horizon: an ATR stop (risk-preset multiple x ATR) below a long / above a short.
    """
    p = params or StrategyParams()
    short = side == "short"
    atr_ok = atr is not None and np.isfinite(atr) and atr > 0 and price > 0
    if not p.atr_stop and p.exit_below_sma is not None and sma_slow is not None and np.isfinite(sma_slow):
        if short:
            level = sma_slow * (1.0 + p.exit_below_sma)
            if level > price:
                return float(level), f"200-day SMA plus {p.exit_below_sma * 100:.0f}% buffer", "sma"
        else:
            level = sma_slow * (1.0 - p.exit_below_sma)
            if level < price:
                return float(level), f"200-day SMA minus {p.exit_below_sma * 100:.0f}% buffer", "sma"
    if atr_ok:
        dist = min(max(profile.stop_atr_multiple * atr / price, 0.02), 0.5)
        level = price * (1.0 + dist) if short else price * (1.0 - dist)
        return float(level), f"{profile.stop_atr_multiple:g} x ATR {'stop above the entry' if short else 'trailing stop'}", "atr"
    return None, "no stop (volatility unknown)", "none"


def risk_sizing(price: float, stop_pct: Optional[float], realized_vol: float, profile: InvestorProfile,
                rationale: Optional[List[str]] = None) -> tuple:
    """The three independent risk rules, as a % of the TOTAL account. Returns (base_pct, binding_rule).

    1. Max-loss rule   : if the stop is hit, lose at most `max_loss_pct` of the account.
    2. Volatility rule : scale the position so its expected volatility matches the profile's target.
    3. Concentration   : never exceed the profile's maximum single-position size.
    The smallest wins. Works for longs and shorts (the stop distance is a positive fraction either way).
    """
    rationale = rationale if rationale is not None else []
    candidates = {"concentration cap": profile.max_position_pct}
    if stop_pct:
        candidates["max-loss rule"] = profile.max_loss_pct / stop_pct
        rationale.append(f"Max-loss rule: losing {profile.max_loss_pct:g}% of the account (${profile.max_loss_value:,.0f}) at the stop "
                         f"caps the position at {candidates['max-loss rule']:.1f}% of the account.")
    if realized_vol and np.isfinite(realized_vol) and realized_vol > 0:
        rv = max(realized_vol, 0.05)
        candidates["volatility target"] = 100.0 * profile.target_vol / rv
        rationale.append(f"Volatility rule: {rv * 100:.0f}% realised vol vs a {profile.target_vol * 100:.0f}% target caps the position at "
                         f"{candidates['volatility target']:.1f}% of the account.")
    binding = min(candidates, key=candidates.get)
    return float(candidates[binding]), binding


def dollars_to_shares(pct: float, price: float, profile: InvestorProfile, binding: str,
                      rationale: List[str]) -> tuple:
    """Turn a % of the account into dollars and whole shares, capped by the cash available.
    Returns (pct_of_account, value, shares, binding)."""
    value = profile.account_value * pct / 100.0
    if value > profile.cash + 1e-9:
        if value > 0:
            rationale.append(f"Cash cap: the rules allow ${value:,.0f}, but only ${profile.cash:,.0f} of cash is available to invest.")
        value = profile.cash
        binding = "cash available"
    shares = int(math.floor(value / price)) if price > 0 else 0
    if value > 0 and shares == 0:
        rationale.append(f"${value:,.0f} is less than the price of one share (${price:,.2f}); consider fractional shares if your broker offers them.")
    pct_out = 100.0 * value / profile.account_value if profile.account_value > 0 else 0.0
    return pct_out, value, shares, binding


def size_position(price: float, atr: float, realized_vol: float, score: float, verdict: str,
                  profile: InvestorProfile, regime_size_multiplier: float = 1.0, sma_slow: Optional[float] = None,
                  params: Optional[StrategyParams] = None) -> PositionPlan:
    """Size a NEW long position from the cash available.

    The size is the smallest of the max-loss, volatility and concentration rules (as % of the total account),
    scaled by conviction (score strength) and the regime multiplier, then capped by the cash you can invest.
    """
    rationale: List[str] = []
    stop_price, stop_label, _ = stop_level(price, atr, sma_slow, profile, params)
    stop_pct: Optional[float] = None
    if stop_price is not None and price > 0:
        # floor of 1% only (a wider assumed distance gives a smaller, safer size); no upper clamp, because
        # understating a wide stop's distance would let the real loss at the stop exceed the investor's limit
        stop_pct = max(1.0 - stop_price / price, 0.01)
        rationale.append(f"Exit level {stop_pct * 100:.1f}% below the current price at ${stop_price:,.2f} ({stop_label}).")

    base_pct, binding = risk_sizing(price, stop_pct, realized_vol, profile, rationale)

    conviction = conviction_from_score(score)
    if verdict in {"Strong Buy", "Buy"}:
        pct = base_pct * conviction * regime_size_multiplier
        rationale.append(f"Scaled by conviction {conviction:.2f} (score {score:.0f}) and market-regime multiplier {regime_size_multiplier:g}.")
    elif verdict == "Hold":
        pct = 0.0
        rationale.append("Hold: keep what you own; no new capital.")
    elif verdict == "Reduce":
        pct = 0.0
        rationale.append("Reduce: trim roughly half of an existing position; no new capital.")
    else:
        pct = 0.0
        rationale.append("Sell: exit the position.")
    pct = float(min(pct, profile.max_position_pct))
    pct, value, shares, binding = dollars_to_shares(pct, price, profile, binding, rationale)
    max_loss_value = shares * price * (stop_pct or 0.0)
    return PositionPlan(verdict=verdict, position_pct=pct, position_value=value, shares=shares, entry_price=price,
                        stop_price=stop_price, stop_pct=stop_pct, max_loss_value=max_loss_value, conviction=conviction,
                        binding_constraint=binding, rationale=rationale, side="long", cash_available=profile.cash)


def backtest_size_fraction(atr: float, realized_vol: float, score: float, profile: InvestorProfile,
                           regime_size_multiplier: float = 1.0, mode: str = "vol_target") -> float:
    """Fraction of the *sleeve* (capital earmarked for this stock) to invest when a signal fires.

    mode="full"       : always 100% (shows raw signal quality)
    mode="vol_target" : volatility targeting x regime multiplier (the sleeve-level rules; conviction and
                        the portfolio caps are applied on top in the live position plan)
    """
    if mode == "full":
        return 1.0
    frac = 1.0
    if realized_vol and np.isfinite(realized_vol) and realized_vol > 0:
        frac = min(1.0, profile.target_vol / max(realized_vol, 0.05))
    return float(np.clip(frac * regime_size_multiplier, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Assemble the decision
# --------------------------------------------------------------------------- #
def make_decision(ticker: str, as_of, price: float, atr: float, realized_vol: float,
                  fundamental: Optional[float], technical: float, beacons: List[Beacon],
                  regime: Optional[RegimeSnapshot], profile: InvestorProfile,
                  params: Optional[StrategyParams] = None, warnings: Optional[List[str]] = None,
                  sma_slow: Optional[float] = None) -> Decision:
    p = params or StrategyParams()
    adj = regime_adjustments(regime.is_bear if regime else False, regime.is_turbulent if regime else False, p)
    score = composite_score(fundamental, technical, profile, adj["score_penalty"])
    verdict = verdict_from_score(score)
    warns = list(warnings or [])
    if fundamental is None:
        warns.append("Fundamental data unavailable: the score is technical-only and should be treated with extra caution.")
    if adj["block_entries"] and verdict in {"Buy", "Strong Buy"}:
        verdict = "Hold"
        warns.append("Market is Bear / Turbulent: new entries are blocked by the regime rule, so the verdict is capped at Hold.")
    plan = size_position(price, atr, realized_vol, score, verdict, profile, adj["size_multiplier"], sma_slow, p)
    return Decision(ticker=ticker, as_of=str(as_of)[:10], price=float(price), profile=profile,
                    fundamental_score=fundamental, technical_score=float(technical), regime=regime,
                    regime_penalty=adj["score_penalty"], composite_score=score, verdict=verdict,
                    beacons=beacons, plan=plan, warnings=warns)
