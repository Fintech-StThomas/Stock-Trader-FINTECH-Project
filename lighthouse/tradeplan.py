"""
Trade plans: what to buy (or sell short) right now with the cash you have, where you are wrong (stop-loss),
where to take profit, and how to manage the trade once it is on.

The plan reuses the verdict, the stop rules and the three risk rules of the decision engine, so a long ticket
here and the Position plan in the Analyse tab agree. Shorts mirror the long rules (see `SHORT_MAX_SCORE`).

Trade-management rules (the same rules drive the ticket, the historical setup test and the position monitor):
  1. Initial stop  : the horizon's stop rule (long horizon: 200-day SMA minus a buffer; short horizon: k x ATR).
                     Shorts always use the short-horizon (tactical) rules - an ATR stop and the short time stop -
                     because a short's loss is open-ended and it pays borrow fees for as long as it is held.
  2. Target 1      : entry +/- m1 x R (R = distance from entry to the initial stop). Close `partial_exit` of the
                     position there and move the stop on the rest to breakeven (the entry price).
  3. Trailing stop : after target 1, trail the stop at the best close since entry -/+ k x ATR (a chandelier exit);
                     it only ever tightens.
  4. Target 2      : entry +/- m2 x R closes the rest.
  5. Time stop     : if the trade is still open after `time_stop_days` sessions, close it at that session's close.

Long-horizon longs follow a trend plan by default instead (TradeSettings.long_exit = "trend"): the same initial stop,
placed as an order and left there, and no fixed profit target. At every review (the last session of each month) the
position is sold at the next open if the close is below the 200-day SMA minus the buffer, a level that rises with the
average and so locks in gains, or if the technical score has fallen to the exit score. These are the long-horizon
backtest's own exits; on the sample pack they earned several times more per trade than R-multiple targets, which sat
30-80% away (docs/METHODOLOGY.md, section 9).

Nothing here is a guarantee: a stop is an order, not a price, and an overnight gap can fill it worse.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import InvestorProfile, StrategyParams
from .decision import conviction_from_score, dollars_to_shares, risk_sizing, stop_level
from .regime import regime_adjustments
from .technicals import beacon_scores

SIDES = ("long", "short")
# Shorts mirror the long rule around 50: a long needs a composite >= 60 (Buy); a short needs <= 40.
SHORT_MAX_SCORE = 40.0
# Default time stops (sessions) per horizon: short = weeks to a few months, long = about a year.
TIME_STOP_DEFAULT = {"short": 40, "long": 252}
# Long-horizon longs: "trend" = the backtest's review-day trend and signal exits, no fixed target; "targets" = R targets.
LONG_EXITS = ("trend", "targets")
REG_T_INITIAL_SHORT = 0.50        # Regulation T: 150% of the short value = the sale proceeds + 50% deposited
FINRA_SHORT_MAINT_PCT = 0.30      # FINRA Rule 4210(c): short stock >= $5: greater of $5/share or 30% of market value
FINRA_SHORT_MAINT_PER_SHARE = 5.0
FINRA_LOW_PRICE_MAINT_PER_SHARE = 2.5   # short stock < $5: greater of $2.50/share or 100% of market value


class StaleDataError(RuntimeError):
    """A trade ticket was asked for on data that is not current (bundled sample data, a stale cache, a past date, or
    a series missing finished sessions). Pass allow_stale=True (CLI --allow-stale-data) to override, for testing only."""


ALLOW_STALE_ENV = "LIGHTHOUSE_ALLOW_STALE_DATA"
LIVE_SOURCES = {"yfinance", "stooq", "cache"}       # a fresh cache holds the last finished session (it expires at each close)


def allow_stale_default() -> bool:
    """The override from the environment (LIGHTHOUSE_ALLOW_STALE_DATA=1); off unless set."""
    return os.environ.get(ALLOW_STALE_ENV, "").strip().lower() in {"1", "true", "yes"}


def data_freshness(result, now: Optional[pd.Timestamp] = None) -> Tuple[bool, str]:
    """(current?, why). Current means: prices from a live feed or a fresh cache, not a "what would it have said" date,
    and a last bar that is the last finished trading session (NYSE calendar)."""
    from .data import sessions_behind
    attrs = getattr(result.prices, "attrs", {}) or {}
    src = attrs.get("source")
    last = result.prices.index[-1]
    if src == "sample":
        return False, f"these are the bundled sample prices (they end {last.date()})"
    if src == "stale-cache":
        return False, f"these prices come from an old cache copy (they end {last.date()}): the live feed could not be reached"
    if src not in LIVE_SOURCES:
        return False, f"the prices' source ({src or 'unknown'}) is not a live feed"
    behind = sessions_behind(last, now)
    if behind > 0:
        return False, (f"the prices end {last.date()}, {behind} finished trading session{'s' if behind > 1 else ''} ago "
                       "(a past date or an outdated download)")
    return True, f"current: {src} prices through {last.date()}, the last finished session"


def require_current_data(result, allow_stale: Optional[bool] = None, now: Optional[pd.Timestamp] = None) -> str:
    """Raise StaleDataError unless the data is current or the override is given. Returns the freshness note."""
    ok, why = data_freshness(result, now)
    allowed = allow_stale_default() if allow_stale is None else bool(allow_stale)
    if not ok and not allowed:
        raise StaleDataError(f"{result.ticker}: no trade ticket, because {why}. Tickets are orders: they need current "
                             "prices. Refresh with a live connection, or override for testing only (CLI --allow-stale-data, "
                             f"app 'allow stale data', or {ALLOW_STALE_ENV}=1).")
    return why if ok else f"OVERRIDE: {why}"


@dataclass
class TradeSettings:
    """How the trade is managed. Defaults are conventional, not optimised."""
    reward_multiples: Tuple[float, float] = (2.0, 3.0)   # target 1 and target 2, in R
    partial_exit: float = 0.5                            # share of the position closed at target 1
    time_stop_days: Optional[int] = None                 # None = horizon default (TIME_STOP_DEFAULT)
    alert_atr: float = 0.5                               # "near stop" warning when price is within this many ATR
    chase_atr: float = 0.5                               # do not chase: skip if price runs this many ATR past the entry
    max_ideas: int = 5                                   # trade ideas funded from the cash at once
    max_total_risk_pct: float = 6.0                      # "portfolio heat": all open risk across ideas, % of account
    min_fill_ratio: float = 0.25                         # fund an idea only if the cash left buys >= 25% of its full size
    long_exit: str = "trend"                             # long-horizon longs: "trend" (monthly trend/signal exits) | "targets"

    def __post_init__(self) -> None:
        m1, m2 = self.reward_multiples
        if not (0 < m1 <= m2):
            raise ValueError("reward_multiples must be positive and ascending, e.g. (2, 3)")
        if not (0.0 <= self.partial_exit <= 1.0):
            raise ValueError("partial_exit must be between 0 and 1")
        if self.time_stop_days is not None and self.time_stop_days < 1:
            raise ValueError("time_stop_days must be at least 1")
        if self.max_ideas < 1 or self.max_total_risk_pct <= 0:
            raise ValueError("max_ideas must be >= 1 and max_total_risk_pct > 0")
        if not (0.0 <= self.min_fill_ratio <= 1.0):
            raise ValueError("min_fill_ratio must be between 0 and 1")
        if self.long_exit not in LONG_EXITS:
            raise ValueError(f"long_exit must be one of {LONG_EXITS}")

    def time_stop_for(self, horizon: str) -> int:
        return int(self.time_stop_days or TIME_STOP_DEFAULT.get(horizon, 40))


def trend_plan_applies(side: str, profile: InvestorProfile, params: StrategyParams, settings: TradeSettings) -> bool:
    """True when a trade follows the trend plan: a long, on the long horizon, with the 200-day SMA exit switched on."""
    return (side == "long" and profile.horizon == "long" and settings.long_exit == "trend" and not params.atr_stop
            and params.exit_below_sma is not None)


REVIEW_PERIOD = {"daily": "session", "weekly": "week", "monthly": "month"}


def review_days(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """True on the sessions a rule is reviewed: the last session of each week or month (every session for "daily").
    The last row counts only when the next business day starts a new period, so today is not mistaken for a month end."""
    n = len(index)
    if frequency == "daily" or n == 0:
        return np.ones(n, dtype=bool)
    rule = "W" if frequency == "weekly" else "M"
    period = index.to_period(rule).asi8
    out = np.zeros(n, dtype=bool)
    out[:-1] = period[:-1] != period[1:]
    nxt = (index[-1] + pd.offsets.BDay(1)).to_period(rule).ordinal
    out[-1] = nxt != period[-1]
    return out


@dataclass
class Target:
    r_multiple: float
    price: float
    move_pct: float          # signed % move from the entry (negative for a short's targets)
    shares: int              # shares closed at this level
    profit: float            # $ profit on those shares if the level fills


@dataclass
class SetupStats:
    """How these exact trade rules played out on this stock's own history (non-overlapping trades)."""
    n_trades: int
    win_rate: float
    avg_r: float             # expectancy in R per trade, after costs
    median_r: float
    profit_factor: float     # sum of winning R / sum of losing R
    hit_target1: float       # share of trades that reached target 1
    stopped_out: float       # share of trades stopped out before target 1
    timed_out: float         # share closed by the time stop before target 1 (the three shares sum to 1; for a trend
                             # plan, stopped_out + timed_out + trend_exited sum to 1)
    avg_days: float
    avg_return_pct: float    # average % return of the position per trade (sign-adjusted for shorts)
    years: float
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    exit_style: str = "targets"     # "targets" | "trend"
    trend_exited: float = 0.0       # trend plans: share of trades ended by the review-day trend or signal exit

    @property
    def reliable(self) -> bool:
        return self.n_trades >= 20


@dataclass
class TradePlan:
    ticker: str
    name: str
    side: str                       # "long" | "short"
    status: str                     # "actionable" | "watch" | "avoid"
    reasons: List[str]
    as_of: str                      # last finished session the signals use
    price: float                    # reference entry price: the live quote when there is one, else the last close
    entry_limit: float              # do-not-chase price (long: max price to pay; short: min price to sell at)
    stop: Optional[float]
    stop_label: str
    stop_pct: Optional[float]       # distance from entry to stop, fraction of price
    risk_per_share: float
    shares: int
    notional: float                 # shares x price (cash spent for a long, collateral for a short)
    dollars_at_risk: float          # shares x risk per share (gaps can make it worse)
    risk_pct_of_account: float
    targets: List[Target]
    trail_atr_multiple: float
    time_stop_days: Optional[int]   # None: no time stop (the trend plan's default)
    typical_move_pct: float         # 1-sigma move over `typical_days` sessions, from realised volatility
    atr: float
    composite_score: float
    technical_score: float
    fundamental_score: Optional[float]
    verdict: str
    regime: str
    conviction: float
    size_binding: str
    cash_available: float
    account_value: float
    sizing_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    margin: Dict[str, float] = field(default_factory=dict)
    stats: Optional[SetupStats] = None
    priced_from: str = ""           # where `price` comes from (live quote with its time, or the close)
    horizon: str = "long"           # the rules the trade follows ("short" for every short sale)
    partial_exit: float = 0.5       # share of the position sold at target 1 (TradeSettings.partial_exit)
    exit_style: str = "targets"     # "targets" | "trend" (long-horizon longs: review-day trend and signal exits)
    trend_exit_level: Optional[float] = None    # trend plans: a review-day close below this ends the trade (today's value)
    trend_exit_label: str = ""                  # e.g. "200-day SMA minus 8%"
    trend_exit_pace_pct: Optional[float] = None  # how fast that level rose over the last 21 sessions, % per month
    signal_exit_score: Optional[float] = None   # trend plans: a review-day technical score at or below this ends the trade
    review: str = "month"                       # the review period of a trend plan: "month" | "week" | "session"
    typical_days: int = 252                     # window of `typical_move_pct`
    data_status: str = ""                       # data_freshness note: "current: ..." or "OVERRIDE: ..."

    @property
    def is_actionable(self) -> bool:
        return self.status == "actionable" and self.shares > 0

    @property
    def is_trend_plan(self) -> bool:
        return self.exit_style == "trend"

    @property
    def reward_risk(self) -> float:
        return self.targets[0].r_multiple if self.targets else float("nan")

    def _trend_rule_text(self) -> str:
        """The review-day exits of a trend plan, in one sentence."""
        pace = (f"; it follows the average, {self.trend_exit_pace_pct:+.1f}% over the last month"
                if self.trend_exit_pace_pct is not None and np.isfinite(self.trend_exit_pace_pct) else "")
        lvl = f"${self.trend_exit_level:,.2f}" if self.trend_exit_level is not None else "n/a"
        return (f"the close is below the {self.trend_exit_label} (today {lvl}{pace}) or the technical score is "
                f"{self.signal_exit_score:g} or lower (today {self.technical_score:.0f})")

    @property
    def profit_if_all_targets(self) -> float:
        return float(sum(t.profit for t in self.targets))

    @property
    def cash_left(self) -> float:
        return float(max(self.cash_available - self.notional, 0.0))

    def management_steps(self) -> List[str]:
        """The plan in plain English, in the order the investor acts on it."""
        verb, unverb = ("Buy", "sell") if self.side == "long" else ("Sell short", "buy back")
        best = "highest" if self.side == "long" else "lowest"
        sign = "−" if self.side == "long" else "+"
        if self.shares <= 0:
            levels = [f"stop ${self.stop:,.2f}"] if self.stop is not None else []
            levels += [f"target {k + 1} ${t.price:,.2f}" for k, t in enumerate(self.targets)]
            if self.is_trend_plan and self.trend_exit_level is not None:
                levels.append(f"trend exit ${self.trend_exit_level:,.2f} (review-day close)")
            return ["No position: " + ("the setup is not actionable" if self.status == "avoid" else "the cash available does not "
                                       "buy a whole share") + (". Reference levels: " + ", ".join(levels) + "." if levels else ".")]
        steps = []
        if self.shares > 0:
            chase = "above" if self.side == "long" else "below"
            steps.append(f"{verb} {self.shares} shares of {self.ticker} with a limit order near ${self.price:,.2f}; "
                         f"skip the trade if it is already trading {chase} ${self.entry_limit:,.2f} (do not chase).")
        if self.stop is not None:
            steps.append(f"Immediately place a stop order at ${self.stop:,.2f} ({self.stop_label}): if hit, {unverb} everything. "
                         f"Planned loss ≈ ${self.dollars_at_risk:,.0f} ({self.risk_pct_of_account:.2f}% of the account).")
        if self.is_trend_plan:
            steps.append(f"No fixed profit target: hold while the trend lasts. At every {self.review}'s last close, review the "
                         f"trade and {unverb.lower()} everything at the next open if {self._trend_rule_text()}. This is how "
                         "the profit is taken: the exit level climbs behind a rising trend.")
            steps.append("Leave the stop order in place for the whole trade: it caps the loss if the price falls between reviews.")
            if self.time_stop_days:
                steps.append(f"Time stop: if the trade is still open after {self.time_stop_days} trading sessions, close it.")
            return steps
        if self.targets:
            t1 = self.targets[0]
            t2 = self.targets[1] if len(self.targets) > 1 else None
            trail = (f"Then trail the stop at the {best} close since entry {sign} {self.trail_atr_multiple:g} × ATR "
                     f"(≈ ${self.trail_atr_multiple * self.atr:,.2f} today); never loosen it.")
            if t2 is not None and t1.shares and t2.shares:
                steps.append(f"Target 1 at ${t1.price:,.2f} ({t1.r_multiple:g}R, {t1.move_pct:+.1f}%): {unverb} {t1.shares} shares "
                             f"(+${t1.profit:,.0f}) and move the stop on the rest to breakeven (${self.price:,.2f}).")
                steps.append(trail)
                steps.append(f"Target 2 at ${t2.price:,.2f} ({t2.r_multiple:g}R, {t2.move_pct:+.1f}%): {unverb} the remaining "
                             f"{t2.shares} shares (+${t2.profit:,.0f}).")
            elif t2 is not None and t2.shares and not t1.shares:
                steps.append(f"At target 1 (${t1.price:,.2f}, {t1.r_multiple:g}R) sell nothing yet: move the stop to breakeven "
                             f"(${self.price:,.2f}).")
                steps.append(trail)
                steps.append(f"Target 2 at ${t2.price:,.2f} ({t2.r_multiple:g}R, {t2.move_pct:+.1f}%): {unverb} all {t2.shares} "
                             f"shares (+${t2.profit:,.0f}).")
            else:
                steps.append(f"Take profit at ${t1.price:,.2f} ({t1.r_multiple:g}R, {t1.move_pct:+.1f}%): {unverb} all {t1.shares} "
                             f"shares (+${t1.profit:,.0f}).")
        steps.append(f"Time stop: if the trade is still open after {self.time_stop_days} trading sessions, close it.")
        return steps

    def exit_plan(self) -> List["ExitRule"]:
        """When to get out, for a loss or a profit: every exit the plan has, its trigger price, what it does to the
        whole position's P&L if the trade ends that way, and how often it happened on this stock's history."""
        short = self.side == "short"
        unverb = "Buy back" if short else "Sell"
        t1 = self.targets[0] if self.targets else None
        t2 = self.targets[1] if len(self.targets) > 1 else None
        if self.shares > 0:
            q1, q2, total = (t1.shares if t1 else 0), (t2.shares if t2 else 0), float(self.shares)
        else:   # no position: express the plan per unit of position (fractions instead of shares)
            f1 = self.partial_exit if t2 is not None and self.partial_exit < 1.0 else 1.0
            q1, q2, total = f1, 1.0 - f1, 1.0
        risk = self.risk_per_share
        money = self.shares > 0

        def pnl(units_r: float) -> Optional[float]:
            return float(units_r * risk) if money else None

        hist = _exit_history(self.stats)
        rules: List[ExitRule] = []
        if self.is_trend_plan:
            return self._trend_exit_plan(unverb, total, pnl, hist)
        if self.stop is not None:
            rules.append(ExitRule(
                "loss", "Stop-loss (before target 1)", self.stop, (self.stop / self.price - 1.0) * 100.0, self.shares or None,
                pnl(-total), -1.0, f"{unverb} everything at the stop order ({self.stop_label}). The trade was wrong; take the planned loss.",
                hist.get("stop", "")))
        if t1 is not None and t2 is not None and q2 and not q1:
            # nothing is sold at target 1 (a 0% partial): it only moves the stop to breakeven
            rules.append(ExitRule(
                "profit", "Target 1: stop to breakeven", t1.price, t1.move_pct, None, pnl(0.0), 0.0,
                f"Sell nothing; move the stop to breakeven (${self.price:,.2f}). If the position is then stopped there, the "
                "result is zero.", hist.get("target 1", "")))
        if t1 is not None and q1:
            locked = q1 * t1.r_multiple
            if t2 is not None and q2:
                what = f"{t1.shares} shares" if t1.shares else f"{q1 / total:.0%} of the position"
                rules.append(ExitRule(
                    "profit", "Target 1: partial profit", t1.price, t1.move_pct, t1.shares or None, pnl(locked), locked / total,
                    f"{unverb} {what} and move the stop on the rest to breakeven (${self.price:,.2f}). If the rest is then "
                    "stopped at breakeven, this is the result.", hist.get("target 1", "")))
            else:
                rules.append(ExitRule(
                    "profit", "Target 1: take profit", t1.price, t1.move_pct, t1.shares or None, pnl(locked), locked / total,
                    f"{unverb} the whole position.", hist.get("target 1", "")))
        if t2 is not None and q2:
            best = q1 * (t1.r_multiple if t1 else 0.0) + q2 * t2.r_multiple
            trail_dir = "−" if not short else "+"
            rules.append(ExitRule(
                "profit", "Trailing stop (after target 1)", None, None, t2.shares or None, None, None,
                f"Trail the stop on the rest at the {'lowest' if short else 'highest'} close since entry {trail_dir} "
                f"{self.trail_atr_multiple:g} × ATR (≈ ${self.trail_atr_multiple * self.atr:,.2f} today), never loosening it. "
                f"It locks in more of the run; the result lands between target 1's and target 2's.",
                hist.get("trailing", "")))
            rules.append(ExitRule(
                "profit", "Target 2: the rest", t2.price, t2.move_pct, t2.shares or None, pnl(best), best / total,
                f"{unverb} the remaining {t2.shares or ''} shares. The best planned outcome.".replace("  ", " "),
                hist.get("target 2", "")))
        rules.append(ExitRule(
            "time", f"Time stop: {self.time_stop_days} sessions", None, None, None, None, None,
            f"{unverb} at the market if neither the stop nor target 1 has been reached by then: the idea has not worked and "
            "the cash is better used elsewhere. The result lies between the stop's loss and target 1's profit.",
            hist.get("time stop", "")))
        rules.append(ExitRule(
            "evidence", "Verdict turns", None, None, None, None, None,
            (f"Re-run Lighthouse every {'week' if self.horizon == 'short' else 'month'} (the review cycle these rules were "
             "tested with). If the verdict turns " + ("Buy or better while you are short, buy back early." if short else
                                                      "Reduce or Sell while you are long, sell early.")),
            ""))
        return rules

    def _trend_exit_plan(self, unverb: str, total: float, pnl, hist: Dict[str, str]) -> List["ExitRule"]:
        """Exits of a trend plan: the stop order, then the review-day trend and signal exits (and a time stop if set)."""
        rules: List[ExitRule] = []
        when = f"At the last close of each {self.review}"
        if self.stop is not None:
            rules.append(ExitRule(
                "loss", "Stop-loss (stop order)", self.stop, (self.stop / self.price - 1.0) * 100.0, self.shares or None,
                pnl(-total), -1.0,
                f"{unverb} everything at the stop order ({self.stop_label}). Placed at entry and never moved: it caps the "
                "loss if the price falls between reviews.", hist.get("stop", "")))
        if self.trend_exit_level is not None:
            notes = []
            if self.stop is not None and abs(self.trend_exit_level - self.stop) < 0.005 * self.price:
                notes.append("today it sits at the stop")
            if self.trend_exit_pace_pct is not None and np.isfinite(self.trend_exit_pace_pct):
                notes.append(f"it moved {self.trend_exit_pace_pct:+.1f}% over the last month")
            pace = f" ({'; '.join(notes).capitalize()}.)" if notes else ""
            rules.append(ExitRule(
                "profit", "Trend exit (review)", self.trend_exit_level, (self.trend_exit_level / self.price - 1.0) * 100.0,
                self.shares or None, None, None,
                f"{when}, if the close is below the {self.trend_exit_label}, {unverb.lower()} everything at the next "
                f"open. The level rises with the average, so it locks in more of the gain the longer the trend runs: "
                f"this is how the profit is taken.{pace}", hist.get("trend exit", "")))
        if self.signal_exit_score is not None:
            rules.append(ExitRule(
                "evidence", "Signal exit (review)", None, None, self.shares or None, None, None,
                f"{when}, if the technical score is {self.signal_exit_score:g} or lower (today {self.technical_score:.0f}), "
                f"{unverb.lower()} everything at the next open: the momentum behind the trade has gone.",
                hist.get("signal exit", "")))
        if self.time_stop_days:
            rules.append(ExitRule(
                "time", f"Time stop: {self.time_stop_days} sessions", None, None, None, None, None,
                f"{unverb} at the market if the trade is still open after {self.time_stop_days} sessions.",
                hist.get("time stop", "")))
        return rules


@dataclass
class ExitRule:
    """One way out of a trade."""
    kind: str                     # "loss" | "profit" | "time" | "evidence"
    name: str
    price: Optional[float]        # trigger price (None: no fixed level)
    move_pct: Optional[float]     # signed % from the entry
    shares: Optional[int]         # shares closed by this exit
    pnl: Optional[float]          # $ result of the WHOLE position if the trade ends this way (None: not fixed in advance)
    r: Optional[float]            # the same in R
    action: str
    history: str                  # how often (and how fast) it happened on this stock's history


def _exit_history(stats: Optional["SetupStats"]) -> Dict[str, str]:
    """Plain-English frequency and median duration of each exit in the historical setup test."""
    if stats is None or stats.trades is None or not len(stats.trades):
        return {}
    tr = stats.trades
    n = len(tr)
    parts = tr["outcome"].str.split(" + ", regex=False)
    first, last = parts.str[0], parts.str[-1]

    def line(mask, label: Optional[str], with_r: bool = False) -> str:
        k = int(mask.sum())
        if k == 0:
            return f"never in {n} past setups"
        when = f", median {int(tr.loc[mask, 'days'].median())} sessions {label}" if label else ""
        avg = f", average {tr.loc[mask, 'r'].mean():+.2f}R" if with_r else ""
        return f"{k} of {n} past setups ({k / n:.0%}){when}{avg}"
    if stats.exit_style == "trend":
        return {
            "stop": line(first == "stop", "to the stop"),
            "trend exit": line(last == "trend exit", "held", with_r=True),
            "signal exit": line(last == "signal exit", "held", with_r=True),
            "time stop": line(first == "time stop", "held", with_r=True),
        }
    return {
        "stop": line(first == "stop", "to the stop"),
        "target 1": line(tr["hit_target1"].astype(bool), None),
        "trailing": line(last == "breakeven/trailing stop", "to the exit"),
        "target 2": line(last == "target 2", "to target 2"),
        "time stop": line(first == "time stop", "held"),
    }


# --------------------------------------------------------------------------- #
# Setup rules
# --------------------------------------------------------------------------- #
def trade_rules(profile: InvestorProfile, params: StrategyParams, side: str) -> Tuple[InvestorProfile, StrategyParams]:
    """The profile and parameters whose stop, trailing-stop and time-stop rules apply to `side`.
    Longs follow the investor's horizon; shorts always follow the short-horizon rules (ATR stop)."""
    if side == "short":
        return profile.with_overrides(horizon="short"), params.copy(atr_stop=True)
    return profile, params


def _trend_score(decision) -> float:
    for b in decision.technical_beacons:
        if b.key == "trend":
            return float(b.score)
    return float("nan")


def evaluate_setup(decision, side: str, params: StrategyParams) -> Tuple[str, List[str], float]:
    """Is there a trade right now? Returns (status, reasons, regime size multiplier)."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    trend = _trend_score(decision)
    score, tech, verdict = decision.composite_score, decision.technical_score, decision.verdict
    reg = decision.regime
    reasons: List[str] = []
    if side == "long":
        mult = regime_adjustments(reg.is_bear if reg else False, reg.is_turbulent if reg else False, params)["size_multiplier"]
        bullish = verdict in {"Buy", "Strong Buy"}
        trend_up = np.isfinite(trend) and trend > 0
        reasons.append(f"Verdict {verdict} (score {score:.0f}; a buy needs ≥ 60).")
        reasons.append(f"Trend beacon {trend:+.2f}: " + ("price and the 50-day are above the 200-day (up-trend)." if trend_up
                                                         else "not in an up-trend yet."))
        if any("new entries are blocked" in w for w in decision.warnings):
            reasons.append("Market is Bear / Turbulent: new long entries are blocked by the regime rule.")
        if mult < 1:
            reasons.append(f"Bear market: position size × {mult:g}.")
        if bullish and trend_up:
            return "actionable", reasons, mult
        if (bullish and not trend_up) or (verdict == "Hold" and trend_up and tech >= params.entry_score):
            return "watch", reasons, mult
        return "avoid", reasons, mult

    # short side: mirror of the long rule
    bull = reg is not None and not reg.is_bear
    calm_bull = bull and not reg.is_turbulent
    mult = 0.5 if bull else 1.0
    bearish = score <= SHORT_MAX_SCORE
    trend_down = np.isfinite(trend) and trend < 0
    tech_weak = tech <= 100.0 - params.entry_score
    reasons.append(f"Score {score:.0f} (a short needs ≤ {SHORT_MAX_SCORE:.0f}: the mirror of a buy's ≥ 60).")
    reasons.append(f"Trend beacon {trend:+.2f}: " + ("price and the 50-day are below the 200-day (down-trend)." if trend_down
                                                     else "not in a down-trend."))
    reasons.append(f"Technical score {tech:.0f} (a short needs ≤ {100 - params.entry_score:.0f}).")
    if calm_bull:
        reasons.append("Market is Bull / Calm: shorts fight the tide, so new shorts are blocked (mirror of the long rule's "
                       "Bear / Turbulent block).")
    elif bull:
        reasons.append("Bull market: short size × 0.5.")
    if bearish and trend_down and tech_weak and not calm_bull:
        return "actionable", reasons, mult
    if score < 45 and trend_down:
        return "watch", reasons, mult
    return "avoid", reasons, mult


# --------------------------------------------------------------------------- #
# Ticket
# --------------------------------------------------------------------------- #
def _split_shares(shares: int, partial: float) -> Tuple[int, int]:
    """Shares closed at target 1 and target 2."""
    if shares <= 0:
        return 0, 0
    if partial <= 0.0:
        return 0, shares                 # nothing sold at target 1; it only moves the stop to breakeven
    if shares == 1 or partial >= 1.0:
        return shares, 0                 # a single share cannot be split: all of it goes at target 1
    n1 = min(max(int(round(shares * partial)), 1), shares - 1)
    return n1, shares - n1


def build_trade_plan(result, side: str = "long", settings: Optional[TradeSettings] = None, cash: Optional[float] = None,
                     with_stats: bool = True, allow_stale: Optional[bool] = None) -> TradePlan:
    """The trade ticket for one analysed stock. `cash` overrides the profile's cash available (used when several
    ideas share one pot of cash).

    Refuses (StaleDataError) when the data is not current (bundled sample, a stale cache, a past date, missing
    sessions) unless `allow_stale` (default: the LIGHTHOUSE_ALLOW_STALE_DATA environment variable) says otherwise; an
    overridden ticket says so in its first warning."""
    freshness = require_current_data(result, allow_stale)
    s = settings or TradeSettings()
    d, prof, p = result.decision, result.profile, result.params
    if cash is not None:
        prof = prof.with_overrides(cash_available=float(min(max(cash, 0.0), prof.account_value)))
    last = result.indicators.iloc[-1]
    price, atr = float(d.price), float(last["atr"])
    rv = float(last["sizing_vol"]) if np.isfinite(last["sizing_vol"]) else float("nan")
    status, reasons, mult = evaluate_setup(d, side, p)
    short = side == "short"
    warnings: List[str] = []
    notes: List[str] = []

    rprof, rparams = trade_rules(prof, p, side)
    trend = trend_plan_applies(side, rprof, rparams, s)
    stop, stop_label, _ = stop_level(price, atr, float(last["sma_slow"]), rprof, rparams, side=side)
    stop_label = stop_label.replace("trailing stop", "stop below the entry")   # the ticket only trails after target 1
    stop_pct = None if stop is None else max(abs(stop / price - 1.0), 0.01)
    risk_ps = 0.0 if stop is None else abs(price - stop)

    if short:
        conviction = conviction_from_score(100.0 - d.composite_score)
        entry_limit = price - s.chase_atr * atr
    else:
        conviction = conviction_from_score(d.composite_score)
        entry_limit = price + s.chase_atr * atr

    shares, value, pct, binding = 0, 0.0, 0.0, "no trade"
    if status in {"actionable", "watch"} and stop is not None:
        base_pct, binding = risk_sizing(price, stop_pct, rv, prof, notes)
        pct = float(min(base_pct * conviction * mult, prof.max_position_pct))
        notes.append(f"Scaled by conviction {conviction:.2f} and regime multiplier {mult:g}.")
        if short:
            notes.append("Short collateral is capped at the cash available (1:1), more conservative than Regulation T's 50% "
                         "initial margin, which would allow up to 2:1.")
        pct, value, shares, binding = dollars_to_shares(pct, price, prof, binding, notes)
        if status == "watch":
            notes.insert(0, "Conditional ticket: only place it if the setup turns actionable.")
    elif stop is None:
        warnings.append("No stop could be computed (volatility unknown), so no position is sized.")

    t_days = s.time_stop_days if trend else s.time_stop_for(rprof.horizon)
    typical_days = int(t_days or TIME_STOP_DEFAULT.get(rprof.horizon, 252))
    m1, m2 = s.reward_multiples
    n1, n2 = _split_shares(shares, s.partial_exit)
    targets: List[Target] = []
    for m, n in ((m1, n1), (m2, n2)):
        if risk_ps <= 0 or trend:
            break
        tp = price - m * risk_ps if short else price + m * risk_ps
        if tp <= 0.01 * price:
            warnings.append(f"The {m:g}R target would be at or below zero; it is not reachable for this short.")
            continue
        targets.append(Target(m, float(tp), float((tp / price - 1.0) * 100.0), int(n), float(n * m * risk_ps)))

    typical = float(rv * math.sqrt(typical_days / p.trading_days) * 100.0) if np.isfinite(rv) else float("nan")
    trend_fields: Dict[str, object] = {}
    if trend:
        sma = result.indicators["sma_slow"]
        buffer = rparams.exit_below_sma
        level = float(sma.iloc[-1] * (1.0 - buffer)) if np.isfinite(sma.iloc[-1]) else None
        pace = float((sma.iloc[-1] / sma.iloc[-22] - 1.0) * 100.0) if len(sma) > 22 and sma.iloc[-22] > 0 else None
        trend_fields = dict(exit_style="trend", trend_exit_level=level, trend_exit_pace_pct=pace,
                            trend_exit_label=f"200-day SMA minus {buffer * 100:.0f}%", signal_exit_score=float(rparams.exit_score),
                            review=REVIEW_PERIOD.get(rparams.decision_frequency, "month"))
        if level is not None and level >= price:
            warnings.append(f"Price is already below the trend-exit level (${level:,.2f}): the next review would sell.")
    if targets and np.isfinite(typical) and typical > 0:
        z = abs(targets[0].move_pct) / typical
        if z > 1.5:
            warnings.append(f"Target 1 needs a {abs(targets[0].move_pct):.1f}% move, {z:.1f}× the typical 1σ move of "
                            f"{typical:.1f}% over {t_days} sessions: ambitious for this window.")

    as_of = pd.Timestamp(d.as_of)
    if (pd.Timestamp.today().normalize() - as_of).days > 4:
        warnings.append(f"Prices are from {d.as_of}. Refresh with live data and re-check every level before placing an order.")
    warnings.append("Stops are orders, not guarantees: an overnight gap (earnings, news) can fill beyond the stop. "
                    "Check the next earnings date before entering.")

    margin: Dict[str, float] = {}
    if short:
        if prof.horizon == "long":
            notes.append("Shorts use the short-horizon rules (ATR stop, short time stop) even for a long-term investor: a short's "
                         "loss is open-ended and borrow fees accrue every day it is held.")
        notional = shares * price
        maint = (max(FINRA_SHORT_MAINT_PCT * notional, FINRA_SHORT_MAINT_PER_SHARE * shares) if price >= 5
                 else max(notional, FINRA_LOW_PRICE_MAINT_PER_SHARE * shares))
        margin = {"reg_t_initial": REG_T_INITIAL_SHORT * notional, "finra_maintenance": maint}
        warnings += [
            "Short selling needs a margin account (not a cash account or an IRA) and shares your broker can borrow; "
            "hard-to-borrow stocks charge a daily borrow fee and the lender can recall the shares.",
            "A short's loss is not capped by the purchase price: if the stock gaps up through the stop (or is squeezed), "
            "the loss can exceed the plan.",
            "Short sellers owe any dividends paid while short (payment in lieu).",
            "SEC Rule 201: after a 10% one-day fall, shorts may only be sold above the national best bid for the rest of "
            "that day and the next.",
        ]

    plan = TradePlan(
        ticker=d.ticker, name=result.fundamentals.name or d.ticker, side=side, status=status, reasons=reasons, as_of=d.as_of,
        price=price, entry_limit=float(entry_limit), stop=stop, stop_label=stop_label, stop_pct=stop_pct,
        risk_per_share=float(risk_ps), shares=int(shares), notional=float(shares * price),
        dollars_at_risk=float(shares * risk_ps), risk_pct_of_account=float(100.0 * shares * risk_ps / prof.account_value),
        targets=targets, trail_atr_multiple=float(rprof.stop_atr_multiple), time_stop_days=t_days, typical_move_pct=typical,
        atr=atr, composite_score=float(d.composite_score), technical_score=float(d.technical_score),
        fundamental_score=d.fundamental_score, verdict=d.verdict, regime=d.regime.label if d.regime else "n/a",
        conviction=float(conviction), size_binding=binding, cash_available=prof.cash, account_value=prof.account_value,
        sizing_notes=notes, warnings=warnings, margin=margin,
        priced_from=(f"live quote {result.quote.describe()}" if getattr(result, "quote", None) is not None
                     else f"close of {d.as_of}"),
        horizon=rprof.horizon, partial_exit=s.partial_exit, typical_days=typical_days, **trend_fields,
    )
    plan.data_status = freshness
    if freshness.startswith("OVERRIDE"):
        plan.warnings.insert(0, f"Not for trading: {freshness[len('OVERRIDE: '):]}. This ticket was made with the "
                                "stale-data override.")
    if with_stats:
        attach_stats(plan, result, s)
    return plan


def attach_stats(plan: TradePlan, result, settings: Optional[TradeSettings] = None) -> TradePlan:
    """Run the historical setup test for `plan` and add a warning when its record is weak or thin."""
    try:
        st = setup_statistics(result, plan.side, settings)
    except ValueError as exc:
        plan.warnings.append(f"Historical setup test unavailable: {exc}")
        return plan
    plan.stats = st
    if st.avg_r <= 0:
        plan.warnings.insert(0, f"On this stock's own history these {plan.side} rules LOST money on average "
                                f"({st.avg_r:+.2f}R per trade over {st.n_trades} trades): treat the idea with suspicion.")
    if not st.reliable:
        plan.warnings.append(f"Only {st.n_trades} historical trades: too few to judge the rules on this stock.")
    return plan


# --------------------------------------------------------------------------- #
# Trade simulation (shared by the setup test and the position monitor)
# --------------------------------------------------------------------------- #
def signed_usd(v: float) -> str:
    """Signed dollars for results: '+$1,234' / '−$1,234' (a true minus sign)."""
    return f"{'−' if v < 0 else '+'}${abs(v):,.0f}"


def _bar_exit(o: float, h: float, l: float, c: float, stop: float, target: Optional[float], check_open: bool,
              real_ohlc: bool) -> Optional[Tuple[str, float]]:
    """Long-space exit test for one bar. Gaps fill at the open; if the stop and the target are both inside the bar's
    range the stop is assumed to come first (conservative). Close-only data fills at the observed close."""
    if not real_ohlc:
        if c <= stop:
            return "stop", c
        if target is not None and c >= target:
            return "target", c
        return None
    if check_open:
        if o <= stop:
            return "stop", o
        if target is not None and o >= target:
            return "target", o
    if l <= stop:
        return "stop", stop
    if target is not None and h >= target:
        return "target", target
    return None


def _simulate(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, atr: np.ndarray, start: int, entry: float,
              stop: float, tp1: Optional[float], tp2: Optional[float], partial: float, trail_k: float, max_bars: Optional[int],
              real_ohlc: bool, check_first_bar: bool) -> Dict[str, object]:
    """Walk one long-space trade forward from bar `start` (the entry bar). Returns the exits and the live state.

    Shorts are simulated by negating prices (and swapping high/low) before calling this, so one engine serves both.
    """
    n = len(c)
    remaining, exits = 1.0, []          # exits: (fraction, price, reason, bar)
    stage1 = True
    cur_stop, best_close = stop, entry
    tp1_bar = None
    end = n if max_bars is None else min(n, start + max_bars)
    for j in range(start, end):
        first = j == start
        if not first or check_first_bar:
            target = tp1 if stage1 else tp2
            hit = _bar_exit(o[j], h[j], l[j], c[j], cur_stop, target, check_open=not first, real_ohlc=real_ohlc)
            if hit is not None:
                reason, px = hit
                if reason == "stop":
                    exits.append((remaining, px, "stop" if stage1 else "breakeven/trailing stop", j))
                    remaining = 0.0
                    break
                if stage1:
                    frac = partial if (tp2 is not None and partial < 1.0) else 1.0
                    exits.append((remaining * frac, px, "target 1", j))
                    remaining -= remaining * frac
                    stage1, tp1_bar = False, j
                    if remaining <= 1e-12:
                        remaining = 0.0
                        break
                    cur_stop = max(cur_stop, entry)      # breakeven on the rest
                else:
                    exits.append((remaining, px, "target 2", j))
                    remaining = 0.0
                    break
        best_close = max(best_close, c[j])
        if not stage1 and np.isfinite(atr[j]):
            cur_stop = max(cur_stop, best_close - trail_k * atr[j])
    last_bar = min(end, n) - 1
    timed_out = remaining > 0 and max_bars is not None and start + max_bars <= n
    if timed_out:
        exits.append((remaining, c[last_bar], "time stop", last_bar))
        remaining = 0.0
    return {"exits": exits, "remaining": remaining, "stop": cur_stop, "tp1_hit": tp1_bar is not None,
            "tp1_bar": tp1_bar, "last_bar": last_bar, "best_close": best_close}


def _simulate_trend(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, start: int, stop: float,
                    exit_level: np.ndarray, tech: np.ndarray, review: np.ndarray, exit_score: float, max_bars: Optional[int],
                    real_ohlc: bool, check_first_bar: bool) -> Dict[str, object]:
    """Walk one long trade under the trend plan from bar `start` (the entry bar).

    The stop order stays where it was placed (a gap below it fills at the open). On a review day, a technical score at
    or below `exit_score` ("signal exit") or a close below `exit_level` ("trend exit": the 200-day SMA minus the buffer)
    sells everything at the next session's open (its close for close-only data): the long-horizon backtest's exits.
    `pending` is the exit decided at the last close when the data ends before it could be filled.
    """
    n = len(c)
    end = n if max_bars is None else min(n, start + max_bars)
    exits: list = []
    pending: Optional[str] = None
    last = start
    for j in range(start, end):
        last = j
        if pending is not None:
            exits.append((1.0, o[j] if real_ohlc else c[j], pending, j))
            pending = None
            break
        if j > start or check_first_bar:
            hit = _bar_exit(o[j], h[j], l[j], c[j], stop, None, check_open=j > start, real_ohlc=real_ohlc)
            if hit is not None:
                exits.append((1.0, hit[1], "stop", j))
                break
        if review[j]:
            if np.isfinite(tech[j]) and tech[j] <= exit_score:
                pending = "signal exit"
            elif np.isfinite(exit_level[j]) and c[j] < exit_level[j]:
                pending = "trend exit"
    if not exits and pending is None and max_bars is not None and start + max_bars <= n:
        exits.append((1.0, c[end - 1], "time stop", end - 1))
    return {"exits": exits, "remaining": 0.0 if exits else 1.0, "pending": pending if not exits else None,
            "stop": stop, "last_bar": last}


def _to_long_space(df: pd.DataFrame, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    if side == "short":
        return -o, -l, -h, -c
    return o, h, l, c


def _has_real_ohlc(df: pd.DataFrame) -> bool:
    c = df["Close"].to_numpy(dtype=float)
    return bool(np.nanmean(np.abs(df["High"].to_numpy(dtype=float) - df["Low"].to_numpy(dtype=float)) / c) > 1e-6)


# --------------------------------------------------------------------------- #
# Historical setup test
# --------------------------------------------------------------------------- #
def _historical_conditions(result, side: str) -> np.ndarray:
    """Point-in-time version of `evaluate_setup` (actionable only) for every date."""
    ind, p, prof = result.indicators, result.params, result.profile
    tech = result.technical_scores.reindex(ind.index).to_numpy(dtype=float)
    trend = beacon_scores(ind, p)["trend"].to_numpy(dtype=float)
    pit = result.pit_scores.reindex(ind.index).to_numpy(dtype=float) if result.pit_scores is not None else np.full(len(ind), np.nan)
    blended = np.where(np.isfinite(pit), prof.fundamental_weight * pit + prof.technical_weight * tech, tech)
    if result.regimes is not None:
        rt = result.regimes.reindex(ind.index.union(result.regimes.index)).ffill().reindex(ind.index)
        known = rt["regime"].notna().to_numpy()
        bull = (rt["bull"].to_numpy() == 1.0) & known
        bear = (rt["bull"].to_numpy() == 0.0) & known
        turb = (rt["turbulent"].to_numpy() == 1.0) & known
        calm = (rt["turbulent"].to_numpy() == 0.0) & known
    else:
        bull = bear = turb = calm = np.zeros(len(ind), dtype=bool)
    penalty = np.where(bear, p.bear_score_penalty, 0.0) + np.where(turb, p.turbulent_score_penalty, 0.0)
    composite = np.clip(blended - penalty, 0.0, 100.0)
    with np.errstate(invalid="ignore"):
        if side == "long":
            cond = (composite >= 60.0) & (trend > 0) & ~(bear & turb)
        else:
            cond = (composite <= SHORT_MAX_SCORE) & (trend < 0) & (tech <= 100.0 - p.entry_score) & ~(bull & calm)
    return cond & np.isfinite(tech) & np.isfinite(trend)


def setup_statistics(result, side: str = "long", settings: Optional[TradeSettings] = None) -> SetupStats:
    """Replay the trade-management rules on every historical setup of this stock, one trade at a time.

    Entry: the session after the signal, at the open (the close for close-only data). Stops and targets are set from
    the fill exactly as the live ticket sets them. Costs: commission + slippage on both sides, from StrategyParams.
    Trades still open at the end of the data are excluded. Results are in R (multiples of the initial risk).
    """
    s = settings or TradeSettings()
    ind = result.indicators
    prof, p = trade_rules(result.profile, result.params, side)
    trend = trend_plan_applies(side, prof, p, s)
    cond = _historical_conditions(result, side)
    o, h, l, c = _to_long_space(ind, side)
    atr = ind["atr"].to_numpy(dtype=float)
    sma = ind["sma_slow"].to_numpy(dtype=float)
    real = _has_real_ohlc(ind)
    cost = (p.cost_bps + p.slippage_bps) / 10_000.0
    t_days = s.time_stop_days if trend else s.time_stop_for(prof.horizon)
    if trend:
        exit_level, tech_now, review = _trend_inputs(result, p)
    m1, m2 = s.reward_multiples
    rows = []
    i, n = 0, len(ind)
    while i < n - 1:
        if not cond[i]:
            i += 1
            continue
        e = i + 1
        entry_raw = abs(o[e]) if real else abs(c[e])
        stop_raw, _, _ = stop_level(entry_raw, atr[i], sma[i], prof, p, side=side)
        if stop_raw is None or not np.isfinite(stop_raw):
            i += 1
            continue
        risk = abs(entry_raw - stop_raw)
        sgn = -1.0 if side == "short" else 1.0
        entry_ls, stop_ls = sgn * entry_raw, sgn * stop_raw
        tp1_ls, tp2_ls = entry_ls + m1 * risk, entry_ls + m2 * risk
        if side == "short" and entry_raw - m1 * risk <= 0.01 * entry_raw:
            i += 1
            continue
        if side == "short" and entry_raw - m2 * risk <= 0.01 * entry_raw:
            tp2_ls = None
        if trend:
            sim = _simulate_trend(o, h, l, c, e, stop_ls, exit_level, tech_now, review, p.exit_score, t_days, real,
                                  check_first_bar=real)
        else:
            sim = _simulate(o, h, l, c, atr, e, entry_ls, stop_ls, tp1_ls, tp2_ls, s.partial_exit, prof.stop_atr_multiple,
                            t_days, real, check_first_bar=real)
        if sim["remaining"] > 0:      # still open when the data ends: no outcome yet
            break
        exits = sim["exits"]
        gross_r = sum(f * (px - entry_ls) for f, px, _, _ in exits) / risk
        exit_avg = sum(f * abs(px) for f, px, _, _ in exits)
        cost_r = cost * (entry_raw + exit_avg) / risk
        r = gross_r - cost_r
        reasons = [x[2] for x in exits]
        exit_bar = max(x[3] for x in exits)
        rows.append({"signal_date": ind.index[i], "entry_date": ind.index[e], "entry": entry_raw, "stop": stop_raw,
                     "exit_date": ind.index[exit_bar], "outcome": " + ".join(dict.fromkeys(reasons)), "r": r,
                     "return_pct": r * risk / entry_raw * 100.0, "days": int(exit_bar - e + 1),
                     "hit_target1": "target 1" in reasons})
        i = exit_bar          # a new signal may come at the close of the exit session
    years = (ind.index[-1] - ind.index[0]).days / 365.25
    if not rows:
        raise ValueError(f"no historical {side} setups in {years:.1f} years of data")
    tr = pd.DataFrame(rows)
    rr = tr["r"].to_numpy()
    wins, losses = rr[rr > 0].sum(), -rr[rr <= 0].sum()
    first = tr["outcome"].str.split(" + ", regex=False).str[0]
    return SetupStats(
        n_trades=len(tr), win_rate=float((rr > 0).mean()), avg_r=float(rr.mean()), median_r=float(np.median(rr)),
        profit_factor=float(wins / losses) if losses > 0 else float("inf"), hit_target1=float(tr["hit_target1"].mean()),
        stopped_out=float((first == "stop").mean()), timed_out=float((first == "time stop").mean()),
        avg_days=float(tr["days"].mean()), avg_return_pct=float(tr["return_pct"].mean()), years=float(years), trades=tr,
        exit_style="trend" if trend else "targets",
        trend_exited=float(first.isin(["trend exit", "signal exit"]).mean()))


def _trend_inputs(result, p: StrategyParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per session: the trend-exit level (200-day SMA minus the buffer), the technical score the backtest tests (smoothed
    like the backtest's) and whether the session is a review day."""
    ind = result.indicators
    exit_level = (ind["sma_slow"] * (1.0 - p.exit_below_sma)).to_numpy(dtype=float)
    tech = result.technical_scores.reindex(ind.index)
    if p.score_smoothing > 1:
        tech = tech.rolling(p.score_smoothing, min_periods=1).mean()
    return exit_level, tech.to_numpy(dtype=float), review_days(ind.index, p.decision_frequency)


# --------------------------------------------------------------------------- #
# Trade ideas for a pot of cash
# --------------------------------------------------------------------------- #
def allocate_ideas(results: Sequence, side: str = "long", settings: Optional[TradeSettings] = None,
                   include_watch: bool = True, with_stats: bool = False,
                   allow_stale: Optional[bool] = None) -> Tuple[List[TradePlan], List[TradePlan], pd.DataFrame]:
    """Fund the strongest actionable setups from the profile's cash, best first.

    Each idea is sized by the usual rules against the cash still unspent, and the total planned risk across ideas is
    capped at `max_total_risk_pct` of the account ("portfolio heat"). Returns (funded, watch list, summary table).
    """
    s = settings or TradeSettings()
    results = [r for r in results if r is not None]
    if not results:
        return [], [], pd.DataFrame()
    prof = results[0].profile
    current, refused = [], []
    for r in results:
        try:
            require_current_data(r, allow_stale)
            current.append(r)
        except StaleDataError as exc:
            refused.append((r.ticker, str(exc)))
    if not current:
        raise StaleDataError(refused[0][1] + (f" ({len(refused) - 1} more tickers refused for the same reason.)"
                                              if len(refused) > 1 else ""))
    results = current
    plans = [build_trade_plan(r, side, s, with_stats=False, allow_stale=allow_stale) for r in results]
    key = (lambda pl: -pl.composite_score) if side == "long" else (lambda pl: pl.composite_score)
    candidates = sorted([pl for pl in plans if pl.status == "actionable"], key=key)
    watch = sorted([pl for pl in plans if pl.status == "watch"], key=key) if include_watch else []
    by_ticker = {r.ticker: r for r in results}
    cash_left = prof.cash
    heat_left = prof.account_value * s.max_total_risk_pct / 100.0
    funded: List[TradePlan] = []
    for cand in candidates:
        if len(funded) >= s.max_ideas or heat_left <= 0 or cash_left <= 0:
            break
        if cash_left < cand.price:
            continue                      # a cheaper idea further down may still fit
        pl = build_trade_plan(by_ticker[cand.ticker], side, s, cash=cash_left, with_stats=False, allow_stale=allow_stale)
        if funded and pl.notional < s.min_fill_ratio * cand.notional:
            continue                      # the leftover cash would only buy a crumb of this idea
        if pl.shares > 0 and pl.dollars_at_risk > heat_left:
            keep = int(math.floor(heat_left / pl.risk_per_share)) if pl.risk_per_share > 0 else 0
            pl = _resize(pl, keep, s, f"Portfolio heat: total planned risk across ideas is capped at "
                                      f"{s.max_total_risk_pct:g}% of the account, so this idea is cut to {keep} shares.")
        if pl.shares <= 0:
            continue
        funded.append(pl)
        cash_left -= pl.notional
        heat_left -= pl.dollars_at_risk
    if with_stats:
        for pl in funded + watch:
            attach_stats(pl, by_ticker[pl.ticker], s)
    table = ideas_table(funded + watch)
    if refused:      # left out: their data is not current
        table = pd.concat([table, pd.DataFrame([{"ticker": t, "status": "refused: data not current", "binding": why}
                                                for t, why in refused])], ignore_index=True)
    return funded, watch, table


def _resize(pl: TradePlan, shares: int, s: TradeSettings, note: str) -> TradePlan:
    n1, n2 = _split_shares(shares, s.partial_exit)
    for t, n in zip(pl.targets, (n1, n2)):
        t.shares = int(n)
        t.profit = float(n * t.r_multiple * pl.risk_per_share)
    pl.shares = int(shares)
    pl.notional = float(shares * pl.price)
    pl.dollars_at_risk = float(shares * pl.risk_per_share)
    pl.risk_pct_of_account = float(100.0 * pl.dollars_at_risk / pl.account_value)
    pl.size_binding = "portfolio heat"
    pl.sizing_notes.append(note)
    return pl


def ideas_table(plans: Iterable[TradePlan]) -> pd.DataFrame:
    rows = []
    for pl in plans:
        t1 = pl.targets[0] if pl.targets else None
        t2 = pl.targets[1] if len(pl.targets) > 1 else None
        rows.append({
            "ticker": pl.ticker, "name": pl.name, "side": pl.side, "status": pl.status, "verdict": pl.verdict,
            "score": round(pl.composite_score, 1), "price": round(pl.price, 2), "shares": pl.shares, "cost": round(pl.notional, 2),
            "stop": None if pl.stop is None else round(pl.stop, 2),
            "stop_pct": None if pl.stop_pct is None else round(pl.stop_pct * 100, 1),
            "risk_usd": round(pl.dollars_at_risk, 2), "risk_pct": round(pl.risk_pct_of_account, 2),
            "target1": None if t1 is None else round(t1.price, 2), "target2": None if t2 is None else round(t2.price, 2),
            "profit_t1": None if t1 is None else round(t1.profit, 2), "profit_all": round(pl.profit_if_all_targets, 2),
            "rr": None if t1 is None else t1.r_multiple, "exit": "trend" if pl.is_trend_plan else "targets",
            "trend_exit": None if pl.trend_exit_level is None else round(pl.trend_exit_level, 2),
            "regime": pl.regime, "binding": pl.size_binding, "as_of": pl.as_of,
            "priced_from": pl.priced_from,
            "hist_trades": pl.stats.n_trades if pl.stats else None,
            "hist_win_rate": None if not pl.stats else round(pl.stats.win_rate * 100, 0),
            "hist_avg_r": None if not pl.stats else round(pl.stats.avg_r, 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Open-position monitor (stop-loss lookout)
# --------------------------------------------------------------------------- #
@dataclass
class PositionStatus:
    ticker: str
    side: str
    entry_price: float
    shares: int
    entry_date: str
    as_of: str
    last_price: float
    days_held: int
    pnl: float                      # realised + unrealised, assuming the plan was followed (partial sold at target 1)
    pnl_pct: float                  # pnl as % of the cost of the original position
    r_multiple: float               # pnl in R (multiples of the initial risk on the whole position)
    initial_stop: float
    current_stop: float
    stop_note: str
    distance_to_stop_pct: float     # how far price can move against you before the stop (fraction, >= 0 when not hit)
    distance_to_stop_atr: float
    targets: List[Tuple[float, float, bool]]   # (R, price, reached?)
    status: str                     # STOP HIT | NEAR STOP | TARGET 2 HIT | TARGET 1 HIT | TIME STOP DUE | OK, and for a
                                    # trend plan TREND EXIT | SIGNAL EXIT | SELL AT THE OPEN
    action: str
    alerts: List[str]
    verdict_now: str
    score_now: float
    shares_open: int = 0            # shares still held after the plan's exits
    realised_pnl: float = 0.0       # locked in by the exits already taken (target 1 partial, target 2, stops)
    unrealised_pnl: float = 0.0     # on the shares still open, at the last price
    exits: List[Tuple[str, int, float, str]] = field(default_factory=list)   # (date, shares, price, reason)


def monitor_position(result, side: str, entry_price: float, shares: int, entry_date, stop: Optional[float] = None,
                     targets: Optional[Sequence[float]] = None, settings: Optional[TradeSettings] = None) -> PositionStatus:
    """Re-check an open position against its plan: has the stop or a target been hit, is price close to the stop,
    where should the (trailing) stop sit now, and does the latest verdict still support the trade?

    The path since entry is replayed with the same rules as the setup test, starting the session after entry. A long on
    the long horizon follows the trend plan (review-day trend and signal exits) unless `targets` are given.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    if entry_price <= 0 or shares <= 0:
        raise ValueError("entry_price and shares must be positive")
    s = settings or TradeSettings()
    ind, d = result.indicators, result.decision
    prof, p = trade_rules(result.profile, result.params, side)
    entry_ts = pd.Timestamp(entry_date)
    alerts: List[str] = []
    if entry_ts > ind.index[-1]:
        raise ValueError(f"entry date {entry_ts.date()} is after the last price in the data ({ind.index[-1].date()})")
    pos = int(ind.index.searchsorted(entry_ts))        # first session on/after the entry date
    if entry_ts < ind.index[0]:
        alerts.append(f"Entry date is before the loaded history; replay starts {ind.index[0].date()}.")
    ref = max(pos - 1, 0)          # indicators known when the order went in: the previous close
    if ind.index[pos] == entry_ts:
        lo_d, hi_d = float(ind["Low"].iloc[pos]), float(ind["High"].iloc[pos])
        if not (lo_d * 0.98 <= entry_price <= hi_d * 1.02):
            alerts.append(f"Entry ${entry_price:,.2f} is outside the {entry_ts.date()} price range (${lo_d:,.2f}–${hi_d:,.2f}): "
                          "check the entry price and date (and whether the data is split-adjusted).")
    atr_entry, sma_entry = float(ind["atr"].iloc[ref]), float(ind["sma_slow"].iloc[ref])
    if stop is None:
        stop_raw, stop_note, _ = stop_level(entry_price, atr_entry, sma_entry, prof, p, side=side)
        if stop_raw is None:
            raise ValueError("no stop could be derived (ATR unknown); enter your own stop")
        stop_note = f"plan stop: {stop_note}"
    else:
        stop_raw, stop_note = float(stop), "your stop"
    if (side == "long" and stop_raw >= entry_price) or (side == "short" and stop_raw <= entry_price):
        raise ValueError("the stop must be below the entry for a long and above it for a short")
    risk = abs(entry_price - stop_raw)
    if not targets and trend_plan_applies(side, prof, p, s):
        return _monitor_trend(result, entry_price, int(shares), entry_ts, pos, float(stop_raw), stop_note, s, alerts)
    m1, m2 = s.reward_multiples
    if targets:
        tps = sorted((float(t) for t in targets), reverse=(side == "short"))[:2]
        if any((t <= entry_price) if side == "long" else (t >= entry_price) for t in tps):
            raise ValueError("targets must be above the entry for a long and below it for a short")
    else:
        tps = [entry_price + m * risk if side == "long" else entry_price - m * risk for m in (m1, m2)]
        tps = [t for t in tps if t > 0.01 * entry_price]
    sgn = -1.0 if side == "short" else 1.0
    o, h, l, c = _to_long_space(ind, side)
    atr = ind["atr"].to_numpy(dtype=float)
    real = _has_real_ohlc(ind)
    sim = _simulate(o, h, l, c, atr, pos, sgn * entry_price, sgn * stop_raw, sgn * tps[0] if tps else None,
                    sgn * tps[1] if len(tps) > 1 else None, s.partial_exit, prof.stop_atr_multiple, None, real,
                    check_first_bar=False)
    last_price = float(ind["Close"].iloc[-1])
    last_atr = float(ind["atr"].iloc[-1])
    cur_stop = abs(sim["stop"])
    days_held = len(ind) - 1 - pos
    # Whole shares per exit, as the ticket splits them: target 1 sells `partial_exit` of the shares when a second target
    # exists; every other exit closes whatever is still open. P&L is realised on the exits, unrealised on the rest.
    n1, _ = _split_shares(shares, s.partial_exit) if len(tps) > 1 else (shares, 0)
    open_sh, realised, done = int(shares), 0.0, []
    for _frac, px, reason, bar in sim["exits"]:
        n = min(n1, open_sh) if reason == "target 1" else open_sh
        if n <= 0:
            continue
        realised += sgn * (abs(px) - entry_price) * n
        open_sh -= n
        done.append((str(ind.index[bar].date()), int(n), float(abs(px)), reason))
    unrealised = sgn * (last_price - entry_price) * open_sh
    pnl = realised + unrealised
    r_now = pnl / (risk * shares)
    dist = sgn * (last_price - cur_stop) / last_price
    dist_atr = sgn * (last_price - cur_stop) / last_atr if last_atr > 0 else float("nan")
    reached = []
    for k, tp in enumerate(tps):
        r_mult = abs(tp - entry_price) / risk
        hit = sim["tp1_hit"] if k == 0 else any(x[2] == "target 2" for x in sim["exits"])
        reached.append((float(r_mult), float(tp), bool(hit)))
    t_days = s.time_stop_for(prof.horizon)
    unverb = "Sell" if side == "long" else "Buy back"
    closing = done[-1] if open_sh == 0 and done else None     # the exit that closed the whole position, if any
    result_txt = f"Result {signed_usd(pnl)} ({r_now:+.2f}R)."
    if closing is not None and "stop" in closing[3]:
        status = "STOP HIT"
        action = (f"{unverb} the {'remaining ' if len(done) > 1 else ''}shares: the {closing[3]} was hit on {closing[0]} "
                  f"(fill ${closing[2]:,.2f}). {result_txt}")
    elif closing is not None and closing[3] == "target 2":
        status = "TARGET 2 HIT"
        action = (f"Target 2 (${tps[1]:,.2f}) was reached on {closing[0]} (fill ${closing[2]:,.2f}): take the rest of the profit. "
                  f"{result_txt}")
    elif closing is not None:
        status = "TARGET 1 HIT"
        action = (f"Target 1 (${tps[0]:,.2f}) was reached on {closing[0]} (fill ${closing[2]:,.2f}): the whole position closes "
                  f"there. {result_txt}")
    elif dist <= s.alert_atr * last_atr / last_price:
        status = "NEAR STOP"
        action = (f"Price is {dist * 100:.1f}% ({dist_atr:.2f} ATR) from the stop at ${cur_stop:,.2f}. Keep the stop order in "
                  "place; do not move it further away.")
    elif sim["tp1_hit"]:
        status = "TARGET 1 HIT"
        if len(tps) > 1 and open_sh < shares:
            action = (f"Target 1 was reached: {unverb.lower()} {shares - open_sh} shares there if you have not "
                      f"({signed_usd(realised)} locked in), keep the stop on the other {open_sh} at ${cur_stop:,.2f} (breakeven "
                      f"or better, trailing), and let them run to ${tps[1]:,.2f}.")
        elif len(tps) > 1:     # nothing is sold at target 1 (a 0% partial)
            action = (f"Target 1 was reached: keep all {open_sh} shares with the stop at ${cur_stop:,.2f} (breakeven or better, "
                      f"trailing) and let them run to ${tps[1]:,.2f}.")
        else:
            action = "Target 1 was reached: take the profit."
    elif days_held >= t_days:
        status = "TIME STOP DUE"
        action = f"Held {days_held} sessions without reaching target 1 (time stop {t_days}): close the trade and free the capital."
    else:
        status = "OK"
        action = f"On plan. Stop ${cur_stop:,.2f}, target 1 ${tps[0]:,.2f}." if tps else f"On plan. Stop ${cur_stop:,.2f}."
    if side == "long" and d.verdict in {"Reduce", "Sell"}:
        alerts.append(f"Lighthouse now says {d.verdict} (score {d.composite_score:.0f}): the evidence has turned against this long.")
    if side == "short" and d.verdict in {"Buy", "Strong Buy"}:
        alerts.append(f"Lighthouse now says {d.verdict} (score {d.composite_score:.0f}): the evidence has turned against this short.")
    if (pd.Timestamp.today().normalize() - ind.index[-1]).days > 4:
        alerts.append(f"Prices end {ind.index[-1].date()}: refresh live data for a current status.")
    return PositionStatus(
        ticker=result.ticker, side=side, entry_price=float(entry_price), shares=int(shares), entry_date=str(entry_ts.date()),
        as_of=str(ind.index[-1].date()), last_price=last_price, days_held=int(days_held), pnl=float(pnl),
        pnl_pct=float(100.0 * pnl / (entry_price * shares)), r_multiple=float(r_now), initial_stop=float(stop_raw),
        current_stop=float(cur_stop), stop_note=stop_note, distance_to_stop_pct=float(dist), distance_to_stop_atr=float(dist_atr),
        targets=reached, status=status, action=action, alerts=alerts, verdict_now=d.verdict, score_now=float(d.composite_score),
        shares_open=int(open_sh), realised_pnl=float(realised), unrealised_pnl=float(unrealised), exits=done)


def _monitor_trend(result, entry_price: float, shares: int, entry_ts: pd.Timestamp, pos: int, stop: float, stop_note: str,
                   s: TradeSettings, alerts: List[str]) -> PositionStatus:
    """`monitor_position` for a trend plan: the stop order, then the review-day trend and signal exits."""
    ind, d, p = result.indicators, result.decision, result.params
    exit_level, tech, review = _trend_inputs(result, p)
    o, h, l, c = _to_long_space(ind, "long")
    sim = _simulate_trend(o, h, l, c, pos, stop, exit_level, tech, review, p.exit_score, None, _has_real_ohlc(ind),
                          check_first_bar=False)
    last_price, last_atr = float(ind["Close"].iloc[-1]), float(ind["atr"].iloc[-1])
    risk = entry_price - stop
    days_held = len(ind) - 1 - pos
    done = [(str(ind.index[bar].date()), int(shares), float(px), reason) for _f, px, reason, bar in sim["exits"]]
    open_sh = 0 if done else shares
    realised = sum((px - entry_price) * n for _, n, px, _ in done)
    unrealised = (last_price - entry_price) * open_sh
    pnl = realised + unrealised
    r_now = pnl / (risk * shares)
    dist = (last_price - stop) / last_price
    dist_atr = (last_price - stop) / last_atr if last_atr > 0 else float("nan")
    period = REVIEW_PERIOD.get(p.decision_frequency, "month")
    label = f"200-day SMA minus {p.exit_below_sma * 100:.0f}%"
    level_now, tech_now = float(exit_level[-1]), float(tech[-1])
    result_txt = f"Result {signed_usd(pnl)} ({r_now:+.2f}R)."
    why = {"trend exit": f"a close below the {label}", "signal exit": f"a technical score at or below {p.exit_score:g}"}
    if done and done[-1][3] == "stop":
        status = "STOP HIT"
        action = f"Sell the shares: the stop was hit on {done[-1][0]} (fill ${done[-1][2]:,.2f}). {result_txt}"
    elif done:
        day, _, px, reason = done[-1]
        status = reason.upper()
        action = (f"Sell the shares: the {period}'s review found {why[reason]}; the plan sold at the open of {day} "
                  f"(${px:,.2f}). {result_txt}")
    elif sim["pending"] is not None:
        status = "SELL AT THE OPEN"
        action = (f"The review at the close of {ind.index[-1].date()} found {why[sim['pending']]}: sell everything at the "
                  f"next open. {result_txt}")
    elif dist <= s.alert_atr * last_atr / last_price:
        status = "NEAR STOP"
        action = (f"Price is {dist * 100:.1f}% ({dist_atr:.2f} ATR) from the stop at ${stop:,.2f}. Keep the stop order in "
                  "place; do not move it further away.")
    elif s.time_stop_days and days_held >= s.time_stop_days:
        status = "TIME STOP DUE"
        action = f"Held {days_held} sessions (time stop {s.time_stop_days}): close the trade and free the capital."
    else:
        status = "OK"
        action = (f"On plan: hold. Stop order ${stop:,.2f}. At the {period}'s last close, sell at the next open if the close "
                  f"is below ${level_now:,.2f} ({label}; it rises with the average) or the technical score is "
                  f"{p.exit_score:g} or lower (today {tech_now:.0f}).")
    if d.verdict in {"Reduce", "Sell"} and not done:
        alerts.append(f"Lighthouse now says {d.verdict} (score {d.composite_score:.0f}): the evidence has turned against this long.")
    if (pd.Timestamp.today().normalize() - ind.index[-1]).days > 4:
        alerts.append(f"Prices end {ind.index[-1].date()}: refresh live data for a current status.")
    return PositionStatus(
        ticker=result.ticker, side="long", entry_price=float(entry_price), shares=int(shares), entry_date=str(entry_ts.date()),
        as_of=str(ind.index[-1].date()), last_price=last_price, days_held=int(days_held), pnl=float(pnl),
        pnl_pct=float(100.0 * pnl / (entry_price * shares)), r_multiple=float(r_now), initial_stop=float(stop),
        current_stop=float(stop), stop_note=stop_note, distance_to_stop_pct=float(dist), distance_to_stop_atr=float(dist_atr),
        targets=[], status=status, action=action, alerts=alerts, verdict_now=d.verdict, score_now=float(d.composite_score),
        shares_open=int(open_sh), realised_pnl=float(realised), unrealised_pnl=float(unrealised), exits=done)
