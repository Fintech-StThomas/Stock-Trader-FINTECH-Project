"""
Honest, point-in-time backtest of the Lighthouse trading rule.

Rule (long-only, reviewed at the close on the decision day - monthly for the
long horizon, weekly for the short horizon; stops are checked daily):
  ENTER  when the trend beacon is positive (price and 50-day SMA above the
         200-day SMA), the Technical Score >= entry threshold (+ penalty in a
         bear market), the market is not Bear/Turbulent, and the point-in-time
         Fundamental Score (if known) is above the gate.
  EXIT   when the Technical Score <= exit threshold, when the close is more than
         the buffer below the 200-day SMA, or (short horizon) on the ATR stop.
  SIZE   volatility-targeted fraction of the sleeve x regime multiplier.
  FILL   orders decided on day t are filled at day t+1's open (or close when no
         open data exists) and pay commission + slippage on both sides.

No indicator looks forward, statements are used from the session after their
SEC filing, and every order is executed strictly after the information that
triggered it. Every result is shown next to buy-and-hold and next
to the same average exposure held without timing (`BacktestResult.same_exposure`): a
rule that sits in cash half the time has to beat holding half as much stock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .config import InvestorProfile, StrategyParams
from .decision import backtest_size_fraction, stop_level
from .indicators import max_drawdown
from .regime import regime_table
from .technicals import compute_indicators, technical_score, beacon_scores


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    size_fraction: float
    return_pct: float          # price return of the trade, net of costs
    pnl_pct: float             # contribution to sleeve equity (return x size)
    holding_days: int
    exit_reason: str


@dataclass
class BacktestResult:
    ticker: str
    equity: pd.Series                       # strategy, starts at 1.0
    buy_hold: pd.Series                     # same window, starts at 1.0
    benchmark: Optional[pd.Series]          # same window, starts at 1.0 (None if not given)
    exposure: pd.Series                     # fraction of the sleeve invested each day
    signals: pd.DataFrame                   # tech_score, trend, fund_score, regime flags, in_market
    trades: List[Trade]
    params: StrategyParams
    profile: InvestorProfile
    options: Dict[str, object] = field(default_factory=dict)
    risk_free: Optional[pd.Series] = None   # daily T-bill rate (annual fraction); None = params.risk_free_rate

    @property
    def rf(self) -> Union[float, pd.Series]:
        """The risk-free rate the Sharpe and Sortino ratios use: the T-bill series when there is one, else the fixed rate."""
        return self.risk_free if self.risk_free is not None else self.params.risk_free_rate

    # -- metrics ----------------------------------------------------------------
    @property
    def metrics(self) -> Dict[str, float]:
        m = compute_metrics(self.equity, self.rf, self.params.trading_days)
        m.update(trade_metrics(self.trades))
        m["exposure"] = float(self.exposure.mean())
        return m

    @property
    def same_exposure(self) -> pd.Series:
        """The stock and cash in a fixed mix at the rule's average exposure, rebalanced daily and never timed: what
        the rule's lower risk would have cost without any signal. Cash earns what the rule's cash earns."""
        return constant_mix(self.buy_hold, float(self.exposure.mean()), self.rf if self.params.cash_yield else 0.0,
                            self.params.trading_days)

    def summary(self) -> pd.DataFrame:
        rows = {"Lighthouse strategy": self.metrics,
                "Same exposure, never traded": compute_metrics(self.same_exposure, self.rf, self.params.trading_days),
                "Buy & hold": compute_metrics(self.buy_hold, self.rf, self.params.trading_days)}
        if self.benchmark is not None:
            rows["S&P 500"] = compute_metrics(self.benchmark, self.rf, self.params.trading_days)
        df = pd.DataFrame(rows).T
        cols = ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar",
                "n_trades", "win_rate", "avg_win", "avg_loss", "profit_factor", "exposure", "years"]
        return df.reindex(columns=[c for c in cols if c in df.columns])

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["entry_date", "entry_price", "exit_date", "exit_price", "size_fraction",
                                         "return_pct", "pnl_pct", "holding_days", "exit_reason"])
        return pd.DataFrame([t.__dict__ for t in self.trades])

    @property
    def start(self) -> pd.Timestamp:
        return self.equity.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.equity.index[-1]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rate_on(rate: pd.Series, index: pd.Index) -> pd.Series:
    """An annual rate published on some days (e.g. the T-bill) as a value for every date of `index`: the latest
    published value (dates before the first one take the first)."""
    s = rate.reindex(rate.index.union(index)).ffill().reindex(index)
    return s.fillna(float(rate.iloc[0])) if len(rate) else s


def constant_mix(buy_hold: pd.Series, weight: float, cash_rate: Union[float, pd.Series] = 0.0,
                 trading_days: int = 252) -> pd.Series:
    """Equity of `weight` in the stock (its buy-and-hold curve) and the rest in cash at `cash_rate` (annual, fixed or by
    date), rebalanced every day, starting at 1.0."""
    r = buy_hold.pct_change().fillna(0.0)
    if isinstance(cash_rate, pd.Series):
        rf = rate_on(cash_rate, buy_hold.index) / trading_days
    else:
        rf = pd.Series(float(cash_rate) / trading_days, index=buy_hold.index)
    daily = weight * r + (1.0 - weight) * rf
    daily.iloc[0] = 0.0
    return (1.0 + daily).cumprod().rename("same_exposure")


def compute_metrics(equity: pd.Series, risk_free_rate: Union[float, pd.Series] = 0.02, trading_days: int = 252) -> Dict[str, float]:
    """Return and risk metrics of an equity curve. `risk_free_rate`: a fixed annual rate, or a series of annual rates
    by date (the 3-month T-bill) applied day by day."""
    eq = equity.dropna()
    if len(eq) < 2:
        return {k: float("nan") for k in ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "years"]}
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(r.std(ddof=0) * np.sqrt(trading_days))
    if isinstance(risk_free_rate, pd.Series):
        rf_daily = rate_on(risk_free_rate, r.index) / trading_days
        excess = float((r - rf_daily).mean() * trading_days)
    else:
        rf_daily = risk_free_rate / trading_days
        excess = r.mean() * trading_days - risk_free_rate
    sharpe = float(excess / vol) if vol > 0 else float("nan")
    # Sortino & Price (1994): downside deviation below the target (here the risk-free rate), averaged over ALL
    # periods - not the standard deviation of the negative returns alone, which measures how losses vary around
    # their own average rather than how far returns fall short of the target.
    shortfall = np.minimum(r - rf_daily, 0.0)
    downside = float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(trading_days))
    sortino = float(excess / downside) if downside > 0 else float("nan")
    mdd = max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float("nan")
    return {"total_return": total, "cagr": cagr, "volatility": vol, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": float(mdd), "calmar": calmar, "years": float(years)}


def trade_metrics(trades: Sequence[Trade]) -> Dict[str, float]:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "avg_win": float("nan"), "avg_loss": float("nan"),
                "profit_factor": float("nan"), "avg_holding_days": float("nan"), "best_trade": float("nan"), "worst_trade": float("nan")}
    rets = np.array([t.return_pct for t in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    # profit factor = gross profit / gross loss in money terms, so each trade is weighted by its size
    pnl = np.array([t.pnl_pct for t in trades])
    gross_win, gross_loss = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return {"n_trades": n, "win_rate": float(len(wins) / n),
            "avg_win": float(wins.mean()) if len(wins) else float("nan"),
            "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_holding_days": float(np.mean([t.holding_days for t in trades])),
            "best_trade": float(rets.max()), "worst_trade": float(rets.min())}


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def run_backtest(prices: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None,
                 params: Optional[StrategyParams] = None, profile: Optional[InvestorProfile] = None,
                 fundamentals_pit: Optional[pd.Series] = None, use_regime: bool = True,
                 use_fundamental_gate: bool = True, use_stops: bool = True, sizing: str = "vol_target",
                 start=None, end=None, risk_free: Optional[pd.Series] = None) -> BacktestResult:
    """`risk_free`: the daily 3-month T-bill rate for the Sharpe and Sortino ratios (data.load_risk_free); None uses
    the fixed `params.risk_free_rate`. It changes no trade."""
    p = params or StrategyParams()
    prof = profile or InvestorProfile()
    ind = compute_indicators(prices, p)
    tscore_raw = technical_score(ind, prof.horizon, p)
    tscore = tscore_raw.rolling(p.score_smoothing, min_periods=1).mean() if p.score_smoothing > 1 else tscore_raw
    trend = beacon_scores(ind, p)["trend"]
    sma_dist = ind["Close"] / ind["sma_slow"] - 1.0
    if p.decision_frequency == "daily":
        decide = pd.Series(True, index=ind.index)
    else:
        rule = "W" if p.decision_frequency == "weekly" else "M"
        period = ind.index.to_period(rule).asi8          # integer period codes: comparing Period objects boxed every date
        decide = pd.Series(period != np.roll(period, -1), index=ind.index)   # last trading day of each week/month
        decide.iloc[-1] = True

    if benchmark is not None and use_regime:
        rt = regime_table(benchmark, p).reindex(ind.index.union(benchmark.index)).ffill().reindex(ind.index)
        bull = rt["bull"].fillna(1.0).values == 1.0
        turb = rt["turbulent"].fillna(0.0).values == 1.0
        regime_known = rt["regime"].notna().values
    else:
        bull = np.ones(len(ind), dtype=bool)
        turb = np.zeros(len(ind), dtype=bool)
        regime_known = np.zeros(len(ind), dtype=bool)

    if fundamentals_pit is not None and use_fundamental_gate:
        fscore = fundamentals_pit.reindex(ind.index).to_numpy(dtype=float, copy=True)
    else:
        fscore = np.full(len(ind), np.nan)

    # analysis window: first date with a defined technical score (and inside start/end if given); a copy, because
    # pandas 3 hands out a read-only view
    valid = tscore.notna().to_numpy(copy=True)
    if start is not None:
        valid &= (ind.index >= pd.Timestamp(start))
    if end is not None:
        valid &= (ind.index <= pd.Timestamp(end))
    if valid.sum() < 30:
        raise ValueError("not enough history to run a backtest (need the 200-day SMA / 52-week high to warm up)")
    first = int(np.argmax(valid))
    last = len(ind) - 1 - int(np.argmax(valid[::-1]))
    idx = ind.index[first:last + 1]
    n = len(idx)

    close = ind["Close"].values[first:last + 1]
    open_ = ind["Open"].values[first:last + 1]
    low = ind["Low"].values[first:last + 1]
    has_open = bool(np.nanmean(np.abs(open_ / close - 1.0)) > 1e-6)   # sample files without real opens use the close
    has_range = bool(np.nanmean(np.abs(ind["High"].values[first:last + 1] - low) / close) > 1e-6)
    atr = ind["atr"].values[first:last + 1]
    sma = ind["sma_slow"].values[first:last + 1]
    rvol = ind["sizing_vol"].values[first:last + 1]
    ts = tscore.values[first:last + 1]
    tr = trend.values[first:last + 1]
    sd = sma_dist.values[first:last + 1]
    dec = decide.values[first:last + 1]
    fs = fscore[first:last + 1]
    bull, turb, regime_known = bull[first:last + 1], turb[first:last + 1], regime_known[first:last + 1]

    # costs per side: fixed, or commission + Corwin-Schultz half spread (floored at the slippage) + square-root impact
    fixed_cost = (p.cost_bps + p.slippage_bps) / 10_000.0
    liquid = p.cost_model == "liquidity"
    if liquid:
        from .costs import MAX_COST_BPS, corwin_schultz_spread, dollar_volume, impact_bps
        half = (corwin_schultz_spread(ind["High"], ind["Low"]) * 1e4 / 2.0).fillna(0.0).clip(lower=p.slippage_bps)
        half = half.values[first:last + 1]
        lr = np.log(ind["Close"] / ind["Close"].shift(1))
        dvol = lr.rolling(21, min_periods=5).std().values[first:last + 1]
        adv = dollar_volume(ind["Close"], ind["Volume"] if "Volume" in ind else None).values[first:last + 1]

    def side_cost(i: int, weight: float) -> float:
        """Cost of trading `weight` of the sleeve on day i: the order is that share of the investor's account."""
        if not liquid:
            return fixed_cost
        dollars = p.order_value if p.order_value is not None else abs(weight) * prof.account_value
        # capped: a higher figure means a series without volume, not a price anyone pays (and an uncapped cost
        # above 100% would turn a sale into a negative fill)
        return float(min(p.cost_bps + half[i] + impact_bps(dollars, adv[i], dvol[i]), MAX_COST_BPS) / 10_000.0)

    # uninvested cash earns the risk-free rate (the T-bill series when given, else the fixed rate)
    if p.cash_yield:
        rf_d = (rate_on(risk_free, idx).to_numpy() if risk_free is not None and len(risk_free)
                else np.full(n, p.risk_free_rate)) / p.trading_days
    else:
        rf_d = np.zeros(n)
    resting = use_stops and p.resting_stops

    cash, shares = 1.0, 0.0
    equity = np.empty(n)
    exposure = np.zeros(n)
    in_market = np.zeros(n, dtype=bool)
    trades: List[Trade] = []
    pending: Optional[Tuple[str, float, str]] = None   # (side, size_fraction, reason)
    entry_i = entry_px = size_frac = 0.0
    trail_high = stop = np.nan
    stop_order = np.nan                                  # the resting stop order (NaN: none)

    def close_trade(i: int, px_raw: float, reason: str) -> None:
        nonlocal cash, shares
        c = side_cost(i, size_frac)
        px = px_raw * (1.0 - c)
        cash += shares * px
        ret = px / entry_px - 1.0
        trades.append(Trade(idx[int(entry_i)], float(entry_px), idx[i], float(px), float(size_frac),
                            float(ret), float(ret * size_frac), int(i - entry_i), reason))
        shares = 0.0

    for i in range(n):
        if i > 0:
            cash *= 1.0 + rf_d[i]
        fill = open_[i] if has_open else close[i]
        entered_today = False
        # 1) execute yesterday's decision at the open
        if pending is not None:
            side, frac, reason = pending
            if side == "rebalance" and shares > 0:
                target_value = frac * (cash + shares * fill)
                delta_shares = target_value / fill - shares
                c = side_cost(i, frac - size_frac)
                if delta_shares > 0:
                    px = fill * (1.0 + c); cash -= delta_shares * px
                else:
                    px = fill * (1.0 - c); cash += -delta_shares * px
                shares += delta_shares
                size_frac = frac
            elif side == "buy" and shares == 0:
                c = side_cost(i, frac)
                px = fill * (1.0 + c)
                alloc = cash * frac
                shares = alloc / px
                cash -= alloc
                entry_i, entry_px, size_frac = i, px, frac
                entered_today = True
                trail_high = close[i - 1] if i > 0 else fill
                stop = trail_high - prof.stop_atr_multiple * (atr[i - 1] if i > 0 else atr[i])
                if resting:
                    if p.atr_stop:
                        stop_order = stop
                    else:   # long horizon: the ticket's stop order, the 200-day SMA minus the buffer, left in place
                        lvl, _, _ = stop_level(fill, atr[i - 1] if i > 0 else atr[i], sma[i - 1] if i > 0 else sma[i],
                                               prof, p)
                        stop_order = lvl if lvl is not None else np.nan
            elif side == "sell" and shares > 0:
                close_trade(i, fill, reason)
                stop_order = np.nan
            pending = None
        # 2) the resting stop during the day: at the open on a gap through it, else at the stop price
        if resting and shares > 0 and np.isfinite(stop_order):
            hit = None
            if not has_range:                      # close-only data: the close is all that is known
                if close[i] <= stop_order:
                    hit = close[i]
            elif not entered_today and has_open and open_[i] <= stop_order:
                hit = open_[i]
            elif low[i] <= stop_order:
                hit = stop_order
            if hit is not None:
                close_trade(i, hit, "stop order")
                stop_order = np.nan
        # 3) mark to market
        equity[i] = cash + shares * close[i]
        exposure[i] = (shares * close[i]) / equity[i] if equity[i] > 0 else 0.0
        in_market[i] = shares > 0
        # 4) decide for tomorrow (only information up to today's close is used)
        if i == n - 1:
            break
        if shares > 0:
            trail_high = max(trail_high, close[i])
            if np.isfinite(atr[i]):
                stop = max(stop, trail_high - prof.stop_atr_multiple * atr[i]) if np.isfinite(stop) else trail_high - prof.stop_atr_multiple * atr[i]
                if resting and p.atr_stop:
                    stop_order = stop              # the trailing stop order is raised at the close, never lowered
            if dec[i] and np.isfinite(ts[i]) and ts[i] <= p.exit_score:
                pending = ("sell", 0.0, "signal exit")
            elif dec[i] and p.exit_below_sma is not None and np.isfinite(sd[i]) and sd[i] < -p.exit_below_sma:
                pending = ("sell", 0.0, "below 200-day SMA")
            elif dec[i] and p.trend_exit is not None and np.isfinite(tr[i]) and tr[i] < p.trend_exit:
                pending = ("sell", 0.0, "trend broken")
            elif use_stops and p.atr_stop and not resting and np.isfinite(stop) and close[i] < stop:
                pending = ("sell", 0.0, "trailing stop")
            elif dec[i] and sizing != "full":
                # review day: steer the position back toward its volatility target
                score_for_size = ts[i] if not np.isfinite(fs[i]) else (prof.fundamental_weight * fs[i] + prof.technical_weight * ts[i])
                mult = (p.bear_size_multiplier if (use_regime and not bull[i]) else 1.0)
                target = backtest_size_fraction(atr[i], rvol[i], score_for_size, prof, mult, mode=sizing)
                current = (shares * close[i]) / equity[i] if equity[i] > 0 else 0.0
                if abs(target - current) > p.rebalance_threshold:
                    pending = ("rebalance", target, "rebalance")
        else:
            if not dec[i] or not np.isfinite(ts[i]):
                continue
            threshold = p.entry_score + (p.bear_entry_penalty if (use_regime and not bull[i]) else 0.0)
            blocked = use_regime and p.block_entries_in_bear_turbulent and (not bull[i]) and turb[i]
            gate_ok = (not use_fundamental_gate) or (not np.isfinite(fs[i])) or fs[i] >= p.fundamental_gate_min
            trend_ok = (not p.trend_gate) or (np.isfinite(tr[i]) and tr[i] > 0.0)
            if p.max_entry_distance is not None and np.isfinite(sd[i]) and sd[i] > p.max_entry_distance:
                trend_ok = False
            if p.min_entry_distance is not None and not (np.isfinite(sd[i]) and sd[i] > p.min_entry_distance):
                trend_ok = False
            if ts[i] >= threshold and not blocked and gate_ok and trend_ok:
                score_for_size = ts[i] if not np.isfinite(fs[i]) else (prof.fundamental_weight * fs[i] + prof.technical_weight * ts[i])
                mult = (p.bear_size_multiplier if (use_regime and not bull[i]) else 1.0)
                frac = backtest_size_fraction(atr[i], rvol[i], score_for_size, prof, mult, mode=sizing)
                if frac > 0:
                    pending = ("buy", frac, "signal entry")

    # close any open position at the final close so the trade list is complete
    if shares > 0:
        c = side_cost(n - 1, size_frac)
        px = close[-1] * (1.0 - c)
        ret = px / entry_px - 1.0
        trades.append(Trade(idx[int(entry_i)], float(entry_px), idx[-1], float(px), float(size_frac),
                            float(ret), float(ret * size_frac), int(n - 1 - entry_i), "end of data (open)"))

    eq = pd.Series(equity, index=idx, name="strategy")
    bh = pd.Series(close / close[0], index=idx, name="buy_hold")
    bench = None
    if benchmark is not None:
        b = benchmark["Close"].reindex(idx.union(benchmark.index)).ffill().reindex(idx)
        bench = (b / b.iloc[0]).rename("benchmark")
    signals = pd.DataFrame({"tech_score": ts, "trend": tr, "fund_score": fs, "bull": bull.astype(float), "turbulent": turb.astype(float),
                            "regime_known": regime_known.astype(float), "in_market": in_market.astype(float)}, index=idx)
    return BacktestResult(prices.attrs.get("ticker", ""), eq, bh, bench, pd.Series(exposure, index=idx), signals, trades,
                          p, prof, {"use_regime": use_regime, "use_fundamental_gate": use_fundamental_gate,
                                    "use_stops": use_stops, "sizing": sizing, "fills": "next open" if has_open else "next close",
                                    "risk_free": describe_risk_free(risk_free, idx, p.risk_free_rate),
                                    "costs": ("commission + Corwin-Schultz half spread (floor "
                                              f"{p.slippage_bps:g} bps) + square-root impact" if liquid else
                                              f"fixed {p.cost_bps + p.slippage_bps:g} bps per side"),
                                    "stops": ("resting orders (gaps fill at the open)" if resting else
                                              "checked at the close, filled next open" if use_stops else "off"),
                                    "cash": "earns the risk-free rate" if p.cash_yield else "earns 0%"},
                          risk_free=risk_free)


def describe_risk_free(risk_free: Optional[pd.Series], index: pd.Index, fixed: float) -> str:
    """One line on the risk-free rate behind the Sharpe and Sortino ratios."""
    if risk_free is None or not len(risk_free) or not len(index):
        return f"a fixed {fixed * 100:g}% a year (no T-bill series: offline or sample data)"
    on = rate_on(risk_free, index)
    return (f"the 3-month T-bill day by day (FRED DTB3): {on.mean() * 100:.1f}% a year on average over the test, "
            f"{on.iloc[-1] * 100:.2f}% at its end")


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def trend_rule(prices, params: StrategyParams, profile: InvestorProfile, start=None, risk_free=None) -> BacktestResult:
    """The plain 200-day trend rule on the same engine as the product's rule: on each review day, fully invested when
    the close is above the 200-day SMA, in cash (earning the risk-free rate) when below; the same review days, fills,
    costs and cash yield, no score, no market regime, no stops. The yardstick for what the beacons add over one line
    (METHODOLOGY section 8d)."""
    p = params.copy(entry_score=0.0, exit_score=-1.0, trend_gate=False, min_entry_distance=0.0, exit_below_sma=0.0,
                    trend_exit=None, max_entry_distance=None)
    return run_backtest(prices, None, p, profile, use_regime=False, use_stops=False, sizing="full", start=start,
                        risk_free=risk_free)


def strategy_layers(prices, benchmark, params, profile, fundamentals_pit=None, start=None, risk_free=None) -> pd.DataFrame:
    """Add the product's layers one at a time to show what each contributes, then the two yardsticks every timing rule
    has to beat: the full rule's average exposure held without timing, and the plain 200-day trend rule."""
    variants = [
        ("Technical signals only (fully invested)", dict(use_regime=False, use_fundamental_gate=False, use_stops=False, sizing="full")),
        ("+ market regime filter", dict(use_regime=True, use_fundamental_gate=False, use_stops=False, sizing="full")),
        ("+ fundamental gate (point-in-time)", dict(use_regime=True, use_fundamental_gate=True, use_stops=False, sizing="full")),
    ]
    if params.atr_stop:
        variants.append(("+ trailing ATR stop", dict(use_regime=True, use_fundamental_gate=True, use_stops=True, sizing="full")))
    variants.append(("+ volatility-targeted sizing (full Lighthouse)", dict(use_regime=True, use_fundamental_gate=True, use_stops=True, sizing="vol_target")))
    rows = {}
    base = full = None
    for name, opts in variants:
        res = run_backtest(prices, benchmark, params, profile, fundamentals_pit, start=start, risk_free=risk_free, **opts)
        rows[name] = res.metrics
        base = base or res
        full = res
    same = compute_metrics(full.same_exposure, full.rf, params.trading_days)
    same["exposure"] = float(full.exposure.mean())
    rows["Same exposure, never traded"] = same
    rows["Plain 200-day trend rule"] = trend_rule(prices, params, profile, start=start, risk_free=risk_free).metrics
    rows["Buy & hold"] = compute_metrics(base.buy_hold, base.rf, params.trading_days)
    if base.benchmark is not None:
        rows["S&P 500"] = compute_metrics(base.benchmark, base.rf, params.trading_days)
    df = pd.DataFrame(rows).T
    return df.reindex(columns=["cagr", "volatility", "sharpe", "max_drawdown", "n_trades", "win_rate", "exposure"])


def sensitivity_grid(prices, benchmark, params, profile, fundamentals_pit=None,
                     entry_values=(50, 55, 60, 65, 70), exit_values=(25, 30, 35, 40, 45), start=None,
                     risk_free=None) -> pd.DataFrame:
    """Backtest every (entry, exit) threshold pair. A robust rule has a *plateau*, not a single spike."""
    rows = []
    for e_in, e_out in product(entry_values, exit_values):
        if e_out >= e_in:
            continue
        res = run_backtest(prices, benchmark, params.copy(entry_score=e_in, exit_score=e_out), profile, fundamentals_pit, start=start,
                           risk_free=risk_free)
        m = res.metrics
        rows.append({"entry": e_in, "exit": e_out, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "max_drawdown": m["max_drawdown"], "n_trades": m["n_trades"], "win_rate": m["win_rate"]})
    return pd.DataFrame(rows)


def parameter_sweep(prices, benchmark, params, profile, fundamentals_pit, name: str, values: Sequence, start=None) -> pd.DataFrame:
    rows = []
    for v in values:
        if name == "stop_atr_multiple":
            res = run_backtest(prices, benchmark, params, profile.with_overrides(stop_atr_override=float(v)), fundamentals_pit, start=start)
        elif name == "target_vol":
            res = run_backtest(prices, benchmark, params, profile.with_overrides(target_vol_override=float(v)), fundamentals_pit, start=start)
        else:
            res = run_backtest(prices, benchmark, params.copy(**{name: v}), profile, fundamentals_pit, start=start)
        m = res.metrics
        rows.append({name: v, "cagr": m["cagr"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                     "n_trades": m["n_trades"], "win_rate": m["win_rate"]})
    return pd.DataFrame(rows)


def walk_forward(prices, benchmark, params, profile, fundamentals_pit=None, split: float = 0.6,
                 entry_values=(50, 55, 60, 65, 70), exit_values=(25, 30, 35, 40, 45), risk_free=None) -> Dict[str, object]:
    """Tune thresholds on the first `split` of history, then test them - untouched - on the rest.

    Reports the default rule and the tuned rule on the out-of-sample window, so the
    investor can see whether tuning helped or merely fitted noise.
    """
    base = run_backtest(prices, benchmark, params, profile, fundamentals_pit, risk_free=risk_free)
    rf = base.rf
    idx = base.equity.index
    split_date = idx[int(len(idx) * split)]
    grid = sensitivity_grid(prices, benchmark, params, profile, fundamentals_pit, entry_values, exit_values, start=None,
                            risk_free=risk_free)
    # in-sample: metrics restricted to the training window
    rows = []
    for _, g in grid.iterrows():
        res = run_backtest(prices, benchmark, params.copy(entry_score=g["entry"], exit_score=g["exit"]), profile, fundamentals_pit)
        is_eq = res.equity[res.equity.index < split_date]
        m = compute_metrics(is_eq, rf, params.trading_days)
        rows.append({"entry": g["entry"], "exit": g["exit"], "is_sharpe": m["sharpe"], "is_cagr": m["cagr"],
                     "is_trades": sum(1 for t in res.trades if t.exit_date < split_date)})
    is_df = pd.DataFrame(rows)
    eligible = is_df[is_df["is_trades"] >= 3] if (is_df["is_trades"] >= 3).any() else is_df
    best = eligible.sort_values("is_sharpe", ascending=False).iloc[0]
    tuned = params.copy(entry_score=float(best["entry"]), exit_score=float(best["exit"]))

    def oos(p):
        res = run_backtest(prices, benchmark, p, profile, fundamentals_pit)
        eq = res.equity[res.equity.index >= split_date]
        eq = eq / eq.iloc[0]
        m = compute_metrics(eq, rf, params.trading_days)
        m["n_trades"] = sum(1 for t in res.trades if t.entry_date >= split_date)
        return m, eq, res

    m_default, eq_default, _ = oos(params)
    m_tuned, eq_tuned, _ = oos(tuned)
    bh = base.buy_hold[base.buy_hold.index >= split_date]
    bh = bh / bh.iloc[0]
    m_bh = compute_metrics(bh, rf, params.trading_days)
    table = pd.DataFrame({f"Default rule ({params.entry_score:.0f}/{params.exit_score:.0f})": m_default, f"Tuned in-sample ({best['entry']:.0f}/{best['exit']:.0f})": m_tuned,
                          "Buy & hold": m_bh}).T
    return {"split_date": split_date, "in_sample": is_df, "best_in_sample": best, "tuned_params": tuned,
            "out_of_sample": table.reindex(columns=["total_return", "cagr", "volatility", "sharpe", "max_drawdown", "n_trades"]),
            "oos_equity": pd.DataFrame({"default": eq_default, "tuned": eq_tuned, "buy_hold": bh})}


def regime_breakdown(result: BacktestResult) -> pd.DataFrame:
    """Strategy vs buy-and-hold performance inside each market regime."""
    sig = result.signals
    r_s = result.equity.pct_change()
    r_b = result.buy_hold.pct_change()
    label = np.where(sig["bull"] == 1.0, "Bull", "Bear")
    label = [f"{t} / {'Turbulent' if v == 1.0 else 'Calm'}" for t, v in zip(label, sig["turbulent"])]
    df = pd.DataFrame({"regime": label, "strategy": r_s, "buy_hold": r_b, "exposure": result.exposure})
    df = df[sig["regime_known"] == 1.0]
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("regime")
    out = pd.DataFrame({
        "days": g.size(),
        "share_of_time": g.size() / len(df),
        "strategy_ann_return": g["strategy"].mean() * result.params.trading_days,
        "buy_hold_ann_return": g["buy_hold"].mean() * result.params.trading_days,
        "avg_exposure": g["exposure"].mean(),
    })
    return out.sort_values("days", ascending=False)
