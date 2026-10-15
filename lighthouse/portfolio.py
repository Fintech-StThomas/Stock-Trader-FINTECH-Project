"""
Portfolio construction and risk: turn a basket of stocks (yours, or one picked for a goal) into weights, dollars and
shares, measure the portfolio's risk, and test the allocation rule out-of-sample.

Estimation
----------
* Covariance : Ledoit & Wolf (2004) shrinkage of the sample covariance toward a scaled identity. The sample matrix is
               noisy when stocks are many relative to the history; shrinkage makes it stable and invertible.
* Returns    : not used. Historical means are the noisiest input in finance; no method here needs them.

Allocation methods (all long-only, fully invested before any volatility-target cash buffer)
-----------------------------------------------------------------------------------------
equal_weight   1/N. Hard to beat out-of-sample (DeMiguel, Garlappi & Uppal 2009); the honest baseline, and the best
               Sharpe ratio on the app's default basket.
risk_parity    equal risk contribution (Maillard, Roncalli & Teïletche 2010), solved with Spinu's (2013) convex form.
min_variance   the lowest-volatility mix (uses correlations; tends to concentrate in defensive names); the smallest
               drawdown on the default basket.
Inverse volatility, maximum Sharpe (Bayes-Stein returns) and hierarchical risk parity were removed in October 2026: on
the default basket none beat 1/N, and maximum Sharpe did worst with the most turnover (docs/METHODOLOGY.md section 10).

Weight caps (per stock, per sector) are applied inside the optimisers, or by projecting a heuristic's weights onto the
caps (closest feasible weights). A volatility target holds cash when the expected volatility is above the target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .backtest import compute_metrics
from .config import StrategyParams
from .data import DataUnavailable, load_prices
from .indicators import max_drawdown
from .parallel import map_tickers
from .technicals import compute_indicators, technical_score
from .universe import UNKNOWN_SECTOR, sectors_for

TRADING_DAYS = 252
METHODS: Dict[str, str] = {
    "equal_weight": "Equal weight (1/N)",
    "risk_parity": "Risk parity (equal risk contribution)",
    "min_variance": "Minimum variance",
}
# Market stress windows (S&P 500 closing peak -> trough) used for the stress test when the data covers them.
STRESS_WINDOWS: List[Tuple[str, str, str]] = [
    ("Dot-com bust", "2000-03-24", "2002-10-09"),
    ("Global financial crisis", "2007-10-09", "2009-03-09"),
    ("2011 US downgrade", "2011-04-29", "2011-10-03"),
    ("Q4 2018 sell-off", "2018-09-20", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 rate shock", "2022-01-03", "2022-10-12"),
]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioConstraints:
    max_weight: float = 0.25               # per stock
    min_weight: float = 0.0                # per stock (only for stocks that are held)
    max_sector_weight: Optional[float] = None   # per sector, e.g. 0.40; None = no sector cap
    target_vol: Optional[float] = None     # annual; above it the portfolio holds cash (volatility targeting)

    def __post_init__(self) -> None:
        if not (0 < self.max_weight <= 1):
            raise ValueError("max_weight must be in (0, 1]")
        if not (0 <= self.min_weight <= self.max_weight):
            raise ValueError("min_weight must be between 0 and max_weight")
        if self.max_sector_weight is not None and not (0 < self.max_sector_weight <= 1):
            raise ValueError("max_sector_weight must be in (0, 1]")
        if self.target_vol is not None and self.target_vol <= 0:
            raise ValueError("target_vol must be positive")

    def check_feasible(self, tickers: Sequence[str], sectors: Optional[Sequence[str]] = None) -> None:
        n = len(tickers)
        if n == 0:
            raise ValueError("the basket is empty")
        if n * self.max_weight < 1 - 1e-9:
            raise ValueError(f"{n} stocks with a {self.max_weight:.0%} cap cannot add up to 100%: add stocks or raise the cap "
                             f"(need at least {math.ceil(1 / self.max_weight - 1e-9)} stocks)")
        if n * self.min_weight > 1 + 1e-9:
            raise ValueError(f"{n} stocks with a {self.min_weight:.0%} minimum exceed 100%: lower the minimum")
        if self.max_sector_weight is not None and sectors is not None:
            if sector_room(sectors, self.max_sector_weight, self.max_weight) < 1 - 1e-9:
                counts = pd.Series(list(sectors)).value_counts()
                raise ValueError(f"the sector cap of {self.max_sector_weight:.0%} cannot be met with these sectors "
                                 f"({', '.join(f'{s}: {c}' for s, c in counts.items())}): add sectors or raise the cap")


def sector_room(sectors: Sequence[str], sector_cap: float, max_weight: float) -> float:
    """The most weight the caps allow in total. A stock with an unknown sector is not counted against any sector cap,
    so it only adds its own per-stock cap."""
    known = [s for s in sectors if s != UNKNOWN_SECTOR]
    counts = pd.Series(known, dtype=object).value_counts()
    return float(sum(min(sector_cap, c * max_weight) for c in counts) + (len(sectors) - len(known)) * max_weight)


@dataclass
class PriceUniverse:
    closes: pd.DataFrame                   # aligned daily closes (common dates), one column per ticker
    frames: Dict[str, pd.DataFrame]        # full OHLCV frame per ticker (for indicators)
    sectors: Dict[str, str]
    benchmark: Optional[pd.Series]         # benchmark closes on the same dates
    excluded: Dict[str, str]               # ticker -> why it was left out
    sources: Dict[str, str]

    @property
    def tickers(self) -> List[str]:
        return list(self.closes.columns)

    @property
    def returns(self) -> pd.DataFrame:
        return self.closes.pct_change().iloc[1:]


def load_universe(tickers: Iterable[str], years: float = 10, source: str = "auto", benchmark: Optional[str] = "^GSPC",
                  min_days: int = 260, stale_days: int = 10) -> PriceUniverse:
    """Load and align prices. Tickers that fail, are too short, or whose data ends well before the others' are left
    out (and reported), so one stale series cannot cut everyone's history short."""
    frames, excluded, sources = {}, {}, {}
    loaded, errors = map_tickers(lambda t: load_prices(t, years=years, source=source), tickers, source=source)
    excluded.update(errors)                  # one bad ticker must not kill the portfolio
    for t, df in loaded.items():
        if len(df) < min_days:
            excluded[t] = f"only {len(df)} trading days"
            continue
        frames[t], sources[t] = df, str(df.attrs.get("source"))
    if not frames:
        raise DataUnavailable("no ticker in the basket could be loaded: " + "; ".join(f"{k} ({v})" for k, v in excluded.items()))
    latest = max(df.index[-1] for df in frames.values())
    for t in list(frames):
        if frames[t].index[-1] < latest - pd.Timedelta(days=stale_days):
            excluded[t] = f"data ends {frames[t].index[-1].date()}, before the others ({latest.date()})"
            del frames[t]
    closes = pd.concat({t: df["Close"] for t, df in frames.items()}, axis=1, join="inner").dropna()
    if len(closes) < min_days:
        raise DataUnavailable(f"the stocks share only {len(closes)} common trading days; need {min_days}")
    bench = None
    if benchmark:
        try:
            b = load_prices(benchmark, years=None, start=closes.index[0], end=closes.index[-1], source=source)["Close"]
            bench = b.reindex(closes.index.union(b.index)).ffill().reindex(closes.index)
        except DataUnavailable:
            bench = None
    return PriceUniverse(closes=closes, frames=frames, sectors=sectors_for(closes.columns, source), benchmark=bench,
                         excluded=excluded, sources=sources)


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def ledoit_wolf(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage covariance toward mu * I. `returns` is T x N. Returns (covariance, intensity)."""
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    mu = np.trace(s) / n
    target = mu * np.eye(n)
    d2 = np.sum((s - target) ** 2) / n
    row_sq = np.sum(xc ** 2, axis=1)
    b2_bar = (np.sum(row_sq ** 2) - t * np.sum(s ** 2)) / (t ** 2) / n
    b2 = min(b2_bar, d2)
    shrink = float(b2 / d2) if d2 > 0 else 1.0
    return shrink * target + (1.0 - shrink) * s, shrink


@dataclass
class Estimates:
    cov: np.ndarray            # annualised covariance (Ledoit-Wolf)
    vol: np.ndarray            # annualised volatility per stock
    mu_hist: np.ndarray        # annualised historical mean return (arithmetic): reported, never optimised on
    shrinkage: float


def estimate(window: pd.DataFrame) -> Estimates:
    x = window.to_numpy(dtype=float)
    cov_d, shrink = ledoit_wolf(x)
    cov = cov_d * TRADING_DAYS
    return Estimates(cov=cov, vol=np.sqrt(np.diag(cov)), mu_hist=x.mean(axis=0) * TRADING_DAYS, shrinkage=shrink)


# --------------------------------------------------------------------------- #
# Optimisers
# --------------------------------------------------------------------------- #
def _sector_matrix(sectors: Optional[Sequence[str]], n: int) -> Tuple[List[str], np.ndarray]:
    """Membership matrix (sector x stock) for the sector caps. Stocks with an unknown sector belong to no row."""
    if sectors is None:
        return [], np.zeros((0, n))
    names = sorted(set(sectors) - {UNKNOWN_SECTOR})
    if not names:
        return [], np.zeros((0, n))
    return names, np.array([[1.0 if s == name else 0.0 for s in sectors] for name in names])


def _weight_constraints(n: int, cons: PortfolioConstraints, sectors: Optional[Sequence[str]]) -> list:
    out = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]
    if cons.max_sector_weight is not None and sectors is not None and len(_sector_matrix(sectors, n)[0]):
        _, m = _sector_matrix(sectors, n)
        cap = cons.max_sector_weight
        out.append({"type": "ineq", "fun": lambda w, m=m: cap - m @ w, "jac": lambda w, m=m: -m})
    return out


def _feasible(w: np.ndarray, cons: PortfolioConstraints, sectors: Optional[Sequence[str]], tol: float = 1e-6) -> bool:
    if abs(w.sum() - 1.0) > tol or (w < cons.min_weight - tol).any() or (w > cons.max_weight + tol).any():
        return False
    if cons.max_sector_weight is not None and sectors is not None:
        _, m = _sector_matrix(sectors, len(w))
        if (m @ w > cons.max_sector_weight + tol).any():
            return False
    return True


def _solve(fun, jac, x0: np.ndarray, cons: PortfolioConstraints, sectors: Optional[Sequence[str]]) -> np.ndarray:
    from scipy.optimize import minimize   # imported on first use: scipy adds ~0.25 s to start-up
    n = len(x0)
    res = minimize(fun, x0, jac=jac, method="SLSQP", bounds=[(cons.min_weight, cons.max_weight)] * n,
                   constraints=_weight_constraints(n, cons, sectors), options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    if not _feasible(w, cons, sectors, tol=1e-4):
        raise RuntimeError(f"the optimiser could not satisfy the constraints ({res.message})")
    return w


def project_to_constraints(w0: np.ndarray, cons: PortfolioConstraints, sectors: Optional[Sequence[str]] = None) -> np.ndarray:
    """Closest weights (least squares) that satisfy the caps; returns w0 unchanged when it already does."""
    w0 = np.asarray(w0, dtype=float)
    if _feasible(w0, cons, sectors):
        return w0
    return _solve(lambda w: float(np.sum((w - w0) ** 2)), lambda w: 2.0 * (w - w0), np.full(len(w0), 1.0 / len(w0)), cons, sectors)


def _min_variance(cov: np.ndarray, cons, sectors) -> np.ndarray:
    n = len(cov)
    return _solve(lambda w: float(w @ cov @ w), lambda w: 2.0 * cov @ w, np.full(n, 1.0 / n), cons, sectors)


def _risk_parity(cov: np.ndarray, cons, sectors) -> Tuple[np.ndarray, str]:
    from scipy.optimize import minimize
    n = len(cov)
    # Spinu (2013): the minimiser of 1/2 y'Σy - (1/n) Σ log y gives equal risk contributions after normalising
    fun = lambda y: 0.5 * float(y @ cov @ y) - float(np.sum(np.log(y))) / n          # noqa: E731
    jac = lambda y: cov @ y - 1.0 / (n * y)                                          # noqa: E731
    y0 = 1.0 / np.sqrt(np.diag(cov))
    res = minimize(fun, y0 / y0.sum(), jac=jac, method="L-BFGS-B", bounds=[(1e-12, None)] * n, options={"maxiter": 2000})
    w = res.x / res.x.sum()
    if _feasible(w, cons, sectors):
        return w, ""

    # caps bind: equalise risk contributions as far as the caps allow
    def rc_err(w):
        port = float(w @ cov @ w)
        rc = w * (cov @ w) / port
        return float(np.sum((rc - 1.0 / n) ** 2))
    w = _solve(rc_err, None, project_to_constraints(w, cons, sectors), cons, sectors)
    return w, "The weight caps bind, so risk contributions are equalised as far as the caps allow."


def optimize(window: pd.DataFrame, method: str, constraints: Optional[PortfolioConstraints] = None,
             sectors: Optional[Sequence[str]] = None, risk_free_rate: float = 0.02,
             est: Optional[Estimates] = None) -> Tuple[pd.Series, List[str]]:
    """Weights (summing to 1, before any volatility-target cash) for the stocks in `window` (daily returns)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    cons = constraints or PortfolioConstraints()
    tickers = list(window.columns)
    cons.check_feasible(tickers, sectors)
    est = est or estimate(window)
    notes: List[str] = []
    n = len(tickers)
    if method == "equal_weight":
        w = project_to_constraints(np.full(n, 1.0 / n), cons, sectors)
    elif method == "min_variance":
        w = _min_variance(est.cov, cons, sectors)
    else:
        w, note = _risk_parity(est.cov, cons, sectors)
        notes += [note] if note else []
    w = np.where(w < 1e-6, 0.0, w)
    return pd.Series(w / w.sum(), index=tickers, name="weight"), notes


def vol_target_scale(weights: np.ndarray, cov: np.ndarray, target_vol: Optional[float]) -> float:
    """Fraction to invest so the expected volatility does not exceed the target (the rest stays in cash)."""
    if target_vol is None:
        return 1.0
    vol = float(np.sqrt(weights @ cov @ weights))
    return float(min(1.0, target_vol / vol)) if vol > 0 else 1.0


# --------------------------------------------------------------------------- #
# Risk report
# --------------------------------------------------------------------------- #
def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of portfolio volatility from each stock (sums to 1): w_i (Σw)_i / w'Σw."""
    port = float(weights @ cov @ weights)
    return weights * (cov @ weights) / port if port > 0 else np.zeros_like(weights)


def risk_report(weights: pd.Series, returns: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None,
                capital: float = 0.0, risk_free_rate: float = 0.02, est: Optional[Estimates] = None) -> Dict[str, object]:
    """Risk of holding `weights` over the `returns` window. Weights may sum to less than 1: the rest is cash at 0%.

    Forward-looking volatility uses the shrunk covariance. Return, VaR / expected shortfall, drawdown, beta and the
    stress tests are historical simulations of today's weights held constant (rebalanced daily) through the window.
    """
    w = weights.reindex(returns.columns).fillna(0.0).to_numpy(dtype=float)
    invested = float(w.sum())
    w_norm = w / invested if invested > 0 else w
    est = est or estimate(returns)
    rp = pd.Series(returns.to_numpy(dtype=float) @ w, index=returns.index)
    eq = (1.0 + rp).cumprod()
    vol_exp = float(np.sqrt(w @ est.cov @ w))
    q = rp.quantile(0.05)
    r21 = np.expm1(np.log1p(rp).rolling(21).sum()).dropna()      # overlapping 21-session (about 1 month) returns
    q21 = r21.quantile(0.05) if len(r21) else float("nan")
    sd = np.sqrt(np.diag(est.cov))
    m = compute_metrics(eq, risk_free_rate, TRADING_DAYS)
    vol_norm = float(np.sqrt(w_norm @ est.cov @ w_norm))
    out: Dict[str, object] = {
        "expected_vol": vol_exp,
        "hist_return": float(rp.mean() * TRADING_DAYS),
        "hist_vol": m["volatility"],
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "max_drawdown": float(max_drawdown(eq)),
        "var95_1d": float(-q),
        "es95_1d": float(-rp[rp <= q].mean()),
        "var95_21d": float(-q21),
        "es95_21d": float(-r21[r21 <= q21].mean()) if len(r21) else float("nan"),
        "diversification_ratio": float(w_norm @ sd / vol_norm) if vol_norm > 0 else float("nan"),
        "effective_n": float(1.0 / np.sum(w_norm ** 2)) if np.sum(w_norm ** 2) > 0 else float("nan"),
        "cov_shrinkage": est.shrinkage,
        "window_start": returns.index[0], "window_end": returns.index[-1], "days": len(returns),
        "cash_weight": 1.0 - invested,
    }
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(returns.index).fillna(0.0)
        var_b = float(b.var())
        out["beta"] = float(np.cov(rp, b)[0, 1] / var_b) if var_b > 0 else float("nan")
        out["correlation_to_benchmark"] = float(np.corrcoef(rp, b)[0, 1])
        out["tracking_error"] = float((rp - b).std() * np.sqrt(TRADING_DAYS))
    if capital:
        for k in ("var95_1d", "es95_1d", "var95_21d", "es95_21d"):
            out[k + "_usd"] = out[k] * capital
    out["risk_contributions"] = pd.Series(risk_contributions(w, est.cov), index=returns.columns, name="risk share")
    out["correlation"] = returns.corr()
    stress = []
    bench_eq = None if benchmark_returns is None else (1.0 + benchmark_returns.reindex(returns.index).fillna(0.0)).cumprod()
    for name, s_, e_ in STRESS_WINDOWS:
        s_, e_ = pd.Timestamp(s_), pd.Timestamp(e_)
        if returns.index[0] > s_ or returns.index[-1] < e_:
            continue
        seg = rp[(rp.index > s_) & (rp.index <= e_)]
        row = {"window": name, "from": s_.date(), "to": e_.date(), "portfolio": float((1 + seg).prod() - 1)}
        if bench_eq is not None:
            row["S&P 500"] = float(bench_eq.asof(e_) / bench_eq.asof(s_) - 1)
        stress.append(row)
    out["stress"] = pd.DataFrame(stress)
    return out


# --------------------------------------------------------------------------- #
# Dollars and shares
# --------------------------------------------------------------------------- #
def to_shares(weights: pd.Series, prices: pd.Series, cash: float) -> Tuple[pd.DataFrame, float]:
    """Whole shares for `cash` (weights may sum to < 1; the rest is a deliberate cash buffer).

    Floors every position, then spends the leftover one share at a time on the position furthest below its target,
    never overshooting the cash. Returns (table, cash left)."""
    w = weights[weights > 0]
    px = prices.reindex(w.index).astype(float)
    target = w * cash
    shares = np.floor(target / px).astype(int)
    left = cash - float((shares * px).sum())
    budget = cash * float(w.sum())             # do not spend the cash buffer
    spent = cash - left
    while True:
        gap = (target - shares * px) / px      # shortfall in shares
        gap = gap[(px <= left + 1e-9) & (px <= budget - spent + 1e-9)]
        if gap.empty or gap.max() < 0.5:
            break
        t = gap.idxmax()
        shares[t] += 1
        left -= float(px[t])
        spent += float(px[t])
    value = shares * px
    table = pd.DataFrame({"target_weight": w, "price": px, "shares": shares, "value": value,
                          "actual_weight": value / cash if cash > 0 else 0.0})
    return table, float(left)


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioBacktest:
    equity: pd.Series
    weights: pd.DataFrame          # risky weights decided at each rebalance (rows = decision dates)
    turnover: pd.Series            # one-way turnover per executed rebalance
    metrics: Dict[str, float]
    label: str


def _rebalance_flags(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    rule = {"monthly": "M", "quarterly": "Q", "weekly": "W"}[freq]
    period = index.to_period(rule)
    flags = np.asarray(period != np.roll(period, -1))
    flags[-1] = False                 # nothing to execute after the last day
    return flags


def backtest_weights(returns: pd.DataFrame, weight_fn: Callable[..., Tuple[np.ndarray, float]], lookback: int = 252,
                     rebalance: str = "monthly", cost_bps: float = 10.0, risk_free_rate: Union[float, pd.Series] = 0.02,
                     label: str = "portfolio", pass_holdings: bool = False, cash_yield: bool = True) -> PortfolioBacktest:
    """Walk forward: on each rebalance day t, `weight_fn(t)` sees returns up to and including day t and returns
    (risky weights, invested fraction); with `pass_holdings` it is called as `weight_fn(t, w)` with the weights held
    at that close (after drift), so a rule can keep what it owns. The trade executes at the NEXT day's close (no
    look-ahead), pays `cost_bps` on every dollar traded (turnover = buys + sells as a share of the portfolio), and the
    weights then drift with prices until the next rebalance. The part not invested (a volatility target's cash
    buffer) earns `risk_free_rate` (the T-bill series day by day, or a fixed rate) unless `cash_yield` is False."""
    r = returns.to_numpy(dtype=float)
    dates = returns.index
    if not cash_yield:
        rf_d = np.zeros(len(dates))
    elif isinstance(risk_free_rate, pd.Series):
        from .backtest import rate_on
        rf_d = rate_on(risk_free_rate, dates).to_numpy() / TRADING_DAYS
    else:
        rf_d = np.full(len(dates), float(risk_free_rate) / TRADING_DAYS)
    t_n, n = r.shape
    flags = _rebalance_flags(dates, rebalance)
    flags[: lookback - 1] = False
    starts = np.where(flags)[0]
    if len(starts) == 0:
        raise ValueError(f"need more than {lookback} days of history for a walk-forward test")
    t0 = int(starts[0])
    cost = cost_bps / 10_000.0
    v, w = 1.0, np.zeros(n)
    pending = None
    eq, w_rows, w_dates, turns, turn_dates = [], [], [], [], []
    for t in range(t0, t_n):
        if t > t0:
            rp = float(w @ r[t]) + max(0.0, 1.0 - float(w.sum())) * rf_d[t]
            v *= 1.0 + rp
            if 1.0 + rp > 0:
                w = w * (1.0 + r[t]) / (1.0 + rp)
        if pending is not None:
            turnover = float(np.abs(pending - w).sum())
            v *= 1.0 - turnover * cost
            w = pending.copy()
            turns.append(turnover)
            turn_dates.append(dates[t])
            pending = None
        if flags[t]:
            wt, invested = weight_fn(t, w.copy()) if pass_holdings else weight_fn(t)
            pending = np.asarray(wt, dtype=float) * invested
            w_rows.append(pending.copy())
            w_dates.append(dates[t])
        eq.append(v)
    equity = pd.Series(eq, index=dates[t0:], name=label)
    m = compute_metrics(equity, risk_free_rate, TRADING_DAYS)
    tv = pd.Series(turns, index=turn_dates, name="turnover", dtype=float)
    m["turnover_annual"] = float(tv.sum() / m["years"]) if m.get("years") else float("nan")
    return PortfolioBacktest(equity=equity, weights=pd.DataFrame(w_rows, index=w_dates, columns=returns.columns),
                             turnover=tv, metrics=m, label=label)


def _metrics_rf(risk_free: Optional[pd.Series], risk_free_rate: float):
    """The rate behind walk-forward Sharpe ratios: the daily T-bill when given (rates moved from ~0% to 5% over the
    last decade), else the fixed rate. The optimiser's maximum-Sharpe method always uses the fixed (today's) rate."""
    return risk_free if risk_free is not None and len(risk_free) else risk_free_rate


def method_weight_fn(returns: pd.DataFrame, method: str, cons: PortfolioConstraints, sectors: Optional[Sequence[str]],
                     lookback: int, risk_free_rate: float) -> Callable[[int], Tuple[np.ndarray, float]]:
    def fn(t: int):
        window = returns.iloc[t - lookback + 1: t + 1]
        est = estimate(window)
        w, _ = optimize(window, method, cons, sectors, risk_free_rate, est=est)
        return w.to_numpy(), vol_target_scale(w.to_numpy(), est.cov, cons.target_vol)
    return fn


def compare_methods(universe: PriceUniverse, methods: Optional[Sequence[str]] = None,
                    constraints: Optional[PortfolioConstraints] = None, lookback: int = 252, rebalance: str = "monthly",
                    cost_bps: float = 10.0, risk_free_rate: float = 0.02,
                    risk_free: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Dict[str, PortfolioBacktest]]:
    """Walk-forward every allocation method on the same basket: the evidence for choosing one."""
    rets = universe.returns
    cons = constraints or PortfolioConstraints()
    sectors = [universe.sectors[t] for t in rets.columns] if cons.max_sector_weight is not None else None
    runs: Dict[str, PortfolioBacktest] = {}
    for m in methods or list(METHODS):
        runs[m] = backtest_weights(rets, method_weight_fn(rets, m, cons, sectors, lookback, risk_free_rate), lookback,
                                   rebalance, cost_bps, _metrics_rf(risk_free, risk_free_rate), METHODS[m])
    rows = {METHODS[m]: bt.metrics for m, bt in runs.items()}
    if universe.benchmark is not None and runs:
        first = next(iter(runs.values())).equity.index
        b = universe.benchmark.reindex(first)
        rows["S&P 500 (buy & hold)"] = compute_metrics(b / b.iloc[0], _metrics_rf(risk_free, risk_free_rate), TRADING_DAYS)
        runs["benchmark"] = PortfolioBacktest(b / b.iloc[0], pd.DataFrame(), pd.Series(dtype=float), rows["S&P 500 (buy & hold)"], "S&P 500")
    table = pd.DataFrame(rows).T
    cols = ["cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "turnover_annual", "years"]
    return table.reindex(columns=[c for c in cols if c in table.columns]), runs


# --------------------------------------------------------------------------- #
# Goal-based selection
# --------------------------------------------------------------------------- #
@dataclass
class Goal:
    key: str
    label: str
    method: str
    rank_by: str               # "low_vol" | "momentum" | "technical"
    description: str
    n_stocks: int = 10
    max_weight: float = 0.20
    max_sector_weight: Optional[float] = 0.40
    target_vol: Optional[float] = None
    require_uptrend: bool = False
    min_technical: Optional[float] = None   # skip stocks the engine rates bearish (technical score below this)
    # Rank buffer: a stock already held is kept while it still passes the goal's filters and ranks within the top
    # keep_rank x n_stocks (inf = for as long as it qualifies); only the freed places are re-filled by rank. None:
    # re-select from scratch at every review. Without it the fast-moving technical and momentum ranks replaced most of
    # the portfolio every month (balanced: 18.5x a year on the large caps); see docs/METHODOLOGY.md section 10.
    keep_rank: Optional[float] = None


GOALS: Dict[str, Goal] = {
    "preservation": Goal("preservation", "Capital preservation", "min_variance", "low_vol",
                         "The calmest stocks, combined for the lowest volatility. For money you cannot afford to see fall far.",
                         n_stocks=10, max_weight=0.15, max_sector_weight=0.35, target_vol=0.12),
    "balanced": Goal("balanced", "Balanced risk and reward", "risk_parity", "technical",
                     "The stocks with the strongest Lighthouse technical scores, with risk spread evenly so no single stock "
                     "or sector dominates.", n_stocks=10, max_weight=0.20, max_sector_weight=0.40, min_technical=45.0,
                     keep_rank=float("inf")),
    "growth": Goal("growth", "Growth", "equal_weight", "momentum",
                   "The strongest 12-month momentum among stocks in an up-trend, in equal weights (no estimate of their "
                   "returns, the input that misleads optimisers most). Higher volatility.",
                   n_stocks=8, max_weight=0.25, max_sector_weight=0.50, require_uptrend=True, keep_rank=float("inf")),
    "diversified": Goal("diversified", "Maximum diversification", "risk_parity", "technical",
                        "Many stocks across sectors (15, at most 15% each and 30% per sector), with risk spread evenly.", n_stocks=15, max_weight=0.15, max_sector_weight=0.30, min_technical=45.0,
                        keep_rank=float("inf")),
}


def selection_panels(universe: PriceUniverse, params: Optional[StrategyParams] = None) -> Dict[str, pd.DataFrame]:
    """Point-in-time ranking inputs for every date (all causal): technical score, 12-1 momentum, 1-year volatility,
    and the up-trend flag (close above the 200-day SMA)."""
    p = params or StrategyParams()
    idx = universe.closes.index
    tech, trend = {}, {}
    for t in universe.tickers:
        ind = compute_indicators(universe.frames[t], p)
        tech[t] = technical_score(ind, "long", p).reindex(idx)
        trend[t] = (ind["Close"] > ind["sma_slow"]).astype(float).where(ind["sma_slow"].notna()).reindex(idx)
    c = universe.closes
    return {
        "technical": pd.DataFrame(tech),
        "momentum": c.shift(21) / c.shift(252) - 1.0,                # 12-1 month momentum (Jegadeesh & Titman 1993)
        "low_vol": -np.log(c).diff().rolling(252).std() * np.sqrt(TRADING_DAYS),   # higher = calmer
        "uptrend": pd.DataFrame(trend),
    }


def select(panels: Dict[str, pd.DataFrame], t: int, goal: Goal, sectors: Dict[str, str], max_per_sector: int = 3,
           tickers: Optional[Sequence[str]] = None, held: Optional[Iterable[str]] = None) -> List[str]:
    """Pick the goal's stocks as of row t: rank, optional up-trend filter, at most `max_per_sector` per sector.
    Stocks in `held` that still qualify and sit inside the goal's rank buffer (`Goal.keep_rank`) are kept first."""
    score = panels[goal.rank_by].iloc[t]
    if tickers is not None:
        score = score.reindex(tickers)
    score = score.dropna()
    if goal.require_uptrend:
        up = panels["uptrend"].iloc[t].reindex(score.index)
        score = score[up == 1.0]
    if goal.min_technical is not None:
        tech = panels["technical"].iloc[t].reindex(score.index)
        score = score[tech >= goal.min_technical]
    order = list(score.sort_values(ascending=False).index)
    keep: List[str] = []
    if held is not None and goal.keep_rank is not None:
        rank = {tk: i for i, tk in enumerate(order)}
        keep = sorted((tk for tk in set(held) if tk in rank and rank[tk] < goal.keep_rank * goal.n_stocks), key=rank.get)
    picked, per = [], {}
    for tk in keep + [tk for tk in order if tk not in keep]:
        s = sectors.get(tk, UNKNOWN_SECTOR)
        if s != UNKNOWN_SECTOR and per.get(s, 0) >= max_per_sector:
            continue                     # an unknown sector is not a sector: it never counts against the limit
        picked.append(tk)
        per[s] = per.get(s, 0) + 1
        if len(picked) >= goal.n_stocks:
            break
    return picked


def _goal_constraints(goal: Goal, picked: Sequence[str], sectors: Dict[str, str], cons: PortfolioConstraints) -> PortfolioConstraints:
    """Relax the caps just enough to be feasible for a short pick list (e.g. early in a backtest)."""
    n = max(len(picked), 1)
    max_w = max(cons.max_weight, 1.0 / n)
    sec = cons.max_sector_weight
    if sec is not None and sector_room([sectors.get(t, UNKNOWN_SECTOR) for t in picked], sec, max_w) < 1 - 1e-9:
        sec = None
    return PortfolioConstraints(max_weight=max_w, min_weight=0.0, max_sector_weight=sec, target_vol=cons.target_vol)


# --------------------------------------------------------------------------- #
# One-call builders
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioResult:
    method: str
    label: str
    weights: pd.Series                 # risky weights (sum to the invested fraction; the rest is cash)
    cash_weight: float
    allocation: pd.DataFrame           # target weight, price, shares, value, actual weight per stock
    cash_left: float                   # uninvested dollars after buying whole shares (includes any cash buffer)
    capital: float
    risk: Dict[str, object]
    estimates: Estimates
    universe: PriceUniverse
    as_of: str
    notes: List[str] = field(default_factory=list)
    backtest: Optional[PortfolioBacktest] = None
    comparison: Optional[pd.DataFrame] = None
    comparison_runs: Dict[str, PortfolioBacktest] = field(default_factory=dict)
    selection: Optional[pd.DataFrame] = None   # goal-based: why each stock was picked

    def table(self) -> pd.DataFrame:
        a = self.allocation.copy()
        a["sector"] = [self.universe.sectors.get(t, UNKNOWN_SECTOR) for t in a.index]
        a["risk_share"] = self.risk["risk_contributions"].reindex(a.index)
        a["volatility"] = pd.Series(self.estimates.vol, index=self.weights.index).reindex(a.index)
        return a


def _finish(universe: PriceUniverse, picked: List[str], method: str, cons: PortfolioConstraints, capital: float,
            lookback: int, risk_free_rate: float, notes: List[str]) -> PortfolioResult:
    rets = universe.returns[picked]
    window = rets.iloc[-lookback:]
    sectors = [universe.sectors[t] for t in picked] if cons.max_sector_weight is not None else None
    est = estimate(window)
    w, more = optimize(window, method, cons, sectors, risk_free_rate, est=est)
    notes = notes + more
    unfinished = sorted({(universe.frames[t].attrs.get("partial_bar") or {}).get("date") for t in picked} - {None})
    if unfinished:
        notes.append(f"Orders are sized from the {universe.closes.index[-1].date()} closes: today's session "
                     f"({unfinished[0]}) is still trading, so check each live price before you place the orders.")
    unknown = [t for t in picked if universe.sectors.get(t, UNKNOWN_SECTOR) == UNKNOWN_SECTOR]
    if unknown:
        notes.append(f"No sector found for {', '.join(unknown)} (not in the curated list and none in the fundamentals, "
                     "e.g. an ETF): not counted against the per-sector limits.")
    invested = vol_target_scale(w.to_numpy(), est.cov, cons.target_vol)
    if invested < 1.0:
        notes.append(f"Volatility target {cons.target_vol:.0%}: the stocks' mix is expected to run at "
                     f"{float(np.sqrt(w.to_numpy() @ est.cov @ w.to_numpy())):.1%}, so {1 - invested:.0%} is held in cash.")
    risky = w * invested
    alloc, left = to_shares(risky, universe.closes[picked].iloc[-1], capital) if capital > 0 else (
        pd.DataFrame({"target_weight": risky, "price": universe.closes[picked].iloc[-1], "shares": 0, "value": 0.0,
                      "actual_weight": 0.0}), 0.0)
    if capital > 0 and (alloc["shares"] == 0).any():
        zero = list(alloc.index[alloc["shares"] == 0])
        notes.append(f"${capital:,.0f} is too little to buy a whole share of {', '.join(zero)} at its target weight.")
    bench_r = universe.benchmark.pct_change().reindex(window.index) if universe.benchmark is not None else None
    risk = risk_report(risky, window, bench_r, capital, risk_free_rate, est)
    return PortfolioResult(method=method, label=METHODS[method], weights=risky, cash_weight=1.0 - invested, allocation=alloc,
                           cash_left=left, capital=capital, risk=risk, estimates=est, universe=universe,
                           as_of=str(universe.closes.index[-1].date()), notes=notes)


def build_portfolio(tickers: Iterable[str], method: str = "risk_parity", constraints: Optional[PortfolioConstraints] = None,
                    capital: float = 10_000, years: float = 10, source: str = "auto", lookback: int = 252,
                    rebalance: str = "monthly", cost_bps: float = 10.0, risk_free_rate: float = 0.02,
                    run_backtest: bool = True, compare: bool = True, universe: Optional[PriceUniverse] = None,
                    risk_free: Optional[pd.Series] = None) -> PortfolioResult:
    """Allocate YOUR basket: weights, shares for `capital`, risk report, and a walk-forward test of the method
    (plus every other method for comparison)."""
    cons = constraints or PortfolioConstraints()
    uni = universe or load_universe(tickers, years=years, source=source)
    notes = [f"Left out {t}: {why}" for t, why in uni.excluded.items()]
    if len(uni.returns) < lookback + 21:
        lookback = max(63, len(uni.returns) - 21)
        notes.append(f"Short common history: covariance estimated on {lookback} days.")
    res = _finish(uni, uni.tickers, method, cons, capital, lookback, risk_free_rate, notes)
    if run_backtest:
        try:
            if compare:
                res.comparison, res.comparison_runs = compare_methods(uni, None, cons, lookback, rebalance, cost_bps, risk_free_rate,
                                                                      risk_free)
                res.backtest = res.comparison_runs[method]
            else:
                sectors = [uni.sectors[t] for t in uni.tickers] if cons.max_sector_weight is not None else None
                res.backtest = backtest_weights(uni.returns, method_weight_fn(uni.returns, method, cons, sectors, lookback, risk_free_rate),
                                                lookback, rebalance, cost_bps, _metrics_rf(risk_free, risk_free_rate), METHODS[method])
        except (ValueError, RuntimeError) as exc:
            res.notes.append(f"Walk-forward test skipped: {exc}")
    return res


def goal_portfolio(goal: str, tickers: Optional[Iterable[str]] = None, capital: float = 10_000, years: float = 10,
                   source: str = "auto", n_stocks: Optional[int] = None, constraints: Optional[PortfolioConstraints] = None,
                   max_per_sector: int = 3, lookback: int = 252, rebalance: str = "monthly", cost_bps: float = 10.0,
                   risk_free_rate: float = 0.02, run_backtest: bool = True,
                   universe: Optional[PriceUniverse] = None, held: Optional[Iterable[str]] = None,
                   risk_free: Optional[pd.Series] = None) -> PortfolioResult:
    """Build a portfolio aimed at a goal from a universe ("all available tickers" by default).

    Today's picks use the same point-in-time ranking as the walk-forward test, which re-selects the stocks at every
    rebalance using only data available then - so the test does not benefit from knowing today's winners. `held`: the
    stocks you already own for this goal; they are kept while they still qualify (the goal's rank buffer), as the
    walk-forward test keeps its own holdings.
    """
    from .universe import available_universe
    if goal not in GOALS:
        raise ValueError(f"goal must be one of {list(GOALS)}")
    g = GOALS[goal]
    if n_stocks is not None:
        g = Goal(**{**g.__dict__, "n_stocks": int(n_stocks)})
    cons = constraints or PortfolioConstraints(max_weight=g.max_weight, max_sector_weight=g.max_sector_weight, target_vol=g.target_vol)
    uni = universe or load_universe(tickers or available_universe(source), years=years, source=source)
    panels = selection_panels(uni)
    t_last = len(uni.closes) - 1
    notes = [f"Left out {t}: {why}" for t, why in uni.excluded.items()]
    # A universe chosen for one or two sectors (e.g. "Technology") cannot fill the goal under the per-sector limit:
    # raise the limit just enough. Stocks without a sector are never limited, so they count toward the room.
    labels = [uni.sectors.get(t, UNKNOWN_SECTOR) for t in uni.tickers]
    counts = pd.Series([x for x in labels if x != UNKNOWN_SECTOR], dtype=object).value_counts()
    n_unknown = len(labels) - int(counts.sum())

    def room(limit: int) -> int:
        return int(counts.clip(upper=limit).sum()) + n_unknown
    if len(counts) and room(max_per_sector) < g.n_stocks and room(int(counts.max())) > room(max_per_sector):
        raised = max_per_sector
        while room(raised) < g.n_stocks and raised < int(counts.max()):
            raised += 1
        extra = f" (plus {n_unknown} without one)" if n_unknown else ""
        notes.append(f"This universe has {len(counts)} sector{'s' if len(counts) > 1 else ''}{extra}: too few for "
                     f"{g.n_stocks} stocks at {max_per_sector} per sector, so up to {raised} stocks per sector are allowed.")
        max_per_sector = raised
    held = [t.upper().strip() for t in held] if held is not None else None
    picked = select(panels, t_last, g, uni.sectors, max_per_sector, held=held)
    if held:
        kept = [t for t in picked if t in held]
        dropped = [t for t in held if t not in picked]
        notes.append(f"Kept {len(kept)} of the {len(held)} stocks you hold" + (f" ({', '.join(kept)})" if kept else "") +
                     (f"; {', '.join(dropped)} no longer qualif{'ies' if len(dropped) == 1 else 'y'} (or left the "
                      "universe): sell" if dropped else "") + ".")
    if len(picked) < 2:
        raise ValueError(f"only {len(picked)} stock(s) qualify for the '{g.label}' goal right now; widen the universe")
    if len(picked) < g.n_stocks:
        notes.append(f"Only {len(picked)} stocks qualified (asked for {g.n_stocks}).")
    cons_now = _goal_constraints(g, picked, uni.sectors, cons)
    if cons_now.max_weight > cons.max_weight + 1e-9:
        notes.append(f"The per-stock cap was raised to {cons_now.max_weight:.0%} so the weights can add up to 100% with "
                     f"{len(picked)} stocks.")
    if cons.max_sector_weight and cons_now.max_sector_weight is None:
        notes.append(f"The {cons.max_sector_weight:.0%} sector cap was dropped: these stocks' sectors cannot hold 100% "
                     "under it.")
    res = _finish(uni, picked, g.method, cons_now, capital, min(lookback, len(uni.returns)), risk_free_rate, notes)
    row = {k: panels[k].iloc[t_last] for k in ("technical", "momentum", "low_vol", "uptrend")}
    sel = pd.DataFrame(row).loc[picked]
    sel["low_vol"] = -sel["low_vol"]
    res.selection = sel.rename(columns={"technical": "technical score", "momentum": "12-1 momentum",
                                        "low_vol": "1y volatility", "uptrend": "above 200-day"})
    res.label = f"{g.label} · {METHODS[g.method]}"
    res.notes.insert(0, g.description)
    if run_backtest:
        rets = uni.returns
        offset = 1                       # returns start one row after closes
        names = list(rets.columns)

        def weight_fn(goal: Goal):
            def fn(t: int, w_now: np.ndarray):
                held_now = [names[i] for i in np.flatnonzero(w_now > 1e-9)]
                pk = select(panels, t + offset, goal, uni.sectors, max_per_sector, held=held_now)
                w_full = np.zeros(len(names))
                if len(pk) < 2:
                    return w_full, 0.0       # not enough qualifying stocks: stay in cash
                c = _goal_constraints(goal, pk, uni.sectors, cons)
                window = rets[pk].iloc[t - lookback + 1: t + 1]
                est = estimate(window)
                w, _ = optimize(window, goal.method, c, [uni.sectors[x] for x in pk] if c.max_sector_weight is not None else None,
                                risk_free_rate, est=est)
                w_full[[names.index(x) for x in pk]] = w.to_numpy()
                return w_full, vol_target_scale(w.to_numpy(), est.cov, c.target_vol)
            return fn

        warm = max(lookback, 253)        # momentum and volatility ranks need a year of history
        every = {"monthly": "month", "quarterly": "quarter", "weekly": "week"}.get(rebalance, rebalance)
        try:
            kept = g.keep_rank is not None
            label = f"{g.label} (reviewed every {every}" + ("; holdings kept while they qualify)" if kept else ")")
            rf_m = _metrics_rf(risk_free, risk_free_rate)
            res.backtest = backtest_weights(rets, weight_fn(g), warm, rebalance, cost_bps, rf_m, label, pass_holdings=True)
            runs = {"goal": res.backtest}
            rows = {res.backtest.label: res.backtest.metrics}
            if kept:                     # the same goal re-selected from scratch at every review: what the buffer saves
                g0 = Goal(**{**g.__dict__, "keep_rank": None})
                scratch = backtest_weights(rets, weight_fn(g0), warm, rebalance, cost_bps, rf_m,
                                           f"{g.label} (re-selected from scratch every {every})", pass_holdings=True)
                runs["goal_from_scratch"] = scratch
                rows[scratch.label] = scratch.metrics
            ew = backtest_weights(rets, lambda t: (np.full(len(rets.columns), 1.0 / len(rets.columns)), 1.0), warm, rebalance,
                                  cost_bps, rf_m, "Equal weight, whole universe")
            runs["equal_weight_universe"] = ew
            rows[ew.label] = ew.metrics
            if uni.benchmark is not None:
                b = uni.benchmark.reindex(res.backtest.equity.index)
                rows["S&P 500 (buy & hold)"] = compute_metrics(b / b.iloc[0], rf_m, TRADING_DAYS)
                runs["benchmark"] = PortfolioBacktest(b / b.iloc[0], pd.DataFrame(), pd.Series(dtype=float), rows["S&P 500 (buy & hold)"], "S&P 500")
            cols = ["cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "turnover_annual", "years"]
            table = pd.DataFrame(rows).T
            res.comparison = table.reindex(columns=[c for c in cols if c in table.columns])
            res.comparison_runs = runs
        except (ValueError, RuntimeError) as exc:
            res.notes.append(f"Walk-forward test skipped: {exc}")
    return res
