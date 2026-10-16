"""
The cross-sectional research harness: does a signal rank stocks by their next month's return, after costs, out of
sample, beyond known factors, on a universe that was investable at the time?

Protocol (fixed before any result was seen; docs/METHODOLOGY.md section 14)
  universe     point-in-time S&P 500 members at each month end (lighthouse.membership), priced members only;
               coverage (priced / members) is reported with every result
  timing       the signal is read at the last close of month t (only data up to it), the return runs from that close
               to the last close of month t+1 (total return)
  delistings   a member whose prices stop during month t+1 (well before the end of the data) earns its last price
               ("exclude") or -100% ("pessimistic"); every result is computed under both
  tests        monthly rank information coefficient (Spearman) with plain and Newey-West t-statistics; equal-weight
               quintile portfolios; the long-short spread (top minus bottom quintile) and the long-only top quintile
               minus the equal-weight universe; turnover; returns after per-stock costs (commission + Corwin-Schultz half
               spread + square-root impact at a $1m reference portfolio) and, for the short leg, a borrow fee;
               regressions on the Fama-French five factors plus momentum (Newey-West t)
  split        tuning 2009-01 to 2017-12, freely explored; hold-out 2018-01 onwards, evaluated once per signal and
               configuration (a repeat raises unless explicitly allowed, and counts as another trial); the deflated
               Sharpe ratio (Bailey & Lopez de Prado 2014) uses the number of trials in the experiment log
  log          every evaluation appends a row (settings, data version, code commit, results) to
               reports/research/experiments.csv; hold-out evaluations also to reports/research/holdout_log.csv
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..costs import MAX_COST_BPS
from ..data import REPO_DIR
from ..total_return import DELISTING_RULES

RESEARCH_DIR = REPO_DIR / "reports" / "research"
TUNING = ("2009-01-01", "2017-12-31")
HOLDOUT = ("2018-01-01", None)
MIN_NAMES = 30
EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """Month-end cross-sections of a point-in-time universe."""
    dates: pd.DatetimeIndex                     # month-end trading days (the last close of each month)
    prices: pd.DataFrame                        # month-end total-return close per ticker (NaN: no price that month end)
    members: pd.DataFrame                       # bool: a member at that month end
    fwd: Dict[str, pd.DataFrame]                # rule -> next month's return of each member priced at t (NaN: not in)
    last_price: pd.Series                       # ticker -> date of its last price
    data_end: pd.Timestamp
    universe_name: str = ""
    sources: List[str] = field(default_factory=list)
    member_count: Optional[pd.Series] = None    # ALL members at each month end, priced or not (the coverage denominator)
    tradable: Optional[pd.DataFrame] = None     # bool: traded on at least MIN_TRADED_DAYS of the last 21 sessions

    @property
    def investable(self) -> pd.DataFrame:
        inv = self.members & self.prices.notna()
        return inv & self.tradable.reindex_like(inv).fillna(False).astype(bool) if self.tradable is not None else inv

    def members_on(self, d) -> int:
        """Every member at month end d, including those without any price (they cannot be tested)."""
        if self.member_count is not None:
            return int(self.member_count.get(d, 0))
        return int(self.members.loc[d].sum())

    def coverage(self) -> pd.Series:
        m = self.member_count if self.member_count is not None else self.members.sum(axis=1)
        return (self.investable.sum(axis=1) / m.replace(0, np.nan)).rename("coverage")


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each month present in `index`."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


MIN_TRADED_DAYS = 15          # of the last 21 sessions: a series with no volume is stale or the wrong security


def traded_days(frames: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, window: int = 21) -> pd.DataFrame:
    """Sessions with positive volume among the last `window`, at each month end (all of them when a feed has no
    volume column at all, e.g. an index)."""
    out = {}
    for t, df in frames.items():
        if "Volume" not in df or df["Volume"].isna().all():
            out[t] = pd.Series(float(window), index=dates)
            continue
        n = (df["Volume"].fillna(0) > 0).astype(float).rolling(window, min_periods=1).sum()
        out[t] = n.reindex(n.index.union(dates)).ffill(limit=5).reindex(dates)
    return pd.DataFrame(out, index=dates)


def build_panel(universe, frames: Dict[str, pd.DataFrame], start="2008-01-01", end=None,
                calendar: Optional[pd.DatetimeIndex] = None) -> Panel:
    """Month-end prices, membership and forward returns under both delisting rules. A member is tested only in the
    months it traded on at least MIN_TRADED_DAYS of the last 21 sessions (`Panel.tradable`): free price feeds carry
    stale series and other companies' data under reused tickers, and both show up as days without volume."""
    closes = {t: df["Close"] for t, df in frames.items() if len(df)}
    cal = calendar if calendar is not None else pd.DatetimeIndex(sorted(set().union(*[c.index for c in closes.values()])))
    cal = cal[(cal >= pd.Timestamp(start)) & ((cal <= pd.Timestamp(end)) if end is not None else True)]
    dates = month_ends(cal)
    data_end = cal[-1]
    tickers = sorted(set(universe.intervals["ticker"]) & set(closes)) if universe is not None else sorted(closes)
    px = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    last = {}
    for t in tickers:
        c = closes[t].dropna()
        last[t] = c.index[-1]
        # the close on the month-end day, or the last close within the 5 sessions before it (a halt or holiday)
        a = c.reindex(c.index.union(dates)).ffill(limit=5).reindex(dates)
        a[dates > c.index[-1]] = np.nan
        a[dates < c.index[0]] = np.nan
        px[t] = a
    if universe is not None:
        full = universe.membership_matrix(dates)
        member_count = full.sum(axis=1)
        mem = full.reindex(columns=tickers, fill_value=False)
    else:
        mem = pd.DataFrame(True, index=dates, columns=tickers)
        member_count = mem.sum(axis=1)
    last_price = pd.Series(last)
    # the month t during whose next month each stock stopped trading (its last price after month end t, on or before
    # month end t+1), when that is well before the end of the data: a delisting (or a merger, or a ticker change)
    died = pd.DataFrame(False, index=dates, columns=tickers)
    last_close = {}
    for t in tickers:
        lp = last_price[t]
        last_close[t] = float(closes[t].dropna().iloc[-1])
        k = int(dates.searchsorted(lp, side="right")) - 1
        if 0 <= k < len(dates) - 1 and lp < data_end - pd.Timedelta(days=10) and pd.notna(px.iloc[k][t]):
            died.iloc[k, died.columns.get_loc(t)] = True
    nxt = px.shift(-1)
    fwd = {}
    for rule in DELISTING_RULES:
        r = nxt / px - 1.0
        if rule == "pessimistic":
            r = r.mask(died, -1.0)
        else:
            r = r.mask(died, pd.DataFrame({t: last_close[t] / px[t] - 1.0 for t in tickers}, index=dates))
        r = r.where(px.notna() & mem)
        r.iloc[-1] = np.nan                      # no next month after the last month end
        fwd[rule] = r
    tradable = traded_days({t: frames[t] for t in tickers}, dates) >= MIN_TRADED_DAYS
    return Panel(dates=dates, prices=px, members=mem, fwd=fwd, last_price=last_price, data_end=data_end,
                 universe_name=getattr(universe, "name", ""), sources=list(getattr(universe, "sources", [])),
                 member_count=member_count, tradable=tradable.reindex(columns=tickers, fill_value=False))


def sample_at_month_ends(daily: Dict[str, pd.Series], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """A daily signal per ticker read at each month end (the last value on or before it, within 5 sessions)."""
    out = {}
    for t, s in daily.items():
        s = s.dropna()
        if s.empty:
            continue
        a = s.reindex(s.index.union(dates)).ffill(limit=5).reindex(dates)
        a[dates > s.index[-1] + pd.Timedelta(days=7)] = np.nan
        out[t] = a
    return pd.DataFrame(out, index=dates)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def newey_west_se(x: np.ndarray, lags: int = 3) -> float:
    """Newey-West (Bartlett) standard error of the mean of `x`."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lags + 1.0)) * (e[k:] @ e[:-k]) / n
    return float(math.sqrt(max(s, 0.0) / n))


def t_stats(x) -> Tuple[float, float, float]:
    """(mean, plain t, Newey-West t with 3 lags)."""
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    t = m / (sd / math.sqrt(len(x))) if sd > 0 else float("nan")
    nw = newey_west_se(x)
    return m, t, (m / nw if nw and nw > 0 else float("nan"))


def ols_nw(y: pd.Series, X: pd.DataFrame, lags: int = 3) -> Dict[str, object]:
    """OLS of y on X (an intercept added) with Newey-West standard errors."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < X.shape[1] + 6:
        return {"n": len(d)}
    Y = d["y"].to_numpy()
    Z = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    e = Y - Z @ beta
    n, k = Z.shape
    ZtZi = np.linalg.pinv(Z.T @ Z)
    S = (Z * e[:, None]).T @ (Z * e[:, None])
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = (Z[l:] * e[l:, None]).T @ (Z[:-l] * e[:-l, None])
        S += w * (G + G.T)
    V = ZtZi @ S @ ZtZi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    names = ["alpha"] + list(X.columns)
    r2 = 1.0 - (e @ e) / (((Y - Y.mean()) ** 2).sum()) if len(Y) > 1 else float("nan")
    return {"n": n, "coef": dict(zip(names, beta)), "t": dict(zip(names, beta / np.where(se > 0, se, np.nan))),
            "r2": float(r2)}


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, sr_std: float) -> float:
    """Probability that the true Sharpe ratio exceeds the best one expected from `n_trials` unskilled tries (Bailey &
    Lopez de Prado 2014). `sr` and `sr_std` (the standard deviation of the trials' Sharpe ratios) are per period;
    `kurt` is the plain (not excess) kurtosis."""
    from scipy.stats import norm
    if not (np.isfinite(sr) and n_obs > 2):
        return float("nan")
    n_trials = max(int(n_trials), 1)
    if n_trials > 1 and np.isfinite(sr_std) and sr_std > 0:
        sr0 = sr_std * ((1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e)))
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    return float(norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denom))


# --------------------------------------------------------------------------- #
# Costs per stock and month
# --------------------------------------------------------------------------- #
MAX_COST = MAX_COST_BPS / 1e4    # per side: anything above is a data error (no volume), not a price anyone pays


def cost_panel(frames: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex, portfolio_value: float = 1e6,
               names_per_leg: int = 100, commission_bps: float = 5.0, floor_bps: float = 5.0,
               fixed_bps: Optional[float] = None) -> pd.DataFrame:
    """Per-side cost (as a fraction) of trading each stock at each month end: commission + Corwin-Schultz half spread
    (floored) + square-root impact for an order of portfolio_value / names_per_leg (a $1m portfolio of 100 names per
    leg: $10,000 orders), capped at MAX_COST. From daily highs and lows the Corwin-Schultz estimator reads part of the
    day's volatility as spread: about 35 bps for AAPL or MSFT in 2009-2017 against a quoted spread of 1-3 bps, so these
    costs are conservative for large companies. `fixed_bps` replaces the model with one cost for every stock (the
    sensitivity check)."""
    if fixed_bps is not None:
        return pd.DataFrame(fixed_bps / 1e4, index=dates, columns=list(frames))
    from ..costs import corwin_schultz_spread, dollar_volume, impact_bps
    order = portfolio_value / max(names_per_leg, 1)
    out = {}
    for t, df in frames.items():
        half = (corwin_schultz_spread(df["High"], df["Low"]) * 1e4 / 2.0).fillna(0.0).clip(lower=floor_bps)
        vol = np.log(df["Close"] / df["Close"].shift(1)).rolling(21, min_periods=5).std()
        adv = dollar_volume(df["Close"], df["Volume"] if "Volume" in df else None)
        c = (commission_bps + half + pd.Series(impact_bps(order, adv.to_numpy(), vol.to_numpy()), index=df.index)) / 1e4
        out[t] = c.reindex(c.index.union(dates)).ffill(limit=5).reindex(dates).clip(upper=MAX_COST)
    return pd.DataFrame(out, index=dates)


# --------------------------------------------------------------------------- #
# One signal
# --------------------------------------------------------------------------- #
@dataclass
class SignalResult:
    name: str
    rule: str
    period: Tuple[str, Optional[str]]
    ic: pd.Series
    quintiles: pd.DataFrame          # monthly gross return of each quintile (1 = lowest signal)
    spread: pd.Series                # Q5 - Q1, gross
    spread_net: pd.Series            # after costs and the short leg's borrow fee
    long_only: pd.Series             # Q5 - equal-weight universe, gross
    long_only_net: pd.Series
    turnover: pd.Series              # one-way turnover of the top quintile per month
    coverage: pd.Series
    regression: Dict[str, object]
    summary: Dict[str, object]


def _qbins(x: pd.Series, q: int = 5) -> pd.Series:
    r = x.rank(method="first")
    return pd.qcut(r, q, labels=False) + 1


def evaluate(panel: Panel, signal: pd.DataFrame, name: str, rule: str = "exclude",
             period: Tuple[str, Optional[str]] = TUNING, costs: Optional[pd.DataFrame] = None,
             factors: Optional[pd.DataFrame] = None, borrow_fee: float = 0.005, quantiles: int = 5,
             min_names: int = MIN_NAMES) -> SignalResult:
    """Every statistic of the protocol for one signal (month-end x ticker; higher = expected to do better)."""
    lo, hi = pd.Timestamp(period[0]), (pd.Timestamp(period[1]) if period[1] else panel.dates[-1])
    dates = [d for d in panel.dates if lo <= d <= hi]
    fwd = panel.fwd[rule]
    sig = signal.reindex(index=panel.dates, columns=panel.prices.columns)
    ic, qret, spread, spread_net, lo_only, lo_net, turn, cov = {}, {}, {}, {}, {}, {}, {}, {}
    prev_w: Dict[int, pd.Series] = {}
    for d in dates:
        f = fwd.loc[d]
        s = sig.loc[d]
        ok = f.notna() & s.notna() & panel.members.loc[d]
        if panel.tradable is not None:
            ok &= panel.tradable.loc[d].reindex(ok.index, fill_value=False)
        cov[d] = ok.sum() / max(panel.members_on(d), 1)       # tested names / every member, priced or not
        if ok.sum() < min_names or not np.isfinite(f[ok]).all():
            continue
        fs, ss = f[ok], s[ok]
        ic[d] = float(ss.rank().corr(fs.rank()))
        b = _qbins(ss, quantiles)
        rets = fs.groupby(b).mean()
        qret[d] = rets
        spread[d] = float(rets.get(quantiles, np.nan) - rets.get(1, np.nan))
        lo_only[d] = float(rets.get(quantiles, np.nan) - fs.mean())
        # costs: equal weights in each leg; what is traded = the change in weights since last month (drift ignored)
        leg_cost = {}
        for k in (quantiles, 1):
            names = b[b == k].index
            w = pd.Series(1.0 / len(names), index=names)
            dw = w.subtract(prev_w.get(k, pd.Series(dtype=float)), fill_value=0.0).abs()
            c = costs.loc[d].reindex(dw.index).fillna(0.002) if costs is not None else pd.Series(0.001, index=dw.index)
            leg_cost[k] = float((dw * c).sum())
            if k == quantiles:
                turn[d] = float(dw.sum() / 2.0)
            prev_w[k] = w
        spread_net[d] = spread[d] - leg_cost[quantiles] - leg_cost[1] - borrow_fee / 12.0
        lo_net[d] = lo_only[d] - leg_cost[quantiles]
    ic_s = pd.Series(ic, name="ic")
    q = pd.DataFrame(qret).T.sort_index()
    sp, spn = pd.Series(spread, name="spread"), pd.Series(spread_net, name="spread_net")
    lo_s, lon = pd.Series(lo_only, name="long_only"), pd.Series(lo_net, name="long_only_net")
    tv = pd.Series(turn, name="turnover")
    reg = {}
    if factors is not None and len(spn):
        # the spread dated month end t is the return over month t+1: it is explained by month t+1's factors
        fac = factors.reindex((spn.index.to_period("M") + 1).to_timestamp("M"))
        fac.index = spn.index
        X = fac[[c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in fac]]
        reg = ols_nw(spn, X)
        reg_lo = ols_nw(lon, X)
        reg = {"long_short": reg, "long_only": reg_lo}
    summ = summarize_result(name, rule, period, ic_s, sp, spn, lo_s, lon, tv, pd.Series(cov), reg, q)
    return SignalResult(name, rule, period, ic_s, q, sp, spn, lo_s, lon, tv, pd.Series(cov, name="coverage"), reg, summ)


def _ann(x: pd.Series) -> Tuple[float, float, float]:
    x = x.dropna()
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    m, sd = x.mean() * 12, x.std(ddof=1) * math.sqrt(12)
    return float(m), float(sd), float(m / sd) if sd > 0 else float("nan")


def summarize_result(name, rule, period, ic, sp, spn, lo, lon, tv, cov, reg, q) -> Dict[str, object]:
    ic_m, ic_t, ic_nw = t_stats(ic)
    s_m, s_t, s_nw = t_stats(spn)
    ann, vol, sharpe = _ann(spn)
    l_ann, l_vol, l_sharpe = _ann(lon)
    _, l_t, l_nw = t_stats(lon)
    g_ann, _, g_sharpe = _ann(sp)
    x = spn.dropna()
    out = {"signal": name, "rule": rule, "start": str(period[0]), "end": str(period[1] or "latest"),
           "months": int(len(ic)), "coverage": float(cov.mean()) if len(cov) else float("nan"),
           "ic_mean": ic_m, "ic_t": ic_t, "ic_nw_t": ic_nw, "ic_hit": float((ic > 0).mean()) if len(ic) else float("nan"),
           "spread_gross_ann": g_ann, "spread_gross_sharpe": g_sharpe,
           "spread_net_ann": ann, "spread_net_vol": vol, "spread_net_sharpe": sharpe, "spread_net_nw_t": s_nw,
           "long_only_net_ann": l_ann, "long_only_net_sharpe": l_sharpe, "long_only_net_nw_t": l_nw,
           "top_turnover": float(tv.mean()) if len(tv) else float("nan"),
           "monotonic": _monotonic(q), "skew": float(x.skew()) if len(x) > 3 else float("nan"),
           "kurt": float(x.kurt() + 3.0) if len(x) > 3 else float("nan")}
    ls = (reg or {}).get("long_short") or {}
    if "coef" in ls:
        out.update({"alpha_ann": float(ls["coef"]["alpha"] * 12), "alpha_nw_t": float(ls["t"]["alpha"]),
                    "r2": ls["r2"], "loadings": {k: round(float(v), 3) for k, v in ls["coef"].items() if k != "alpha"}})
    lo_r = (reg or {}).get("long_only") or {}
    if "coef" in lo_r:
        out.update({"long_only_alpha_ann": float(lo_r["coef"]["alpha"] * 12), "long_only_alpha_nw_t": float(lo_r["t"]["alpha"])})
    return out


def _monotonic(q: pd.DataFrame) -> float:
    """Spearman correlation between quintile number and its average return (1 = perfectly ordered)."""
    if q is None or q.empty:
        return float("nan")
    m = q.mean()
    return float(pd.Series(m.index, index=m.index).rank().corr(m.rank()))


# --------------------------------------------------------------------------- #
# The log and the hold-out guard
# --------------------------------------------------------------------------- #
def config_hash(name: str, config: Dict[str, object]) -> str:
    return hashlib.sha1(json.dumps({"name": name, **config}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def signal_config(name: str, description: str = "", **extra) -> Dict[str, object]:
    """The configuration a signal of the research run is logged with (scripts/research_signals.py): one signal with one
    configuration is one trial. `extra` adds what changed a signal's values since its first trial (for example the
    share-count method: fundamental_signals.config_extra), so a corrected signal is a new trial. The forward watch
    list (research/watch.py) records its signals under the same configuration, so its rows carry the config_hash of
    the logged trial."""
    return {"signal": name, "universe": "sp500_pit", "quantiles": 5, "costs": "cs_spread+sqrt_impact@1m/100",
            "borrow": 0.005, "description": description, **extra}


class HoldoutUsed(RuntimeError):
    """The hold-out of this signal and configuration was already evaluated."""


LOG_COLUMNS = ["run_at", "holdout", "period", "delisting_rule", "holdout_id", "config_hash", "signal", "rule", "start",
               "end", "months", "coverage", "ic_mean", "ic_t", "ic_nw_t", "ic_hit", "spread_gross_ann",
               "spread_gross_sharpe", "spread_net_ann", "spread_net_vol", "spread_net_sharpe", "spread_net_nw_t",
               "long_only_net_ann", "long_only_net_sharpe", "long_only_net_nw_t", "top_turnover", "monotonic", "skew",
               "kurt", "alpha_ann", "alpha_nw_t", "r2", "loadings", "long_only_alpha_ann", "long_only_alpha_nw_t",
               "trials", "deflated_sharpe", "long_only_deflated_sharpe", "extra", "config", "data_version", "code"]


class ExperimentLog:
    """Every evaluation, appended (never edited) to experiments.csv; hold-outs also to holdout_log.csv. The columns are
    fixed (LOG_COLUMNS): a statistic added later goes into the `extra` JSON column, so old and new rows always line up."""

    def __init__(self, folder: Optional[Path] = None) -> None:
        self.folder = Path(folder or RESEARCH_DIR)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "experiments.csv"
        self.holdout_path = self.folder / "holdout_log.csv"

    def rows(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def holdouts(self) -> pd.DataFrame:
        return pd.read_csv(self.holdout_path) if self.holdout_path.exists() else pd.DataFrame()

    def trials(self) -> int:
        """Distinct (signal, configuration) pairs tried so far, any period or delisting rule: the multiple-testing
        count for the deflated Sharpe ratio."""
        r = self.rows()
        return int(r["config_hash"].nunique()) if len(r) and "config_hash" in r else 0

    def trial_sharpe_std(self, default: float = 0.1) -> float:
        """The dispersion of monthly net Sharpe ratios across every configuration tried (the deflated Sharpe ratio's
        variance of trials): the latest tuning row of each configuration and delisting rule, both the long-short and
        the long-only Sharpe. Taken from the whole log, so a run that evaluates one signal is judged against every
        trial, not against itself."""
        r = self.rows()
        if r.empty or "period" not in r:
            return default
        t = r[r["period"] == "tuning"].drop_duplicates(["config_hash", "delisting_rule"], keep="last")
        sr = pd.concat([pd.to_numeric(t[c], errors="coerce") for c in ("spread_net_sharpe", "long_only_net_sharpe")
                        if c in t]) / math.sqrt(12)
        sr = sr[np.isfinite(sr)]
        return float(sr.std(ddof=1)) if len(sr) > 2 else default

    def append(self, summary: Dict[str, object], config: Dict[str, object], data_version: Dict[str, object],
               holdout: bool = False, rule: str = "", period: str = "", holdout_id: str = "") -> None:
        """`config` describes the signal (not the rule or period), so one configuration is one trial."""
        from ..journal import code_version
        row = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "holdout": holdout,
               "period": period, "delisting_rule": rule, "holdout_id": holdout_id,
               "config_hash": config_hash(str(summary.get("signal")), config),
               **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in summary.items()},
               "config": json.dumps(config, sort_keys=True, default=str),
               "data_version": json.dumps(data_version, sort_keys=True, default=str), "code": code_version()}
        extra = {k: v for k, v in row.items() if k not in LOG_COLUMNS}
        row = {k: row.get(k) for k in LOG_COLUMNS}
        row["extra"] = json.dumps(extra, sort_keys=True, default=str) if extra else ""
        df = pd.DataFrame([row], columns=LOG_COLUMNS)
        for path in ([self.path, self.holdout_path] if holdout else [self.path]):
            self._check_header(path)
            df.to_csv(path, mode="a", index=False, header=not path.exists())

    @staticmethod
    def _check_header(path: Path) -> None:
        if path.exists():
            with open(path) as fh:
                head = fh.readline().strip().split(",")
            if head != LOG_COLUMNS:
                raise RuntimeError(f"{path} has other columns than LOG_COLUMNS: appending would misalign its rows")

    def check_holdout(self, name: str, config: Dict[str, object], allow_repeat: bool = False) -> int:
        """How many times this exact hold-out was evaluated before; raises HoldoutUsed unless allowed."""
        h = self.holdouts()
        n = int(h.loc[h["config_hash"] == config_hash(name, config), "holdout_id"].nunique()) if len(h) else 0
        if n and not allow_repeat:
            raise HoldoutUsed(f"the hold-out of {name} with this configuration was already evaluated {n} time(s); "
                              "a repeat counts as a new trial (allow_repeat=True)")
        return n


def evaluate_holdout(panel: Panel, signal: pd.DataFrame, name: str, config: Dict[str, object], log: ExperimentLog,
                     data_version: Dict[str, object], costs=None, factors=None, allow_repeat: bool = False,
                     n_trials: Optional[int] = None, sr_std: Optional[float] = None) -> Dict[str, SignalResult]:
    """The hold-out, once, under both delisting rules; logged, with the deflated Sharpe ratio."""
    log.check_holdout(name, config, allow_repeat)
    hid = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%f") + "-" + config_hash(name, config)
    out = {}
    trials = n_trials if n_trials is not None else max(log.trials(), 1)
    for rule in DELISTING_RULES:
        res = evaluate(panel, signal, name, rule, HOLDOUT, costs, factors)
        sr_m = res.spread_net.mean() / res.spread_net.std(ddof=1) if res.spread_net.std(ddof=1) > 0 else float("nan")
        res.summary["trials"] = trials
        res.summary["deflated_sharpe"] = deflated_sharpe(float(sr_m), len(res.spread_net), res.summary["skew"],
                                                         res.summary["kurt"], trials, sr_std if sr_std is not None else 0.1)
        lo = res.long_only_net
        lo_sr = lo.mean() / lo.std(ddof=1) if lo.std(ddof=1) > 0 else float("nan")
        res.summary["long_only_deflated_sharpe"] = deflated_sharpe(float(lo_sr), len(lo), float(lo.skew()),
                                                                   float(lo.kurt() + 3), trials,
                                                                   sr_std if sr_std is not None else 0.1)
        log.append(res.summary, config, data_version, holdout=True, rule=rule, period="holdout", holdout_id=hid)
        out[rule] = res
    return out


__all__ = ["Panel", "build_panel", "sample_at_month_ends", "evaluate", "evaluate_holdout", "ExperimentLog",
           "HoldoutUsed", "deflated_sharpe", "newey_west_se", "t_stats", "ols_nw", "cost_panel", "TUNING", "HOLDOUT",
           "config_hash", "signal_config"]
