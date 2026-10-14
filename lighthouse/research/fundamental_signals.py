"""
Fundamental signals from SEC filings, each known from the session after the filing that made it public.

From the point-in-time quarterly statements (lighthouse.pit: values as first reported, dated by filing), trailing-twelve-
month (TTM) flows and latest balance-sheet stocks; market values from the cover-page share count of the latest filing x
the as-traded price. A value older than 15 months (a company that stopped filing) is not used.

Earnings      sue            standardised unexpected earnings, seasonal random walk: (EPS_q - EPS_q-4) / standard
                             deviation of that change over the previous 8 quarters (Foster, Olsen & Shevlin 1984;
                             Bernard & Thomas 1989: the post-earnings drift follows it)
Quality       gp_assets      gross profit / total assets (Novy-Marx 2013)
              roic           after-tax operating income / (equity + debt - cash); roic_trend: its change over a year
              accruals       minus (net income - operating cash flow) / total assets (Sloan 1996)
              piotroski      F-score, 0-9 (Piotroski 2000)
              altman_z       Z-score (Altman 1968): higher = further from distress
              beneish        minus the M-score (Beneish 1999): higher = less sign of earnings manipulation
              fcf_sbc_yield  (operating cash flow - capex - stock-based pay) / market value
              cash_conversion operating cash flow / net income (profitable firms)
Value         earnings_yield, book_to_market, fcf_yield, ebitda_ev, and value_own_history (earnings yield vs the
              stock's own last five years, a z-score)
Capital       net_issuance   minus the one-year change in split-adjusted shares outstanding (Pontiff & Woodgate 2008)
allocation    shareholder_yield (dividends + buybacks - shares issued) / market value
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import store
from ..pit import load_statements, ttm
from ..total_return import load_actions, price_only_close, split_factor_after, to_units_of
from .harness import sample_at_month_ends

MAX_AGE_DAYS = 460
FUNDAMENTAL_SIGNALS = ["sue", "gp_assets", "roic", "roic_trend", "accruals", "piotroski", "altman_z", "beneish",
                       "fcf_sbc_yield", "cash_conversion", "earnings_yield", "book_to_market", "fcf_yield", "ebitda_ev",
                       "value_own_history", "net_issuance", "shareholder_yield"]
DESCRIPTIONS = {
    "sue": "standardised unexpected earnings (seasonal random walk)", "gp_assets": "gross profit / assets",
    "roic": "return on invested capital", "roic_trend": "1-year change in ROIC", "accruals": "low accruals",
    "piotroski": "Piotroski F-score", "altman_z": "Altman Z-score", "beneish": "minus the Beneish M-score",
    "fcf_sbc_yield": "FCF after stock-based pay / market value", "cash_conversion": "operating cash flow / net income",
    "earnings_yield": "earnings / price", "book_to_market": "book / market", "fcf_yield": "FCF / market value",
    "ebitda_ev": "EBITDA / enterprise value", "value_own_history": "earnings yield vs own 5-year history (z)",
    "net_issuance": "minus net share issuance (1 year)", "shareholder_yield": "dividends + buybacks - issuance / market value",
}


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((np.asarray(b, float) != 0) & np.isfinite(b), np.asarray(a, float) / np.asarray(b, float), np.nan)
    return out


def quarterly_features(q: pd.DataFrame) -> pd.DataFrame:
    """Per quarterly row (as first reported, dated `filed`): TTM flows, balance-sheet items and the derived ratios
    that do not need a price."""
    f = pd.DataFrame(index=q.index)
    f["filed"] = pd.to_datetime(q["filed"])
    for c in ("revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "capex", "sbc",
              "pretax_income", "income_tax", "dividends_paid", "buybacks", "stock_issued", "d_and_a", "sga", "eps"):
        f[f"{c}_ttm"] = ttm(q, c) if c in q else np.nan
    for c in ("total_assets", "equity", "total_debt", "cash", "current_assets", "current_liabilities", "receivables",
              "ppe", "retained_earnings", "total_liabilities", "long_term_debt", "cover_shares", "shares"):
        f[c] = q[c] if c in q else np.nan
    if "gross_profit_ttm" in f and f["gross_profit_ttm"].isna().all() and {"revenue", "cost_of_revenue"} <= set(q.columns):
        f["gross_profit_ttm"] = ttm(q.assign(gp=q["revenue"] - q["cost_of_revenue"]), "gp")
    a = f["total_assets"]
    f["gp_assets"] = _safe_div(f["gross_profit_ttm"], a)
    tax = np.clip(_safe_div(f["income_tax_ttm"], f["pretax_income_ttm"]), 0.0, 0.35)
    tax = np.where(np.isfinite(tax), tax, 0.21)
    nopat = f["operating_income_ttm"] * (1.0 - tax)
    ic = f["equity"] + f["total_debt"].fillna(0.0) - f["cash"].fillna(0.0)
    f["roic"] = np.where(ic > 0, _safe_div(nopat, ic), np.nan)
    f["roic_trend"] = f["roic"] - pd.Series(f["roic"]).shift(4)
    f["accruals"] = -_safe_div(f["net_income_ttm"] - f["operating_cash_flow_ttm"], a)
    f["cash_conversion"] = np.where(f["net_income_ttm"] > 0,
                                    np.clip(_safe_div(f["operating_cash_flow_ttm"], f["net_income_ttm"]), -5, 5), np.nan)
    f["piotroski"] = _piotroski(f)
    f["beneish"] = -_beneish(f)
    eps_d = q["eps"] - q["eps"].shift(4) if "eps" in q else pd.Series(np.nan, index=q.index)
    sd = eps_d.shift(1).rolling(8, min_periods=4).std()
    f["sue"] = _safe_div(eps_d, sd)
    return f


def _piotroski(f: pd.DataFrame) -> np.ndarray:
    a = f["total_assets"]
    roa = _safe_div(f["net_income_ttm"], a)
    lag = lambda s: pd.Series(s, index=f.index).shift(4).to_numpy()   # noqa: E731
    lev = _safe_div(f["long_term_debt"], a)
    cr = _safe_div(f["current_assets"], f["current_liabilities"])
    gm = _safe_div(f["gross_profit_ttm"], f["revenue_ttm"])
    at = _safe_div(f["revenue_ttm"], a)
    sh = f["shares"].to_numpy() if "shares" in f else np.full(len(f), np.nan)
    parts = [roa > 0, f["operating_cash_flow_ttm"].to_numpy() > 0, roa > lag(roa),
             f["operating_cash_flow_ttm"].to_numpy() > f["net_income_ttm"].to_numpy(),
             lev < lag(lev), cr > lag(cr), sh <= lag(sh) * 1.005, gm > lag(gm), at > lag(at)]
    known = np.isfinite(roa) & np.isfinite(lag(roa)) & np.isfinite(at) & np.isfinite(lag(at))
    score = np.sum([np.nan_to_num(p.astype(float)) for p in parts], axis=0)
    return np.where(known, score, np.nan)


def _beneish(f: pd.DataFrame) -> np.ndarray:
    lag = lambda s: pd.Series(s, index=f.index).shift(4).to_numpy()   # noqa: E731
    rev, a = f["revenue_ttm"].to_numpy(), f["total_assets"].to_numpy()
    dsri = _safe_div(_safe_div(f["receivables"], rev), lag(_safe_div(f["receivables"], rev)))
    gm = _safe_div(f["gross_profit_ttm"], rev)
    gmi = _safe_div(lag(gm), gm)
    aq = 1.0 - _safe_div(f["current_assets"].to_numpy() + f["ppe"].to_numpy(), a)
    aqi = _safe_div(aq, lag(aq))
    sgi = _safe_div(rev, lag(rev))
    dep = _safe_div(f["d_and_a_ttm"], f["d_and_a_ttm"].to_numpy() + f["ppe"].to_numpy())
    depi = _safe_div(lag(dep), dep)
    sgai = _safe_div(_safe_div(f["sga_ttm"], rev), lag(_safe_div(f["sga_ttm"], rev)))
    lv = _safe_div(f["total_liabilities"], a)
    lvgi = _safe_div(lv, lag(lv))
    tata = _safe_div(f["net_income_ttm"].to_numpy() - f["operating_cash_flow_ttm"].to_numpy(), a)
    # missing components are set to their neutral value (1 for indices, 0 for accruals), as is usual in practice
    fill = lambda x, v: np.where(np.isfinite(x), x, v)   # noqa: E731
    m = (-4.84 + 0.920 * fill(dsri, 1) + 0.528 * fill(gmi, 1) + 0.404 * fill(aqi, 1) + 0.892 * fill(sgi, 1)
         + 0.115 * fill(depi, 1) - 0.172 * fill(sgai, 1) + 4.679 * fill(tata, 0) - 0.327 * fill(lvgi, 1))
    return np.where(np.isfinite(sgi) & np.isfinite(tata), m, np.nan)


def daily_values(ticker: str, frame: pd.DataFrame, source: str = "auto") -> Dict[str, pd.Series]:
    """Every fundamental signal for one stock, daily, from the session after each filing."""
    h = load_statements(ticker, source=source)
    q = h.quarterly
    if q is None or q.empty or "filed" not in q:
        return {}
    feats = quarterly_features(q.sort_index())
    feats = feats[feats["filed"].notna()].sort_values("filed")
    feats = feats[~feats["filed"].duplicated(keep="last")]           # one 10-K reporting two periods: the later one
    idx = frame.index
    acts = load_actions(ticker, source=source, max_age_hours=None)
    acts_u = to_units_of(acts, idx[-1]) if not acts.empty else acts
    po = price_only_close(frame["Close"], acts_u) if not acts.empty else frame["Close"]
    traded = po * split_factor_after(idx, acts_u, idx[-1]) if not acts.empty else po

    def daily(col: str) -> pd.Series:
        s = pd.Series(np.asarray(feats[col], float), index=pd.DatetimeIndex(feats["filed"]) + pd.Timedelta(days=1))
        s = s[~s.index.duplicated(keep="last")]
        on = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        known_at = pd.Series(s.index, index=s.index).reindex(s.index.union(idx)).ffill().reindex(idx)
        age = (pd.Series(idx, index=idx) - known_at).dt.days
        return on.where(age <= MAX_AGE_DAYS)

    shares = daily("cover_shares")
    mcap = traded * shares
    out = {c: daily(c) for c in ("sue", "gp_assets", "roic", "roic_trend", "accruals", "piotroski", "beneish",
                                 "cash_conversion")}
    ni, eq, ocf, capex, sbc = (daily(c) for c in ("net_income_ttm", "equity", "operating_cash_flow_ttm", "capex_ttm", "sbc_ttm"))
    debt, cash = daily("total_debt").fillna(0.0), daily("cash").fillna(0.0)
    ebit, da = daily("operating_income_ttm"), daily("d_and_a_ttm")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["earnings_yield"] = ni / mcap
        out["book_to_market"] = (eq / mcap).where(eq > 0)
        out["fcf_yield"] = (ocf - capex.fillna(0.0)) / mcap
        out["fcf_sbc_yield"] = (ocf - capex.fillna(0.0) - sbc.fillna(0.0)) / mcap
        ev = mcap + debt - cash
        out["ebitda_ev"] = ((ebit + da.fillna(0.0)) / ev).where(ev > 0)
        wc = daily("current_assets") - daily("current_liabilities")
        ta, tl = daily("total_assets"), daily("total_liabilities")
        out["altman_z"] = (1.2 * wc / ta + 1.4 * daily("retained_earnings") / ta + 3.3 * ebit / ta
                           + 0.6 * mcap / tl + 1.0 * daily("revenue_ttm") / ta)
        ey = out["earnings_yield"]
        hist_m = ey.rolling(1260, min_periods=500).mean()
        hist_s = ey.rolling(1260, min_periods=500).std()
        out["value_own_history"] = (ey - hist_m) / hist_s
        # net issuance: shares in today's units (a split is not an issue of new shares)
        adj = shares * split_factor_after(idx, acts_u, idx[-1]) if not acts.empty else shares
        out["net_issuance"] = -(adj / adj.shift(252) - 1.0)
        payout = daily("dividends_paid_ttm").fillna(0.0) + daily("buybacks_ttm").fillna(0.0) - daily("stock_issued_ttm").fillna(0.0)
        out["shareholder_yield"] = (payout / mcap).where(mcap > 0)
    out["market_cap"] = mcap
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def monthly(frames: Dict[str, pd.DataFrame], panel, refresh: bool = False, source: str = "auto") -> Dict[str, pd.DataFrame]:
    """{signal: month-end x ticker} for every fundamental signal (and market_cap), cached as Parquet."""
    names = FUNDAMENTAL_SIGNALS + ["market_cap"]
    cached = {n: store.read_frame("research_signals", n) for n in names} if not refresh else {}
    if not refresh and all(v is not None and v.index.equals(panel.dates) for v in cached.values()):
        return cached
    from ..parallel import map_tickers
    daily: Dict[str, Dict[str, pd.Series]] = {n: {} for n in names}
    # network-bound (SEC filings, dividends): threads share the SEC rate limit; one company's odd filing must not stop
    # the panel, but a bug that fails most of them must not pass silently
    got, errors = map_tickers(lambda t: daily_values(t, frames[t], source), list(frames), max_workers=8)
    for t, vals in got.items():
        for n, s in vals.items():
            if n in daily:
                daily[n][t] = s
    if errors and len(errors) > 0.5 * max(len(frames), 1):     # ... but a bug that fails most of them must not pass
        raise RuntimeError(f"fundamental signals failed for {len(errors)} of {len(frames)} stocks, e.g. "
                           + "; ".join(f"{k}: {v}" for k, v in list(errors.items())[:3]))
    store.write_json("research", "fundamental_signal_errors", errors)
    out = {}
    for n in names:
        m = sample_at_month_ends(daily[n], panel.dates)
        store.write_frame("research_signals", n, m)
        out[n] = m
    return out


__all__ = ["FUNDAMENTAL_SIGNALS", "DESCRIPTIONS", "quarterly_features", "daily_values", "monthly"]
