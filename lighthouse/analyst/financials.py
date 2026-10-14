"""
A company's numbers as they were public on a date: trailing-twelve-month flows from the last four quarters filed,
the latest balance sheet, the share count from the latest cover page, the price and market value that day.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..fundamentals import StatementHistory
from ..pit import as_of, ttm
from ..total_return import price_only_close, split_factor_after, to_units_of

FLOWS = ["revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income", "operating_cash_flow", "capex",
         "d_and_a", "sbc", "interest_expense", "income_tax", "pretax_income", "dividends_paid", "buybacks",
         "stock_issued", "rnd", "sga", "acquisitions", "eps"]
STOCKS = ["total_assets", "current_assets", "current_liabilities", "cash", "short_investments", "cash_and_investments",
          "receivables",
          "inventory", "ppe", "goodwill", "accounts_payable", "total_liabilities", "long_term_debt", "short_term_debt",
          "total_debt", "equity", "retained_earnings", "cover_shares", "debt_due_1y", "debt_due_2y", "debt_due_3y",
          "debt_due_4y", "debt_due_5y", "debt_due_later"]


@dataclass
class Financials:
    ticker: str
    as_of: pd.Timestamp
    price: float                         # as traded that day
    shares: float                        # cover-page shares outstanding (latest filing)
    ttm: Dict[str, float]                # trailing twelve months, from the last four quarters filed by `as_of`
    ttm_prev: Dict[str, float]           # the twelve months before those
    bs: Dict[str, float]                 # the latest balance sheet filed by `as_of`
    annual: pd.DataFrame                 # fiscal years filed by `as_of`
    quarterly: pd.DataFrame              # quarters filed by `as_of`
    last_filed: Optional[pd.Timestamp] = None
    notes: List[str] = field(default_factory=list)

    @property
    def market_cap(self) -> float:
        return float(self.price * self.shares) if self.shares and np.isfinite(self.shares) else float("nan")

    @property
    def liquid_assets(self) -> float:
        """Cash and short-term investments (the larger of their sum and the combined line some companies report)."""
        parts = (self.bs.get("cash") or 0.0) + (self.bs.get("short_investments") or 0.0)
        return float(max(parts, self.bs.get("cash_and_investments") or 0.0))

    @property
    def net_debt(self) -> float:
        debt = self.bs.get("total_debt")
        return float((debt if debt is not None and np.isfinite(debt) else 0.0) - self.liquid_assets)

    @property
    def enterprise_value(self) -> float:
        return self.market_cap + self.net_debt

    def get(self, key: str, default=np.nan) -> float:
        v = self.ttm.get(key, self.bs.get(key, default))
        return float(v) if v is not None and np.isfinite(v) else default

    @property
    def tax_rate(self) -> float:
        t = self.get("income_tax") / self.get("pretax_income") if self.get("pretax_income") > 0 else np.nan
        return float(np.clip(t, 0.0, 0.35)) if np.isfinite(t) else 0.21

    @property
    def fcf(self) -> float:
        return self.get("operating_cash_flow") - (self.get("capex", 0.0) or 0.0)

    def ratios(self) -> Dict[str, float]:
        """The headline valuation, quality and growth ratios."""
        mc, ev = self.market_cap, self.enterprise_value
        ebitda = self.get("operating_income") + (self.get("d_and_a", 0.0) or 0.0)
        inv_cap = self.get("equity") + (self.get("total_debt", 0.0) or 0.0) - self.liquid_assets
        nopat = self.get("operating_income") * (1.0 - self.tax_rate)
        r = {
            "pe": mc / self.get("net_income") if self.get("net_income") > 0 else np.nan,
            "ev_ebitda": ev / ebitda if ebitda > 0 else np.nan,
            "ev_sales": ev / self.get("revenue") if self.get("revenue") > 0 else np.nan,
            "pb": mc / self.get("equity") if self.get("equity") > 0 else np.nan,
            "fcf_yield": self.fcf / mc if mc > 0 else np.nan,
            "fcf_sbc_yield": (self.fcf - (self.get("sbc", 0.0) or 0.0)) / mc if mc > 0 else np.nan,
            "gross_margin": self.get("gross_profit") / self.get("revenue") if self.get("revenue") > 0 else np.nan,
            "operating_margin": self.get("operating_income") / self.get("revenue") if self.get("revenue") > 0 else np.nan,
            "net_margin": self.get("net_income") / self.get("revenue") if self.get("revenue") > 0 else np.nan,
            "roic": nopat / inv_cap if inv_cap > 0 else np.nan,
            "roe": self.get("net_income") / self.get("equity") if self.get("equity") > 0 else np.nan,
            "revenue_growth": (self.get("revenue") / self.ttm_prev.get("revenue", np.nan) - 1.0)
            if self.ttm_prev.get("revenue") else np.nan,
            "eps_growth": (self.get("net_income") / self.ttm_prev.get("net_income", np.nan) - 1.0)
            if (self.ttm_prev.get("net_income") or 0) > 0 else np.nan,
            "net_debt_ebitda": self.net_debt / ebitda if ebitda > 0 else np.nan,
            "interest_cover": self.get("operating_income") / self.get("interest_expense")
            if self.get("interest_expense", 0) > 0 else np.nan,
            "current_ratio": self.get("current_assets") / self.get("current_liabilities")
            if self.get("current_liabilities") > 0 else np.nan,
            "sbc_revenue": self.get("sbc", 0.0) / self.get("revenue") if self.get("revenue") > 0 else np.nan,
            "rnd_revenue": self.get("rnd", np.nan) / self.get("revenue") if self.get("revenue") > 0 else np.nan,
            "capex_revenue": self.get("capex", np.nan) / self.get("revenue") if self.get("revenue") > 0 else np.nan,
        }
        return {k: (float(v) if v is not None and np.isfinite(v) else float("nan")) for k, v in r.items()}


def financials_as_of(ticker: str, statements: StatementHistory, prices: pd.DataFrame, actions: Optional[pd.DataFrame],
                     date=None) -> Financials:
    """The company's numbers public on `date` (default: the last price) and the as-traded price that day. `prices` is the
    total-return price frame (adjusted at its last date); `actions` its dividends and splits."""
    d = pd.Timestamp(date) if date is not None else prices.index[-1]
    q = as_of(statements.quarterly, d)
    a = as_of(statements.annual, d)
    notes: List[str] = []
    if q is None or q.empty:
        raise ValueError(f"{ticker}: no quarterly statements filed by {d.date()}")
    q = q.sort_index()
    t_now: Dict[str, float] = {}
    t_prev: Dict[str, float] = {}
    for c in FLOWS:
        if c in q:
            s = ttm(q, c)
            t_now[c] = float(s.iloc[-1]) if len(s) and np.isfinite(s.iloc[-1]) else float("nan")
            t_prev[c] = float(s.iloc[-5]) if len(s) > 4 and np.isfinite(s.iloc[-5]) else float("nan")
    if "gross_profit" in t_now and not np.isfinite(t_now["gross_profit"]) and np.isfinite(t_now.get("cost_of_revenue", np.nan)):
        t_now["gross_profit"] = t_now["revenue"] - t_now["cost_of_revenue"]
    last = q.iloc[-1]
    bs = {c: float(last[c]) for c in STOCKS if c in q and pd.notna(last[c])}
    # balance-sheet items missing in the last quarter (e.g. debt maturities, which only the 10-K reports): the latest
    # annual value
    if len(a):
        la = a.sort_index().iloc[-1]
        for c in STOCKS:
            if c not in bs and c in a and pd.notna(la[c]):
                bs[c] = float(la[c])
    px = prices["Close"][prices.index <= d]
    if px.empty:
        raise ValueError(f"{ticker}: no price on or before {d.date()}")
    if actions is not None and not actions.empty:
        acts = to_units_of(actions, prices.index[-1])
        po = price_only_close(prices["Close"], acts)
        price = float(po[po.index <= d].iloc[-1] * split_factor_after([px.index[-1]], acts, prices.index[-1])[0])
    else:
        price = float(px.iloc[-1])
        notes.append("no dividend/split history: the price is the adjusted close (valuation ratios may be off)")
    shares = bs.get("cover_shares", float("nan"))
    if not np.isfinite(shares) and "shares" in q:
        shares = float(q["shares"].dropna().iloc[-1])
        notes.append("shares outstanding from the weighted-average diluted count (no cover-page count)")
    return Financials(ticker=ticker, as_of=d, price=price, shares=shares, ttm=t_now, ttm_prev=t_prev, bs=bs, annual=a,
                      quarterly=q, last_filed=pd.to_datetime(q["filed"]).max(), notes=notes)


__all__ = ["Financials", "financials_as_of", "FLOWS", "STOCKS"]
