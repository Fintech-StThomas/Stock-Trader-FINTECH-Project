"""
A small linked three-statement model, projected 3-5 years from a handful of assumptions.

Income statement   revenue grows at the year's growth rate; operating income = revenue x operating margin;
                   interest = debt x the interest rate (less interest earned on cash); tax at the tax rate
Balance sheet      net working capital = revenue x NWC intensity; PP&E rolls forward with capex and D&A; debt follows
                   the debt policy; equity rolls forward with net income, dividends and buybacks; cash is the plug
Cash flow          operating cash flow = net income + D&A + stock-based pay - the change in working capital;
                   free cash flow to the firm (FCFF) = EBIT x (1 - tax) + D&A + stock-based pay - capex - change in
                   working capital (the cash view: pay in shares is added back like D&A). The valuation uses FCFF after
                   stock-based pay (`fcff_after_sbc`, i.e. without the add-back): the operating margin already deducts it,
                   and it is a real cost to shareholders, paid in dilution

The balance sheet balances by construction (assets = debt + equity + other liabilities held flat); the check is part
of every projection. Assumptions start from the company's own trailing numbers (`Assumptions.from_financials`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .financials import Financials


@dataclass
class Assumptions:
    revenue: float                       # the starting (TTM) revenue
    growth: List[float]                  # revenue growth per projected year
    operating_margin: List[float]        # per projected year
    tax_rate: float = 0.21
    da_pct: float = 0.04                 # D&A / revenue
    capex_pct: float = 0.05              # capex / revenue
    nwc_pct: float = 0.05                # net working capital / revenue
    sbc_pct: float = 0.0                 # stock-based pay / revenue (a real cost: subtracted from FCFF in `valuation`)
    payout: float = 0.3                  # dividends / net income
    buyback_pct: float = 0.0             # buybacks / net income
    interest_rate: float = 0.05          # on debt
    cash_rate: float = 0.03              # earned on cash
    debt_policy: str = "flat"            # "flat" | "paydown" (repay 10% a year)
    start_cash: float = 0.0
    start_debt: float = 0.0
    start_ppe: float = 0.0
    start_equity: float = 0.0
    start_other_liabilities: float = 0.0
    shares: float = 1.0
    label: str = "base"

    @property
    def years(self) -> int:
        return len(self.growth)

    @classmethod
    def from_financials(cls, f: Financials, years: int = 5, growth: Optional[List[float]] = None,
                        margin: Optional[List[float]] = None, long_run_growth: float = 0.03, label: str = "base") -> "Assumptions":
        """Start from the company's trailing year: growth fades from its last year's rate (capped at +/-25%) to the long-run
        rate; the margin stays at today's; intensities are today's ratios; the tax rate is today's effective rate held to
        15-30% (one-off tax items do not last for five years)."""
        r = f.ratios()
        g0 = r["revenue_growth"] if np.isfinite(r["revenue_growth"]) else long_run_growth
        g0 = float(np.clip(g0, -0.25, 0.25))
        g = growth or [g0 + (long_run_growth - g0) * k / max(years - 1, 1) for k in range(years)]
        m0 = r["operating_margin"] if np.isfinite(r["operating_margin"]) else 0.10
        m = margin or [m0] * years
        rev = f.get("revenue")
        pct = lambda key, d: float(np.clip(f.get(key, np.nan) / rev, 0, 1)) if rev and np.isfinite(f.get(key, np.nan)) else d  # noqa: E731
        # operating working capital: receivables + inventory - payables (cash and investments are not working capital)
        nwc = (f.get("receivables", 0.0) if np.isfinite(f.get("receivables", np.nan)) else 0.0) + \
              (f.get("inventory", 0.0) if np.isfinite(f.get("inventory", np.nan)) else 0.0) - \
              (f.get("accounts_payable", 0.0) if np.isfinite(f.get("accounts_payable", np.nan)) else 0.0)
        capex_pct = pct("capex", 0.05)
        da_pct = pct("d_and_a", np.nan)
        if not np.isfinite(da_pct):
            da_pct = capex_pct                  # not reported under a standard concept: assume D&A = capex (steady state)
        ni = f.get("net_income")
        payout = float(np.clip(f.get("dividends_paid", 0.0) / ni, 0, 1.5)) if ni and ni > 0 else 0.0
        bb = float(np.clip(f.get("buybacks", 0.0) / ni, 0, 2.0)) if ni and ni > 0 else 0.0
        debt = f.get("total_debt", 0.0)
        debt = 0.0 if not np.isfinite(debt) else debt
        ie = f.get("interest_expense", np.nan)
        rate = float(np.clip(ie / debt, 0.01, 0.12)) if debt > 0 and np.isfinite(ie) else 0.05
        cash = f.liquid_assets
        equity = f.get("equity", 0.0)
        other = (f.get("total_assets", np.nan) - debt - equity) if np.isfinite(f.get("total_assets", np.nan)) else 0.0
        return cls(revenue=float(rev), growth=[float(x) for x in g], operating_margin=[float(x) for x in m],
                   tax_rate=float(np.clip(f.tax_rate, 0.15, 0.30)), da_pct=da_pct, capex_pct=capex_pct,
                   nwc_pct=float(np.clip(nwc / rev, -0.5, 0.5)) if rev else 0.05, sbc_pct=pct("sbc", 0.0),
                   payout=payout, buyback_pct=bb, interest_rate=rate, start_cash=float(cash), start_debt=float(debt),
                   start_ppe=float(f.get("ppe", 0.0) if np.isfinite(f.get("ppe", np.nan)) else 0.0),
                   start_equity=float(equity) if np.isfinite(equity) else 0.0,
                   start_other_liabilities=float(other) if np.isfinite(other) else 0.0,
                   shares=float(f.shares) if np.isfinite(f.shares) else 1.0, label=label)

    def scaled(self, growth_shift: float = 0.0, margin_shift: float = 0.0, label: Optional[str] = None) -> "Assumptions":
        return replace(self, growth=[g + growth_shift for g in self.growth],
                       operating_margin=[m + margin_shift for m in self.operating_margin], label=label or self.label)

    def to_dict(self) -> dict:
        return asdict(self)


def project(a: Assumptions) -> pd.DataFrame:
    """The projection: one row per year (1..n), every line of the three statements, and the balance check."""
    rows = []
    rev, cash, debt, ppe, eq = a.revenue, a.start_cash, a.start_debt, a.start_ppe, a.start_equity
    nwc = a.revenue * a.nwc_pct
    other = a.start_other_liabilities
    other_assets = other + eq + debt - cash - nwc - ppe          # everything else on the balance sheet, held flat
    for k in range(a.years):
        rev_n = rev * (1.0 + a.growth[k])
        ebit = rev_n * a.operating_margin[k]
        da = rev_n * a.da_pct
        capex = rev_n * a.capex_pct
        sbc = rev_n * a.sbc_pct
        nwc_n = rev_n * a.nwc_pct
        d_nwc = nwc_n - nwc
        debt_n = debt * (0.9 if a.debt_policy == "paydown" else 1.0)
        interest = debt * a.interest_rate - cash * a.cash_rate
        pretax = ebit - interest
        tax = max(pretax, 0.0) * a.tax_rate
        ni = pretax - tax
        div = max(ni, 0.0) * a.payout
        bb = max(ni, 0.0) * a.buyback_pct
        cfo = ni + da + sbc - d_nwc
        fcff = ebit * (1.0 - a.tax_rate) + da + sbc - capex - d_nwc
        cash_n = cash + cfo - capex - div - bb + (debt_n - debt)
        ppe_n = ppe + capex - da
        eq_n = eq + ni + sbc - div - bb
        assets = cash_n + nwc_n + ppe_n + other_assets
        liab_eq = debt_n + eq_n + other
        rows.append({"year": k + 1, "revenue": rev_n, "growth": a.growth[k], "operating_income": ebit,
                     "operating_margin": a.operating_margin[k], "interest": interest, "pretax_income": pretax, "tax": tax,
                     "net_income": ni, "eps": ni / a.shares if a.shares else np.nan, "d_and_a": da, "sbc": sbc,
                     "capex": capex, "change_nwc": d_nwc, "operating_cash_flow": cfo, "fcff": fcff,
                     "fcff_after_sbc": fcff - sbc, "dividends": div, "buybacks": bb, "cash": cash_n, "nwc": nwc_n,
                     "ppe": ppe_n, "debt": debt_n, "equity": eq_n, "total_assets": assets,
                     "balance_check": float(assets - liab_eq)})
        rev, cash, debt, ppe, eq, nwc = rev_n, cash_n, debt_n, ppe_n, eq_n, nwc_n
    return pd.DataFrame(rows).set_index("year")


__all__ = ["Assumptions", "project"]
