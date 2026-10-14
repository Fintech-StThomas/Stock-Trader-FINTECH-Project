"""
The checks an analyst runs before trusting the numbers: earnings quality, balance-sheet risk, how management has used
the cash, and what is coming up.

Earnings quality   accruals (net income well above operating cash flow), weak cash conversion, receivables growing
                   much faster than revenue, inventory building, stock-based pay large relative to revenue, a Beneish
                   M-score above -1.78 (the threshold Beneish 1999 used for likely manipulators), restated filings
                   (10-K/A, 10-Q/A) in the last three years, goodwill a large share of assets
Balance sheet      debt coming due in the next one and two years (the 10-K's maturity footnote, XBRL
                   LongTermDebtMaturitiesRepaymentsOfPrincipal...) against cash and free cash flow, interest cover,
                   net debt / EBITDA, the current ratio, Altman Z
Capital allocation five years of cash flow: what went to dividends, buybacks and acquisitions against the free cash flow
                   earned; the change in share count; the ROIC trend
Catalysts          the next earnings release, 10-Q / 10-K and annual meeting, estimated from the same filings in past
                   years (an estimate: companies move their dates)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .financials import Financials


@dataclass
class Flag:
    area: str                 # "earnings quality" | "balance sheet" | "capital allocation"
    severity: str             # "red" | "amber" | "green"
    text: str


def _yoy(f: Financials, key: str) -> float:
    now, prev = f.ttm.get(key, np.nan), f.ttm_prev.get(key, np.nan)
    return float(now / prev - 1.0) if prev and np.isfinite(prev) and prev != 0 and np.isfinite(now) else float("nan")


def earnings_quality(f: Financials, filings: Optional[pd.DataFrame] = None) -> List[Flag]:
    out: List[Flag] = []
    ni, ocf, ta = f.get("net_income"), f.get("operating_cash_flow"), f.get("total_assets")
    if np.isfinite(ni) and np.isfinite(ocf) and np.isfinite(ta) and ta > 0:
        acc = (ni - ocf) / ta
        if acc > 0.05:
            out.append(Flag("earnings quality", "red", f"Accruals are {acc:.1%} of assets: net income runs well ahead of "
                                                       "operating cash flow."))
        elif acc < -0.05:
            out.append(Flag("earnings quality", "green", f"Operating cash flow exceeds net income by {-acc:.1%} of assets."))
    if np.isfinite(ni) and ni > 0 and np.isfinite(ocf):
        cc = ocf / ni
        if cc < 0.8:
            out.append(Flag("earnings quality", "amber", f"Only {cc:.0%} of net income arrived as operating cash flow."))
    g_rev = _yoy(f, "revenue")
    q = f.quarterly
    for key, label in (("receivables", "Receivables"), ("inventory", "Inventory")):
        if key in q and len(q) >= 5 and np.isfinite(g_rev):
            s = q[key].dropna()
            if len(s) >= 5 and s.iloc[-5] > 0:
                g = s.iloc[-1] / s.iloc[-5] - 1.0
                if g - g_rev > 0.15:
                    acq = f.get("acquisitions", 0.0)
                    why = (f" Acquisitions of ${acq / 1e9:,.1f}bn in the year may explain part of it."
                           if np.isfinite(acq) and np.isfinite(ta) and ta > 0 and acq > 0.05 * ta else "")
                    out.append(Flag("earnings quality", "amber",
                                    f"{label} grew {g:.0%} in a year against revenue {g_rev:+.0%}.{why}"))
    sbc_r = f.ratios()["sbc_revenue"]
    if np.isfinite(sbc_r) and sbc_r > 0.10:
        out.append(Flag("earnings quality", "amber", f"Stock-based pay is {sbc_r:.0%} of revenue: a real cost that adjusted "
                                                     "earnings leave out."))
    gw = f.bs.get("goodwill")
    if gw and np.isfinite(ta) and ta > 0 and gw / ta > 0.4:
        out.append(Flag("earnings quality", "amber", f"Goodwill is {gw / ta:.0%} of assets: past acquisitions that could be "
                                                     "written down."))
    if filings is not None and len(filings):
        recent = filings[(filings["filed"] >= f.as_of - pd.Timedelta(days=3 * 365)) & (filings["filed"] <= f.as_of)]
        n = int(recent["form"].isin(["10-K/A", "10-Q/A"]).sum())
        if n:
            out.append(Flag("earnings quality", "amber" if n == 1 else "red",
                            f"{n} amended periodic report{'s' if n > 1 else ''} (10-K/A or 10-Q/A) in the last three years."))
    try:
        from ..research.fundamental_signals import _beneish, quarterly_features
        feats = quarterly_features(q)
        m = float(_beneish(feats)[-1]) if len(feats) else np.nan     # the M-score itself (the signal is its negative)
        if np.isfinite(m) and m > -1.78:
            out.append(Flag("earnings quality", "red", f"Beneish M-score {m:.2f}, above the -1.78 line of likely "
                                                       "manipulators in Beneish (1999)."))
    except Exception:  # noqa: BLE001 - too few quarters for the score
        pass
    if not out:
        out.append(Flag("earnings quality", "green", "No earnings-quality flag in the latest filings."))
    return out


def balance_sheet(f: Financials) -> List[Flag]:
    out: List[Flag] = []
    r = f.ratios()
    liquid = f.liquid_assets
    fcf = f.fcf
    due1, due2 = f.bs.get("debt_due_1y"), f.bs.get("debt_due_2y")
    if due1 is not None and np.isfinite(due1) and due1 > 0:
        cover = (liquid + max(fcf, 0.0)) / due1
        sev = "red" if cover < 1 else "amber" if cover < 2 else "green"
        out.append(Flag("balance sheet", sev, f"${due1 / 1e9:,.2f}bn of debt falls due within a year"
                        + (f" and ${due2 / 1e9:,.2f}bn in the second" if due2 and np.isfinite(due2) else "")
                        + f"; cash, investments and a year's free cash flow cover the first {cover:.1f}x."))
    ic = r["interest_cover"]
    if np.isfinite(ic):
        out.append(Flag("balance sheet", "red" if ic < 2 else "amber" if ic < 4 else "green",
                        f"Operating income covers interest {ic:.1f}x."))
    nd = r["net_debt_ebitda"]
    if np.isfinite(nd):
        out.append(Flag("balance sheet", "red" if nd > 4 else "amber" if nd > 2.5 else "green",
                        f"Net debt is {nd:.1f}x EBITDA" + (" (net cash)." if nd < 0 else ".")))
    cr = r["current_ratio"]
    if np.isfinite(cr) and cr < 1:
        out.append(Flag("balance sheet", "amber", f"Current ratio {cr:.2f}: short-term liabilities exceed short-term assets."))
    return out


def capital_allocation(f: Financials, years: int = 5) -> Dict[str, object]:
    """The last `years` fiscal years: free cash flow earned and where the cash went; the share-count change."""
    a = f.annual.sort_index().tail(years)
    out: Dict[str, object] = {"years": int(len(a))}
    if a.empty:
        return out
    get = lambda c: float(a[c].sum()) if c in a and a[c].notna().any() else 0.0   # noqa: E731
    fcf = get("operating_cash_flow") - get("capex")
    out.update({"fcf": fcf, "dividends": get("dividends_paid"), "buybacks": get("buybacks"),
                "acquisitions": get("acquisitions"), "issued": get("stock_issued"), "sbc": get("sbc")})
    out["returned_share_of_fcf"] = (out["dividends"] + out["buybacks"]) / fcf if fcf > 0 else float("nan")
    if "cover_shares" in a and a["cover_shares"].notna().sum() >= 2:
        s = a["cover_shares"].dropna()
        out["share_change"] = float(s.iloc[-1] / s.iloc[0] - 1.0)
    lines = []
    if fcf > 0:
        lines.append(f"Over {len(a)} fiscal years it earned ${fcf / 1e9:,.1f}bn of free cash flow and paid "
                     f"${out['dividends'] / 1e9:,.1f}bn in dividends and ${out['buybacks'] / 1e9:,.1f}bn in buybacks "
                     f"({out['returned_share_of_fcf']:.0%} of it)"
                     + (f"; ${out['acquisitions'] / 1e9:,.1f}bn went to acquisitions." if out["acquisitions"] else "."))
    if "share_change" in out:
        lines.append(f"Shares outstanding changed {out['share_change']:+.1%} (cover pages, not split-adjusted).")
    out["lines"] = lines
    return out


def catalysts(filings: Optional[pd.DataFrame], as_of, fiscal_year_end: Optional[str] = None) -> pd.DataFrame:
    """Expected next dates, each estimated as last year's date of the same event plus a year (moved to a weekday)."""
    if filings is None or filings.empty:
        return pd.DataFrame(columns=["event", "expected", "based_on"])
    d0 = pd.Timestamp(as_of)
    past = filings[filings["filed"] <= d0]
    events = {
        "earnings release (8-K Item 2.02)": past[past["form"].isin(["8-K"]) & past["items"].str.contains(r"\b2\.02\b")],
        "quarterly report (10-Q)": past[past["form"] == "10-Q"],
        "annual report (10-K)": past[past["form"] == "10-K"],
        "proxy statement / annual meeting (DEF 14A)": past[past["form"] == "DEF 14A"],
    }
    rows = []
    for name, ev in events.items():
        if ev.empty:
            continue
        dates = ev["filed"].sort_values()
        last_year = dates[(dates > d0 - pd.Timedelta(days=380)) & (dates <= d0 - pd.Timedelta(days=0))]
        cands = sorted(x + pd.DateOffset(years=1) for x in last_year)
        nxt = [c for c in cands if c > d0]
        if not nxt:
            continue
        e = nxt[0]
        while e.weekday() >= 5:
            e += pd.Timedelta(days=1)
        rows.append({"event": name, "expected": e.date(), "based_on": f"filed {(e - pd.DateOffset(years=1)).date()} last year"})
    if fiscal_year_end:
        mm, dd = int(fiscal_year_end[:2]), int(fiscal_year_end[2:])
        fye = pd.Timestamp(year=d0.year, month=mm, day=min(dd, 28)) + pd.offsets.MonthEnd(0) if dd >= 28 else \
            pd.Timestamp(year=d0.year, month=mm, day=dd)
        if fye <= d0:
            fye = fye + pd.DateOffset(years=1)
        rows.append({"event": "fiscal year end", "expected": fye.date(), "based_on": f"fiscal year end {fiscal_year_end}"})
    return pd.DataFrame(rows).sort_values("expected").reset_index(drop=True) if rows else \
        pd.DataFrame(columns=["event", "expected", "based_on"])


__all__ = ["Flag", "earnings_quality", "balance_sheet", "capital_allocation", "catalysts"]
