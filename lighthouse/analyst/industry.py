"""
Industry metrics for the companies the generic ratios misread, chosen by the SIC code of the company's EDGAR profile.

Why       A bank's debt is its raw material and its "revenue" is net of the interest it pays, so EV/EBITDA, debt/equity
          and operating margin say little about it; a REIT's net income is after depreciating buildings that usually
          hold their value (funds from operations add it back); a subscription company books revenue long after it
          signs the contract (remaining performance obligations, deferred revenue); an insurer lives on what claims
          cost against the premiums it earns.
Metrics   bank (SIC 6021-6036)    net interest income; NII / average total assets (a proxy for the net interest
                                  margin: XBRL has no earning-assets total); efficiency ratio = non-interest expense /
                                  (NII + non-interest income); loans / deposits; tangible book value per share =
                                  (equity - preferred - goodwill - other intangibles) / cover-page shares; P/TBV (a
                                  bank that has never tagged preferred stock, under any preferred concept, counts it
                                  as 0 and says so; one that tagged it at other dates but not this one gets none)
          REIT (6798)             FFO as NAREIT defines it = net income + depreciation & amortisation - gains on
                                  property sales + real-estate impairments (none added when the period tags none,
                                  which the metric says); P/FFO; FFO payout = common dividends paid / FFO; the
                                  impairments shown beside it
          software (7370-7374)    remaining performance obligations (RPO); RPO / revenue; RPO and deferred-revenue
                                  growth year on year
          insurer (6311-6399)     net premiums earned; loss ratio = losses, benefits and claims / premiums earned;
                                  premium growth
Periods   A flow is the twelve months to the latest period end: the fiscal year when that end is a year end, else
          four consecutive quarters (quarters derived from year-to-date values as in `lighthouse.pit`); when an input
          has no twelve months there, the latest fiscal year instead. A balance is at the same end (else at the latest
          fiscal year end). The two numbers of a ratio always cover the same period, which the metric states.
Point in  The metric on date D uses only facts filed before D: a value is usable from the session after the filing
time      that first reported it (as in `fundamentals.point_in_time_scores`), and as first reported (a restatement is
          never used early). Each metric carries the latest filing date among its inputs.
Never     A metric whose input is not tagged for the period, under the standard us-gaap concepts listed in
invented  `pit.INDUSTRY_ITEMS`, is None with a note naming what is missing. Companies that tag a line with their own
          extension concept (EDGAR's companyfacts carries the standard taxonomies only) therefore get None for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import pit

INDUSTRY_LABELS = {"bank": "bank", "reit": "real estate investment trust", "software": "software / subscription",
                   "insurer": "insurer"}
SECTOR_OF_INDUSTRY = {"bank": "Financial Services", "insurer": "Financial Services", "reit": "Real Estate",
                      "software": "Technology"}
YEAR_BEFORE_DAYS = (350, 380)            # "the same date a year earlier": a period end 350-380 days before


def industry_for_sic(sic) -> Optional[str]:
    """"bank" | "reit" | "software" | "insurer" for a SIC code, else None. Banks: commercial banks and savings
    institutions (6021, 6022, 6029, 6035, 6036); REITs: 6798; software: computer programming, prepackaged software,
    integrated systems and data processing (7370-7374); insurers: life, health, managed care, property & casualty,
    surety, title (6311-6399; the agents and brokers of 6411 do not underwrite)."""
    try:
        s = int(str(sic).strip())
    except (TypeError, ValueError):
        return None
    if s in (6021, 6022, 6029, 6035, 6036):
        return "bank"
    if s == 6798:
        return "reit"
    if 7370 <= s <= 7374:
        return "software"
    if 6311 <= s <= 6399:
        return "insurer"
    return None


@dataclass
class IndustryMetric:
    key: str
    label: str
    value: Optional[float]
    kind: str                            # "money" | "ratio" (shown as %) | "multiple" (x) | "per_share"
    period: str                          # e.g. "12 months to 2025-09-30 (four quarters)", "at 2025-09-30"
    filed: Optional[pd.Timestamp]        # the latest filing among the inputs (usable from the next session)
    formula: str
    note: str = ""

    def text(self) -> str:
        v = self.value
        if v is None:
            return "n/a"
        if self.kind == "money":
            return f"${v / 1e9:,.2f}bn"
        if self.kind == "ratio":
            return f"{v * 100:.1f}%"
        if self.kind == "multiple":
            return f"{v:.2f}x"
        return f"${v:,.2f}"


@dataclass
class IndustryView:
    industry: str                        # "bank" | "reit" | "software" | "insurer"
    sic: str
    as_of: pd.Timestamp
    metrics: List[IndustryMetric]
    notes: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return INDUSTRY_LABELS[self.industry]

    def get(self, key: str) -> Optional[IndustryMetric]:
        return next((m for m in self.metrics if m.key == key), None)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"metric": m.label, "value": m.text(), "period": m.period,
                              "filed": str(m.filed.date()) if m.filed is not None else "n/a",
                              "how": m.formula + (f" ({m.note})" if m.note else "")} for m in self.metrics]
                            ).set_index("metric")


# --------------------------------------------------------------------------- #
# Values as public before a date
# --------------------------------------------------------------------------- #
_Val = Tuple[float, pd.Timestamp]         # (value, filed)


class _Book:
    """One company's facts as public before `date`, with each line item's quarterly / annual / balance values."""

    def __init__(self, facts: pd.DataFrame, date) -> None:
        self.date = pd.Timestamp(date)
        f = facts[pd.to_datetime(facts["filed"]) < self.date] if facts is not None and len(facts) else facts
        self.facts = f if f is not None else pd.DataFrame(columns=["concept", "unit", "start", "end", "val", "filed"])
        self.orig = pit.original_facts(self.facts) if len(self.facts) else self.facts
        self._cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    def values(self, key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if key not in self._cache:
            item = pit.INDUSTRY.get(key) or pit.ITEMS[key]
            empty = pd.DataFrame(columns=["val", "filed"])
            self._cache[key] = pit.item_values(self.orig, item) if len(self.orig) else (empty, empty)
        return self._cache[key]

    def concepts_of(self, key: str) -> Sequence[str]:
        return (pit.INDUSTRY.get(key) or pit.ITEMS[key]).concepts

    def never_tagged(self, key: str) -> bool:
        """True when no filing before the date carries any of the item's concepts, at any period."""
        return not len(self.facts) or not self.facts["concept"].isin(self.concepts_of(key)).any()

    def latest_end(self) -> Optional[pd.Timestamp]:
        """The latest period the company has reported: its last quarter (or year) of net income or balance sheet."""
        ends = []
        for k in ("net_income", "equity", "total_assets"):
            q, a = self.values(k)
            ends += list(q.index) + list(a.index)
        return max(ends) if ends else None

    def latest_year_end(self) -> Optional[pd.Timestamp]:
        _, a = self.values("net_income")
        return a.index.max() if len(a) else None

    def twelve_months(self, key: str, end: pd.Timestamp) -> Optional[Tuple[float, pd.Timestamp, str]]:
        """(value, filed, basis) of a flow over the twelve months to `end`: the fiscal-year value ending there, else
        the sum of four consecutive quarters ending there; None when neither exists."""
        q, a = self.values(key)
        if end in a.index and np.isfinite(a.loc[end, "val"]):
            return float(a.loc[end, "val"]), pd.Timestamp(a.loc[end, "filed"]), "fiscal year"
        if end not in q.index:
            return None
        qq = q[q.index <= end].sort_index().tail(4)
        if len(qq) < 4 or qq.index[-1] != end or not np.isfinite(qq["val"].astype(float)).all():
            return None
        gaps = pd.Series(qq.index).diff().dt.days.dropna()
        if not gaps.between(*pit.QUARTER_DAYS).all():
            return None
        return float(qq["val"].sum()), pd.Timestamp(pd.to_datetime(qq["filed"]).max()), "four quarters"

    def balance(self, key: str, end: pd.Timestamp) -> Optional[_Val]:
        q, _ = self.values(key)
        if end in q.index and np.isfinite(q.loc[end, "val"]):
            return float(q.loc[end, "val"]), pd.Timestamp(q.loc[end, "filed"])
        return None

    def year_before(self, key: str, end: pd.Timestamp) -> Optional[pd.Timestamp]:
        """The period end of `key` closest to one year before `end` (350-380 days), if reported."""
        q, a = self.values(key)
        ends = sorted(set(q.index) | set(a.index))
        cands = [e for e in ends if YEAR_BEFORE_DAYS[0] <= (end - e).days <= YEAR_BEFORE_DAYS[1]]
        return min(cands, key=lambda e: abs((end - e).days - 365)) if cands else None

    def cover_shares(self, by: pd.Timestamp) -> Optional[_Val]:
        """The cover-page share count of the latest filing up to `by` (a filing date): the exact count, which
        `CommonStockSharesOutstanding` is not always (some banks tag it rounded to the hundred million)."""
        c = pit.cover_shares(self.facts) if len(self.facts) else pd.Series(dtype=float)
        c = c[c.index <= by + pd.Timedelta(days=pit.SAME_FILING_DAYS)]
        if c.empty:
            return None
        return float(c.iloc[-1]), pd.Timestamp(c.index[-1])


def _label_end(e: pd.Timestamp) -> str:
    return str(pd.Timestamp(e).date())


def _missing(book: _Book, keys: Sequence[str], where: str) -> str:
    names = [f"{k} ({' / '.join(book.concepts_of(k)[:2])}{' ...' if len(book.concepts_of(k)) > 2 else ''})"
             for k in keys]
    return f"not tagged for {where}: " + ", ".join(names)


def _flows(book: _Book, keys: Sequence[str], ends: Sequence[Optional[pd.Timestamp]]) -> tuple:
    """({key: value}, filed, period text, end, "") for flows over the same twelve months: the first of `ends` where
    every key has them; else (None, None, None, None, note naming what is missing at each end tried)."""
    missing: List[str] = []
    for e in dict.fromkeys(x for x in ends if x is not None):
        got = {k: book.twelve_months(k, e) for k in keys}
        miss = [k for k, v in got.items() if v is None]
        if not miss:
            bases = {v[2] for v in got.values()}
            basis = "fiscal year" if bases == {"fiscal year"} else "four quarters"
            return ({k: v[0] for k, v in got.items()}, max(v[1] for v in got.values()),
                    f"12 months to {_label_end(e)} ({basis})", e, "")
        missing.append(_missing(book, miss, f"the 12 months to {_label_end(e)}"))
    return None, None, None, None, "; ".join(missing) or "no period reported"


def _balances(book: _Book, keys: Sequence[str], ends: Sequence[Optional[pd.Timestamp]]) -> tuple:
    """({key: value}, filed, period text, end, note) for balances at the same date (the first of `ends` with all)."""
    missing: List[str] = []
    for e in dict.fromkeys(x for x in ends if x is not None):
        got = {k: book.balance(k, e) for k in keys}
        miss = [k for k, v in got.items() if v is None]
        if not miss:
            return ({k: v[0] for k, v in got.items()}, max(v[1] for v in got.values()), f"at {_label_end(e)}", e, "")
        missing.append(_missing(book, miss, _label_end(e)))
    return None, None, None, None, "; ".join(missing) or "no period reported"


def _metric(key, label, kind, formula, value=None, period="", filed=None, note="") -> IndustryMetric:
    """A metric; a missing or non-finite value is None, with no filing date and "n/a" for a period not known."""
    ok = value is not None and np.isfinite(value)
    return IndustryMetric(key, label, float(value) if ok else None, kind, (period or "n/a"), filed if ok else None,
                          formula, note or "")


def _ratio(a: Optional[float], b: Optional[float], positive: bool = True) -> Optional[float]:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0 or (positive and b < 0):
        return None
    return float(a / b)


# --------------------------------------------------------------------------- #
# The four industries
# --------------------------------------------------------------------------- #
def _bank(book: _Book, ends, market_cap: Optional[float]) -> List[IndustryMetric]:
    out: List[IndustryMetric] = []
    v, filed, period, end, note = _flows(book, ["net_interest_income"], ends)
    out.append(_metric("net_interest_income", "Net interest income", "money", "InterestIncomeExpenseNet",
                       v and v["net_interest_income"], period, filed, note))
    # NII over the average of the total assets at the period's end and a year before
    nim = None
    if v is not None:
        a1 = book.balance("total_assets", end)
        prev = book.year_before("total_assets", end)
        a0 = book.balance("total_assets", prev) if prev is not None else None
        if a1 is not None and a0 is not None:
            nim = _metric("nii_to_assets", "NII / average assets", "ratio",
                          "net interest income / mean of total assets at the start and end (a net interest margin "
                          "proxy: XBRL has no earning-assets total)",
                          _ratio(v["net_interest_income"], (a1[0] + a0[0]) / 2.0), period, max(filed, a1[1], a0[1]))
        else:
            nim = _metric("nii_to_assets", "NII / average assets", "ratio", "net interest income / average total "
                          "assets", note=f"total assets not tagged at {_label_end(end)} and a year before")
    out.append(nim or _metric("nii_to_assets", "NII / average assets", "ratio",
                              "net interest income / average total assets", note=note))
    # efficiency: non-interest expense over revenue net of interest expense (NII + non-interest income, else the
    # company's own net-revenue line)
    v, filed, period, end, note = _flows(book, ["noninterest_expense", "net_interest_income", "noninterest_income"], ends)
    formula = "NoninterestExpense / (InterestIncomeExpenseNet + NoninterestIncome)"
    if v is not None:
        eff = _ratio(v["noninterest_expense"], v["net_interest_income"] + v["noninterest_income"])
    else:
        v2, filed2, period2, _, note2 = _flows(book, ["noninterest_expense", "revenue_net_of_interest"], ends)
        if v2 is not None:
            eff, filed, period, note = _ratio(v2["noninterest_expense"], v2["revenue_net_of_interest"]), filed2, period2, ""
            formula = "NoninterestExpense / RevenuesNetOfInterestExpense"
        else:
            eff = None
    out.append(_metric("efficiency_ratio", "Efficiency ratio", "ratio", formula, eff, period, filed, note))
    v, filed, period, end, note = _balances(book, ["loans", "deposits"], ends)
    out.append(_metric("loans_to_deposits", "Loans / deposits", "ratio", "loans (gross where tagged) / Deposits",
                       v and _ratio(v["loans"], v["deposits"]), period, filed, note))
    out += _tangible_book(book, ends, market_cap)
    return out


def _tangible_book(book: _Book, ends, market_cap: Optional[float]) -> List[IndustryMetric]:
    """Tangible book value per share = (equity - preferred - goodwill - other intangibles) / cover-page shares, every
    component at the same balance-sheet date (before the deferred-tax offset some banks apply to goodwill). The
    intangibles are the total line, else the finite-lived plus (when tagged) the indefinite-lived ones; a combined
    goodwill-and-intangibles line replaces both."""
    label, formula = "Tangible book value / share", "(StockholdersEquity - preferred - Goodwill - intangibles) / shares"
    notes: List[str] = []
    for e in dict.fromkeys(x for x in ends if x is not None):
        eq, pref = book.balance("equity", e), book.balance("preferred_equity", e)
        pref_note = ""
        if pref is None and eq is not None and book.never_tagged("preferred_equity"):
            pref, pref_note = (0.0, eq[1]), "no preferred stock tagged in any filing: counted as 0"
        gwi = book.balance("goodwill_and_intangibles", e)
        gw = book.balance("goodwill", e)
        intang = book.balance("intangibles", e)
        if intang is None:
            fin, ind = book.balance("finite_intangibles", e), book.balance("indefinite_intangibles", e)
            if fin is not None:
                intang = (fin[0] + (ind[0] if ind is not None else 0.0), max(fin[1], ind[1] if ind else fin[1]))
        deduct = gwi if gwi is not None else ((gw[0] + intang[0], max(gw[1], intang[1]))
                                              if gw is not None and intang is not None else None)
        miss = [n for n, x in (("equity", eq), ("preferred stock", pref), ("goodwill", gw if gwi is None else gwi),
                               ("intangible assets", intang if gwi is None else gwi)) if x is None]
        if miss:
            notes.append(f"not tagged at {_label_end(e)}: " + ", ".join(miss))
            continue
        tbv = eq[0] - pref[0] - deduct[0]
        filed = max(eq[1], pref[1], deduct[1])
        sh = book.cover_shares(filed)
        sh_note = "cover-page shares"
        if sh is None:
            sh = book.balance("shares_outstanding", e)
            sh_note = "CommonStockSharesOutstanding"
        if sh is None or sh[0] <= 0:
            return [_metric("tbv_per_share", label, "per_share", formula, note=f"no share count by {_label_end(e)}"),
                    _metric("price_to_tbv", "Price / tangible book", "multiple", "market value / tangible book",
                            note="no tangible book value")]
        per = _metric("tbv_per_share", label, "per_share", formula + f" ({sh_note})", tbv / sh[0], f"at {_label_end(e)}",
                      max(filed, sh[1]), pref_note)
        ptbv = _metric("price_to_tbv", "Price / tangible book", "multiple", "market value / tangible book",
                       _ratio(market_cap, tbv), f"tangible book at {_label_end(e)}", filed,
                       "; ".join(x for x in (pref_note, "" if market_cap else "no market value") if x))
        return [per, ptbv]
    note = "; ".join(notes) or "no period reported"
    return [_metric("tbv_per_share", label, "per_share", formula, note=note),
            _metric("price_to_tbv", "Price / tangible book", "multiple", "market value / tangible book",
                    note="no tangible book value")]


def _reit(book: _Book, ends, market_cap: Optional[float]) -> List[IndustryMetric]:
    v, filed, period, end, note = _flows(book, ["net_income", "d_and_a", "real_estate_gains"], ends)
    formula = "net income + depreciation & amortisation - gains on property sales + real-estate impairments (NAREIT)"
    ffo, imp = None, None
    if v is not None:
        imp = book.twelve_months("real_estate_impairment", end)
        ffo = v["net_income"] + v["d_and_a"] - v["real_estate_gains"] + (imp[0] if imp is not None else 0.0)
        filed = max(filed, imp[1]) if imp is not None else filed
        if imp is None:
            note = "no real-estate impairment tagged for the period: none added back"
    p_note = note if ffo is None else "no market value" if not market_cap else "FFO not positive" if ffo <= 0 else ""
    out = [_metric("ffo", "Funds from operations (FFO)", "money", formula, ffo, period, filed, note),
           _metric("p_ffo", "Price / FFO", "multiple", "market value / FFO",
                   _ratio(market_cap, ffo) if ffo is not None and ffo > 0 else None, period, filed, p_note)]
    if v is not None:
        d = book.twelve_months("common_dividends", end)
        out.append(_metric("ffo_payout", "FFO payout (common dividends)", "ratio", "common dividends paid / FFO",
                           _ratio(d[0], ffo) if d is not None else None, period, max(filed, d[1]) if d else None,
                           "" if d is not None else _missing(book, ["common_dividends"], f"the 12 months to {_label_end(end)}")))
        out.append(_metric("re_impairments", "Real-estate impairments (added back in FFO)", "money", "ImpairmentOfRealEstate",
                           imp[0] if imp is not None else None, period, imp[1] if imp else None,
                           "" if imp is not None else "not tagged for the period (none reported under the standard "
                                                      "concept)"))
    else:
        out.append(_metric("ffo_payout", "FFO payout (common dividends)", "ratio", "common dividends paid / FFO",
                           note="no FFO"))
    return out


def _deferred(book: _Book, e: pd.Timestamp) -> Optional[Tuple[str, float, pd.Timestamp]]:
    """(basis, value, filed) of deferred revenue at `e`: the total line, else current + non-current, else current."""
    tot = book.balance("deferred_revenue", e)
    cur, non = book.balance("deferred_revenue_current", e), book.balance("deferred_revenue_noncurrent", e)
    out = {}
    if tot is not None:
        out["total"] = tot
    elif cur is not None and non is not None:
        out["total"] = (cur[0] + non[0], max(cur[1], non[1]))
    if cur is not None:
        out["current"] = cur
    return out or None


def _software(book: _Book, ends) -> List[IndustryMetric]:
    out: List[IndustryMetric] = []
    v, filed, period, end, note = _balances(book, ["rpo"], ends)
    out.append(_metric("rpo", "Remaining performance obligations", "money", "RevenueRemainingPerformanceObligation",
                       v and v["rpo"], period, filed, note))
    if v is not None:
        rev = book.twelve_months("revenue", end)
        out.append(_metric("rpo_to_revenue", "RPO / revenue", "multiple", "RPO / revenue over the 12 months to the same date",
                           _ratio(v["rpo"], rev[0]) if rev else None, period, max(filed, rev[1]) if rev else None,
                           "" if rev else _missing(book, ["revenue"], f"the 12 months to {_label_end(end)}")))
        prev = book.year_before("rpo", end)
        p = book.balance("rpo", prev) if prev is not None else None
        out.append(_metric("rpo_growth", "RPO growth (year on year)", "ratio", f"RPO {_label_end(end)} / a year before - 1",
                           _ratio(v["rpo"], p[0]) - 1.0 if p and _ratio(v["rpo"], p[0]) is not None else None,
                           period, filed, "" if p else "no RPO tagged a year before"))
    else:
        out.append(_metric("rpo_to_revenue", "RPO / revenue", "multiple", "RPO / revenue", note=note))
    # deferred revenue growth, on one basis at both dates (total if both have it, else the current portion)
    dr = None
    for e in dict.fromkeys(x for x in ends if x is not None):
        now = _deferred(book, e)
        prev = book.year_before("deferred_revenue_current", e) or book.year_before("deferred_revenue", e)
        then = _deferred(book, prev) if prev is not None else None
        if now and then:
            basis = "total" if "total" in now and "total" in then else "current" if "current" in now and "current" in then else None
            if basis:
                g = _ratio(now[basis][0], then[basis][0])
                dr = _metric("deferred_revenue_growth", "Deferred revenue growth (year on year)", "ratio",
                             f"deferred revenue ({basis}{' portion' if basis == 'current' else ''}) / a year before - 1",
                             g - 1.0 if g is not None else None, f"at {_label_end(e)} vs {_label_end(prev)}",
                             max(now[basis][1], then[basis][1]))
                break
    out.append(dr or _metric("deferred_revenue_growth", "Deferred revenue growth (year on year)", "ratio",
                             "deferred revenue / a year before - 1",
                             note="deferred revenue (ContractWithCustomerLiability / DeferredRevenue, or their current "
                                  "portion) not tagged at both dates"))
    return out


def _insurer(book: _Book, ends) -> List[IndustryMetric]:
    v, filed, period, end, note = _flows(book, ["premiums_earned", "losses_and_benefits"], ends)
    out = [_metric("premiums_earned", "Net premiums earned", "money", "PremiumsEarnedNet",
                   v and v["premiums_earned"], period, filed, note),
           _metric("loss_ratio", "Loss ratio", "ratio", "losses, benefits and claims incurred / net premiums earned",
                   v and _ratio(v["losses_and_benefits"], v["premiums_earned"]), period, filed, note)]
    if v is None:                         # premiums alone, when the losses are not tagged
        pv, filed, period, end, pnote = _flows(book, ["premiums_earned"], ends)
        if pv is not None:
            out[0] = _metric("premiums_earned", "Net premiums earned", "money", "PremiumsEarnedNet",
                             pv["premiums_earned"], period, filed)
    growth = _metric("premium_growth", "Premium growth (year on year)", "ratio", "premiums earned / a year before - 1",
                     note="no premiums a year before")
    if out[0].value is not None:
        prev = book.year_before("premiums_earned", end)
        p = book.twelve_months("premiums_earned", prev) if prev is not None else None
        if p is not None:
            g = _ratio(out[0].value, p[0])
            growth = _metric("premium_growth", "Premium growth (year on year)", "ratio",
                             "premiums earned / the 12 months a year before - 1", g - 1.0 if g is not None else None,
                             out[0].period, max(out[0].filed, p[1]))
    out.append(growth)
    return out


def industry_metrics(facts: pd.DataFrame, sic, date, market_cap: Optional[float] = None) -> Optional[IndustryView]:
    """The industry metrics on `date` from a company's XBRL facts (`edgar.company_facts`), using only facts filed
    before `date`; None when the SIC code is none of the four industries. `market_cap` (the memo's, on `date`) adds
    P/TBV and P/FFO."""
    kind = industry_for_sic(sic)
    if kind is None:
        return None
    book = _Book(facts, date)
    latest, year_end = book.latest_end(), book.latest_year_end()
    notes: List[str] = []
    if latest is None:
        notes.append(f"no filings before {pd.Timestamp(date).date()}")
    ends = [latest, year_end]
    if kind == "bank":
        metrics = _bank(book, ends, market_cap)
    elif kind == "reit":
        metrics = _reit(book, ends, market_cap)
    elif kind == "software":
        metrics = _software(book, ends)
    else:
        metrics = _insurer(book, ends)
    return IndustryView(kind, str(sic), pd.Timestamp(date), metrics, notes)


__all__ = ["industry_for_sic", "industry_metrics", "IndustryMetric", "IndustryView", "SECTOR_OF_INDUSTRY",
           "INDUSTRY_LABELS"]
