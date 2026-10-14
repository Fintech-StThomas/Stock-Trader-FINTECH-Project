"""
Point-in-time financial statements from SEC XBRL facts, keyed by the date each number became public.

Every XBRL fact carries the filing that reported it and that filing's date. A company reports the same period many
times (the 10-Q, the 10-K's comparative columns, restatements); this module keeps the value **as first reported** and
dates it by that first filing. A backtest that reads a value only on or after its filing date can never use a number
before investors could see it, and never uses a later restatement early. (The cost: a restated value is never used, even
after it became public. That is conservative, not look-ahead.)

Quarterly figures: income-statement items usually come as 3-month values in the 10-Q; cash-flow items only as
year-to-date values (3, 6, 9 months), and the fourth quarter only as the full year in the 10-K. Quarters are derived by
differencing year-to-date values with the same start date, and dated by the later of the two filings.

XBRL starts in 2009 for large companies and 2011 for everyone else; before a company's first XBRL filing its
fundamentals are unknown (the backtest's fundamental gate is then off).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .edgar import PERIODIC_FORMS
from .fundamentals import StatementHistory


@dataclass(frozen=True)
class LineItem:
    key: str
    concepts: Tuple[str, ...]     # XBRL concepts in priority order (the first one reported for a period wins)
    kind: str                     # "flow" (over a period), "stock" (at a date), "per_share" (flow per share)
    unit: str = "USD"


LINE_ITEMS: List[LineItem] = [
    # income statement
    LineItem("revenue", ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                         "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
                         "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense", "TotalRevenuesAndOtherIncome"), "flow"),
    LineItem("cost_of_revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices",
                                 "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"), "flow"),
    LineItem("gross_profit", ("GrossProfit",), "flow"),
    LineItem("rnd", ("ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"), "flow"),
    LineItem("sga", ("SellingGeneralAndAdministrativeExpense",), "flow"),
    LineItem("operating_income", ("OperatingIncomeLoss",), "flow"),
    LineItem("interest_expense", ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
                                  "InterestAndDebtExpense"), "flow"),
    LineItem("pretax_income", ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
                               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"), "flow"),
    LineItem("income_tax", ("IncomeTaxExpenseBenefit",), "flow"),
    LineItem("net_income", ("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"), "flow"),
    LineItem("eps", ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"), "per_share",
             "USD/shares"),
    LineItem("shares", ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"),
             "flow", "shares"),
    # cash flow
    LineItem("operating_cash_flow", ("NetCashProvidedByUsedInOperatingActivities",
                                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"), "flow"),
    LineItem("capex", ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"), "flow"),
    LineItem("d_and_a", ("DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                         "DepreciationAndAmortization", "Depreciation"), "flow"),
    LineItem("sbc", ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"), "flow"),
    LineItem("dividends_paid", ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"), "flow"),
    LineItem("buybacks", ("PaymentsForRepurchaseOfCommonStock",), "flow"),
    LineItem("stock_issued", ("ProceedsFromIssuanceOfCommonStock",), "flow"),
    LineItem("acquisitions", ("PaymentsToAcquireBusinessesNetOfCashAcquired",), "flow"),
    # balance sheet
    LineItem("total_assets", ("Assets",), "stock"),
    LineItem("current_assets", ("AssetsCurrent",), "stock"),
    LineItem("cash", ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                      "Cash"), "stock"),
    LineItem("short_investments", ("ShortTermInvestments", "MarketableSecuritiesCurrent",
                                   "AvailableForSaleSecuritiesDebtSecuritiesCurrent"), "stock"),
    LineItem("receivables", ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"), "stock"),
    LineItem("inventory", ("InventoryNet",), "stock"),
    LineItem("ppe", ("PropertyPlantAndEquipmentNet",), "stock"),
    LineItem("goodwill", ("Goodwill",), "stock"),
    LineItem("current_liabilities", ("LiabilitiesCurrent",), "stock"),
    LineItem("accounts_payable", ("AccountsPayableCurrent",), "stock"),
    LineItem("total_liabilities", ("Liabilities",), "stock"),
    LineItem("long_term_debt", ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt",
                                "LongTermDebtAndFinanceLeaseObligationsNoncurrent"), "stock"),
    LineItem("short_term_debt", ("LongTermDebtCurrent", "DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
                                 "ShortTermBorrowings", "CommercialPaper"), "stock"),
    LineItem("equity", ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "stock"),
    LineItem("retained_earnings", ("RetainedEarningsAccumulatedDeficit",), "stock"),
    LineItem("shares_outstanding", ("CommonStockSharesOutstanding",), "stock", "shares"),
    # debt maturities (10-K footnote), for the balance-sheet risk check
    LineItem("debt_due_1y", ("LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",), "stock"),
    LineItem("debt_due_2y", ("LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",), "stock"),
    LineItem("debt_due_3y", ("LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",), "stock"),
    LineItem("debt_due_4y", ("LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",), "stock"),
    LineItem("debt_due_5y", ("LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",), "stock"),
    LineItem("debt_due_later", ("LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive",), "stock"),
]
ITEMS: Dict[str, LineItem] = {li.key: li for li in LINE_ITEMS}
COVER_CONCEPTS = ("EntityCommonStockSharesOutstanding", "EntityPublicFloat")    # dei: cover page of each filing
CONCEPTS: List[str] = sorted({c for li in LINE_ITEMS for c in li.concepts} | set(COVER_CONCEPTS))

QUARTER_DAYS = (70, 110)
YEAR_DAYS = (340, 390)
SAME_FILING_DAYS = 5      # an item first filed more than this after its row's filing was not known at that date


def original_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Each (concept, unit, start, end) as first reported in a 10-K/10-Q (or 20-F/40-F) filing."""
    f = facts[facts["form"].isin(PERIODIC_FORMS)] if "form" in facts else facts
    f = f.sort_values(["filed", "accn"])
    keys = ["concept", "unit", "start", "end"]
    return f.drop_duplicates([k for k in keys if k in f.columns], keep="first")


def _days(f: pd.DataFrame) -> pd.Series:
    return (f["end"] - f["start"]).dt.days


def flow_values(f: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(quarterly, annual) values of one concept's duration facts, each indexed by period end with columns val, filed.
    Quarters: reported 3-month values, else the difference of two year-to-date values with the same start."""
    f = f.dropna(subset=["start"])
    if f.empty:
        empty = pd.DataFrame(columns=["val", "filed"])
        return empty, empty
    d = _days(f)
    annual = f[(d >= YEAR_DAYS[0]) & (d <= YEAR_DAYS[1])]
    annual = annual.sort_values("filed").drop_duplicates("end").set_index("end")[["val", "filed"]]
    direct = f[(d >= QUARTER_DAYS[0]) & (d <= QUARTER_DAYS[1])]
    direct = direct.sort_values("filed").drop_duplicates("end").set_index("end")[["val", "filed"]]
    derived = []
    ytd = f[(d >= QUARTER_DAYS[0]) & (d <= YEAR_DAYS[1])].sort_values("filed").drop_duplicates(["start", "end"])
    for _, g in ytd.groupby("start"):
        g = g.sort_values("end")
        ends, vals, filed = g["end"].to_list(), g["val"].to_numpy(float), g["filed"].to_list()
        for k in range(1, len(g)):
            gap = (ends[k] - ends[k - 1]).days
            if QUARTER_DAYS[0] <= gap <= QUARTER_DAYS[1]:
                derived.append((ends[k], vals[k] - vals[k - 1], max(filed[k], filed[k - 1])))
    der = pd.DataFrame(derived, columns=["end", "val", "filed"]).sort_values("filed").drop_duplicates("end").set_index("end")
    quarterly = direct.combine_first(der) if len(der) else direct
    return quarterly.sort_index(), annual.sort_index()


def stock_values(f: pd.DataFrame) -> pd.DataFrame:
    """Instant values of one concept, indexed by date, as first reported."""
    if f.empty:
        return pd.DataFrame(columns=["val", "filed"])
    return f.sort_values("filed").drop_duplicates("end").set_index("end")[["val", "filed"]].sort_index()


def _item_values(orig: pd.DataFrame, item: LineItem) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(quarterly, annual) values of a line item. Companies switch concepts over time and sometimes report the same
    number under two of them in different filings (equity "including non-controlling interests" in the 10-K, the plain
    concept only in a later 10-Q's comparative column). Each period therefore takes the value that became public
    first, whichever of the item's concepts carried it; equal dates go to the concept listed first."""
    q_parts, a_parts = [], []
    for rank, concept in enumerate(item.concepts):
        f = orig[(orig["concept"] == concept) & (orig["unit"] == item.unit)]
        if f.empty:
            continue
        if item.kind == "stock":
            q = a = stock_values(f)
        else:
            q, a = flow_values(f)
        q_parts.append(q.assign(rank=rank))
        a_parts.append(a.assign(rank=rank))

    def merge(parts):
        parts = [p for p in parts if len(p)]
        if not parts:
            return pd.DataFrame(columns=["val", "filed"])
        allp = pd.concat(parts).rename_axis("end").reset_index()
        allp = allp.sort_values(["filed", "rank"]).drop_duplicates("end", keep="first")
        return allp.set_index("end")[["val", "filed"]].sort_index()
    return merge(q_parts), merge(a_parts)


def _period_table(values: Dict[str, pd.DataFrame], ends: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per period end: each item's value, and the row's filing date (the first filing that reported the period's
    core numbers). Values first filed well after the row's date are dropped: they were not public on that date."""
    if not len(ends):
        return pd.DataFrame()
    vals = pd.DataFrame({k: v["val"].reindex(ends) for k, v in values.items()}, index=ends)
    filed = pd.DataFrame({k: pd.to_datetime(v["filed"]).reindex(ends) for k, v in values.items()}, index=ends)
    core = [c for c in ("revenue", "net_income", "eps", "operating_income", "total_assets", "equity") if c in filed]
    row_filed = filed[core].min(axis=1) if core else filed.min(axis=1)
    row_filed = row_filed.fillna(filed.min(axis=1))
    late = filed.gt(row_filed + pd.Timedelta(days=SAME_FILING_DAYS), axis=0)
    vals = vals.mask(late)
    vals["filed"] = row_filed
    vals.index.name = "period_end"
    return vals.dropna(subset=["filed"]).sort_index()


def build_statements(facts: pd.DataFrame, source: str = "SEC EDGAR XBRL") -> StatementHistory:
    """Annual and quarterly point-in-time statements from a company's XBRL facts (see the module docstring).

    Columns: every LINE_ITEMS key that was reported, the classic aliases the beacons use (total_debt = long + short
    debt, fcf = operating cash flow - capex, ebitda = operating income + D&A), the cover-page shares outstanding of the
    same filing (`cover_shares`), and `filed`: the date the row became public."""
    if facts is None or facts.empty:
        return StatementHistory(source=source)
    orig = original_facts(facts)
    q_vals, a_vals = {}, {}
    for item in LINE_ITEMS:
        q, a = _item_values(orig, item)
        if len(q):
            q_vals[item.key] = q
        if len(a):
            a_vals[item.key] = a
    # periods: fiscal years from the annual income statement; quarters from the quarterly one
    a_ends = pd.DatetimeIndex(sorted(set().union(*[a_vals[k].index for k in ("revenue", "net_income") if k in a_vals])))
    flow_q = [q_vals[k].index for k in ("revenue", "net_income", "eps") if k in q_vals]
    q_ends = pd.DatetimeIndex(sorted(set().union(*flow_q))) if flow_q else pd.DatetimeIndex([])
    annual = _period_table({k: v for k, v in a_vals.items()}, a_ends)
    quarterly = _period_table({k: v for k, v in q_vals.items()}, q_ends)
    cover = _cover_shares(facts)
    for tab in (annual, quarterly):
        if tab.empty:
            continue
        _add_aliases(tab)
        if cover is not None:
            tab["cover_shares"] = _match_cover(tab, cover)
    return StatementHistory(annual=annual, quarterly=quarterly, source=source)


def _add_aliases(tab: pd.DataFrame) -> None:
    """Derived columns, each computed only from numbers in the same row (so from the same filing)."""
    lt = tab["long_term_debt"] if "long_term_debt" in tab else pd.Series(np.nan, index=tab.index)
    st = tab["short_term_debt"] if "short_term_debt" in tab else pd.Series(np.nan, index=tab.index)
    tab["total_debt"] = lt.fillna(0.0) + st.fillna(0.0)
    tab.loc[lt.isna() & st.isna(), "total_debt"] = np.nan
    if {"operating_cash_flow", "capex"} <= set(tab.columns):
        tab["fcf"] = tab["operating_cash_flow"] - tab["capex"].fillna(0.0)
    if {"revenue", "cost_of_revenue"} <= set(tab.columns):
        gp = tab["revenue"] - tab["cost_of_revenue"]
        tab["gross_profit"] = tab["gross_profit"].fillna(gp) if "gross_profit" in tab else gp
    # operating income where the company reports none (many banks, pharma, energy firms): pre-tax income plus interest
    # expense, an EBIT approximation; `operating_income_is_proxy` marks those rows
    if "pretax_income" in tab:
        proxy = tab["pretax_income"] + (tab["interest_expense"].fillna(0.0) if "interest_expense" in tab else 0.0)
        oi = tab["operating_income"] if "operating_income" in tab else pd.Series(np.nan, index=tab.index)
        tab["operating_income_is_proxy"] = oi.isna() & proxy.notna()
        tab["operating_income"] = oi.fillna(proxy)
    if {"operating_income", "d_and_a"} <= set(tab.columns):
        tab["ebitda"] = tab["operating_income"] + tab["d_and_a"]


def _cover_shares(facts: pd.DataFrame) -> Optional[pd.DataFrame]:
    f = facts[(facts["concept"] == "EntityCommonStockSharesOutstanding") & (facts["unit"] == "shares")]
    if f.empty:
        return None
    # one value per filing (a company with several share classes reports each; their sum is the share count)
    g = f.groupby("accn").agg(val=("val", "sum"), filed=("filed", "min"), end=("end", "max")).sort_values("filed")
    return g


def _match_cover(tab: pd.DataFrame, cover: pd.DataFrame) -> pd.Series:
    """The cover-page share count reported with each row's filing (same filing date), else the latest one filed
    before it."""
    # two filings on one day (a 10-K/A and a 10-Q) can carry different counts: the most recent cover date wins
    c = cover.reset_index().dropna(subset=["filed", "val"]).sort_values(["filed", "end", "accn"])
    c = c.drop_duplicates("filed", keep="last")[["filed", "val"]]
    left = pd.DataFrame({"filed": pd.to_datetime(tab["filed"]).to_numpy(), "_row": np.arange(len(tab))}).sort_values("filed")
    m = pd.merge_asof(left, c, on="filed", direction="backward").sort_values("_row")
    return pd.Series(m["val"].to_numpy(), index=tab.index)


def ttm(quarterly: pd.DataFrame, column: str) -> pd.Series:
    """Trailing-twelve-month sum of a quarterly flow, defined only when the four quarters are consecutive."""
    if quarterly is None or quarterly.empty or column not in quarterly:
        return pd.Series(dtype=float)
    s = quarterly[column]
    idx = quarterly.index
    gaps = pd.Series(idx, index=idx).diff().dt.days
    consecutive = gaps.rolling(3).max() <= QUARTER_DAYS[1]
    out = s.rolling(4, min_periods=4).sum()
    return out.where(consecutive)


def as_of(statements: pd.DataFrame, date) -> pd.DataFrame:
    """The rows of a point-in-time table that were public on `date` (filed on or before it)."""
    if statements is None or statements.empty:
        return statements
    return statements[pd.to_datetime(statements["filed"]) <= pd.Timestamp(date)]


def load_statements(ticker: str, source: str = "auto", refresh: bool = False) -> StatementHistory:
    """Point-in-time statements for a ticker from EDGAR (cache, live, or the sample pack's recorded facts).
    Returns an empty StatementHistory when there are none (a non-US filer, an ETF, no connection)."""
    from .edgar import company_facts
    from .data import DataUnavailable
    try:
        facts = company_facts(ticker=ticker, refresh=refresh, source=source)
    except DataUnavailable as exc:
        h = StatementHistory(source="unavailable")
        h.notes.append(f"EDGAR statements unavailable: {exc}")
        return h
    except Exception as exc:  # noqa: BLE001 - missing configuration (SEC e-mail) or a malformed reply
        h = StatementHistory(source="unavailable")
        h.notes.append(f"EDGAR statements unavailable: {type(exc).__name__}: {str(exc)[:160]}")
        return h
    return build_statements(facts)
