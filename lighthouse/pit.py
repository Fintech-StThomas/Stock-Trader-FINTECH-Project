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
fundamentals are unknown (the memo then has no statements to read).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from . import store
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
                                   "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent"),
             "stock"),
    LineItem("cash_and_investments", ("CashCashEquivalentsAndShortTermInvestments",), "stock"),
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

# Industry line items, read by `lighthouse.analyst.industry` for the companies whose SIC code calls for them (banks,
# REITs, software / subscription, insurers). They are not columns of `build_statements` (the generic statements, the
# fundamental score and the research signals never see them) but they are in CONCEPTS, so the facts cache downloads
# them. Concepts chosen from what large filers actually tag (JPM, BAC, WFC, USB; O, PLD, SPG; CRM, ADBE, NOW; TRV, ALL,
# PGR, MET, UNH; companyfacts read in October 2026). Loans list the gross concepts first; the net-of-allowance ones
# only fill periods a bank tagged no gross total.
INDUSTRY_ITEMS: List[LineItem] = [
    # banks
    LineItem("net_interest_income", ("InterestIncomeExpenseNet",), "flow"),
    LineItem("noninterest_income", ("NoninterestIncome",), "flow"),
    LineItem("noninterest_expense", ("NoninterestExpense",), "flow"),
    LineItem("revenue_net_of_interest", ("RevenuesNetOfInterestExpense",), "flow"),
    LineItem("loans", ("FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
                       "LoansAndLeasesReceivableNetOfDeferredIncome", "LoansAndLeasesReceivableGrossCarryingAmount",
                       "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
                       "LoansAndLeasesReceivableNetReportedAmount"), "stock"),
    LineItem("deposits", ("Deposits",), "stock"),
    LineItem("preferred_equity", ("PreferredStockValue", "PreferredStockIncludingAdditionalPaidInCapitalNetOfDiscount",
                                  "PreferredStockIncludingAdditionalPaidInCapital", "PreferredStockValueOutstanding"),
             "stock"),
    LineItem("intangibles", ("IntangibleAssetsNetExcludingGoodwill", "OtherIntangibleAssetsNet"), "stock"),
    LineItem("finite_intangibles", ("FiniteLivedIntangibleAssetsNet",), "stock"),
    LineItem("indefinite_intangibles", ("IndefiniteLivedIntangibleAssetsExcludingGoodwill",), "stock"),
    LineItem("goodwill_and_intangibles", ("IntangibleAssetsNetIncludingGoodwill",), "stock"),
    # REITs
    LineItem("real_estate_gains", ("GainLossOnSaleOfProperties", "GainsLossesOnSalesOfInvestmentRealEstate"), "flow"),
    LineItem("real_estate_impairment", ("ImpairmentOfRealEstate",), "flow"),
    LineItem("common_dividends", ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends", "DividendsCommonStockCash"),
             "flow"),
    # software / subscription
    LineItem("rpo", ("RevenueRemainingPerformanceObligation",), "stock"),
    LineItem("deferred_revenue", ("ContractWithCustomerLiability", "DeferredRevenue"), "stock"),
    LineItem("deferred_revenue_current", ("ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"), "stock"),
    LineItem("deferred_revenue_noncurrent", ("ContractWithCustomerLiabilityNoncurrent", "DeferredRevenueNoncurrent"),
             "stock"),
    # insurers (property & casualty, life, health)
    LineItem("premiums_earned", ("PremiumsEarnedNet", "PremiumsEarnedNetPropertyAndCasualty"), "flow"),
    LineItem("losses_and_benefits", ("PolicyholderBenefitsAndClaimsIncurredNet", "IncurredClaimsPropertyCasualtyAndLiability",
                                     "PolicyholderBenefitsAndClaimsIncurredHealthCare"), "flow"),
]
INDUSTRY: Dict[str, LineItem] = {li.key: li for li in INDUSTRY_ITEMS}
COVER_CONCEPTS = ("EntityCommonStockSharesOutstanding", "EntityPublicFloat")    # dei: cover page of each filing
CONCEPTS: List[str] = sorted({c for li in LINE_ITEMS + INDUSTRY_ITEMS for c in li.concepts} | set(COVER_CONCEPTS))

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
    f = f[f["start"].notna()]
    if f.empty:
        empty = pd.DataFrame(columns=["val", "filed"])
        return empty, empty
    d = _days(f)
    annual = f[(d >= YEAR_DAYS[0]) & (d <= YEAR_DAYS[1])]
    annual = annual.sort_values("filed").drop_duplicates("end").set_index("end")[["val", "filed"]]
    direct = f[(d >= QUARTER_DAYS[0]) & (d <= QUARTER_DAYS[1])]
    direct = direct.sort_values("filed").drop_duplicates("end").set_index("end")[["val", "filed"]]
    ytd = f[(d >= QUARTER_DAYS[0]) & (d <= YEAR_DAYS[1])].sort_values("filed").drop_duplicates(["start", "end"])
    # consecutive year-to-date values with the same start, one quarter apart: their difference is that quarter
    ytd = ytd.sort_values(["start", "end"], kind="stable")
    gap = ytd["end"].diff().dt.days.to_numpy()
    ok = (ytd["start"] == ytd["start"].shift()).to_numpy() & (gap >= QUARTER_DAYS[0]) & (gap <= QUARTER_DAYS[1])
    val, filed = ytd["val"].to_numpy(float), ytd["filed"]
    before = filed.shift()
    der = pd.DataFrame({"end": ytd["end"].to_numpy()[ok], "val": (val - np.roll(val, 1))[ok],
                        "filed": filed.where(filed >= before, before).to_numpy()[ok]})
    der = der.sort_values("filed").drop_duplicates("end").set_index("end")
    quarterly = direct.combine_first(der) if len(der) else direct
    return quarterly.sort_index(), annual.sort_index()


def stock_values(f: pd.DataFrame) -> pd.DataFrame:
    """Instant values of one concept, indexed by date, as first reported."""
    if f.empty:
        return pd.DataFrame(columns=["val", "filed"])
    return f.sort_values("filed").drop_duplicates("end").set_index("end")[["val", "filed"]].sort_index()


def concept_groups(orig: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """{(concept, unit): its facts (start, end, val, filed)} of `original_facts(...)`, split once: every line item
    then looks its concepts up instead of filtering the whole table again (about a hundred times per company)."""
    if orig is None or orig.empty:
        return {}
    slim = orig[["concept", "unit", "start", "end", "val", "filed"]]
    return {(str(c), str(u)): g[["start", "end", "val", "filed"]]
            for (c, u), g in slim.groupby(["concept", "unit"], sort=False)}


def _item_values(groups: Dict[Tuple[str, str], pd.DataFrame], item: LineItem) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(quarterly, annual) values of a line item, from `concept_groups(original_facts(...))`. Companies switch concepts over time and sometimes report the same
    number under two of them in different filings (equity "including non-controlling interests" in the 10-K, the plain
    concept only in a later 10-Q's comparative column). Each period therefore takes the value that became public
    first, whichever of the item's concepts carried it; equal dates go to the concept listed first."""
    q_parts, a_parts = [], []
    for rank, concept in enumerate(item.concepts):
        f = groups.get((concept, item.unit))
        if f is None or f.empty:
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


def item_values(orig, item: LineItem) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(quarterly, annual) values of one line item, each indexed by period end with `val` and `filed` (the filing
    that first made the value public; for a quarter derived from two year-to-date values, the later of the two).
    `orig` is `original_facts(...)`, or `concept_groups` of it when many items are read from one company (the split
    is then made once). The public form of the per-item step of `build_statements`, for metrics that
    need each value's own filing date (`lighthouse.analyst.industry`)."""
    return _item_values(orig if isinstance(orig, dict) else concept_groups(orig), item)


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
    same filing or the latest before it (`cover_shares`, filed on `cover_filed`), and `filed`: the date the row became
    public."""
    if facts is None or facts.empty:
        return StatementHistory(source=source)
    groups = concept_groups(original_facts(facts))
    q_vals, a_vals = {}, {}
    for item in LINE_ITEMS:
        q, a = _item_values(groups, item)
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
    cover = cover_shares(facts)
    for tab in (annual, quarterly):
        if tab.empty:
            continue
        _add_aliases(tab)
        if len(cover):
            tab[["cover_shares", "cover_filed"]] = _match_cover(tab, cover)
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


def cover_shares(facts: pd.DataFrame) -> pd.Series:
    """Shares outstanding from each filing's cover page (dei:EntityCommonStockSharesOutstanding), indexed by the date
    it was filed: the count public from that date. A company with several share classes reports each; their sum is
    the count. Two filings on one day (a 10-K/A and a 10-Q) can carry different counts: the latest cover date wins.
    The only reader of the cover page: the statements (`cover_shares` column) and membership's market cap use it."""
    f = facts[(facts["concept"] == "EntityCommonStockSharesOutstanding") & (facts["unit"] == "shares")]
    if f.empty:
        return pd.Series(dtype=float, name="shares")
    g = f.groupby("accn").agg(val=("val", "sum"), filed=("filed", "min"), end=("end", "max")).reset_index()
    g = g.dropna(subset=["filed", "val"]).sort_values(["filed", "end", "accn"]).drop_duplicates("filed", keep="last")
    return pd.Series(g["val"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(g["filed"])), name="shares")


def _match_cover(tab: pd.DataFrame, cover: pd.Series) -> pd.DataFrame:
    """The cover-page share count reported with each row's filing (same filing date), else the latest one filed
    before it, with the date that count was filed (`cover_filed`: a split after it changes the count's units)."""
    c = pd.DataFrame({"filed": cover.index, "val": cover.to_numpy(), "cover_filed": cover.index})
    left = pd.DataFrame({"filed": pd.to_datetime(tab["filed"]).to_numpy(), "_row": np.arange(len(tab))}).sort_values("filed")
    m = pd.merge_asof(left, c, on="filed", direction="backward").sort_values("_row")
    return pd.DataFrame({"cover_shares": m["val"].to_numpy(), "cover_filed": m["cover_filed"].to_numpy()}, index=tab.index)


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
    Returns an empty StatementHistory when there are none (a non-US filer, an ETF, no connection).

    Building the tables from the facts takes most of a second per company, so the result is kept on disk
    (`cached_statements`) and reused while the facts and this module are unchanged; `refresh=True` rebuilds it."""
    from .edgar import company_facts, normalize_ticker
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
    return cached_statements(normalize_ticker(ticker), facts, refresh=refresh)


# --------------------------------------------------------------------------- #
# The built statements, kept on disk
# --------------------------------------------------------------------------- #
_STAMP_COLUMN = "_built_from"
_code_stamp_value: List[str] = []


def _code_stamp() -> str:
    """A fingerprint of the code that builds statements: this file and the forms it reads. Any edit to this module
    changes it, so a cached table can never outlive the rules that made it."""
    if not _code_stamp_value:
        import hashlib
        from pathlib import Path
        h = hashlib.sha1(Path(__file__).read_bytes())
        h.update(repr(PERIODIC_FORMS).encode())
        _code_stamp_value.append(h.hexdigest()[:16])
    return _code_stamp_value[0]


def facts_fingerprint(facts: pd.DataFrame) -> str:
    """Identifies one facts table and the code that reads it: a hash of every row (in order) and `_code_stamp`."""
    import hashlib
    rows = pd.util.hash_pandas_object(facts.reset_index(drop=True), index=True).to_numpy()
    h = hashlib.sha1(rows.tobytes())
    h.update(",".join(map(str, facts.columns)).encode())
    return f"{h.hexdigest()[:24]}-{_code_stamp()}"


def _read_built(key: str, stamp: str):
    tab = store.read_frame("edgar_statements", key)
    if tab is None or _STAMP_COLUMN not in tab.columns or not len(tab) or not (tab[_STAMP_COLUMN] == stamp).all():
        return None
    tab = tab.drop(columns=_STAMP_COLUMN)
    for c in tab.columns:                          # one date unit, whichever pandas wrote the file (see edgar._ns)
        if pd.api.types.is_datetime64_any_dtype(tab[c]):
            tab[c] = tab[c].astype("datetime64[ns]")
    tab.index = pd.DatetimeIndex(tab.index).astype("datetime64[ns]")
    tab.index.name = "period_end"
    return tab


def cached_statements(key: str, facts: pd.DataFrame, refresh: bool = False) -> StatementHistory:
    """`build_statements(facts)`, read from data/cache/edgar_statements/<key>_{annual,quarterly}.parquet when those
    tables were built from exactly these facts by this version of the module (each file carries the fingerprint it
    was built from, so a half-finished write or a changed rule is rebuilt, never read). A company without both tables
    is built each time (there is little to build)."""
    if facts is None or facts.empty:
        return build_statements(facts)
    try:
        stamp = facts_fingerprint(facts)
    except Exception:  # noqa: BLE001 - an unhashable table is simply built
        return build_statements(facts)
    if not refresh:
        annual, quarterly = _read_built(f"{key}_annual", stamp), _read_built(f"{key}_quarterly", stamp)
        if annual is not None and quarterly is not None:
            return StatementHistory(annual=annual, quarterly=quarterly, source="SEC EDGAR XBRL")
    h = build_statements(facts)
    if len(h.annual) and len(h.quarterly):
        try:
            for name, tab in (("annual", h.annual), ("quarterly", h.quarterly)):
                store.write_frame("edgar_statements", f"{key}_{name}", tab.assign(**{_STAMP_COLUMN: stamp}))
        except Exception:  # noqa: BLE001 - a cache that cannot be written (read-only disk) costs time, not results
            pass
    return h
