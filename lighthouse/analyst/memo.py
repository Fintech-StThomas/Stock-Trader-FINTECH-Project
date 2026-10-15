"""
The investment memo: everything the toolkit knows about one stock on one date, and a recommendation built from it.

Sections   the company (SIC industry, fiscal year, filings used); the numbers (TTM, growth, margins, returns); for
           banks, REITs, software companies and insurers the industry's own metrics (`analyst.industry`); the
           valuation (bull / base / bear DCF, probability-weighted target, upside-to-downside, reverse DCF, the stock's
           own valuation bands, peers); quality and risk (earnings-quality flags, balance sheet, debt coming due);
           capital allocation; catalysts; the variant view (where the base case differs from what the price implies);
           the thesis, its supporting points and risks; what would prove it wrong (checkable kill criteria, followed by
           `lighthouse.analyst.thesis`); a pre-mortem.
Rules      every number comes from structured data (filings, prices, FRED, Ken French) with its source; the text is
           templated from those numbers, so the same inputs always give the same memo. The recommendation combines
           the rule-based verdict (when an `AnalysisResult` is given) with the valuation; it is not a forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..fundamentals import FundamentalSnapshot, fundamental_beacons, fundamental_score, sector_norms
from ..technicals import Beacon
from .checks import Flag, balance_sheet, capital_allocation, catalysts, earnings_quality
from .financials import Financials, financials_as_of
from .industry import SECTOR_OF_INDUSTRY, IndustryView, industry_for_sic, industry_metrics
from .peers import PeerComparison, choose_peers, compare, peer_norms as _peer_norms
from .valuation import Valuation, value


STANCE_RULE = ("Buy when the probability-weighted value is at least 15% above the price, the bull case gains at least "
               "1.5x what the bear case loses, no check is red and the rule-based signal is not Reduce or Sell; avoid "
               "when the value is more than 25% below the price (15% with a red flag or a Reduce / Sell signal); "
               "otherwise hold. A DCF that fades growth to the long-run rate calls most fast growers expensive: read "
               "the reverse DCF next to it.")


@dataclass
class Criterion:
    """A measurable condition that would prove the thesis wrong, checked against each new filing (or each close)."""
    metric: str                 # a key of Financials.ratios(), or "price"
    op: str                     # "<" or ">"
    threshold: float
    text: str

    def breached(self, value: float) -> bool:
        if value is None or not np.isfinite(value):
            return False
        return value < self.threshold if self.op == "<" else value > self.threshold

    def to_dict(self) -> dict:
        return {"metric": self.metric, "op": self.op, "threshold": float(self.threshold), "text": self.text}


@dataclass
class FundamentalView:
    """The fundamental beacons on the memo's own point-in-time numbers (`fundamental_snapshot`), scored against the
    peer medians when the memo found peers, next to the same numbers against the sector norms. Opt-in
    (`build(peer_norms=True)`, CLI `--peer-norms`); the rule-based verdict keeps the sector norms it was tested with."""
    beacons: List[Beacon]                 # against `norms`
    score: Optional[float]
    sector_beacons: List[Beacon]          # the same numbers against fundamentals.SECTOR_NORMS
    sector_score: Optional[float]
    norms: Dict[str, object]              # what the beacons were scored against (peer_norms, or the sector norms)
    source: str                           # which norms: the peers' source string, or why the sector norms were used
    sector: Optional[str] = None
    peer_based: bool = False


@dataclass
class Memo:
    ticker: str
    as_of: pd.Timestamp
    profile: dict
    financials: Financials
    valuation: Optional[Valuation]
    peers: Optional[PeerComparison]
    quality: List[Flag]
    balance: List[Flag]
    capital: Dict[str, object]
    catalysts: pd.DataFrame
    stance: str                                     # "buy" | "hold" | "avoid"
    thesis: str
    supporting: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    variant_view: List[str] = field(default_factory=list)
    criteria: List[Criterion] = field(default_factory=list)
    premortem: List[str] = field(default_factory=list)
    rule_verdict: Optional[str] = None
    rule_score: Optional[float] = None
    sources: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    fundamental_view: Optional[FundamentalView] = None   # opt-in: beacons against peer medians (build(peer_norms=True))
    industry: Optional[IndustryView] = None              # bank / REIT / software / insurer metrics, by SIC code

    def markdown(self) -> str:
        from ..report import build_memo
        return build_memo(self)


# --------------------------------------------------------------------------- #
# Peers
# --------------------------------------------------------------------------- #
def peer_candidates(source: str = "auto") -> List[str]:
    """Who can be a peer: today's S&P 500 and the curated universes (live); the sample pack (offline)."""
    from ..data import offline_mode
    from ..universe import UNIVERSES, get_universe, SAMPLE_UNIVERSE_NAME
    if source == "sample" or offline_mode():
        return get_universe(SAMPLE_UNIVERSE_NAME)
    names: List[str] = []
    try:
        from ..membership import sp500
        names += sp500().members_on(pd.Timestamp.today().normalize())
    except Exception:  # noqa: BLE001 - Wikipedia / GitHub unreachable: the curated lists alone
        pass
    for v in UNIVERSES.values():
        names += list(v)
    return list(dict.fromkeys(n.upper() for n in names if not n.startswith("^")))


def find_peers(ticker: str, f: Financials, sic: Optional[str], candidates: Iterable[str], source: str = "auto",
               n: int = 8) -> Optional[PeerComparison]:
    """Peers by SIC code then market value, each with its own point-in-time numbers on the memo's date."""
    from ..data import load_prices
    from ..edgar import company_filings
    from ..parallel import map_tickers
    from ..pit import load_statements
    from ..total_return import load_actions
    if not sic:
        return None
    cands = [c for c in candidates if c != ticker]

    def sic_of(t):
        return company_filings(t, source=source)[0].get("sic")
    sics, _ = map_tickers(sic_of, cands, max_workers=8, source=source)
    same = [t for t, s in sics.items() if s and str(s).zfill(4)[:2] == str(sic).zfill(4)[:2]]
    if not same:
        return None

    def numbers(t):
        px = load_prices(t, years=2, end=f.as_of, source=source)
        return financials_as_of(t, load_statements(t, source=source), px, load_actions(t, source=source), f.as_of)
    fin, _ = map_tickers(numbers, same, max_workers=6, source=source)
    caps = {t: x.market_cap for t, x in fin.items()}
    caps[ticker] = f.market_cap
    chosen, level = choose_peers(ticker, {**{t: sics[t] for t in fin}, ticker: sic}, caps, n=n)
    if not chosen:
        return None
    return compare(f, {t: fin[t] for t in chosen}, level)


# --------------------------------------------------------------------------- #
# Fundamental beacons on the memo's numbers (opt-in: against the peer medians)
# --------------------------------------------------------------------------- #
def _finite(x) -> Optional[float]:
    return float(x) if x is not None and np.isfinite(x) else None


def fundamental_snapshot(f: Financials, sector: Optional[str] = None) -> FundamentalSnapshot:
    """The beacons' inputs from the memo's point-in-time numbers (TTM flows, latest balance sheet, the price that day):
    the same definitions `Financials.ratios` gives the peers, so a peer median compares like with like. A loss gives
    P/E -1 (the snapshot's "negative earnings"); forward P/E and PEG need estimates, which free data lacks (None)."""
    r = f.ratios()
    pe = _finite(r["pe"])
    if pe is None and np.isfinite(f.get("net_income")) and f.get("net_income") <= 0 and f.market_cap > 0:
        pe = -1.0
    eq, debt = f.get("equity"), f.bs.get("total_debt")
    de = float(debt / eq) if debt is not None and np.isfinite(debt) and eq > 0 else None
    return FundamentalSnapshot(ticker=f.ticker, sector=sector, price=f.price, market_cap=_finite(f.market_cap),
                               pe_trailing=pe, pb=_finite(r["pb"]), ev_ebitda=_finite(r["ev_ebitda"]),
                               roe=_finite(r["roe"]), operating_margin=_finite(r["operating_margin"]),
                               profit_margin=_finite(r["net_margin"]), debt_to_equity=de,
                               current_ratio=_finite(r["current_ratio"]), fcf=_finite(f.fcf),
                               fcf_yield=_finite(r["fcf_yield"]), revenue_growth=_finite(r["revenue_growth"]),
                               eps_growth=_finite(r["eps_growth"]), as_of=str(f.as_of.date()),
                               source="SEC EDGAR XBRL, point in time")


def fundamental_view(f: Financials, peer_cmp: Optional[PeerComparison], sector: Optional[str] = None,
                     why_no_peers: str = "no peers found") -> FundamentalView:
    """The beacons on the memo's numbers against the peer medians (`peers.peer_norms`) when there are peers, else
    against the sector norms with the reason. The earnings-stability beacon is left out: EDGAR's annual EPS is as
    reported, so a share split between two years would read as an earnings collapse."""
    snap = fundamental_snapshot(f, sector)
    base = fundamental_beacons(snap)
    if peer_cmp is not None and peer_cmp.peers:
        norms = _peer_norms(peer_cmp)
        beacons = fundamental_beacons(snap, norms=norms)
        return FundamentalView(beacons, fundamental_score(beacons), base, fundamental_score(base), norms,
                               str(norms["source"]), sector, True)
    norms = dict(sector_norms(sector))
    return FundamentalView(base, fundamental_score(base), base, fundamental_score(base), norms,
                           f"sector norms for {sector or 'the overall market'} ({why_no_peers})", sector, False)


def _memo_sector(ticker: str, result, sic=None) -> Optional[str]:
    """The sector label the beacons use: the analysis snapshot's, else the curated list's, else the one the SIC code
    implies for a bank, insurer, REIT or software company; None (the default norms) when unknown."""
    from ..universe import UNKNOWN_SECTOR, sector_of
    s = getattr(getattr(result, "fundamentals", None), "sector", None) or sector_of(ticker, None)
    if not s or s == UNKNOWN_SECTOR:
        s = SECTOR_OF_INDUSTRY.get(industry_for_sic(sic) or "")
    return None if not s or s == UNKNOWN_SECTOR else s


def industry_view(ticker: str, sic, f: Financials, source: str = "auto",
                  notes: Optional[List[str]] = None) -> Optional[IndustryView]:
    """The industry metrics (`analyst.industry`) on the memo's date when the SIC code is a bank, REIT, software or
    insurer code; None otherwise or when the facts cannot be read (a note says why). The facts come from the same
    cache as the statements, and only those filed before the memo's date are used."""
    from ..data import DataUnavailable
    from ..edgar import company_facts
    if industry_for_sic(sic) is None:
        return None
    try:
        facts = company_facts(ticker=ticker, source=source)
    except Exception as exc:  # noqa: BLE001 - unreachable EDGAR, no SEC contact e-mail: the memo goes on without them
        if notes is not None:
            kind = "unavailable" if isinstance(exc, DataUnavailable) else type(exc).__name__
            notes.append(f"no industry metrics (facts {kind}: {str(exc)[:120]})")
        return None
    return industry_metrics(facts, sic, f.as_of, market_cap=f.market_cap if np.isfinite(f.market_cap) else None)


# --------------------------------------------------------------------------- #
# The memo
# --------------------------------------------------------------------------- #
def _pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def build(ticker: str, as_of=None, source: str = "auto", result=None, peers: bool = True,
          candidates: Optional[Iterable[str]] = None, factors: Optional[pd.DataFrame] = None,
          risk_free: Optional[float] = None, equity_premium: Optional[float] = None,
          peer_norms: bool = False) -> Memo:
    """The memo for `ticker` on `as_of` (default: the last close). `result` (an `engine.AnalysisResult`) adds the
    rule-based verdict; `peers=False` skips the peer search (it reads every same-industry candidate's filings).
    `peer_norms=True` adds the fundamental beacons on the memo's numbers scored against the peer medians (needs the
    peer search; without peers they are scored against the sector norms and say why). Nothing else changes: the
    stance, the rule-based verdict and the decision keep the sector norms."""
    from ..data import load_prices
    from ..edgar import company_filings
    from ..pit import load_statements
    from ..total_return import load_actions
    from ..data import DataUnavailable
    t = ticker.upper().strip()
    notes: List[str] = []
    prices = result.prices if result is not None else load_prices(t, years=11, end=as_of, source=source)
    actions = result.actions if result is not None and result.actions is not None else load_actions(t, source=source)
    statements = result.pit_statements if result is not None and result.pit_statements is not None else \
        load_statements(t, source=source)
    if statements.quarterly is None or statements.quarterly.empty:
        raise DataUnavailable(f"{t}: no SEC filings to build a memo from ({'; '.join(statements.notes) or 'not a US filer'})")
    f = financials_as_of(t, statements, prices, actions, as_of)
    notes += f.notes
    try:
        profile, filings = company_filings(t, source=source)
    except DataUnavailable as exc:
        profile, filings = {"name": t}, None
        notes.append(f"no filing list ({exc}): no catalysts or amendment check")
    close = prices["Close"][prices.index <= f.as_of]

    # valuation, with the stock's own bands
    val = None
    try:
        from ..research.fundamental_signals import daily_values
        dv = daily_values(t, prices[prices.index <= f.as_of], source=source)
        bands = {k: dv[k] for k in ("earnings_yield", "fcf_yield", "ebitda_ev") if k in dv}
    except Exception as exc:  # noqa: BLE001
        bands = {}
        notes.append(f"no valuation bands ({type(exc).__name__})")
    try:
        val = value(f, close, factors=factors, bands=bands,
                    rate=None if risk_free is None and equity_premium is None else _rate(f, close, risk_free,
                                                                                         equity_premium, factors))
    except Exception as exc:  # noqa: BLE001 - e.g. no revenue line: the memo still reports the rest
        notes.append(f"no DCF: {type(exc).__name__}: {str(exc)[:120]}")

    peer_cmp = None
    if peers:
        try:
            peer_cmp = find_peers(t, f, profile.get("sic"), candidates or peer_candidates(source), source=source)
            if peer_cmp is None:
                notes.append("no peers with prices and filings in the same SIC group")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"peer search failed: {type(exc).__name__}: {str(exc)[:120]}")

    q_flags = earnings_quality(f, filings)
    b_flags = balance_sheet(f)
    cap = capital_allocation(f)
    cats = catalysts(filings, f.as_of, profile.get("fiscalYearEnd"))
    view = None
    if peer_norms:
        view = fundamental_view(f, peer_cmp, _memo_sector(t, result, profile.get("sic")),
                                why_no_peers="peer search off" if not peers else "no peers found")
    ind = industry_view(t, profile.get("sic"), f, source=source, notes=notes)
    memo = Memo(ticker=t, as_of=f.as_of, profile=profile, financials=f, valuation=val, peers=peer_cmp,
                quality=q_flags, balance=b_flags, capital=cap, catalysts=cats, stance="hold", thesis="",
                rule_verdict=getattr(result, "verdict", None), rule_score=getattr(result, "score", None), notes=notes,
                fundamental_view=view, industry=ind)
    _write_argument(memo)
    memo.sources = _sources(memo, prices)
    return memo


def _rate(f, close, risk_free, equity_premium, factors):
    from .valuation import discount_rate
    return discount_rate(f, close, risk_free=risk_free, equity_premium=equity_premium, factors=factors)


def _write_argument(m: Memo) -> None:
    """Stance, thesis, supporting points, risks, variant view, kill criteria and pre-mortem, from the numbers."""
    f, v, r = m.financials, m.valuation, m.financials.ratios()
    reds = [x for x in m.quality + m.balance if x.severity == "red"]
    greens = [x for x in m.quality + m.balance if x.severity == "green"]
    ambers = [x for x in m.quality + m.balance if x.severity == "amber"]

    # stance (STANCE_RULE): valuation first; a red flag or a Reduce / Sell signal blocks a buy and lowers the bar to avoid
    up = v.target / f.price - 1.0 if v is not None and f.price > 0 else np.nan
    ud = v.upside_downside if v is not None else np.nan
    negative = m.rule_verdict in ("Reduce", "Sell")              # config.VERDICT_BANDS
    if np.isfinite(up) and up > 0.15 and (not np.isfinite(ud) or ud > 1.5) and not reds and not negative:
        stance = "buy"
    elif np.isfinite(up) and (up < -0.25 or (up < -0.15 and (reds or negative))):
        stance = "avoid"
    else:
        stance = "hold"
    m.stance = stance

    g_now = r["revenue_growth"]
    if v is not None:
        base = v.cases["base"]
        m.thesis = (f"At ${f.price:,.2f}, the probability-weighted DCF value is ${v.target:,.2f} ({up:+.0%}); the base case "
                    f"(revenue growth fading from {_pct(base.assumptions.growth[0])} to {_pct(base.assumptions.growth[-1])}, "
                    f"operating margin {_pct(base.assumptions.operating_margin[-1])}) is worth ${base.per_share:,.2f}. "
                    + (f"The price implies {_pct(v.implied_growth)} revenue growth a year for five years at today's "
                       f"margin." if v.implied_growth is not None else "No constant growth rate between -30% and +60% "
                       "reproduces the price at today's margin."))
    else:
        m.thesis = "No DCF could be built from the filings; the memo reports the quality and balance-sheet checks only."
    if m.rule_verdict:
        m.thesis += f" The rule-based signals say {m.rule_verdict}" + (f" (score {m.rule_score:.0f}/100)." if m.rule_score is not None else ".")

    sup = [x.text for x in greens]
    if np.isfinite(r["roic"]) and r["roic"] > 0.15:
        sup.append(f"Return on invested capital {_pct(r['roic'])}: the business earns well above a typical cost of capital.")
    if np.isfinite(r["fcf_yield"]) and r["fcf_yield"] > 0.05:
        sup.append(f"Free cash flow yield {_pct(r['fcf_yield'])} at today's price.")
    if v is not None and not v.bands.empty and "earnings_yield" in v.bands.index:
        p = v.bands.loc["earnings_yield", "percentile"]
        if p > 0.7:
            sup.append(f"The earnings yield is higher than on {p:.0%} of days since {v.bands.loc['earnings_yield', 'since']}: "
                       "cheap against its own history.")
    if m.peers is not None and "ev_ebitda" in m.peers.premium and m.peers.premium["ev_ebitda"] < -0.15:
        sup.append(f"EV/EBITDA is {abs(m.peers.premium['ev_ebitda']):.0%} below the peer median.")
    sup += list(m.capital.get("lines", []))[:1]
    m.supporting = sup

    risks = [x.text for x in reds + ambers]
    if v is not None:
        risks.append(f"{v.cases['base'].terminal_share:.0%} of the base-case value lies beyond year five (the terminal "
                     "value): small changes in long-run growth or the discount rate move it a lot.")
        if v.rate.beta > 1.2:
            risks.append(f"Beta {v.rate.beta:.2f}: the stock has moved more than the market.")
        if not v.bands.empty and "earnings_yield" in v.bands.index and v.bands.loc["earnings_yield", "percentile"] < 0.3:
            risks.append(f"The earnings yield is lower than on {1 - v.bands.loc['earnings_yield', 'percentile']:.0%} of days "
                         "in its last ten years: expensive against its own history.")
    if m.peers is not None and "ev_ebitda" in m.peers.premium and m.peers.premium["ev_ebitda"] > 0.25:
        risks.append(f"EV/EBITDA is {m.peers.premium['ev_ebitda']:.0%} above the peer median.")
    m.risks = risks

    vv = []
    if v is not None and v.implied_growth is not None:
        trend = v.cases["base"].assumptions.growth
        avg = float(np.mean(trend))
        gap = v.implied_growth - avg
        if abs(gap) < 0.02:
            vv.append(f"The price implies {_pct(v.implied_growth)} growth, close to the company's own trend "
                      f"({_pct(avg)} a year on average in the base case): no variant view on growth.")
        else:
            vv.append(f"The price implies {_pct(v.implied_growth)} growth a year; the company's own trend gives {_pct(avg)}. "
                      + ("The market expects more than the trend: the bet is that growth slows less than history says."
                         if gap > 0 else "The market expects less than the trend: a long position bets that the trend holds."))
    if v is not None and v.implied_margin is not None and np.isfinite(r["operating_margin"]):
        vv.append(f"At the base case's growth, the price implies an operating margin of {_pct(v.implied_margin)} against "
                  f"{_pct(r['operating_margin'])} today.")
    m.variant_view = vv

    crit: List[Criterion] = []
    if v is not None:
        bear = v.cases["bear"].assumptions
        crit.append(Criterion("operating_margin", "<", float(bear.operating_margin[-1]),
                              f"TTM operating margin falls below the bear case's {_pct(bear.operating_margin[-1])}."))
        floor = min(bear.growth[0], (v.implied_growth if v.implied_growth is not None else bear.growth[0]) - 0.05)
        crit.append(Criterion("revenue_growth", "<", float(floor),
                              f"TTM revenue growth falls below {_pct(floor)} (the bear case's first year, or 5 points under "
                              "what the price implies)."))
        crit.append(Criterion("price", "<", float(v.cases["bear"].per_share),
                              f"The price closes below the bear-case value ${v.cases['bear'].per_share:,.2f}: the market "
                              "prices worse than the bad case."))
    nd = r["net_debt_ebitda"]
    crit.append(Criterion("net_debt_ebitda", ">", float(max(3.0, nd + 1.0) if np.isfinite(nd) else 3.0),
                          f"Net debt rises above {max(3.0, nd + 1.0) if np.isfinite(nd) else 3.0:.1f}x EBITDA."))
    crit.append(Criterion("interest_cover", "<", 3.0, "Operating income covers interest less than 3x."))
    m.criteria = crit

    pm = []
    if v is not None:
        bear = v.cases["bear"]
        pm.append(f"If this is wrong in a year, the likely path is the bear case: growth {_pct(bear.assumptions.growth[0])} "
                  f"and a margin of {_pct(bear.assumptions.operating_margin[-1])}, worth ${bear.per_share:,.2f} "
                  f"({bear.per_share / f.price - 1:+.0%}).")
        pm.append(f"The discount rate ({_pct(v.rate.wacc)}) rests on a beta of {v.rate.beta:.2f} and an equity premium of "
                  f"{_pct(v.rate.equity_premium)}; one point higher cuts the value materially.")
    for x in reds[:3]:
        pm.append(f"A red flag that was visible beforehand: {x.text}")
    pm.append("What the filings cannot show: management, competition, regulation and the next product cycle. Free data "
              "has no consensus estimates; the reverse DCF stands in for them.")
    m.premortem = pm


def _sources(m: Memo, prices: pd.DataFrame) -> List[str]:
    f = m.financials
    out = [f"Statements: SEC EDGAR XBRL filings, as first reported; the latest used was filed "
           f"{f.last_filed.date() if f.last_filed is not None else 'n/a'}.",
           f"Prices: {prices.attrs.get('source', 'price feed')} to {prices.index[-1].date()}; the as-traded price "
           f"${f.price:,.2f} on {f.as_of.date()}."]
    if m.valuation is not None:
        out += [f"Discount rate: {n}." for n in m.valuation.rate.notes]
    out.append("Ken French Data Library (Fama-French factors, market premium): "
               "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html")
    return out


__all__ = ["Criterion", "Memo", "FundamentalView", "STANCE_RULE", "build", "find_peers", "peer_candidates",
           "fundamental_snapshot", "fundamental_view", "industry_view"]
