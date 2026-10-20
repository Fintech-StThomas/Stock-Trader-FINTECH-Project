"""
Fundamental analysis: valuation, quality and growth beacons, for the analyst memo (lighthouse.analyst.memo).

Each beacon compares a ratio with a sector norm (editable in SECTOR_NORMS) or a peer median, or with an absolute band,
and votes between -1 (expensive / weak) and +1 (cheap / strong). The weighted result is the Fundamental Score (0-100).
It is a view for the investor, not an input to the trading rule: a fundamental gate cost return in every backtest and
the score had no ranking ability across the S&P 500 (docs/METHODOLOGY.md sections 8 and 8b).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FUNDAMENTAL_WEIGHTS, StrategyParams
from .technicals import Beacon

# Typical sector medians (approximate, long-run; edit to taste - they are assumptions, not data).
SECTOR_NORMS: Dict[str, Dict[str, Optional[float]]] = {
    #                         P/E    P/B   EV/EBITDA  op. margin
    "Technology":             {"pe": 28.0, "pb": 7.0, "ev_ebitda": 20.0, "op_margin": 0.22},
    "Communication Services": {"pe": 20.0, "pb": 3.0, "ev_ebitda": 10.0, "op_margin": 0.15},
    "Healthcare":             {"pe": 22.0, "pb": 4.0, "ev_ebitda": 15.0, "op_margin": 0.15},
    "Financial Services":     {"pe": 13.0, "pb": 1.4, "ev_ebitda": None, "op_margin": 0.30},
    "Consumer Cyclical":      {"pe": 20.0, "pb": 4.0, "ev_ebitda": 13.0, "op_margin": 0.10},
    "Consumer Defensive":     {"pe": 20.0, "pb": 4.0, "ev_ebitda": 13.0, "op_margin": 0.10},
    "Industrials":            {"pe": 20.0, "pb": 3.5, "ev_ebitda": 13.0, "op_margin": 0.12},
    "Energy":                 {"pe": 12.0, "pb": 1.8, "ev_ebitda": 6.0,  "op_margin": 0.15},
    "Utilities":              {"pe": 17.0, "pb": 1.8, "ev_ebitda": 11.0, "op_margin": 0.18},
    "Real Estate":            {"pe": 30.0, "pb": 2.0, "ev_ebitda": 18.0, "op_margin": 0.25},
    "Basic Materials":        {"pe": 15.0, "pb": 2.0, "ev_ebitda": 8.0,  "op_margin": 0.12},
    "default":                {"pe": 20.0, "pb": 3.0, "ev_ebitda": 13.0, "op_margin": 0.12},
}


def sector_norms(sector: Optional[str]) -> Dict[str, Optional[float]]:
    return SECTOR_NORMS.get(sector or "", SECTOR_NORMS["default"])


@dataclass
class FundamentalSnapshot:
    ticker: str
    name: str = ""
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: str = "USD"
    price: Optional[float] = None
    market_cap: Optional[float] = None
    # valuation
    pe_trailing: Optional[float] = None     # -1 means "negative earnings"
    pe_forward: Optional[float] = None
    peg: Optional[float] = None
    pb: Optional[float] = None
    ev_ebitda: Optional[float] = None
    # quality / financial health
    roe: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    fcf: Optional[float] = None
    fcf_yield: Optional[float] = None
    # growth
    revenue_growth: Optional[float] = None
    eps_growth: Optional[float] = None
    # context
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    analyst_target: Optional[float] = None
    analyst_count: Optional[float] = None
    as_of: str = ""
    source: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return any(getattr(self, k) is not None for k in
                   ["pe_trailing", "pe_forward", "peg", "pb", "ev_ebitda", "roe", "operating_margin",
                    "debt_to_equity", "fcf_yield", "revenue_growth", "eps_growth"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FundamentalSnapshot":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class StatementHistory:
    """Annual and quarterly statements from SEC EDGAR, one row per period end, with a `filed` column: the date each row
    became public (lighthouse.pit)."""
    annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def has_filing_dates(self) -> bool:
        return self.annual is not None and not self.annual.empty and "filed" in self.annual and \
            self.annual["filed"].notna().any()

    @property
    def years(self) -> int:
        return int(len(self.annual))


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def score_multiple_vs_norm(value: Optional[float], norm: Optional[float], spread: float = 0.5) -> Optional[float]:
    """Cheap relative to the norm -> +1 (at (1-spread)x norm); expensive -> -1 (at (1+spread)x norm)."""
    if value is None or norm is None or norm <= 0:
        return None
    if value <= 0:
        return -0.75   # negative earnings / book / EBITDA: valuation cannot support the price
    return _clip((norm - value) / (norm * spread))


def score_band(value: Optional[float], neutral: float, full: float) -> Optional[float]:
    """0 at `neutral`, +1 at neutral + full, -1 at neutral - full (linear, clipped)."""
    if value is None:
        return None
    return _clip((value - neutral) / full)


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _num(x: Optional[float], nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# Beacons from a snapshot
# --------------------------------------------------------------------------- #
NORM_KEYS = ("pe", "pb", "ev_ebitda", "op_margin")


def _norms_in_use(sector: Optional[str], norms: Optional[Dict[str, object]]) -> tuple:
    """(values, per-key label or None, source) for the beacons. Without `norms`: the sector norms, label None (the
    beacons keep their sector wording). With `norms` (e.g. `analyst.peers.peer_norms`): each usable value replaces the
    sector norm and is labelled (norms["label"], e.g. "peer median"); a missing or unusable one (None, not finite, a
    non-positive multiple) keeps the sector norm. EV/EBITDA stays off for financials whatever the norms say (debt is a
    bank's raw material, so enterprise value means nothing there)."""
    base = dict(sector_norms(sector))
    labels: Dict[str, Optional[str]] = {k: None for k in NORM_KEYS}
    if norms is None:
        return base, labels, None
    tag = str(norms.get("label") or "given norm")
    for k in NORM_KEYS:
        v = norms.get(k)
        if v is None or not np.isfinite(float(v)) or (k != "op_margin" and float(v) <= 0):
            continue
        base[k], labels[k] = float(v), tag
    if sector == "Financial Services":
        base["ev_ebitda"], labels["ev_ebitda"] = None, None
    return base, labels, str(norms.get("source") or tag)


def fundamental_beacons(snap: FundamentalSnapshot, history: Optional[StatementHistory] = None,
                        params: Optional[StrategyParams] = None,
                        norms: Optional[Dict[str, object]] = None) -> List[Beacon]:
    """Explainable valuation / quality / growth beacons for the current snapshot.

    `norms` (opt-in, e.g. the peer medians from `analyst.peers.peer_norms`) replaces the sector norms for P/E, P/B,
    EV/EBITDA and operating margin; each beacon that used one says so in its reading ("vs peer median") and names the
    norms' source in its explanation. Left at None (the decision, the backtest and every published number) the
    beacons are exactly the sector-norm ones."""
    norms, label, source = _norms_in_use(snap.sector, norms)
    sector_label = snap.sector or "the overall market"
    is_financial = (snap.sector == "Financial Services")
    beacons: List[Beacon] = []

    def vs(k: str) -> str:            # the reading: "vs Technology ~28" / "vs peer median ~31"
        return f"{label[k]} " if label[k] else ""

    def cite(k: str) -> str:          # appended to the explanation when a beacon used a given norm
        return f" Norm: {label[k]}, {source}." if label[k] else ""

    # ---------------- Valuation ----------------
    val: List[Beacon] = []
    pe = snap.pe_trailing
    if pe is not None:
        s = score_multiple_vs_norm(pe, norms["pe"])
        if pe <= 0:
            txt, why = "negative (loss-making)", "The company is not profitable on a trailing basis, so the share price is not supported by current earnings."
        elif label["pe"]:
            txt = f"P/E {pe:.1f} vs {label['pe']} ~{norms['pe']:.0f}"
            why = (f"Trailing P/E of {pe:.1f} is {'below' if s > 0 else 'above'} the {label['pe']} multiple "
                   f"(~{norms['pe']:.0f}): you pay {'less' if s > 0 else 'more'} per dollar of earnings than for a typical peer."
                   + cite("pe"))
        else:
            txt = f"P/E {pe:.1f} vs {sector_label} ~{norms['pe']:.0f}"
            why = (f"Trailing P/E of {pe:.1f} is {'below' if s > 0 else 'above'} the typical {sector_label} multiple "
                   f"(~{norms['pe']:.0f}): you pay {'less' if s > 0 else 'more'} per dollar of earnings than for a typical peer.")
        val.append(Beacon("pe", "Price / Earnings (trailing)", "fundamental", txt, s, 0, why))
    if snap.pe_forward is not None and snap.pe_forward > 0:
        s = score_multiple_vs_norm(snap.pe_forward, norms["pe"] * 0.9)
        val.append(Beacon("pe_fwd", "Price / Earnings (forward)", "fundamental", f"fwd P/E {snap.pe_forward:.1f}", s, 0,
                          f"Analysts' next-12-month earnings put the forward P/E at {snap.pe_forward:.1f}; "
                          f"{'cheaper' if s > 0 else 'richer'} than the {label['pe'] or 'sector norm'} after expected growth."
                          + cite("pe")))
    if snap.peg is not None and snap.peg > 0:
        s = _clip((1.5 - snap.peg) / 1.0)
        val.append(Beacon("peg", "PEG ratio", "fundamental", f"PEG {snap.peg:.2f}", s, 0,
                          "PEG near or below 1 means the growth rate justifies the earnings multiple." if snap.peg <= 1.2 else
                          "PEG well above 1.5 means investors pay a premium that growth alone does not justify." if snap.peg >= 1.8 else
                          "PEG between 1.2 and 1.8: the multiple is roughly in line with growth."))
    if snap.pb is not None:
        s = score_multiple_vs_norm(snap.pb, norms["pb"], spread=0.6)
        val.append(Beacon("pb", "Price / Book", "fundamental", f"P/B {snap.pb:.2f} vs {vs('pb')}~{norms['pb']:.1f}", s, 0,
                          f"Price-to-book of {snap.pb:.2f} is {'below' if s > 0 else 'above'} the "
                          f"{label['pb'] or sector_label + ' norm'}; "
                          + ("book value matters most for asset-heavy and financial companies." if is_financial else
                             "for asset-light companies this is a secondary check.") + cite("pb")))
    if snap.ev_ebitda is not None and norms["ev_ebitda"]:
        s = score_multiple_vs_norm(snap.ev_ebitda, norms["ev_ebitda"])
        val.append(Beacon("ev_ebitda", "EV / EBITDA", "fundamental",
                          f"EV/EBITDA {snap.ev_ebitda:.1f} vs {vs('ev_ebitda')}~{norms['ev_ebitda']:.0f}", s, 0,
                          "Enterprise value per dollar of operating cash earnings, which is capital-structure neutral, "
                          f"is {'below' if s > 0 else 'above'} the {label['ev_ebitda'] or 'sector norm'}."
                          + cite("ev_ebitda")))
    hist_beacon = _pe_vs_own_history(snap, history)
    if hist_beacon is not None:
        val.append(hist_beacon)

    # ---------------- Quality ----------------
    qual: List[Beacon] = []
    if snap.roe is not None:
        s = score_band(snap.roe, 0.10, 0.15)
        qual.append(Beacon("roe", "Return on equity", "fundamental", f"ROE {_pct(snap.roe)}", s, 0,
                           "Return on equity above 20% signals a business that compounds shareholder capital efficiently." if snap.roe >= 0.20 else
                           "Return on equity near 10% is average: the business earns roughly its cost of capital." if snap.roe >= 0.05 else
                           "Low or negative return on equity: the business is destroying or barely earning on shareholder capital."))
    if snap.operating_margin is not None:
        s = score_band(snap.operating_margin, norms["op_margin"], 0.15)
        qual.append(Beacon("margin", "Operating margin", "fundamental",
                           f"op. margin {_pct(snap.operating_margin)} vs {vs('op_margin')}~{_pct(norms['op_margin'])}", s, 0,
                           f"Operating margin is {'above' if s > 0 else 'below'} the "
                           f"{label['op_margin'] or sector_label + ' norm'}, which "
                           f"{'suggests pricing power or cost advantages' if s > 0 else 'leaves less cushion if revenue slows'}."
                           + cite("op_margin")))
    if snap.debt_to_equity is not None and not is_financial:
        s = _clip((1.0 - snap.debt_to_equity) / 1.0)
        qual.append(Beacon("leverage", "Debt / Equity", "fundamental", f"D/E {_num(snap.debt_to_equity, 2)}", s, 0,
                           "Debt below equity: a balance sheet that can absorb a downturn." if snap.debt_to_equity < 1 else
                           "Debt above equity: leverage amplifies both profits and losses, and raises refinancing risk."))
    if snap.fcf_yield is not None:
        s = score_band(snap.fcf_yield, 0.03, 0.03)
        qual.append(Beacon("fcf", "Free-cash-flow yield", "fundamental", f"FCF yield {_pct(snap.fcf_yield)}", s, 0,
                           "Free cash flow relative to market value: above 5% is a genuine cash return; below 1% means the "
                           "price relies on future growth rather than cash generated today."))
    if snap.current_ratio is not None and not is_financial:
        s = _clip((snap.current_ratio - 1.0) / 1.0, -1.0, 0.5)
        qual.append(Beacon("liquidity", "Current ratio", "fundamental", f"current ratio {_num(snap.current_ratio, 2)}", s, 0,
                           "Short-term assets cover short-term liabilities." if snap.current_ratio >= 1 else
                           "Short-term liabilities exceed short-term assets: watch liquidity."))

    # ---------------- Growth ----------------
    gro: List[Beacon] = []
    if snap.revenue_growth is not None:
        s = score_band(snap.revenue_growth, 0.05, 0.15)
        gro.append(Beacon("rev_growth", "Revenue growth (YoY)", "fundamental", f"revenue {_pct(snap.revenue_growth)}", s, 0,
                          "Double-digit revenue growth: demand for the product is expanding." if snap.revenue_growth >= 0.10 else
                          "Revenue growing around the pace of the economy." if snap.revenue_growth >= 0 else
                          "Revenue is shrinking, which pressures every other metric."))
    if snap.eps_growth is not None:
        s = score_band(snap.eps_growth, 0.08, 0.20)
        gro.append(Beacon("eps_growth", "EPS growth (YoY)", "fundamental", f"EPS {_pct(snap.eps_growth)}", s, 0,
                          "Earnings per share are growing faster than the market's long-run ~8%." if snap.eps_growth >= 0.08 else
                          "Earnings per share are growing below the market's long-run pace." if snap.eps_growth >= 0 else
                          "Earnings per share are falling."))

    # assign group weights: each group's weight is split equally among its available beacons
    for group_key, items in (("valuation", val), ("quality", qual), ("growth", gro)):
        if items:
            w = FUNDAMENTAL_WEIGHTS[group_key] / len(items)
            for b in items:
                b.key = f"{group_key}:{b.key}"
                b.weight = w
        beacons.extend(items)
    return beacons


def _pe_vs_own_history(snap: FundamentalSnapshot, history: Optional[StatementHistory]) -> Optional[Beacon]:
    """Earnings-stability beacon: how trustworthy is the P/E anchor?

    Uses the dispersion of the company's own annual EPS (needs >= 3 years). Steady
    earnings make a multiple meaningful; erratic earnings make it a weak anchor.
    """
    if history is None or history.annual.empty or "eps" not in history.annual or snap.pe_trailing is None or snap.pe_trailing <= 0:
        return None
    eps = history.annual["eps"].dropna()
    if len(eps) < 3 or not eps.mean():
        return None
    cv = float(eps.std(ddof=0) / abs(eps.mean()))
    if not np.isfinite(cv):
        return None
    s = _clip((0.35 - cv) / 0.35, -1.0, 0.5)   # stable earnings (+), erratic earnings (-)
    return Beacon("eps_stability", "Earnings stability (annual EPS)", "fundamental",
                  f"EPS volatility {cv * 100:.0f}% of mean over {len(eps)} years", s, 0,
                  "Annual earnings have been steady, which makes the P/E multiple more trustworthy." if cv < 0.25 else
                  "Annual earnings swing widely, so the headline P/E is a less reliable anchor.")


def fundamental_score(beacons: List[Beacon]) -> Optional[float]:
    """0-100 weighted score; None when nothing is available."""
    usable = [b for b in beacons if b.group == "fundamental" and not np.isnan(b.score)]
    if not usable:
        return None
    total_w = sum(b.weight for b in usable)
    if total_w <= 0:
        return None
    return float(50.0 + 50.0 * sum(b.weight * b.score for b in usable) / total_w)
