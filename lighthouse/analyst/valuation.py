"""
Valuation: what the business is worth under explicit assumptions, and what today's price assumes.

Discount rate  WACC = E/V x (risk-free + beta x equity premium) + D/V x cost of debt x (1 - tax). The risk-free rate is
               the latest 3-month T-bill (FRED DTB3, `data.load_risk_free`; 2% offline). Practitioners often use the
               10-year yield instead; the T-bill gives a lower discount rate when the curve is upward sloping.
               Beta: 60 monthly excess returns on the Ken French market factor, shrunk toward 1 (Blume 1975:
               0.67 x beta + 0.33). Equity premium: the Ken French market factor's excess return since July 1963,
               compounded (the geometric mean, 6.2% a year in data to August 2026): a DCF discounts over many years,
               and the arithmetic mean (7.2%) is the expected return of one year, which overstates the rate a
               multi-year discount should compound at. The arithmetic premium is shown beside the base case as a sensitivity
               (the user's decision, 15 October 2026; docs/NEXT_SESSION.md).
DCF            free cash flow to the firm after stock-based pay (a real cost, paid in shares: it is not added back),
               for the projection years, plus a terminal value by the value-driver formula (Koller, Goedhart & Wessels,
               "Valuation"): NOPAT in the year after the projection x (1 - g / RONIC) / (WACC - g). Growth g needs net
               investment g / RONIC of NOPAT, whatever capex was in the projection years; RONIC, the return on new
               capital, is today's ROIC, never below the WACC (new capital at least earns its cost), in the base and
               bull cases; in the bear case it is the WACC (growth after the projection adds no value: the advantage
               behind today's ROIC is competed away; Koller et al.'s conservative default). A Gordon value on
               the last year's free cash flow would carry that year's capex forever: a company in a build-out (MSFT,
               2026: capex 35% of revenue against D&A of 10%) was worth a fraction of its operating profit at any growth.
               Enterprise value - net debt = equity; per share at the cover-page share count.
Scenarios      base = the company's own trend (lighthouse.analyst.model); bull / bear shift growth and margin up and
               down (defaults: +/- 4 points of growth, +/- 3 points of margin); weights 25 / 50 / 25 for the target.
Reverse DCF    the constant revenue growth (margins held at today's) at which the DCF equals today's price: the
               expectations in the price, the free substitute for consensus estimates.
Bands          where today's earnings yield, FCF yield and EV/EBITDA sit in the stock's own last 10 years.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .financials import Financials
from .model import Assumptions, project

ERP_FALLBACK = 0.055          # when the Ken French library cannot be read (offline): a conventional equity premium


@dataclass
class DiscountRate:
    risk_free: float
    beta: float
    equity_premium: float
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    equity_weight: float
    wacc: float
    notes: List[str] = field(default_factory=list)
    arithmetic_premium: Optional[float] = None    # the arithmetic mean of the same data, for the sensitivity

    def with_premium(self, premium: float) -> "DiscountRate":
        """The same rate with another equity premium (risk-free, beta, debt and weights unchanged)."""
        ke = self.risk_free + self.beta * premium
        wacc = self.equity_weight * ke + (1.0 - self.equity_weight) * self.cost_of_debt * (1.0 - self.tax_rate)
        return replace(self, equity_premium=float(premium), cost_of_equity=float(ke), wacc=float(wacc), notes=[],
                       arithmetic_premium=None)


def estimate_beta(prices: pd.Series, months: int = 60, factors: Optional[pd.DataFrame] = None) -> Tuple[float, str]:
    """Blume-adjusted beta on monthly returns vs the Ken French market factor (or 1.0 with a note when unavailable)."""
    try:
        if factors is None:
            from .. import factors as ff
            factors = ff.load_factors("monthly")
        m = prices.groupby(prices.index.to_period("M")).last().pct_change().dropna()
        f = factors.copy()
        f.index = f.index.to_period("M")
        d = pd.concat([m.rename("r"), f["Mkt-RF"], f["RF"]], axis=1).dropna().tail(months)
        if len(d) < 24:
            return 1.0, "beta 1.0 (fewer than 24 months of returns)"
        x, y = d["Mkt-RF"].to_numpy(), (d["r"] - d["RF"]).to_numpy()
        raw = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        return 0.67 * raw + 0.33, f"beta {raw:.2f} raw over {len(d)} months, {0.67 * raw + 0.33:.2f} Blume-adjusted"
    except Exception as exc:  # noqa: BLE001 - no factor data: a neutral beta, said out loud
        return 1.0, f"beta 1.0 (market factor unavailable: {type(exc).__name__})"


def discount_rate(f: Financials, prices: pd.Series, risk_free: Optional[float] = None, equity_premium: Optional[float] = None,
                  beta: Optional[float] = None, factors: Optional[pd.DataFrame] = None) -> DiscountRate:
    notes: List[str] = []
    arithmetic = None
    if risk_free is None:
        from ..data import load_risk_free
        rf = load_risk_free()
        risk_free = float(rf.iloc[-1]) if rf is not None and len(rf) else 0.02
        notes.append(f"risk-free {risk_free * 100:.2f}% ({'FRED DTB3, ' + str(rf.index[-1].date()) if rf is not None and len(rf) else 'fixed 2%, offline'})")
    if equity_premium is None:
        try:
            from .. import factors as ff
            mp = ff.market_premium()
            if not np.isfinite(mp["geometric"]):
                raise ValueError("no market factor months")
            equity_premium, arithmetic = float(mp["geometric"]), float(mp["arithmetic"])
            notes.append(f"equity premium {equity_premium * 100:.1f}% (Ken French market factor since 1963, "
                         f"geometric; the arithmetic mean is {arithmetic * 100:.1f}%)")
        except Exception:  # noqa: BLE001
            equity_premium = ERP_FALLBACK
            notes.append(f"equity premium {ERP_FALLBACK * 100:.1f}% (conventional; the Ken French library was unavailable)")
    if beta is None:
        beta, why = estimate_beta(prices, factors=factors)
        notes.append(why)
    ke = risk_free + beta * equity_premium
    debt = max(f.bs.get("total_debt", 0.0) or 0.0, 0.0)
    ie = f.get("interest_expense", np.nan)
    kd = float(np.clip(ie / debt, risk_free, risk_free + 0.08)) if debt > 0 and np.isfinite(ie) else risk_free + 0.015
    e = max(f.market_cap, 0.0)
    w = e / (e + debt) if (e + debt) > 0 else 1.0
    wacc = w * ke + (1 - w) * kd * (1 - f.tax_rate)
    return DiscountRate(risk_free, float(beta), float(equity_premium), float(ke), float(kd), float(f.tax_rate), float(w),
                        float(wacc), notes, arithmetic)


@dataclass
class DCFResult:
    label: str
    per_share: float
    equity_value: float
    enterprise_value: float
    terminal_share: float            # share of enterprise value from the terminal value
    projection: pd.DataFrame
    assumptions: Assumptions
    ronic: float = float("nan")      # return on new capital behind the terminal value's reinvestment


def terminal_ronic(f: Financials, rate: float) -> float:
    """The return on new invested capital after the projection: today's ROIC, floored at the discount rate."""
    roic = f.ratios().get("roic", np.nan)
    return float(max(roic, rate)) if roic is not None and np.isfinite(roic) else float(rate)


def terminal_value(a: Assumptions, p: pd.DataFrame, rate: float, terminal_growth: float, ronic: float) -> float:
    """Value at the end of the projection: NOPAT_{n+1} x (1 - g / RONIC) / (rate - g). A loss in the last year has
    nothing to reinvest: it is capitalised as it is."""
    nopat = float(p["operating_income"].iloc[-1]) * (1.0 + terminal_growth) * (1.0 - a.tax_rate)
    reinvest = terminal_growth / ronic if nopat > 0 and ronic > 0 else 0.0
    return nopat * (1.0 - reinvest) / (rate - terminal_growth)


def dcf(a: Assumptions, f: Financials, rate: float, terminal_growth: float = 0.025,
        ronic: Optional[float] = None) -> DCFResult:
    """Enterprise value = discounted FCFF after stock-based pay + the discounted value-driver terminal value (module
    docstring); `ronic` defaults to `terminal_ronic`."""
    if rate <= terminal_growth:
        raise ValueError("the discount rate must exceed the terminal growth rate")
    p = project(a)
    fc = p["fcff_after_sbc"].to_numpy()
    disc = (1.0 + rate) ** -np.arange(1, len(fc) + 1)
    rn = terminal_ronic(f, rate) if ronic is None else float(ronic)
    tv = terminal_value(a, p, rate, terminal_growth, rn)
    pv = float((fc * disc).sum())
    pv_tv = float(tv * disc[-1])
    ev = pv + pv_tv
    eq = ev - f.net_debt
    ps = eq / f.shares if f.shares and np.isfinite(f.shares) else float("nan")
    return DCFResult(a.label, float(ps), float(eq), float(ev), float(pv_tv / ev) if ev else float("nan"), p, a, rn)


@dataclass
class Valuation:
    price: float
    rate: DiscountRate
    cases: Dict[str, DCFResult]
    weights: Dict[str, float]
    implied_growth: Optional[float]          # reverse DCF: constant growth the price implies (margins held)
    implied_margin: Optional[float]          # the operating margin the price implies at base growth
    terminal_growth: float
    bands: pd.DataFrame = field(default_factory=pd.DataFrame)
    # the base case at the arithmetic equity premium: {"equity_premium", "wacc", "per_share"} (empty when the premium
    # was given, or the historical one could not be read)
    premium_sensitivity: Dict[str, float] = field(default_factory=dict)

    @property
    def target(self) -> float:
        return float(sum(self.weights[k] * self.cases[k].per_share for k in self.cases))

    @property
    def upside_downside(self) -> float:
        """(bull - price) / (price - bear): above 1, more to gain in the good case than to lose in the bad one."""
        up = self.cases["bull"].per_share - self.price
        down = self.price - self.cases["bear"].per_share
        if down <= 0:
            return float("inf") if up > 0 else float("nan")
        return float(up / down)

    def table(self) -> pd.DataFrame:
        rows = []
        for k, c in self.cases.items():
            a = c.assumptions
            rows.append({"case": k, "weight": self.weights[k], "value_per_share": c.per_share,
                         "vs_price": c.per_share / self.price - 1.0, "growth_y1": a.growth[0], "growth_last": a.growth[-1],
                         "margin_last": a.operating_margin[-1], "terminal_share": c.terminal_share, "ronic": c.ronic})
        return pd.DataFrame(rows).set_index("case")


def _solve(fn, lo: float, hi: float, target: float, iters: int = 80) -> Optional[float]:
    """Bisection for fn(x) = target on [lo, hi] (fn increasing); None when the target is outside the range."""
    flo, fhi = fn(lo), fn(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or not (flo <= target <= fhi):
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reverse_dcf(f: Financials, base: Assumptions, rate: float, terminal_growth: float = 0.025) -> Tuple[Optional[float], Optional[float]]:
    """(the constant revenue growth the price implies at today's margin, the margin the price implies at base growth)."""
    m0 = base.operating_margin[0]

    def at_growth(g: float) -> float:
        return dcf(Assumptions(**{**base.to_dict(), "growth": [g] * base.years, "operating_margin": [m0] * base.years}),
                   f, rate, terminal_growth).per_share

    def at_margin(m: float) -> float:
        return dcf(Assumptions(**{**base.to_dict(), "operating_margin": [m] * base.years}), f, rate, terminal_growth).per_share
    g = _solve(at_growth, -0.30, 0.60, f.price)
    m = _solve(at_margin, -0.20, 0.80, f.price)
    return g, m


def valuation_bands(daily: Dict[str, pd.Series], years: int = 10) -> pd.DataFrame:
    """Today's value of each daily valuation series and its percentile in its own last `years` years (higher earnings
    or FCF yield = cheaper)."""
    rows = []
    for k, s in daily.items():
        s = s.dropna()
        if s.empty:
            continue
        w = s[s.index >= s.index[-1] - pd.Timedelta(days=int(365.25 * years))]
        now = float(w.iloc[-1])
        rows.append({"measure": k, "now": now, "p10": float(w.quantile(0.1)), "median": float(w.median()),
                     "p90": float(w.quantile(0.9)), "percentile": float((w <= now).mean()), "since": str(w.index[0].date())})
    return pd.DataFrame(rows).set_index("measure") if rows else pd.DataFrame()


def value(f: Financials, prices: pd.Series, rate: Optional[DiscountRate] = None, years: int = 5,
          terminal_growth: float = 0.025, growth_shift: float = 0.04, margin_shift: float = 0.03,
          weights: Optional[Dict[str, float]] = None, factors: Optional[pd.DataFrame] = None,
          bands: Optional[Dict[str, pd.Series]] = None) -> Valuation:
    """Bull / base / bear DCFs, the probability-weighted target, the reverse DCF and the bands. The bear case's
    RONIC is the WACC; base and bull use `terminal_ronic`."""
    r = rate or discount_rate(f, prices, factors=factors)
    base = Assumptions.from_financials(f, years, long_run_growth=terminal_growth + 0.005, label="base")
    cases = {"bear": dcf(base.scaled(-growth_shift, -margin_shift, "bear"), f, r.wacc, terminal_growth, ronic=r.wacc),
             "base": dcf(base, f, r.wacc, terminal_growth),
             "bull": dcf(base.scaled(growth_shift, margin_shift, "bull"), f, r.wacc, terminal_growth)}
    g, m = reverse_dcf(f, base, r.wacc, terminal_growth)
    sens: Dict[str, float] = {}
    ap = r.arithmetic_premium
    if ap is not None and np.isfinite(ap) and abs(ap - r.equity_premium) > 1e-9:
        alt = r.with_premium(ap)
        if alt.wacc > terminal_growth:
            sens = {"equity_premium": float(ap), "wacc": alt.wacc,
                    "per_share": dcf(base, f, alt.wacc, terminal_growth).per_share}
    return Valuation(price=f.price, rate=r, cases=cases, weights=weights or {"bear": 0.25, "base": 0.5, "bull": 0.25},
                     implied_growth=g, implied_margin=m, terminal_growth=terminal_growth,
                     bands=valuation_bands(bands or {}), premium_sensitivity=sens)


__all__ = ["DiscountRate", "estimate_beta", "discount_rate", "DCFResult", "dcf", "terminal_value", "terminal_ronic",
           "Valuation", "reverse_dcf", "valuation_bands", "value"]
