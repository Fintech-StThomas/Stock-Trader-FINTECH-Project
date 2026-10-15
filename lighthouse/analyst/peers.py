"""
Real peers instead of fixed sector norms: companies in the same SIC industry (EDGAR), closest in market value.

Peer choice   the 4-digit SIC code first; when it gives fewer than `min_peers`, the 3-digit, then the 2-digit group.
              Within the group, the `n` companies closest to the target in log market value. Candidates are the
              companies the caller can price (by default today's S&P 500 and the curated large caps): free data has
              no full list of every listed company with its price.
Comparison    valuation (P/E, EV/EBITDA, EV/sales, P/B, FCF yield), quality (gross and operating margin, ROIC) and
              growth, each against the peer median.
Explanation   the premium or discount on EV/EBITDA and P/E, next to the gaps in growth, margin and ROIC; with six or
              more peers, the EV/EBITDA the peers' own relation to growth and ROIC would predict (least squares), so a
              premium can be called explained or not.
Norms         the peer medians feed the fundamental beacons in place of `fundamentals.SECTOR_NORMS` (`peer_norms`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .financials import Financials

COMPARE = ["pe", "ev_ebitda", "ev_sales", "pb", "fcf_yield", "gross_margin", "operating_margin", "roic",
           "revenue_growth", "net_debt_ebitda"]
LABELS = {"pe": "P/E", "ev_ebitda": "EV/EBITDA", "ev_sales": "EV/sales", "pb": "P/B", "fcf_yield": "FCF yield",
          "gross_margin": "gross margin", "operating_margin": "operating margin", "roic": "ROIC",
          "revenue_growth": "revenue growth", "net_debt_ebitda": "net debt/EBITDA"}


def choose_peers(ticker: str, sics: Dict[str, Optional[str]], market_caps: Dict[str, float], n: int = 8,
                 min_peers: int = 4) -> tuple:
    """(peer tickers, the SIC level used, e.g. '4-digit SIC 7372')."""
    sic = sics.get(ticker)
    mc0 = market_caps.get(ticker)
    if not sic or not mc0 or not np.isfinite(mc0) or mc0 <= 0:
        return [], "no SIC code or market value"
    sic = str(sic).zfill(4)
    for digits in (4, 3, 2):
        group = [t for t, s in sics.items() if t != ticker and s and str(s).zfill(4)[:digits] == sic[:digits]
                 and market_caps.get(t) and np.isfinite(market_caps[t]) and market_caps[t] > 0]
        if len(group) >= min_peers or digits == 2:
            ranked = sorted(group, key=lambda t: abs(np.log(market_caps[t] / mc0)))
            return ranked[:n], f"{digits}-digit SIC {sic[:digits]}"
    return [], "no peers"


@dataclass
class PeerComparison:
    ticker: str
    level: str
    peers: List[str]
    table: pd.DataFrame                  # rows: the company, each peer, 'peer median'; columns: COMPARE
    premium: Dict[str, float]            # company / peer median - 1, for the valuation multiples
    explanation: List[str] = field(default_factory=list)
    predicted_ev_ebitda: Optional[float] = None


def compare(target: Financials, peers: Dict[str, Financials], level: str = "") -> PeerComparison:
    rows = {target.ticker: target.ratios()}
    for t, f in peers.items():
        rows[t] = f.ratios()
    tab = pd.DataFrame(rows).T.reindex(columns=COMPARE)
    peer_part = tab.drop(index=target.ticker)
    med = peer_part.median(numeric_only=True)
    tab.loc["peer median"] = med
    me = tab.loc[target.ticker]
    prem = {k: float(me[k] / med[k] - 1.0) for k in ("pe", "ev_ebitda", "ev_sales", "pb")
            if np.isfinite(me[k]) and np.isfinite(med[k]) and med[k] > 0}
    text = []
    for k in ("ev_ebitda", "pe"):
        if k in prem:
            text.append(f"{target.ticker} trades at {me[k]:.1f}x {LABELS[k]} against a peer median of {med[k]:.1f}x: a "
                        f"{abs(prem[k]):.0%} {'premium' if prem[k] > 0 else 'discount'}.")
    gaps = []
    for k, unit in (("revenue_growth", "points of revenue growth"), ("operating_margin", "points of operating margin"),
                    ("roic", "points of ROIC")):
        if np.isfinite(me[k]) and np.isfinite(med[k]):
            gaps.append(f"{(me[k] - med[k]) * 100:+.1f} {unit}")
    if gaps:
        text.append("Against the median peer it has " + ", ".join(gaps) + ".")
    pred = None
    ok = peer_part[["ev_ebitda", "revenue_growth", "roic"]].replace([np.inf, -np.inf], np.nan).dropna()
    ok = ok[(ok["ev_ebitda"] > 0) & (ok["ev_ebitda"] < 100)]
    if len(ok) >= 6 and np.isfinite(me[["revenue_growth", "roic"]].astype(float)).all():
        X = np.column_stack([np.ones(len(ok)), ok["revenue_growth"], ok["roic"]])
        beta, *_ = np.linalg.lstsq(X, ok["ev_ebitda"].to_numpy(), rcond=None)
        pred = float(beta @ np.array([1.0, me["revenue_growth"], me["roic"]]))
        if np.isfinite(me["ev_ebitda"]) and pred > 0:
            gap = me["ev_ebitda"] / pred - 1.0
            text.append(f"The peers' own relation between EV/EBITDA, growth and ROIC predicts {pred:.1f}x for "
                        f"{target.ticker}; it trades {abs(gap):.0%} {'above' if gap > 0 else 'below'} that, so "
                        + ("the premium is not explained by growth and returns alone." if gap > 0.15 else
                           "the discount is not explained by growth and returns alone." if gap < -0.15 else
                           "its multiple is in line with its growth and returns."))
    elif len(peers) < 6:
        text.append(f"Only {len(peers)} peers: too few to fit the multiple to growth and returns.")
    return PeerComparison(target.ticker, level, list(peers), tab, prem, text, pred)


def peer_norms(cmp: PeerComparison) -> Dict[str, object]:
    """Peer medians in the shape of `fundamentals.SECTOR_NORMS` (P/E, P/B, EV/EBITDA, operating margin), plus the
    `label` and `source` that `fundamentals.fundamental_beacons(norms=...)` writes into each beacon that used them.
    A median that is missing or not positive (a multiple) is None: the beacon then keeps the sector norm."""
    med = cmp.table.loc["peer median"]
    clean = lambda v: float(v) if v is not None and np.isfinite(v) and v > 0 else None   # noqa: E731
    return {"pe": clean(med.get("pe")), "pb": clean(med.get("pb")), "ev_ebitda": clean(med.get("ev_ebitda")),
            "op_margin": float(med["operating_margin"]) if np.isfinite(med.get("operating_margin", np.nan)) else None,
            "label": "peer median",
            "source": f"median of {len(cmp.peers)} peers ({cmp.level}): {', '.join(cmp.peers)}"}


__all__ = ["choose_peers", "PeerComparison", "compare", "peer_norms", "COMPARE", "LABELS"]
