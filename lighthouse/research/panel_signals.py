"""
Signals that need the whole panel or outside series, computed at month ends.

  industry_mom   the 12-1 month return of the stock's Fama-French 49 industry (value-weighted, Ken French), the
                 industry from the company's SEC SIC code (Moskowitz & Grinblatt 1999)
  residual_mom   the stock's 12-1 month return after removing its exposure to the Fama-French three factors, estimated
                 over the previous 36 months, divided by the volatility of those residuals (Blitz, Huij & Martens 2011)

Factor and industry returns are realised history: a month's value is used only from the month end after it. (French
publishes them a month or two later; the returns themselves were set by prices public at the time.)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import factors as ff
from .. import store

PANEL_SIGNALS = ["industry_mom", "residual_mom"]
DESCRIPTIONS = {"industry_mom": "12-1 month return of the stock's FF49 industry",
                "residual_mom": "12-1 month Fama-French-3 residual return / residual volatility"}


def sic_codes(tickers, source: str = "auto") -> Dict[str, Optional[str]]:
    """Ticker -> SIC code from EDGAR submissions (cached; None when the ticker has no CIK today)."""
    from ..edgar import cik_for, submissions
    from ..parallel import map_tickers

    def one(t):
        cik = cik_for(t)
        return submissions(cik)[0].get("sic") if cik is not None else None
    got, _ = map_tickers(one, list(tickers), max_workers=8)
    return {t: got.get(t) for t in tickers}


def industry_momentum(panel, sics: Dict[str, Optional[str]]) -> pd.DataFrame:
    ind = ff.load_industries(49, "value")
    ind.index = ind.index.to_period("M")
    mom = (1 + ind).rolling(11).apply(np.prod, raw=True).shift(1) - 1.0      # months t-11 .. t-1 (skip month t)
    out = pd.DataFrame(np.nan, index=panel.dates, columns=panel.prices.columns)
    per = panel.dates.to_period("M")
    for t in out.columns:
        name = ff.sic_industry(sics.get(t)) if sics.get(t) else None
        if name is None or name not in mom.columns:
            continue
        out[t] = mom[name].reindex(per).to_numpy()
    return out


def residual_momentum(panel) -> pd.DataFrame:
    f = ff.load_factors("monthly")
    f.index = f.index.to_period("M")
    rets = panel.prices.pct_change()                        # the return of the month ending at each month end
    per = panel.dates.to_period("M")
    F = f.reindex(per)[["Mkt-RF", "SMB", "HML"]].to_numpy()
    rf = f.reindex(per)["RF"].to_numpy()
    out = pd.DataFrame(np.nan, index=panel.dates, columns=panel.prices.columns)
    R = rets.to_numpy() - rf[:, None]
    n = len(panel.dates)
    for k in range(37, n):
        win = slice(k - 36, k)                               # months k-36 .. k-1 (month k, the latest, is skipped)
        X = F[win]
        if not np.isfinite(X).all():
            continue
        Z = np.column_stack([np.ones(len(X)), X])
        Y = R[win]
        ok = np.isfinite(Y).all(axis=0)
        if not ok.any():
            continue
        beta, *_ = np.linalg.lstsq(Z, Y[:, ok], rcond=None)
        res = Y[:, ok] - Z @ beta
        last11 = res[-11:]
        sd = res.std(axis=0, ddof=1)
        out.iloc[k, np.where(ok)[0]] = np.where(sd > 0, last11.sum(axis=0) / sd, np.nan)
    return out


def monthly(frames, panel, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    out = {}
    for n in PANEL_SIGNALS:
        hit = None if refresh else store.read_frame("research_signals", n)
        if hit is not None and hit.index.equals(panel.dates):
            out[n] = hit
    if "industry_mom" not in out:
        sics = sic_codes(list(panel.prices.columns))
        store.write_json("research", "sic_codes", sics)
        out["industry_mom"] = industry_momentum(panel, sics)
        store.write_frame("research_signals", "industry_mom", out["industry_mom"])
    if "residual_mom" not in out:
        out["residual_mom"] = residual_momentum(panel)
        store.write_frame("research_signals", "residual_mom", out["residual_mom"])
    return out


__all__ = ["PANEL_SIGNALS", "DESCRIPTIONS", "industry_momentum", "residual_momentum", "sic_codes", "monthly"]
