"""
The sector- and size-neutral blend of the signals that survive the pre-registered rule (Phase 4).

A signal that survives on its own may still be an industry bet or a size bet in disguise (value loads on banks and
energy, low volatility on utilities, net issuance on small firms). Before survivors are combined, each one is made
neutral to both, month by month, across the names tested that month:

  1. winsorise at the 1st and 99th percentiles of the month's cross-section (one bad filing must not set the scale)
  2. z-score
  3. regress on Fama-French 49 industry dummies (from the SEC SIC code; an industry with fewer than `min_group` names
     that month is pooled with the others into one "small industries" group) and log market value; keep the residual
     (a name without a market value gets the month's median, so it is industry-neutral only)
  4. z-score the residual again

The blend is the equal-weight mean of the survivors' neutral scores, over the survivors a name has (at least half of
them). Equal weights on purpose: the survivors were chosen with the hold-out, so fitting weights as well would tune
more on data that is no longer out of sample. The blend is evaluated as one more signal and counts as one more trial.

Everything here is cross-sectional within a month: a month's value uses only that month's inputs, so the blend is as
point-in-time as the signals, industries (today's SIC code, see METHODOLOGY section 8b) and market values it is given.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

MIN_GROUP = 3            # names an industry needs in a month to get its own dummy
MIN_NAMES = 30           # below this a month's cross-section is too thin to neutralise (the harness's MIN_NAMES)


def neutralize_row(x: pd.Series, industry: pd.Series, log_mcap: Optional[pd.Series] = None,
                   min_group: int = MIN_GROUP) -> pd.Series:
    """One month: steps 1-4 of the module docstring for the names where `x` is known. Returns NaN elsewhere."""
    x = x.astype(float)
    ok = x.notna() & np.isfinite(x)
    out = pd.Series(np.nan, index=x.index)
    if ok.sum() < 3:
        return out
    v = x[ok]
    lo, hi = v.quantile(0.01), v.quantile(0.99)
    v = v.clip(lo, hi)
    sd = v.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return out
    z = (v - v.mean()) / sd
    ind = industry.reindex(z.index).fillna("Unknown").astype(str)
    counts = ind.value_counts()
    ind = ind.where(ind.map(counts) >= min_group, "_small")
    X = pd.get_dummies(ind, dtype=float)
    if log_mcap is not None:
        m = log_mcap.reindex(z.index).astype(float)
        m = m.where(np.isfinite(m))
        if m.notna().sum() >= 3:
            m = m.fillna(m.median())
            msd = m.std(ddof=1)
            if np.isfinite(msd) and msd > 0:
                X["log_mcap"] = (m - m.mean()) / msd
    beta, *_ = np.linalg.lstsq(X.to_numpy(), z.to_numpy(), rcond=None)
    res = z.to_numpy() - X.to_numpy() @ beta
    rsd = res.std(ddof=1)
    if not np.isfinite(rsd) or rsd <= 0:
        return out
    out[z.index] = (res - res.mean()) / rsd
    return out


def neutral_scores(signal: pd.DataFrame, industries: Dict[str, Optional[str]], market_cap: Optional[pd.DataFrame] = None,
                   mask: Optional[pd.DataFrame] = None, min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Month-end x ticker neutral z-scores of one signal. `mask` (month-end x ticker, True = in the month's tested
    universe, e.g. members that traded) limits each month's cross-section; outside it the score is NaN."""
    sig = signal.astype(float)
    ind = pd.Series({t: industries.get(t) for t in sig.columns}, dtype=object)
    lm = None
    if market_cap is not None:
        mc = market_cap.reindex(index=sig.index, columns=sig.columns).astype(float)
        lm = np.log(mc.where(mc > 0))
    if mask is not None:
        sig = sig.where(mask.reindex(index=sig.index, columns=sig.columns, fill_value=False).astype(bool))
    out = pd.DataFrame(np.nan, index=sig.index, columns=sig.columns)
    for d in sig.index:
        row = sig.loc[d]
        if row.notna().sum() < min_names:
            continue
        out.loc[d] = neutralize_row(row, ind, None if lm is None else lm.loc[d])
    return out


def blend(signals: Dict[str, pd.DataFrame], industries: Dict[str, Optional[str]],
          market_cap: Optional[pd.DataFrame] = None, mask: Optional[pd.DataFrame] = None,
          min_share: float = 0.5) -> pd.DataFrame:
    """Equal-weight mean of the signals' neutral scores (see the module docstring), where a name has at least
    `min_share` of them."""
    if not signals:
        raise ValueError("no signals to blend")
    scores = [neutral_scores(s, industries, market_cap, mask) for s in signals.values()]
    idx = scores[0].index
    cols = sorted(set().union(*[set(s.columns) for s in scores]))
    stack = np.stack([s.reindex(index=idx, columns=cols).to_numpy() for s in scores])
    have = np.isfinite(stack).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isfinite(stack), stack, np.nan), axis=0)
    mean[have < max(1, int(np.ceil(min_share * len(scores))))] = np.nan
    return pd.DataFrame(mean, index=idx, columns=cols)


def industries_from_sic(sics: Dict[str, Optional[str]], tickers: Iterable[str]) -> Dict[str, Optional[str]]:
    """Ticker -> Fama-French 49 industry name (None without a SIC code)."""
    from .. import factors as ff
    return {t: (ff.sic_industry(sics.get(t)) if sics.get(t) else None) for t in tickers}


__all__ = ["neutralize_row", "neutral_scores", "blend", "industries_from_sic", "MIN_GROUP"]
