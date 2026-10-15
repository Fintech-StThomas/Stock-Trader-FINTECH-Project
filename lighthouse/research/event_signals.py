"""
Event signals for the research harness: the market's reaction to earnings, and the language of the annual report.

ear_3d             the earnings announcement return (Brandt, Kishore, Santa-Clara & Venkatachalam 2008): the stock's
                   return minus the equal-weighted universe's over the three sessions around its latest earnings release
                   (the 8-K filed under Item 2.02: the session before, the release session, the session after). Known
                   at the close of the session after; kept 95 days (until the next release). Higher = buy.
text_similarity    cosine similarity of the latest 10-K's words with the previous 10-K's (Cohen, Malloy & Nguyen 2020,
                   "Lazy Prices"). Higher (fewer changes) = buy.
risk_similarity    the same for Item 1A, Risk Factors. Higher = buy.
lm_uncertainty     share of Loughran-McDonald uncertainty words in the 10-K. Lower = buy (sign registered before testing).
lm_litigious       share of litigious words. Lower = buy.
lm_negative        share of negative words (Loughran & McDonald 2011). Lower = buy.
fog                Gunning fog index of the 10-K (Li 2008: harder-to-read reports come with lower, less persistent
                   earnings). Lower = buy.
A 10-K's text signals are known from the session after its filing date and kept 15 months.
eightk_tone        the change in Loughran-McDonald net tone ((positive - negative words) / words) of the company's
                   earnings press release (the EX-99 exhibit of its Item 2.02 8-K) from its previous release: a release
                   more upbeat than the company's own last one (Davis, Piger & Sedor 2012; the change, not the level,
                   so a company's house style cancels). Known from the day after the filing date, kept 95 days. Run on
                   a fixed half of the members (EIGHTK_SAMPLE: chosen by a hash of the ticker before any result), as
                   reading every release since 2009 (about 34,000, two requests each) takes most of a day at the SEC's
                   rate limit.

Coverage: companies the SEC's ticker table maps to a CIK today (delisted companies are missing unless their ticker
still maps), and their filing lists from EDGAR submissions. 10-K documents are read once and not stored: only the numbers
computed from them are cached (data/cache/research_text).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import store

EAR_MAX_AGE_DAYS = 95
TEXT_MAX_AGE_DAYS = 456
TEXT_FEATURES = {"text_similarity": ("similarity", 1.0), "risk_similarity": ("risk_similarity", 1.0),
                 "lm_uncertainty": ("lm_uncertainty", -1.0), "lm_litigious": ("lm_litigious", -1.0),
                 "lm_negative": ("lm_negative", -1.0), "fog": ("fog", -1.0)}
EIGHTK_MAX_AGE_DAYS = 95
EIGHTK_SAMPLE = 0.5
EIGHTK_START = "2009-01-01"
DESCRIPTIONS = {
    "eightk_tone": "change in Loughran-McDonald net tone of the earnings release (8-K Item 2.02, EX-99) from the last",
    "ear_3d": "earnings announcement return, 3 sessions around the 8-K Item 2.02, minus the equal-weight universe",
    "text_similarity": "10-K word similarity with the previous 10-K (Lazy Prices)",
    "risk_similarity": "Item 1A risk-factor similarity with the previous 10-K",
    "lm_uncertainty": "low share of Loughran-McDonald uncertainty words in the 10-K",
    "lm_litigious": "low share of litigious words in the 10-K",
    "lm_negative": "low share of negative words in the 10-K",
    "fog": "low Gunning fog index of the 10-K (easier to read)",
}


# --------------------------------------------------------------------------- #
# Earnings announcement return
# --------------------------------------------------------------------------- #
def announcement_returns(close: pd.Series, market: pd.Series, dates: pd.Series) -> pd.Series:
    """Per release: (the stock's three-session return - the market's), indexed by the session after the window ends
    (the first session the number could be traded on)."""
    idx = close.index
    r = close.pct_change()
    m = market.reindex(idx)
    out = {}
    for d in pd.to_datetime(pd.Series(dates)).dropna():
        k = idx.searchsorted(d)                       # the release session: the first session on or after the filing
        if k < 1 or k + 2 >= len(idx):
            continue
        win = slice(k - 1, k + 2)
        sr = float(np.prod(1.0 + r.iloc[win].to_numpy()) - 1.0)
        mr = float(np.prod(1.0 + m.iloc[win].to_numpy()) - 1.0)
        if np.isfinite(sr) and np.isfinite(mr):
            out[idx[k + 2]] = sr - mr
    return pd.Series(out, dtype=float).sort_index()


def hold_daily(events: pd.Series, index: pd.DatetimeIndex, max_age_days: int) -> pd.Series:
    """An event value carried forward on `index` from its date until a newer one or `max_age_days`."""
    if events.empty:
        return pd.Series(np.nan, index=index)
    events = events[~events.index.duplicated(keep="last")].sort_index()
    on = events.reindex(events.index.union(index)).ffill().reindex(index)
    known = pd.Series(events.index, index=events.index).reindex(events.index.union(index)).ffill().reindex(index)
    age = (pd.Series(index, index=index) - known).dt.days
    return on.where(age <= max_age_days)


def equal_weight_market(frames: Dict[str, pd.DataFrame]) -> pd.Series:
    rets = pd.DataFrame({t: f["Close"].pct_change() for t, f in frames.items()})
    return rets.clip(-0.9, 3.0).mean(axis=1)


def earnings_monthly(frames: Dict[str, pd.DataFrame], panel, refresh: bool = False, source: str = "auto") -> Dict[str, pd.DataFrame]:
    from ..edgar import company_filings, earnings_release_dates
    from ..parallel import map_tickers
    from .harness import sample_at_month_ends
    hit = None if refresh else store.read_frame("research_signals", "ear_3d")
    if hit is not None and hit.index.equals(panel.dates):
        return {"ear_3d": hit}
    market = equal_weight_market(frames)

    def one(t):
        dates = earnings_release_dates(company_filings(t, source=source)[1])
        ev = announcement_returns(frames[t]["Close"], market, dates)
        return hold_daily(ev, frames[t].index, EAR_MAX_AGE_DAYS)
    got, errors = map_tickers(one, list(frames), max_workers=8)
    store.write_json("research", "ear_3d_errors", errors)
    m = sample_at_month_ends({t: s for t, s in got.items() if s is not None}, panel.dates)
    store.write_frame("research_signals", "ear_3d", m)
    return {"ear_3d": m}


# --------------------------------------------------------------------------- #
# Earnings press releases (8-K Item 2.02, EX-99)
# --------------------------------------------------------------------------- #
_EX99 = re.compile(r"ex[-_]?99", re.I)


def in_eightk_sample(ticker: str, share: float = EIGHTK_SAMPLE) -> bool:
    """A fixed, reproducible sub-sample: the ticker's SHA-1 hash, not anything about the company."""
    h = int(hashlib.sha1(str(ticker).upper().encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share


def release_document(index: pd.DataFrame) -> Optional[str]:
    """The press release among a filing's documents: an HTML or text file named like an EX-99 exhibit ('ex99_1',
    'ex-99.1', 'ex991'); the first such exhibit when there are several (99.1 is the release, 99.2 often slides)."""
    if index is None or index.empty or "name" not in index:
        return None
    names = [str(n) for n in index["name"] if _EX99.search(str(n)) and str(n).lower().endswith((".htm", ".html", ".txt"))]
    return sorted(names)[0] if names else None


def release_tone(text_or_html: str, lm) -> Dict[str, float]:
    """Net tone and word count of one release (Loughran-McDonald word lists)."""
    from ..text import html_to_text, tokens, tone
    words = tokens(html_to_text(text_or_html))
    t = tone(words, lm)
    n = len(words)
    return {"words": float(n), "net_tone": float(t["positive"] - t["negative"]) if n else np.nan}


def eightk_history(ticker: str, lm, start: str = EIGHTK_START, refresh: bool = False,
                   source: str = "auto") -> pd.DataFrame:
    """Every earnings release the company filed since `start`, oldest first, indexed by filing date: accession,
    document, words, net tone (cached; the documents themselves are not stored)."""
    from ..edgar import cik_for, company_filings, fetch_document, filing_documents, registrant_cik
    hit = None if refresh else store.read_frame("research_8k", ticker)
    if hit is not None:
        return hit
    cik = cik_for(ticker)
    _, filings = company_filings(ticker, source=source)
    rows = []
    if cik is not None and filings is not None and not filings.empty:
        m = (filings["form"].isin(["8-K"]) & filings["items"].fillna("").str.contains(r"\b2\.02\b", regex=True)
             & (filings["filed"] >= start))
        for _, f in filings[m].sort_values("filed").iterrows():
            filer = registrant_cik(f, cik)
            doc = release_document(filing_documents(filer, f["accn"]))
            if doc is None:
                continue
            rows.append({"filed": f["filed"], "accn": f["accn"], "doc": doc,
                         **release_tone(fetch_document(filer, f["accn"], doc, cache=False), lm)})
    out = (pd.DataFrame(rows).set_index("filed") if rows else
           pd.DataFrame(columns=["accn", "doc", "words", "net_tone"]).rename_axis("filed"))
    store.write_frame("research_8k", ticker, out)
    return out


def eightk_tone_daily(history: pd.DataFrame, index: pd.DatetimeIndex, min_words: int = 100) -> pd.Series:
    """The tone change of the latest release, known from the day after its filing date, kept EIGHTK_MAX_AGE_DAYS.
    A release with fewer than `min_words` words (a stub pointing to a web page) is skipped."""
    h = history[pd.to_numeric(history["words"], errors="coerce") >= min_words].sort_index()
    h = h[~h.index.duplicated(keep="last")]
    change = pd.to_numeric(h["net_tone"], errors="coerce").diff()
    change.index = pd.DatetimeIndex(change.index) + pd.Timedelta(days=1)
    return hold_daily(change.dropna(), index, EIGHTK_MAX_AGE_DAYS).rename("eightk_tone")


def eightk_monthly(frames: Dict[str, pd.DataFrame], panel, refresh: bool = False, source: str = "auto",
                   share: float = EIGHTK_SAMPLE) -> Dict[str, pd.DataFrame]:
    """{'eightk_tone': month-end x ticker} on the fixed sub-sample (other members are missing, so they are untested)."""
    from ..parallel import map_tickers
    from ..text import lm_dictionary
    from .harness import sample_at_month_ends
    hit = None if refresh else store.read_frame("research_signals", "eightk_tone")
    if hit is not None and hit.index.equals(panel.dates):
        return {"eightk_tone": hit}
    lm = lm_dictionary()
    names = [t for t in frames if in_eightk_sample(t, share)]
    got, errors = map_tickers(lambda t: eightk_tone_daily(eightk_history(t, lm, refresh=refresh, source=source),
                                                          frames[t].index), names, max_workers=4)
    store.write_json("research", "eightk_errors", errors)
    if errors and len(errors) > 0.5 * max(len(names), 1):
        raise RuntimeError(f"8-K tone failed for {len(errors)} of {len(names)} stocks, e.g. "
                           + "; ".join(f"{k}: {v}" for k, v in list(errors.items())[:3]))
    m = sample_at_month_ends({t: v for t, v in got.items() if v is not None}, panel.dates)
    m = m.reindex(columns=list(frames))
    store.write_frame("research_signals", "eightk_tone", m)
    return {"eightk_tone": m}


# --------------------------------------------------------------------------- #
# 10-K text
# --------------------------------------------------------------------------- #
def text_history(ticker: str, lm, start: str = "2006-01-01", refresh: bool = False, source: str = "auto") -> pd.DataFrame:
    """Text features of every 10-K the company filed since `start`, oldest first, indexed by filing date (cached)."""
    from ..edgar import cik_for, company_filings, fetch_document, registrant_cik
    from ..text import features
    hit = None if refresh else store.read_frame("research_text", ticker)
    if hit is not None:
        return hit
    cik = cik_for(ticker)
    _, filings = company_filings(ticker, source=source)
    tenk = filings[filings["form"].isin(["10-K", "10-K405", "10-KT"]) & (filings["filed"] >= start)].sort_values("filed")
    rows, prev = [], None
    for _, f in tenk.iterrows():
        if not f.get("doc"):
            continue
        doc = fetch_document(registrant_cik(f, cik), f["accn"], f["doc"], cache=False)
        feats, counts = features(doc, lm, prev)
        prev = counts
        rows.append({"filed": f["filed"], "accn": f["accn"], **feats})
    out = pd.DataFrame(rows).set_index("filed") if rows else pd.DataFrame(columns=["accn"]).rename_axis("filed")
    store.write_frame("research_text", ticker, out)
    return out


def worker_sec_rate(processes: int) -> float:
    """Each text worker's share of this run's SEC budget, in requests a second. The budget is `http.SEC_RATE` (8, or
    less when LIGHTHOUSE_SEC_RATE gave this run a smaller share of the SEC's 10 because other processes download at
    the same time), or the SEC policy's current rate if something lowered it further. The workers have one rate
    limiter each, so together they stay within the budget."""
    from .. import http
    budget = min(float(http.SEC_RATE), float(http.POLICIES["sec.gov"].rate))
    return budget / max(int(processes), 1)


def _worker_init(sec_rate: float) -> None:
    from .. import http
    http.POLICIES["sec.gov"].rate = sec_rate          # this process's share of the run's SEC budget


def _text_one(args) -> tuple:
    ticker, refresh, source = args
    from ..text import lm_dictionary
    try:
        return ticker, text_history(ticker, lm_dictionary(), refresh=refresh, source=source), None
    except Exception as exc:  # noqa: BLE001 - one company's odd filing must not stop the panel
        return ticker, None, f"{type(exc).__name__}: {str(exc)[:160]}"


def text_monthly(frames: Dict[str, pd.DataFrame], panel, refresh: bool = False, source: str = "auto",
                 processes: int = 3) -> Dict[str, pd.DataFrame]:
    """{signal: month-end x ticker} for the text signals. Parsing is CPU-bound: `processes` worker processes share
    this run's SEC rate limit (`http.SEC_RATE` in total: 8 requests a second, or LIGHTHOUSE_SEC_RATE; see
    worker_sec_rate)."""
    from .harness import sample_at_month_ends
    cached = {n: store.read_frame("research_signals", n) for n in TEXT_FEATURES} if not refresh else {}
    if not refresh and all(v is not None and v.index.equals(panel.dates) for v in cached.values()):
        return cached
    from ..text import lm_dictionary
    lm_dictionary()                                    # download once, before the workers start
    hist: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    jobs = [(t, refresh, source) for t in frames]
    with ProcessPoolExecutor(max_workers=processes, initializer=_worker_init,
                             initargs=(worker_sec_rate(processes),)) as ex:
        for t, h, err in ex.map(_text_one, jobs, chunksize=4):
            if err:
                errors[t] = err
            elif h is not None and len(h):
                hist[t] = h
    store.write_json("research", "text_signal_errors", errors)
    if errors and len(errors) > 0.5 * max(len(frames), 1):
        raise RuntimeError(f"text features failed for {len(errors)} of {len(frames)} stocks, e.g. "
                           + "; ".join(f"{k}: {v}" for k, v in list(errors.items())[:3]))
    out = {}
    for name, (col, sign) in TEXT_FEATURES.items():
        daily = {}
        for t, h in hist.items():
            if col not in h:
                continue
            ev = pd.Series(sign * pd.to_numeric(h[col], errors="coerce").to_numpy(),
                           index=pd.DatetimeIndex(h.index) + pd.Timedelta(days=1)).dropna()
            daily[t] = hold_daily(ev, frames[t].index, TEXT_MAX_AGE_DAYS)
        m = sample_at_month_ends(daily, panel.dates)
        store.write_frame("research_signals", name, m)
        out[name] = m
    return out


def monthly(frames, panel, refresh: bool = False, source: str = "auto", text: bool = True,
            eightk: bool = True) -> Dict[str, pd.DataFrame]:
    out = earnings_monthly(frames, panel, refresh, source)
    if text:
        out.update(text_monthly(frames, panel, refresh, source))
    if eightk:
        out.update(eightk_monthly(frames, panel, refresh, source))
    return out


__all__ = ["announcement_returns", "hold_daily", "equal_weight_market", "earnings_monthly", "text_history",
           "text_monthly", "worker_sec_rate", "monthly", "DESCRIPTIONS", "TEXT_FEATURES", "in_eightk_sample",
           "release_document", "release_tone", "eightk_history", "eightk_tone_daily", "eightk_monthly"]
