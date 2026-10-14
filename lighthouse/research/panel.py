"""
Many price series at once, politely: the disk cache first (any age, research does not need today's bar), Yahoo for the
rest in parallel, a second slower pass for the tickers Yahoo throttled, and a remembered list of tickers that have no
data (dead tickers are asked about at most once a week instead of on every run).

Yahoo throttles bursts (HTTP 429). In October 2026, 49 of 479 failures in an 8-thread pull of every S&P 500 member since
1996 were throttling, not missing data; 28 of them loaded on a 2-thread retry.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .. import store
from ..data import DataUnavailable, _from_cache, clean_prices, load_prices, offline_mode
from ..parallel import map_tickers

MISSING_DAYS = 7


def _throttled(err: str) -> bool:
    e = str(err)
    return "429" in e or "Too Many" in e or "rate limit" in e.lower() or "rate-limit" in e.lower()


def _missing_book() -> Dict[str, str]:
    return store.read_json("research", "missing_prices") or {}


def load_frames(tickers: Iterable[str], start="1994-01-01", source: str = "auto", max_workers: int = 8,
                progress: Optional[Callable[[str, int, int], None]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """({ticker: clean OHLCV frame from `start`}, {ticker: why it is missing}). Offline (or source "cache"), only the
    cache is read."""
    tickers = list(dict.fromkeys(t.upper().strip() for t in tickers if t and t.strip()))
    start_ts = pd.Timestamp(start)
    frames: Dict[str, pd.DataFrame] = {}
    missing: Dict[str, str] = {}
    book = _missing_book()
    now = pd.Timestamp.now()
    todo: List[str] = []
    for t in tickers:
        try:
            raw = _from_cache(t, max_age_hours=None, need_start=start_ts.to_pydatetime())
            df, _ = clean_prices(raw, t, "cache")
            frames[t] = df[df.index >= start_ts]
            continue
        except DataUnavailable:
            pass
        hit = book.get(t)
        if hit and (now - pd.Timestamp(hit.split("|")[0])).days < MISSING_DAYS:
            missing[t] = "no data (remembered) " + hit.split("|", 1)[-1]
            continue
        todo.append(t)
    if offline_mode() or source in {"cache", "sample"} or not todo:
        for t in todo:
            missing[t] = "not in the cache (offline)"
        return frames, missing

    def one(t: str) -> pd.DataFrame:
        return load_prices(t, start=start_ts, years=None, source="yfinance")

    got, err = map_tickers(one, todo, max_workers=max_workers, source="auto", on_done=progress)
    retry = [t for t, e in err.items() if _throttled(e)]
    if retry:
        got2, err2 = map_tickers(one, retry, max_workers=2, source="auto")
        got.update(got2)
        for t in retry:
            err.pop(t, None)
        err.update(err2)
    for t, df in got.items():
        df.attrs = {}
        frames[t] = df
    for t, e in err.items():
        missing[t] = e
        if not _throttled(e):
            book[t] = f"{now.isoformat(timespec='seconds')}|{str(e)[-80:]}"
    store.write_json("research", "missing_prices", book)
    return frames, missing


def closes(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    return {t: df["Close"] for t, df in frames.items()}
