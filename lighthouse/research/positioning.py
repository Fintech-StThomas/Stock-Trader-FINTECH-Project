"""
Positioning: what short sellers are doing, from free regulatory data.

  short_volume     minus the share of reported volume that was short sales over the last 5 sessions of the month
                   (FINRA Reg SHO daily files: the Nasdaq, NYSE and ADF trade reporting facilities until August 2018,
                   the consolidated file since). It counts off-exchange short sales, most of them market-making, so it
                   is a weak proxy for short positions. On the forward watch list (lighthouse/research/watch.py).

The insider (Form 4) and days-to-cover signals were tested once, failed, and were removed in October 2026
(docs/METHODOLOGY.md section 8b).
"""
from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .. import store
from ..data import DataUnavailable, is_trading_day

FINRA_SHVOL = "https://cdn.finra.org/equity/regsho/daily/{name}shvol{date}.txt"
POSITIONING_SIGNALS = ["short_volume"]
DESCRIPTIONS = {"short_volume": "minus the short share of reported volume (5 sessions)"}


def _get(url: str, **kw):
    from ..http import fetch
    return fetch(url, **kw)


def _norm_symbols(s):
    """Upper case, with a class-share separator written as the price panel writes it ('BRK-B'): FINRA's daily files
    write 'BRK/B' (in both the per-facility and the consolidated era), SEC filers 'BRK.B'. Keeps NaN as NaN."""
    return s.str.upper().str.strip().str.replace(r"[./]", "-", regex=True)


# --------------------------------------------------------------------------- #
# FINRA: short sale volume and short interest
# --------------------------------------------------------------------------- #
def _normalised_day(hit: pd.DataFrame) -> pd.DataFrame:
    """A cached session as short_volume_day writes it now: one row per normalised symbol. Files cached before October
    2026 renamed '.' to '-' after summing, so two raw symbols could share a label (and reindexing failed with
    "duplicate labels"), and kept 'BRK/B' as is. They are fixed here when read; the cache file is left as it is."""
    idx = _norm_symbols(pd.Index(hit.index, dtype=object))
    if hit.index.is_unique and idx.equals(hit.index):
        return hit
    out = hit.set_axis(idx).groupby(level=0)[["ShortVolume", "TotalVolume"]].sum()
    out.index.name = hit.index.name
    return out


SHVOL_COLUMNS = {5: ["Date", "Symbol", "ShortVolume", "TotalVolume", "Market"],
                 6: ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"]}


def _read_shvol(txt: str) -> pd.DataFrame:
    """One FINRA daily file as a table. Most start with a header line; some early Nasdaq files have none (FNSQ, 22
    November 2010, starts with its first row), and are read with the columns of their field count."""
    first = txt.lstrip().split("\n", 1)[0].strip()
    if first.startswith("Date|"):
        return pd.read_csv(io.StringIO(txt), sep="|", dtype={"Symbol": str})
    names = SHVOL_COLUMNS.get(first.count("|") + 1)
    if names is None:
        raise ValueError(f"unrecognised FINRA short-volume file starting {first[:60]!r}")
    return pd.read_csv(io.StringIO(txt), sep="|", dtype={"Symbol": str}, header=None, names=names)


# How cdn.finra.org answers a file that does not exist: a facility that did not publish that day (FNQC before it
# opened). Any other failure (a timeout, a server error that outlasted the retries) says nothing about the file.
FINRA_NOT_PUBLISHED = (403, 404, 410)


def _session_files(d: pd.Timestamp):
    """(the files of one session as tables, each facility's answer: 'published' | 'not published' | 'failed')."""
    key = d.strftime("%Y%m%d")
    names = ["CNMS"] if d >= pd.Timestamp("2018-08-01") else ["FNSQ", "FNYX", "FNQC", "FNRA"]
    parts, answers = [], {}
    for n in names:
        try:
            txt = _get(FINRA_SHVOL.format(name=n, date=key), timeout=60).text
        except DataUnavailable as exc:
            answers[n] = "not published" if getattr(exc, "status", None) in FINRA_NOT_PUBLISHED else "failed"
            continue
        df = _read_shvol(txt)
        df = df[pd.to_numeric(df["Date"], errors="coerce").notna()]
        parts.append(df[["Symbol", "ShortVolume", "TotalVolume"]])
        answers[n] = "published"
    return parts, answers


def _short_volume_day(day, refresh: bool = False):
    """(the session's table or None, complete): complete unless a facility's download failed (see short_volume_day)."""
    d = pd.Timestamp(day)
    key = d.strftime("%Y%m%d")
    hit = None if refresh else store.read_frame("finra_shvol", key)
    if hit is not None:
        return _normalised_day(hit), True
    parts, answers = _session_files(d)
    complete = "failed" not in answers.values()
    if not parts:
        return None, complete
    raw = pd.concat(parts)
    raw = raw.assign(Symbol=_norm_symbols(raw["Symbol"]))
    out = raw.groupby("Symbol")[["ShortVolume", "TotalVolume"]].sum()
    if complete:
        store.write_frame("finra_shvol", key, out)
        store.write_json("finra_shvol", key, {"facilities": answers, "rows": int(len(out))})
    return out, complete


def short_volume_day(day, refresh: bool = False) -> Optional[pd.DataFrame]:
    """Symbol, short volume and total volume for one session (all trade reporting facilities summed).

    Until July 2018 FINRA published one file per facility (FNSQ, FNQC: the Nasdaq TRFs in Carteret and Chicago; FNYX:
    the NYSE TRF; FNRA: the ADF); a facility that did not exist yet (FNQC in January 2015) is answered with HTTP 403
    and skipped. Each file ends with a line holding its record count, which has no symbol and is dropped. Class shares
    are written 'BRK/B' and stored as 'BRK-B'.

    A session is cached only when every facility answered: with its file, or with "no such file" (FINRA_NOT_PUBLISHED).
    When a download failed (a timeout, a server error through every retry) the session is returned from the files that
    came, as a session without any file is skipped, but it is not cached, and the next call tries again. The cache
    records each facility's answer (finra_shvol/<date>.json). A session cached before October 2026 has no record and
    may lack a facility that failed then: its table holds only the summed volumes, so it cannot be told apart from a
    complete one; `refresh=True` downloads a session again. A session without any file is not cached."""
    return _short_volume_day(day, refresh)[0]


def short_volume_signal(dates: pd.DatetimeIndex, tickers: Iterable[str], sessions: int = 5) -> pd.DataFrame:
    """Minus the short share of volume over the `sessions` trading days before each month's last session. FINRA posts
    a session's file in the evening, after the close at which the signal is read, so the month-end session itself is
    not known yet. The sessions whose download failed in part or in full (see short_volume_day) are listed in
    `attrs["incomplete_sessions"]`: the values use what came, and `monthly` does not cache a signal built on them."""
    tickers = list(tickers)
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    incomplete: List[str] = []
    for d in dates:
        if d < pd.Timestamp("2009-08-03"):
            continue
        days, x = [], d - pd.Timedelta(days=1)
        while len(days) < sessions:
            if is_trading_day(x):
                days.append(x)
            x -= pd.Timedelta(days=1)
        tot = None
        for day in days:
            sv, complete = _short_volume_day(day)
            if not complete:
                incomplete.append(str(pd.Timestamp(day).date()))
            if sv is None:
                continue
            tot = sv if tot is None else tot.add(sv, fill_value=0.0)
        if tot is None:
            continue
        ratio = (tot["ShortVolume"] / tot["TotalVolume"].replace(0, np.nan)).reindex(tickers)
        out.loc[d] = -ratio.to_numpy(float)
    out.attrs["incomplete_sessions"] = sorted(incomplete)
    return out


def monthly(frames, panel, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    out = {}
    for n in POSITIONING_SIGNALS:
        hit = None if refresh else store.read_frame("research_signals", n)
        if hit is not None and hit.index.equals(panel.dates):
            out[n] = hit
    tickers = list(panel.prices.columns)
    if "short_volume" not in out:
        out["short_volume"] = short_volume_signal(panel.dates, tickers)
        incomplete = out["short_volume"].attrs.get("incomplete_sessions", [])
        store.write_json("research", "short_volume_incomplete", incomplete)
        if not incomplete:         # a session that failed to download is tried again next time, so is the signal
            store.write_frame("research_signals", "short_volume", out["short_volume"])
    return out


__all__ = ["POSITIONING_SIGNALS", "DESCRIPTIONS", "short_volume_day", "short_volume_signal", "monthly"]
