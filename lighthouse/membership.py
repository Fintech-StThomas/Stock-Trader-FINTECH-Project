"""
Survivorship-aware universes: who was in the index on each date, not who is in it today.

Testing a rule on today's S&P 500 tests it on the companies that survived and grew into the index; the ones that
shrank, merged or went bust are missing, and every strategy looks better than it was. This module rebuilds historical
membership from free sources and says how much of it the free price feeds can actually see.

Sources
  * fja05680/sp500 on GitHub (open data): daily S&P 500 membership from 1996, built from Andreas Clenow's list (to
    2019) and Wikipedia's change log since; maintained by hand every few months.
  * Wikipedia, "Historical components of the S&P 500": the change log (effective date, added, removed), used to bring
    the GitHub file up to date, and "List of S&P 500 companies" for today's members (a cross-check).
  * SEC EDGAR: Form 25 / Form 15 filings mark delistings and deregistrations (`edgar.delistings`); cover-page share
    counts give market capitalisation (`market_cap_universe`).

Coverage: free price feeds (Yahoo) usually drop delisted tickers and sometimes reuse a ticker for a new company.
Every universe reports the share of member-days that have a price (`coverage`). A member without prices cannot be
backtested; research results are run under both delisting rules (total_return.DELISTING_RULES) and printed with it.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import store
from .data import DataUnavailable, offline_mode
from .edgar import normalize_ticker

FJA_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
           "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
WIKI_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
WIKI_CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FRESH_HOURS = 24.0 * 7          # the sources change a few times a month


def _get(url: str, **kw):
    from .http import fetch
    return fetch(url, **kw)


# --------------------------------------------------------------------------- #
# Parsing the sources
# --------------------------------------------------------------------------- #
def parse_fja_snapshots(text: str) -> pd.DataFrame:
    """The GitHub file (date, "TICKER,TICKER,...") as one row per (date, ticker)."""
    df = pd.read_csv(io.StringIO(text))
    rows = []
    for d, tickers in zip(pd.to_datetime(df["date"]), df["tickers"].astype(str)):
        for t in tickers.split(","):
            t = t.strip()
            if t:
                rows.append((d, normalize_ticker(t)))
    return pd.DataFrame(rows, columns=["date", "ticker"])


def snapshots_to_intervals(snaps: pd.DataFrame, last_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Membership intervals (ticker, start, end) from dated snapshots: a ticker is a member from the first snapshot
    that lists it until the day before the first later snapshot that does not (end NaT = still a member)."""
    dates = pd.DatetimeIndex(sorted(snaps["date"].unique()))
    by_date = snaps.groupby("date")["ticker"].apply(set).reindex(dates)
    open_: Dict[str, pd.Timestamp] = {}
    out = []
    prev: set = set()
    for d, members in by_date.items():
        for t in members - prev:
            open_[t] = d
        for t in prev - members:
            out.append((t, open_.pop(t), d - pd.Timedelta(days=1)))
        prev = members
    for t, s in open_.items():
        out.append((t, s, pd.NaT))
    iv = pd.DataFrame(out, columns=["ticker", "start", "end"]).sort_values(["ticker", "start"]).reset_index(drop=True)
    iv.attrs = {}
    return iv


def parse_wiki_changes(html: str) -> pd.DataFrame:
    """Wikipedia's change log as rows (date, added, removed, reason), tickers normalised; blanks are None."""
    tables = pd.read_html(io.StringIO(html))
    tab = None
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        if any("Effective Date" in c for c in cols) and any("Added" in c for c in cols):
            tab = t
            tab.columns = cols
            break
    if tab is None:
        raise DataUnavailable("the S&P 500 change table was not found on the Wikipedia page")
    col = lambda key: next(c for c in tab.columns if key in c)   # noqa: E731
    out = pd.DataFrame({
        "date": pd.to_datetime(tab[col("Effective Date")], errors="coerce"),
        "added": tab[col("Added Ticker")], "removed": tab[col("Removed Ticker")],
        "added_name": tab[col("Added Security")], "removed_name": tab[col("Removed Security")],
        "reason": tab[next((c for c in tab.columns if "Reason" in c), col("Effective Date"))]})
    for c in ("added", "removed"):
        out[c] = out[c].map(lambda x: normalize_ticker(x) if isinstance(x, str) and x.strip() and x.strip() != "nan" else None)
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def parse_wiki_current(html: str) -> pd.DataFrame:
    """Today's members from Wikipedia: ticker, name, GICS sector and sub-industry, CIK."""
    t = pd.read_html(io.StringIO(html))[0]
    out = pd.DataFrame({"ticker": t["Symbol"].map(normalize_ticker), "name": t["Security"],
                        "gics_sector": t["GICS Sector"], "gics_sub_industry": t["GICS Sub-Industry"],
                        "cik": pd.to_numeric(t.get("CIK"), errors="coerce")})
    return out


def apply_changes(intervals: pd.DataFrame, changes: pd.DataFrame, after: pd.Timestamp) -> Tuple[pd.DataFrame, List[str]]:
    """Bring membership intervals forward with the change log entries dated after `after`. Returns (intervals,
    notes on entries that did not match: an addition already in, a removal not in)."""
    iv = intervals.copy()
    notes = []
    for _, ch in changes[changes["date"] > after].iterrows():
        d = ch["date"]
        if ch["removed"]:
            m = (iv["ticker"] == ch["removed"]) & iv["end"].isna()
            if m.any():
                iv.loc[m, "end"] = d - pd.Timedelta(days=1)
            else:
                notes.append(f"{d.date()}: removal of {ch['removed']} does not match a current member")
        if ch["added"]:
            if ((iv["ticker"] == ch["added"]) & iv["end"].isna()).any():
                notes.append(f"{d.date()}: {ch['added']} is already a member")
            else:
                iv = pd.concat([iv, pd.DataFrame([{"ticker": ch["added"], "start": d, "end": pd.NaT}])], ignore_index=True)
    return iv.sort_values(["ticker", "start"]).reset_index(drop=True), notes


# --------------------------------------------------------------------------- #
# The universe object
# --------------------------------------------------------------------------- #
@dataclass
class Universe:
    """Index membership over time. `intervals`: ticker, start, end (NaT = still a member)."""
    name: str
    intervals: pd.DataFrame
    sources: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    as_of: Optional[pd.Timestamp] = None

    def members_on(self, date) -> List[str]:
        d = pd.Timestamp(date)
        iv = self.intervals
        m = (iv["start"] <= d) & (iv["end"].isna() | (iv["end"] >= d))
        return sorted(iv.loc[m, "ticker"].unique())

    def ever_members(self, start=None, end=None) -> List[str]:
        iv = self.intervals
        m = pd.Series(True, index=iv.index)
        if end is not None:
            m &= iv["start"] <= pd.Timestamp(end)
        if start is not None:
            m &= iv["end"].isna() | (iv["end"] >= pd.Timestamp(start))
        return sorted(iv.loc[m, "ticker"].unique())

    def membership_matrix(self, dates: Sequence) -> pd.DataFrame:
        """Boolean frame (dates x tickers): was the ticker a member on that date?"""
        dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        tickers = sorted(self.intervals["ticker"].unique())
        out = pd.DataFrame(False, index=dates, columns=tickers)
        for t, s, e in self.intervals[["ticker", "start", "end"]].itertuples(index=False):
            m = (dates >= s) & ((dates <= e) if pd.notna(e) else True)
            if m.any():
                out.loc[m, t] = True
        return out

    def coverage(self, priced: Dict[str, pd.DatetimeIndex], dates: Sequence) -> pd.DataFrame:
        """Survivorship coverage: per date, members, members with a price that day (within 5 sessions), and the share.
        `priced` maps ticker -> the dates it has prices."""
        mm = self.membership_matrix(dates)
        have = pd.DataFrame(False, index=mm.index, columns=mm.columns)
        for t in mm.columns:
            idx = priced.get(t)
            if idx is None or not len(idx):
                continue
            idx = pd.DatetimeIndex(idx)
            pos = idx.searchsorted(mm.index, side="right") - 1
            ok = pos >= 0
            last = pd.DatetimeIndex(np.where(ok, idx[np.clip(pos, 0, len(idx) - 1)], pd.NaT))
            have[t] = ok & ((mm.index - last) <= pd.Timedelta(days=7)) & (mm.index <= idx[-1] + pd.Timedelta(days=7))
        members = mm.sum(axis=1)
        with_px = (mm & have).sum(axis=1)
        return pd.DataFrame({"members": members, "with_prices": with_px, "coverage": with_px / members.replace(0, np.nan)})

    def summary(self) -> str:
        iv = self.intervals
        return (f"{self.name}: {iv['ticker'].nunique():,} tickers, {len(iv):,} membership spells, "
                f"{pd.Timestamp(iv['start'].min()).date()} to {self.as_of.date() if self.as_of is not None else 'today'}")


def sp500(refresh: bool = False, extend: bool = True) -> Universe:
    """Historical S&P 500 membership since 1996 (see the module docstring for the sources). Cached for a week."""
    cached = store.read_frame("universe", "sp500_intervals")
    meta = store.read_json("universe", "sp500_meta") or {}
    path = store.path_for("universe", "sp500_intervals")
    if cached is not None and not refresh and (offline_mode() or store.age_hours(path) < FRESH_HOURS):
        return Universe("S&P 500 (point in time)", cached, meta.get("sources", []), meta.get("notes", []),
                        pd.Timestamp(meta["as_of"]) if meta.get("as_of") else None)
    try:
        text = _get(FJA_URL, timeout=60).text
    except DataUnavailable:
        if cached is not None:
            return Universe("S&P 500 (point in time)", cached, meta.get("sources", []), meta.get("notes", []),
                            pd.Timestamp(meta["as_of"]) if meta.get("as_of") else None)
        raise
    snaps = parse_fja_snapshots(text)
    last = snaps["date"].max()
    iv = snapshots_to_intervals(snaps)
    sources = [f"fja05680/sp500 (GitHub), {snaps['date'].min().date()} to {last.date()}"]
    notes: List[str] = []
    as_of = last
    if extend:
        try:
            changes = parse_wiki_changes(_get(WIKI_CHANGES_URL, timeout=60).text)
            iv, n = apply_changes(iv, changes, last)
            notes += n
            newer = changes[changes["date"] > last]
            if len(newer):
                as_of = newer["date"].max()
                sources.append(f"Wikipedia change log, {len(newer)} changes after {last.date()} to {as_of.date()}")
            current = parse_wiki_current(_get(WIKI_CURRENT_URL, timeout=60).text)
            mine = set(iv.loc[iv["end"].isna(), "ticker"])
            theirs = set(current["ticker"])
            if mine != theirs:
                notes.append(f"vs Wikipedia's current list: {len(mine - theirs)} extra ({', '.join(sorted(mine - theirs)[:8])}), "
                             f"{len(theirs - mine)} missing ({', '.join(sorted(theirs - mine)[:8])})")
            store.write_frame("universe", "sp500_current", current)
        except DataUnavailable as exc:
            notes.append(f"not brought up to date with Wikipedia ({exc}); membership ends {last.date()}")
    store.write_frame("universe", "sp500_intervals", iv)
    store.write_json("universe", "sp500_meta", {"sources": sources, "notes": notes, "as_of": str(as_of.date())})
    return Universe("S&P 500 (point in time)", iv, sources, notes, as_of)


def from_intervals(name: str, intervals: pd.DataFrame, **kw) -> Universe:
    iv = intervals.copy()
    iv["start"] = pd.to_datetime(iv["start"])
    iv["end"] = pd.to_datetime(iv["end"])
    iv["ticker"] = iv["ticker"].map(normalize_ticker)
    return Universe(name, iv, **kw)


def static_universe(name: str, tickers: Iterable[str], start="1990-01-01") -> Universe:
    """A fixed list (for example today's large caps) as a Universe, so it reports coverage like the others; the name
    says it is survivorship-biased."""
    iv = pd.DataFrame({"ticker": [normalize_ticker(t) for t in tickers], "start": pd.Timestamp(start), "end": pd.NaT})
    return Universe(name, iv, notes=["today's list applied to the past: survivorship-biased"])


def sample_members(universe: Universe, date, n: int, seed: int = 7) -> List[str]:
    """`n` members on `date`, drawn at random (seeded): an unbiased sample of who was in the index then."""
    m = universe.members_on(date)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(m, size=min(n, len(m)), replace=False).tolist())


# --------------------------------------------------------------------------- #
# EDGAR: market capitalisation and delistings
# --------------------------------------------------------------------------- #
def cover_shares_series(ticker: str, source: str = "auto") -> pd.Series:
    """Shares outstanding from each filing's cover page (dei:EntityCommonStockSharesOutstanding, all classes summed),
    indexed by filing date: the count public on that date."""
    from .edgar import company_facts
    f = company_facts(ticker=ticker, source=source, concepts=["EntityCommonStockSharesOutstanding"])
    f = f[(f["concept"] == "EntityCommonStockSharesOutstanding") & (f["unit"] == "shares")]
    if f.empty:
        return pd.Series(dtype=float)
    g = f.groupby("accn").agg(val=("val", "sum"), filed=("filed", "min"), end=("end", "max"))
    g = g.sort_values(["filed", "end"]).drop_duplicates("filed", keep="last")
    return pd.Series(g["val"].to_numpy(), index=pd.DatetimeIndex(g["filed"]), name="shares")


def market_cap_on(date, tickers: Iterable[str], prices: Dict[str, pd.Series], source: str = "auto") -> pd.Series:
    """Market capitalisation on `date` = the latest cover-page share count filed by then x the as-traded close that day
    (`prices` holds split-adjusted closes adjusted at their last date; the split factor puts them back as traded).
    Missing shares or price -> NaN."""
    from .total_return import load_actions, split_factor_after
    d = pd.Timestamp(date)
    out = {}
    for t in tickers:
        px = prices.get(t)
        if px is None or px.empty or px.index[0] > d:
            out[t] = np.nan
            continue
        p = px[px.index <= d]
        if p.empty or (d - p.index[-1]).days > 7:
            out[t] = np.nan
            continue
        try:
            sh = cover_shares_series(t, source)
        except Exception:  # noqa: BLE001 - no EDGAR data: no market cap
            out[t] = np.nan
            continue
        sh = sh[sh.index <= d]
        if sh.empty:
            out[t] = np.nan
            continue
        acts = load_actions(t, source=source)
        traded = float(p.iloc[-1]) * float(split_factor_after([p.index[-1]], acts, px.index[-1])[0])
        out[t] = traded * float(sh.iloc[-1])
    return pd.Series(out, name="market_cap")


def market_cap_universe(candidates: Iterable[str], dates: Sequence, prices: Dict[str, pd.Series], top_n: int = 500,
                        source: str = "auto") -> Universe:
    """The `top_n` largest candidates by EDGAR market cap at each date (held until the next date). Candidates must be
    listed by the caller: free data has no list of every past ticker with its prices, which is the gap this costs."""
    dates = sorted(pd.to_datetime(list(dates)))
    rows = []
    cands = list(candidates)
    for k, d in enumerate(dates):
        mc = market_cap_on(d, cands, prices, source).dropna().sort_values(ascending=False).head(top_n)
        end = dates[k + 1] - pd.Timedelta(days=1) if k + 1 < len(dates) else pd.NaT
        rows += [(t, d, end) for t in mc.index]
    iv = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    return Universe(f"Top {top_n} by EDGAR market cap", iv,
                    notes=["market cap = cover-page shares x as-traded close; candidates supplied by the caller"])


def delisting_dates(names: Dict[str, str], start_year: int = 2001) -> pd.DataFrame:
    """Match company names (ticker -> name, e.g. Wikipedia's removed securities) to EDGAR Form 25 / Form 15 filers by
    normalised name, and return ticker, cik, form, filed for the matches (name matching: check before relying on it)."""
    from .edgar import delistings
    d = delistings(start_year)
    d = d.assign(key=d["name"].map(_name_key))
    rows = []
    for t, name in names.items():
        k = _name_key(name)
        hit = d[d["key"] == k]
        if len(hit):
            h = hit.iloc[0]
            rows.append((t, int(h["cik"]), h["form"], h["filed"], h["name"]))
    return pd.DataFrame(rows, columns=["ticker", "cik", "form", "filed", "edgar_name"])


def _name_key(name: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    stop = {"inc", "corp", "corporation", "co", "company", "ltd", "plc", "the", "holdings", "group", "llc", "lp", "sa", "nv"}
    return " ".join(w for w in s.split() if w not in stop)
