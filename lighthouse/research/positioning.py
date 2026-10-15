"""
Positioning signals: what insiders and short sellers are doing, from free regulatory data.

  insider_buyers   distinct insiders (officers, directors, 10% owners) who bought shares in the open market (Form 4,
                   transaction code P) in the 90 days to the month end, by filing date (Cohen, Malloy & Pomorski 2012:
                   routine trades carry no information, opportunistic purchases do; several buyers at once are rarer
                   and stronger). Source: the SEC's quarterly Insider Transactions Data Sets (2006 onwards).
  insider_net      open-market purchases minus sales by insiders in the same window, as a share of market value
  short_volume     minus the share of reported volume that was short sales over the last 5 sessions of the month
                   (FINRA Reg SHO daily files: the Nasdaq, NYSE and ADF trade reporting facilities until August 2018,
                   the consolidated file since). It counts off-exchange short sales, most of them market-making, so it
                   is a weak proxy for short positions.
  days_to_cover    minus short interest / average daily volume at the last settlement date of the month (FINRA
                   consolidated short interest, available from 2018 only: a hold-out-period-only, exploratory signal)

Borrow fees and availability are not free; shorting costs are an assumption (StrategyParams.short_borrow_fee).
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .. import store
from ..data import DataUnavailable, is_trading_day, offline_mode

SEC_INSIDER_INDEX = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
FINRA_SHVOL = "https://cdn.finra.org/equity/regsho/daily/{name}shvol{date}.txt"
FINRA_SI_API = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
POSITIONING_SIGNALS = ["insider_buyers", "insider_net", "short_volume", "days_to_cover"]
DESCRIPTIONS = {"insider_buyers": "distinct insiders buying in the open market (90 days)",
                "insider_net": "insider open-market net purchases / market value (90 days)",
                "short_volume": "minus the short share of reported volume (5 sessions)",
                "days_to_cover": "minus short interest / average daily volume (FINRA, 2018 onwards)"}


def _get(url: str, **kw):
    from ..http import fetch
    return fetch(url, **kw)


def _norm_symbols(s):
    """Upper case, with a class-share separator written as the price panel writes it ('BRK-B'): FINRA's daily files
    write 'BRK/B' (in both the per-facility and the consolidated era), SEC filers 'BRK.B'. Keeps NaN as NaN."""
    return s.str.upper().str.strip().str.replace(r"[./]", "-", regex=True)


def _by_ticker(values: pd.Series, tickers: List[str]) -> np.ndarray:
    """`values` (indexed by symbol) at `tickers`; a class share missing under its own name ('BRK-B') falls back to the
    symbol without its separator ('BRKB'), as FINRA's short-interest data writes Berkshire and Brown-Forman."""
    v = values[~values.index.duplicated(keep="first")]
    own = v.reindex(tickers).to_numpy(float)
    bare = v.reindex([t.replace("-", "") for t in tickers]).to_numpy(float)
    dashed = np.array(["-" in t for t in tickers], dtype=bool)
    return np.where(np.isnan(own) & dashed, bare, own)


# --------------------------------------------------------------------------- #
# Form 4: insider transactions
# --------------------------------------------------------------------------- #
def insider_files() -> List[str]:
    """URLs of every quarterly Insider Transactions Data Set."""
    hit = store.read_json("insider", "index", None if offline_mode() else 24.0 * 7)
    if hit:
        return hit
    t = _get(SEC_INSIDER_INDEX).text
    links = sorted(set(re.findall(r'href="([^"]+form345\.zip)"', t)))
    urls = ["https://www.sec.gov" + l if l.startswith("/") else l for l in links]
    store.write_json("insider", "index", urls)
    return urls


# Non-derivative rows that are not the issuer's common equity: an officer buying a bank's preferred stock at par
# (JPMorgan, January 2020: $5.1m of Series HH) says little about the common shares the signal ranks.
NOT_COMMON = r"prefer|\bpfd\b|\bnotes?\b|debenture|\bbonds?\b|warrant"
# No insider's open-market trade on one Form 4 line comes near this; a larger value is a typing error.
MAX_TRADE_VALUE = 5e9
# A US exchange name written in front of the symbol ('NYSE: SCS', 'NASDAQ:XYZ'); a foreign one ('ASX:CRN') is left
# alone, so the row cannot match a US ticker by accident.
_EXCHANGE_PREFIX = r"^(?:NYSE(?:\s*(?:AMERICAN|MKT|ARCA))?|NASDAQ(?:\s*[A-Z]{2})?|AMEX)\s*:\s*"


def _issuer_symbols(raw: pd.Series) -> pd.Series:
    """The symbols of ISSUERTRADINGSYMBOL, a free-text field: one list per row. Quotes, brackets, question marks and
    then a US exchange prefix are removed ('(NYSE:FBC)' -> FBC); a field listing several classes is split on commas and
    semicolons ('LEN, LEN-B' -> LEN and LEN-B), never on spaces ('N O G' is Northern Oil & Gas, not N, O and G)."""
    s = raw.str.upper().str.replace(r"[\"'()\[\]?]", "", regex=True).str.strip()
    s = s.str.replace(_EXCHANGE_PREFIX, "", regex=True)
    return s.str.split(r"\s*[,;]\s*", regex=True)


def parse_form345(blob: bytes) -> pd.DataFrame:
    """Open-market purchases (P) and sales (S) of common stock from one data set: filed, issuer CIK and symbol, owner,
    relationship, shares, price, value (+ for purchases).

    Forms 4 and 4/A only (a Form 5 reports a year's exempt transactions months late). Rows in preferred stock, notes,
    debentures, bonds or warrants are dropped (NOT_COMMON). A joint filing counts once, under its first reporting
    owner. A filing that names several share classes appears once per class symbol. A row worth more than
    MAX_TRADE_VALUE is a data-entry error (the total typed into the price field: one 2020 Q1 row came to $1.56
    trillion); it keeps its buyer but its dollar value is missing. `doc` (4 or 4/A) and `trans_date` let
    `insider_trades` drop amendments that repeat an original."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    tr = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str,
                     usecols=["ACCESSION_NUMBER", "SECURITY_TITLE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                              "TRANS_ACQUIRED_DISP_CD", "TRANS_DATE"])
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"])]
    tr = tr[~tr["SECURITY_TITLE"].fillna("").str.contains(NOT_COMMON, case=False, regex=True)]
    own = pd.read_csv(z.open("REPORTINGOWNER.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
    own = own.drop_duplicates("ACCESSION_NUMBER")
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    df = df.assign(_sym=_issuer_symbols(df["ISSUERTRADINGSYMBOL"])).explode("_sym", ignore_index=True)
    sh = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    px = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    sign = np.where(df["TRANS_CODE"] == "P", 1.0, -1.0)
    out = pd.DataFrame({"filed": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
                        "issuer_cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce"),
                        "symbol": _norm_symbols(df["_sym"].astype(object)),
                        "owner": df["RPTOWNERCIK"], "relationship": df["RPTOWNER_RELATIONSHIP"], "code": df["TRANS_CODE"],
                        "shares": sh, "price": px, "value": sign * sh * px, "doc": df["DOCUMENT_TYPE"],
                        "trans_date": pd.to_datetime(df["TRANS_DATE"], format="%d-%b-%Y", errors="coerce")})
    out["value"] = out["value"].where(out["value"].abs() <= MAX_TRADE_VALUE)
    return out.dropna(subset=["filed"]).reset_index(drop=True)


AMENDMENT_KEY = ["owner", "symbol", "trans_date", "code", "shares", "price"]


def drop_repeated_amendments(trades: pd.DataFrame) -> pd.DataFrame:
    """Leave out 4/A rows that repeat a transaction already reported on a Form 4 (same owner, symbol, trade date, code,
    shares and price): an amendment often restates the whole form, and counting it again doubles the trade. A 4/A row
    that differs (a corrected number) is kept; the original it corrects stays too, as it was public first."""
    if not set(AMENDMENT_KEY + ["doc"]) <= set(trades.columns) or trades.empty:
        return trades
    keys = pd.util.hash_pandas_object(trades[AMENDMENT_KEY], index=False)     # missing values hash alike
    seen = set(keys[trades["doc"] == "4"])
    repeat = (trades["doc"] == "4/A") & keys.isin(seen)
    return trades[~repeat].reset_index(drop=True)


def insider_trades(refresh: bool = False) -> pd.DataFrame:
    """Every open-market insider purchase and sale since 2006 (cached per quarter; past quarters never change), with
    repeated amendments left out. A quarter cached by an older parser (without the `doc` column, so without the
    October 2026 fixes: preferred stock, free-text symbols, outliers) is downloaded and parsed again."""
    parts = []
    for url in insider_files():
        key = url.rsplit("/", 1)[-1].replace("_form345.zip", "")
        hit = None if refresh else store.read_frame("insider", key)
        if hit is None or "doc" not in hit.columns:
            hit = parse_form345(_get(url, timeout=180).content)
            store.write_frame("insider", key, hit)
        parts.append(hit)
    return drop_repeated_amendments(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()


_CLASS_SHARE = re.compile(r"^[A-Z]+-[A-Z]$")          # a share class as the price panel writes it: BRK-B, BF-B


def issuer_ciks_for(tickers: Iterable[str], table: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """{member: issuer CIK} for the members whose insider trades are matched by issuer rather than by symbol: a class
    share (BRK-B, BF-B) and every class of an issuer with two or more classes among `tickers` (GOOGL and GOOG, FOXA and
    FOX, NWSA and NWS). Every class of a company files under its one CIK, but a Form 4 names the class traded, in the
    filer's spelling: Berkshire's insiders file as 'BRK.A', Brown-Forman's as 'BFA, BFB'. The CIKs come from EDGAR's
    ticker table (today's tickers, as the fundamental signals use it); empty when it cannot be read, and then trades
    match by symbol only."""
    tickers = list(dict.fromkeys(tickers))
    if table is None:
        from ..edgar import ticker_table
        try:
            table = ticker_table()
        except DataUnavailable:
            return {}
    cik = table.drop_duplicates("ticker").set_index("ticker")["cik"]
    known = {t: int(cik[t]) for t in tickers if t in cik.index and pd.notna(cik[t])}
    per_cik = pd.Series(known).value_counts() if known else pd.Series(dtype=int)
    return {t: c for t, c in known.items() if _CLASS_SHARE.match(t) or per_cik.get(c, 0) > 1}


# Every column of a trade except its symbol: the copies of one Form 4 line listed under several class symbols agree on
# all of them.
_LINE_COLUMNS = ["filed", "issuer_cik", "owner", "relationship", "code", "shares", "price", "value", "doc",
                 "trans_date"]


def member_trades(trades: pd.DataFrame, tickers: Iterable[str],
                  issuer_ciks: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """The trades of `tickers`, one row per trade and member (in `member`). A trade belongs to the member its filed
    symbol names, which also covers companies that no longer exist. A member in `issuer_ciks` (see `issuer_ciks_for`)
    also takes every trade filed under its issuer's CIK, whatever class the symbol names: BRK-B takes Berkshire's
    'BRK.A' filings, and the two classes of an issuer in the index share its trades (a company's insiders trade one
    company, as its two classes share one set of fundamentals). CIKs are never reused, so no other company's trades
    can match this way (Brooks Automation's 'BRKS' or Saul Centers' 'BFS' stay theirs).

    `parse_form345` lists a line that names several classes ('BFA, BFB') once per symbol, and those copies differ only
    in the symbol; when two of them reach one member, the line counts once for it: among the rows that agree on every
    column but the symbol, the member keeps those of one symbol (the most frequent, so two identical lines on one form
    both count). Members matched by symbol only are not touched by this."""
    tickers = list(dict.fromkeys(tickers))
    by_symbol = trades[trades["symbol"].isin(tickers)].assign(member=lambda x: x["symbol"])
    issuer_ciks = {t: int(c) for t, c in (issuer_ciks or {}).items() if t in set(tickers) and pd.notna(c)}
    if not issuer_ciks or "issuer_cik" not in trades.columns:
        return by_symbol
    # members matched by symbol keep their rows in their order (their sums are exactly what they were)
    parts = [by_symbol[~by_symbol["member"].isin(list(issuer_ciks))]]
    grouped = trades[trades["symbol"].isin(list(issuer_ciks)) | trades["issuer_cik"].isin(list(issuer_ciks.values()))]
    for t, cik in issuer_ciks.items():
        hit = grouped[(grouped["symbol"] == t) | (grouped["issuer_cik"] == cik)]
        if hit.empty:
            continue
        cols = [c for c in _LINE_COLUMNS if c in hit.columns]
        frame = pd.DataFrame({"line": pd.util.hash_pandas_object(hit[cols], index=False).to_numpy(),  # NaN hash alike
                              "symbol": hit["symbol"].fillna("").astype(str).to_numpy(), "n": 1})
        frame["n"] = frame.groupby(["line", "symbol"])["n"].transform("size")
        best = frame.sort_values("n", ascending=False, kind="stable").drop_duplicates("line")
        pairs = pd.MultiIndex.from_frame(frame[["line", "symbol"]])
        keep = pairs.isin(pd.MultiIndex.from_frame(best[["line", "symbol"]]))
        parts.append(hit[keep].assign(member=t))
    return pd.concat(parts).reset_index(drop=True)


def insider_signals(trades: pd.DataFrame, dates: pd.DatetimeIndex, tickers: Iterable[str],
                    market_cap: Optional[pd.DataFrame] = None, window_days: int = 90,
                    issuer_ciks: Optional[Dict[str, int]] = None) -> Dict[str, pd.DataFrame]:
    """insider_buyers and insider_net at each month end, by the issuer's trading symbol as filed (which also covers
    companies that no longer exist), and for share classes by the issuer's CIK (`member_trades`). A filing counts
    from the day after its filing date: EDGAR accepts Forms 4 until 10 pm and dates them that day, after the close at
    which the signal is read (as the fundamental and text signals use a filing from the session after it). The window
    is the `window_days` days to the month end, by that date."""
    tickers = list(tickers)
    t = member_trades(trades, tickers, issuer_ciks)
    known = t["filed"] + pd.Timedelta(days=1)
    buyers = pd.DataFrame(0.0, index=dates, columns=tickers)
    net = pd.DataFrame(0.0, index=dates, columns=tickers)
    for d in dates:
        w = t[(known > d - pd.Timedelta(days=window_days)) & (known <= d)]
        if w.empty:
            continue
        b = w[w["code"] == "P"].groupby("member")["owner"].nunique()
        buyers.loc[d, b.index] = b.to_numpy(float)
        v = w.groupby("member")["value"].sum()
        net.loc[d, v.index] = v.to_numpy(float)
    if market_cap is not None:
        net = (net / market_cap.reindex(index=dates, columns=tickers)).where(market_cap.reindex(index=dates, columns=tickers) > 0)
    else:
        net = net * np.nan
    first = known.min() if len(t) else None
    if first is not None:
        buyers.loc[buyers.index < first] = np.nan
        net.loc[net.index < first] = np.nan
    return {"insider_buyers": buyers, "insider_net": net}


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


def short_volume_day(day) -> Optional[pd.DataFrame]:
    """Symbol, short volume and total volume for one session (all trade reporting facilities summed).

    Until July 2018 FINRA published one file per facility (FNSQ, FNQC: the Nasdaq TRFs in Carteret and Chicago; FNYX:
    the NYSE TRF; FNRA: the ADF); a facility that did not exist yet (FNQC in January 2015) is answered with HTTP 403
    and skipped. Each file ends with a line holding its record count, which has no symbol and is dropped. Class shares
    are written 'BRK/B' and stored as 'BRK-B'."""
    d = pd.Timestamp(day)
    key = d.strftime("%Y%m%d")
    hit = store.read_frame("finra_shvol", key)
    if hit is not None:
        return _normalised_day(hit)
    names = ["CNMS"] if d >= pd.Timestamp("2018-08-01") else ["FNSQ", "FNYX", "FNQC", "FNRA"]
    parts = []
    for n in names:
        try:
            txt = _get(FINRA_SHVOL.format(name=n, date=key), timeout=60).text
        except DataUnavailable:
            continue
        df = _read_shvol(txt)
        df = df[pd.to_numeric(df["Date"], errors="coerce").notna()]
        parts.append(df[["Symbol", "ShortVolume", "TotalVolume"]])
    if not parts:
        return None
    raw = pd.concat(parts)
    raw = raw.assign(Symbol=_norm_symbols(raw["Symbol"]))
    out = raw.groupby("Symbol")[["ShortVolume", "TotalVolume"]].sum()
    store.write_frame("finra_shvol", key, out)
    return out


def short_volume_signal(dates: pd.DatetimeIndex, tickers: Iterable[str], sessions: int = 5) -> pd.DataFrame:
    """Minus the short share of volume over the `sessions` trading days before each month's last session. FINRA posts
    a session's file in the evening, after the close at which the signal is read, so the month-end session itself is
    not known yet."""
    tickers = list(tickers)
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
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
            sv = short_volume_day(day)
            if sv is None:
                continue
            tot = sv if tot is None else tot.add(sv, fill_value=0.0)
        if tot is None:
            continue
        ratio = (tot["ShortVolume"] / tot["TotalVolume"].replace(0, np.nan)).reindex(tickers)
        out.loc[d] = -ratio.to_numpy(float)
    return out


def short_interest_on(settlement) -> pd.DataFrame:
    """FINRA consolidated short interest at one settlement date: symbol, short position, average daily volume, days to
    cover (paged; cached). The API answers a date that is not a settlement date with HTTP 204 and no body (an empty
    table). FINRA's own days to cover is short / ADV rounded to 0.01, floored at 1.00 and capped at 999.99 (also when
    the ADV is zero); it is kept as published, and days_to_cover_signal divides the two quantities instead."""
    d = pd.Timestamp(settlement)
    key = str(d.date())
    hit = store.read_frame("finra_si", key)
    if hit is not None:
        return hit
    rows, offset = [], 0
    while True:
        body = {"limit": 5000, "offset": offset,
                "compareFilters": [{"compareType": "equal", "fieldName": "settlementDate", "fieldValue": key}]}
        txt = _get(FINRA_SI_API, method="POST", json_body=body, headers={"Accept": "text/plain"}, timeout=90).text
        if not txt.strip():
            break
        df = pd.read_csv(io.StringIO(txt))
        rows.append(df)
        if len(df) < 5000:
            break
        offset += 5000
    if not rows:
        out = pd.DataFrame(columns=["short", "adv", "days_to_cover"])
    else:
        df = pd.concat(rows)
        out = pd.DataFrame({"short": pd.to_numeric(df["currentShortPositionQuantity"], errors="coerce").to_numpy(),
                            "adv": pd.to_numeric(df["averageDailyVolumeQuantity"], errors="coerce").to_numpy(),
                            "days_to_cover": pd.to_numeric(df["daysToCoverQuantity"], errors="coerce").to_numpy()},
                           index=_norm_symbols(df["symbolCode"].astype(str)))
        out = out[~out.index.duplicated(keep="first")]
    # an empty answer for a date FINRA may not have published yet (it publishes ~7 business days after settlement;
    # a month still running asks for a future date) is not cached: it would stay empty once the data is out
    if len(out) or d < pd.Timestamp.today().normalize() - pd.Timedelta(days=30):
        store.write_frame("finra_si", key, out)
    return out


def settlement_dates(month_end: pd.Timestamp) -> pd.Timestamp:
    """FINRA's month-end settlement date: the last business day of the month (approximately; holidays move it back)."""
    d = pd.Timestamp(month_end) + pd.offsets.MonthEnd(0)
    while not is_trading_day(d):
        d -= pd.Timedelta(days=1)
    return d


def days_to_cover_signal(dates: pd.DatetimeIndex, tickers: Iterable[str], start="2018-01-01") -> pd.DataFrame:
    """Minus days to cover at each month's settlement date (known about a week later, so it is used from the NEXT
    month end: FINRA publishes short interest ~7 business days after settlement).

    Days to cover is short interest / average daily volume, computed from the two quantities: FINRA's published
    figure floors it at 1.00 (AMD, December 2023: 0.95 published as 1.00), which ties the most liquid stocks, and
    writes 999.99 for a zero volume (here: missing). Class shares are matched without their separator ('BRKB')."""
    tickers = list(tickers)
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    prev = None
    for d in dates:
        if prev is not None:
            out.loc[d] = prev
        if d < pd.Timestamp(start):
            continue
        si = short_interest_on(settlement_dates(d))
        if si.empty:
            # the month-end settlement may sit a day earlier: try the previous business days
            x = settlement_dates(d) - pd.Timedelta(days=1)
            for _ in range(3):
                si = short_interest_on(x) if is_trading_day(x) else si
                if not si.empty:
                    break
                x -= pd.Timedelta(days=1)
        if si.empty:
            prev = None
        else:
            ratio = si["short"] / si["adv"].where(si["adv"] > 0)
            prev = -_by_ticker(ratio, tickers)
    return out


def monthly(frames, panel, market_cap: Optional[pd.DataFrame] = None, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    out = {}
    for n in POSITIONING_SIGNALS:
        hit = None if refresh else store.read_frame("research_signals", n)
        if hit is not None and hit.index.equals(panel.dates):
            out[n] = hit
    tickers = list(panel.prices.columns)
    if "insider_buyers" not in out or "insider_net" not in out:
        ins = insider_signals(insider_trades(), panel.dates, tickers, market_cap, issuer_ciks=issuer_ciks_for(tickers))
        for k, v in ins.items():
            store.write_frame("research_signals", k, v)
            out[k] = v
    if "short_volume" not in out:
        out["short_volume"] = short_volume_signal(panel.dates, tickers)
        store.write_frame("research_signals", "short_volume", out["short_volume"])
    if "days_to_cover" not in out:
        out["days_to_cover"] = days_to_cover_signal(panel.dates, tickers)
        store.write_frame("research_signals", "days_to_cover", out["days_to_cover"])
    return out


__all__ = ["POSITIONING_SIGNALS", "DESCRIPTIONS", "parse_form345", "insider_trades", "insider_signals",
           "short_volume_day", "short_volume_signal", "short_interest_on", "days_to_cover_signal", "monthly"]
