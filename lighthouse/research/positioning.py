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


def parse_form345(blob: bytes) -> pd.DataFrame:
    """Open-market purchases (P) and sales (S) of common stock from one data set: filed, issuer CIK and symbol, owner,
    relationship, shares, price, value (+ for purchases)."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    tr = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str,
                     usecols=["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                              "TRANS_ACQUIRED_DISP_CD", "TRANS_DATE"])
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"])]
    own = pd.read_csv(z.open("REPORTINGOWNER.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
    own = own.drop_duplicates("ACCESSION_NUMBER")
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    sh = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    px = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    sign = np.where(df["TRANS_CODE"] == "P", 1.0, -1.0)
    out = pd.DataFrame({"filed": pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
                        "issuer_cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce"),
                        "symbol": df["ISSUERTRADINGSYMBOL"].str.upper().str.strip().str.replace(".", "-", regex=False),
                        "owner": df["RPTOWNERCIK"], "relationship": df["RPTOWNER_RELATIONSHIP"], "code": df["TRANS_CODE"],
                        "shares": sh, "price": px, "value": sign * sh * px})
    return out.dropna(subset=["filed"]).reset_index(drop=True)


def insider_trades(refresh: bool = False) -> pd.DataFrame:
    """Every open-market insider purchase and sale since 2006 (cached per quarter; past quarters never change)."""
    parts = []
    for url in insider_files():
        key = url.rsplit("/", 1)[-1].replace("_form345.zip", "")
        hit = None if refresh else store.read_frame("insider", key)
        if hit is None:
            hit = parse_form345(_get(url, timeout=180).content)
            store.write_frame("insider", key, hit)
        parts.append(hit)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def insider_signals(trades: pd.DataFrame, dates: pd.DatetimeIndex, tickers: Iterable[str],
                    market_cap: Optional[pd.DataFrame] = None, window_days: int = 90) -> Dict[str, pd.DataFrame]:
    """insider_buyers and insider_net at each month end, by the issuer's trading symbol as filed (which also covers
    companies that no longer exist)."""
    tickers = list(tickers)
    t = trades[trades["symbol"].isin(tickers)]
    buyers = pd.DataFrame(0.0, index=dates, columns=tickers)
    net = pd.DataFrame(0.0, index=dates, columns=tickers)
    for d in dates:
        w = t[(t["filed"] > d - pd.Timedelta(days=window_days)) & (t["filed"] <= d)]
        if w.empty:
            continue
        b = w[w["code"] == "P"].groupby("symbol")["owner"].nunique()
        buyers.loc[d, b.index] = b.to_numpy(float)
        v = w.groupby("symbol")["value"].sum()
        net.loc[d, v.index] = v.to_numpy(float)
    if market_cap is not None:
        net = (net / market_cap.reindex(index=dates, columns=tickers)).where(market_cap.reindex(index=dates, columns=tickers) > 0)
    else:
        net = net * np.nan
    first = t["filed"].min() if len(t) else None
    if first is not None:
        buyers.loc[buyers.index < first] = np.nan
        net.loc[net.index < first] = np.nan
    return {"insider_buyers": buyers, "insider_net": net}


# --------------------------------------------------------------------------- #
# FINRA: short sale volume and short interest
# --------------------------------------------------------------------------- #
def short_volume_day(day) -> Optional[pd.DataFrame]:
    """Symbol, short volume and total volume for one session (all trade reporting facilities summed)."""
    d = pd.Timestamp(day)
    key = d.strftime("%Y%m%d")
    hit = store.read_frame("finra_shvol", key)
    if hit is not None:
        return hit
    names = ["CNMS"] if d >= pd.Timestamp("2018-08-01") else ["FNSQ", "FNYX", "FNQC", "FNRA"]
    parts = []
    for n in names:
        try:
            txt = _get(FINRA_SHVOL.format(name=n, date=key), timeout=60).text
        except DataUnavailable:
            continue
        df = pd.read_csv(io.StringIO(txt), sep="|", dtype={"Symbol": str})
        df = df[pd.to_numeric(df["Date"], errors="coerce").notna()]
        parts.append(df[["Symbol", "ShortVolume", "TotalVolume"]])
    if not parts:
        return None
    out = pd.concat(parts).groupby("Symbol")[["ShortVolume", "TotalVolume"]].sum()
    out.index = out.index.str.upper().str.replace(".", "-", regex=False)
    store.write_frame("finra_shvol", key, out)
    return out


def short_volume_signal(dates: pd.DatetimeIndex, tickers: Iterable[str], sessions: int = 5) -> pd.DataFrame:
    """Minus the short share of volume over the last `sessions` trading days of each month."""
    tickers = list(tickers)
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for d in dates:
        if d < pd.Timestamp("2009-08-03"):
            continue
        days, x = [], d
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
    cover (paged; cached)."""
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
                           index=df["symbolCode"].astype(str).str.upper().str.replace(".", "-", regex=False))
        out = out[~out.index.duplicated(keep="first")]
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
    month end: FINRA publishes short interest ~7 business days after settlement)."""
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
        prev = (-si["days_to_cover"]).reindex(tickers).to_numpy(float) if not si.empty else None
    return out


def monthly(frames, panel, market_cap: Optional[pd.DataFrame] = None, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    out = {}
    for n in POSITIONING_SIGNALS:
        hit = None if refresh else store.read_frame("research_signals", n)
        if hit is not None and hit.index.equals(panel.dates):
            out[n] = hit
    tickers = list(panel.prices.columns)
    if "insider_buyers" not in out or "insider_net" not in out:
        ins = insider_signals(insider_trades(), panel.dates, tickers, market_cap)
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
