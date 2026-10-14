"""
Institutional ownership breadth from the SEC's Form 13F data sets (2013 Q2 onward; earlier 13Fs are not published
in structured form).

inst_breadth   the change in the share of 13F filers holding the stock over the quarter (Chen, Hong & Stein 2002,
               "Breadth of ownership and stock returns"): among the managers who filed for both quarters, (holders
               this quarter - holders last quarter) / managers. Higher = buy.

Point in time  a quarter's holdings count only from filings made within 45 days of the quarter end (the 13F deadline),
               and the signal is known from day 46. Amendments (13F-HR/A) and option positions are left out.
Tickers        13F holdings are identified by CUSIP. The SEC's fails-to-deliver files carry CUSIP and ticker side by
               side; each quarter uses the pairs seen in the files nearest its end (within two years), so a ticker that
               was reused by another company later does not borrow its CUSIP. A stock that never appears in those
               files (few fails) is missing: the coverage is reported with the results.
Volume         each quarter's data set is a 2-90 MB zip, read in memory and reduced to holder counts; nothing large
               is written to disk (the counts are cached in data/cache/research).
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import store

SEC = "https://www.sec.gov"
F13_PAGE = SEC + "/data-research/sec-markets-data/form-13f-data-sets"
FTD_PAGE = SEC + "/data-research/sec-markets-data/fails-deliver-data"
DEADLINE_DAYS = 45
DESCRIPTIONS = {"inst_breadth": "change in breadth of 13F institutional ownership (Chen, Hong & Stein 2002)"}


def _norm_symbol(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _links(page_html: str) -> List[str]:
    return re.findall(r'href="([^"]+\.zip)"', page_html)


def list_13f_files(page_html: Optional[str] = None) -> List[str]:
    from ..http import fetch
    page_html = page_html if page_html is not None else fetch(F13_PAGE).text
    return [SEC + u if u.startswith("/") else u for u in _links(page_html) if "form13f" in u.lower()]


def list_ftd_files(page_html: Optional[str] = None) -> pd.DataFrame:
    """Fails-to-deliver files with the half-month they cover (cnsfailsYYYYMM[a|b].zip)."""
    from ..http import fetch
    page_html = page_html if page_html is not None else fetch(FTD_PAGE).text
    rows = []
    for u in _links(page_html):
        m = re.search(r"cnsfails(\d{4})(\d{2})([ab])\.zip", u)
        if m:
            y, mo, half = int(m.group(1)), int(m.group(2)), m.group(3)
            d = pd.Timestamp(year=y, month=mo, day=1 if half == "a" else 16)
            rows.append({"date": d, "url": SEC + u if u.startswith("/") else u})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame(columns=["date", "url"])


def parse_ftd_zip(blob: bytes) -> pd.DataFrame:
    """(cusip, symbol) pairs in one fails-to-deliver file (pipe-delimited text)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_csv(fh, sep="|", dtype=str, usecols=[1, 2], names=["cusip", "symbol"], header=0,
                             encoding="latin-1", on_bad_lines="skip")
    df = df.dropna()
    df["cusip"] = df["cusip"].str.strip().str.upper()
    df["symbol"] = df["symbol"].map(_norm_symbol)
    return df[(df["cusip"].str.len() == 9) & (df["symbol"] != "")].drop_duplicates()


def parse_13f_zip(blob: bytes, cusips: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(holdings, filers) from one 13F data set: holdings = (period, cik, cusip) for on-time original 13F-HR filings
    and the CUSIPs asked for; filers = (period, cik) for every on-time original filing."""
    want = set(c.upper() for c in cusips)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = {n.split("/")[-1].upper(): n for n in z.namelist()}
        with z.open(names["SUBMISSION.TSV"]) as fh:
            sub = pd.read_csv(fh, sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE",
                                                                  "CIK", "PERIODOFREPORT"])
        sub = sub[sub["SUBMISSIONTYPE"].str.strip() == "13F-HR"].copy()
        sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
        sub["period"] = pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
        sub = sub.dropna(subset=["filed", "period"])
        sub = sub[sub["filed"] <= sub["period"] + pd.Timedelta(days=DEADLINE_DAYS)]
        sub["cik"] = pd.to_numeric(sub["CIK"], errors="coerce").astype("Int64")
        sub = sub.sort_values("filed").drop_duplicates(["cik", "period"], keep="first")
        acc = dict(zip(sub["ACCESSION_NUMBER"], zip(sub["period"], sub["cik"])))
        parts = []
        with z.open(names["INFOTABLE.TSV"]) as fh:
            for chunk in pd.read_csv(fh, sep="\t", dtype=str, chunksize=1_000_000, on_bad_lines="skip",
                                     usecols=lambda c: c in {"ACCESSION_NUMBER", "CUSIP", "PUTCALL", "SSHPRNAMTTYPE"}):
                c = chunk[chunk["ACCESSION_NUMBER"].isin(acc.keys())]
                c = c[c["CUSIP"].str.upper().isin(want)]
                if "PUTCALL" in c:
                    c = c[c["PUTCALL"].isna() | (c["PUTCALL"].str.strip() == "")]
                if "SSHPRNAMTTYPE" in c:
                    c = c[c["SSHPRNAMTTYPE"].str.strip().str.upper() == "SH"]
                if len(c):
                    pc = [acc[a] for a in c["ACCESSION_NUMBER"]]
                    parts.append(pd.DataFrame({"period": [p for p, _ in pc], "cik": [k for _, k in pc],
                                               "cusip": c["CUSIP"].str.upper().to_numpy()}))
    hold = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else \
        pd.DataFrame(columns=["period", "cik", "cusip"])
    filers = sub[["period", "cik"]].drop_duplicates().reset_index(drop=True)
    return hold, filers


def breadth_change(holdings: pd.DataFrame, filers: pd.DataFrame) -> pd.DataFrame:
    """period x cusip: the change in breadth over the quarter ending at `period`, among managers filing both quarters."""
    periods = sorted(pd.to_datetime(filers["period"].unique()))
    by_p = {p: set(filers.loc[filers["period"] == p, "cik"]) for p in periods}
    out = {}
    for prev, cur in zip(periods[:-1], periods[1:]):
        if (cur - prev).days > 100:                    # a missing quarter: no change can be measured
            continue
        common = by_p[prev] & by_p[cur]
        if not common:
            continue
        h0 = holdings[(holdings["period"] == prev) & holdings["cik"].isin(common)].groupby("cusip")["cik"].nunique()
        h1 = holdings[(holdings["period"] == cur) & holdings["cik"].isin(common)].groupby("cusip")["cik"].nunique()
        idx = h0.index.union(h1.index)
        out[cur] = (h1.reindex(idx).fillna(0) - h0.reindex(idx).fillna(0)) / len(common)
    return pd.DataFrame(out).T.sort_index() if out else pd.DataFrame()


def cusip_for_quarters(ftd: Dict[pd.Timestamp, pd.DataFrame], tickers: Iterable[str],
                       quarters: Iterable[pd.Timestamp], window_days: int = 730) -> Dict[Tuple[str, pd.Timestamp], str]:
    """(ticker, quarter end) -> CUSIP from the fails-to-deliver pairs nearest the quarter (within `window_days`)."""
    obs = []
    for d, df in ftd.items():
        obs.append(df.assign(date=pd.Timestamp(d)))
    if not obs:
        return {}
    allp = pd.concat(obs, ignore_index=True)
    want = {_norm_symbol(t): t for t in tickers}
    allp = allp[allp["symbol"].isin(want)]
    out = {}
    for sym, g in allp.groupby("symbol"):
        g = g.sort_values("date")
        for q in quarters:
            dist = (g["date"] - q).abs()
            ok = dist <= pd.Timedelta(days=window_days)
            if ok.any():
                out[(want[sym], q)] = g.loc[dist[ok].idxmin(), "cusip"]
    return out


def monthly(frames: Dict[str, pd.DataFrame], panel, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """{'inst_breadth': month-end x ticker}."""
    from ..http import fetch
    from .event_signals import hold_daily
    from .harness import sample_at_month_ends
    hit = None if refresh else store.read_frame("research_signals", "inst_breadth")
    if hit is not None and hit.index.equals(panel.dates):
        return {"inst_breadth": hit}
    tickers = list(frames)
    # CUSIPs: one fails-to-deliver file per quarter end (the first half of the month after it), 2013 on
    ftd_files = list_ftd_files()
    ftd_files = ftd_files[ftd_files["date"] >= "2013-01-01"]
    pick = ftd_files.groupby(ftd_files["date"].dt.to_period("Q")).head(1)
    ftd = {}
    for _, r in pick.iterrows():
        key = f"ftd_{r['date']:%Y%m%d}"
        df = store.read_frame("research_13f", key)
        if df is None:
            df = parse_ftd_zip(fetch(r["url"], timeout=120).content)
            store.write_frame("research_13f", key, df)
        ftd[r["date"]] = df
    quarters = sorted({(pd.Timestamp(d) + pd.offsets.QuarterEnd(0)).normalize() for d in panel.dates
                       if d >= pd.Timestamp("2013-03-31")})
    cmap = cusip_for_quarters(ftd, tickers, quarters)
    cusips = set(cmap.values())
    holds, filers = [], []
    for url in list_13f_files():
        key = "f13_" + url.rsplit("/", 1)[-1].replace(".zip", "")
        h = store.read_frame("research_13f", key + "_hold")
        f = store.read_frame("research_13f", key + "_filers")
        if h is None or f is None or refresh:
            h, f = parse_13f_zip(fetch(url, timeout=600).content, cusips)
            store.write_frame("research_13f", key + "_hold", h)
            store.write_frame("research_13f", key + "_filers", f)
        holds.append(h)
        filers.append(f)
    hold = pd.concat(holds, ignore_index=True).drop_duplicates()
    fil = pd.concat(filers, ignore_index=True).drop_duplicates()
    hold["period"], fil["period"] = pd.to_datetime(hold["period"]), pd.to_datetime(fil["period"])
    bc = breadth_change(hold, fil)
    daily = {}
    for t in tickers:
        vals = {}
        for q in bc.index:
            c = cmap.get((t, q))
            if c is not None and c in bc.columns and np.isfinite(bc.at[q, c]):
                vals[q + pd.Timedelta(days=DEADLINE_DAYS + 1)] = float(bc.at[q, c])
        if vals:
            daily[t] = hold_daily(pd.Series(vals).sort_index(), frames[t].index, 100)
    m = sample_at_month_ends(daily, panel.dates)
    store.write_frame("research_signals", "inst_breadth", m)
    store.write_json("research", "inst_breadth_coverage", {"tickers_with_cusip": len({t for t, _ in cmap}),
                                                           "tickers": len(tickers), "quarters": len(bc)})
    return {"inst_breadth": m}


__all__ = ["list_13f_files", "list_ftd_files", "parse_ftd_zip", "parse_13f_zip", "breadth_change",
           "cusip_for_quarters", "monthly", "DESCRIPTIONS"]
