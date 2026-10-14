"""
SEC EDGAR, the free and official source of US company filings: who files (ticker -> CIK), what they filed and when,
and every XBRL number in their 10-K and 10-Q reports with its filing date.

Endpoints (all free; the SEC asks for a User-Agent with a contact e-mail and at most 10 requests per second, which
`lighthouse.http` enforces):
  www.sec.gov/files/company_tickers_exchange.json     today's tickers, CIKs and exchanges
  data.sec.gov/submissions/CIK##########.json         company profile (SIC code, fiscal year end) and every filing
  data.sec.gov/api/xbrl/companyfacts/CIK##########.json   every XBRL fact, with the filing (accession) and its date
  www.sec.gov/Archives/edgar/full-index/YYYY/QTRn/form.idx  every filing of a quarter by form type (all filers,
                                                      including companies that no longer exist)
  www.sec.gov/files/dera/data/financial-statement-data-sets/YYYYqN.zip   the bulk quarterly statement data sets

XBRL coverage starts in 2009 (large filers) and 2011 (everyone). Before that EDGAR has filings but no machine-readable
numbers, so Lighthouse treats fundamentals before a company's first XBRL filing as unknown.

Everything downloaded is filtered to what Lighthouse uses and cached as Parquet under data/cache/edgar/.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import store
from .data import DataUnavailable, SAMPLE_DIR, offline_mode

TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORM_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.idx"
FSDS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{qtr}.zip"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{doc}"
FRESH_HOURS = 24.0                         # filings arrive daily; a day-old copy is current enough for research

# The delisting and deregistration forms: Form 25 (25-NSE when the exchange files it) removes a security from listing;
# Form 15 ends the duty to file reports (15-12B / 15-12G / 15-15D).
DELISTING_FORMS = ("25", "25-NSE", "15-12B", "15-12G", "15-15D")
ANNUAL_FORMS = ("10-K", "10-K/A", "10-KT", "20-F", "20-F/A", "40-F", "40-F/A")
QUARTERLY_FORMS = ("10-Q", "10-Q/A", "10-QT")
PERIODIC_FORMS = ANNUAL_FORMS + QUARTERLY_FORMS


# Successor issuers. A holding-company reorganisation gives the listed company a new CIK, and today's ticker file then
# points at a registrant with a few months of history; its statements before the change sit under the old CIK. EDGAR
# does not link the two, so the known cases are listed here (new CIK -> predecessor CIKs, oldest last).
PREDECESSOR_CIKS: Dict[int, List[int]] = {
    2115436: [34088],          # ExxonMobil Holdings Corp (2026) <- Exxon Mobil Corp
}


def normalize_ticker(ticker: str) -> str:
    """One spelling for share classes: BRK.B, BRK/B and BRK-B are all BRK-B (the SEC's and Yahoo's form)."""
    return str(ticker).upper().strip().replace(".", "-").replace("/", "-")


def _get(url: str, **kw):
    from .http import fetch
    return fetch(url, **kw)


# --------------------------------------------------------------------------- #
# Who files
# --------------------------------------------------------------------------- #
def ticker_table(refresh: bool = False) -> pd.DataFrame:
    """Today's EDGAR ticker list: one row per ticker with cik, name, ticker and exchange. Companies that no longer
    file have no ticker here; `filing_index` and `delistings` cover them."""
    cached = store.read_frame("edgar", "company_tickers")
    fresh = cached is not None and (offline_mode() or store.age_hours(store.path_for("edgar", "company_tickers")) < FRESH_HOURS)
    if fresh and not refresh:
        return cached
    try:
        payload = _get(TICKERS_URL).json()
    except DataUnavailable:
        if cached is not None:
            return cached
        raise
    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["cik"] = df["cik"].astype(int)
    store.write_frame("edgar", "company_tickers", df)
    return df


def cik_for(ticker: str) -> Optional[int]:
    """The CIK of a ticker that files with the SEC today, or None."""
    t = normalize_ticker(ticker)
    try:
        tab = ticker_table()
    except DataUnavailable:
        return None
    hit = tab.loc[tab["ticker"] == t, "cik"]
    return int(hit.iloc[0]) if len(hit) else None


# --------------------------------------------------------------------------- #
# What they filed
# --------------------------------------------------------------------------- #
FILING_COLUMNS = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "items",
                  "primaryDocument", "isXBRL"]


def _filings_frame(block: dict) -> pd.DataFrame:
    cols = {c: block.get(c, []) for c in FILING_COLUMNS}
    n = max((len(v) for v in cols.values()), default=0)
    cols = {c: (v if len(v) == n else [None] * n) for c, v in cols.items()}
    return pd.DataFrame(cols)


def parse_submissions(payload: dict, older_pages: Sequence[dict] = ()) -> tuple:
    """(profile dict, filings frame) from a submissions JSON and its older pages."""
    profile = {k: payload.get(k) for k in ("cik", "name", "sic", "sicDescription", "fiscalYearEnd", "entityType",
                                           "stateOfIncorporation", "tickers", "exchanges", "category")}
    profile["cik"] = int(profile["cik"]) if profile.get("cik") else None
    profile["former_names"] = [f.get("name") for f in payload.get("formerNames", [])]
    frames = [_filings_frame(payload.get("filings", {}).get("recent", {}))]
    frames += [_filings_frame(p) for p in older_pages]
    f = pd.concat(frames, ignore_index=True)
    if f.empty:
        return profile, f
    f = f.rename(columns={"accessionNumber": "accn", "filingDate": "filed", "reportDate": "report_date",
                          "acceptanceDateTime": "accepted", "primaryDocument": "doc", "isXBRL": "is_xbrl"})
    f["filed"] = pd.to_datetime(f["filed"], errors="coerce")
    f["report_date"] = pd.to_datetime(f["report_date"], errors="coerce")
    f["accepted"] = pd.to_datetime(f["accepted"], errors="coerce", utc=True)
    f["items"] = f["items"].fillna("").astype(str)
    f = f.drop_duplicates("accn").sort_values("filed").reset_index(drop=True)
    return profile, f


def submissions(cik: int, refresh: bool = False) -> tuple:
    """(profile, filings) for one company: SIC code, fiscal year end, and every filing with its date, form and 8-K
    items. Cached for a day (any age offline)."""
    key = f"{int(cik):010d}"
    cached = store.read_frame("edgar_filings", key)
    prof = store.read_json("edgar_profile", key)
    fresh = cached is not None and prof is not None and (
        offline_mode() or store.age_hours(store.path_for("edgar_filings", key)) < FRESH_HOURS)
    if fresh and not refresh:
        return prof, cached
    try:
        payload = _get(SUBMISSIONS_URL.format(cik=int(cik))).json()
        pages = [_get(SUBMISSIONS_PAGE_URL.format(name=p["name"])).json() for p in payload.get("filings", {}).get("files", [])]
    except DataUnavailable:
        if cached is not None and prof is not None:
            return prof, cached
        raise
    prof, filings = parse_submissions(payload, pages)
    store.write_frame("edgar_filings", key, filings)
    store.write_json("edgar_profile", key, prof)
    return prof, filings


SAMPLE_FILING_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "8-K", "8-K/A", "DEF 14A", "S-1", "S-3", "S-4", "424B2",
                       "SC 13D", "SC 13G", "25", "25-NSE", "15-12B", "NT 10-K", "NT 10-Q")


def sample_filings_paths(ticker: str) -> tuple:
    base = SAMPLE_DIR / "edgar"
    t = normalize_ticker(ticker)
    return base / f"{t}_filings.parquet", base / f"{t}_profile.json"


def company_filings(ticker: str, source: str = "auto", refresh: bool = False) -> tuple:
    """(profile, filings) for a ticker: EDGAR submissions (cached), or the sample pack's recorded copy when the source is
    the sample pack or EDGAR cannot be reached. Raises DataUnavailable when neither has the company."""
    import json
    fp, pp = sample_filings_paths(ticker)

    def sample():
        if fp.exists() and pp.exists():
            f = pd.read_parquet(fp)
            for c in ("filed", "report_date"):
                f[c] = pd.to_datetime(f[c], errors="coerce")
            f["items"] = f["items"].fillna("").astype(str)
            return json.loads(pp.read_text()), f
        raise DataUnavailable(f"{ticker}: no EDGAR filing list in the sample pack")
    if source == "sample":
        return sample()
    cik = cik_for(ticker)
    if cik is None:
        return sample()
    try:
        return submissions(cik, refresh=refresh)
    except DataUnavailable:
        return sample()


def earnings_release_dates(filings: pd.DataFrame) -> pd.Series:
    """Dates of the 8-K reports under Item 2.02 ("Results of Operations and Financial Condition"): the earnings
    announcements. Free data has no announcement calendar; these filings are the public record of it."""
    if filings is None or filings.empty:
        return pd.Series(dtype="datetime64[ns]")
    m = filings["form"].isin(["8-K", "8-K/A"]) & filings["items"].str.contains(r"\b2\.02\b", regex=True)
    return filings.loc[m, "filed"].dropna().sort_values().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The numbers: XBRL company facts
# --------------------------------------------------------------------------- #
FACT_COLUMNS = ["taxonomy", "concept", "unit", "start", "end", "val", "accn", "fy", "fp", "form", "filed", "frame"]


def flatten_company_facts(payload: dict, concepts: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Every fact as one row (taxonomy, concept, unit, start, end, val, accn, fy, fp, form, filed, frame), keeping only
    `concepts` ("us-gaap:Revenues" or bare "Revenues") when given."""
    want = None
    if concepts is not None:
        want = {c.split(":")[-1] for c in concepts}
    rows: List[tuple] = []
    for tax, block in (payload.get("facts") or {}).items():
        for concept, spec in block.items():
            if want is not None and concept not in want:
                continue
            for unit, facts in (spec.get("units") or {}).items():
                for f in facts:
                    rows.append((tax, concept, unit, f.get("start"), f.get("end"), f.get("val"), f.get("accn"),
                                 f.get("fy"), f.get("fp"), f.get("form"), f.get("filed"), f.get("frame")))
    df = pd.DataFrame(rows, columns=FACT_COLUMNS)
    return _typed_facts(df)


def _typed_facts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)
    df = df.copy()
    for c in FACT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce")
    for c in ("taxonomy", "concept", "unit", "accn", "fp", "form", "frame"):
        df[c] = df[c].astype("string")
    return df.dropna(subset=["end", "val", "filed"]).reset_index(drop=True)


def sample_facts_path(ticker: str):
    return SAMPLE_DIR / "edgar" / f"{normalize_ticker(ticker)}.parquet"


def company_facts(cik: Optional[int] = None, ticker: Optional[str] = None, concepts: Optional[Iterable[str]] = None,
                  refresh: bool = False, source: str = "auto") -> pd.DataFrame:
    """The company's XBRL facts as a long table (see `flatten_company_facts`), filtered to the concepts Lighthouse uses
    (`pit.CONCEPTS`) unless `concepts` says otherwise. Order: fresh cache -> EDGAR -> stale cache -> sample pack."""
    from .pit import CONCEPTS
    concepts = list(concepts) if concepts is not None else CONCEPTS
    if source == "sample" or (offline_mode() and cik is None and ticker is not None):
        p = sample_facts_path(ticker or "")
        if ticker and p.exists():
            return _typed_facts(pd.read_parquet(p))
        if source == "sample":
            raise DataUnavailable(f"{ticker}: no EDGAR facts in the sample pack")
    if cik is None:
        cik = cik_for(ticker) if ticker else None
        if cik is None:
            p = sample_facts_path(ticker or "")
            if ticker and p.exists():
                return _typed_facts(pd.read_parquet(p))
            raise DataUnavailable(f"{ticker}: no CIK (not a current SEC filer, or EDGAR unreachable)")
    parts = [_facts_for_cik(int(cik), concepts, refresh, ticker)]
    for prev in PREDECESSOR_CIKS.get(int(cik), []):
        try:
            parts.append(_facts_for_cik(prev, concepts, refresh, None))
        except DataUnavailable:
            continue
    return _typed_facts(pd.concat(parts, ignore_index=True)) if len(parts) > 1 else parts[0]


def _facts_for_cik(cik: int, concepts, refresh: bool, ticker: Optional[str]) -> pd.DataFrame:
    key = f"{int(cik):010d}"
    path = store.path_for("edgar_facts", key)
    cached = store.read_frame("edgar_facts", key)
    if cached is not None and not refresh and (offline_mode() or store.age_hours(path) < FRESH_HOURS):
        return _typed_facts(cached)
    try:
        payload = _get(FACTS_URL.format(cik=int(cik)), timeout=60).json()
    except DataUnavailable:
        if cached is not None:
            return _typed_facts(cached)
        p = sample_facts_path(ticker or "")
        if ticker and p.exists():
            return _typed_facts(pd.read_parquet(p))
        raise
    df = flatten_company_facts(payload, concepts)
    store.write_frame("edgar_facts", key, df)
    return df


# --------------------------------------------------------------------------- #
# Every filer, including the dead: the quarterly form index
# --------------------------------------------------------------------------- #
def parse_form_index_lines(lines: Iterable[str], forms: Sequence[str]) -> pd.DataFrame:
    """Rows of a form.idx (fixed-width: form type, company, CIK, date filed, file name) for the given form types.
    The file is sorted by form type, so reading stops once every wanted form has been passed."""
    wanted = set(forms)
    last = max(wanted)
    rows, header_done, started = [], False, False
    for raw in lines:
        s = raw.decode("latin-1") if isinstance(raw, bytes) else raw
        if not header_done:
            header_done = s.startswith("-----")
            continue
        if not s.strip():
            continue
        form = s[:17].strip()
        parts = s[17:].rstrip().rsplit(None, 3)
        if len(parts) < 4:
            continue
        name, cik, filed, fname = parts
        if form in wanted:
            started = True
            rows.append((form, name.strip(), int(cik), filed, fname))
        elif started and form > last and not any(form.startswith(w) for w in wanted):
            break
    df = pd.DataFrame(rows, columns=["form", "name", "cik", "filed", "file"])
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["accn"] = df["file"].str.extract(r"(\d{10}-\d{2}-\d{6})", expand=False)
    return df


def filing_index(year: int, qtr: int, forms: Sequence[str] = DELISTING_FORMS, refresh: bool = False) -> pd.DataFrame:
    """One quarter's filings of the given form types by every EDGAR filer (dead companies included). Past quarters
    never change, so they are cached for good; the current quarter for a day."""
    key = f"{year}Q{qtr}_" + "_".join(sorted(f.replace("/", "") for f in forms))
    path = store.path_for("edgar_index", key)
    cached = store.read_frame("edgar_index", key)
    now = pd.Timestamp.now()
    current = (year, qtr) >= (now.year, (now.month - 1) // 3 + 1)
    if cached is not None and not refresh and (offline_mode() or not current or store.age_hours(path) < FRESH_HOURS):
        return cached
    resp = _get(FORM_INDEX_URL.format(year=year, qtr=qtr), stream=True, timeout=120)
    try:
        df = parse_form_index_lines(resp.iter_lines(), forms)
    finally:
        resp.close()
    store.write_frame("edgar_index", key, df)
    return df


def delistings(start_year: int = 2001, end_year: Optional[int] = None, refresh: bool = False) -> pd.DataFrame:
    """Every Form 25 / 25-NSE (removal from listing) and Form 15 (deregistration) filed in the window, one row per
    subject company and form: cik, name, form, filed, accn. A Form 25-NSE appears in the index under both the
    exchange and the company; the exchanges (filers of many such forms) are dropped."""
    end_year = end_year or datetime.now().year
    now = pd.Timestamp.now()
    parts = []
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            if (y, q) > (now.year, (now.month - 1) // 3 + 1):
                break
            parts.append(filing_index(y, q, DELISTING_FORMS, refresh=refresh))
    if not parts:
        return pd.DataFrame(columns=["cik", "name", "form", "filed", "accn"])
    df = pd.concat(parts, ignore_index=True)
    # exchanges file 25-NSE for many issuers; a CIK with dozens of them in one quarter is an exchange, not a company
    per_q = df[df["form"].str.startswith("25")].groupby([df["filed"].dt.to_period("Q"), "cik"]).size()
    exchanges = set(per_q[per_q > 20].index.get_level_values(1))
    df = df[~df["cik"].isin(exchanges)]
    return df.drop_duplicates(["cik", "form", "accn"]).sort_values("filed").reset_index(drop=True)[
        ["cik", "name", "form", "filed", "accn"]]


# --------------------------------------------------------------------------- #
# Bulk: the quarterly Financial Statement Data Sets
# --------------------------------------------------------------------------- #
def parse_fsds_zip(blob, ciks: Optional[Iterable[int]] = None, tags: Optional[Iterable[str]] = None,
                   chunksize: int = 500_000) -> pd.DataFrame:
    """Facts from one quarterly Financial Statement Data Set (a zip of tab-separated sub.txt / num.txt), in the same
    long format as `flatten_company_facts`, for the given companies and XBRL tags. num.txt is read in chunks and
    filtered as it streams, so a 100 MB+ file never sits in memory whole. Only the company's own, undimensioned
    values are kept (no co-registrant, no segment breakdown)."""
    zf = zipfile.ZipFile(io.BytesIO(blob) if isinstance(blob, (bytes, bytearray)) else blob)
    sub = pd.read_csv(zf.open("sub.txt"), sep="\t", dtype=str, usecols=lambda c: c in {
        "adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed"})
    sub["cik"] = sub["cik"].astype(int)
    if ciks is not None:
        sub = sub[sub["cik"].isin({int(c) for c in ciks})]
    want_tags = None if tags is None else {t.split(":")[-1] for t in tags}
    keep = set(sub["adsh"])
    parts = []
    for chunk in pd.read_csv(zf.open("num.txt"), sep="\t", dtype=str, chunksize=chunksize,
                             usecols=lambda c: c in {"adsh", "tag", "version", "coreg", "ddate", "qtrs", "uom",
                                                     "segments", "value"}):
        m = chunk["adsh"].isin(keep)
        if want_tags is not None:
            m &= chunk["tag"].isin(want_tags)
        if "coreg" in chunk:
            m &= chunk["coreg"].isna()
        if "segments" in chunk:
            m &= chunk["segments"].isna()
        if m.any():
            parts.append(chunk[m])
    if not parts:
        return pd.DataFrame(columns=FACT_COLUMNS)
    num = pd.concat(parts, ignore_index=True).merge(sub, on="adsh", how="left")
    end = pd.to_datetime(num["ddate"], format="%Y%m%d", errors="coerce")
    q = pd.to_numeric(num["qtrs"], errors="coerce").fillna(0).astype(int)
    start = pd.Series(pd.NaT, index=num.index, dtype="datetime64[ns]")
    dur = q > 0
    # a duration of q quarters ends on `ddate` and starts the day after the end q*3 months earlier (a month-end stays a
    # month-end: June 30 minus three months is March 31, so the quarter starts April 1)
    def _start(e: pd.Timestamp, k: int) -> pd.Timestamp:
        s0 = e - pd.DateOffset(months=3 * k)
        if e.is_month_end:
            s0 = s0 + pd.offsets.MonthEnd(0)
        return s0 + pd.Timedelta(days=1)
    start[dur] = [_start(e, k) for e, k in zip(end[dur], q[dur])]
    taxonomy = num["version"].fillna("").str.extract(r"^(us-gaap|dei|ifrs-full|srt)", expand=False).fillna("custom")
    out = pd.DataFrame({
        "taxonomy": taxonomy, "concept": num["tag"], "unit": num["uom"], "start": start, "end": end,
        "val": num["value"], "accn": num["adsh"], "fy": num["fy"], "fp": num["fp"], "form": num["form"],
        "filed": pd.to_datetime(num["filed"], format="%Y%m%d", errors="coerce"), "frame": None,
        "cik": num["cik"].astype(int)})
    return _typed_facts(out)


def bulk_quarter(year: int, qtr: int, ciks: Optional[Iterable[int]] = None, tags: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Download one quarterly data set into memory (no temporary file), filter it, and cache the result."""
    key = f"{year}q{qtr}"
    cached = store.read_frame("edgar_fsds", key)
    if cached is None:
        from .pit import CONCEPTS
        blob = _get(FSDS_URL.format(year=year, qtr=qtr), timeout=300).content
        cached = parse_fsds_zip(blob, None, tags or CONCEPTS)
        store.write_frame("edgar_fsds", key, cached)
    if ciks is not None:
        cached = cached[cached["cik"].isin({int(c) for c in ciks})]
    return cached


def document_url(cik: int, accn: str, doc: str) -> str:
    return ARCHIVE_URL.format(cik=int(cik), accn_nodash=str(accn).replace("-", ""), doc=doc)


def fetch_document(cik: int, accn: str, doc: str) -> str:
    """The text of one filed document (HTML or text), cached by accession number."""
    key = f"{int(cik)}_{str(accn).replace('-', '')}_{doc}"
    hit = store.read_json("edgar_docs", key)
    if hit is not None:
        return hit
    txt = _get(document_url(cik, accn, doc), timeout=60).text
    store.write_json("edgar_docs", key, txt)
    return txt


def filing_documents(cik: int, accn: str) -> pd.DataFrame:
    """The documents inside one filing (name, type, description), from the filing's index JSON."""
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{str(accn).replace('-', '')}/index.json"
    key = f"{int(cik)}_{str(accn).replace('-', '')}_index"
    hit = store.read_json("edgar_docs", key)
    if hit is None:
        hit = _get(url).json()
        store.write_json("edgar_docs", key, hit)
    items = (hit.get("directory") or {}).get("item") or []
    return pd.DataFrame(items)


__all__ = ["ticker_table", "cik_for", "submissions", "company_facts", "flatten_company_facts", "filing_index",
           "delistings", "parse_fsds_zip", "bulk_quarter", "earnings_release_dates", "normalize_ticker",
           "fetch_document", "filing_documents", "document_url", "company_filings", "SAMPLE_FILING_FORMS"]
