"""
The Kenneth R. French Data Library (free): factor returns, industry portfolios and SIC-code industry definitions.

Used to ask whether a result is new or just exposure to known factors (market, size, value, profitability, investment,
momentum), for industry momentum, and for the market risk premium in valuations. Files are zipped CSVs with several
sections (monthly, annual, ...); values are percent; -99.99 / -999 mean missing.

The library's terms ask users to cite it; data is built from CRSP and updated monthly (a month or two behind).
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import store
from .data import DataUnavailable, offline_mode

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff5_monthly": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_monthly": "F-F_Momentum_Factor_CSV.zip",
    "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "ind49": "49_Industry_Portfolios_CSV.zip",
    "ind12": "12_Industry_Portfolios_CSV.zip",
    "sic49": "Siccodes49.zip",
    "sic12": "Siccodes12.zip",
}
FRESH_HOURS = 24.0 * 7
CITATION = "Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"


def _download_text(name: str) -> str:
    from .http import fetch
    blob = fetch(BASE_URL + FILES[name], timeout=90).content
    z = zipfile.ZipFile(io.BytesIO(blob))
    return z.read(z.namelist()[0]).decode("latin-1")


def _cached_text(name: str, refresh: bool = False) -> str:
    """The file's text, cached for a week (any age offline)."""
    hit = store.read_json("french", name)
    path = store.path_for("french", name, "json")
    if hit is not None and not refresh and (offline_mode() or store.age_hours(path) < FRESH_HOURS):
        return hit
    try:
        txt = _download_text(name)
    except DataUnavailable:
        if hit is not None:
            return hit
        raise
    store.write_json("french", name, txt)
    return txt


_DATE = re.compile(r"^\s*(\d{4}|\d{6}|\d{8})\s*,")


def parse_sections(text: str) -> Dict[str, pd.DataFrame]:
    """Every table in a library CSV, keyed by the title line above its header (the first table is also under "main").
    Dates YYYYMM -> month end, YYYYMMDD -> that day, YYYY -> year end; values in percent are returned as fractions."""
    lines = text.splitlines()
    out: Dict[str, pd.DataFrame] = {}
    title, header, rows = "main", None, []

    def flush():
        nonlocal header, rows
        if header is not None and rows:
            df = pd.DataFrame(rows, columns=["date"] + header).set_index("date")
            df = df.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999.0], np.nan) / 100.0
            key = "main" if not out else title          # the first table follows the file's preamble
            k = 2
            while key in out:
                key, k = f"{title} ({k})", k + 1
            if key == "main" and title != "main" and title not in out:
                out[title] = df                          # ... and is also found by its own title when it has one
            out[key] = df
        header, rows = None, []

    for line in lines:
        if line.startswith(",") and not _DATE.match(line):
            flush()
            header = [c.strip() for c in line.split(",")[1:]]
            continue
        m = _DATE.match(line)
        if m and header is not None:
            parts = [p.strip() for p in line.split(",")]
            rows.append([_date(parts[0])] + parts[1:1 + len(header)])
            continue
        if line.strip():
            flush()
            title = line.strip()
    flush()
    return out


def _date(tok: str) -> pd.Timestamp:
    if len(tok) == 8:
        return pd.Timestamp(tok)
    if len(tok) == 6:
        return pd.Timestamp(f"{tok[:4]}-{tok[4:]}-01") + pd.offsets.MonthEnd(0)
    return pd.Timestamp(f"{tok}-12-31")


def load_factors(freq: str = "monthly", refresh: bool = False) -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RMW, CMA, RF and Mom (momentum: prior 12-2 month winners minus losers), as fractions,
    monthly (month-end index) or daily."""
    f5 = parse_sections(_cached_text(f"ff5_{freq}", refresh))
    mom = parse_sections(_cached_text(f"mom_{freq}", refresh))
    base = next(iter(f5.values()))
    m = next(iter(mom.values()))
    m.columns = [c.strip() for c in m.columns]
    out = base.join(m[["Mom"]], how="left")
    out.index.name = "date"
    return out


def load_industries(n: int = 49, weighting: str = "value", refresh: bool = False) -> pd.DataFrame:
    """Monthly returns of the n (12 or 49) industry portfolios, value- or equal-weighted."""
    secs = parse_sections(_cached_text(f"ind{n}", refresh))
    want = "Value Weighted Returns -- Monthly" if weighting == "value" else "Equal Weighted Returns -- Monthly"
    key = next(k for k in secs if want in k)
    df = secs[key]
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_sic_codes(text: str) -> List[Tuple[int, int, str]]:
    """(SIC from, SIC to, industry short name) from a Siccodes file."""
    out, current = [], None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+", line)
        if m and not re.match(r"^\s*\d{4}-\d{4}", line):
            current = m.group(2)
            continue
        r = re.match(r"^\s*(\d{4})-(\d{4})", line)
        if r and current:
            out.append((int(r.group(1)), int(r.group(2)), current))
    return out


def sic_industry(sic, n: int = 49) -> Optional[str]:
    """The Fama-French industry of a 4-digit SIC code (EDGAR's `sic`), or None ("Other" when unlisted, as French does)."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return None
    for lo, hi, name in _sic_table(n):
        if lo <= code <= hi:
            return name
    return "Other"


_SIC_TABLES: Dict[int, List[Tuple[int, int, str]]] = {}


def _sic_table(n: int) -> List[Tuple[int, int, str]]:
    if n not in _SIC_TABLES:
        _SIC_TABLES[n] = parse_sic_codes(_cached_text(f"sic{n}"))
    return _SIC_TABLES[n]


def market_premium(start: str = "1963-07-01", end=None) -> Dict[str, float]:
    """The historical equity premium (Mkt-RF) per year: arithmetic mean of monthly excess returns x 12, and the
    geometric excess return, over the window."""
    f = load_factors("monthly")
    f = f[(f.index >= pd.Timestamp(start)) & ((f.index <= pd.Timestamp(end)) if end is not None else True)]
    ex = f["Mkt-RF"].dropna()
    rf = f["RF"].reindex(ex.index)
    geo = float(((1 + ex + rf).prod() / (1 + rf).prod()) ** (12 / len(ex)) - 1) if len(ex) else float("nan")
    return {"arithmetic": float(ex.mean() * 12), "geometric": geo, "months": int(len(ex)),
            "start": str(ex.index[0].date()) if len(ex) else None, "end": str(ex.index[-1].date()) if len(ex) else None}


__all__ = ["load_factors", "load_industries", "sic_industry", "market_premium", "parse_sections", "parse_sic_codes",
           "CITATION"]
