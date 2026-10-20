"""
Data collection, cleaning and caching.

Sources (tried in this order when source="auto"):
  1. Yahoo Finance via `yfinance`     - prices (dividend & split adjusted) + fundamentals
  2. Stooq CSV endpoint (no API key)  - prices only (split adjusted)
  3. Local cache (data/cache)         - the last successful live download
  4. Bundled sample pack (data/sample)- real historical data shipped with the
     repo so demos, tests and the notebook run with no internet connection

Set the environment variable LIGHTHOUSE_OFFLINE=1 (or source="sample") to skip
the network entirely.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
SAMPLE_DIR = Path(os.environ.get("LIGHTHOUSE_SAMPLE_DIR", REPO_DIR / "data" / "sample"))
CACHE_DIR = Path(os.environ.get("LIGHTHOUSE_CACHE_DIR", REPO_DIR / "data" / "cache"))
CACHE_MAX_AGE_HOURS = 20
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
STOOQ_SYMBOLS = {"^GSPC": "^spx", "^DJI": "^dji", "^IXIC": "^ndq"}


# yfinance and the pandas calls it makes emit deprecation chatter that the investor cannot act on. Silence it once, by
# module, instead of with warnings.catch_warnings(): that context manager changes process-wide state and is not safe
# when several tickers download in parallel threads.
warnings.filterwarnings("ignore", module=r"yfinance(\..*)?$")


def offline_mode() -> bool:
    return os.environ.get("LIGHTHOUSE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


class DataUnavailable(RuntimeError):
    """Raised when no source could provide the requested data."""


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    ticker: str = ""
    source: str = ""
    rows_raw: int = 0
    rows_clean: int = 0
    duplicates_removed: int = 0
    rows_dropped: int = 0
    missing_filled: int = 0
    volume_available: bool = True
    start: Optional[str] = None
    end: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.ticker}: {self.rows_clean:,} trading days from {self.start} to {self.end} (source: {self.source})",
                 f"  raw rows {self.rows_raw:,} | duplicates removed {self.duplicates_removed} | "
                 f"rows dropped {self.rows_dropped} | gaps filled {self.missing_filled} | "
                 f"volume {'available' if self.volume_available else 'NOT available'}"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _f(x) -> Optional[float]:
    """A finite float, or None."""
    try:
        if x is None:
            return None
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df.index.name = "Date"
    return df


def clean_prices(df: pd.DataFrame, ticker: str = "", source: str = "") -> Tuple[pd.DataFrame, CleaningReport]:
    """Standardise an OHLCV frame and report every change that was made."""
    rep = CleaningReport(ticker=ticker, source=source, rows_raw=len(df))
    df = _normalize_index(df)
    # column names
    rename = {c: c.title() for c in df.columns if isinstance(c, str)}
    df = df.rename(columns=rename)
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    for c in PRICE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # order & duplicates
    df = df.sort_index()
    dup = int(df.index.duplicated(keep="last").sum())
    if dup:
        df = df[~df.index.duplicated(keep="last")]
    rep.duplicates_removed = dup

    # rows without a usable close are unusable
    bad = df["Close"].isna() | (df["Close"] <= 0)
    rep.rows_dropped = int(bad.sum())
    df = df[~bad]

    # fill isolated gaps in O/H/L from the close (never forward in time)
    fills = 0
    for c in ["Open", "High", "Low"]:
        m = df[c].isna() | (df[c] <= 0)
        fills += int(m.sum())
        df.loc[m, c] = df.loc[m, "Close"]
    # guarantee High >= max(Open, Close) >= min(Open, Close) >= Low
    hi = df[["Open", "High", "Close"]].max(axis=1)
    lo = df[["Open", "Low", "Close"]].min(axis=1)
    fixed = int(((df["High"] != hi) | (df["Low"] != lo)).sum())
    df["High"], df["Low"] = hi, lo
    rep.missing_filled = fills + fixed

    vol_ok = df["Volume"].notna().mean() > 0.5
    rep.volume_available = bool(vol_ok)
    if not vol_ok:
        df["Volume"] = np.nan
        rep.notes.append("volume missing for most rows: volume-based signals are disabled for this series")
    else:
        df["Volume"] = df["Volume"].astype(float)

    # extreme one-day moves are kept but flagged (could be a split that was not adjusted)
    rets = df["Close"].pct_change().abs()
    jumps = rets[rets > 0.5]
    for d, r in jumps.items():
        rep.notes.append(f"{d.date()}: {r * 100:.0f}% one-day move - check for an unadjusted split or data error")

    rep.rows_clean = len(df)
    if len(df):
        rep.start, rep.end = str(df.index[0].date()), str(df.index[-1].date())
    return df, rep


# --------------------------------------------------------------------------- #
# Market clock and live quotes
# --------------------------------------------------------------------------- #
EXCHANGE_TZ = "America/New_York"
SESSION_OPEN, SESSION_CLOSE = (9, 30), (16, 0)      # US regular session, exchange time


def _now_et() -> pd.Timestamp:
    """The current time on the exchange's clock (a function so tests can freeze it)."""
    return pd.Timestamp.now(tz=EXCHANGE_TZ)


# Full-day NYSE closures outside the holiday rules since 1990, the calendar's first year. Unplanned, so none can be
# listed before it happens: one announced later is added here.
NYSE_SPECIAL_CLOSURES: Dict[str, str] = {
    "1994-04-27": "national day of mourning, President Nixon",
    "2001-09-11": "terrorist attacks of 11 September", "2001-09-12": "terrorist attacks of 11 September",
    "2001-09-13": "terrorist attacks of 11 September", "2001-09-14": "terrorist attacks of 11 September",
    "2004-06-11": "national day of mourning, President Reagan",
    "2007-01-02": "national day of mourning, President Ford",
    "2012-10-29": "Hurricane Sandy", "2012-10-30": "Hurricane Sandy",
    "2018-12-05": "national day of mourning, President George H. W. Bush",
    "2025-01-09": "national day of mourning, President Carter",
}


def nyse_holidays(start_year: int = 1990, end_year: Optional[int] = None) -> pd.DatetimeIndex:
    """Full-day NYSE closures: New Year's Day, Martin Luther King Jr. Day (from 1998), Presidents' Day, Good Friday,
    Memorial Day, Juneteenth (from 2022), Independence Day, Labor Day, Thanksgiving and Christmas, with the exchange's
    weekend rules (Saturday -> the Friday before, except New Year's Day; Sunday -> the Monday after), and the one-off
    closures in NYSE_SPECIAL_CLOSURES (national days of mourning, 9/11, Hurricane Sandy). Early closes (13:00) come
    from the feed's trading-period metadata when it has one."""
    from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
                                        USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday)

    class _NYSE(AbstractHolidayCalendar):
        rules = [
            Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
            Holiday("Martin Luther King Jr. Day", month=1, day=1, offset=USMartinLutherKingJr.offset, start_date="1998-01-01"),
            USPresidentsDay, GoodFriday, USMemorialDay,
            Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
            Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
            USLaborDay, USThanksgivingDay,
            Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        ]
    end_year = end_year or (pd.Timestamp.now().year + 2)
    lo, hi = pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{end_year}-12-31")
    special = pd.DatetimeIndex(sorted(NYSE_SPECIAL_CLOSURES))
    return _NYSE().holidays(lo, hi).union(special[(special >= lo) & (special <= hi)])


_HOLIDAYS: Optional[set] = None


def is_trading_day(day) -> bool:
    """A weekday that is not a full-day NYSE holiday."""
    global _HOLIDAYS
    if _HOLIDAYS is None:
        _HOLIDAYS = {d.date() for d in nyse_holidays()}
    d = pd.Timestamp(day)
    return d.weekday() < 5 and d.date() not in _HOLIDAYS


def last_session_close(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """The most recent 16:00 New York time on a trading day (a weekday that is not an NYSE holiday) at or before
    `now`. Early closes (13:00) are treated as 16:00: the cache then expires up to three hours late on those days."""
    now = _now_et() if now is None else now.tz_convert(EXCHANGE_TZ)
    close = now.normalize() + pd.Timedelta(hours=SESSION_CLOSE[0], minutes=SESSION_CLOSE[1])
    if now < close:
        close -= pd.Timedelta(days=1)
    while not is_trading_day(close.tz_localize(None).normalize()):
        close -= pd.Timedelta(days=1)
    return close


def sessions_behind(last_bar, now: Optional[pd.Timestamp] = None) -> int:
    """How many finished trading sessions the data is missing: 0 when its last bar is the last finished session."""
    target = last_session_close(now).tz_localize(None).normalize()
    d = pd.Timestamp(last_bar).normalize()
    if d >= target:
        return 0
    days = pd.date_range(d + pd.Timedelta(days=1), target, freq="D")
    return int(sum(1 for x in days if is_trading_day(x)))


def _session_window(day: pd.Timestamp, meta: Optional[dict] = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Regular-session start and end on `day` (exchange time): from Yahoo's trading-period metadata when it describes
    that day, otherwise the standard 9:30-16:00 New York session."""
    tz = (meta or {}).get("exchangeTimezoneName") or EXCHANGE_TZ
    reg = ((meta or {}).get("currentTradingPeriod") or {}).get("regular") or {}
    try:
        start, end = pd.Timestamp(reg["start"]), pd.Timestamp(reg["end"])
        if start.tzinfo is None:
            start, end = start.tz_localize(tz), end.tz_localize(tz)
        start, end = start.tz_convert(tz), end.tz_convert(tz)
        if start.normalize().date() == day.date():
            return start, end
    except (KeyError, TypeError, ValueError):
        pass
    d = pd.Timestamp(day.date()).tz_localize(EXCHANGE_TZ)
    return (d + pd.Timedelta(hours=SESSION_OPEN[0], minutes=SESSION_OPEN[1]),
            d + pd.Timedelta(hours=SESSION_CLOSE[0], minutes=SESSION_CLOSE[1]))


def is_partial_bar(bar_date, now: Optional[pd.Timestamp] = None, meta: Optional[dict] = None) -> bool:
    """True when the daily bar dated `bar_date` belongs to a session that has not closed yet (trading now, or a bar a feed
    shows before the open): its high, low, close and volume will still change, so it is not a finished day. Dates are
    compared on the exchange's own calendar (the feed's time zone, New York without one): a Tokyo bar is dated in Tokyo."""
    tz = (meta or {}).get("exchangeTimezoneName") or EXCHANGE_TZ
    now = (_now_et() if now is None else now).tz_convert(tz)
    day = pd.Timestamp(bar_date)
    if day.date() != now.date() or now.weekday() >= 5:
        return False
    _, end = _session_window(day, meta)
    return now < end


@dataclass
class Quote:
    """The latest traded price with its timestamp, as the price feed reports it."""
    ticker: str
    price: float
    time: str                    # ISO timestamp, exchange time
    market_state: str            # "open" | "pre-market" | "after-hours" | "closed"
    source: str = "yfinance"
    tz: str = EXCHANGE_TZ
    note: str = "Yahoo quotes can lag by up to 15 minutes on some exchanges; confirm the price with your broker."

    @property
    def timestamp(self) -> pd.Timestamp:
        t = pd.Timestamp(self.time)
        return t.tz_localize(self.tz) if t.tzinfo is None else t.tz_convert(self.tz)

    def describe(self) -> str:
        t = self.timestamp
        label = {"open": "market open", "pre-market": "pre-market: last regular-session price",
                 "after-hours": "after hours: regular-session close", "closed": "market closed"}.get(self.market_state, self.market_state)
        return f"${self.price:,.2f} at {t.strftime('%Y-%m-%d %H:%M')} {t.tzname() or ''} ({label}, {self.source})".replace("  ", " ")

    def to_dict(self) -> dict:
        return asdict(self)


def _market_state(meta: Optional[dict], now: pd.Timestamp) -> str:
    periods = (meta or {}).get("currentTradingPeriod") or {}
    for key, label in (("regular", "open"), ("pre", "pre-market"), ("post", "after-hours")):
        win = periods.get(key) or {}
        try:
            if pd.Timestamp(win["start"]) <= now < pd.Timestamp(win["end"]):
                return label
        except (KeyError, TypeError, ValueError):
            continue
    if not periods:      # no metadata: fall back to the standard regular session
        start, end = _session_window(now)
        if now.weekday() < 5 and start <= now < end:
            return "open"
    return "closed"


def _quote_from_meta(ticker: str, meta: Optional[dict], now: Optional[pd.Timestamp] = None) -> Optional[Quote]:
    if not meta:
        return None
    price, when = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
    if when is None or (isinstance(when, float) and not np.isfinite(when)) or pd.isna(when):
        return None                     # a price without a time cannot be shown as live
    try:
        price = float(price)
        ts = pd.Timestamp(when, unit="s", tz="UTC") if isinstance(when, (int, float)) else pd.Timestamp(when)
        ts = (ts.tz_localize("UTC") if ts.tzinfo is None else ts).tz_convert(meta.get("exchangeTimezoneName") or EXCHANGE_TZ)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(price) or price <= 0:
        return None
    now = _now_et() if now is None else now
    return Quote(ticker=ticker.upper(), price=price, time=ts.isoformat(), market_state=_market_state(meta, now),
                 tz=str(ts.tz))


def load_quote(ticker: str, source: str = "auto") -> Optional[Quote]:
    """The latest price and its timestamp (one light request, never cached). None offline or when the feed fails."""
    if offline_mode() or source not in {"auto", "yfinance"}:
        return None
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        tk.history(period="5d", interval="1d", auto_adjust=True, actions=False)
        meta = _slim_meta(tk.history_metadata)
    except Exception:  # noqa: BLE001 - a missing quote only means the last close is used
        return None
    return _quote_from_meta(ticker, meta)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def _from_yfinance(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so the package works without it
    tk = yf.Ticker(ticker)
    hist = tk.history(start=start.strftime("%Y-%m-%d"), end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                      auto_adjust=True, actions=False)
    try:
        meta = _slim_meta(tk.history_metadata)
    except Exception:  # noqa: BLE001
        meta = {}
    if hist is None or len(hist) == 0:
        raise DataUnavailable(f"yfinance returned no rows for {ticker}")
    hist.attrs = {"meta": meta}
    return hist


_META_KEYS = ("regularMarketPrice", "regularMarketTime", "exchangeTimezoneName", "currentTradingPeriod", "longName",
              "shortName")


def _slim_meta(meta) -> dict:
    """The few fields of Yahoo's chart metadata the quote needs (the full dict holds DataFrames, which pandas cannot
    carry in `attrs`)."""
    try:
        return {k: meta[k] for k in _META_KEYS if k in meta}
    except TypeError:      # not a mapping
        return {}


def _from_stooq(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    import requests
    sym = STOOQ_SYMBOLS.get(ticker.upper(), f"{ticker.lower()}.us")
    url = (f"https://stooq.com/q/d/l/?s={sym}&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d")
    from .http import contact_headers
    r = requests.get(url, timeout=20, headers=contact_headers())
    r.raise_for_status()
    text = r.text.strip()
    if not text.startswith("Date") or len(text.splitlines()) < 3:
        raise DataUnavailable(f"stooq returned no data for {ticker}")
    from io import StringIO
    df = pd.read_csv(StringIO(text), parse_dates=["Date"], index_col="Date")
    return df


def sample_pack_tickers() -> List[str]:
    d = SAMPLE_DIR / "prices"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.csv"))


def _from_sample(ticker: str) -> pd.DataFrame:
    path = SAMPLE_DIR / "prices" / f"{ticker.upper()}.csv"
    if not path.exists():
        raise DataUnavailable(f"{ticker} is not in the offline sample pack ({', '.join(sample_pack_tickers())})")
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def _cache_path(kind: str, ticker: str) -> Path:
    safe = ticker.upper().replace("^", "_idx_").replace("/", "_")
    return CACHE_DIR / kind / f"{safe}.{ 'csv' if kind == 'prices' else 'json'}"


def _from_cache(ticker: str, max_age_hours: Optional[float] = CACHE_MAX_AGE_HOURS,
                need_start: Optional[datetime] = None, session_check: bool = True,
                need_current: bool = False) -> pd.DataFrame:
    """Cached prices. With `need_start`, the cache must also cover the requested window: a short download
    (e.g. a 6-year watchlist run) must not silently stand in for a later 10-year analysis. With `need_current`, it
    must hold the last finished session: a download made just after a close, before the feed published that day's
    bar, is not trusted for the rest of the day."""
    path = _cache_path("prices", ticker)
    if not path.exists():
        raise DataUnavailable("no cache")
    age_h = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
    if max_age_hours is not None and age_h > max_age_hours:
        raise DataUnavailable("cache stale")
    meta_path = path.with_suffix(".meta.json")
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except (OSError, ValueError):
        meta = {}
    if max_age_hours is not None and session_check:
        # a session that closed after the download has a finished bar the cache does not hold
        try:
            fetched = pd.Timestamp(meta["fetched_at"])
        except (KeyError, TypeError, ValueError):
            fetched = pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime)).tz_localize(datetime.now().astimezone().tzinfo)
        if fetched.tz_convert(EXCHANGE_TZ) < last_session_close():
            raise DataUnavailable("cache stale: a trading session has closed since it was downloaded")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if need_current and len(df):
        # the last row with a real close: a feed can list today's date before the bar is final (an empty close)
        closes = pd.to_numeric(df["Close"], errors="coerce") if "Close" in df else pd.Series(dtype=float)
        valid = closes[closes > 0]
        last = valid.index[-1] if len(valid) else None
        if last is None or sessions_behind(last) > 0:
            raise DataUnavailable(f"cache stale: its last close is {pd.Timestamp(last).date() if last is not None else 'missing'}, "
                                  "before the last finished session")
    if need_start is not None and len(df):
        requested = None
        try:
            requested = pd.Timestamp(meta["requested_start"]) if "requested_start" in meta else None
        except (TypeError, ValueError):
            requested = None
        tolerance = pd.Timedelta(days=7)   # weekends and holidays at the start of the window
        covered = (requested is not None and requested <= pd.Timestamp(need_start) + tolerance) or \
                  (pd.Timestamp(df.index[0]) <= pd.Timestamp(need_start) + tolerance)
        if not covered:
            raise DataUnavailable(f"cache starts {pd.Timestamp(df.index[0]).date()}, after the requested {pd.Timestamp(need_start).date()}")
    return df


def _atomic_write(path: Path, write) -> None:
    """Write through a temporary file and rename it into place, so a reader (another thread or process) sees either
    the old file or the new one, never a half-written file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _to_cache(ticker: str, df: pd.DataFrame, requested_start: Optional[datetime] = None) -> None:
    try:
        path = _cache_path("prices", ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"fetched_at": _now_et().isoformat()}
        if requested_start is not None:
            meta["requested_start"] = pd.Timestamp(requested_start).isoformat()
        _atomic_write(path, df.to_csv)
        _atomic_write(path.with_suffix(".meta.json"), lambda p: p.write_text(json.dumps(meta)))
    except OSError:
        pass


_ticker_locks: Dict[str, threading.Lock] = {}
_ticker_locks_guard = threading.Lock()


def _ticker_lock(ticker: str) -> threading.Lock:
    """One loader per ticker at a time: when 8 threads all need ^GSPC, one downloads it and the others read the cache."""
    with _ticker_locks_guard:
        return _ticker_locks.setdefault(ticker, threading.Lock())


# --------------------------------------------------------------------------- #
# Public loaders
# --------------------------------------------------------------------------- #
def _window(years: Optional[float], start, end) -> Tuple[datetime, datetime]:
    end_dt = pd.Timestamp(end).to_pydatetime() if end is not None else datetime.now()
    if start is not None:
        start_dt = pd.Timestamp(start).to_pydatetime()
    else:
        # extra year so the 200-day SMA / 52-week high are defined from the first analysed day
        start_dt = end_dt - timedelta(days=int(365.25 * ((years or 10) + 1.2)))
    return start_dt, end_dt


def load_prices(ticker: str, years: Optional[float] = 10, start=None, end=None,
                source: str = "auto", use_cache: bool = True, refresh: bool = False,
                keep_partial: bool = False) -> pd.DataFrame:
    """Load a clean daily OHLCV frame for `ticker`.

    The returned frame carries `df.attrs["source"]` and `df.attrs["cleaning"]`
    (a CleaningReport) so the product can tell the investor where its data came from.

    A bar for a session that is still trading (during market hours) is removed unless `keep_partial`: its close, range
    and volume are not final, and the rules were tested on finished days. It is kept in `df.attrs["partial_bar"]`, and
    the feed's latest price and timestamp in `df.attrs["quote"]` (a `Quote`, live sources only). `refresh` skips the
    fresh cache and downloads again (the result is still cached).
    """
    ticker = ticker.upper().strip()
    start_dt, end_dt = _window(years, start, end)
    attempts: List[str] = []
    raw, used = None, None

    order: List[str]
    today = _now_et().normalize()
    reaches_today = (end is None or pd.Timestamp(end).normalize() >= today.tz_localize(None)) and not offline_mode()
    if source == "auto":
        order = ["cache", "sample"] if offline_mode() else ["cache", "yfinance", "stooq", "stale-cache", "sample"]
        if keep_partial and not offline_mode() and reaches_today and is_partial_bar(today):
            order.remove("cache")       # the cache never holds an unfinished session: today's bar needs a download
    else:
        order = [source]

    with _ticker_lock(ticker):          # one download per ticker at a time; the others then read the fresh cache
        for src in order:
            try:
                if src == "yfinance":
                    raw = _from_yfinance(ticker, start_dt, end_dt)
                elif src == "stooq":
                    raw = _from_stooq(ticker, start_dt, end_dt)
                elif src == "cache":
                    # in "auto" a live download follows, so a cache missing a finished session is skipped; asked for
                    # explicitly, the cache is used as long as it is under 20 hours old
                    raw = (_from_cache(ticker, need_start=start_dt, session_check=source == "auto",
                                       need_current=source == "auto" and reaches_today)
                           if use_cache and not refresh else None)
                elif src == "stale-cache":
                    raw = _from_cache(ticker, max_age_hours=None) if use_cache else None
                elif src == "sample":
                    raw = _from_sample(ticker)
                else:
                    raise ValueError(f"unknown source {src!r}")
                if raw is None or len(raw) == 0:
                    raise DataUnavailable("empty")
                used = src
                break
            except Exception as exc:  # noqa: BLE001 - we want to fall through to the next source
                attempts.append(f"{src}: {type(exc).__name__}: {str(exc)[:120]}")
                raw = None

        if raw is None:
            raise DataUnavailable(f"could not load prices for {ticker}. Tried -> " + " | ".join(attempts))

        feed_meta = raw.attrs.get("meta") if used == "yfinance" else None
        quote = _quote_from_meta(ticker, feed_meta) if feed_meta else None
        raw = _normalize_index(raw)
        raw.attrs = {}                  # the feed's metadata is read; it must not ride along into every derived frame
        partial = None
        if used in {"yfinance", "stooq", "cache"} and len(raw) and is_partial_bar(raw.index[-1], meta=feed_meta):
            partial = {"date": str(raw.index[-1].date()), **{k: _f(raw[k].iloc[-1]) for k in PRICE_COLUMNS if k in raw}}
            if not keep_partial:
                raw = raw.iloc[:-1]
        if used in {"yfinance", "stooq"} and use_cache:
            _to_cache(ticker, raw if partial is None or not keep_partial else raw.iloc[:-1], requested_start=start_dt)

    df, report = clean_prices(raw, ticker, used)
    if partial is not None:
        report.notes.insert(0, f"Today's session ({partial['date']}) is still trading: its bar is "
                               + ("kept but not final." if keep_partial else
                                  f"left out, so every signal uses finished sessions (through {report.end})."))
    if used in {"sample", "stale-cache"}:
        report.notes.insert(0, f"OFFLINE DATA ({used}): series ends {report.end}; connect to the internet for live prices")
    # trim to the requested window. Offline sources ignore the live window, so the `years` window is
    # applied relative to the *last available* date (plus the warm-up year the indicators need).
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    elif years is not None and len(df):
        df = df[df.index >= df.index[-1] - timedelta(days=int(365.25 * (years + 1.2)))]
    if df.empty:
        raise DataUnavailable(f"{ticker}: the {used} data ({report.start} to {report.end}) has no rows in the requested window")
    df.attrs["ticker"] = ticker
    df.attrs["source"] = used
    df.attrs["cleaning"] = report
    df.attrs["attempts"] = attempts
    df.attrs["partial_bar"] = partial
    df.attrs["quote"] = quote
    df.attrs["name"] = str((feed_meta or {}).get("longName") or (feed_meta or {}).get("shortName") or "") if used == "yfinance" else ""
    return df


def load_benchmark(symbol: str = "^GSPC", years: Optional[float] = 10, start=None, end=None,
                   source: str = "auto", keep_partial: bool = False) -> pd.DataFrame:
    """Market benchmark (default: S&P 500 index) used for regime detection and comparison."""
    return load_prices(symbol, years=years, start=start, end=end, source=source, keep_partial=keep_partial)


# --------------------------------------------------------------------------- #
# Risk-free rate
# --------------------------------------------------------------------------- #
FRED_TBILL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"   # public CSV endpoint: no API key
_rf_lock = threading.Lock()


def _parse_fred_csv(text: str) -> pd.Series:
    """FRED's CSV (observation_date, value in percent; "." or blank on holidays) as a fraction per date."""
    df = pd.read_csv(io.StringIO(text))
    s = pd.Series(pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy() / 100.0,
                  index=pd.to_datetime(df.iloc[:, 0]), name="DTB3")
    return s.dropna().sort_index()


def _read_risk_free(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return s.rename("DTB3").dropna().sort_index()


def load_risk_free(source: str = "auto", refresh: bool = False) -> Optional[pd.Series]:
    """The 3-month Treasury bill rate (FRED series DTB3: secondary market, annualised, as a fraction) on every day it
    was published, for the Sharpe and Sortino ratios. Downloaded from FRED's public CSV endpoint and cached for up to
    20 hours (a stale cache is used if FRED cannot be reached). Returns None offline, for the sample pack, or when no
    copy can be had: the caller then uses StrategyParams.risk_free_rate (2% a year)."""
    if offline_mode() or source == "sample":
        return None
    path = CACHE_DIR / "riskfree_DTB3.csv"
    with _rf_lock:
        if path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            if source == "cache" or (age_h < CACHE_MAX_AGE_HOURS and not refresh):
                try:
                    return _read_risk_free(path)
                except Exception:  # noqa: BLE001 - a damaged cache file is simply downloaded again
                    pass
        if source != "cache":
            try:
                import requests
                from .http import contact_headers
                # requests' own User-Agent: FRED's CDN stalled (read timeouts) on custom or browser-like ones
                r = requests.get(FRED_TBILL_URL, timeout=15, headers=contact_headers())
                r.raise_for_status()
                s = _parse_fred_csv(r.text)
                if len(s) > 250:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    _atomic_write(path, lambda tmp: s.to_csv(tmp, header=True))
                    return s
            except Exception:  # noqa: BLE001 - network, HTTP or format trouble: fall back below
                pass
        if path.exists():
            try:
                return _read_risk_free(path)
            except Exception:  # noqa: BLE001
                pass
    return None


def align(prices: pd.DataFrame, benchmark: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to their common date range (benchmark is reindexed and forward-filled
    onto the stock's trading days so a missing benchmark day never drops a stock day)."""
    if prices.empty or benchmark.empty:
        raise DataUnavailable("no prices to align with the benchmark")
    start = max(prices.index[0], benchmark.index[0])
    end = min(prices.index[-1], benchmark.index[-1])
    if start > end:      # e.g. live prices with a benchmark that fell back to the 2022 sample file
        raise DataUnavailable(f"the benchmark ({benchmark.index[0].date()} to {benchmark.index[-1].date()}) does not overlap "
                              f"the prices ({prices.index[0].date()} to {prices.index[-1].date()})")
    p = prices[(prices.index >= start) & (prices.index <= end)]
    b = benchmark.reindex(p.index.union(benchmark.index)).ffill().reindex(p.index)
    return p, b


# --------------------------------------------------------------------------- #
# Sectors (Yahoo's profile; fundamentals themselves come from SEC EDGAR: lighthouse.pit, lighthouse.analyst)
# --------------------------------------------------------------------------- #
SECTOR_CACHE_DAYS = 30


def _sector_cache_file() -> Path:
    return CACHE_DIR / "fundamentals" / "_sectors.json"


_NO_SECTOR_TYPES = {"ETF", "MUTUALFUND", "INDEX", "CURRENCY", "CRYPTOCURRENCY", "FUTURE"}


def _sector_from_yfinance(ticker: str) -> Tuple[Optional[str], bool]:
    """(sector, final). "No sector" is final only for instruments that never have one (funds, indices); a stock whose
    answer came back without a sector (a thin or degraded response) is asked again next time."""
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    sector = info.get("sector") or None
    return sector, bool(sector) or str(info.get("quoteType", "")).upper() in _NO_SECTOR_TYPES


def load_sectors(tickers: List[str], source: str = "auto", max_workers: int = 8) -> Dict[str, Optional[str]]:
    """The sector Yahoo's profile reports for each ticker (`info["sector"]`), or None when there is none (ETFs, unknown
    symbols, no connection). A small sector cache first, then Yahoo for the missing ones in parallel; the answers are
    cached for SECTOR_CACHE_DAYS days.
    """
    out: Dict[str, Optional[str]] = {t: None for t in dict.fromkeys(x.upper().strip() for x in tickers if x and x.strip())}
    missing = list(out)
    if not missing:
        return out
    cache_file = _sector_cache_file()
    try:
        cached = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except (OSError, ValueError):
        cached = {}
    now = datetime.now()
    to_fetch = []
    for t in missing:
        hit = cached.get(t)
        fresh = False
        if isinstance(hit, dict):
            try:
                fresh = (now - datetime.fromisoformat(hit.get("as_of", ""))).days <= SECTOR_CACHE_DAYS
            except ValueError:
                fresh = False
        if fresh:
            out[t] = hit.get("sector") or None
        else:
            to_fetch.append(t)
    live = source in {"auto", "yfinance"} and not offline_mode()
    if not to_fetch or not live:
        return out
    from concurrent.futures import ThreadPoolExecutor

    def fetch(t: str):
        try:
            return (t, *_sector_from_yfinance(t))
        except Exception:  # noqa: BLE001 - a failed lookup just leaves the sector unknown (and is retried next time)
            return t, None, False

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(to_fetch)))) as pool:
        answers = list(pool.map(fetch, to_fetch))
    for t, sec, ok in answers:
        out[t] = sec
        if ok:
            cached[t] = {"sector": sec, "as_of": now.isoformat(timespec="seconds")}
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cached, indent=1, sort_keys=True))
    except OSError:
        pass
    return out
