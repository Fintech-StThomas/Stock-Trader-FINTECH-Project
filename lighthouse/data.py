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

from .fundamentals import FundamentalSnapshot, StatementHistory

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


def nyse_holidays(start_year: int = 1990, end_year: Optional[int] = None) -> pd.DatetimeIndex:
    """Full-day NYSE closures: New Year's Day, Martin Luther King Jr. Day (from 1998), Presidents' Day, Good Friday,
    Memorial Day, Juneteenth (from 2022), Independence Day, Labor Day, Thanksgiving and Christmas, with the exchange's
    weekend rules (Saturday -> the Friday before, except New Year's Day; Sunday -> the Monday after). One-off closures
    (national days of mourning, 9/11, Hurricane Sandy) are not listed. Early closes (13:00) come from the feed's
    trading-period metadata when it has one."""
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
    return _NYSE().holidays(pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{end_year}-12-31"))


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


_META_KEYS = ("regularMarketPrice", "regularMarketTime", "exchangeTimezoneName", "currentTradingPeriod")


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
    r = requests.get(url, timeout=20)
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
                # requests' own User-Agent: FRED's CDN stalled (read timeouts) on custom or browser-like ones
                r = requests.get(FRED_TBILL_URL, timeout=15)
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
# Fundamentals
# --------------------------------------------------------------------------- #
_LINE_ITEMS = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"],
    "eps": ["Diluted EPS", "Basic EPS"],
    "operating_income": ["Operating Income", "EBIT", "Total Operating Income As Reported"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "total_debt": ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "fcf": ["Free Cash Flow"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure"],
    "shares": ["Diluted Average Shares", "Basic Average Shares", "Ordinary Shares Number", "Share Issued"],
}


def _pick(frame: Optional[pd.DataFrame], names: List[str]) -> Optional[pd.Series]:
    if frame is None or frame is False or len(frame) == 0:
        return None
    for n in names:
        if n in frame.index:
            s = pd.to_numeric(frame.loc[n], errors="coerce")
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            return s
    return None


def _statements_to_frame(inc, bs, cf) -> pd.DataFrame:
    cols = {}
    for key, names in _LINE_ITEMS.items():
        for frame in (inc, bs, cf):
            s = _pick(frame, names)
            if s is not None:
                cols[key] = s
                break
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df.index.name = "period_end"
    if "fcf" not in df and {"operating_cash_flow", "capex"} <= set(df.columns):
        df["fcf"] = df["operating_cash_flow"] + df["capex"]   # capex is reported negative
    return df.dropna(how="all")


def _f(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _snapshot_from_yfinance(ticker: str, info: dict, hist: StatementHistory, price: Optional[float]) -> FundamentalSnapshot:
    g = lambda k: _f(info.get(k))  # noqa: E731
    a = hist.annual
    q = hist.quarterly
    last = a.iloc[-1] if len(a) else pd.Series(dtype=float)
    prev = a.iloc[-2] if len(a) > 1 else pd.Series(dtype=float)

    price = g("currentPrice") or g("regularMarketPrice") or price
    market_cap = g("marketCap")
    shares = g("sharesOutstanding") or _f(last.get("shares"))
    if market_cap is None and price and shares:
        market_cap = price * shares

    # trailing EPS: prefer the sum of the last four quarters, then the last annual figure
    eps_ttm = None
    if len(q) >= 4 and "eps" in q:
        eps_ttm = _f(q["eps"].tail(4).sum()) if q["eps"].tail(4).notna().all() else None
    eps_ttm = g("trailingEps") or eps_ttm or _f(last.get("eps"))

    equity, debt, cash = _f(last.get("equity")), _f(last.get("total_debt")), _f(last.get("cash"))
    ebitda, revenue, ni = _f(last.get("ebitda")), _f(last.get("revenue")), _f(last.get("net_income"))
    fcf = g("freeCashflow") or _f(last.get("fcf"))

    pe = g("trailingPE") or ((price / eps_ttm) if price and eps_ttm and eps_ttm > 0 else None)
    if pe is None and price and eps_ttm is not None and eps_ttm <= 0:
        pe = -1.0   # sentinel: unprofitable
    pb = g("priceToBook") or ((market_cap / equity) if market_cap and equity and equity > 0 else None)
    ev = g("enterpriseValue") or ((market_cap + (debt or 0) - (cash or 0)) if market_cap else None)
    ev_ebitda = g("enterpriseToEbitda") or ((ev / ebitda) if ev and ebitda and ebitda > 0 else None)

    # growth: annual statement growth is the primary (stable) measure; Yahoo's quarterly YoY is the fallback
    rev_g = None
    if _f(last.get("revenue")) and _f(prev.get("revenue")):
        rev_g = last["revenue"] / prev["revenue"] - 1.0
    rev_g = _f(rev_g) if rev_g is not None else g("revenueGrowth")
    eps_g = None
    if _f(last.get("eps")) and _f(prev.get("eps")) and prev["eps"] > 0:
        eps_g = last["eps"] / prev["eps"] - 1.0
    eps_g = _f(eps_g) if eps_g is not None else g("earningsGrowth")

    peg = g("pegRatio") or g("trailingPegRatio")
    if peg is None and pe and pe > 0 and eps_g and eps_g > 0:
        peg = pe / (eps_g * 100.0)

    roe = g("returnOnEquity")
    if roe is None and ni is not None and equity and equity > 0:
        roe = ni / equity
    op_margin = g("operatingMargins")
    if op_margin is None and _f(last.get("operating_income")) and revenue:
        op_margin = last["operating_income"] / revenue
    profit_margin = g("profitMargins") or ((ni / revenue) if ni is not None and revenue else None)
    de = g("debtToEquity")
    de = de / 100.0 if de is not None else ((debt / equity) if debt is not None and equity and equity > 0 else None)
    cr = g("currentRatio")
    if cr is None and _f(last.get("current_assets")) and _f(last.get("current_liabilities")):
        cr = last["current_assets"] / last["current_liabilities"]
    # dividend yield: Yahoo's `dividendYield` switched from a fraction to a percent (0.0073 -> 0.73), so it is
    # ambiguous for small yields. Prefer the unambiguous annual dividend per share / price, then the trailing fraction.
    div = None
    rate = g("dividendRate")
    if rate is not None and price:
        div = rate / price
    if div is None:
        div = g("trailingAnnualDividendYield")
    if div is None:
        div = g("dividendYield")
        if div is not None and div > 0.3:   # only a percent reading can exceed 30%; anything smaller stays ambiguous
            div = div / 100.0

    return FundamentalSnapshot(
        ticker=ticker, name=info.get("shortName") or info.get("longName") or ticker,
        sector=info.get("sector"), industry=info.get("industry"), currency=info.get("currency") or "USD",
        price=price, market_cap=market_cap, pe_trailing=pe, pe_forward=g("forwardPE"), peg=peg, pb=pb,
        ev_ebitda=ev_ebitda, roe=roe, operating_margin=op_margin, profit_margin=profit_margin,
        debt_to_equity=de, current_ratio=cr, fcf=fcf,
        fcf_yield=(fcf / market_cap) if fcf is not None and market_cap else None,
        revenue_growth=rev_g, eps_growth=eps_g, dividend_yield=div, beta=g("beta"),
        analyst_target=g("targetMeanPrice"), analyst_count=g("numberOfAnalystOpinions"),
        as_of=datetime.now().strftime("%Y-%m-%d"), source="yfinance",
    )


def _fundamentals_from_yfinance(ticker: str, last_price: Optional[float]) -> Tuple[FundamentalSnapshot, StatementHistory]:
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        info = dict(t.info or {})
    except Exception:  # noqa: BLE001
        info = {}
    def grab(attr):
        try:
            v = getattr(t, attr)
            return v if isinstance(v, pd.DataFrame) and len(v) else None
        except Exception:  # noqa: BLE001
            return None
    annual = _statements_to_frame(grab("income_stmt"), grab("balance_sheet"), grab("cashflow"))
    quarterly = _statements_to_frame(grab("quarterly_income_stmt"), grab("quarterly_balance_sheet"), grab("quarterly_cashflow"))
    hist = StatementHistory(annual=annual, quarterly=quarterly)
    if not info and annual.empty:
        raise DataUnavailable(f"yfinance returned no fundamentals for {ticker}")
    snap = _snapshot_from_yfinance(ticker, info, hist, last_price)
    return snap, hist


def _fundamentals_to_json(snap: FundamentalSnapshot, hist: StatementHistory) -> dict:
    def frame_rows(df: pd.DataFrame) -> list:
        if df is None or df.empty:
            return []
        out = df.copy()
        out.insert(0, "period_end", [d.strftime("%Y-%m-%d") for d in out.index])
        return json.loads(out.to_json(orient="records"))
    return {"snapshot": snap.to_dict(), "annual": frame_rows(hist.annual), "quarterly": frame_rows(hist.quarterly)}


def _fundamentals_from_json(path: Path, ticker: str) -> Tuple[FundamentalSnapshot, StatementHistory]:
    payload = json.loads(Path(path).read_text())
    snap = FundamentalSnapshot.from_dict({**payload.get("snapshot", {}), "ticker": ticker})
    def rows_to_frame(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["period_end"] = pd.to_datetime(df["period_end"])
        return df.set_index("period_end").sort_index().apply(pd.to_numeric, errors="coerce")
    return snap, StatementHistory(annual=rows_to_frame(payload.get("annual", [])),
                                  quarterly=rows_to_frame(payload.get("quarterly", [])))


SECTOR_CACHE_DAYS = 30


def _sector_cache_file() -> Path:
    return CACHE_DIR / "fundamentals" / "_sectors.json"


def _sector_from_snapshot_file(path: Path) -> Optional[str]:
    try:
        return json.loads(path.read_text()).get("snapshot", {}).get("sector") or None
    except (OSError, ValueError, AttributeError):
        return None


_NO_SECTOR_TYPES = {"ETF", "MUTUALFUND", "INDEX", "CURRENCY", "CRYPTOCURRENCY", "FUTURE"}


def _sector_from_yfinance(ticker: str) -> Tuple[Optional[str], bool]:
    """(sector, final). "No sector" is final only for instruments that never have one (funds, indices); a stock whose
    answer came back without a sector (a thin or degraded response) is asked again next time."""
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    sector = info.get("sector") or None
    return sector, bool(sector) or str(info.get("quoteType", "")).upper() in _NO_SECTOR_TYPES


def load_sectors(tickers: List[str], source: str = "auto", max_workers: int = 8) -> Dict[str, Optional[str]]:
    """The sector each ticker's fundamentals snapshot reports (Yahoo's `info["sector"]`), or None when there is none
    (ETFs, unknown symbols, no connection).

    Looks in the fundamentals cache and the sample fixtures first (a sector rarely changes, so any age is fine), then a
    small sector cache, then asks Yahoo for the missing ones in parallel (only `info`, not the statements) and caches
    the answers for SECTOR_CACHE_DAYS days.
    """
    out: Dict[str, Optional[str]] = {}
    for t in dict.fromkeys(x.upper().strip() for x in tickers if x and x.strip()):
        sec = None
        for p in (_cache_path("fundamentals", t), SAMPLE_DIR / "fundamentals" / f"{t}.json"):
            if p.exists():
                sec = _sector_from_snapshot_file(p)
                if sec:
                    break
        out[t] = sec
    missing = [t for t, s in out.items() if s is None]
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


def load_fundamentals(ticker: str, source: str = "auto", last_price: Optional[float] = None,
                      overrides: Optional[Dict[str, float]] = None,
                      fixture_path: Optional[str] = None) -> Tuple[FundamentalSnapshot, StatementHistory]:
    """Load the fundamental snapshot and statement history for `ticker`.

    `overrides` lets a non-programmer type in their own numbers (e.g. from a broker
    page) when a feed is missing a field; `fixture_path` loads a JSON file in the
    same format Lighthouse writes to its cache.
    """
    ticker = ticker.upper().strip()
    attempts: List[str] = []
    snap: Optional[FundamentalSnapshot] = None
    hist = StatementHistory()

    if fixture_path:
        order = ["fixture"]
    elif source == "auto":
        order = ["cache", "sample"] if offline_mode() else ["cache", "yfinance", "stale-cache", "sample"]
    else:
        order = [source]

    for src in order:
        try:
            if src == "fixture":
                snap, hist = _fundamentals_from_json(Path(fixture_path), ticker)
            elif src == "yfinance":
                snap, hist = _fundamentals_from_yfinance(ticker, last_price)
                try:
                    p = _cache_path("fundamentals", ticker)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(_fundamentals_to_json(snap, hist), indent=1, default=str))
                except OSError:
                    pass
            elif src in {"cache", "stale-cache"}:
                p = _cache_path("fundamentals", ticker)
                if not p.exists():
                    raise DataUnavailable("no cache")
                age_h = (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600
                if src == "cache" and age_h > CACHE_MAX_AGE_HOURS:
                    raise DataUnavailable("cache stale")
                snap, hist = _fundamentals_from_json(p, ticker)
                snap.source = f"{snap.source} (cached {age_h / 24:.0f}d ago)"
            elif src == "sample":
                p = SAMPLE_DIR / "fundamentals" / f"{ticker}.json"
                if not p.exists():
                    raise DataUnavailable("no offline fixture")
                snap, hist = _fundamentals_from_json(p, ticker)
            else:
                raise ValueError(f"unknown source {src!r}")
            break
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{src}: {type(exc).__name__}: {str(exc)[:120]}")
            snap = None

    if snap is None:
        snap = FundamentalSnapshot(ticker=ticker, name=ticker, source="unavailable",
                                   as_of=datetime.now().strftime("%Y-%m-%d"))
        snap.notes.append("No fundamental data could be loaded (" + " | ".join(attempts) + ")")
    if overrides:
        for k, v in overrides.items():
            if hasattr(snap, k):
                setattr(snap, k, _f(v))
        snap.notes.append("Investor-supplied values override the feed for: " + ", ".join(overrides))
    if snap.price is None and last_price is not None:
        snap.price = last_price
    return snap, hist
