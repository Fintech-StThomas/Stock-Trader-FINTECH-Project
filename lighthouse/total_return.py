"""
Corporate actions and total returns: dividends, splits, and what to assume when a stock stops trading.

Three kinds of price, and when each one is right:
  as traded       the price on the tape that day. Per-share accounting numbers (EPS in a 10-K) are in these units.
  split-adjusted  every earlier price divided by the later splits, so a 4-for-1 split is not a 75% crash. Price-only:
                  dividends are not in it. Valuation ratios (P/E) need this price with EPS in the same share units.
  total return    split-adjusted and dividend-adjusted (Yahoo's "adjusted close", the sample pack). Returns,
                  backtests and portfolios use this one: a dividend is part of what the investor earned.

Yahoo's adjustment: adjusted(t) = close(t) x the product, over every later ex-dividend date d, of
(1 - dividend(d) / close(day before d)). `price_only_close` inverts it exactly, given the dividends.

Delisted stocks: free price feeds usually drop a stock's history once it is delisted, so a test of today's tickers
silently skips every company that went bust. `apply_delisting` applies the two bounds Lighthouse reports side by side:
"pessimistic" (the position is worth nothing after the last price, a -100% return) and "exclude" (the last price stands,
as if the investor got out at it).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import store
from .data import DataUnavailable, SAMPLE_DIR, offline_mode

ACTION_COLUMNS = ["dividend", "split"]
DELISTING_RULES = ("pessimistic", "exclude")


def _empty_actions() -> pd.DataFrame:
    return pd.DataFrame({"dividend": pd.Series(dtype=float), "split": pd.Series(dtype=float)},
                        index=pd.DatetimeIndex([], name="Date"))


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_actions()
    out = pd.DataFrame(index=pd.to_datetime(df.index))
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out["dividend"] = pd.to_numeric(df.get("dividend", df.get("Dividends", 0.0)), errors="coerce").fillna(0.0).to_numpy()
    sp = pd.to_numeric(df.get("split", df.get("Stock Splits", 0.0)), errors="coerce").fillna(0.0).to_numpy()
    out["split"] = np.where(sp > 0, sp, 1.0)
    out.index.name = "Date"
    out = out.groupby(level=0).agg({"dividend": "sum", "split": "prod"})
    return out[(out["dividend"] != 0) | (out["split"] != 1.0)].sort_index()


def sample_actions_path(ticker: str):
    return SAMPLE_DIR / "actions" / f"{ticker.upper().replace('^', '_idx_')}.csv"


def load_actions(ticker: str, source: str = "auto", refresh: bool = False) -> pd.DataFrame:
    """Dividends (per share) and splits (ratio, e.g. 4.0 for 4-for-1) by ex-date, in today's share units (Yahoo's
    convention). Order: fresh cache (20 h) -> Yahoo -> stale cache -> the sample pack's recorded copy; empty when
    nothing is known (the caller then cannot adjust, and says so)."""
    t = ticker.upper().strip()
    sample = sample_actions_path(t)
    if source == "sample" or offline_mode():
        cached = store.read_frame("actions", t)
        if cached is not None and source != "sample":
            return _tidy(cached)
        return _tidy(pd.read_csv(sample, index_col=0, parse_dates=True)) if sample.exists() else _empty_actions()
    cached = store.read_frame("actions", t)
    path = store.path_for("actions", t)
    if cached is not None and not refresh and store.age_hours(path) < 20:
        return _tidy(cached)
    try:
        import yfinance as yf
        raw = yf.Ticker(t).actions
        df = _tidy(raw if raw is not None else None)
        store.write_frame("actions", t, df)
        return df
    except Exception:  # noqa: BLE001 - no actions means no adjustment, reported by the caller
        if cached is not None:
            return _tidy(cached)
        return _tidy(pd.read_csv(sample, index_col=0, parse_dates=True)) if sample.exists() else _empty_actions()


def split_factor_after(dates, actions: pd.DataFrame, end=None) -> np.ndarray:
    """For each date: the product of split ratios with an ex-date after it (and on or before `end`). Dividing an
    as-traded per-share number from that date by this factor puts it in the share units of `end`."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    if actions is None or actions.empty:
        return np.ones(len(d))
    sp = actions["split"]
    sp = sp[sp != 1.0]
    if end is not None:
        sp = sp[sp.index <= pd.Timestamp(end)]
    if sp.empty:
        return np.ones(len(d))
    ex = sp.index.to_numpy()
    cum_after = np.cumprod(sp.to_numpy()[::-1])[::-1]          # product of ratios from each split to the last
    pos = np.searchsorted(ex, d.to_numpy(), side="right")      # first split strictly after each date
    out = np.ones(len(d))
    inside = pos < len(ex)
    out[inside] = cum_after[pos[inside]]
    return out


def to_units_of(actions: pd.DataFrame, end) -> pd.DataFrame:
    """Actions up to `end`, with dividends restated from today's share units to the units of `end` (a price series
    adjusted on that date, such as the sample pack, does not know the splits that came later)."""
    if actions is None or actions.empty:
        return _empty_actions()
    end = pd.Timestamp(end)
    later = actions.loc[actions.index > end, "split"].prod()
    out = actions[actions.index <= end].copy()
    out["dividend"] = out["dividend"] * float(later)
    return out


def price_only_close(adj_close: pd.Series, actions: pd.DataFrame) -> pd.Series:
    """Undo the dividend adjustment of a total-return close (keeping its split adjustment): the split-adjusted closing
    price. `actions` must be in the series' share units (see `to_units_of`). Exact inverse of Yahoo's method: going
    back from the last day, at the day before each ex-date the price is adjusted / factor + dividend."""
    adj = adj_close.astype(float)
    if actions is None or actions.empty or not (actions["dividend"] > 0).any():
        return adj.copy()
    div = actions["dividend"]
    div = div[div > 0]
    idx = adj.index
    # dividend on the session that is the ex-date (or the first session after it)
    pos = np.searchsorted(idx.to_numpy(), div.index.to_numpy(), side="left")
    d_on = np.zeros(len(idx))
    ok = (pos > 0) & (pos < len(idx))
    np.add.at(d_on, pos[ok], div.to_numpy()[ok])
    a = adj.to_numpy()
    out = np.empty(len(a))
    f = 1.0                          # product of (1 - D/C) over ex-dates after the current day
    for i in range(len(a) - 1, -1, -1):
        out[i] = a[i] / f
        if i > 0 and d_on[i] > 0:
            prev = a[i - 1] / f + d_on[i]          # the close before the ex-date, price-only
            f *= (1.0 - d_on[i] / prev) if prev > 0 else 1.0
    return pd.Series(out, index=idx, name=adj_close.name)


def total_return_index(close: pd.Series, actions: pd.DataFrame) -> pd.Series:
    """A total-return index (starts at 1) from a split-adjusted, price-only close and the dividends (in the same
    units): each ex-date's dividend is reinvested at that day's close. Days without a close carry the last value."""
    full = close.index
    c = close.astype(float)
    c = c[np.isfinite(c) & (c > 0)]
    d_on = pd.Series(0.0, index=c.index)
    if actions is not None and not actions.empty:
        div = actions["dividend"][actions["dividend"] > 0]
        pos = np.searchsorted(c.index.to_numpy(), div.index.to_numpy(), side="left")
        ok = (pos > 0) & (pos < len(c))
        vals = np.zeros(len(c))
        np.add.at(vals, pos[ok], div.to_numpy()[ok])
        d_on = pd.Series(vals, index=c.index)
    gross = (c + d_on) / c.shift(1)
    return gross.fillna(1.0).cumprod().reindex(full).ffill().rename("total_return")


def as_traded_close(split_adj_close: pd.Series, actions: pd.DataFrame, end=None) -> pd.Series:
    """The price as it traded each day, from a split-adjusted close (adjusted as of `end`, default its last date)."""
    end = end if end is not None else split_adj_close.index[-1]
    return split_adj_close * split_factor_after(split_adj_close.index, actions, end)


def per_share_in_units_of(values: pd.Series, known_on: pd.Series, actions: pd.DataFrame, end) -> pd.Series:
    """Per-share numbers reported on `known_on` dates (as traded then), restated in the share units of `end`."""
    return values / split_factor_after(pd.to_datetime(known_on), actions, end)


# --------------------------------------------------------------------------- #
# Delistings
# --------------------------------------------------------------------------- #
def is_delisted(prices: pd.DataFrame, data_end, tolerance_days: int = 10) -> bool:
    """A price series that stops well before the rest of the data has stopped trading (delisted, merged or renamed)."""
    return len(prices) > 0 and prices.index[-1] < pd.Timestamp(data_end) - pd.Timedelta(days=tolerance_days)


def apply_delisting(prices: pd.DataFrame, rule: str, data_end=None, tolerance_days: int = 10) -> pd.DataFrame:
    """The price frame under a delisting rule. "exclude": unchanged (the last price stands). "pessimistic": if the
    series stops before `data_end`, one more session is added on which the stock is worth nothing (a -100% return for
    whoever still holds it), so buy-and-hold and any open position lose everything."""
    if rule not in DELISTING_RULES:
        raise ValueError(f"rule must be one of {DELISTING_RULES}")
    if rule == "exclude" or data_end is None or not is_delisted(prices, data_end, tolerance_days):
        return prices
    nxt = prices.index[-1] + pd.offsets.BDay(1)
    tiny = float(prices["Close"].iloc[-1]) * 1e-9          # zero, kept positive so ratios stay defined
    row = pd.DataFrame({c: [tiny] for c in ("Open", "High", "Low", "Close")}, index=pd.DatetimeIndex([nxt], name=prices.index.name))
    if "Volume" in prices:
        row["Volume"] = 0.0
    out = pd.concat([prices, row])
    out.attrs = dict(prices.attrs)
    out.attrs["delisting"] = f"pessimistic: worth nothing on {nxt.date()} (prices end {prices.index[-1].date()})"
    return out


def delisting_return(rule: str) -> Optional[float]:
    """The return booked for the period a stock disappears without a final price: -100% (pessimistic) or none (the
    stock leaves the sample: exclude)."""
    if rule not in DELISTING_RULES:
        raise ValueError(f"rule must be one of {DELISTING_RULES}")
    return -1.0 if rule == "pessimistic" else None


__all__ = ["load_actions", "split_factor_after", "to_units_of", "price_only_close", "total_return_index",
           "as_traded_close", "per_share_in_units_of", "apply_delisting", "is_delisted", "delisting_return",
           "DELISTING_RULES", "DataUnavailable"]
