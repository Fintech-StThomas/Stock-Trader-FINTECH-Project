"""Live-data plumbing without the network: the market clock, partial daily bars, the session-aware cache and orders
priced from a timestamped quote. The feed is replaced by a fake that behaves like yfinance during market hours."""
import json

import numpy as np
import pandas as pd
import pytest

import lighthouse.data as data
from lighthouse import analyze, InvestorProfile
from lighthouse.data import Quote, is_partial_bar, last_session_close, load_prices, _quote_from_meta
from lighthouse.tradeplan import build_trade_plan

ET = "America/New_York"
TUE_1100 = pd.Timestamp("2026-10-13 11:00", tz=ET)          # a Tuesday, mid-session


def _meta(day="2026-10-13", price=101.5, when=None, start="09:30", end="16:00"):
    t = lambda hm: pd.Timestamp(f"{day} {hm}", tz=ET)       # noqa: E731
    return {"regularMarketPrice": price, "regularMarketTime": when or t("10:59"), "exchangeTimezoneName": ET,
            "currentTradingPeriod": {"pre": {"start": t("04:00"), "end": t(start)}, "regular": {"start": t(start), "end": t(end)},
                                     "post": {"start": t(end), "end": t("20:00")}}}


def test_last_session_close():
    assert last_session_close(pd.Timestamp("2026-10-13 05:00", tz=ET)) == pd.Timestamp("2026-10-12 16:00", tz=ET)
    assert last_session_close(pd.Timestamp("2026-10-13 17:00", tz=ET)) == pd.Timestamp("2026-10-13 16:00", tz=ET)
    assert last_session_close(pd.Timestamp("2026-10-17 12:00", tz=ET)) == pd.Timestamp("2026-10-16 16:00", tz=ET)   # Saturday
    assert last_session_close(pd.Timestamp("2026-10-19 09:00", tz=ET)) == pd.Timestamp("2026-10-16 16:00", tz=ET)   # Monday a.m.
    assert last_session_close(pd.Timestamp("2026-10-13 15:00", tz="UTC")) == pd.Timestamp("2026-10-12 16:00", tz=ET)


def test_partial_bar_detection():
    assert is_partial_bar("2026-10-13", TUE_1100)
    assert not is_partial_bar("2026-10-13", pd.Timestamp("2026-10-13 16:30", tz=ET))    # the session has closed
    assert is_partial_bar("2026-10-13", pd.Timestamp("2026-10-13 08:00", tz=ET))        # a bar shown before the open
    assert not is_partial_bar("2026-10-12", TUE_1100)                                    # yesterday's bar is final
    early = _meta(end="13:00")                                                           # an early-close day
    assert not is_partial_bar("2026-10-13", pd.Timestamp("2026-10-13 13:30", tz=ET), early)
    assert is_partial_bar("2026-10-13", pd.Timestamp("2026-10-13 12:30", tz=ET), early)


def test_quote_from_metadata():
    q = _quote_from_meta("abc", _meta(), now=TUE_1100)
    assert q.ticker == "ABC" and q.price == 101.5 and q.market_state == "open"
    assert q.timestamp == pd.Timestamp("2026-10-13 10:59", tz=ET) and "EDT" in q.describe()
    epoch = int(pd.Timestamp("2026-10-12 20:00:01", tz="UTC").timestamp())               # Yahoo also sends epoch seconds
    q = _quote_from_meta("abc", _meta(when=epoch), now=pd.Timestamp("2026-10-13 07:00", tz=ET))
    assert q.market_state == "pre-market" and q.timestamp.hour == 16
    assert _quote_from_meta("abc", _meta(), now=pd.Timestamp("2026-10-13 21:00", tz=ET)).market_state == "closed"
    assert _quote_from_meta("abc", {"regularMarketPrice": None}) is None
    assert _quote_from_meta("abc", {}) is None


def _bars_through(day: str, n: int = 700, seed: int = 3, last_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=day, periods=n)
    close = np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    close = close / close[-1] * last_price
    df = pd.DataFrame({"Open": close * 0.998, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                       "Volume": rng.integers(1e6, 5e6, n).astype(float)}, index=idx)
    df.loc[df.index[-1], "Volume"] = 2e5                          # a partial session: a fraction of a day's volume
    return df


@pytest.fixture
def live_feed(tmp_path, monkeypatch):
    """A fake yfinance during Tuesday's session: daily bars through today, today's bar still forming."""
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    clock = {"now": TUE_1100}
    monkeypatch.setattr(data, "_now_et", lambda: clock["now"])
    calls = []

    def fake_yf(ticker, start, end):
        calls.append(ticker)
        df = _bars_through("2026-10-13", seed=len(ticker), last_price=101.5 if ticker != "^GSPC" else 5000.0)
        df.attrs = {"meta": _meta(price=float(df["Close"].iloc[-1]))}
        return df
    monkeypatch.setattr(data, "_from_yfinance", fake_yf)
    monkeypatch.setattr(data, "_from_stooq", lambda *a: pytest.fail("stooq should not be needed"))
    monkeypatch.setattr(data, "_fundamentals_from_yfinance", lambda *a: (_ for _ in ()).throw(data.DataUnavailable("no feed")))
    import lighthouse.engine as engine
    monkeypatch.setattr(engine, "load_quote", lambda *a, **k: pytest.fail("the download already carried a quote"))
    return {"clock": clock, "calls": calls, "cache": tmp_path}


def test_partial_bar_is_left_out_and_flagged(live_feed):
    df = load_prices("ABC", years=2)
    assert df.index[-1] == pd.Timestamp("2026-10-12")              # signals use finished sessions only
    assert df.attrs["partial_bar"]["date"] == "2026-10-13" and df.attrs["partial_bar"]["Volume"] == 2e5
    assert isinstance(df.attrs["quote"], Quote) and df.attrs["quote"].market_state == "open"
    assert "still trading" in df.attrs["cleaning"].notes[0]
    cached = pd.read_csv(live_feed["cache"] / "prices" / "ABC.csv", index_col=0, parse_dates=True)
    assert cached.index[-1] == pd.Timestamp("2026-10-12")          # an unfinished bar is never cached
    meta = json.loads((live_feed["cache"] / "prices" / "ABC.meta.json").read_text())
    assert pd.Timestamp(meta["fetched_at"]) == TUE_1100
    kept = load_prices("ABC", years=2, keep_partial=True, refresh=True)
    assert kept.index[-1] == pd.Timestamp("2026-10-13")


def test_cache_is_stale_once_a_session_closes(live_feed):
    load_prices("ABC", years=2)
    assert live_feed["calls"] == ["ABC"]
    again = load_prices("ABC", years=2)                             # same session: the cache holds every finished bar
    assert again.attrs["source"] == "cache" and live_feed["calls"] == ["ABC"]
    assert again.attrs["quote"] is None and again.attrs["partial_bar"] is None
    live_feed["clock"]["now"] = pd.Timestamp("2026-10-13 16:05", tz=ET)
    after = load_prices("ABC", years=2)                             # Tuesday has closed: its bar must be fetched
    assert after.attrs["source"] == "yfinance" and live_feed["calls"] == ["ABC", "ABC"]
    assert after.index[-1] == pd.Timestamp("2026-10-13") and after.attrs["partial_bar"] is None
    load_prices("ABC", years=2, refresh=True)                       # refresh always downloads
    assert live_feed["calls"] == ["ABC", "ABC", "ABC"]


def test_orders_are_priced_from_the_quote(live_feed):
    r = analyze("ABC", InvestorProfile.from_cash(10_000, 0), years=2, run_backtest_=False)
    assert r.partial_bar is not None and r.indicators.index[-1] == pd.Timestamp("2026-10-12")
    assert r.quote is not None and r.decision.price == pytest.approx(r.quote.price)
    assert r.decision.as_of == "2026-10-12"
    assert any("priced from the live quote" in w for w in r.warnings)
    pl = build_trade_plan(r, "long", with_stats=False)
    assert pl.price == pytest.approx(r.quote.price) and pl.priced_from.startswith("live quote $")
    hist = analyze("ABC", InvestorProfile.from_cash(10_000, 0), years=2, run_backtest_=False, as_of="2026-06-30")
    assert hist.quote is None                                       # the time machine never uses today's price


def test_quote_needs_a_time_and_partial_bars_use_the_exchange_calendar():
    assert _quote_from_meta("abc", {"regularMarketPrice": 10.0}) is None             # a price without a time
    tokyo = {"exchangeTimezoneName": "Asia/Tokyo",
             "currentTradingPeriod": {"regular": {"start": pd.Timestamp("2026-10-13 09:00", tz="Asia/Tokyo"),
                                                  "end": pd.Timestamp("2026-10-13 15:30", tz="Asia/Tokyo")}}}
    monday_evening_ny = pd.Timestamp("2026-10-12 21:00", tz=ET)                      # = Tuesday 10:00 in Tokyo
    assert is_partial_bar("2026-10-13", monday_evening_ny, tokyo)
    assert not is_partial_bar("2026-10-12", monday_evening_ny, tokyo)


def test_explicit_cache_source_ignores_the_session_rule(live_feed):
    load_prices("ABC", years=2)                                                     # cached at 11:00
    live_feed["clock"]["now"] = pd.Timestamp("2026-10-13 16:05", tz=ET)
    df = load_prices("ABC", years=2, source="cache")                                 # asked for: under 20 hours is fine
    assert df.attrs["source"] == "cache" and live_feed["calls"] == ["ABC"]
    assert not list(live_feed["cache"].glob("prices/.*.tmp"))                        # atomic writes leave no temp files


def test_no_extra_quote_request_after_a_fallback(live_feed, monkeypatch):
    import lighthouse.engine as engine
    load_prices("ABC", years=2)
    down = lambda *a: (_ for _ in ()).throw(data.DataUnavailable("network down"))   # noqa: E731
    monkeypatch.setattr(data, "_from_yfinance", down)
    monkeypatch.setattr(data, "_from_stooq", down)
    live_feed["clock"]["now"] = pd.Timestamp("2026-10-14 11:00", tz=ET)               # the cache is now behind a close
    asked = []
    monkeypatch.setattr(engine, "load_quote", lambda *a, **k: asked.append(a) or None)
    r = analyze("ABC", InvestorProfile.from_cash(10_000, 0), years=2, run_backtest_=False)
    assert r.prices.attrs["source"] == "stale-cache" and r.quote is None and asked == []
    # the benchmark fell back to the 2022 sample file, which does not overlap 2026: regime rules off, no crash
    assert r.regimes is None and any("Benchmark ^GSPC unavailable" in w and "requested window" in w for w in r.warnings)


def test_keep_partial_keeps_todays_bar_for_the_position_replay(live_feed):
    r = analyze("ABC", InvestorProfile.from_cash(10_000, 0), years=2, run_backtest_=False, keep_partial=True)
    assert r.indicators.index[-1] == pd.Timestamp("2026-10-13") and r.partial_bar is not None
    assert "kept but not final" in " ".join(r.warnings)
    assert r.prices.attrs["source"] == "yfinance" and "cleaning" in r.prices.attrs   # metadata is back on the result


def test_keep_partial_downloads_during_the_session_even_with_a_fresh_cache(live_feed):
    load_prices("ABC", years=2)                                   # the cache now holds bars through yesterday
    df = load_prices("ABC", years=2, keep_partial=True)
    assert df.index[-1] == pd.Timestamp("2026-10-13") and live_feed["calls"] == ["ABC", "ABC"]
    load_prices("^GSPC", years=2)
    bench = load_prices("^GSPC", years=None, start=df.index[0], end=df.index[-1], keep_partial=True)   # as analyze asks
    assert bench.index[-1] == pd.Timestamp("2026-10-13")
