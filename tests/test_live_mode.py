"""The live-mode rule: no trade ticket from sample, stale or past data without an explicit override."""
import pandas as pd
import pytest

from lighthouse import analyze, InvestorProfile
from lighthouse import data as data_mod
from lighthouse.data import is_trading_day, last_session_close, nyse_holidays, sessions_behind
from lighthouse.tradeplan import (ALLOW_STALE_ENV, StaleDataError, allocate_ideas, build_trade_plan, data_freshness,
                                  require_current_data)

ET = "America/New_York"


@pytest.fixture(scope="module")
def msft():
    return analyze("MSFT", InvestorProfile.from_cash(5_000, 20_000, horizon="short"), source="sample", years=6,
                   run_backtest_=False)


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.delenv(ALLOW_STALE_ENV, raising=False)


def test_sample_data_is_refused_without_the_override(msft, strict):
    ok, why = data_freshness(msft)
    assert not ok and "sample" in why
    with pytest.raises(StaleDataError, match="no trade ticket"):
        build_trade_plan(msft, "long", with_stats=False)
    with pytest.raises(StaleDataError):
        allocate_ideas([msft], "long")


def test_the_override_is_explicit_and_stamped_on_the_ticket(msft, strict, monkeypatch):
    pl = build_trade_plan(msft, "long", with_stats=False, allow_stale=True)
    assert pl.data_status.startswith("OVERRIDE") and pl.warnings[0].startswith("Not for trading")
    monkeypatch.setenv(ALLOW_STALE_ENV, "1")
    assert build_trade_plan(msft, "long", with_stats=False).data_status.startswith("OVERRIDE")
    with pytest.raises(StaleDataError):                        # an explicit False wins over the environment
        build_trade_plan(msft, "long", with_stats=False, allow_stale=False)


class _Result:
    def __init__(self, source, last, ticker="ABC"):
        idx = pd.bdate_range(end=last, periods=5)
        self.prices = pd.DataFrame({"Close": 1.0}, index=idx)
        self.prices.attrs = {"source": source}
        self.ticker = ticker


def test_freshness_rules():
    tue_evening = pd.Timestamp("2026-10-13 17:00", tz=ET)
    assert data_freshness(_Result("yfinance", "2026-10-13"), tue_evening)[0]
    assert data_freshness(_Result("cache", "2026-10-13"), tue_evening)[0]            # a fresh cache holds the last close
    ok, why = data_freshness(_Result("yfinance", "2026-10-09"), tue_evening)          # a past date: two sessions behind
    assert not ok and "2 finished trading sessions" in why
    assert not data_freshness(_Result("stale-cache", "2026-10-13"), tue_evening)[0]
    assert not data_freshness(_Result("sample", "2022-12-28"), tue_evening)[0]
    # Thanksgiving: on the Friday morning after it the last finished session is Wednesday's
    fri = pd.Timestamp("2026-11-27 10:00", tz=ET)
    assert data_freshness(_Result("yfinance", "2026-11-25"), fri)[0]
    with pytest.raises(StaleDataError):
        require_current_data(_Result("yfinance", "2026-10-09"), allow_stale=False, now=tue_evening)


def test_allocation_skips_stale_tickers_but_needs_one_current(msft, strict, monkeypatch):
    stale = _Result("stale-cache", "2022-12-28", "OLD")
    with pytest.raises(StaleDataError, match="1 more ticker"):
        allocate_ideas([msft, stale], "long")


def test_nyse_calendar():
    h = {d.date() for d in nyse_holidays(2026, 2026)}
    assert pd.Timestamp("2026-04-03").date() in h            # Good Friday
    assert pd.Timestamp("2026-07-03").date() in h            # July 4th on a Saturday: observed Friday
    assert pd.Timestamp("2026-10-12").date() not in h        # Columbus Day: the exchange is open
    assert not is_trading_day("2026-12-25") and is_trading_day("2026-12-24")
    assert last_session_close(pd.Timestamp("2026-07-06 09:00", tz=ET)) == pd.Timestamp("2026-07-02 16:00", tz=ET)
    assert sessions_behind("2026-07-02", pd.Timestamp("2026-07-06 09:00", tz=ET)) == 0
    assert sessions_behind("2026-06-30", pd.Timestamp("2026-07-06 09:00", tz=ET)) == 2


def test_nyse_calendar_knows_the_one_off_closures():
    from lighthouse.data import NYSE_SPECIAL_CLOSURES
    h = {d.date() for d in nyse_holidays(1990, 2026)}
    for day in ("1994-04-27", "2001-09-11", "2001-09-14", "2004-06-11", "2007-01-02", "2012-10-29", "2012-10-30",
                "2018-12-05", "2025-01-09"):
        assert pd.Timestamp(day).date() in h and not is_trading_day(day), day
    assert all(pd.Timestamp(d).weekday() < 5 for d in NYSE_SPECIAL_CLOSURES)       # each one closed a weekday
    assert is_trading_day("2001-09-17") and is_trading_day("2012-10-31") and is_trading_day("2025-01-10")
    # the market clock on the day of mourning for President Carter: the last finished session is the 8th, and a bar
    # of the 8th is current on the morning of the 10th (the old calendar counted the 9th as a missing session)
    assert last_session_close(pd.Timestamp("2025-01-09 17:00", tz=ET)) == pd.Timestamp("2025-01-08 16:00", tz=ET)
    assert sessions_behind("2025-01-08", pd.Timestamp("2025-01-10 09:00", tz=ET)) == 0
    assert sessions_behind("2025-01-08", pd.Timestamp("2025-01-10 17:00", tz=ET)) == 1
    # nothing changes once the closure has passed: the cache expires at the same closes as before
    assert last_session_close(pd.Timestamp("2025-01-13 09:00", tz=ET)) == pd.Timestamp("2025-01-10 16:00", tz=ET)


def test_cli_refuses_and_accepts_the_flag(capsys, strict):
    from lighthouse.__main__ import main
    assert main(["trade", "MSFT", "--source", "sample", "--horizon", "short", "--cash", "5000", "--years", "6"]) == 2
    assert "no trade ticket" in capsys.readouterr().err
    assert main(["trade", "MSFT", "--source", "sample", "--horizon", "short", "--cash", "5000", "--years", "6",
                 "--allow-stale-data"]) == 0
