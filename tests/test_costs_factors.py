"""Liquidity-scaled costs, resting stops that gap, cash at the risk-free rate, borrow fees, and the Ken French files
(recorded excerpts in tests/fixtures/french, October 2026)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lighthouse import factors as ff
from lighthouse.backtest import run_backtest
from lighthouse.config import InvestorProfile, StrategyParams
from lighthouse.costs import CostModel, corwin_schultz_spread, impact_bps
from conftest import make_prices

FIX = Path(__file__).parent / "fixtures" / "french"


def test_corwin_schultz_is_causal_and_zero_without_a_range():
    df = make_prices(400)
    s = corwin_schultz_spread(df["High"], df["Low"])
    assert (s.dropna() >= 0).all() and s.dropna().mean() > 0
    cut = corwin_schultz_spread(df["High"].iloc[:250], df["Low"].iloc[:250])
    pd.testing.assert_series_equal(s.iloc[:250], cut)                         # nothing after t is used
    flat = corwin_schultz_spread(df["Close"], df["Close"])                    # close-only data: no range, no spread
    assert flat.dropna().eq(0).all()


def test_impact_grows_with_the_square_root_of_size():
    a, b = impact_bps(1e5, 1e8, 0.02), impact_bps(4e5, 1e8, 0.02)
    assert b == pytest.approx(2 * a) and impact_bps(1e5, np.nan, 0.02) == 0.0
    c = CostModel().per_side_bps(make_prices(200), 1e4)
    assert (c.dropna() >= 10.0 - 1e-9).all()                                  # commission 5 + half-spread floor 5


def _gap_series():
    """Rises for a year and a half with a 3% daily range (the rule buys; the trailing stop sits ~10% below), then
    gaps down 36% overnight, far through the stop."""
    idx = pd.bdate_range("2015-01-01", periods=420)
    close = np.concatenate([np.linspace(50, 110, 400), np.full(20, 70.0)])
    open_ = np.concatenate([[50.0], close[:-1]])
    open_[400] = 70.0                                                          # the gap: the open is far below the stop
    df = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.015, "Low": np.minimum(open_, close) * 0.985,
                       "Close": close, "Volume": 1e6}, index=idx)
    df.attrs["ticker"] = "GAP"
    return df


def test_a_gap_through_a_resting_stop_fills_at_the_open():
    df = _gap_series()
    p = StrategyParams.for_horizon("short", cost_model="fixed", entry_score=50)
    bt = run_backtest(df, None, p, InvestorProfile(horizon="short"))
    stops = [t for t in bt.trades if t.exit_reason == "stop order"]
    assert stops, "the gap should hit the stop order"
    t = stops[-1]
    assert t.exit_date == df.index[400]
    assert t.exit_price == pytest.approx(70.0 * (1 - 0.001))                   # the open, less costs, not the stop price
    old = run_backtest(df, None, p.copy(resting_stops=False), InvestorProfile(horizon="short"))
    assert not any(x.exit_reason == "stop order" for x in old.trades)          # close-checked stops sell a day later


def test_cash_earns_the_risk_free_rate():
    df = make_prices(700)
    p = StrategyParams(entry_score=101.0)                                      # never enters: always in cash
    bt = run_backtest(df, None, p)
    years = (bt.equity.index[-1] - bt.equity.index[0]).days / 365.25
    assert bt.equity.iloc[-1] == pytest.approx((1 + 0.02 / 252) ** (len(bt.equity) - 1), rel=1e-9)
    assert 0.015 < bt.metrics["cagr"] < 0.025 and years > 1
    zero = run_backtest(df, None, p.copy(cash_yield=False))
    assert zero.equity.iloc[-1] == pytest.approx(1.0)
    tbill = pd.Series(0.05, index=df.index[::5])
    assert run_backtest(df, None, p, risk_free=tbill).equity.iloc[-1] > bt.equity.iloc[-1]


def test_portfolio_cash_buffer_earns_the_rate():
    from lighthouse.portfolio import backtest_weights
    rets = pd.DataFrame(0.0, index=pd.bdate_range("2020-01-01", periods=400), columns=["A", "B"])
    run = backtest_weights(rets, lambda t: (np.array([0.5, 0.5]), 0.5), lookback=60, cost_bps=0.0, risk_free_rate=0.04)
    days = len(run.equity) - 1
    assert run.equity.iloc[-1] == pytest.approx((1 + 0.5 * 0.04 / 252) ** days, rel=1e-3)
    flat = backtest_weights(rets, lambda t: (np.array([0.5, 0.5]), 0.5), lookback=60, cost_bps=0.0, risk_free_rate=0.04,
                            cash_yield=False)
    assert flat.equity.iloc[-1] == pytest.approx(1.0)


def test_borrow_fee_lowers_the_short_record():
    from lighthouse import analyze
    from lighthouse.tradeplan import setup_statistics
    r = analyze("XOM", InvestorProfile(horizon="short"), source="sample", years=10, run_backtest_=False)
    cheap = setup_statistics(r, "short")
    r.params = r.params.copy(short_borrow_fee=0.20)
    dear = setup_statistics(r, "short")
    assert dear.n_trades == cheap.n_trades and dear.avg_r < cheap.avg_r


def test_french_files_parse():
    secs = ff.parse_sections((FIX / "ff5_monthly_excerpt.csv").read_text())
    main = secs["main"]
    assert list(main.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert main.index[0] == pd.Timestamp("1963-07-31") and main.loc["1963-07-31", "Mkt-RF"] == pytest.approx(-0.0039)
    annual = next(v for k, v in secs.items() if "Annual" in k)
    assert annual.index[0] == pd.Timestamp("1964-12-31")
    mom = ff.parse_sections((FIX / "mom_monthly_excerpt.csv").read_text())
    assert "Mom" in [c.strip() for c in next(iter(mom.values())).columns]
    sic = ff.parse_sic_codes((FIX / "siccodes49_excerpt.txt").read_text())
    names = {n for _, _, n in sic}
    assert {"Agric", "Food"} <= names and all(lo <= hi for lo, hi, _ in sic)


def test_industry_file_first_table_is_found_by_its_title():
    """The 49-industry file opens with the value-weighted monthly table (recorded excerpt, data to August 2026)."""
    secs = ff.parse_sections((FIX / "ind49_monthly_excerpt.csv").read_text())
    vw = secs["Average Value Weighted Returns -- Monthly"]
    assert vw is secs["main"] and vw.shape[1] == 49
    assert vw.loc["1926-07-31", "Agric"] == pytest.approx(0.0236)
    assert np.isnan(vw.loc["1926-07-31", "Soda"])                      # -99.99 = missing
    ew = secs["Average Equal Weighted Returns -- Monthly"]
    assert ew.shape[1] == 49 and len(ew) >= 4
