"""Regression tests for the issues found in the code review (each test fails on the previous code)."""
import argparse
import json

import numpy as np
import pandas as pd
import pytest

from lighthouse import analyze, InvestorProfile, StrategyParams
from lighthouse import data as data_mod
from lighthouse.backtest import compute_metrics, trade_metrics, Trade
from lighthouse.decision import size_position


def test_sortino_uses_downside_deviation_over_all_periods():
    r = pd.Series([0.01, -0.02, 0.015, -0.005, 0.0, 0.02, -0.01, 0.005] * 40)
    eq = pd.Series((1 + r).cumprod().values, index=pd.bdate_range("2020-01-01", periods=len(r)))
    m = compute_metrics(eq, risk_free_rate=0.02)
    rets = eq.pct_change().dropna()
    dd = np.sqrt(np.mean(np.minimum(rets - 0.02 / 252, 0) ** 2)) * np.sqrt(252)
    assert m["sortino"] == pytest.approx((rets.mean() * 252 - 0.02) / dd)


def test_profit_factor_weights_trades_by_size():
    d = pd.Timestamp("2020-01-01")
    trades = [Trade(d, 100, d, 110, 1.0, 0.10, 0.10, 5, "x"), Trade(d, 100, d, 90, 0.2, -0.10, -0.02, 5, "x")]
    assert trade_metrics(trades)["profit_factor"] == pytest.approx(0.10 / 0.02)


def test_wide_stop_never_lets_the_loss_exceed_the_limit():
    prof = InvestorProfile(risk="aggressive", max_loss_pct=2.0, portfolio_value=100_000)
    # price 100, 200-day SMA 32.6 -> long-horizon stop at 30: a 70% stop distance
    plan = size_position(100.0, 2.0, 0.30, 85, "Strong Buy", prof, sma_slow=32.6, params=StrategyParams())
    assert plan.stop_price == pytest.approx(32.6 * 0.92)
    real_loss = plan.shares * (100.0 - plan.stop_price)
    assert real_loss <= prof.max_loss_value + 1e-6
    assert plan.max_loss_value == pytest.approx(real_loss)


def test_position_is_capped_by_the_cash_available():
    prof = InvestorProfile.from_cash(1_000, 99_000, risk="aggressive", max_loss_pct=5)
    plan = size_position(50.0, 1.0, 0.15, 90, "Strong Buy", prof, params=StrategyParams(atr_stop=True))
    assert plan.cost <= 1_000 and plan.binding_constraint == "cash available" and plan.shares == 20
    assert plan.cash_left == pytest.approx(1_000 - plan.cost)


def test_profile_cash_validation_and_description():
    with pytest.raises(ValueError):
        InvestorProfile(portfolio_value=1_000, cash_available=2_000)
    with pytest.raises(ValueError):
        InvestorProfile.from_cash(-1)
    p = InvestorProfile.from_cash(5_000, 20_000)
    assert p.account_value == 25_000 and p.cash == 5_000 and p.holdings_value == 20_000
    assert "$5,000 cash to invest in a $25,000 account" in p.describe()
    legacy = InvestorProfile(portfolio_value=10_000)
    assert legacy.cash == 10_000 and "$10,000 cash to invest" in legacy.describe()


def test_cli_short_horizon_uses_the_short_rule():
    from lighthouse.__main__ import _params, _profile
    a = argparse.Namespace(horizon="short", params=None, cash=3_000.0, holdings=7_000.0, risk="balanced", max_loss=2.0,
                           portfolio=10_000.0)
    p = _params(a)
    assert p.atr_stop and p.decision_frequency == "weekly"
    a.params = json.dumps({"entry_score": 65})
    assert _params(a).entry_score == 65 and _params(a).atr_stop
    prof = _profile(a)
    assert prof.cash == 3_000 and prof.account_value == 10_000


def test_fresh_cache_must_cover_the_requested_window(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "CACHE_DIR", tmp_path)
    idx = pd.bdate_range("2018-01-01", "2020-12-31")
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)
    df.index.name = "Date"
    data_mod._to_cache("ZZZ", df, requested_start=pd.Timestamp("2018-01-01"))
    assert len(data_mod._from_cache("ZZZ", need_start=pd.Timestamp("2018-01-03").to_pydatetime())) == len(idx)
    with pytest.raises(data_mod.DataUnavailable, match="after the requested"):
        data_mod._from_cache("ZZZ", need_start=pd.Timestamp("2012-01-01").to_pydatetime())
    assert len(data_mod._from_cache("ZZZ", max_age_hours=None)) == len(idx)       # stale fallback accepts anything


def test_dashboard_draws_the_band_centre_and_the_regime_shading():
    pytest.importorskip("plotly")
    from lighthouse import interactive as ix
    r = analyze("AAPL", source="sample", years=6)
    fig = ix.dashboard(r, last_days=252)
    centre = [t for t in fig.data if t.name and t.name == "Band centre (trend filter)"]
    assert len(centre) == 1 and not any(t.name and "Bollinger" in t.name for t in fig.data)
    assert np.allclose(np.asarray(centre[0].y, dtype=float), r.indicators["band_mid"].tail(252).values, equal_nan=True)
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert rects, "2022 was a bear market: the regime shading must be drawn"
    bt = ix.backtest(r.backtest, r.regimes)
    assert any(s.type == "rect" for s in bt.layout.shapes)
