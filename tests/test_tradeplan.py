"""Trade tickets (long and short), the shared trade simulator, the historical setup test, idea allocation and the
open-position monitor."""
import numpy as np
import pandas as pd
import pytest

from lighthouse import analyze, InvestorProfile
from lighthouse.data import sample_pack_tickers
from lighthouse.tradeplan import (TradeSettings, build_trade_plan, setup_statistics, allocate_ideas, monitor_position,
                                  _simulate, _split_shares, SHORT_MAX_SCORE)


@pytest.fixture(scope="module")
def msft_short_horizon():
    prof = InvestorProfile.from_cash(5_000, 20_000, horizon="short", risk="balanced", max_loss_pct=1.0)
    return analyze("MSFT", prof, source="sample", years=10)


@pytest.fixture(scope="module")
def sample_results():
    prof = InvestorProfile.from_cash(5_000, 20_000, horizon="short", risk="balanced", max_loss_pct=1.0)
    tickers = [t for t in sample_pack_tickers() if not t.startswith("^")]
    return [analyze(t, prof, source="sample", years=6, run_backtest_=False) for t in tickers]


# --------------------------------------------------------------------------- ticket
def test_long_ticket_matches_the_position_plan(msft_short_horizon):
    r = msft_short_horizon
    pl = build_trade_plan(r, "long", with_stats=False)
    assert r.decision.verdict in {"Buy", "Strong Buy"} and pl.status == "actionable"
    assert pl.shares == r.decision.plan.shares and pl.stop == pytest.approx(r.decision.plan.stop_price)
    assert pl.notional <= r.profile.cash + 1e-9                                  # never spends more than the cash
    assert pl.dollars_at_risk <= r.profile.max_loss_value + 1e-6                 # never risks more than the limit
    assert pl.stop < pl.price < pl.entry_limit
    m1, m2 = TradeSettings().reward_multiples
    assert pl.targets[0].price - pl.price == pytest.approx(m1 * pl.risk_per_share)
    assert pl.targets[1].price - pl.price == pytest.approx(m2 * pl.risk_per_share)
    assert sum(t.shares for t in pl.targets) == pl.shares
    steps = " ".join(pl.management_steps())
    assert "stop order" in steps and "breakeven" in steps and "Time stop" in steps


def test_short_ticket_mirrors_the_long_one(sample_results):
    shorts = [build_trade_plan(r, "short", with_stats=False) for r in sample_results]
    act = [p for p in shorts if p.status == "actionable"]
    assert act, "the 2022 bear market in the sample pack should offer at least one short"
    for pl in act:
        assert pl.composite_score <= SHORT_MAX_SCORE
        assert pl.stop > pl.price > pl.entry_limit
        assert all(t.price < pl.price for t in pl.targets)
        assert pl.notional <= pl.cash_available + 1e-9 and pl.dollars_at_risk <= 0.01 * pl.account_value + 1e-6
        if pl.shares:
            assert pl.margin["reg_t_initial"] == pytest.approx(0.5 * pl.notional)
            assert pl.margin["finra_maintenance"] == pytest.approx(max(0.3 * pl.notional, 5.0 * pl.shares)) if pl.price >= 5 else True
        assert any("margin account" in w for w in pl.warnings)


def test_short_is_blocked_in_a_calm_bull_market(msft_short_horizon):
    pl = build_trade_plan(msft_short_horizon, "short", with_stats=False)
    assert msft_short_horizon.decision.regime.label == "Bull / Calm"
    assert pl.status == "avoid" and pl.shares == 0
    assert pl.management_steps()[0].startswith("No position")


def test_split_shares():
    assert _split_shares(0, 0.5) == (0, 0)
    assert _split_shares(1, 0.5) == (1, 0)
    assert _split_shares(10, 0.5) == (5, 5)
    assert _split_shares(3, 0.5) == (2, 1)
    assert _split_shares(10, 0.0) == (0, 10)
    assert _split_shares(10, 1.0) == (10, 0)


def test_settings_validation():
    with pytest.raises(ValueError):
        TradeSettings(reward_multiples=(3, 2))
    with pytest.raises(ValueError):
        TradeSettings(partial_exit=1.5)
    with pytest.raises(ValueError):
        TradeSettings(time_stop_days=0)


# --------------------------------------------------------------------------- simulator
def _bars(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.full(len(a), 1.0)


def test_simulator_stop_gap_target_and_time_stop():
    # entry 100 at bar 0, stop 95, targets 110 / 115
    o, h, l, c, atr = _bars([[100, 101, 99, 100], [100, 101, 94, 96]])
    sim = _simulate(o, h, l, c, atr, 0, 100, 95, 110, 115, 0.5, 2.0, 40, True, True)
    assert sim["exits"][0][2] == "stop" and sim["exits"][0][1] == 95          # intraday stop fills at the stop
    o, h, l, c, atr = _bars([[100, 101, 99, 100], [90, 92, 89, 91]])
    sim = _simulate(o, h, l, c, atr, 0, 100, 95, 110, 115, 0.5, 2.0, 40, True, True)
    assert sim["exits"][0][1] == 90                                            # gap through the stop fills at the open
    o, h, l, c, atr = _bars([[100, 101, 99, 100], [100, 111, 94, 105]])
    sim = _simulate(o, h, l, c, atr, 0, 100, 95, 110, 115, 0.5, 2.0, 40, True, True)
    assert sim["exits"][0][2] == "stop"                                        # stop and target in one bar: stop first
    o, h, l, c, atr = _bars([[100, 101, 99, 100], [100, 111, 99, 110], [110, 111, 99, 100]])
    sim = _simulate(o, h, l, c, atr, 0, 100, 95, 110, 115, 0.5, 50.0, 40, True, True)    # trail too wide to matter
    assert [x[2] for x in sim["exits"]] == ["target 1", "breakeven/trailing stop"]
    assert sim["exits"][0][0] == pytest.approx(0.5) and sim["exits"][1][1] == pytest.approx(100.0)   # breakeven
    sim = _simulate(o, h, l, c, atr, 0, 100, 95, 110, 115, 0.5, 2.0, 40, True, True)     # trail: 110 - 2 x ATR(1)
    assert sim["exits"][1][1] == pytest.approx(108.0)
    o, h, l, c, atr = _bars([[100, 101, 99, 100], [100, 111, 99, 110], [110, 116, 109, 115]])
    sim = _simulate(o, h, l, c, atr, 0, 100, 95, 110, 115, 0.5, 5.0, 40, True, True)
    assert [x[2] for x in sim["exits"]] == ["target 1", "target 2"]
    o, h, l, c, atr = _bars([[100, 101, 99, 100]] * 5)
    sim = _simulate(o, h, l, c, atr, 0, 100, 95, 110, 115, 0.5, 2.0, 3, True, True)
    assert sim["exits"][-1][2] == "time stop" and sim["exits"][-1][3] == 2 and sim["remaining"] == 0
    sim = _simulate(o, h, l, c, atr, 0, 100, 95, 110, 115, 0.5, 2.0, 40, True, True)
    assert sim["remaining"] == 1.0                                             # data ends before the time stop: still open


def test_simulator_handles_shorts_by_negation():
    # short at 100, stop 105, target 90: price falls to 89
    o, h, l, c, atr = _bars([[100, 101, 99, 100], [98, 99, 89, 90]])
    sim = _simulate(-o, -l, -h, -c, atr, 0, -100, -105, -90, None, 0.5, 2.0, 40, True, True)
    assert sim["exits"][0][2] == "target 1" and abs(sim["exits"][0][1]) == 90
    gain_r = sum(f * (px - (-100)) for f, px, _, _ in sim["exits"]) / 5.0
    assert gain_r == pytest.approx(2.0)


# --------------------------------------------------------------------------- historical setup test
def test_setup_statistics_partition_and_no_lookahead(msft_short_horizon):
    st = setup_statistics(msft_short_horizon, "long")
    assert st.n_trades >= 20 and st.reliable
    assert st.hit_target1 + st.stopped_out + st.timed_out == pytest.approx(1.0)
    tr = st.trades
    assert (tr["entry_date"] > tr["signal_date"]).all() and (tr["exit_date"] >= tr["entry_date"]).all()
    assert tr["entry_date"].is_monotonic_increasing
    assert (tr["entry_date"].iloc[1:].values >= tr["exit_date"].iloc[:-1].values).all()     # one trade at a time
    # trades that closed before a cut date are identical when the future is removed
    cut = tr["exit_date"].iloc[len(tr) // 2]
    r2 = analyze("MSFT", msft_short_horizon.profile, source="sample", years=10, as_of=str(cut.date()))
    st2 = setup_statistics(r2, "long")
    common = tr[tr["exit_date"] < cut - pd.Timedelta(days=5)]
    early = st2.trades.set_index("entry_date").reindex(common["entry_date"])
    assert np.allclose(early["r"].values, common["r"].values)


# --------------------------------------------------------------------------- ideas
def test_allocate_ideas_respects_cash_heat_and_count(sample_results):
    s = TradeSettings(max_ideas=3, max_total_risk_pct=2.0)
    for side in ("long", "short"):
        funded, watch, table = allocate_ideas(sample_results, side, s)
        prof = sample_results[0].profile
        assert len(funded) <= 3
        assert sum(p.notional for p in funded) <= prof.cash + 1e-6
        assert sum(p.dollars_at_risk for p in funded) <= 0.02 * prof.account_value + 1e-6
        assert all(p.status == "actionable" and p.shares > 0 for p in funded)
        assert all(p.status == "watch" for p in watch)
        if funded:
            assert set(table["ticker"]) >= {p.ticker for p in funded}


# --------------------------------------------------------------------------- monitor
def test_monitor_reports_stop_and_targets(msft_short_horizon):
    r = msft_short_horizon
    ms = monitor_position(r, "long", 73.9, 30, "2017-09-01")       # MSFT ran from ~74 to ~84 by Nov 2017
    assert ms.status == "TARGET 2 HIT" and all(hit for _, _, hit in ms.targets)
    assert ms.pnl == pytest.approx((ms.last_price - 73.9) * 30)
    ms = monitor_position(r, "long", 69.0, 30, "2017-05-01")       # the 2017-05-15 low (66.83) hit the plan stop
    assert ms.status == "STOP HIT" and "2017-05-15" in ms.action
    ms = monitor_position(r, "short", 74.0, 10, "2017-09-01")
    assert ms.status == "STOP HIT" and ms.r_multiple < 0
    assert any("turned against this short" in a for a in ms.alerts)
    ms = monitor_position(r, "long", 83.0, 10, "2017-11-09", stop=82.5)   # 2017-11-10 high 84.10 reaches the 2R target 84.00
    assert ms.status == "TARGET 1 HIT" and ms.current_stop >= 83.0          # stop moved to breakeven or better
    ms = monitor_position(r, "long", 84.5, 10, "2017-11-09", stop=83.1, targets=[95, 100],
                          settings=TradeSettings(alert_atr=1.0))            # last close 83.87 is < 1 ATR above 83.10
    assert ms.status == "NEAR STOP" and ms.current_stop == pytest.approx(83.1)
    ms = monitor_position(r, "long", 84.5, 10, "2017-11-09", stop=80.0, targets=[95, 100])
    assert ms.status == "OK" and ms.distance_to_stop_pct > 0
    ms = monitor_position(r, "long", 80.0, 20, "2017-09-01")
    assert any("outside the 2017-09-01 price range" in a for a in ms.alerts)


def test_monitor_validates_inputs(msft_short_horizon):
    r = msft_short_horizon
    with pytest.raises(ValueError):
        monitor_position(r, "long", 80.0, 10, "2017-09-01", stop=85.0)       # stop above a long's entry
    with pytest.raises(ValueError):
        monitor_position(r, "short", 80.0, 10, "2017-09-01", targets=[85.0])  # short target above entry
    with pytest.raises(ValueError):
        monitor_position(r, "long", 80.0, 10, "2030-01-01")                  # entry after the data
    with pytest.raises(ValueError):
        monitor_position(r, "sideways", 80.0, 10, "2017-09-01")
