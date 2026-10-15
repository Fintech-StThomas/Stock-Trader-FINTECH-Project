"""Trade tickets (long only), their exit plans, the shared trade simulator, the historical setup test, idea
allocation and the open-position replay."""
import numpy as np
import pandas as pd
import pytest

from lighthouse import analyze, InvestorProfile
from lighthouse.data import sample_pack_tickers
from lighthouse.tradeplan import (TradeSettings, build_trade_plan, setup_statistics, allocate_ideas, monitor_position,
                                  review_days, _simulate, _simulate_trend, _split_shares)


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
    funded, watch, table = allocate_ideas(sample_results, "long", s)
    prof = sample_results[0].profile
    assert len(funded) <= 3
    assert sum(p.notional for p in funded) <= prof.cash + 1e-6
    assert sum(p.dollars_at_risk for p in funded) <= 0.02 * prof.account_value + 1e-6
    assert all(p.status == "actionable" and p.shares > 0 for p in funded)
    assert all(p.status == "watch" for p in watch)
    if funded:
        assert set(table["ticker"]) >= {p.ticker for p in funded}
    with pytest.raises(ValueError, match="short tickets were removed"):
        allocate_ideas(sample_results, "short", s)


# --------------------------------------------------------------------------- exit plan
def test_exit_plan_gives_every_way_out_with_the_whole_positions_result(msft_short_horizon):
    r = msft_short_horizon
    pl = build_trade_plan(r, "long")
    rules = {x.name: x for x in pl.exit_plan()}
    assert [x.kind for x in pl.exit_plan()] == ["loss", "profit", "profit", "profit", "time", "evidence"]
    stop, t1, t2 = rules["Stop-loss (before target 1)"], rules["Target 1: partial profit"], rules["Target 2: the rest"]
    risk, n1, n2 = pl.risk_per_share, pl.targets[0].shares, pl.targets[1].shares
    assert stop.price == pl.stop and stop.r == -1.0 and stop.pnl == pytest.approx(-pl.dollars_at_risk)
    assert t1.pnl == pytest.approx(n1 * 2 * risk) and t1.r == pytest.approx(n1 * 2 / pl.shares)
    assert t2.pnl == pytest.approx(n1 * 2 * risk + n2 * 3 * risk) == pytest.approx(pl.profit_if_all_targets)
    assert rules["Time stop: 40 sessions"].pnl is None and "week" in rules["Verdict turns"].action
    # the history column counts the setup test's trades by how they ended
    st = pl.stats
    first = st.trades["outcome"].str.split(" + ", regex=False).str[0]
    assert stop.history.startswith(f"{int((first == 'stop').sum())} of {st.n_trades} past setups")
    one = TradeSettings(partial_exit=0.3)
    tiny = build_trade_plan(r, "long", one, cash=1.0, with_stats=False)   # $1 buys no share: a plan per unit
    assert tiny.shares == 0
    t1_unit = next(x for x in tiny.exit_plan() if x.name.startswith("Target 1"))
    assert t1_unit.pnl is None and t1_unit.r == pytest.approx(0.3 * 2)    # per unit, with the configured partial


# --------------------------------------------------------------------------- monitor
def test_monitor_reports_stop_and_targets(msft_short_horizon):
    r = msft_short_horizon
    ms = monitor_position(r, "long", 73.9, 30, "2017-09-01")       # MSFT ran from ~74 to ~84 by Nov 2017
    assert ms.status == "TARGET 2 HIT" and all(hit for _, _, hit in ms.targets)
    # half sold at target 1, the rest at target 2 (filled at the open after the 2017-10-27 earnings gap): P&L is
    # realised on those fills, not marked to the last price as if all 30 shares were still held
    assert [(n, why) for _, n, _, why in ms.exits] == [(15, "target 1"), (15, "target 2")]
    assert ms.exits[0][2] == pytest.approx(ms.targets[0][1]) and ms.exits[1][2] >= ms.targets[1][1]
    assert ms.pnl == pytest.approx(sum(n * (px - 73.9) for _, n, px, _ in ms.exits)) and ms.shares_open == 0
    assert ms.unrealised_pnl == 0 and ms.pnl != pytest.approx((ms.last_price - 73.9) * 30)
    risk = (ms.targets[0][1] - 73.9) / 2
    assert ms.r_multiple == pytest.approx(ms.pnl / (30 * risk))
    ms = monitor_position(r, "long", 69.0, 30, "2017-05-01")       # the 2017-05-15 low (66.83) hit the plan stop
    assert ms.status == "STOP HIT" and "2017-05-15" in ms.action
    assert ms.r_multiple == pytest.approx(-1.0) and ms.pnl == pytest.approx(30 * (ms.initial_stop - 69.0))
    ms = monitor_position(r, "long", 83.0, 10, "2017-11-09", stop=82.5)   # 2017-11-10 high 84.10 reaches the 2R target 84.00
    assert ms.status == "TARGET 1 HIT" and ms.current_stop >= 83.0          # stop moved to breakeven or better
    assert ms.shares_open == 5 and ms.realised_pnl == pytest.approx(5 * 1.0)  # 5 sold at 84.00
    assert ms.unrealised_pnl == pytest.approx(5 * (ms.last_price - 83.0))
    assert ms.r_multiple == pytest.approx((ms.realised_pnl + ms.unrealised_pnl) / (10 * 0.5))
    one = monitor_position(r, "long", 83.0, 1, "2017-11-09", stop=82.5)   # a single share cannot be split
    assert one.status == "TARGET 1 HIT" and one.shares_open == 0 and one.pnl == pytest.approx(1.0)
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
        monitor_position(r, "long", 80.0, 10, "2017-09-01", targets=[75.0])   # target below the entry
    with pytest.raises(ValueError, match="short tickets were removed"):
        monitor_position(r, "short", 80.0, 10, "2017-09-01")
    with pytest.raises(ValueError):
        monitor_position(r, "long", 80.0, 10, "2030-01-01")                  # entry after the data
    with pytest.raises(ValueError):
        monitor_position(r, "sideways", 80.0, 10, "2017-09-01")


def test_exit_plan_and_replay_with_no_partial_sale(msft_short_horizon):
    r = msft_short_horizon
    s0 = TradeSettings(partial_exit=0.0)
    pl = build_trade_plan(r, "long", s0, with_stats=False)
    names = [x.name for x in pl.exit_plan()]
    assert "Target 1: stop to breakeven" in names and "Target 1: partial profit" not in names
    t1 = next(x for x in pl.exit_plan() if x.name == "Target 1: stop to breakeven")
    assert t1.pnl == 0.0 and t1.r == 0.0 and "Sell nothing" in t1.action
    ms = monitor_position(r, "long", 83.0, 10, "2017-11-09", stop=82.5, settings=s0)
    assert ms.status == "TARGET 1 HIT" and ms.shares_open == 10 and "keep all 10 shares" in ms.action
    none = build_trade_plan(r, "long", TradeSettings(partial_exit=0.3), cash=1.0, with_stats=False)
    t1s = next(x for x in none.exit_plan() if x.name.startswith("Target 1"))
    assert "30% of the position" in t1s.action                     # not "half"


def test_screen_accepts_a_generator():
    from lighthouse.screener import screen
    df = screen((t for t in ["MSFT", "KO"]), source="sample", years=5, progress=False)
    assert list(df["ticker"]) and set(df["ticker"]) == {"MSFT", "KO"}


# --------------------------------------------------------------------------- long-horizon trend plan
@pytest.fixture(scope="module")
def msft_long_horizon():
    prof = InvestorProfile.from_cash(25_000, horizon="long", risk="balanced")
    return analyze("MSFT", prof, source="sample", years=15)


def test_long_horizon_buy_follows_the_trend_plan(msft_long_horizon):
    r = msft_long_horizon
    pl = build_trade_plan(r, "long")
    assert pl.is_trend_plan and pl.status == "actionable" and pl.shares > 0
    assert pl.targets == [] and pl.time_stop_days is None
    sma = float(r.indicators["sma_slow"].iloc[-1])
    assert pl.trend_exit_level == pytest.approx(sma * (1 - r.params.exit_below_sma))
    assert pl.signal_exit_score == r.params.exit_score and pl.review == "month"
    names = [x.name for x in pl.exit_plan()]
    assert names == ["Stop-loss (stop order)", "Trend exit (review)", "Signal exit (review)"]
    steps = " ".join(pl.management_steps())
    assert "No fixed profit target" in steps and "month's last close" in steps and "stop order" in steps
    st = pl.stats
    assert st.exit_style == "trend" and st.hit_target1 == 0.0
    assert st.stopped_out + st.timed_out + st.trend_exited == pytest.approx(1.0)
    assert set(st.trades["outcome"]) <= {"stop", "trend exit", "signal exit"}
    assert all("average" in x.history for x in pl.exit_plan()[1:] if not x.history.startswith("never"))
    # the R-target plan stays available
    old = build_trade_plan(r, "long", TradeSettings(long_exit="targets"), with_stats=False)
    assert not old.is_trend_plan and len(old.targets) == 2 and old.time_stop_days == 252
    with pytest.raises(ValueError):
        TradeSettings(long_exit="moon")


def test_trend_simulator_stop_review_exits_and_pending():
    o, h, l, c, _ = _bars([[100, 101, 99, 100]] * 6)
    level, tech = np.full(6, 90.0), np.full(6, 70.0)
    review = np.array([False, False, True, False, False, True])
    sim = _simulate_trend(o, h, l, c, 0, 95.0, level, tech, review, 35.0, None, True, True)
    assert sim["remaining"] == 1.0 and sim["pending"] is None and sim["exits"] == []        # on plan, still open
    lvl = level.copy(); lvl[2] = 100.5                       # review-day close (100) below the trend level
    sim = _simulate_trend(o, h, l, c, 0, 95.0, lvl, tech, review, 35.0, None, True, True)
    assert sim["exits"] == [(1.0, 100.0, "trend exit", 3)]   # sold at the next open
    lvl[2] = 90.0; lvl[3] = 100.5                            # not a review day: ignored
    assert _simulate_trend(o, h, l, c, 0, 95.0, lvl, tech, review, 35.0, None, True, True)["exits"] == []
    t = tech.copy(); t[5] = 30.0                             # signal exit decided on the last bar: pending
    sim = _simulate_trend(o, h, l, c, 0, 95.0, level, t, review, 35.0, None, True, True)
    assert sim["exits"] == [] and sim["pending"] == "signal exit"
    o2, h2, l2, c2, _ = _bars([[100, 101, 99, 100], [100, 101, 94, 96], [96, 97, 95, 96]])
    sim = _simulate_trend(o2, h2, l2, c2, 0, 95.0, np.full(3, 90.0), np.full(3, 70.0), np.ones(3, bool), 35.0, None, True, True)
    assert sim["exits"] == [(1.0, 95.0, "stop", 1)]          # the stop order fills intraday
    sim = _simulate_trend(o, h, l, c, 0, 95.0, level, tech, review, 35.0, 4, True, True)
    assert sim["exits"] == [(1.0, 100.0, "time stop", 3)]


def test_review_days_are_period_ends():
    idx = pd.bdate_range("2024-01-24", "2024-03-05")
    m = review_days(idx, "monthly")
    assert [d.strftime("%Y-%m-%d") for d in idx[m]] == ["2024-01-31", "2024-02-29"]   # 2024-03-05 is not a month end
    assert review_days(pd.bdate_range("2024-01-24", "2024-01-31"), "monthly")[-1]       # ... but 2024-01-31 is
    assert review_days(idx, "daily").all()
    assert idx[review_days(idx, "weekly")][0].strftime("%A") == "Friday"


def test_position_replay_reproduces_a_historical_trend_exit(msft_long_horizon):
    r = msft_long_horizon
    tr = setup_statistics(r, "long").trades
    t = tr[tr["outcome"] == "trend exit"].iloc[0]
    ms = monitor_position(r, "long", float(t.entry), 10, t.entry_date)
    assert ms.status == "TREND EXIT" and ms.shares_open == 0 and len(ms.exits) == 1
    day, n, px, why = ms.exits[0]
    assert why == "trend exit" and day == str(t.exit_date.date()) and n == 10
    assert ms.r_multiple == pytest.approx(t.r + (r.params.cost_bps + r.params.slippage_bps) / 1e4 * (t.entry + px)
                                          / (t.entry - t.stop), abs=1e-6)          # the setup test's R before costs
    fresh = monitor_position(r, "long", 80.0, 10, "2017-10-02")
    assert fresh.status in {"OK", "NEAR STOP"} and "200-day SMA" in fresh.action and fresh.targets == []
