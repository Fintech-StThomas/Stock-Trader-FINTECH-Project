"""The forward journal (append-only), virtual positions, the scorecard and the Alpaca paper adapter, offline."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from lighthouse import analyze, InvestorProfile
from lighthouse.broker import AlpacaPaper, PAPER_URL, mirror, reconcile, review_exit_order, ticket_orders
from lighthouse.journal import Journal, check_positions, decision_row, record_run
from lighthouse.scorecard import brier, brier_skill, build_scorecard, calibration, score_verdicts
from lighthouse.tradeplan import build_trade_plan, TradeSettings


@pytest.fixture(scope="module")
def past_and_now():
    prof = InvestorProfile.from_cash(10_000, 30_000, horizon="long", risk="balanced")
    then = analyze("MSFT", prof, source="sample", years=10, as_of="2014-01-31", run_backtest_=False)   # a Strong Buy
    now = analyze("MSFT", prof, source="sample", years=10, run_backtest_=False)
    return then, now


def test_journal_is_append_only(tmp_path, past_and_now):
    then, _ = past_and_now
    j = Journal(tmp_path / "j.sqlite")
    pl = build_trade_plan(then, "long", allow_stale=True)
    did = j.append_decision(decision_row(then, pl, run_id="t1"))
    j.append_event(did, "MSFT", "alert", date="2016-05-02", detail={"x": 1})
    con = sqlite3.connect(str(j.path))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE decisions SET verdict='Sell' WHERE id=?", (did,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM events")
    con.close()
    d = j.decisions().iloc[0]
    assert d["ticker"] == "MSFT" and d["as_of"] == "2014-01-31" and d["verdict"] == then.decision.verdict
    inputs, ver = json.loads(d["inputs"]), json.loads(d["data_version"])
    assert inputs["profile"]["horizon"] == "long" and "trend" in inputs["beacons"]
    assert ver["last_bar"] == "2014-01-31" and ver["price_source"] == "sample" and len(ver["closes_sha1"]) == 16
    assert d["code_version"].startswith("lighthouse ")
    paths = j.export(tmp_path / "out")
    assert all(p.exists() for p in paths) and len(pd.read_csv(paths[0])) == 1


def test_stale_data_is_never_journaled(tmp_path, past_and_now, monkeypatch):
    monkeypatch.delenv("LIGHTHOUSE_ALLOW_STALE_DATA", raising=False)
    j = Journal(tmp_path / "j.sqlite")
    rep = record_run([past_and_now[0]], ["long"], j)
    assert rep.recorded == 0 and "MSFT" in rep.refused and j.decisions().empty


def test_virtual_position_opens_and_its_exit_is_recorded(tmp_path, past_and_now):
    then, now = past_and_now
    j = Journal(tmp_path / "j.sqlite")
    rep = record_run([then], ["long"], j, allow_stale=True)
    pl = build_trade_plan(then, "long", allow_stale=True)
    assert rep.recorded == 1 and pl.is_actionable and pl.is_trend_plan
    assert len(rep.opened) == 1 and len(j.open_positions()) == 1
    again = record_run([then], ["long"], j, allow_stale=True)       # one open position per ticker and side
    assert again.opened == []
    checks = check_positions({"MSFT": now}, j, run_id="t2")
    # bought at the open of 3 Feb 2014; the replay with the ticket's rules ends it at a month-end trend review
    assert [(c["kind"], c["status"]) for c in checks] == [("exit", "TREND EXIT")]
    ev = j.events()
    ex = ev[ev["kind"] == "exit"].iloc[0]
    assert json.loads(ex["detail"])["entry_date"] == "2014-02-03" and pd.Timestamp(ex["date"]) > pd.Timestamp("2014-02-03")
    assert j.open_positions().empty
    assert check_positions({"MSFT": now}, j, run_id="t3") == []        # closed: nothing more to record
    # the closed trade is scored against its forecast
    sc = build_scorecard(j, lambda t: now.prices["Close"] if t == "MSFT" else None)
    assert len(sc.tickets) == 1 and sc.tickets["p_win"].notna().all() and sc.summary()["closed_tickets"] == 1


def test_an_open_position_stays_open(tmp_path):
    prof = InvestorProfile.from_cash(10_000, 30_000, horizon="long", risk="balanced")
    then = analyze("MSFT", prof, source="sample", years=10, as_of="2016-07-29", run_backtest_=False)
    now = analyze("MSFT", prof, source="sample", years=10, run_backtest_=False)
    j = Journal(tmp_path / "j.sqlite")
    record_run([then], ["long"], j, allow_stale=True)
    assert check_positions({"MSFT": now}, j) == [] and len(j.open_positions()) == 1


def test_scorecard_math():
    assert brier([1.0, 0.0], [1, 0]) == 0.0 and brier([0.5, 0.5], [1, 0]) == 0.25
    assert brier_skill([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]) > 0 > brier_skill([0.1, 0.9, 0.2, 0.8], [1, 0, 1, 0])
    cal = calibration([0.15, 0.18, 0.85, 0.9], [0, 0, 1, 1], bins=5)
    assert cal["n"].tolist() == [2, 2] and cal["observed"].tolist() == [0.0, 1.0]


def test_verdicts_are_scored_only_after_their_window(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=400)
    stock = pd.Series(np.linspace(100, 200, 400), index=idx)
    spy = pd.Series(np.linspace(100, 120, 400), index=idx)
    closes = {"ABC": stock, "SPY": spy}.get
    d = pd.DataFrame([{"id": 1, "ticker": "ABC", "side": "long", "as_of": str(idx[10].date()), "verdict": "Buy", "horizon_days": 21},
                      {"id": 2, "ticker": "ABC", "side": "long", "as_of": str(idx[60].date()), "verdict": "Sell", "horizon_days": 21},
                      {"id": 3, "ticker": "ABC", "side": "long", "as_of": str(idx[390].date()), "verdict": "Buy", "horizon_days": 21}])
    v = score_verdicts(d, closes)
    assert set(v["id"]) == {1, 2}                                     # the last window has not finished
    assert v.set_index("id").loc[1, "hit"] == 1.0 and v.set_index("id").loc[2, "hit"] == 0.0
    assert v.set_index("id").loc[2, "follow"] == 0.0                   # a Sell means out of the stock


def test_paper_orders_mirror_the_ticket(past_and_now):
    then, _ = past_and_now
    pl = build_trade_plan(then, "long", allow_stale=True, with_stats=False)
    assert pl.is_actionable
    trend = ticket_orders(pl, 7)
    assert len(trend) == 1 and trend[0].payload["order_class"] == "oto"
    assert trend[0].payload["stop_loss"]["stop_price"] == round(pl.stop, 2)
    assert trend[0].payload["limit_price"] == round(pl.entry_limit, 2) and trend[0].payload["client_order_id"] == "lh-7-entry"
    tgt = build_trade_plan(then, "long", TradeSettings(long_exit="targets"), allow_stale=True, with_stats=False)
    legs = ticket_orders(tgt, 8)
    assert [o.payload["order_class"] for o in legs] == ["bracket"] * len(legs)
    assert sum(int(o.payload["qty"]) for o in legs) == tgt.shares
    assert legs[0].payload["take_profit"]["limit_price"] == round(tgt.targets[0].price, 2)
    ex = review_exit_order("MSFT", "long", 10, 7, "trend exit")
    assert ex.payload["time_in_force"] == "opg" and ex.payload["side"] == "sell" and ex.payload["type"] == "market"


def test_paper_adapter_is_paper_only_and_dry_without_keys(monkeypatch, past_and_now):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(ValueError):
        AlpacaPaper(base_url="https://api.alpaca.markets")
    b = AlpacaPaper()
    assert b.dry_run and b.positions().empty
    then, _ = past_and_now
    pl = build_trade_plan(then, "long", allow_stale=True, with_stats=False)
    out = mirror(b, {1: pl}, held={"MSFT"})
    assert out == []                                                   # already held: nothing sent
    out = mirror(b, {1: pl})
    assert len(out) == 1 and out[0]["dry_run"] and out[0]["symbol"] == "MSFT" and len(b.sent) == 1


def test_paper_adapter_sends_signed_requests(monkeypatch):
    calls = []

    class R:
        status_code, headers = 200, {}

        def __init__(self, body):
            self.text = json.dumps(body)

        def json(self):
            return json.loads(self.text)

        def close(self):
            pass

    class S:
        def request(self, method, url, params=None, headers=None, timeout=None, stream=False, json=None):
            calls.append((method, url, headers, json))
            return R({"id": "abc", "status": "accepted"} if method == "POST" else [])

    import lighthouse.http as http
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    monkeypatch.setattr(http, "_session", lambda: S())
    b = AlpacaPaper(key_id="k", secret="s")
    assert not b.dry_run
    assert b.submit(review_exit_order("MSFT", "long", 3, 1, "x"))["status"] == "accepted"
    m, url, h, body = calls[-1]
    assert m == "POST" and url == PAPER_URL + "/v2/orders" and h["APCA-API-KEY-ID"] == "k" and body["qty"] == "3"


def test_reconcile_compares_fills_with_the_backtest_assumption():
    orders = pd.DataFrame([
        {"client_order_id": "lh-5-entry", "symbol": "ABC", "side": "buy", "type": "limit", "qty": "10", "filled_qty": "10",
         "filled_avg_price": "101.0", "filled_at": "2026-10-13T13:31:00Z",
         "legs": [{"client_order_id": "lh-5-entry-sl", "symbol": "ABC", "side": "sell", "type": "stop", "stop_price": "95",
                   "filled_avg_price": "94.5", "filled_at": "2026-10-20T15:00:00Z", "filled_qty": "10"}]}])
    opens = {"ABC": pd.Series([100.0], index=[pd.Timestamp("2026-10-13")])}
    r = reconcile(orders, opens).set_index("client_order_id")
    assert r.loc["lh-5-entry", "slippage_bps"] == pytest.approx(100.0)     # paid 1% above the open
    assert r.loc["lh-5-entry-sl", "slippage_bps"] == pytest.approx(-(94.5 / 95 - 1) * 1e4)   # sold below the stop: worse
    assert r.loc["lh-5-entry", "decision_id"] == "5"


def test_win_rate_forecast_is_shrunk_toward_the_pooled_rate():
    from lighthouse.journal import shrunk_win_rate, POOLED_WIN_RATE, PRIOR_TRADES
    assert shrunk_win_rate(1.0, 3, "trend") == pytest.approx((3 + PRIOR_TRADES * 0.45) / (3 + PRIOR_TRADES))
    assert shrunk_win_rate(None, None, "targets") == POOLED_WIN_RATE["targets"]
    assert abs(shrunk_win_rate(0.6, 1000, "trend") - 0.6) < 0.01          # a long history speaks for itself
