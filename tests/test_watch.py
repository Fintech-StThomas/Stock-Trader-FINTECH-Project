"""The forward watch list (lighthouse/research/watch.py) and its score (lighthouse/scorecard.score_watch), offline: the
NYSE month-end clock, the journal's append-only watch tables and their schema upgrade, the recorder (it takes the last
finished session and nothing else), no look-ahead in any of the three signals, late snapshots, the scoring checked by
hand, and that nothing of it reaches a verdict, a ticket or lighthouse/decision.py."""
import ast
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lighthouse import data
from lighthouse.data import DataUnavailable
from lighthouse.journal import SCHEMA_VERSION, Journal, JournalVersionError
from lighthouse.research import harness as H
from lighthouse.research import watch as W
from lighthouse.scorecard import FORWARD_MIN_MONTHS, score_watch

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Synthetic month ends
# --------------------------------------------------------------------------- #
def _frames(n=40, start="2025-01-02", end="2027-03-31", seed=3):
    rng = np.random.default_rng(seed)
    idx = W.sessions(start, end)
    out = {}
    for k in range(n):
        c = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx))))
        out[f"S{k:02d}"] = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6},
                                        index=idx)
    return out


def _toy(inp):
    """A test signal: the stock's return over the last 21 sessions, read at the snapshot's close."""
    v = {t: float(inp.frames[t]["Close"].iloc[-1] / inp.frames[t]["Close"].iloc[-22] - 1) for t in inp.priced_members()}
    return W.SignalValues(pd.Series(v, dtype=float), {"toy": "21-session return"})


TOY = {"toy": W.WatchSignal("toy", "21-session return (a test signal)", "0123456789ab", "long_only", _toy)}


def _inputs(frames, members=None):
    def load(as_of, extra=(), source="auto"):
        inp = W.WatchInputs(as_of, sorted(members or frames), frames, source)
        inp.info = {"current_share": 1.0, "price_source": "synthetic", "members": len(inp.members)}
        return inp
    return load


def _evening(day, hour=22):
    """`hour` UTC on `day`: after the 16:00 New York close in summer and in winter."""
    return pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=hour)


def _record(j, day, frames, signals=TOY, at=None):
    t = at if at is not None else _evening(day)
    return W.record_watch(j, now=t, clock=lambda: t, inputs=_inputs(frames), signals=signals)


# --------------------------------------------------------------------------- #
# The clock
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("day, month_end", [
    ("2026-10-30", True), ("2026-10-29", False), ("2026-10-31", False),     # 31 October 2026 is a Saturday
    ("2024-03-28", True), ("2024-03-29", False),                            # Good Friday ends March 2024 early
    ("2021-05-28", True),                                                   # Memorial Day, Monday 31 May 2021
    ("2022-12-30", True),                                                   # 31 December 2022 is a Saturday
    ("2027-12-31", True),         # New Year's Day 2028 is a Saturday: the NYSE does not close the Friday before
    ("2024-11-29", True),         # the day after Thanksgiving (an early close) is November 2024's last session
    ("2026-12-24", False), ("2026-12-31", True), ("2026-07-31", True), ("2026-07-03", False)])
def test_month_end_sessions_follow_the_nyse_calendar(day, month_end):
    assert W.is_month_end_session(day) == month_end


def test_the_deadline_is_the_next_open_in_summer_and_winter():
    assert W.next_open_utc("2026-09-30") == pd.Timestamp("2026-10-01 13:30", tz="UTC")     # New York on summer time
    assert W.next_open_utc("2026-10-30") == pd.Timestamp("2026-11-02 14:30", tz="UTC")     # a Friday; winter time
    assert W.next_open_utc("2024-03-28") == pd.Timestamp("2024-04-01 13:30", tz="UTC")     # over Good Friday
    assert W.on_time("2026-10-01T13:29:59+00:00", "2026-09-30")
    assert not W.on_time("2026-10-01T13:30:00+00:00", "2026-09-30")
    assert W.next_month_end_session("2026-10-30") == pd.Timestamp("2026-11-30")
    assert W.next_month_end_session("2024-02-29") == pd.Timestamp("2024-03-28")
    # the last finished session: 16:00 New York is 20:00 UTC in October
    assert W.last_finished_session(pd.Timestamp("2026-10-30 19:59", tz="UTC")) == pd.Timestamp("2026-10-29")
    assert W.last_finished_session(pd.Timestamp("2026-10-30 20:01", tz="UTC")) == pd.Timestamp("2026-10-30")
    assert W.last_finished_session(pd.Timestamp("2026-11-01 12:00", tz="UTC")) == pd.Timestamp("2026-10-30")
    assert W.is_due(pd.Timestamp("2026-10-31 15:00", tz="UTC")) and not W.is_due(pd.Timestamp("2026-11-02 22:00", tz="UTC"))


# --------------------------------------------------------------------------- #
# The journal's watch tables
# --------------------------------------------------------------------------- #
def _row(ticker="AAA", as_of="2026-10-30", signal="toy", value=1.0):
    return {"recorded_at": "2026-10-30T22:00:00+00:00", "as_of": as_of, "signal": signal, "ticker": ticker,
            "value": value, "rank": 1.0, "pct": 1.0, "quintile": None, "n_ranked": 1, "tradable": 1, "cost": 0.002,
            "config_hash": "0123456789ab", "data_version": "{}", "code_version": "test", "run_id": "r1"}


def _outcome(ticker="AAA", as_of="2026-10-30", ret=0.01):
    return {"recorded_at": "2026-11-30T22:00:00+00:00", "as_of": as_of, "end_date": "2026-11-30", "ticker": ticker,
            "ret_exclude": ret, "ret_pessimistic": ret, "data_version": "{}", "code_version": "test", "run_id": "r2"}


def test_watch_tables_are_append_only_with_one_row_per_key(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    assert j.append_watch_rows([_row("AAA"), _row("BBB")]) == 2
    assert j.append_watch_outcomes([_outcome("AAA")]) == 1
    con = sqlite3.connect(str(j.path))
    for sql in ("UPDATE watch_signals SET value = 0", "DELETE FROM watch_signals",
                "UPDATE watch_outcomes SET ret_exclude = 0", "DELETE FROM watch_outcomes"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            con.execute(sql)
    con.close()
    with pytest.raises(sqlite3.IntegrityError):                      # the same signal, configuration, date and ticker
        j.append_watch_rows([_row("AAA", value=2.0)])
    with pytest.raises(sqlite3.IntegrityError):                      # a snapshot is written whole or not at all
        j.append_watch_rows([_row("CCC"), _row("BBB")])
    with pytest.raises(sqlite3.IntegrityError):
        j.append_watch_outcomes([_outcome("AAA", ret=0.5)])
    s = j.watch_signals()
    assert sorted(s["ticker"]) == ["AAA", "BBB"] and s["value"].tolist() == [1.0, 1.0]
    assert j.append_watch_rows([_row("AAA", signal="toy_quarterly")]) == 1    # a variant is another signal name
    paths = j.export(tmp_path / "out")
    assert {p.name for p in paths} >= {"journal_watch_signals.csv", "journal_watch_outcomes.csv"}
    assert len(pd.read_csv(tmp_path / "out" / "journal_watch_signals.csv")) == 3


# The schema of a journal written before the watch tables existed (lighthouse/journal.py at commit 9c98e9a), verbatim.
V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL, as_of TEXT NOT NULL, ticker TEXT NOT NULL,
  horizon TEXT, risk TEXT, side TEXT, verdict TEXT, status TEXT, composite REAL, fundamental REAL, technical REAL,
  regime TEXT, price REAL, priced_from TEXT, entry_limit REAL, stop REAL, target1 REAL, target2 REAL, trend_exit REAL,
  signal_exit REAL, exit_style TEXT, time_stop INTEGER, shares INTEGER, notional REAL, risk_usd REAL, account REAL,
  cash REAL, p_win REAL, hist_trades INTEGER, horizon_days INTEGER, inputs TEXT, data_version TEXT, code_version TEXT,
  run_id TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL, decision_id INTEGER, ticker TEXT NOT NULL,
  kind TEXT NOT NULL, date TEXT, price REAL, shares INTEGER, detail TEXT, run_id TEXT);
CREATE INDEX IF NOT EXISTS decisions_ticker ON decisions (ticker, as_of);
CREATE INDEX IF NOT EXISTS events_decision ON events (decision_id);
CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
"""


def test_a_version_1_journal_is_upgraded_in_place(tmp_path):
    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(str(p))
    con.executescript(V1_SCHEMA)
    con.execute("INSERT INTO meta (key, value) VALUES ('schema_version', '1')")
    con.execute("INSERT INTO meta (key, value) VALUES ('created_at', '2026-10-13T22:20:00+00:00')")
    con.execute("INSERT INTO decisions (recorded_at, as_of, ticker, verdict) VALUES "
                "('2026-10-13T22:20:00+00:00', '2026-10-13', 'MSFT', 'Strong Buy')")
    con.execute("INSERT INTO events (recorded_at, decision_id, ticker, kind) VALUES "
                "('2026-10-13T22:20:00+00:00', 1, 'MSFT', 'opened')")
    con.execute("CREATE TABLE theses (id INTEGER PRIMARY KEY, ticker TEXT)")      # another module's table rides along
    con.execute("INSERT INTO theses (ticker) VALUES ('MSFT')")
    con.commit()
    con.close()
    j = Journal(p)
    assert j.schema_version() == SCHEMA_VERSION == 2
    meta = dict(sqlite3.connect(str(p)).execute("SELECT key, value FROM meta").fetchall())
    assert meta["created_at"] == "2026-10-13T22:20:00+00:00" and "upgraded_1_to_2" in meta
    d = j.decisions()
    assert len(d) == 1 and d.iloc[0]["verdict"] == "Strong Buy" and len(j.events()) == 1
    assert sqlite3.connect(str(p)).execute("SELECT COUNT(*) FROM theses").fetchone()[0] == 1
    assert j.watch_signals().empty and j.watch_outcomes().empty
    j.append_watch_rows([_row()])
    con = sqlite3.connect(str(p))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE watch_signals SET value = 0")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM decisions")                                      # the old triggers still hold
    con.close()
    Journal(p)                                                                    # opening it again changes nothing
    assert dict(sqlite3.connect(str(p)).execute("SELECT key, value FROM meta").fetchall()) == meta
    assert len(Journal(p).watch_signals()) == 1


def test_a_journal_from_a_newer_version_is_refused(tmp_path):
    p = tmp_path / "j.sqlite"
    Journal(p)
    con = sqlite3.connect(str(p))
    con.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    con.commit()
    con.close()
    with pytest.raises(JournalVersionError, match="schema 99"):
        Journal(p)


# --------------------------------------------------------------------------- #
# The recorder
# --------------------------------------------------------------------------- #
def test_the_recorder_takes_the_last_finished_session_and_nothing_else(tmp_path):
    frames = _frames()
    path = tmp_path / "j.sqlite"
    # 29 October 2026 is not October's last session: nothing happens, and no journal is created
    rep = _record(path, "2026-10-29", frames)
    assert not rep.due and not path.exists() and "2026-10-30" in rep.lines()[0]
    # on the 30th before the close (16:00 New York, 20:00 UTC) the last finished session is the 29th
    rep = _record(path, None, frames, at=pd.Timestamp("2026-10-30 19:59", tz="UTC"))
    assert not rep.due and rep.as_of == pd.Timestamp("2026-10-29") and not path.exists()
    rep = _record(path, "2026-10-30", frames)
    assert rep.due and rep.recorded == {"toy": 40} and not rep.errors
    j = Journal(path)
    s = j.watch_signals()
    assert len(s) == 40 and set(s["as_of"]) == {"2026-10-30"} and set(s["config_hash"]) == {"0123456789ab"}
    assert set(s["recorded_at"]) == {"2026-10-30T22:00:00+00:00"} and s["code_version"].str.startswith("lighthouse").all()
    # the value is the signal at the close of the 30th; rank, percentile and quintile among the tested members
    cut = {t: f[f.index <= "2026-10-30"] for t, f in frames.items()}
    want = pd.Series({t: f["Close"].iloc[-1] / f["Close"].iloc[-22] - 1 for t, f in cut.items()})
    got = s.set_index("ticker")
    np.testing.assert_allclose(got["value"].reindex(want.index), want.to_numpy())
    assert (got["tradable"] == 1).all() and (got["n_ranked"] == 40).all()
    top = want.idxmax()
    assert got.loc[top, "rank"] == 40 and got.loc[top, "pct"] == 1.0 and got.loc[top, "quintile"] == 5
    assert (got["quintile"].value_counts().sort_index() == 8).all()
    costs = H.cost_panel(cut, pd.DatetimeIndex(["2026-10-30"])).iloc[0]
    np.testing.assert_allclose(got["cost"].reindex(costs.index), costs.to_numpy())
    # a second run the same evening, and the Saturday after (whose last finished session is still the 30th), add nothing
    again = _record(j, None, frames, at=_evening("2026-10-30", 23))
    assert again.recorded == {} and again.already == ["toy"]
    sat = _record(j, "2026-10-31", frames)
    assert sat.as_of == pd.Timestamp("2026-10-30") and sat.recorded == {} and len(j.watch_signals()) == 40


def test_the_recorder_has_no_date_option_and_refuses_stale_prices(tmp_path):
    import inspect
    from lighthouse.__main__ import main
    with pytest.raises(SystemExit):                                   # the command line cannot backdate
        main(["journal", "watch", "--as-of", "2026-09-30"])
    params = inspect.signature(W.record_watch).parameters
    assert "as_of" not in params and "date" not in params
    frames = _frames()

    def stale(as_of, extra=(), source="auto"):
        inp = _inputs(frames)(as_of)
        inp.info["current_share"] = 0.5                               # half the members' prices end the day before
        return inp
    t = _evening("2026-10-30")
    rep = W.record_watch(tmp_path / "j.sqlite", now=t, clock=lambda: t, inputs=stale, signals=TOY)
    assert "prices" in rep.errors and Journal(tmp_path / "j.sqlite").watch_signals().empty


def test_outcomes_are_recorded_once_the_next_month_end_has_closed(tmp_path):
    frames = _frames()
    frames["S00"] = frames["S00"][frames["S00"].index <= "2026-11-10"]          # stops trading in November
    j = Journal(tmp_path / "j.sqlite")
    _record(j, "2026-10-30", frames)
    assert j.watch_outcomes().empty
    mid = _record(j, "2026-11-13", frames)                                      # not a month end: nothing
    assert not mid.due and j.watch_outcomes().empty
    rep = _record(j, "2026-11-30", frames)
    assert rep.outcomes == {"2026-10-30": 40}
    o = j.watch_outcomes().set_index("ticker")
    assert set(o["as_of"]) == {"2026-10-30"} and set(o["end_date"]) == {"2026-11-30"}
    c = frames["S01"]["Close"]
    assert o.loc["S01", "ret_exclude"] == pytest.approx(c.loc["2026-11-30"] / c.loc["2026-10-30"] - 1)
    assert o.loc["S01", "ret_pessimistic"] == o.loc["S01", "ret_exclude"]
    dead = frames["S00"]["Close"]
    assert o.loc["S00", "ret_pessimistic"] == -1.0                              # the two delisting rules
    assert o.loc["S00", "ret_exclude"] == pytest.approx(dead.iloc[-1] / dead.loc["2026-10-30"] - 1)
    # in November's snapshot S00 has no month-end price: recorded, not tested, not ranked
    nov = j.watch_signals()
    nov = nov[nov["as_of"] == "2026-11-30"].set_index("ticker")
    assert nov.loc["S00", "tradable"] == 0 and np.isnan(nov.loc["S00", "rank"]) and nov["n_ranked"].iloc[0] == 39
    # the outcome is written once
    assert _record(j, None, frames, at=_evening("2026-11-30", 23)).outcomes == {}
    assert len(j.watch_outcomes()) == 40


def test_a_snapshot_does_not_change_when_later_data_is_appended(tmp_path):
    frames = _frames()
    cut = {t: f[f.index <= "2026-10-30"] for t, f in frames.items()}
    a = Journal(tmp_path / "a.sqlite")
    b = Journal(tmp_path / "b.sqlite")
    _record(a, "2026-10-30", cut)                   # only what existed at the close of 30 October
    _record(b, "2026-10-30", frames)                # five more months of prices passed in: none of it is used
    cols = ["ticker", "value", "rank", "pct", "quintile", "tradable", "cost"]
    pd.testing.assert_frame_equal(a.watch_signals()[cols], b.watch_signals()[cols])


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_late_snapshots_are_kept_but_not_scored(tmp_path):
    frames = _frames()
    j = Journal(tmp_path / "j.sqlite")
    for day in ("2026-07-31", "2026-08-31", "2026-09-30", "2026-10-30", "2026-11-30", "2026-12-31"):
        at = pd.Timestamp("2026-10-01 14:00", tz="UTC") if day == "2026-09-30" else None    # after the 13:30 open
        rep = _record(j, day, frames, at=at)
        assert rep.recorded == {"toy": 40}
    wc = score_watch(j, signals=TOY)
    late = wc.late()
    assert late["as_of"].tolist() == ["2026-09-30"] and late["deadline"].iloc[0].startswith("2026-10-01T13:30")
    assert len(j.watch_signals()) == 6 * 40                                     # kept in the journal
    ex = wc.table[wc.table["rule"] == "exclude"].iloc[0]
    assert ex["snapshots"] == 6 and ex["late"] == 1
    # scored: July, August, October, November (September is late, December's outcome is not known yet)
    scored = sorted(wc.monthly.loc[wc.monthly["rule"] == "exclude", "as_of"].dt.strftime("%Y-%m-%d"))
    assert scored == ["2026-07-31", "2026-08-31", "2026-10-30", "2026-11-30"] and ex["months"] == 4
    assert "late" in wc.text() and "2026-09-30" in wc.text()
    # the same months through the harness on the same data give the same numbers
    cal = W.sessions("2026-06-01", "2026-12-31")
    panel = H.build_panel(None, frames, start=cal[0], calendar=cal)
    sig = pd.DataFrame({t: f["Close"] / f["Close"].shift(21) - 1 for t, f in frames.items()}).reindex(panel.dates)
    sig.loc[pd.Timestamp("2026-09-30")] = np.nan
    res = H.evaluate(panel, sig, "toy", "exclude", ("2026-07-01", None), H.cost_panel(frames, panel.dates), None)
    assert ex["ic_mean"] == pytest.approx(res.ic.mean()) and ex["ic_nw_t"] == pytest.approx(res.summary["ic_nw_t"])
    assert ex["long_only_net_ann"] == pytest.approx(res.long_only_net.mean() * 12)
    assert ex["top_turnover"] == pytest.approx(res.summary["top_turnover"])


def _hand_journal(path):
    """Five stocks, three scored month ends and a fourth without an outcome yet, rows written as the recorder would."""
    j = Journal(path)
    names = ["A", "B", "C", "D", "E"]
    values = {"2026-01-30": [1, 2, 3, 4, 5], "2026-02-27": [5, 4, 3, 2, 1], "2026-03-31": [1, 2, 3, 4, 5],
              "2026-04-30": [1, 2, 3, 4, 5]}
    rets = {"2026-01-30": [0.01, 0.02, 0.03, 0.04, 0.05], "2026-02-27": [0.01, 0.02, 0.03, 0.04, 0.05],
            "2026-03-31": [0.0, 0.0, 0.0, 0.0, 0.10]}
    for d, vals in values.items():
        at = (pd.Timestamp(d) + pd.Timedelta(hours=22)).tz_localize("UTC").isoformat(timespec="seconds")
        j.append_watch_rows([{**_row(t, d, "hand", float(v)), "recorded_at": at, "cost": 0.002, "rank": float(v),
                              "pct": v / 5.0, "quintile": int(v), "n_ranked": 5}
                             for t, v in zip(names, vals)])
    for d, rr in rets.items():
        rows = [{**_outcome(t, d, r), "end_date": str(W.next_month_end_session(d).date())} for t, r in zip(names, rr)]
        if d == "2026-03-31":
            rows[3]["ret_pessimistic"] = -1.0                       # D stops trading during April: -100% if worthless
        j.append_watch_outcomes(rows)
    return j


def test_scoring_on_a_small_example_checked_by_hand(tmp_path):
    j = _hand_journal(tmp_path / "j.sqlite")
    spec = {"hand": W.WatchSignal("hand", "hand-made", "0123456789ab", "long_only", _toy)}
    wc = score_watch(j, min_names=5, signals=spec, power=False)
    ex = wc.table.set_index("rule").loc["exclude"]
    # monthly Spearman ICs: +1 (same order), -1 (reversed), and with A-D tied at 0% and E at +10%:
    # ranks y = 2.5, 2.5, 2.5, 2.5, 5 against x = 1..5 -> 5 / sqrt(10 x 5) = 0.7071
    ic = wc.monthly[wc.monthly["rule"] == "exclude"].set_index("as_of")["ic"]
    np.testing.assert_allclose(ic.to_numpy(), [1.0, -1.0, 5 / math.sqrt(50)])
    assert ex["months"] == 3                                   # April's snapshot has no outcome yet: not scored
    assert ex["ic_mean"] == pytest.approx((5 / math.sqrt(50)) / 3) and ex["ic_hit"] == pytest.approx(2 / 3)
    # Newey-West t with 3 lags (Bartlett weights, at most n - 1 = 2 lags here), written out
    x = np.array([1.0, -1.0, 5 / math.sqrt(50)])
    e = x - x.mean()
    s = e @ e / 3 + 2 * (1 - 1 / 4) * (e[1:] @ e[:-1]) / 3 + 2 * (1 - 2 / 4) * (e[2:] @ e[:-2]) / 3
    assert ex["ic_nw_t"] == pytest.approx(x.mean() / math.sqrt(s / 3))
    # the top quintile is one stock: E, then A, then E. Its return minus the equal-weight universe's (3%, 3%, 2%):
    # +2%, -2%, +8% gross; 0.2% a side, paid on the change in weights (1, then 2, then 2): +1.8%, -2.4%, +7.6%
    assert ex["long_only_gross_ann"] == pytest.approx((0.02 - 0.02 + 0.08) / 3 * 12)
    assert ex["long_only_net_ann"] == pytest.approx((0.018 - 0.024 + 0.076) / 3 * 12)
    assert ex["long_only_net_ann_10bps"] == pytest.approx((0.019 - 0.022 + 0.078) / 3 * 12)
    assert ex["top_turnover"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)          # one-way: half of the weight changes
    # long-short: top minus bottom quintile (+4%, -4%, +10%), both legs' costs and 0.5% a year of borrow
    assert ex["long_short_gross_ann"] == pytest.approx(0.10 / 3 * 12)
    assert ex["long_short_net_ann"] == pytest.approx(((0.04 - 0.004) + (-0.04 - 0.008) + (0.10 - 0.008)) / 3 * 12
                                                     - 0.005)
    assert ex["coverage"] == pytest.approx(1.0)
    # the pessimistic rule: D is worth nothing in April. Ranks of (0, 0, 0, -1, 0.10) are 3, 3, 3, 1, 5 against
    # x = 1..5 -> 2 / sqrt(10 x 8); E beats the universe's -18% by 28%
    pe = wc.monthly[wc.monthly["rule"] == "pessimistic"].set_index("as_of")
    assert pe["ic"].iloc[-1] == pytest.approx(2 / math.sqrt(80))
    assert pe["long_only"].iloc[-1] == pytest.approx(0.10 - (-0.9 / 5))
    st = wc.status.iloc[0]
    assert st["status"].startswith("collecting") and st["look"] is None


def test_the_forward_rule_is_read_at_the_december_look(tmp_path):
    """Three years of forward months from a signal that knows next month's return: the rule is met at the look of
    December (and only says a new pre-registered test may be considered); an incomplete record is not judged."""
    names = [f"T{k:02d}" for k in range(35)]

    def planted(path, dates, seed=5):
        rng = np.random.default_rng(seed)
        j = Journal(path)
        for d in dates:
            r = rng.normal(0.01, 0.05, len(names))
            v = r + rng.normal(0, 0.05, len(names))              # a planted signal: knows next month's return
            at = (d + pd.Timedelta(hours=22)).tz_localize("UTC").isoformat(timespec="seconds")
            j.append_watch_rows([{**_row(t, str(d.date()), "fcf_yield", float(x)), "recorded_at": at,
                                  "config_hash": W.WATCH_SIGNALS["fcf_yield"].config_hash} for t, x in zip(names, v)])
            j.append_watch_outcomes([{**_outcome(t, str(d.date()), float(x)), "end_date": "x"}
                                     for t, x in zip(names, r)])
        return score_watch(j, power=False).status.set_index("signal").loc["fcf_yield"]
    month_ends = [d for d in W.sessions("2027-01-01", "2033-03-31") if W.is_month_end_session(d)]
    # 35 months to December 2029: still collecting
    st = planted(tmp_path / "a.sqlite", month_ends[1:37])
    assert st["look"] == "2029-12-31" and st["months_at_look"] == 35 and st["status"].startswith("collecting: 35 of 36")
    # 36 months: met at the look of December 2029 (which only makes the case for a new test)
    st = planted(tmp_path / "b.sqlite", month_ends[:38])
    assert st["look"] == "2029-12-31" and st["months_at_look"] == 36 >= FORWARD_MIN_MONTHS
    assert st["status"].startswith("met at the look of 2029-12-31") and "nothing is weighted" in st["status"]
    # six years with every fourth month end missing (each December among them): enough months, but the record is
    # incomplete, whatever the numbers. The look is dated by the calendar: a missing December does not skip it, and it
    # is read once a later month (January 2033) has been scored
    st = planted(tmp_path / "c.sqlite", [d for k, d in enumerate(month_ends[:73]) if k % 4 != 3])
    assert st["look"] == "2032-12-31" and st["months_at_look"] == 54
    assert st["status"].startswith("incomplete record at the look of 2032-12-31: 18 of 72")
    # a signal with no edge: not met
    rng = np.random.default_rng(9)
    j = Journal(tmp_path / "d.sqlite")
    for d in month_ends[:38]:
        at = (d + pd.Timedelta(hours=22)).tz_localize("UTC").isoformat(timespec="seconds")
        j.append_watch_rows([{**_row(t, str(d.date()), "fcf_yield", float(x)), "recorded_at": at,
                              "config_hash": W.WATCH_SIGNALS["fcf_yield"].config_hash}
                             for t, x in zip(names, rng.normal(size=len(names)))])
        j.append_watch_outcomes([{**_outcome(t, str(d.date()), float(x)), "end_date": "x"}
                                 for t, x in zip(names, rng.normal(0.01, 0.05, len(names)))])
    st = score_watch(j, power=False).status.set_index("signal").loc["fcf_yield"]
    assert st["status"].startswith("not met at the look of 2029-12-31") and "Newey-West" in st["status"]


def test_months_needed_follow_from_the_logged_ic():
    p = W.months_needed().set_index(["signal", "period"])
    log = H.ExperimentLog().rows()
    for name, spec in W.WATCH_SIGNALS.items():
        rows = log[(log["config_hash"] == spec.config_hash) & (log["delisting_rule"] == "exclude")]
        for period in ("tuning", "holdout"):
            x = rows[rows["period"] == period].iloc[-1]
            got = p.loc[(name, period)]
            assert got["ic_sd"] == pytest.approx(x["ic_mean"] / x["ic_t"] * math.sqrt(x["months"]))
            if x["ic_mean"] > 0:
                assert got["months_t2"] == pytest.approx(x["months"] * (2 / x["ic_nw_t"]) ** 2)
                assert got["months_80"] > got["months_t2"]
            else:
                assert got["months_t2"] == math.inf
    # the numbers docs/METHODOLOGY.md section 8c quotes
    assert round(p.loc[("risk_similarity", "holdout"), "months_t2"]) == 82
    assert round(p.loc[("fcf_yield", "tuning"), "months_t2"]) == 75
    assert round(p.loc[("fcf_yield", "holdout"), "months_t2"]) == 801
    assert round(p.loc[("short_volume", "tuning"), "months_t2"]) == 34
    assert round(p.loc[("short_volume", "holdout"), "months_t2"]) == 123


def test_watch_signals_are_the_logged_trials():
    log = H.ExperimentLog().rows()
    for name, spec in W.WATCH_SIGNALS.items():
        assert H.config_hash(name, spec.config) == spec.config_hash          # the configuration, hashed as logged
        logged = log[log["signal"] == name]
        # the watched trial, and only the earlier trials it replaced before any forward snapshot (W.SUPERSEDED)
        assert spec.config_hash in set(logged["config_hash"])
        assert set(logged["config_hash"]) <= {spec.config_hash, *W.SUPERSEDED.get(name, ())}
        assert json.loads(logged.loc[logged["config_hash"] == spec.config_hash, "config"].iloc[-1]) == spec.config
    from lighthouse.research import event_signals as EV
    from lighthouse.research import fundamental_signals as FS
    from lighthouse.research import positioning as PO
    assert W.WATCH_SIGNALS["fcf_yield"].description == FS.DESCRIPTIONS["fcf_yield"]
    assert W.WATCH_SIGNALS["short_volume"].description == PO.DESCRIPTIONS["short_volume"]
    assert W.WATCH_SIGNALS["risk_similarity"].description == EV.DESCRIPTIONS["risk_similarity"]
    verdicts = pd.read_csv(H.RESEARCH_DIR / "signal_verdicts.csv").set_index("signal")
    for name, spec in W.WATCH_SIGNALS.items():
        assert verdicts.loc[name, "use"] == spec.use and not verdicts.loc[name, "survives"]


# --------------------------------------------------------------------------- #
# The three signals: the harness's values, and no look-ahead
# --------------------------------------------------------------------------- #
def test_fcf_yield_snapshot_is_the_harness_value_and_ignores_later_filings(monkeypatch):
    from lighthouse import edgar
    from lighthouse.data import load_prices
    from lighthouse.research import fundamental_signals as FS
    from lighthouse.total_return import _empty_actions
    monkeypatch.setattr(FS, "load_actions", lambda *a, **k: _empty_actions())   # the prices are taken as traded
    tickers = ["AAPL", "JNJ", "KO", "XOM"]
    frames = {}
    for t in tickers:
        df = load_prices(t, source="sample", years=None)
        df.attrs = {}
        frames[t] = df
    facts = {t: edgar.company_facts(ticker=t, source="sample") for t in tickers}
    as_of = pd.Timestamp("2016-06-30")
    assert W.is_month_end_session(as_of)
    monkeypatch.setattr(edgar, "company_facts", lambda **kw: facts[kw["ticker"]])
    later = W.fcf_yield_values(W.WatchInputs(as_of, tickers, frames, "sample")).values   # six more years of filings
    cut = as_of
    monkeypatch.setattr(edgar, "company_facts", lambda **kw: facts[kw["ticker"]][facts[kw["ticker"]]["filed"] <= cut])
    then = W.fcf_yield_values(W.WatchInputs(as_of, tickers, frames, "sample")).values
    pd.testing.assert_series_equal(later, then)
    assert then.notna().all() and (then > 0).all()
    # the harness's monthly panel on all the data gives the same value at that month end
    monkeypatch.setattr(edgar, "company_facts", lambda **kw: facts[kw["ticker"]])
    daily = {t: FS.daily_values(t, frames[t], "sample")["fcf_yield"] for t in tickers}
    me = H.month_ends(frames["AAPL"].index[frames["AAPL"].index <= "2016-12-31"])
    harness = H.sample_at_month_ends(daily, me).loc[as_of]
    np.testing.assert_allclose(then.to_numpy(), harness.reindex(then.index).to_numpy())
    # a filing made after the close of the snapshot date is not used, one made the day before is
    cut = as_of - pd.Timedelta(days=1)
    monkeypatch.setattr(edgar, "company_facts", lambda **kw: facts[kw["ticker"]][facts[kw["ticker"]]["filed"] <= cut])
    pd.testing.assert_series_equal(W.fcf_yield_values(W.WatchInputs(as_of, tickers, frames, "sample")).values, then)


LM_CSV = """Word,Seq_num,Word Count,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining,Syllables
LOSS,1,10,2009,0,0,0,0,0,0,1
MAY,2,10,0,0,2009,0,0,2009,0,1
LAWSUIT,4,10,0,0,0,2009,0,0,0,2
GAIN,5,10,0,2009,0,0,0,0,0,1
"""


def _tenk(words: str) -> str:
    risk = "Item 1A. Risk Factors " + words * 120 + " Item 1B. Unresolved Staff Comments None. Item 2. Properties offices."
    return f"<html><body><p>Annual report. {words}</p><p>{risk}</p></body></html>"


DOCS = {"a1": _tenk("We may face a lawsuit and a loss. "), "a2": _tenk("We may face a lawsuit and a gain. "),
        "a3": _tenk("Competition may reduce our sales and our gain. "), "a4": _tenk("Rates may rise and cause a loss. ")}
FILINGS = pd.DataFrame({
    "accn": ["a1", "a2", "q1", "a3", "a4"],
    "filed": pd.to_datetime(["2014-02-14", "2015-02-13", "2015-05-01", "2016-02-29", "2017-02-10"]),
    "form": ["10-K", "10-K", "10-Q", "10-K", "10-K"], "doc": ["k1.htm", "k2.htm", "q1.htm", "k3.htm", "k4.htm"],
    "items": ["", "", "", "", ""]})


def _serve_tenks(monkeypatch, cache, until=None):
    from lighthouse import edgar, text
    lm = text.parse_lm_csv(io.StringIO(LM_CSV))
    monkeypatch.setattr(data, "CACHE_DIR", cache)
    monkeypatch.setattr(text, "lm_dictionary", lambda refresh=False: lm)
    monkeypatch.setattr(edgar, "cik_for", lambda t: 1234)
    f = FILINGS if until is None else FILINGS[FILINGS["filed"] <= pd.Timestamp(until)]
    monkeypatch.setattr(edgar, "company_filings", lambda t, source="auto", refresh=False: ({}, f.copy()))
    fetched = []

    def fetch(cik, accn, doc, cache=True):
        fetched.append((accn, cache))
        return DOCS[accn]
    monkeypatch.setattr(edgar, "fetch_document", fetch)
    return lm, fetched


def test_risk_similarity_snapshot_is_the_harness_value_and_causal(monkeypatch, tmp_path):
    from lighthouse.research import event_signals as EV
    idx = W.sessions("2013-01-02", "2018-12-31")
    frames = {"ABC": pd.DataFrame({"Close": 10.0, "High": 10.1, "Low": 9.9, "Open": 10.0, "Volume": 1e6}, index=idx)}
    lm, _ = _serve_tenks(monkeypatch, tmp_path / "harness")
    hist = EV.text_history("ABC", lm)                                           # every 10-K, as the research run
    daily = EV.text_signal_daily(hist, idx, "risk_similarity", 1.0)
    me = H.month_ends(idx)
    harness = H.sample_at_month_ends({"ABC": daily}, me)["ABC"]
    _, fetched = _serve_tenks(monkeypatch, tmp_path / "watch")                  # a cache without the research run
    for d in ["2014-02-28", "2015-01-30", "2015-02-27", "2016-02-29", "2016-03-31", "2017-01-31", "2017-02-28",
              "2017-12-29", "2018-06-29"]:
        got = W.risk_similarity_values(W.WatchInputs(d, ["ABC"], frames), processes=1).values["ABC"]
        want = harness.loc[pd.Timestamp(d)]
        assert (np.isnan(got) and np.isnan(want)) or got == pytest.approx(want), d
    assert np.isnan(harness.loc["2014-02-28"])                     # the first 10-K has nothing to compare with
    assert harness.loc["2016-02-29"] == pytest.approx(hist["risk_similarity"].iloc[1])   # filed that day: not yet
    assert harness.loc["2016-03-31"] == pytest.approx(hist["risk_similarity"].iloc[2])
    assert np.isnan(harness.loc["2018-06-29"])                     # older than 15 months
    assert all(cache is False for _, cache in fetched)             # 10-K documents are not stored
    # the snapshot of 31 March 2016 is the same whether the 2017 10-K exists yet or not, and read from the cache
    d = "2016-03-31"
    before = W.risk_similarity_values(W.WatchInputs(d, ["ABC"], frames), processes=1)
    _, fetched = _serve_tenks(monkeypatch, tmp_path / "watch", until=d)
    after = W.risk_similarity_values(W.WatchInputs(d, ["ABC"], frames), processes=1)
    assert after.values["ABC"] == before.values["ABC"] and fetched == []
    assert after.detail["ABC"] == {"10-K": "a3", "filed": "2016-02-29"}
    # a research cache that holds the filing is used as it is
    _, fetched = _serve_tenks(monkeypatch, tmp_path / "harness")
    W.risk_similarity_values(W.WatchInputs("2017-02-28", ["ABC"], frames), processes=1)
    assert fetched == []


def test_the_watch_reads_each_10k_under_the_registrant_that_filed_it(monkeypatch, tmp_path):
    # a successor issuer's merged filing list (edgar.submissions) holds its predecessor's 10-Ks, which sit under the
    # predecessor's CIK; synthetic filings, as in test_text_events
    from lighthouse import edgar, text
    lm = text.parse_lm_csv(io.StringIO(LM_CSV))
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    filings = pd.DataFrame({"accn": ["a-2025", "a-2027"], "form": ["10-K", "10-K"],
                            "filed": pd.to_datetime(["2025-02-19", "2027-02-17"]), "doc": ["k25.htm", "k27.htm"],
                            "cik": [34088, 2115436]})
    monkeypatch.setattr(edgar, "cik_for", lambda t: 2115436)
    monkeypatch.setattr(edgar, "company_filings", lambda t, source="auto", refresh=False: ({}, filings))
    asked = []
    monkeypatch.setattr(edgar, "fetch_document", lambda cik, accn, doc, cache=True: asked.append((cik, accn)) or
                        "We expect a loss. Our risk factors are uncertain. Revenue may decline.")
    row = W.known_tenk_rows("XOM", "2027-03-31", lm=lm)
    assert asked == [(34088, "a-2025"), (2115436, "a-2027")] and list(row["accn"]) == ["a-2027"]


def _serve_finra(monkeypatch, cache, until=None):
    from lighthouse.research import positioning as PO
    fix = Path(__file__).parent / "fixtures" / "positioning"
    monkeypatch.setattr(data, "CACHE_DIR", cache)
    cut = pd.Timestamp(until) if until is not None else None

    def fake(url, **kw):
        name = url.rsplit("/", 1)[-1]
        f = fix / name
        if not f.exists() or (cut is not None and pd.Timestamp(name[-12:-4]) > cut):
            raise DataUnavailable(f"HTTP 403 from cdn.finra.org for {url}")
        body = f.read_bytes()
        return SimpleNamespace(content=body, text=body.decode("utf-8"), status_code=200)
    monkeypatch.setattr(PO, "_get", fake)


def test_short_volume_snapshot_is_the_harness_value_and_causal(monkeypatch, tmp_path):
    from lighthouse.research import positioning as PO
    tickers = ["AAPL", "AMD", "BRK-B", "MSFT", "XOM"]
    idx = W.sessions("2022-06-01", "2024-01-31")
    frames = {t: pd.DataFrame({"Close": 10.0}, index=idx) for t in tickers}
    as_of = "2023-12-29"
    _serve_finra(monkeypatch, tmp_path / "full")                    # the month end's own file and January's exist
    full = W.short_volume_values(W.WatchInputs(as_of, tickers, frames))
    harness = PO.short_volume_signal(pd.DatetimeIndex([as_of]), tickers).loc[as_of]
    pd.testing.assert_series_equal(full.values, harness, check_names=False)
    short = 6_472_070 + 3_852_742 + 6_315_121 + 3_891_667
    total = 13_573_593 + 11_354_301 + 18_507_950 + 12_447_595
    assert full.values["AAPL"] == pytest.approx(-short / total)
    assert full.info["finra_sessions"] == ["2023-12-21", "2023-12-22", "2023-12-26", "2023-12-27", "2023-12-28"]
    assert full.info["finra_files_found"] == ["2023-12-22", "2023-12-26", "2023-12-27", "2023-12-28"]
    _serve_finra(monkeypatch, tmp_path / "then", until="2023-12-28")          # what FINRA had published by the close
    then = W.short_volume_values(W.WatchInputs(as_of, tickers, frames))
    pd.testing.assert_series_equal(full.values, then.values)


# --------------------------------------------------------------------------- #
# Watch-only: nothing reaches a verdict or a ticket
# --------------------------------------------------------------------------- #
DECISION_MODULES = ["decision", "engine", "tradeplan", "screener", "portfolio", "fundamentals", "technicals", "regime",
                    "backtest", "broker", "config", "costs", "universe", "journal"]   # journal.py writes the verdicts


def _imports(path: Path) -> set:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
            out |= {"." * node.level + (node.module or "") + "." + a.name for a in node.names}
    return out


def test_watch_signals_never_reach_decisions(tmp_path, monkeypatch):
    for m in DECISION_MODULES:
        names = _imports(ROOT / "lighthouse" / f"{m}.py")
        assert not any("watch" in n or "scorecard" in n for n in names), (m, names)
    # decision.py makes the verdict from the configuration, the regime and the beacons, nothing else
    assert _imports(ROOT / "lighthouse" / "decision.py") <= {
        "__future__", "__future__.annotations", "math", "dataclasses", "dataclasses.dataclass", "dataclasses.field",
        "typing", "typing.List", "typing.Optional", "numpy", ".config", ".config.InvestorProfile",
        ".config.StrategyParams", ".config.verdict_from_score", ".regime", ".regime.RegimeSnapshot",
        ".regime.regime_adjustments", ".technicals", ".technicals.Beacon"}
    # an analysis and a ticket never import the watch list or the scorecard
    code = ("import sys\nfrom lighthouse import analyze\nfrom lighthouse.tradeplan import build_trade_plan\n"
            "r = analyze('MSFT', source='sample', years=6, run_backtest_=False)\n"
            "build_trade_plan(r, 'long', allow_stale=True)\n"
            "print(sorted(m for m in sys.modules if 'watch' in m or m == 'lighthouse.scorecard'))\n")
    env = {**os.environ, "LIGHTHOUSE_OFFLINE": "1", "LIGHTHOUSE_ALLOW_STALE_DATA": "1", "MPLBACKEND": "Agg"}
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"
    # and the verdict and the ticket are the same with a journal that ranks MSFT top on every watch signal
    from lighthouse import analyze
    from lighthouse.tradeplan import build_trade_plan
    monkeypatch.setenv("LIGHTHOUSE_JOURNAL", str(tmp_path / "none.sqlite"))
    r0 = analyze("MSFT", source="sample", years=6, run_backtest_=False)
    p0 = build_trade_plan(r0, "long", allow_stale=True)
    j = Journal(tmp_path / "watch.sqlite")
    for name, spec in W.WATCH_SIGNALS.items():
        j.append_watch_rows([{**_row("MSFT", str(r0.decision.as_of)[:10], name, 9.9), "config_hash": spec.config_hash,
                              "quintile": 5, "pct": 1.0}])
    monkeypatch.setenv("LIGHTHOUSE_JOURNAL", str(j.path))
    r1 = analyze("MSFT", source="sample", years=6, run_backtest_=False)
    p1 = build_trade_plan(r1, "long", allow_stale=True)
    assert (r1.decision.verdict, r1.decision.composite_score, r1.decision.technical_score) == \
        (r0.decision.verdict, r0.decision.composite_score, r0.decision.technical_score)
    assert (p1.status, p1.stop, p1.shares, p1.entry_limit) == (p0.status, p0.stop, p0.shares, p0.entry_limit)


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #
def test_journal_watch_command(monkeypatch, tmp_path, capsys):
    from lighthouse.__main__ import main
    path = tmp_path / "j.sqlite"
    monkeypatch.setattr(W, "utc_now", lambda: pd.Timestamp("2026-10-29 22:30", tz="UTC"))
    assert main(["journal", "watch", "--due"]) == 0 and capsys.readouterr().out.strip() == "false"
    assert main(["journal", "watch", "--journal", str(path)]) == 0
    assert "Nothing to record" in capsys.readouterr().out and not path.exists()
    monkeypatch.setattr(W, "utc_now", lambda: pd.Timestamp("2026-10-30 22:30", tz="UTC"))
    assert main(["journal", "watch", "--due"]) == 0 and capsys.readouterr().out.strip() == "true"
    # due, but offline: the snapshot needs live data, and nothing is written
    assert main(["journal", "watch", "--journal", str(path)]) == 1
    assert "needs live data" in capsys.readouterr().out and Journal(path).watch_signals().empty
    assert main(["journal", "watch", "MSFT", "--journal", str(path)]) == 1          # no tickers: the whole index


def test_journal_score_has_a_watch_only_section(tmp_path, capsys):
    from lighthouse.__main__ import main
    frames = _frames()
    j = Journal(tmp_path / "j.sqlite")
    for day in ("2026-08-31", "2026-09-30", "2026-10-30", "2026-11-30"):
        _record(j, day, frames)
    out = tmp_path / "out"
    assert main(["journal", "score", "--journal", str(j.path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "WATCH-ONLY SIGNALS" in text and "no weight" in text and "toy" in text
    card = pd.read_csv(out / "watch_scorecard.csv")
    assert set(card["rule"]) == {"exclude", "pessimistic"} and (card["months"] == 3).all()
    assert len(pd.read_csv(out / "watch_snapshots.csv")) == 4
    # an empty journal scores too (the daily workflow runs `journal score` before any watch snapshot exists)
    assert main(["journal", "score", "--journal", str(tmp_path / "empty.sqlite")]) == 0
    assert "no watch snapshots yet" in capsys.readouterr().out
