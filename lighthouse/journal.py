"""
The forward decision journal: every verdict and ticket Lighthouse produces, written down the day it is made and never
edited. Forward results cannot contain look-ahead, cost nothing and become more valuable every month, so they are the
primary evidence for (or against) the rules.

Storage: one SQLite file (default data/journal/lighthouse_journal.sqlite, gitignored; LIGHTHOUSE_JOURNAL moves it) with
these tables, all append-only (SQLite triggers abort any UPDATE or DELETE):

  decisions  one row per (ticker, side) per run: the verdict, the scores, the ticket's levels (entry, stop, targets or
             trend exit, time stop), size, the forecast it implies (`p_win`: the ticket rules' historical win rate on
             that stock), the exact inputs (profile, parameters, beacon readings) and the data version (price source,
             last bar, a hash of the closes, the latest filing used, the code commit)
  events     what happened afterwards: a virtual position opened at the next session's open, an exit fired (stop,
             target, trend or signal review, time stop), an alert, a paper-trading order or fill, a reconciliation
  watch_signals   (schema 2) the watch-only research signals (lighthouse/research/watch.py): one row per (signal,
             ticker, month end) for every point-in-time S&P 500 member, with its value, cross-sectional rank and
             percentile, the configuration hash of the logged research trial, the data and code version
  watch_outcomes  (schema 2) each member's total return from a month end to the next, under both delisting rules,
             recorded once the next month end has closed (what the watch signals are scored against)

The watch tables are read only by the scorecard (lighthouse/scorecard.py, `score_watch`); no verdict or ticket uses
them.

A decision is a *virtual position* when its ticket was actionable with shares: the journal replays it from the session
after its signal with the ticket's own rules (tradeplan.monitor_position), exactly as the backtest would, and records
the exit when one fires. Paper trading (lighthouse.broker) mirrors the same tickets at Alpaca and records real fills.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import REPO_DIR

DEFAULT_PATH = REPO_DIR / "data" / "journal" / "lighthouse_journal.sqlite"
# 1: decisions and events. 2 (October 2026): + watch_signals and watch_outcomes. A file written by an older version is
# upgraded when opened (the new tables are created; no existing row is touched) and its meta row says so.
SCHEMA_VERSION = 2
HORIZON_DAYS = {"long": 252, "short": 21}          # the window a verdict is scored over (sessions)
# A stock's own setup history is a handful of trades (3-20): its raw win rate is a poor probability (3 wins in 3 trades
# is not certainty). The forecast shrinks it toward the pooled win rate of the same exit plan on the 20 sample stocks,
# 1990-2022 (docs/METHODOLOGY.md section 9: trend plan 45%, R targets 63%), weighted as PRIOR_TRADES trades.
POOLED_WIN_RATE = {"trend": 0.45, "targets": 0.63}
PRIOR_TRADES = 10


def shrunk_win_rate(win_rate: Optional[float], n: Optional[int], exit_style: str) -> Optional[float]:
    """(wins + PRIOR_TRADES x pooled rate) / (trades + PRIOR_TRADES): the ticket's forecast of ending with a profit."""
    p0 = POOLED_WIN_RATE.get(exit_style, 0.5)
    if win_rate is None or not n or not np.isfinite(win_rate):
        return p0
    return float((win_rate * n + PRIOR_TRADES * p0) / (n + PRIOR_TRADES))

DECISION_COLUMNS = [
    "recorded_at", "as_of", "ticker", "horizon", "risk", "side", "verdict", "status", "composite", "fundamental",
    "technical", "regime", "price", "priced_from", "entry_limit", "stop", "target1", "target2", "trend_exit",
    "signal_exit", "exit_style", "time_stop", "shares", "notional", "risk_usd", "account", "cash", "p_win",
    "hist_trades", "horizon_days", "inputs", "data_version", "code_version", "run_id", "note"]
EVENT_COLUMNS = ["recorded_at", "decision_id", "ticker", "kind", "date", "price", "shares", "detail", "run_id"]
WATCH_COLUMNS = ["recorded_at", "as_of", "signal", "ticker", "value", "rank", "pct", "quintile", "n_ranked", "tradable",
                 "cost", "config_hash", "data_version", "code_version", "run_id"]
OUTCOME_COLUMNS = ["recorded_at", "as_of", "end_date", "ticker", "ret_exclude", "ret_pessimistic", "data_version",
                   "code_version", "run_id"]

_SCHEMA = """
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
# Schema 2: the watch-only signals and their outcomes. One row per (signal, configuration, month end, ticker) and per
# (month end, ticker): a second recording of the same thing is refused by the UNIQUE constraints.
_SCHEMA_WATCH = """
CREATE TABLE IF NOT EXISTS watch_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL, as_of TEXT NOT NULL, signal TEXT NOT NULL,
  ticker TEXT NOT NULL, value REAL, rank REAL, pct REAL, quintile INTEGER, n_ranked INTEGER, tradable INTEGER,
  cost REAL, config_hash TEXT NOT NULL, data_version TEXT, code_version TEXT, run_id TEXT,
  UNIQUE (signal, config_hash, as_of, ticker));
CREATE TABLE IF NOT EXISTS watch_outcomes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL, as_of TEXT NOT NULL, end_date TEXT NOT NULL,
  ticker TEXT NOT NULL, ret_exclude REAL, ret_pessimistic REAL, data_version TEXT, code_version TEXT, run_id TEXT,
  UNIQUE (as_of, ticker));
CREATE INDEX IF NOT EXISTS watch_signals_date ON watch_signals (signal, as_of);
CREATE TRIGGER IF NOT EXISTS watch_signals_no_update BEFORE UPDATE ON watch_signals
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS watch_signals_no_delete BEFORE DELETE ON watch_signals
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS watch_outcomes_no_update BEFORE UPDATE ON watch_outcomes
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS watch_outcomes_no_delete BEFORE DELETE ON watch_outcomes
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
"""


class JournalVersionError(RuntimeError):
    """The journal file was written by a newer version of Lighthouse than this one."""


def journal_path(path=None) -> Path:
    return Path(path or os.environ.get("LIGHTHOUSE_JOURNAL") or DEFAULT_PATH)


def code_version() -> str:
    """The git commit the code runs from (GITHUB_SHA in Actions), with the package version."""
    from . import __version__
    sha = os.environ.get("GITHUB_SHA", "")
    if not sha:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001 - not a git checkout
            sha = ""
    return f"lighthouse {__version__} @ {sha[:12] or 'unknown'}"


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


class Journal:
    """The append-only store. `record_*` add rows; nothing changes or removes one."""

    def __init__(self, path=None) -> None:
        self.path = journal_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            found = _stored_schema_version(c)
            if found is not None and found > SCHEMA_VERSION:
                raise JournalVersionError(f"{self.path} has journal schema {found}; this version of Lighthouse knows "
                                          f"schema {SCHEMA_VERSION} at most: update Lighthouse before writing to it")
            c.executescript(_SCHEMA)
            c.executescript(_SCHEMA_WATCH)                 # schema 2; creating tables never touches existing rows
            c.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', ?)", (_now(),))
            if found is None:
                c.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif found < SCHEMA_VERSION:
                # meta is the one table that is not append-only: it describes the file, not what was recorded
                c.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))
                c.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
                          (f"upgraded_{found}_to_{SCHEMA_VERSION}", _now()))

    def schema_version(self) -> int:
        with self._conn() as c:
            return int(_stored_schema_version(c) or 0)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(str(self.path), timeout=30)
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ------------------------------------------------------------------ writing
    def append_decision(self, row: Dict[str, object]) -> int:
        cols = [k for k in DECISION_COLUMNS if k in row]
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO decisions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                            [_sql(row[k]) for k in cols])
            return int(cur.lastrowid)

    def append_event(self, decision_id: Optional[int], ticker: str, kind: str, date=None, price=None, shares=None,
                     detail: Optional[dict] = None, run_id: str = "") -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO events (recorded_at, decision_id, ticker, kind, date, price, shares, detail, run_id) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (_now(), decision_id, ticker, kind, None if date is None else str(pd.Timestamp(date).date()),
                             _sql(price), _sql(shares), json.dumps(detail or {}, default=str), run_id))
            return int(cur.lastrowid)

    def append_watch_rows(self, rows: Sequence[Dict[str, object]]) -> int:
        """One watch-signal snapshot (lighthouse/research/watch.py), written in one transaction: all its rows or none."""
        return self._append_many("watch_signals", WATCH_COLUMNS, rows)

    def append_watch_outcomes(self, rows: Sequence[Dict[str, object]]) -> int:
        """The month's outcomes of one snapshot date, in one transaction."""
        return self._append_many("watch_outcomes", OUTCOME_COLUMNS, rows)

    def _append_many(self, table: str, columns: Sequence[str], rows: Sequence[Dict[str, object]]) -> int:
        if not rows:
            return 0
        with self._conn() as c:
            c.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                          [[_sql(r.get(k)) for k in columns] for r in rows])
        return len(rows)

    # ------------------------------------------------------------------ reading
    def decisions(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM decisions ORDER BY id", c)

    def events(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM events ORDER BY id", c)

    def watch_signals(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM watch_signals ORDER BY id", c)

    def watch_outcomes(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM watch_outcomes ORDER BY id", c)

    def export(self, out_dir, fmt: str = "csv") -> List[Path]:
        """Write every table (CSV or JSON lines) to `out_dir`. Returns the paths."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, df in (("decisions", self.decisions()), ("events", self.events()),
                         ("watch_signals", self.watch_signals()), ("watch_outcomes", self.watch_outcomes())):
            p = out / f"journal_{name}.{'csv' if fmt == 'csv' else 'jsonl'}"
            if fmt == "csv":
                df.to_csv(p, index=False)
            else:
                df.to_json(p, orient="records", lines=True)
            paths.append(p)
        return paths

    def open_positions(self) -> pd.DataFrame:
        """Virtual positions opened and not yet closed: decisions with an 'opened' event and no 'exit' event."""
        d, e = self.decisions(), self.events()
        if d.empty or e.empty:
            return d.iloc[0:0]
        opened = set(e.loc[e["kind"] == "opened", "decision_id"].dropna().astype(int))
        closed = set(e.loc[e["kind"] == "exit", "decision_id"].dropna().astype(int))
        return d[d["id"].isin(opened - closed)]


def _stored_schema_version(c: sqlite3.Connection) -> Optional[int]:
    """The schema version a journal file records (None for a new file)."""
    if not c.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone():
        return None
    row = c.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row[0]) if row and str(row[0]).isdigit() else None


def _sql(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str)
    return v


# --------------------------------------------------------------------------- #
# What gets written
# --------------------------------------------------------------------------- #
def data_version(result) -> Dict[str, object]:
    """Enough to tell later exactly which data a decision saw."""
    px = result.prices
    tail = px["Close"].tail(300).to_numpy(dtype=float)
    h = hashlib.sha1(np.round(tail, 6).tobytes()).hexdigest()[:16]
    attrs = getattr(px, "attrs", {}) or {}
    rf = getattr(result, "risk_free", None)
    return {"price_source": attrs.get("source"), "last_bar": str(px.index[-1].date()), "rows": int(len(px)),
            "closes_sha1": h,
            "risk_free_last": None if rf is None or not len(rf) else str(rf.index[-1].date()),
            "quote": result.quote.to_dict() if getattr(result, "quote", None) is not None else None}


def decision_row(result, plan, run_id: str = "", note: str = "") -> Dict[str, object]:
    """One journal row from an analysis and its ticket (TradePlan)."""
    d, prof, p = result.decision, result.profile, result.params
    beacons = {b.key: (None if not np.isfinite(b.score) else round(float(b.score), 4)) for b in d.beacons}
    t1 = plan.targets[0].price if plan.targets else None
    t2 = plan.targets[1].price if len(plan.targets) > 1 else None
    st = plan.stats
    return {
        "recorded_at": _now(), "as_of": d.as_of, "ticker": d.ticker, "horizon": plan.horizon, "risk": prof.risk,
        "side": plan.side, "verdict": d.verdict, "status": plan.status, "composite": d.composite_score,
        "technical": d.technical_score, "regime": plan.regime, "price": plan.price,
        "priced_from": plan.priced_from, "entry_limit": plan.entry_limit, "stop": plan.stop, "target1": t1, "target2": t2,
        "trend_exit": plan.trend_exit_level, "signal_exit": plan.signal_exit_score, "exit_style": plan.exit_style,
        "time_stop": plan.time_stop_days, "shares": plan.shares, "notional": plan.notional, "risk_usd": plan.dollars_at_risk,
        "account": plan.account_value, "cash": plan.cash_available,
        "p_win": shrunk_win_rate(None if st is None else st.win_rate, None if st is None else st.n_trades, plan.exit_style),
        "hist_trades": None if st is None else st.n_trades,
        "horizon_days": HORIZON_DAYS.get(prof.horizon, 252),
        "inputs": {"profile": prof.to_dict(), "params": p.to_dict(), "beacons": beacons,
                   "hist_win_rate": None if st is None else st.win_rate, "hist_avg_r": None if st is None else st.avg_r,
                   "data_status": plan.data_status, "warnings": plan.warnings[:6]},
        "data_version": data_version(result), "code_version": code_version(), "run_id": run_id, "note": note,
    }


# --------------------------------------------------------------------------- #
# The daily run
# --------------------------------------------------------------------------- #
@dataclass
class RunReport:
    run_id: str
    recorded: int = 0
    opened: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    refused: Dict[str, str] = field(default_factory=dict)
    paper: List[str] = field(default_factory=list)

    def markdown(self) -> str:
        lines = [f"## Lighthouse journal run {self.run_id}", "",
                 f"* decisions recorded: {self.recorded}", f"* virtual positions opened: {', '.join(self.opened) or 'none'}"]
        if self.alerts:
            lines += ["", "### Alerts (exits that fired on journaled positions)", ""] + [f"* {a}" for a in self.alerts]
        if self.paper:
            lines += ["", "### Paper trading", ""] + [f"* {x}" for x in self.paper]
        if self.refused:
            lines += ["", "### Refused (data not current)", ""] + [f"* {t}: {w}" for t, w in self.refused.items()]
        if self.errors:
            lines += ["", "### Errors", ""] + [f"* {t}: {e}" for t, e in self.errors.items()]
        return "\n".join(lines)


def record_run(results: Sequence, sides: Sequence[str] = ("long",), journal: Optional[Journal] = None,
               settings=None, run_id: Optional[str] = None, allow_stale: Optional[bool] = None,
               open_positions: bool = True) -> RunReport:
    """Write today's decision for every analysed ticker and side, and open a virtual position for each actionable
    ticket that has shares (one open position per ticker and side at a time, as in the backtest). Stale data is
    refused, never journaled."""
    from .tradeplan import StaleDataError, build_trade_plan
    j = journal or Journal()
    run_id = run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    rep = RunReport(run_id)
    already = j.open_positions()
    held = set(zip(already["ticker"], already["side"])) if len(already) else set()
    for r in results:
        for side in sides:
            try:
                plan = build_trade_plan(r, side, settings, with_stats=True, allow_stale=allow_stale)
            except StaleDataError as exc:
                rep.refused[r.ticker] = str(exc)
                continue
            except Exception as exc:  # noqa: BLE001 - one ticker must not stop the run
                rep.errors[f"{r.ticker} {side}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                continue
            did = j.append_decision(decision_row(r, plan, run_id))
            rep.recorded += 1
            if open_positions and plan.is_actionable and (r.ticker, side) not in held:
                j.append_event(did, r.ticker, "opened", date=plan.as_of, price=plan.price, shares=plan.shares,
                               detail={"rule": "fills at the next session's open", "stop": plan.stop,
                                       "exit_style": plan.exit_style}, run_id=run_id)
                held.add((r.ticker, side))
                rep.opened.append(f"{side} {r.ticker} {plan.shares} sh")
    return rep


def check_positions(results: Dict[str, object], journal: Optional[Journal] = None, settings=None,
                    run_id: str = "") -> List[Dict[str, object]]:
    """Replay every open virtual position with its ticket's rules and record the exits that fired (stop, target 2,
    trend or signal review, time stop) and the alerts. Returns one dict per new event: decision_id, ticker, side,
    kind ("exit" | "alert"), status, shares, stop_now, text (plain English)."""
    from .tradeplan import monitor_position
    j = journal or Journal()
    alerts = []
    for _, row in j.open_positions().iterrows():
        r = results.get(row["ticker"])
        if r is None:
            continue
        entry_day = _next_session(r.indicators.index, pd.Timestamp(row["as_of"]))
        if entry_day is None:
            continue                                        # the entry session has not happened yet
        entry_px = float(r.indicators.loc[entry_day, "Open"])
        try:
            ms = monitor_position(r, row["side"], entry_px, int(row["shares"]), entry_day, stop=row["stop"],
                                  targets=[row["target1"], row["target2"]] if row["exit_style"] == "targets"
                                  and pd.notna(row["target1"]) else None, settings=settings)
        except ValueError as exc:
            alerts.append({"decision_id": int(row["id"]), "ticker": row["ticker"], "side": row["side"], "kind": "error",
                           "status": "ERROR", "shares": int(row["shares"]), "stop_now": None,
                           "text": f"{row['ticker']}: cannot replay the position ({exc})"})
            continue
        # closed when the replay shows every share sold (stop, target, trend or signal review filled at the open), or
        # when the time stop is due (the ticket says close at the market); a review that says "sell at the next open",
        # a partial target and a near stop are alerts. Each status is recorded once per position.
        if ms.shares_open == 0 or ms.status == "TIME STOP DUE":
            kind = "exit"
        elif ms.status in {"SELL AT THE OPEN", "NEAR STOP", "TARGET 1 HIT"}:
            kind = "alert"
        else:
            continue
        closing = ms.exits[-1] if ms.exits and ms.shares_open == 0 else None
        seen = j.events()
        seen = seen[(seen["decision_id"] == row["id"]) & (seen["kind"] == kind)]
        if any(json.loads(x or "{}").get("status") == ms.status for x in seen["detail"]):
            continue
        j.append_event(int(row["id"]), row["ticker"], kind, date=closing[0] if closing else ms.as_of,
                       price=closing[2] if closing else ms.last_price, shares=int(row["shares"]),
                       detail={"status": ms.status, "action": ms.action, "pnl": ms.pnl, "r": ms.r_multiple,
                               "entry": entry_px, "entry_date": str(entry_day.date())}, run_id=run_id)
        alerts.append({"decision_id": int(row["id"]), "ticker": row["ticker"], "side": row["side"], "kind": kind,
                       "status": ms.status, "shares": int(ms.shares_open), "stop_now": ms.current_stop,
                       "text": f"{row['side']} {row['ticker']} (journal #{row['id']}): {ms.status}. {ms.action}"})
    return alerts


def _next_session(index: pd.DatetimeIndex, day: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = index.searchsorted(day, side="right")
    return index[pos] if pos < len(index) else None


__all__ = ["Journal", "decision_row", "data_version", "record_run", "check_positions", "RunReport", "journal_path",
           "HORIZON_DAYS", "code_version", "SCHEMA_VERSION", "WATCH_COLUMNS", "OUTCOME_COLUMNS", "JournalVersionError"]
