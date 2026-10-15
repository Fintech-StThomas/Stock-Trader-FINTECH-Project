"""
The forward watch list: the research harness's near misses, recorded in the forward journal at every month end and
scored on data that did not exist when they were tested (docs/METHODOLOGY.md section 8c).

Watch-only. None of these signals passed the pre-registered survival rule (METHODOLOGY section 8b), so none carries any
weight: nothing in the verdicts, the tickets or lighthouse/decision.py reads this module or its tables
(tests/test_watch.py checks it), and a good forward score does not change that. It can only make the case for a new
pre-registered test, which is the user's decision.

  fcf_yield        free cash flow / market value: passed tuning (IC +0.024, Newey-West t 2.3), hold-out +0.009 (t 0.7)
  short_volume     minus the short share of FINRA-reported volume over the 5 sessions before the month's last: IC t 3.5
                   in tuning and 1.9 in the hold-out, but its top quintile turns over 60-66% a month
  risk_similarity  Item 1A (risk factors) similarity with the previous 10-K (Lazy Prices): flat in tuning, t 2.3 in the
                   hold-out only

Each is computed exactly as the research run logged it in reports/research/experiments.csv: the same functions (the
harness's fundamental, positioning and text code), the same parameters and the same configuration, whose config_hash
is written on every row, so the forward rows are more data for the same trial, not a new trial. A variant (for example
short volume rebalanced quarterly) is a new configuration: log it as a new trial first, then add a WatchSignal under
its own name.

Timing (no look-ahead, no backdating)
  * The snapshot date is the last finished NYSE session, and only the month's last session counts: on any other day
    the recorder does nothing, and it takes no date argument. Everything used was public by that close: prices to it,
    a filing from the session after its filing date, FINRA files of the sessions before it.
  * A snapshot is forward evidence only if it was written before the next session's open (9:30 New York: 13:30 UTC in
    summer, 14:30 UTC in winter), that is before any return it predicts exists. A late one stays in the journal, is
    reported as late and is not scored.
  * Each member's outcome (its total return to the next month's last session, under both delisting rules) is recorded
    once that session has closed; nothing is scored before its outcome is recorded.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import store
from ..data import EXCHANGE_TZ, SESSION_OPEN, DataUnavailable, is_trading_day, last_session_close, offline_mode
from ..total_return import DELISTING_RULES
from . import harness as H

PRICE_YEARS = 3              # price history read for each member (enough for costs, the volume screen and FCF yield)
MIN_SESSIONS = 260           # scripts/research_signals.load_inputs keeps series with more than 260 sessions only
MIN_CURRENT = 0.90           # share of priced members whose prices must reach the snapshot date
OUTCOME_LOOKBACK_DAYS = 45   # calendar window before a snapshot date that the outcome panel is built on
TEXT_PROCESSES = 3           # 10-K parsing is CPU-bound: worker processes sharing the SEC rate (as the harness does)


# --------------------------------------------------------------------------- #
# The NYSE clock
# --------------------------------------------------------------------------- #
def sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions (weekdays that are not full-day holidays) from `start` to `end`, both included."""
    days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    return pd.DatetimeIndex([d for d in days if is_trading_day(d)])


def month_end_session(day) -> pd.Timestamp:
    """The last NYSE session of the month that `day` falls in."""
    d = pd.Timestamp(day).normalize()
    last = d + pd.offsets.MonthEnd(0)
    while not is_trading_day(last):
        last -= pd.Timedelta(days=1)
    return last


def is_month_end_session(day) -> bool:
    """True when `day` is the last NYSE session of its month (Good Friday or Memorial Day can move it earlier)."""
    d = pd.Timestamp(day).normalize()
    return is_trading_day(d) and d == month_end_session(d)


def next_month_end_session(day) -> pd.Timestamp:
    """The last NYSE session of the month after the one `day` falls in."""
    return month_end_session(pd.Timestamp(day).normalize() + pd.offsets.MonthBegin(1))


def next_session(day) -> pd.Timestamp:
    d = pd.Timestamp(day).normalize() + pd.Timedelta(days=1)
    while not is_trading_day(d):
        d += pd.Timedelta(days=1)
    return d


def next_open_utc(as_of) -> pd.Timestamp:
    """The opening bell (9:30 New York) of the session after `as_of`, in UTC: the deadline of a forward snapshot."""
    d = next_session(as_of)
    bell = (d + pd.Timedelta(hours=SESSION_OPEN[0], minutes=SESSION_OPEN[1])).tz_localize(EXCHANGE_TZ)
    return bell.tz_convert("UTC")


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def last_finished_session(now=None) -> pd.Timestamp:
    """The date of the last NYSE session that has closed at `now` (default: the clock)."""
    now = _utc(now) if now is not None else utc_now()
    return last_session_close(now).tz_localize(None).normalize()


def is_due(now=None) -> bool:
    """True when the last finished session at `now` is the last session of its month: a watch snapshot is due."""
    return is_month_end_session(last_finished_session(now))


def on_time(recorded_at, as_of) -> bool:
    """A snapshot is forward evidence only if it was written before the next session's open."""
    return _utc(recorded_at) < next_open_utc(as_of)


# --------------------------------------------------------------------------- #
# The signals
# --------------------------------------------------------------------------- #
@dataclass
class WatchInputs:
    """What a snapshot is computed from: everything public at the close of `as_of`."""
    as_of: pd.Timestamp
    members: List[str]                                     # every point-in-time S&P 500 member on as_of, priced or not
    frames: Dict[str, pd.DataFrame]                        # daily OHLCV through as_of (members with > 260 sessions;
    #                                                        former members too when an outcome needs them)
    source: str = "auto"
    info: Dict[str, object] = field(default_factory=dict)  # data version: membership, prices

    def __post_init__(self) -> None:
        # nothing after the snapshot's close can reach it, whatever the caller passes
        self.as_of = pd.Timestamp(self.as_of).normalize()
        self.frames = {t: f[f.index <= self.as_of] for t, f in self.frames.items()}
        self.frames = {t: f for t, f in self.frames.items() if len(f)}

    def priced_members(self) -> List[str]:
        """The members in the research panel on as_of: priced, with more than MIN_SESSIONS sessions."""
        return sorted(t for t in self.members if t in self.frames and len(self.frames[t]) > MIN_SESSIONS)


@dataclass
class SignalValues:
    values: pd.Series                                       # ticker -> value on as_of (NaN: not known)
    info: Dict[str, object] = field(default_factory=dict)   # what the values rest on (data version)
    detail: Dict[str, dict] = field(default_factory=dict)   # ticker -> its own data version (e.g. the 10-K used)
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WatchSignal:
    name: str
    description: str               # as logged (part of the configuration, so of its hash)
    config_hash: str               # the logged trial's hash in reports/research/experiments.csv
    use: str                       # the better use the tuning results chose: "long_only" or "long_short"
    compute: Callable[[WatchInputs], SignalValues]
    why: str = ""

    @property
    def config(self) -> Dict[str, object]:
        return H.signal_config(self.name, self.description)


def _at(daily: Dict[str, pd.Series], as_of, tickers) -> pd.Series:
    """Daily signals read on the snapshot date exactly as the harness reads them at a month end."""
    m = H.sample_at_month_ends(daily, pd.DatetimeIndex([pd.Timestamp(as_of)]))
    out = m.iloc[0] if len(m.columns) else pd.Series(dtype=float)
    return out.reindex(list(tickers)).astype(float)


def fcf_yield_values(inp: WatchInputs) -> SignalValues:
    """(operating cash flow - capex, trailing twelve months) / market value, from the session after each filing:
    research/fundamental_signals.daily_values on prices through as_of, read as the harness reads it."""
    from ..parallel import map_tickers
    from . import fundamental_signals as FS
    tickers = inp.priced_members()
    got, errors = map_tickers(lambda t: FS.daily_values(t, inp.frames[t], inp.source).get("fcf_yield"), tickers,
                              max_workers=8)
    if errors and len(errors) > 0.5 * max(len(tickers), 1):        # as FS.monthly: a bug must not pass silently
        raise RuntimeError(f"fcf_yield failed for {len(errors)} of {len(tickers)} stocks, e.g. "
                           + "; ".join(f"{k}: {v}" for k, v in list(errors.items())[:3]))
    daily = {t: s for t, s in got.items() if s is not None}
    return SignalValues(_at(daily, inp.as_of, tickers),
                        {"filings": "SEC XBRL company facts, each from the session after its filing date",
                         "with_statements": len(daily), "errors": len(errors)}, errors=errors)


def short_volume_values(inp: WatchInputs) -> SignalValues:
    """Minus the short share of FINRA-reported volume over the 5 sessions before the month's last:
    research/positioning.short_volume_signal at the snapshot date."""
    from . import positioning as PO
    tickers = inp.priced_members()
    sig = PO.short_volume_signal(pd.DatetimeIndex([inp.as_of]), tickers)
    before = sessions(inp.as_of - pd.Timedelta(days=14), inp.as_of - pd.Timedelta(days=1))[-5:]
    found = [str(d.date()) for d in before if PO.short_volume_day(d) is not None]      # cached by the call above
    return SignalValues(sig.iloc[0].reindex(tickers).astype(float),
                        {"finra_sessions": [str(d.date()) for d in before], "finra_files_found": found})


def risk_similarity_values(inp: WatchInputs, processes: int = TEXT_PROCESSES) -> SignalValues:
    """Cosine similarity of the latest 10-K's Item 1A words with the previous 10-K's, from the session after its
    filing date, kept 15 months: the harness's text features (research/event_signals) for the latest 10-K known on
    as_of, computed as text_history computes it (see known_tenk_rows)."""
    from . import event_signals as EV
    col, sign = EV.TEXT_FEATURES["risk_similarity"]
    tickers = inp.priced_members()
    rows, errors = tenk_rows_for(tickers, inp.as_of, inp.source, processes)
    if errors and len(errors) > 0.5 * max(len(tickers), 1):        # as EV.text_monthly
        raise RuntimeError(f"risk_similarity failed for {len(errors)} of {len(tickers)} stocks, e.g. "
                           + "; ".join(f"{k}: {v}" for k, v in list(errors.items())[:3]))
    daily, detail = {}, {}
    for t, h in rows.items():
        if h is None or not len(h) or col not in h:
            continue
        daily[t] = EV.text_signal_daily(h, inp.frames[t].index, col, sign)
        detail[t] = {"10-K": str(h["accn"].iloc[-1]), "filed": str(pd.Timestamp(h.index[-1]).date())}
    return SignalValues(_at(daily, inp.as_of, tickers),
                        {"filings": "latest 10-K filed before the snapshot date, Item 1A vs the 10-K before it",
                         "with_10k": len(daily), "errors": len(errors)}, detail, errors)


WATCH_SIGNALS: Dict[str, WatchSignal] = {
    "fcf_yield": WatchSignal(
        "fcf_yield", "FCF / market value", "a86f74152373", "long_short", fcf_yield_values,
        "passed tuning (IC +0.024, NW t 2.3); hold-out IC +0.009 (t 0.7)"),
    "short_volume": WatchSignal(
        "short_volume", "minus the short share of reported volume (5 sessions)", "84d37c55a0d9", "long_only",
        short_volume_values, "IC NW t 3.5 (tuning) and 1.9 (hold-out), but 60-66% monthly turnover"),
    "risk_similarity": WatchSignal(
        "risk_similarity", "Item 1A risk-factor similarity with the previous 10-K", "7207f773a97c", "long_only",
        risk_similarity_values, "flat in tuning; hold-out IC +0.011 (NW t 2.3)"),
}


# --------------------------------------------------------------------------- #
# 10-K text, incrementally
# --------------------------------------------------------------------------- #
def _has_doc(d) -> bool:
    return isinstance(d, str) and bool(d)


def known_tenk_rows(ticker: str, as_of, source: str = "auto", lm=None) -> pd.DataFrame:
    """The row of event_signals.text_history (10-K text features, indexed by filing date) that the text signals use on
    `as_of`: the latest 10-K filed before it (a filing counts from the session after its filing date), compared with
    the 10-K before it, exactly as text_history computes it. Read from the research run's cache when that holds the
    filing, else from this module's cache (watch_text), else computed: the two documents are read and not stored, only
    the numbers are cached. Empty when no 10-K is known."""
    from ..edgar import cik_for, company_filings, fetch_document, registrant_cik
    from ..text import features, lm_dictionary
    from . import event_signals as EV
    as_of = pd.Timestamp(as_of)
    _, filings = company_filings(ticker, source=source)
    empty = pd.DataFrame(columns=["accn"]).rename_axis("filed")
    if filings is None or filings.empty:
        return empty
    tenk = EV.tenk_filings(filings)
    tenk = tenk[tenk["doc"].map(_has_doc)]                          # text_history skips filings without a document
    tenk = tenk[pd.to_datetime(tenk["filed"]) + pd.Timedelta(days=1) <= as_of]
    if tenk.empty:
        return empty
    last = tenk.iloc[-1]
    for kind in ("research_text", "watch_text"):
        hit = store.read_frame(kind, ticker)
        if hit is not None and "accn" in hit and (hit["accn"] == last["accn"]).any():
            return hit[hit["accn"] == last["accn"]].tail(1)
    lm = lm if lm is not None else lm_dictionary()
    cik = cik_for(ticker)
    prev = None
    if len(tenk) > 1:
        p = tenk.iloc[-2]
        prev = features(fetch_document(registrant_cik(p, cik), p["accn"], p["doc"], cache=False), lm)[1]
    feats, _ = features(fetch_document(registrant_cik(last, cik), last["accn"], last["doc"], cache=False), lm, prev)
    row = pd.DataFrame([{"filed": pd.Timestamp(last["filed"]), "accn": last["accn"], **feats}]).set_index("filed")
    old = store.read_frame("watch_text", ticker)
    store.write_frame("watch_text", ticker, row if old is None or old.empty else pd.concat([old, row]))
    return row


def _tenk_one(args) -> tuple:
    ticker, as_of, source = args
    try:
        return ticker, known_tenk_rows(ticker, as_of, source), None
    except Exception as exc:  # noqa: BLE001 - one company's odd filing must not stop the snapshot
        return ticker, None, f"{type(exc).__name__}: {str(exc)[:160]}"


def tenk_rows_for(tickers: Sequence[str], as_of, source: str = "auto",
                  processes: int = TEXT_PROCESSES) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """known_tenk_rows for many companies: in worker processes that share this run's SEC rate (as the harness's text
    workers do), or in this process when `processes` is 1."""
    jobs = [(t, pd.Timestamp(as_of), source) for t in tickers]
    rows: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    if processes <= 1 or len(jobs) < 2:
        results = map(_tenk_one, jobs)
    else:
        from ..text import lm_dictionary
        from . import event_signals as EV
        lm_dictionary()                                            # download once, before the workers start
        ex = ProcessPoolExecutor(max_workers=processes, initializer=EV._worker_init,
                                 initargs=(EV.worker_sec_rate(processes),))
        results = ex.map(_tenk_one, jobs, chunksize=4)
    try:
        for t, h, err in results:
            if err:
                errors[t] = err
            elif h is not None:
                rows[t] = h
    finally:
        if processes > 1 and len(jobs) >= 2:
            ex.shutdown()
    return rows, errors


# --------------------------------------------------------------------------- #
# Inputs (live)
# --------------------------------------------------------------------------- #
def load_current_frames(tickers: Iterable[str], as_of, source: str = "auto",
                        max_workers: int = 8) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Daily OHLCV for `tickers` through `as_of`, PRICE_YEARS long. With the "auto" source a cached file is used only
    when it holds the last finished session (lighthouse.data.load_prices); otherwise Yahoo is asked, and throttled
    tickers are retried on two threads (as research/panel.load_frames does)."""
    from ..data import load_prices
    from ..parallel import map_tickers
    from .panel import _throttled
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.DateOffset(years=PRICE_YEARS)

    def one(t: str) -> pd.DataFrame:
        df = load_prices(t, start=start, years=None, source=source)
        df.attrs = {}
        return df[df.index <= as_of]
    got, err = map_tickers(one, list(tickers), max_workers=max_workers, source=source)
    retry = [t for t, e in err.items() if _throttled(e)]
    if retry:
        got2, err2 = map_tickers(one, retry, max_workers=2, source=source)
        got.update(got2)
        for t in retry:
            err.pop(t, None)
        err.update(err2)
    return {t: df for t, df in got.items() if df is not None and len(df)}, err


def live_inputs(as_of, extra: Iterable[str] = (), source: str = "auto") -> WatchInputs:
    """The point-in-time S&P 500 members on `as_of` (membership refreshed from its sources) and their prices through
    `as_of`; `extra` adds former members whose outcome is still to be recorded."""
    from .. import membership
    if offline_mode():
        raise DataUnavailable("the watch snapshot needs live data (Yahoo prices, SEC EDGAR, FINRA); offline mode is on")
    as_of = pd.Timestamp(as_of)
    uni = membership.sp500(refresh=True)
    members = uni.members_on(as_of)
    frames, missing = load_current_frames(sorted(set(members) | set(extra)), as_of, source)
    inp = WatchInputs(as_of, members, frames, source)
    priced = inp.priced_members()
    current = [t for t in priced if frames[t].index[-1] >= as_of]
    inp.info = {"universe": uni.name, "membership_as_of": str(uni.as_of.date()) if uni.as_of is not None else None,
                "members": len(members), "priced": len(priced), "prices_current": len(current),
                "current_share": round(len(current) / max(len(priced), 1), 4), "price_source": source,
                "prices_missing": len(set(members) - set(frames))}
    return inp


# --------------------------------------------------------------------------- #
# Snapshot and outcome rows
# --------------------------------------------------------------------------- #
def month_end_panel(frames: Dict[str, pd.DataFrame], start, end) -> Optional[H.Panel]:
    """The harness's panel (month-end prices, the volume screen, next-month returns under both delisting rules) on the
    NYSE calendar from `start` to `end`, for `frames` cut at `end`."""
    end = pd.Timestamp(end)
    cut = {t: f[f.index <= end] for t, f in frames.items()}
    cut = {t: f for t, f in cut.items() if len(f) and f["Close"].notna().any()}
    cal = sessions(start, end)
    if not cut or not len(cal):
        return None
    return H.build_panel(None, cut, start=cal[0], calendar=cal)


def snapshot_frame(inp: WatchInputs, values: pd.Series) -> pd.DataFrame:
    """One row per member on as_of: the value, whether the harness would test the stock that month (priced at the
    month end and traded on 15 of the last 21 sessions), its per-side cost under the harness's cost model, and its
    rank (1 = lowest), percentile and quintile among the tested members with a value."""
    as_of = pd.Timestamp(inp.as_of)
    members = sorted(inp.members)
    out = pd.DataFrame(index=pd.Index(members, name="ticker"))
    out["value"] = values.reindex(members).astype(float)
    priced = {t: inp.frames[t] for t in inp.priced_members()}
    tradable = pd.Series(np.nan, index=members)
    cost = pd.Series(np.nan, index=members)
    if priced:
        panel = month_end_panel(priced, as_of - pd.Timedelta(days=OUTCOME_LOOKBACK_DAYS), as_of)
        if panel is not None and panel.dates[-1] == as_of:
            tradable = panel.investable.loc[as_of].reindex(members).astype(float)
            tradable[~tradable.index.isin(list(priced))] = np.nan
        cost = H.cost_panel(priced, pd.DatetimeIndex([as_of])).iloc[0].reindex(members)
    out["tradable"] = tradable
    out["cost"] = cost
    ranked = out["value"].notna() & (out["tradable"] == 1.0)
    v = out.loc[ranked, "value"]
    out["rank"] = v.rank()
    out["pct"] = out["rank"] / max(len(v), 1)
    out["quintile"] = H._qbins(v, 5) if len(v) >= 5 else np.nan
    out["n_ranked"] = int(len(v))
    return out


def _num(x, cast=float):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else cast(x)


def _dumps(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def snapshot_rows(spec: WatchSignal, inp: WatchInputs, sv: SignalValues, recorded_at: str, run_id: str,
                  code: str) -> List[Dict[str, object]]:
    frame = snapshot_frame(inp, sv.values)
    common = {"prices": {k: inp.info.get(k) for k in ("price_source", "priced", "prices_current", "members")},
              "membership_as_of": inp.info.get("membership_as_of"), **sv.info}
    rows = []
    for t, r in frame.iterrows():
        dv = dict(common, **({"own": sv.detail[t]} if t in sv.detail else {}))
        rows.append({"recorded_at": recorded_at, "as_of": str(pd.Timestamp(inp.as_of).date()), "signal": spec.name,
                     "ticker": t, "value": _num(r["value"]), "rank": _num(r["rank"]), "pct": _num(r["pct"]),
                     "quintile": _num(r["quintile"], int), "n_ranked": _num(r["n_ranked"], int),
                     "tradable": _num(r["tradable"], int), "cost": _num(r["cost"]), "config_hash": spec.config_hash,
                     "data_version": _dumps(dv), "code_version": code, "run_id": run_id})
    return rows


def outcome_frame(frames: Dict[str, pd.DataFrame], members: Sequence[str], as_of, end) -> Optional[pd.DataFrame]:
    """Each member's total return from the close of `as_of` to the close of `end` (the next month's last session)
    under both delisting rules, by the harness's panel; None when the prices do not reach `end` yet."""
    as_of, end = pd.Timestamp(as_of), pd.Timestamp(end)
    have = {t: frames[t] for t in members if t in frames}
    panel = month_end_panel(have, as_of - pd.Timedelta(days=OUTCOME_LOOKBACK_DAYS), end) if have else None
    if panel is None or as_of not in panel.dates or panel.dates[-1] != end or panel.data_end != end:
        return None
    out = pd.DataFrame({f"ret_{rule}": panel.fwd[rule].loc[as_of] for rule in DELISTING_RULES})
    return out.reindex(sorted(members))


def snapshot_table(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per recorded snapshot (signal, configuration, month end): rows, members ranked, when it was written, its
    deadline (the next session's open) and whether it was on time."""
    cols = ["signal", "config_hash", "as_of", "rows", "ranked", "recorded_at", "deadline", "on_time"]
    if snaps is None or snaps.empty:
        return pd.DataFrame(columns=cols)
    g = snaps.groupby(["signal", "config_hash", "as_of"], sort=True)
    out = g.agg(rows=("ticker", "size"), ranked=("rank", "count"), recorded_at=("recorded_at", "max")).reset_index()
    out["deadline"] = [next_open_utc(d).isoformat() for d in out["as_of"]]
    out["on_time"] = [on_time(r, d) for r, d in zip(out["recorded_at"], out["as_of"])]
    return out[cols]


def pending_outcomes(snaps: pd.DataFrame, outcomes: pd.DataFrame, as_of) -> List[Tuple[pd.Timestamp, pd.Timestamp, List[str]]]:
    """(snapshot date, its next month end, members) for every snapshot date whose outcome is not recorded yet and
    whose next month end has closed by `as_of`."""
    if snaps is None or snaps.empty:
        return []
    done = set(outcomes["as_of"]) if outcomes is not None and len(outcomes) else set()
    out = []
    for d, g in snaps.groupby("as_of"):
        end = next_month_end_session(d)
        if d not in done and end <= pd.Timestamp(as_of):
            out.append((pd.Timestamp(d), end, sorted(set(g["ticker"]))))
    return out


# --------------------------------------------------------------------------- #
# The recorder
# --------------------------------------------------------------------------- #
@dataclass
class WatchReport:
    now: pd.Timestamp
    as_of: pd.Timestamp
    due: bool
    recorded: Dict[str, int] = field(default_factory=dict)       # signal -> rows written
    outcomes: Dict[str, int] = field(default_factory=dict)       # snapshot date -> outcome rows written
    already: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    run_id: str = ""

    def lines(self) -> List[str]:
        d = str(self.as_of.date())
        if not self.due:
            return [f"Nothing to record: the last finished session is {d}; the month's last is "
                    f"{month_end_session(self.as_of).date()} (watch snapshots are taken at month ends only)."]
        out = [f"Watch-only signals, month end {d} (not survivors of the research rule; no weight in any verdict):"]
        out += [f"  recorded {s}: {n} members" for s, n in self.recorded.items()]
        out += [f"  already recorded {s}" for s in self.already]
        out += [f"  outcomes of {k}: {n} members" for k, n in self.outcomes.items()]
        out += [f"  {x}" for x in self.notes]
        out += [f"  ERROR {k}: {v}" for k, v in self.errors.items()]
        if not (self.recorded or self.outcomes or self.errors):
            out.append("  nothing new to record")
        return out

    def markdown(self) -> str:
        return "\n".join(["## Lighthouse watch list " + str(self.as_of.date()), ""] + [f"* {x.strip()}" for x in self.lines()])


def record_watch(journal=None, now=None, clock: Optional[Callable[[], pd.Timestamp]] = None,
                 inputs: Optional[Callable[..., WatchInputs]] = None,
                 signals: Optional[Dict[str, WatchSignal]] = None, source: str = "auto") -> WatchReport:
    """Record the month-end snapshot of every watch signal and the outcomes that have become known, if the last
    finished session is a month end; otherwise do nothing (and open no journal). Append-only: a snapshot already
    recorded is not written again. `journal`: a Journal, a path, or None for the default file. `now`, `clock` and
    `inputs` exist for tests (a frozen clock, synthetic data); the command line passes none, so it cannot backdate."""
    from ..journal import Journal, code_version
    clock = clock or utc_now
    now = _utc(now) if now is not None else clock()
    as_of = last_finished_session(now)
    rep = WatchReport(now=now, as_of=as_of, due=is_month_end_session(as_of))
    if not rep.due:
        return rep
    j = journal if isinstance(journal, Journal) else Journal(journal)
    specs = signals if signals is not None else WATCH_SIGNALS
    snaps, outs = j.watch_signals(), j.watch_outcomes()
    d = str(as_of.date())
    have = set(zip(snaps["signal"], snaps["config_hash"], snaps["as_of"])) if len(snaps) else set()
    todo = [s for s in specs.values() if (s.name, s.config_hash, d) not in have]
    rep.already = [s.name for s in specs.values() if (s.name, s.config_hash, d) in have]
    pending = pending_outcomes(snaps, outs, as_of)
    if not todo and not pending:
        return rep
    rep.run_id = "watch-" + now.strftime("%Y%m%dT%H%M%SZ")
    extra = sorted({t for _, _, m in pending for t in m})
    try:
        inp = (inputs or live_inputs)(as_of, extra=extra, source=source)
    except Exception as exc:  # noqa: BLE001 - reported; nothing is written
        rep.errors["inputs"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return rep
    share = float(inp.info.get("current_share", 1.0))
    if not inp.members:
        rep.errors["members"] = f"no point-in-time S&P 500 members known for {d}: nothing recorded"
        return rep
    if share < MIN_CURRENT:
        rep.errors["prices"] = (f"only {share:.0%} of priced members have a price for {d} (at least {MIN_CURRENT:.0%} "
                                "needed): the feed has not published the close yet; nothing recorded, run again later")
        return rep
    code = code_version()
    for spec in todo:                                    # the snapshot first: it has a deadline (the next open)
        try:
            sv = spec.compute(inp)
        except Exception as exc:  # noqa: BLE001 - one signal's data source must not stop the others
            rep.errors[spec.name] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        rows = snapshot_rows(spec, inp, sv, clock().isoformat(timespec="seconds"), rep.run_id, code)
        rep.recorded[spec.name] = j.append_watch_rows(rows)
        if sv.errors:
            rep.notes.append(f"{spec.name}: {len(sv.errors)} members without a value because of an error "
                             f"(e.g. {next(iter(sv.errors.items()))[0]})")
        if not on_time(rows[0]["recorded_at"], as_of):
            rep.notes.append(f"{spec.name}: written after the next open ({next_open_utc(as_of).isoformat()}): late, "
                             "kept but not scored")
    for t, end, members in pending:
        frame = outcome_frame(inp.frames, members, t, end)
        if frame is None:
            rep.notes.append(f"outcomes of {t.date()}: prices do not reach {end.date()} yet")
            continue
        stamp = clock().isoformat(timespec="seconds")
        dv = _dumps({"prices_through": str(end.date()), "price_source": inp.info.get("price_source"),
                     "priced": int(frame.notna().any(axis=1).sum())})
        rows = [{"recorded_at": stamp, "as_of": str(t.date()), "end_date": str(end.date()), "ticker": tk,
                 "ret_exclude": _num(r["ret_exclude"]), "ret_pessimistic": _num(r["ret_pessimistic"]),
                 "data_version": dv, "code_version": code, "run_id": rep.run_id} for tk, r in frame.iterrows()]
        rep.outcomes[str(t.date())] = j.append_watch_outcomes(rows)
    return rep


# --------------------------------------------------------------------------- #
# Power: how long the forward test takes
# --------------------------------------------------------------------------- #
def months_needed(log: Optional[pd.DataFrame] = None, target_t: float = 2.0,
                  signals: Optional[Dict[str, WatchSignal]] = None) -> pd.DataFrame:
    """For each watch signal and each logged period (tuning, hold-out): the monthly IC's mean, standard deviation
    (mean / plain t x sqrt(months)) and Newey-West t, and the forward months a Newey-West t of `target_t` would take if
    the true monthly IC were that mean with the same autocorrelation: t grows with the square root of the months, so
    n = months x (target / logged NW t)^2. `months_80` is the length that gives an 80% chance of reaching it
    (n = months x ((target + 0.84) / NW t)^2). Read from reports/research/experiments.csv (the exclude rule; both rules
    give the same numbers), never written."""
    from scipy.stats import norm
    log = log if log is not None else H.ExperimentLog().rows()
    specs = signals if signals is not None else WATCH_SIGNALS
    rows = []
    for s in specs.values():
        r = log[(log["config_hash"] == s.config_hash) & (log["delisting_rule"] == "exclude")]
        for period in ("tuning", "holdout"):
            p = r[r["period"] == period]
            if p.empty:
                continue
            x = p.iloc[-1]                                           # the latest evaluation of that period
            m, t, nw, n = (float(x["ic_mean"]), float(x["ic_t"]), float(x["ic_nw_t"]), int(x["months"]))
            sd = m / t * math.sqrt(n) if t else float("nan")
            ok = m > 0 and nw > 0
            rows.append({"signal": s.name, "period": period, "months": n, "ic_mean": m, "ic_sd": sd, "ic_nw_t": nw,
                         "months_t2": n * (target_t / nw) ** 2 if ok else float("inf"),
                         "months_80": n * ((target_t + norm.ppf(0.8)) / nw) ** 2 if ok else float("inf")})
    return pd.DataFrame(rows)


__all__ = ["WATCH_SIGNALS", "WatchSignal", "WatchInputs", "SignalValues", "WatchReport", "record_watch", "is_due",
           "is_month_end_session", "month_end_session", "next_month_end_session", "next_open_utc", "on_time",
           "last_finished_session", "sessions", "snapshot_frame", "snapshot_rows", "outcome_frame", "snapshot_table",
           "pending_outcomes", "known_tenk_rows", "tenk_rows_for", "live_inputs", "load_current_frames",
           "months_needed", "fcf_yield_values", "short_volume_values", "risk_similarity_values"]
