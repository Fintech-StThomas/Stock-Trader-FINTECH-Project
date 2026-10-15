"""
Score the forward journal: were the calls right, were the stated probabilities honest, and did following them beat
buy-and-hold and the S&P 500 over the same windows?

Two kinds of forecast are scored, each only once its outcome is known (nothing is scored early):

  verdicts  a Buy / Strong Buy says "beat SPY over the horizon" (21 sessions short-term, 252 long-term); a Reduce / Sell
            says "lag SPY". Hold makes no call. Hit rate, and the average return of following the verdict (long when
            Buy, in cash otherwise) next to buy-and-hold of the same stock and SPY over the same windows.
  tickets   each virtual position from an actionable ticket ends with a profit or a loss (journal events). Its forecast
            is `p_win`, the rules' historical win rate on that stock; calibration (forecast vs observed by bin) and the
            Brier score, against the Brier score of always forecasting the overall win rate (skill > 0 = useful).

Daily runs journal overlapping windows for the same stock: they are not independent. `per_month=True` keeps the first
decision per stock and month, and the summary reports how many windows were scored.

A third, separate part scores the watch-only research signals (lighthouse/research/watch.py, `score_watch`): monthly
rank IC with its Newey-West t, the top quintile against the equal-weight universe and the long-short spread before and
after the harness's costs, and turnover, computed by the research harness's own `evaluate` on the journal's forward
snapshots and outcomes. They are not survivors and carry no weight: nothing here feeds a verdict or a ticket.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BULLISH = {"Buy", "Strong Buy"}
BEARISH = {"Reduce", "Sell"}


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def brier_skill(p, y) -> float:
    """1 - Brier / Brier(climatology): above 0 the forecasts beat always saying the base rate."""
    y = np.asarray(y, float)
    if not len(y):
        return float("nan")
    ref = brier(np.full(len(y), y.mean()), y)
    return float(1.0 - brier(p, y) / ref) if ref > 0 else float("nan")


def calibration(p, y, bins: int = 10) -> pd.DataFrame:
    """Forecast vs observed frequency in equal-width probability bins (empty bins left out)."""
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "n", "forecast", "observed"])
    edges = np.linspace(0, 1, bins + 1)
    df["bin"] = pd.cut(df["p"], edges, include_lowest=True)
    g = df.groupby("bin", observed=True)
    out = pd.DataFrame({"n": g.size(), "forecast": g["p"].mean(), "observed": g["y"].mean()}).reset_index()
    out["bin"] = out["bin"].astype(str)
    return out


def score_verdicts(decisions: pd.DataFrame, closes: Callable[[str], Optional[pd.Series]], benchmark: str = "SPY",
                   per_month: bool = True) -> pd.DataFrame:
    """One row per decision whose horizon has passed: the stock's and SPY's return from the decision's close over
    `horizon_days` sessions, the call, and whether it was right."""
    d = decisions.copy()
    if d.empty:
        return pd.DataFrame()
    d = d[d["side"] == "long"] if "side" in d else d          # the verdict is the same for both sides: score it once
    d["as_of"] = pd.to_datetime(d["as_of"])
    if per_month:
        d = d.sort_values("id").groupby([d["ticker"], d["as_of"].dt.to_period("M")]).head(1)
    bench = closes(benchmark)
    rows = []
    for t, g in d.groupby("ticker"):
        px = closes(t)
        if px is None or bench is None or px.empty:
            continue
        for _, r in g.iterrows():
            h = int(r.get("horizon_days") or 252)
            i0 = px.index.searchsorted(r["as_of"], side="right") - 1
            if i0 < 0 or i0 + h >= len(px):
                continue                                         # the window has not finished (or no prices then)
            t0, t1 = px.index[i0], px.index[i0 + h]
            b = bench.reindex(bench.index.union([t0, t1])).ffill()
            ret = float(px.iloc[i0 + h] / px.iloc[i0] - 1.0)
            spy = float(b.loc[t1] / b.loc[t0] - 1.0)
            call = 1 if r["verdict"] in BULLISH else (-1 if r["verdict"] in BEARISH else 0)
            rows.append({"id": r["id"], "ticker": t, "as_of": t0, "end": t1, "verdict": r["verdict"], "call": call,
                         "ret": ret, "spy": spy, "excess": ret - spy,
                         "hit": (np.nan if call == 0 else float((ret - spy) * call > 0)),
                         "follow": ret if call == 1 else 0.0})
    return pd.DataFrame(rows)


def score_tickets(decisions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Closed virtual positions with their forecast `p_win` and outcome (profit = 1)."""
    if decisions.empty or events.empty:
        return pd.DataFrame()
    ex = events[events["kind"] == "exit"].copy()
    if ex.empty:
        return pd.DataFrame()
    ex["detail"] = ex["detail"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    ex["pnl"] = ex["detail"].map(lambda x: x.get("pnl"))
    ex["r"] = ex["detail"].map(lambda x: x.get("r"))
    ex["status"] = ex["detail"].map(lambda x: x.get("status"))
    m = ex.merge(decisions[["id", "ticker", "side", "as_of", "p_win", "hist_trades", "exit_style"]],
                 left_on="decision_id", right_on="id", suffixes=("", "_d"))
    m["won"] = (pd.to_numeric(m["pnl"], errors="coerce") > 0).astype(float)
    return m[["decision_id", "ticker", "side", "as_of", "date", "status", "pnl", "r", "won", "p_win", "hist_trades",
              "exit_style"]]


@dataclass
class Scorecard:
    verdicts: pd.DataFrame
    tickets: pd.DataFrame

    def summary(self) -> Dict[str, object]:
        v, t = self.verdicts, self.tickets
        out: Dict[str, object] = {"verdict_windows": int(len(v))}
        if len(v):
            called = v[v["call"] != 0]
            out.update({
                "calls": int(len(called)), "hit_rate": float(called["hit"].mean()) if len(called) else float("nan"),
                "follow_avg_return": float(v["follow"].mean()), "buy_hold_avg_return": float(v["ret"].mean()),
                "spy_avg_return": float(v["spy"].mean()),
                "buys": int((v["call"] == 1).sum()),
                "buy_excess_vs_spy": float(v.loc[v["call"] == 1, "excess"].mean()) if (v["call"] == 1).any() else float("nan"),
            })
        out["closed_tickets"] = int(len(t))
        if len(t):
            p = t["p_win"].astype(float)
            ok = p.notna()
            out.update({"ticket_win_rate": float(t["won"].mean()), "ticket_avg_r": float(pd.to_numeric(t["r"]).mean()),
                        "forecast_win_rate": float(p[ok].mean()) if ok.any() else float("nan"),
                        "brier": brier(p[ok], t.loc[ok, "won"]), "brier_skill": brier_skill(p[ok], t.loc[ok, "won"])})
        return out

    def calibration(self, bins: int = 10) -> pd.DataFrame:
        t = self.tickets
        if t.empty:
            return calibration([], [], bins)
        ok = t["p_win"].notna()
        return calibration(t.loc[ok, "p_win"], t.loc[ok, "won"], bins)


def build_scorecard(journal, closes: Callable[[str], Optional[pd.Series]], per_month: bool = True) -> Scorecard:
    d, e = journal.decisions(), journal.events()
    return Scorecard(score_verdicts(d, closes, per_month=per_month), score_tickets(d, e))


def price_closes(source: str = "auto", start="2000-01-01") -> Callable[[str], Optional[pd.Series]]:
    """A cached loader of total-return closes for the scorecard."""
    from .data import DataUnavailable, load_prices
    memo: Dict[str, Optional[pd.Series]] = {}

    def get(t: str) -> Optional[pd.Series]:
        if t not in memo:
            try:
                memo[t] = load_prices(t, start=start, years=None, source=source)["Close"]
            except DataUnavailable:
                memo[t] = None
        return memo[t]
    return get


# --------------------------------------------------------------------------- #
# Watch-only signals: forward evidence for the research near misses (lighthouse/research/watch.py)
# --------------------------------------------------------------------------- #
WATCH_LABEL = ("watch-only: research near misses, not survivors of the pre-registered rule; they carry no weight in any "
               "verdict, ticket or decision")
# The forward rule (docs/METHODOLOGY.md section 8c), written down on 15 October 2026, before any forward data existed.
# It is read once a year, on the forward months to December, and when met it only makes the case for a new
# pre-registered test: it never gives a signal weight.
FORWARD_MIN_MONTHS = 36
FORWARD_IC_T = 2.0
FORWARD_MAX_GAP_SHARE = 1.0 / 6.0
FLAT_BPS = 10.0              # the harness's cost sensitivity: one cost per side for every stock


def forward_panel(snaps: pd.DataFrame, outcomes: pd.DataFrame):
    """(harness Panel, signal, per-side costs) from one watch signal's snapshots and the recorded outcomes: the month
    ends that have both. None when there is no such month end. Members, the volume screen and the costs are the ones
    recorded with the snapshot; the next-month returns are the recorded outcomes under each delisting rule."""
    from .research import harness as H
    from .total_return import DELISTING_RULES
    if snaps is None or snaps.empty or outcomes is None or outcomes.empty:
        return None
    dates = sorted(set(snaps["as_of"]) & set(outcomes["as_of"]))
    if not dates:
        return None
    s = snaps[snaps["as_of"].isin(dates)]
    o = outcomes[outcomes["as_of"].isin(dates)]
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    tickers = sorted(set(s["ticker"]))

    def pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
        p = df.pivot_table(index="as_of", columns="ticker", values=col, aggfunc="last", dropna=False)
        p.index = pd.DatetimeIndex(pd.to_datetime(p.index))
        return p.reindex(index=idx, columns=tickers).astype(float)
    members = pd.DataFrame(False, index=idx, columns=tickers)
    for d, g in s.groupby("as_of"):
        members.loc[pd.Timestamp(d), sorted(set(g["ticker"]))] = True
    fwd = {rule: pivot(o, f"ret_{rule}").where(members) for rule in DELISTING_RULES}
    prices = fwd["exclude"].notna().astype(float).where(lambda x: x > 0)        # 1.0: priced at both month ends
    tradable = pivot(s, "tradable").fillna(0.0) > 0.5
    counts = s.groupby("as_of")["ticker"].nunique()
    counts.index = pd.DatetimeIndex(pd.to_datetime(counts.index))
    panel = H.Panel(dates=idx, prices=prices, members=members, fwd=fwd, last_price=pd.Series(dtype=object),
                    data_end=idx[-1], universe_name="S&P 500 (point in time), forward",
                    member_count=counts.reindex(idx), tradable=tradable)
    return panel, pivot(s, "value"), pivot(s, "cost")


def _watch_summary(res, alt) -> Dict[str, object]:
    """The forward statistics of one signal and delisting rule (res: harness cost model; alt: flat 10 bps)."""
    s = res.summary

    def ann(x: pd.Series) -> float:                    # the mean monthly return x 12, as the harness annualises
        return float(x.mean() * 12) if len(x) else float("nan")
    return {"months": int(len(res.ic)), "coverage": s["coverage"],
            "ic_mean": float(res.ic.mean()) if len(res.ic) else float("nan"), "ic_nw_t": s["ic_nw_t"],
            "ic_hit": s["ic_hit"], "long_only_gross_ann": ann(res.long_only), "long_only_net_ann": ann(res.long_only_net),
            f"long_only_net_ann_{FLAT_BPS:g}bps": ann(alt.long_only_net), "long_short_gross_ann": ann(res.spread),
            "long_short_net_ann": ann(res.spread_net), f"long_short_net_ann_{FLAT_BPS:g}bps": ann(alt.spread_net),
            "top_turnover": s["top_turnover"]}


def _forward_status(name: str, use: Optional[str], fp, snaps_tab: pd.DataFrame, min_names: int) -> Dict[str, object]:
    """The forward rule at the last annual look (the scored months to the latest December month end)."""
    from .research import harness as H
    from .research import watch as W
    from .total_return import DELISTING_RULES
    out: Dict[str, object] = {"signal": name, "look": None, "months_at_look": 0}
    looks = [d for d in (fp[0].dates if fp is not None else []) if d.month == 12]
    if not looks:
        n = 0 if fp is None else len(fp[0].dates)
        out["status"] = (f"collecting: {n} forward month(s) with an outcome; the rule is read once a year, on the months "
                         f"to December, from {FORWARD_MIN_MONTHS} months on")
        return out
    panel, value, costs = fp
    look = looks[-1]
    out["look"] = str(look.date())
    res = {r: H.evaluate(panel, value, name, r, (str(panel.dates[0].date()), str(look.date())), costs, None,
                         min_names=min_names) for r in DELISTING_RULES}
    months = min(len(v.ic) for v in res.values())
    out["months_at_look"] = months
    due = [d for d in W.sessions(pd.Timestamp(snaps_tab["as_of"].min()), look) if W.is_month_end_session(d)]
    have = set(pd.to_datetime(snaps_tab.loc[snaps_tab["on_time"].astype(bool), "as_of"]))
    gaps = sum(1 for d in due if d not in have)
    key = "long_only_net" if use == "long_only" else "spread_net"
    checks = {
        f"IC > 0 with Newey-West t >= {FORWARD_IC_T:g}": all(
            v.ic.mean() > 0 and v.summary["ic_nw_t"] >= FORWARD_IC_T for v in res.values()),
        "IC > 0 in more than half the months": all(v.summary["ic_hit"] > 0.5 for v in res.values()),
        f"{(use or 'long_short').replace('_', '-')} return after costs > 0": all(
            getattr(v, key).mean() > 0 for v in res.values()),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if months < FORWARD_MIN_MONTHS:
        out["status"] = f"collecting: {months} of {FORWARD_MIN_MONTHS} forward months at the look of {look.date()}"
    elif due and gaps / len(due) > FORWARD_MAX_GAP_SHARE:
        out["status"] = (f"incomplete record at the look of {look.date()}: {gaps} of {len(due)} month ends missing or "
                         "late")
    elif failed:
        out["status"] = f"not met at the look of {look.date()}: " + "; ".join(f"no {k}" for k in failed)
    else:
        out["status"] = (f"met at the look of {look.date()}: the case for a new pre-registered test (the user decides; "
                         "nothing is weighted)")
    return out


@dataclass
class WatchScorecard:
    table: pd.DataFrame                  # one row per (signal, configuration, delisting rule)
    monthly: pd.DataFrame                # one row per (signal, rule, month end scored)
    snapshots: pd.DataFrame              # one row per snapshot: when written, its deadline, on time or late
    status: pd.DataFrame                 # the forward rule per signal
    power: pd.DataFrame = field(default_factory=pd.DataFrame)   # months a Newey-West t of 2 would take (from the log)

    def late(self) -> pd.DataFrame:
        s = self.snapshots
        return s[~s["on_time"].astype(bool)] if len(s) else s

    def text(self) -> str:
        lines = [f"WATCH-ONLY SIGNALS ({WATCH_LABEL})"]
        if self.snapshots.empty:
            lines.append("no watch snapshots yet (python -m lighthouse journal watch records them at month ends)")
            return "\n".join(lines)
        ok = int(self.snapshots["on_time"].astype(bool).sum())
        lines.append(f"snapshots: {ok} on time, {len(self.snapshots) - ok} late (kept, not scored)")
        for _, r in self.late().iterrows():
            lines.append(f"  late: {r['signal']} {r['as_of']} written {r['recorded_at']}, after the open at {r['deadline']}")
        if len(self.table):
            cols = [c for c in ("signal", "rule", "months", "coverage", "ic_mean", "ic_nw_t", "ic_hit",
                                "long_only_gross_ann", "long_only_net_ann", f"long_only_net_ann_{FLAT_BPS:g}bps",
                                "long_short_net_ann", "top_turnover") if c in self.table]
            with pd.option_context("display.width", 250, "display.max_columns", 30):
                lines.append(self.table[cols].round(4).to_string(index=False))
        for _, r in self.status.iterrows():
            lines.append(f"forward rule, {r['signal']}: {r['status']}")
        if len(self.power):
            for name, g in self.power.groupby("signal", sort=False):
                parts = [f"{int(round(x))} at the {p} IC" if np.isfinite(x) else f"never at the {p} IC (not positive)"
                         for p, x in zip(g["period"].str.replace("holdout", "hold-out"), g["months_t2"])]
                lines.append(f"months for an expected Newey-West t of 2, {name}: " + ", ".join(parts))
        return "\n".join(lines)


def score_watch(journal, min_names: Optional[int] = None, signals=None, power: bool = True) -> WatchScorecard:
    """Score the watch-only signals on their on-time forward snapshots whose outcome is recorded (nothing earlier),
    with the research harness's own statistics, costs and delisting rules. Late snapshots are listed, not scored."""
    from .research import harness as H
    from .research import watch as W
    from .total_return import DELISTING_RULES
    snaps, outs = journal.watch_signals(), journal.watch_outcomes()
    tab = W.snapshot_table(snaps)
    specs = W.WATCH_SIGNALS if signals is None else signals
    min_names = H.MIN_NAMES if min_names is None else int(min_names)
    rows: List[Dict[str, object]] = []
    monthly: List[pd.DataFrame] = []
    status: List[Dict[str, object]] = []
    groups = snaps.groupby(["signal", "config_hash"], sort=False) if len(snaps) else []
    for (name, ch), g in groups:
        st = tab[(tab["signal"] == name) & (tab["config_hash"] == ch)]
        on = set(st.loc[st["on_time"].astype(bool), "as_of"])
        base = {"signal": name, "config_hash": ch, "snapshots": int(len(st)),
                "late": int((~st["on_time"].astype(bool)).sum()), "first": st["as_of"].min(), "last": st["as_of"].max()}
        fp = forward_panel(g[g["as_of"].isin(on)], outs)
        if fp is None:
            rows += [{**base, "rule": r, "months": 0} for r in DELISTING_RULES]
        else:
            panel, value, costs = fp
            flat = H.cost_panel(dict.fromkeys(panel.prices.columns), panel.dates, fixed_bps=FLAT_BPS)
            period = (str(panel.dates[0].date()), None)
            for rule in DELISTING_RULES:
                res = H.evaluate(panel, value, name, rule, period, costs, None, min_names=min_names)
                alt = H.evaluate(panel, value, name, rule, period, flat, None, min_names=min_names)
                rows.append({**base, "rule": rule, **_watch_summary(res, alt)})
                m = pd.DataFrame({"ic": res.ic, "long_only": res.long_only, "long_only_net": res.long_only_net,
                                  "spread": res.spread, "spread_net": res.spread_net, "turnover": res.turnover})
                if len(m):
                    monthly.append(m.rename_axis("as_of").reset_index().assign(signal=name, rule=rule))
        spec = specs.get(name)
        status.append(_forward_status(name, spec.use if spec is not None and spec.config_hash == ch else None, fp, st,
                                      min_names))
    pw = pd.DataFrame()
    if power:
        try:
            pw = W.months_needed(signals=specs)
        except Exception:  # noqa: BLE001 - the experiment log is optional for scoring
            pw = pd.DataFrame()
    return WatchScorecard(pd.DataFrame(rows), pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame(),
                          tab, pd.DataFrame(status), pw)
