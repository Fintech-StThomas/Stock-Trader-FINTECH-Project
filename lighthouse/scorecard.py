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
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Optional

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
