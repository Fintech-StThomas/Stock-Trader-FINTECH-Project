"""
The senior-analyst layer: theses written down, checked against every new filing, sized by conviction and reviewed
when they end.

Thesis tracker   `open_thesis` records a memo's stance, price, values and kill criteria in the journal's SQLite file
                 (append-only tables `theses` and `thesis_checks`). `check_theses` reads each open thesis's filings:
                 when a 10-Q or 10-K has been filed since the last check it recomputes the numbers as of that filing
                 and tests every criterion; the price criterion is tested on every run. A breach marks the thesis
                 broken (it stays open until closed: the analyst decides).
Coverage monitor `coverage` lists every thesis: stance, return since it was opened against SPY, distance to the
                 value, criteria breached, filings since, and whether the memo is stale (a new filing since, or more
                 than 100 days old); with the journal's own calibration for the stock when its verdicts have been scored.
Conviction       `conviction` turns a memo into 0-1 (upside, upside-to-downside, flags, agreement with the rule-based
                 signal); `conviction_size` scales a position between half and the full size the risk rules allow.
                 It never raises a size above the risk rules, and nothing uses it unless asked.
Post-mortem      `close_thesis` writes what was expected (value, implied growth, base-case growth and margin) against
                 what happened (the return and SPY's, revenue growth and margin since, which criterion broke), and
                 splits the price change into the change in earnings and the change in the multiple.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from ..journal import Journal, _now, _sql
from .financials import financials_as_of

STALE_DAYS = 100

_SCHEMA = """
CREATE TABLE IF NOT EXISTS theses (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL, ticker TEXT NOT NULL, as_of TEXT NOT NULL,
  stance TEXT, price REAL, target REAL, bear REAL, base REAL, bull REAL, implied_growth REAL, base_growth REAL,
  base_margin REAL, revenue REAL, operating_margin REAL, net_income REAL, pe REAL, last_filed TEXT, conviction REAL,
  criteria TEXT, thesis TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS thesis_checks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT NOT NULL, thesis_id INTEGER NOT NULL, ticker TEXT NOT NULL,
  date TEXT, kind TEXT NOT NULL, last_filed TEXT, price REAL, metrics TEXT, breaches TEXT, detail TEXT);
CREATE TRIGGER IF NOT EXISTS theses_no_update BEFORE UPDATE ON theses
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS theses_no_delete BEFORE DELETE ON theses
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS thesis_checks_no_update BEFORE UPDATE ON thesis_checks
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS thesis_checks_no_delete BEFORE DELETE ON thesis_checks
  BEGIN SELECT RAISE(ABORT, 'the journal is append-only'); END;
"""


class ThesisBook:
    """The theses and their checks, in the journal's SQLite file."""

    def __init__(self, journal: Optional[Journal] = None, path=None) -> None:
        self.journal = journal or Journal(path)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator:
        with self.journal._conn() as c:      # noqa: SLF001 - same file, same connection settings
            yield c

    def theses(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM theses ORDER BY id", c)

    def checks(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM thesis_checks ORDER BY id", c)

    def append_check(self, thesis_id: int, ticker: str, kind: str, date=None, last_filed=None, price=None,
                     metrics: Optional[dict] = None, breaches: Optional[list] = None, detail: Optional[dict] = None) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO thesis_checks (recorded_at, thesis_id, ticker, date, kind, last_filed, price, "
                            "metrics, breaches, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (_now(), int(thesis_id), ticker, None if date is None else str(pd.Timestamp(date).date()), kind,
                             None if last_filed is None else str(pd.Timestamp(last_filed).date()), _sql(price),
                             json.dumps(metrics or {}, default=str), json.dumps(breaches or [], default=str),
                             json.dumps(detail or {}, default=str)))
            return int(cur.lastrowid)

    def open_ids(self) -> List[int]:
        t, c = self.theses(), self.checks()
        closed = set(c.loc[c["kind"] == "closed", "thesis_id"].astype(int)) if not c.empty else set()
        return [int(i) for i in t["id"] if int(i) not in closed]


# --------------------------------------------------------------------------- #
# Conviction
# --------------------------------------------------------------------------- #
def conviction(memo) -> float:
    """0-1: how strongly the evidence in the memo points one way (for a buy: upside, asymmetry, clean checks, the
    rule-based signal agreeing). Avoid and hold stances get the same scale for the short or no-trade side."""
    v = memo.valuation
    if v is None or not memo.financials.price:
        return 0.0
    up = v.target / memo.financials.price - 1.0
    ud = v.upside_downside
    flags = memo.quality + memo.balance
    reds = sum(x.severity == "red" for x in flags)
    ambers = sum(x.severity == "amber" for x in flags)
    s_up = float(np.clip(abs(up) / 0.40, 0, 1))                        # 40% from fair value = full marks
    s_ud = float(np.clip((ud - 1.0) / 2.0, 0, 1)) if np.isfinite(ud) and up > 0 else (1.0 if up > 0 else 0.5)
    s_flags = float(np.clip(1.0 - 0.35 * reds - 0.1 * ambers, 0, 1))
    agree = 0.5
    if memo.rule_score is not None:
        agree = float(np.clip((memo.rule_score - 45.0) / 27.0, 0, 1)) if up > 0 else \
            float(np.clip((45.0 - memo.rule_score) / 27.0, 0, 1))
    return float(np.clip(0.35 * s_up + 0.2 * s_ud + 0.25 * s_flags + 0.2 * agree, 0, 1))


def conviction_size(max_shares: int, conviction_value: float, floor: float = 0.5) -> int:
    """Between `floor` and all of the shares the risk rules allow; never more."""
    c = float(np.clip(conviction_value, 0, 1))
    return int(np.floor(max_shares * (floor + (1.0 - floor) * c)))


# --------------------------------------------------------------------------- #
# Opening, checking, closing
# --------------------------------------------------------------------------- #
def open_thesis(memo, book: Optional[ThesisBook] = None, note: str = "") -> int:
    book = book or ThesisBook()
    f, v, r = memo.financials, memo.valuation, memo.financials.ratios()
    row = {"recorded_at": _now(), "ticker": memo.ticker, "as_of": str(f.as_of.date()), "stance": memo.stance,
           "price": f.price, "target": v.target if v else None,
           "bear": v.cases["bear"].per_share if v else None, "base": v.cases["base"].per_share if v else None,
           "bull": v.cases["bull"].per_share if v else None, "implied_growth": v.implied_growth if v else None,
           "base_growth": v.cases["base"].assumptions.growth[0] if v else None,
           "base_margin": v.cases["base"].assumptions.operating_margin[0] if v else None,
           "revenue": f.get("revenue"), "operating_margin": r["operating_margin"], "net_income": f.get("net_income"),
           "pe": r["pe"], "last_filed": str(f.last_filed.date()) if f.last_filed is not None else None,
           "conviction": conviction(memo), "criteria": json.dumps([c.to_dict() for c in memo.criteria]),
           "thesis": memo.thesis, "note": note}
    cols = list(row)
    with book._conn() as c:                  # noqa: SLF001
        cur = c.execute(f"INSERT INTO theses ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [_sql(row[k]) for k in cols])
        return int(cur.lastrowid)


@dataclass
class ThesisStatus:
    thesis_id: int
    ticker: str
    stance: str
    opened: str
    price_then: float
    price_now: float
    new_filing: Optional[str]
    breaches: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def broken(self) -> bool:
        return bool(self.breaches)


def _criteria(row) -> List:
    from .memo import Criterion
    return [Criterion(**c) for c in json.loads(row["criteria"] or "[]")]


def _latest(ticker: str, source: str, date=None):
    from ..data import load_prices
    from ..pit import load_statements
    from ..total_return import load_actions
    px = load_prices(ticker, years=2, end=date, source=source)
    st = load_statements(ticker, source=source)
    return financials_as_of(ticker, st, px, load_actions(ticker, source=source), date), px


def check_theses(book: Optional[ThesisBook] = None, source: str = "auto", date=None) -> List[ThesisStatus]:
    """Check every open thesis: the price criterion always, the others when a filing has arrived since the last check."""
    book = book or ThesisBook()
    t, c = book.theses(), book.checks()
    out: List[ThesisStatus] = []
    for tid in book.open_ids():
        row = t.loc[t["id"] == tid].iloc[0]
        try:
            f, _ = _latest(row["ticker"], source, date)
        except Exception as exc:  # noqa: BLE001 - one stock's data must not stop the others
            book.append_check(tid, row["ticker"], "error", date=date, detail={"error": f"{type(exc).__name__}: {exc}"})
            continue
        mine = c[(c["thesis_id"] == tid) & c["last_filed"].notna()] if not c.empty else c
        seen = max([pd.Timestamp(row["last_filed"])] + [pd.Timestamp(x) for x in mine["last_filed"]]) \
            if row["last_filed"] else None
        new = f.last_filed is not None and (seen is None or f.last_filed > seen)
        ratios = f.ratios()
        metrics = {"price": f.price, **{k: ratios[k] for k in ("operating_margin", "revenue_growth", "net_debt_ebitda",
                                                                  "interest_cover", "pe")}}
        breaches = []
        for crit in _criteria(row):
            if crit.metric != "price" and not new:
                continue
            val = metrics.get(crit.metric, ratios.get(crit.metric, np.nan))
            if crit.breached(val):
                breaches.append(f"{crit.text} (now {val:,.3g})")
        kind = "filing" if new else "price"
        book.append_check(tid, row["ticker"], kind, date=f.as_of, last_filed=f.last_filed if new else None,
                          price=f.price, metrics=metrics, breaches=breaches)
        out.append(ThesisStatus(tid, row["ticker"], row["stance"], row["as_of"], float(row["price"]), f.price,
                                str(f.last_filed.date()) if new else None, breaches, metrics))
    return out


def _ttm_end(quarterly: Optional[pd.DataFrame], by) -> Optional[pd.Timestamp]:
    """The period end of the latest quarter filed by `by`: where a trailing-twelve-month sum known then ends."""
    if quarterly is None or quarterly.empty or "filed" not in quarterly:
        return None
    known = quarterly[pd.to_datetime(quarterly["filed"]) <= pd.Timestamp(by)]
    return pd.Timestamp(known.index.max()) if len(known) else None


def post_mortem(row, f_now, spy_return: Optional[float] = None, broke: Optional[List[str]] = None) -> List[str]:
    """Expected against what happened, from the thesis row and the numbers at close."""
    p0, p1 = float(row["price"]), f_now.price
    ret = p1 / p0 - 1.0
    r_now = f_now.ratios()
    lines = [f"Opened {row['as_of']} at ${p0:,.2f} with stance {str(row['stance']).upper()} and a value of "
             f"${float(row['target'] or np.nan):,.2f}; closed {f_now.as_of.date()} at ${p1:,.2f} ({ret:+.1%})"
             + (f" against SPY {spy_return:+.1%}." if spy_return is not None and np.isfinite(spy_return) else ".")]
    rev0, rev1 = row["revenue"], f_now.get("revenue")
    end0, end1 = _ttm_end(f_now.quarterly, row["as_of"]), _ttm_end(f_now.quarterly, f_now.as_of)
    if rev0 and np.isfinite(rev1):
        # trailing revenue moves when a quarter is reported, so the growth rate is per year of reported quarters
        # (the quarter ends behind the two trailing sums), not per calendar year between the memo dates
        if end0 is not None and end1 is not None:
            years = (end1 - end0).days / 365.25
            span = f" (trailing twelve months to {end0.date()} vs to {end1.date()})"
        else:
            years, span = (f_now.as_of - pd.Timestamp(row["as_of"])).days / 365.25, ""
        expected = (f"against {float(row['base_growth'] or np.nan):.1%} in the base case and "
                    f"{float(row['implied_growth']) if row['implied_growth'] is not None else float('nan'):.1%} "
                    "implied by the price")
        if years < 0.2:
            lines.append(f"Revenue: no new quarter reported since the thesis was opened ({expected}).")
        else:
            g = (rev1 / rev0) ** (1 / years) - 1.0
            lines.append(f"Revenue grew {g:+.1%} a year{span}, {expected}.")
    if row["operating_margin"] is not None and np.isfinite(r_now["operating_margin"]):
        lines.append(f"Operating margin went from {float(row['operating_margin']):.1%} to {r_now['operating_margin']:.1%}.")
    e0, e1 = row["net_income"], f_now.get("net_income")
    if e0 and e0 > 0 and np.isfinite(e1) and e1 > 0 and row["pe"] and np.isfinite(r_now["pe"]):
        earn = e1 / e0 - 1.0
        mult = r_now["pe"] / float(row["pe"]) - 1.0
        lines.append(f"Of the price change, earnings (net income) moved {earn:+.1%} and the P/E {mult:+.1%} "
                     f"({float(row['pe']):.1f}x to {r_now['pe']:.1f}x); share-count changes and dividends make up the rest.")
    if broke:
        lines.append("Criteria that broke: " + "; ".join(broke))
    right = (ret > 0) if row["stance"] == "buy" else (ret < 0) if row["stance"] == "avoid" else None
    if right is not None:
        lines.append("The stance was " + ("right" if right else "wrong") + " on the price over this period; whether for the "
                     "reasons in the thesis is the question the lines above answer.")
    return lines


def close_thesis(thesis_id: int, reason: str = "", book: Optional[ThesisBook] = None, source: str = "auto",
                 date=None) -> List[str]:
    book = book or ThesisBook()
    t = book.theses()
    row = t.loc[t["id"] == int(thesis_id)].iloc[0]
    f, _ = _latest(row["ticker"], source, date)
    spy = None
    try:
        from ..data import load_prices
        s = load_prices("SPY", years=None, start=row["as_of"], end=f.as_of, source=source)["Close"]
        spy = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 1 else None
    except Exception:  # noqa: BLE001
        pass
    c = book.checks()
    broke = []
    if not c.empty:
        for b in c.loc[c["thesis_id"] == int(thesis_id), "breaches"]:
            broke += [x for x in json.loads(b or "[]") if x not in broke]
    lines = post_mortem(row, f, spy, broke)
    book.append_check(int(thesis_id), row["ticker"], "closed", date=f.as_of, price=f.price,
                      detail={"reason": reason, "post_mortem": lines})
    return lines


def coverage(book: Optional[ThesisBook] = None, source: str = "auto", date=None, prices_now: bool = True) -> pd.DataFrame:
    """One row per thesis: where it stands now."""
    book = book or ThesisBook()
    t, c = book.theses(), book.checks()
    if t.empty:
        return pd.DataFrame()
    calib = _journal_calibration(book.journal, source) if prices_now else {}
    rows = []
    today = pd.Timestamp(date) if date is not None else pd.Timestamp.today().normalize()
    for _, row in t.iterrows():
        mine = c[c["thesis_id"] == row["id"]] if not c.empty else c
        closed = bool((mine["kind"] == "closed").any()) if len(mine) else False
        breaches = sorted({x for b in mine["breaches"] for x in json.loads(b or "[]")}) if len(mine) else []
        filings = int((mine["kind"] == "filing").sum()) if len(mine) else 0
        price_now = float(mine["price"].dropna().iloc[-1]) if len(mine) and mine["price"].notna().any() else np.nan
        age = (today - pd.Timestamp(row["as_of"])).days
        rows.append({"id": int(row["id"]), "ticker": row["ticker"], "opened": row["as_of"], "stance": row["stance"],
                     "conviction": row["conviction"], "price_then": row["price"], "price_now": price_now,
                     "return": price_now / row["price"] - 1.0 if np.isfinite(price_now) else np.nan,
                     "value": row["target"], "to_value": row["target"] / price_now - 1.0
                     if row["target"] and np.isfinite(price_now) else np.nan,
                     "filings_since": filings, "breaches": len(breaches), "status": "closed" if closed else
                     "broken" if breaches else "open",
                     "stale": bool(filings > 0 or age > STALE_DAYS) and not closed, "age_days": age,
                     "journal_hit_rate": calib.get(row["ticker"], {}).get("hit_rate", np.nan),
                     "journal_brier_skill": calib.get(row["ticker"], {}).get("brier_skill", np.nan)})
    return pd.DataFrame(rows).set_index("id")


def _journal_calibration(journal: Journal, source: str) -> Dict[str, Dict[str, float]]:
    """Per ticker: the hit rate of its scored verdicts and the Brier skill of its tickets (when any are scored)."""
    try:
        from ..scorecard import brier_skill, build_scorecard, price_closes
        sc = build_scorecard(journal, price_closes(source))
    except Exception:  # noqa: BLE001 - nothing scored yet, or no prices
        return {}
    out: Dict[str, Dict[str, float]] = {}
    v, k = sc.verdicts, sc.tickets
    if len(v):
        for t, g in v[v["call"] != 0].groupby("ticker"):
            out.setdefault(t, {})["hit_rate"] = float(g["hit"].mean())
    if len(k) and {"p_win", "won"} <= set(k.columns):
        for t, g in k.dropna(subset=["p_win", "won"]).groupby("ticker"):
            if len(g) >= 3:
                out.setdefault(t, {})["brier_skill"] = float(brier_skill(g["p_win"], g["won"]))
    return out


__all__ = ["ThesisBook", "ThesisStatus", "conviction", "conviction_size", "open_thesis", "check_theses", "close_thesis",
           "post_mortem", "coverage", "STALE_DAYS"]
