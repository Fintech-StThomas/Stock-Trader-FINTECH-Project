"""Plain-English report generation (Markdown)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

LIGHT_ICON = {"green": "🟢", "amber": "🟡", "red": "🔴", "grey": "⚪"}


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}", index: bool = True) -> str:
    """Tiny Markdown table writer (avoids a `tabulate` dependency)."""
    cols = list(df.columns)
    header = ([df.index.name or ""] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] if index else []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Human-friendly version of a metrics table (percentages, 2-dp ratios)."""
    out = pd.DataFrame(index=df.index)
    pct = {"total_return": "Total return", "cagr": "CAGR", "volatility": "Volatility", "max_drawdown": "Max drawdown",
           "win_rate": "Win rate", "avg_win": "Avg win", "avg_loss": "Avg loss", "exposure": "Time invested",
           "strategy_ann_return": "Strategy ann. return", "buy_hold_ann_return": "Buy & hold ann. return",
           "share_of_time": "Share of time", "avg_exposure": "Avg exposure", "turnover_annual": "Turnover / yr"}
    for c in df.columns:
        if c in pct:
            out[pct[c]] = df[c].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:.1f}%")
        elif c in {"sharpe", "sortino", "calmar", "profit_factor", "is_sharpe"}:
            out[c.replace("_", " ").title()] = df[c].map(lambda v: "n/a" if pd.isna(v) else ("inf" if np.isinf(v) else f"{v:.2f}"))
        elif c in {"n_trades", "days", "n_trades"}:
            out["Trades" if c == "n_trades" else c.title()] = df[c].map(lambda v: "n/a" if pd.isna(v) else f"{int(v)}")
        elif c == "years":
            out["Years"] = df[c].map(lambda v: f"{v:.1f}")
        else:
            out[c] = df[c]
    return out


def beacon_table(beacons) -> str:
    rows = []
    for b in beacons:
        rows.append({"signal": b.name, "reading": b.value, "light": LIGHT_ICON[b.light],
                     "vote": "n/a" if np.isnan(b.score) else f"{b.score:+.2f}",
                     "weight": f"{b.weight * 100:.0f}%",
                     "points": "n/a" if np.isnan(b.score) else f"{b.contribution:+.1f}", "why": b.rationale})
    df = pd.DataFrame(rows)
    return _md_table(df, index=False)


def beacon_frame_df(beacons) -> pd.DataFrame:
    """Beacons as a DataFrame (for notebooks / apps)."""
    return pd.DataFrame([{"light": LIGHT_ICON[b.light], "signal": b.name, "reading": b.value,
                          "vote": None if np.isnan(b.score) else round(b.score, 2), "weight": f"{b.weight * 100:.0f}%",
                          "points": None if np.isnan(b.score) else round(b.contribution, 1), "why": b.rationale} for b in beacons])


def build_report(result, include_backtest: bool = True) -> str:
    d, f, prof = result.decision, result.fundamentals, result.profile
    L: List[str] = []
    L.append(f"# Lighthouse report: {f.name or d.ticker} ({d.ticker})")
    q = getattr(result, "quote", None)
    priced = f"price {q.describe()}" if q is not None else f"price ${d.price:,.2f} (close)"
    L.append(f"*Signals as of {d.as_of} · {priced} · investor: {prof.describe()}*\n")
    L.append(f"## Verdict: **{d.verdict}** — Lighthouse score {d.composite_score:.0f} / 100")
    L.append(d.plan.action + "\n")

    # score composition
    L.append("### How the score is built")
    if d.fundamental_score is not None:
        L.append(f"- Fundamental score **{d.fundamental_score:.0f}** × weight {prof.fundamental_weight:.0%} = {d.fundamental_score * prof.fundamental_weight:.1f}")
        L.append(f"- Technical score **{d.technical_score:.0f}** × weight {prof.technical_weight:.0%} = {d.technical_score * prof.technical_weight:.1f}")
    else:
        L.append(f"- Technical score **{d.technical_score:.0f}** (fundamentals unavailable, weight 100%)")
    if d.regime is not None:
        L.append(f"- Market-weather adjustment: **{('−' + format(d.regime_penalty, '.0f')) if d.regime_penalty else '0'}** ({d.regime.label})")
    L.append(f"- **Composite = {d.composite_score:.1f}** → {d.verdict} "
             f"(bands: ≥72 Strong Buy, ≥60 Buy, ≥45 Hold, ≥32 Reduce, else Sell)\n")

    # fundamentals
    L.append(f"### Fundamental beacons ({'score ' + format(d.fundamental_score, '.0f') if d.fundamental_score is not None else 'unavailable'})")
    meta = [x for x in [f.sector, f.industry] if x]
    if meta:
        L.append(f"*{' · '.join(meta)} · data: {f.source} ({f.as_of})*\n")
    if d.fundamental_beacons:
        L.append(beacon_table(d.fundamental_beacons) + "\n")
    else:
        L.append("_No fundamental data was available for this ticker._\n")

    # technicals
    L.append(f"### Technical beacons (score {d.technical_score:.0f})")
    L.append(beacon_table(d.technical_beacons) + "\n")

    # regime
    L.append("### Market weather")
    if d.regime is not None:
        L.append(d.regime.describe())
        if d.regime.is_bear:
            L.append(f"In a bear market Lighthouse removes {result.params.bear_score_penalty:.0f} points, raises the entry bar "
                     f"and halves new position sizes.")
        if d.regime.is_turbulent:
            L.append(f"Elevated volatility removes {result.params.turbulent_score_penalty:.0f} more points; the volatility rule "
                     f"automatically shrinks the position.")
        L.append("")
    else:
        L.append("_Benchmark unavailable; no regime adjustment applied._\n")

    # position plan
    pl = d.plan
    L.append("### Position plan")
    if pl.position_pct > 0:
        L.append(f"- Buy **{pl.shares} shares** near ${pl.entry_price:,.2f} = **${pl.cost:,.0f}** of your ${prof.cash:,.0f} cash "
                 f"({pl.position_pct:.1f}% of the ${prof.account_value:,.0f} account); ${pl.cash_left:,.0f} stays in cash.")
        if pl.stop_price:
            L.append(f"- Exit level **${pl.stop_price:,.2f}** ({pl.stop_pct * 100:.1f}% below): worst case ≈ ${pl.max_loss_value:,.0f} "
                     f"({pl.max_loss_value / prof.account_value * 100:.2f}% of the account, limit {prof.max_loss_pct:g}%) if the stop "
                     "fills at its price; an overnight gap through the stop can lose more.")
        L.append(f"- Size was limited by the **{pl.binding_constraint}**.")
    else:
        L.append(f"- No new capital. {pl.action}")
        if pl.stop_price and d.verdict in {"Hold"}:
            L.append(f"- If you already own it, a trailing stop near ${pl.stop_price:,.2f} ({pl.stop_pct * 100:.1f}% below) keeps the downside defined.")
    for r in pl.rationale:
        L.append(f"  - {r}")
    L.append("")

    # backtest
    bt = result.backtest
    if include_backtest and bt is not None:
        L.append(f"### How this rule behaved in the past ({bt.start.date()} → {bt.end.date()})")
        pit = getattr(result, "pit_statements", None)
        dated = ("statements dated by their SEC filing date" if pit is not None and pit.has_filing_dates else
                 f"statements lagged {result.params.reporting_lag_days} days")
        L.append(f"Long-only, reviewed {result.params.decision_frequency} at the close, filled at the {bt.options.get('fills')}; "
                 f"costs: {bt.options.get('costs', 'fixed bps per side')}; stops: {bt.options.get('stops', 'n/a')}; cash "
                 f"{bt.options.get('cash', 'earns 0%')}; {dated}. Returns are for the capital allocated to this stock. "
                 f"Sharpe and Sortino use {bt.options.get('risk_free', 'a fixed risk-free rate')}.\n")
        L.append(_md_table(format_metrics(bt.summary())) + "\n")
        layers = result.extras.get("layers")
        if layers is not None:
            L.append("**What each layer contributes**\n")
            L.append(_md_table(format_metrics(layers)) + "\n")
        rb = result.extras.get("regime_breakdown")
        if rb is not None and len(rb):
            L.append("**Behaviour by market regime** (annualised daily returns)\n")
            L.append(_md_table(format_metrics(rb)) + "\n")
        wf = result.extras.get("walk_forward")
        if wf is not None:
            L.append(f"**Out-of-sample check** (thresholds tuned before {wf['split_date'].date()}, tested after it)\n")
            L.append(_md_table(format_metrics(wf["out_of_sample"])) + "\n")
        sens = result.extras.get("sensitivity")
        if sens is not None and len(sens):
            best, worst = sens.loc[sens["sharpe"].idxmax()], sens.loc[sens["sharpe"].idxmin()]
            L.append(f"**Sensitivity**: across {len(sens)} entry/exit threshold pairs the Sharpe ratio ranges from "
                     f"{worst['sharpe']:.2f} (entry {worst['entry']:.0f}/exit {worst['exit']:.0f}) to {best['sharpe']:.2f} "
                     f"(entry {best['entry']:.0f}/exit {best['exit']:.0f}); median {sens['sharpe'].median():.2f}.\n")

    # warnings & limitations
    L.append("### Data notes and warnings")
    if result.warnings:
        L += [f"- {w}" for w in result.warnings]
    else:
        L.append("- none")
    src = result.prices.attrs.get("source", "?")
    L.append(f"- Prices: {src}; {len(result.prices):,} trading days {result.prices.index[0].date()} → {result.prices.index[-1].date()}.")
    L.append("")
    L.append("### Limitations you should know")
    L += [
        "- Sector norms are long-run approximations, not live peer data; a cheap stock in an expensive sector can still be expensive.",
        "- The fundamental beacons use the latest reported figures; the backtest's fundamental gate uses SEC filings as "
        "first reported, from the day after each was filed (when EDGAR is reachable).",
        "- The backtest is long-only; costs are estimated from the stock's own spread and volume, stops rest as orders (an "
        "overnight gap fills at the open), and it ignores taxes. Run `python -m lighthouse memo` for the analyst view.",
        "- Past behaviour of a rule is evidence about its *design*, not a promise about its future returns.",
        "- Lighthouse is a decision-support tool, not investment advice.",
    ]
    return "\n".join(L)



# --------------------------------------------------------------------------- #
# The investment memo (lighthouse.analyst.memo)
# --------------------------------------------------------------------------- #
def _p(x, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _x(x, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}x"


def build_memo(memo) -> str:
    """The memo as Markdown: the recommendation first, then the evidence for it."""
    m, f = memo, memo.financials
    r = f.ratios()
    v = m.valuation
    name = m.profile.get("name") or m.ticker
    L: List[str] = [f"# Investment memo: {name} ({m.ticker})",
                    f"*As of {f.as_of.date()} · price ${f.price:,.2f} · market value ${f.market_cap / 1e9:,.1f}bn · "
                    f"SIC {m.profile.get('sic') or 'n/a'} {m.profile.get('sicDescription') or ''}*\n"]
    from .analyst.memo import STANCE_RULE
    L.append(f"## Recommendation: **{m.stance.upper()}**")
    L.append(m.thesis + "\n")
    L.append(f"*{STANCE_RULE}*\n")
    if m.supporting:
        L.append("**Why**")
        L += [f"- {x}" for x in m.supporting]
        L.append("")
    if m.risks:
        L.append("**Risks**")
        L += [f"- {x}" for x in m.risks]
        L.append("")
    if m.variant_view:
        L.append("## Variant view: what the price assumes")
        L += [f"- {x}" for x in m.variant_view]
        L.append("")

    L.append("## The numbers (trailing twelve months)")
    rows = [("Revenue", f"${f.get('revenue') / 1e9:,.2f}bn" if np.isfinite(f.get('revenue')) else "n/a", _p(r["revenue_growth"])),
            ("Operating income", f"${f.get('operating_income') / 1e9:,.2f}bn" if np.isfinite(f.get('operating_income')) else "n/a",
             _p(r["operating_margin"]) + " margin"),
            ("Net income", f"${f.get('net_income') / 1e9:,.2f}bn" if np.isfinite(f.get('net_income')) else "n/a",
             _p(r["eps_growth"]) + " growth"),
            ("Free cash flow", f"${f.fcf / 1e9:,.2f}bn" if np.isfinite(f.fcf) else "n/a", _p(r["fcf_yield"]) + " yield"),
            ("ROIC", _p(r["roic"]), ""), ("Gross margin", _p(r["gross_margin"]), ""),
            ("Net debt", f"${f.net_debt / 1e9:,.2f}bn", _x(r["net_debt_ebitda"]) + " EBITDA"),
            ("Stock-based pay", _p(r["sbc_revenue"]) + " of revenue", "")]
    L.append(_md_table(pd.DataFrame(rows, columns=["line", "value", "note"]).set_index("line")) + "\n")
    ind = getattr(m, "industry", None)
    if ind is not None:
        L += _industry_lines(ind, m.profile.get("sicDescription"))
    from .analyst.segments import memo_lines                     # optional: only when the 10-K tags segments
    L += memo_lines(getattr(m, "segments", None))

    L.append("## Valuation")
    if v is not None:
        t = v.table()
        show = pd.DataFrame({"weight": t["weight"].map(lambda x: f"{x:.0%}"),
                             "value / share": t["value_per_share"].map(lambda x: f"${x:,.2f}"),
                             "vs price": t["vs_price"].map(lambda x: f"{x:+.0%}"),
                             "growth yr 1 → 5": [f"{_p(a)} → {_p(b)}" for a, b in zip(t["growth_y1"], t["growth_last"])],
                             "margin yr 5": t["margin_last"].map(_p),
                             "terminal share": t["terminal_share"].map(lambda x: f"{x:.0%}")}, index=t.index)
        L.append(_md_table(show) + "\n")
        ud = v.upside_downside
        ud_text = ("the bull case is below the price" if v.cases["bull"].per_share < f.price else
                   "the bear case is above the price" if not np.isfinite(ud) or ud == float("inf") else
                   f"upside-to-downside {ud:.2f} (bull gain ÷ bear loss)")
        L.append(f"- Probability-weighted value **${v.target:,.2f}** ({v.target / f.price - 1:+.0%}); {ud_text}.")
        L.append(f"- Reverse DCF: the price implies **{_p(v.implied_growth)}** revenue growth a year for five years at today's "
                 f"margin, or a **{_p(v.implied_margin)}** operating margin at the base case's growth.")
        L.append(f"- Discount rate (WACC) {_p(v.rate.wacc, 2)}: risk-free {_p(v.rate.risk_free, 2)}, beta {v.rate.beta:.2f}, "
                 f"equity premium {_p(v.rate.equity_premium)}, cost of debt {_p(v.rate.cost_of_debt, 2)}; terminal growth "
                 f"{_p(v.terminal_growth)}, reinvesting growth / RONIC of operating profit after tax (RONIC "
                 f"{_p(v.cases['base'].ronic)}: today's ROIC, at least the WACC). Free cash flow is after stock-based pay.")
        if not v.bands.empty:
            b = v.bands.copy()
            labels = {"earnings_yield": "earnings yield", "fcf_yield": "FCF yield", "ebitda_ev": "EBITDA / EV"}
            show = pd.DataFrame({"now": b["now"].map(_p), "10th pct": b["p10"].map(_p), "median": b["median"].map(_p),
                                 "90th pct": b["p90"].map(_p), "cheaper than": b["percentile"].map(lambda x: f"{x:.0%} of days"),
                                 "since": b["since"]})
            show.index = [labels.get(i, i) for i in b.index]
            L.append("\n**Against its own history** (higher yield = cheaper)\n")
            L.append(_md_table(show) + "\n")
    else:
        L.append("_No DCF: the filings lack the lines it needs._\n")
    if m.peers is not None:
        pc = m.peers
        L.append(f"**Peers** ({pc.level}, closest in market value)\n")
        tab = pc.table.copy()
        fmt = {"pe": _x, "ev_ebitda": _x, "ev_sales": _x, "pb": _x, "net_debt_ebitda": _x}
        show = pd.DataFrame({LABELS_PEERS.get(c, c): tab[c].map(fmt.get(c, _p)) for c in tab.columns}, index=tab.index)
        L.append(_md_table(show) + "\n")
        L += [f"- {x}" for x in pc.explanation]
        L.append("")
    fv = getattr(m, "fundamental_view", None)
    if fv is not None:
        L += _fundamental_view_lines(fv)

    L.append("## Quality and balance sheet")
    for x in m.quality + m.balance:
        L.append(f"- {LIGHT_ICON.get(x.severity, '')} *{x.area}*: {x.text}")
    L.append("")
    if m.capital.get("lines"):
        L.append("## Capital allocation")
        L += [f"- {x}" for x in m.capital["lines"]]
        L.append("")
    if m.catalysts is not None and len(m.catalysts):
        L.append("## Catalyst calendar (estimated from last year's filing dates)")
        L.append(_md_table(m.catalysts.set_index("event")) + "\n")

    L.append("## What would prove this wrong")
    L += [f"- {c.text}" for c in m.criteria]
    L.append("\n`python -m lighthouse thesis` checks these against every new filing.\n")
    L.append("## Pre-mortem")
    L += [f"- {x}" for x in m.premortem]
    L.append("")
    L.append("## Sources and notes")
    L += [f"- {x}" for x in m.sources]
    L += [f"- Note: {x}" for x in m.notes]
    L.append("- The text is generated from the numbers by fixed rules; it cannot be backtested as judgment, and it is not "
             "investment advice.")
    return "\n".join(L)


def _industry_lines(ind, sic_description: Optional[str] = None) -> List[str]:
    """The memo's industry section: the metrics the generic ratios misread for this kind of company, each with the
    period it covers and the filing that made it public."""
    L = [f"## Industry metrics: {ind.label} (SIC {ind.sic}{' ' + sic_description if sic_description else ''})",
         f"*From the filings public before {ind.as_of.date()}: a value is used from the session after its filing date. "
         "n/a means the company did not tag an input under the standard us-gaap concept for that period (the note "
         "names it); nothing is estimated.*\n",
         _md_table(ind.table()) + "\n"]
    if ind.notes:
        L += [f"- Note: {x}" for x in ind.notes] + [""]
    return L


def _fundamental_view_lines(fv) -> List[str]:
    """The memo's opt-in section: its own numbers' fundamental beacons against the peer medians (or, without peers,
    the sector norms and why), with the score both ways."""
    score = lambda s: "n/a" if s is None else f"{s:.0f}/100"   # noqa: E731
    head = "against the peer medians" if fv.peer_based else "against the sector norms"
    L = [f"## Fundamental beacons {head}",
         f"*Norms: {fv.source}. Fundamental score {score(fv.score)} {head}"
         + (f" ({score(fv.sector_score)} against the sector norms for {fv.sector or 'the overall market'})"
            if fv.peer_based else "")
         + ". The rule-based verdict keeps the sector norms it was tested with; P/E, P/B, EV/EBITDA and operating "
           "margin are scored against the norms, the other beacons against fixed bands.*\n",
         beacon_table(fv.beacons) + "\n"]
    return L


LABELS_PEERS = {"pe": "P/E", "ev_ebitda": "EV/EBITDA", "ev_sales": "EV/sales", "pb": "P/B", "fcf_yield": "FCF yield",
                "gross_margin": "gross margin", "operating_margin": "op. margin", "roic": "ROIC",
                "revenue_growth": "rev. growth", "net_debt_ebitda": "ND/EBITDA"}
