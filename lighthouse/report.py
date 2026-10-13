"""Plain-English report generation (Markdown)."""
from __future__ import annotations

from typing import List

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
        L.append(f"Long-only, reviewed {result.params.decision_frequency} at the close, filled at the {bt.options.get('fills')}, "
                 f"{result.params.cost_bps + result.params.slippage_bps:.0f} bps commission + slippage per side, statements lagged "
                 f"{result.params.reporting_lag_days} days. Returns are for the capital allocated to this stock. "
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
        "- Fundamentals are the latest reported figures; the backtest's fundamental gate uses only the years of statements the feed provides.",
        "- The backtest is long-only with simple costs; it ignores taxes, dividends on cash, and intraday stop execution.",
        "- Past behaviour of a rule is evidence about its *design*, not a promise about its future returns.",
        "- Lighthouse is a decision-support tool, not investment advice.",
    ]
    return "\n".join(L)
