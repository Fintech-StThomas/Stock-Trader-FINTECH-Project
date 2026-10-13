"""
Interactive charts (Plotly) for the Streamlit app.

Same visual system as `charts.py` (strategy = blue, buy & hold = orange, S&P 500 = aqua;
beacon lights = green / amber / red) but every figure can be hovered, zoomed and
downloaded, and every mark carries its explanation in the tooltip.

The matplotlib charts in `charts.py` remain the ones used by the CLI, the Markdown
report and the notebook; this module is only imported by the app.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .palette import SERIES, STATUS, REGIME_FILL, INK2, GRID
from .config import VERDICT_BANDS

VERDICT_COLOR = {"Strong Buy": STATUS["green"], "Buy": "#1baf7a", "Hold": STATUS["amber"], "Reduce": "#ec835a", "Sell": STATUS["red"]}
LIGHT_WORD = {"green": "bullish", "amber": "neutral", "red": "bearish", "grey": "unavailable"}
FONT = dict(family="Source Sans Pro, Segoe UI, Helvetica, Arial, sans-serif", size=13)
_MODEBAR_REMOVE = ["lasso2d", "select2d", "autoScale2d"]
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": _MODEBAR_REMOVE, "responsive": True,
                 "toImageButtonOptions": {"format": "png", "scale": 2}}


def _base_layout(fig: go.Figure, title: Optional[str] = None, height: Optional[int] = None, **kw) -> go.Figure:
    layout = dict(
        template="plotly_white", font=FONT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=56 if title else 24, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=12)),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
    )
    layout.update(kw)   # a caller's choice (e.g. hovermode="closest") wins over the defaults
    fig.update_layout(**layout)
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return fig


def _regime_spans(regimes: Optional[pd.DataFrame], index: pd.Index) -> List[Tuple[pd.Timestamp, pd.Timestamp, str, str]]:
    """Contiguous (start, end, label, colour) spans of the non-benign regimes over `index`."""
    if regimes is None or len(index) == 0:
        return []
    r = regimes["regime"].reindex(index.union(regimes.index)).ffill().reindex(index)
    spans = []
    for label, color in REGIME_FILL.items():
        mask = (r == label).values
        if not mask.any():
            continue
        edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            spans.append((index[s], index[min(e, len(index) - 1)], label, color))
    return spans


def _shade_regimes(fig: go.Figure, regimes: Optional[pd.DataFrame], index: pd.Index, row: int = 1) -> List[Tuple[str, str]]:
    """Draw the regime rectangles behind panel `row`; returns the (label, colour) pairs used, for `_regime_legend`."""
    used: Dict[str, str] = {}
    try:
        sub = fig.get_subplot(row, 1)
        xref, yref = sub.xaxis.plotly_name.replace("axis", ""), sub.yaxis.plotly_name.replace("axis", "")
    except Exception:  # noqa: BLE001 - a figure without subplots
        xref, yref = "x", "y"
    shapes = []
    for x0, x1, label, color in _regime_spans(regimes, index):
        # one layout update for all the spans: fig.add_vrect re-validates the whole layout on every call, which took
        # seconds for a long history. Shapes set directly are also never dropped for a subplot without traces yet
        # (the add_vrect default that once hid the shading).
        shapes.append(dict(type="rect", xref=xref, yref=f"{yref} domain", x0=x0, x1=x1, y0=0, y1=1, fillcolor=color,
                           opacity=0.55, layer="below", line_width=0))
        used[label] = color
    if shapes:
        fig.update_layout(shapes=list(fig.layout.shapes or ()) + shapes)
    return list(used.items())


def _regime_legend(fig: go.Figure, used: List[Tuple[str, str]], row: int = 1) -> None:
    """Legend swatches for the regimes. Added AFTER the real traces: Plotly infers an axis's type from its first
    trace, and a swatch's null x would turn a date axis into a linear one (every date then becomes NaN)."""
    for label, color in used:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label, hoverinfo="skip",
                                 marker=dict(size=12, symbol="square", color=color)), row=row, col=1)
    fig.update_xaxes(type="date", row=row, col=1)


def _range_buttons(x: float = 1.0, xanchor: str = "right") -> dict:
    return dict(buttons=[dict(count=6, label="6m", step="month", stepmode="backward"),
                         dict(count=1, label="1y", step="year", stepmode="backward"),
                         dict(count=3, label="3y", step="year", stepmode="backward"),
                         dict(step="all", label="all")],
                bgcolor="rgba(0,0,0,0)", activecolor="#dde9f8", font=dict(size=11), x=x, y=1.0, xanchor=xanchor, yanchor="bottom")


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def score_gauge(decision, height: int = 230) -> go.Figure:
    """Lighthouse score on a 0-100 dial with the verdict bands behind it."""
    bands = sorted(VERDICT_BANDS)   # ascending thresholds
    steps = []
    for i, (lo, label) in enumerate(bands):
        hi = bands[i + 1][0] if i + 1 < len(bands) else 100
        steps.append(dict(range=[lo, hi], color=VERDICT_COLOR[label], thickness=0.7))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(decision.composite_score),
        number=dict(suffix=" / 100", font=dict(size=30)),
        title=dict(text=f"<b>{decision.verdict}</b>", font=dict(size=18, color=VERDICT_COLOR[decision.verdict])),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 32, 45, 60, 72, 100], tickfont=dict(size=11)),
                   bar=dict(color="#0b0b0b", thickness=0.25), bgcolor="rgba(0,0,0,0)", borderwidth=0, steps=steps,
                   threshold=dict(line=dict(color="#0b0b0b", width=3), thickness=0.8, value=decision.composite_score)),
    ))
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", font=FONT)
    return fig


def score_waterfall(decision, profile, height: int = 300) -> go.Figure:
    """How the composite is built: fundamental x weight + technical x weight - market weather = score."""
    labels, values, text = [], [], []
    if decision.fundamental_score is not None:
        labels.append(f"Fundamental {decision.fundamental_score:.0f} × {profile.fundamental_weight:.0%}")
        values.append(decision.fundamental_score * profile.fundamental_weight)
        labels.append(f"Technical {decision.technical_score:.0f} × {profile.technical_weight:.0%}")
        values.append(decision.technical_score * profile.technical_weight)
    else:
        labels.append(f"Technical {decision.technical_score:.0f} × 100% (fundamentals unavailable)")
        values.append(decision.technical_score)
    labels.append("Market weather" + (f" ({decision.regime.label})" if decision.regime else ""))
    values.append(-decision.regime_penalty)
    text = [f"{v:+.1f}" for v in values]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"] * len(values) + ["total"],
        x=labels + ["Lighthouse score"], y=values + [decision.composite_score], text=text + [f"{decision.composite_score:.0f}"],
        textposition="outside", connector=dict(line=dict(color=INK2, width=1)),
        increasing=dict(marker=dict(color=SERIES["strategy"])), decreasing=dict(marker=dict(color=STATUS["red"])),
        totals=dict(marker=dict(color=VERDICT_COLOR[decision.verdict])),
        hovertemplate="%{x}<br>%{text} points<extra></extra>",
    ))
    for thr, label in VERDICT_BANDS[:-1]:
        fig.add_hline(y=thr, line=dict(color=INK2, width=0.7, dash="dot"), annotation_text=f"{label} ≥{thr}",
                      annotation_position="right", annotation_font=dict(size=10, color=INK2))
    _base_layout(fig, title="How the score is built", height=height, showlegend=False, hovermode="closest")
    fig.update_yaxes(range=[0, 105], title="points")
    return fig


# --------------------------------------------------------------------------- #
# Beacons
# --------------------------------------------------------------------------- #
def score_breakdown(decision, height: Optional[int] = None, beacons=None) -> go.Figure:
    """Each beacon's contribution in points; hover shows the reading and the reason. `beacons` filters the list shown."""
    beacons = [b for b in (decision.beacons if beacons is None else beacons) if not np.isnan(b.score)]
    if not beacons:
        fig = go.Figure()
        fig.add_annotation(text="No beacons available", showarrow=False)
        return _base_layout(fig, height=200)
    names = [f"{'F' if b.group == 'fundamental' else 'T'} · {b.name}" for b in beacons][::-1]
    vals = [b.contribution for b in beacons][::-1]
    colors = [STATUS[b.light] for b in beacons][::-1]
    custom = [[b.value, LIGHT_WORD[b.light], f"{b.score:+.2f}", f"{b.weight * 100:.0f}%", b.rationale] for b in beacons][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=colors), customdata=custom,
        text=[f"{v:+.1f}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>reading: %{customdata[0]}<br>light: %{customdata[1]} · vote %{customdata[2]} · weight %{customdata[3]}"
                      "<br>points: %{x:+.1f}<br><i>%{customdata[4]}</i><extra></extra>", showlegend=False,
    ))
    for light, word in (("green", "bullish"), ("amber", "neutral"), ("red", "bearish")):   # legend swatches only
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, symbol="square", color=STATUS[light]),
                                 name=word, hoverinfo="skip"))
    lim = max(abs(min(vals)), abs(max(vals)), 5) * 1.35
    f = f"Fundamental {decision.fundamental_score:.0f}" if decision.fundamental_score is not None else "Fundamental n/a"
    weather = f"−{decision.regime_penalty:.0f}" if decision.regime_penalty else "0"
    _base_layout(fig, title=f"Why {decision.ticker} scores {decision.composite_score:.0f}: {f} · Technical {decision.technical_score:.0f} · weather {weather} → {decision.verdict}",
                 height=height or int(30 * len(beacons) + 120), hovermode="closest", barmode="overlay")
    fig.add_vline(x=0, line=dict(color=INK2, width=1))
    fig.update_xaxes(range=[-lim, lim], title="points added to the group score (−50 … +50 scale, weighted)")
    fig.update_yaxes(tickfont=dict(size=12))
    return fig


# --------------------------------------------------------------------------- #
# Price dashboard
# --------------------------------------------------------------------------- #
def _band_spec(p) -> dict:
    """Columns and labels of the price-band overlay: a Hull-MA centre +/- k standard deviations of the closes around it."""
    return dict(up="band_upper", lo="band_lower", mid="band_mid", name=f"Price bands (±{p.band_std:g}σ)",
                mid_name=f"Band centre (Hull MA {p.band_window})", short=f"Hull {p.band_window}",
                title=f"price bands (Hull MA {p.band_window} ± {p.band_std:g}σ)", extra="price bands")


def _add_bands(fig, ind: pd.DataFrame, spec: dict, alpha: float, **rc) -> None:
    fig.add_trace(go.Scatter(x=ind.index, y=ind[spec["up"]], line=dict(width=0), hoverinfo="skip", showlegend=False), **rc)
    fig.add_trace(go.Scatter(x=ind.index, y=ind[spec["lo"]], line=dict(width=0), fill="tonexty",
                             fillcolor=f"rgba(158,197,244,{alpha})", name=spec["name"], hoverinfo="skip"), **rc)
    # The centre line is added after the lower band so the fill above still runs upper-to-lower.
    fig.add_trace(go.Scatter(x=ind.index, y=ind[spec["mid"]], name=spec["mid_name"],
                             line=dict(color=SERIES["band_mid"], width=1.2, dash="dash"),
                             customdata=np.column_stack([ind[spec["up"]], ind[spec["lo"]]]),
                             hovertemplate=spec["short"] + " %{y:,.2f}<br>bands %{customdata[1]:,.2f} – %{customdata[0]:,.2f}"
                                           f"<extra>{spec['extra']}</extra>"), **rc)


def dashboard(result, last_days: int = 756, height: Optional[int] = None) -> go.Figure:
    """Price with trend/bands/trades/regimes, volume, RSI and MACD: four linked panels with range buttons."""
    ind = result.indicators.tail(last_days)
    p = result.params
    spec = _band_spec(p)
    has_vol = ind["Volume"].notna().any()
    rows = 4 if has_vol else 3
    heights = [0.5, 0.14, 0.18, 0.18] if has_vol else [0.56, 0.22, 0.22]
    titles = [f"Price, trend and {spec['title']}"] + (["Volume"] if has_vol else []) + \
             [f"RSI ({p.rsi_window}) · dashed = {p.rsi_overbought:.0f} / {p.rsi_oversold:.0f}", f"MACD ({p.macd_fast}/{p.macd_slow}/{p.macd_signal})"]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.045, row_heights=heights, subplot_titles=titles)

    used = _shade_regimes(fig, result.regimes, ind.index, row=1)
    _add_bands(fig, ind, spec, 0.35, row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color=SERIES["price"], width=1.8),
                             hovertemplate="close $%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["sma_fast"], name=f"SMA {p.sma_fast}", line=dict(color=SERIES["sma_fast"], width=1.3),
                             hovertemplate="SMA %{y:,.2f}<extra>" + f"SMA {p.sma_fast}" + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["sma_slow"], name=f"SMA {p.sma_slow}", line=dict(color=SERIES["sma_slow"], width=1.3),
                             hovertemplate="SMA %{y:,.2f}<extra>" + f"SMA {p.sma_slow}" + "</extra>"), row=1, col=1)
    if result.backtest is not None:
        ent = [(t.entry_date, t.entry_price, t.size_fraction) for t in result.backtest.trades if t.entry_date >= ind.index[0]]
        ext = [(t.exit_date, t.exit_price, t.exit_reason, t.return_pct) for t in result.backtest.trades
               if t.exit_date >= ind.index[0] and not t.exit_reason.startswith("end")]
        if ent:
            fig.add_trace(go.Scatter(x=[e[0] for e in ent], y=[e[1] for e in ent], mode="markers", name="strategy entry",
                                     marker=dict(symbol="triangle-up", size=13, color=STATUS["green"], line=dict(color="white", width=1.2)),
                                     customdata=[[e[2] * 100] for e in ent],
                                     hovertemplate="entry $%{y:,.2f}<br>size %{customdata[0]:.0f}% of sleeve<extra>entry</extra>"), row=1, col=1)
        if ext:
            fig.add_trace(go.Scatter(x=[e[0] for e in ext], y=[e[1] for e in ext], mode="markers", name="strategy exit",
                                     marker=dict(symbol="triangle-down", size=13, color=STATUS["red"], line=dict(color="white", width=1.2)),
                                     customdata=[[e[2], e[3] * 100] for e in ext],
                                     hovertemplate="exit $%{y:,.2f}<br>%{customdata[0]}<br>trade %{customdata[1]:+.1f}%<extra>exit</extra>"), row=1, col=1)
    r = 2
    if has_vol:
        up = ind["Close"] >= ind["Close"].shift(1)
        fig.add_trace(go.Bar(x=ind.index, y=ind["Volume"], name="Volume", marker=dict(color=np.where(up, "#86b6ef", "#c3c2b7"), line_width=0),
                             hovertemplate="volume %{y:,.0f}<extra></extra>", showlegend=False), row=r, col=1)
        r += 1
    fig.add_trace(go.Scatter(x=ind.index, y=ind["rsi"], name="RSI", line=dict(color=SERIES["price"], width=1.3),
                             hovertemplate="RSI %{y:.1f}<extra></extra>", showlegend=False), row=r, col=1)
    fig.add_hline(y=p.rsi_overbought, line=dict(color=STATUS["red"], width=0.9, dash="dash"), row=r, col=1)
    fig.add_hline(y=p.rsi_oversold, line=dict(color=STATUS["green"], width=0.9, dash="dash"), row=r, col=1)
    fig.update_yaxes(range=[0, 100], row=r, col=1)
    r += 1
    fig.add_trace(go.Bar(x=ind.index, y=ind["macd_hist"], name="MACD histogram",
                         marker=dict(color=np.where(ind["macd_hist"] >= 0, "#86b6ef", "#f1b0b0"), line_width=0),
                         hovertemplate="histogram %{y:.3f}<extra></extra>", showlegend=False), row=r, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["macd"], name="MACD", line=dict(color=SERIES["price"], width=1.2),
                             hovertemplate="MACD %{y:.3f}<extra></extra>"), row=r, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["macd_signal"], name="signal", line=dict(color=SERIES["sma_fast"], width=1.2),
                             hovertemplate="signal %{y:.3f}<extra></extra>"), row=r, col=1)

    _regime_legend(fig, used, row=1)
    _base_layout(fig, height=height or (820 if has_vol else 700), bargap=0)
    fig.update_layout(legend=dict(y=1.04), margin=dict(t=70))
    fig.update_xaxes(rangeselector=_range_buttons(), row=1, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor=INK2)
    fig.update_yaxes(title="price ($)", row=1, col=1)
    for ann in fig.layout.annotations:
        ann.update(x=0, xanchor="left", font=dict(size=12, color=INK2))
    return fig


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(bt, regimes: Optional[pd.DataFrame] = None, height: int = 640) -> go.Figure:
    """Equity curves (log), drawdown and exposure, with regime shading and end-of-period values."""
    from .indicators import drawdown
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.56, 0.26, 0.18],
                        subplot_titles=["Growth of $1 (log scale)", "Drawdown from peak", "Fraction of the sleeve invested"])
    used = _shade_regimes(fig, regimes, bt.equity.index, row=1)
    series = [("Lighthouse strategy", bt.equity, SERIES["strategy"]), ("Buy & hold", bt.buy_hold, SERIES["buy_hold"])]
    if bt.benchmark is not None:
        series.append(("S&P 500", bt.benchmark, SERIES["benchmark"]))
    for name, s, color in series:
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name, line=dict(color=color, width=1.7),
                                 hovertemplate="%{y:.2f}x<extra>" + name + "</extra>"), row=1, col=1)
        fig.add_annotation(x=s.index[-1], y=np.log10(s.iloc[-1]), text=f"{s.iloc[-1]:.2f}x", showarrow=False, xanchor="left",
                           xshift=4, font=dict(size=11, color=color), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.equity.index, y=drawdown(bt.equity), name="strategy drawdown", fill="tozeroy",
                             line=dict(color=SERIES["strategy"], width=1), fillcolor="rgba(42,120,214,0.35)",
                             hovertemplate="%{y:.1%}<extra>strategy drawdown</extra>", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=bt.buy_hold.index, y=drawdown(bt.buy_hold), name="buy & hold drawdown",
                             line=dict(color=SERIES["buy_hold"], width=1.2), hovertemplate="%{y:.1%}<extra>buy & hold drawdown</extra>",
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=bt.exposure.index, y=bt.exposure, name="invested", fill="tozeroy", line=dict(color="#86b6ef", width=0.8),
                             fillcolor="rgba(134,182,239,0.6)", hovertemplate="%{y:.0%} invested<extra></extra>", showlegend=False), row=3, col=1)
    _regime_legend(fig, used, row=1)
    m = bt.metrics
    _base_layout(fig, title=f"{bt.ticker} · {bt.start.date()} → {bt.end.date()} · strategy CAGR {m['cagr'] * 100:.1f}%, "
                            f"max drawdown {m['max_drawdown'] * 100:.0f}%, {m['n_trades']} trades, win rate {m['win_rate'] * 100:.0f}%",
                 height=height)
    fig.update_layout(legend=dict(y=1.02), margin=dict(t=90))
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=3, col=1)
    fig.update_xaxes(rangeselector=_range_buttons(), row=1, col=1)
    for ann in fig.layout.annotations:
        if ann.text in ("Growth of $1 (log scale)", "Drawdown from peak", "Fraction of the sleeve invested"):
            ann.update(x=0, xanchor="left", font=dict(size=12, color=INK2))
    return fig


def layers(df: pd.DataFrame, height: Optional[int] = None) -> go.Figure:
    """Return and risk by strategy layer, side by side."""
    names = list(df.index)
    colors = [SERIES["buy_hold"] if n == "Buy & hold" else SERIES["benchmark"] if n == "S&P 500" else SERIES["strategy"] for n in names]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04, subplot_titles=["CAGR (%)", "Max drawdown (%)"])
    fig.add_trace(go.Bar(y=names[::-1], x=(df["cagr"] * 100)[::-1], orientation="h", marker=dict(color=colors[::-1]),
                         text=[f"{v:.1f}%" for v in (df["cagr"] * 100)[::-1]], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}<br>CAGR %{x:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(y=names[::-1], x=(df["max_drawdown"] * 100)[::-1], orientation="h", marker=dict(color=colors[::-1]),
                         text=[f"{v:.0f}%" for v in (df["max_drawdown"] * 100)[::-1]], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}<br>max drawdown %{x:.0f}%<extra></extra>"), row=1, col=2)
    _base_layout(fig, height=height or int(44 * len(names) + 110), showlegend=False, hovermode="closest")
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=1)
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12, color=INK2))
    return fig


def sensitivity(grid: pd.DataFrame, metric: str = "sharpe", height: int = 420) -> go.Figure:
    """Entry x exit threshold heatmap: a robust rule shows a plateau, not a spike."""
    piv = grid.pivot(index="exit", columns="entry", values=metric).sort_index(ascending=False)
    vals = piv.values.astype(float)
    fmt = ".2f" if metric in {"sharpe", "sortino"} else ".1%"
    fig = go.Figure(go.Heatmap(
        z=vals, x=[f"{c:.0f}" for c in piv.columns], y=[f"{r:.0f}" for r in piv.index],
        colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]], colorbar=dict(title=metric, thickness=12),
        text=[[("" if not np.isfinite(v) else format(v, fmt)) for v in row] for row in vals], texttemplate="%{text}",
        hovertemplate="entry %{x} · exit %{y}<br>" + metric + " %{z:" + fmt + "}<extra></extra>", hoverongaps=False,
    ))
    _base_layout(fig, title=f"Sensitivity of {metric} to the entry / exit thresholds", height=height, hovermode="closest")
    fig.update_xaxes(title="entry threshold (technical score)", type="category")
    fig.update_yaxes(title="exit threshold", type="category")
    return fig


def walk_forward(wf: dict, height: int = 380) -> go.Figure:
    eq = wf["oos_equity"]
    fig = go.Figure()
    for col, name, color in (("default", "Default rule", SERIES["strategy"]), ("tuned", "Rule tuned in-sample", "#4a3aa7"),
                             ("buy_hold", "Buy & hold", SERIES["buy_hold"])):
        fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name=name, line=dict(color=color, width=1.7),
                                 hovertemplate="%{y:.2f}x<extra>" + name + "</extra>"))
    _base_layout(fig, title=f"Out-of-sample test: thresholds chosen before {wf['split_date'].date()}, applied after it", height=height)
    fig.update_yaxes(title="growth of $1 (out-of-sample)")
    return fig


# --------------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------------- #
def screener(df: pd.DataFrame, height: Optional[int] = None) -> go.Figure:
    d = df.dropna(subset=["score"]).copy()
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="No ticker could be scored", showarrow=False)
        return _base_layout(fig, height=200)
    labels = [f"{t}  ({v})" for t, v in zip(d["ticker"], d["verdict"])][::-1]
    custom = d[["name", "fundamental", "technical", "regime", "position_pct"]].fillna("n/a").values.tolist()[::-1]
    fig = go.Figure(go.Bar(
        x=d["score"][::-1], y=labels, orientation="h", marker=dict(color=[VERDICT_COLOR.get(v, STATUS["grey"]) for v in d["verdict"]][::-1]),
        text=[f"{s:.0f}" for s in d["score"]][::-1], textposition="outside", cliponaxis=False, customdata=custom,
        hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>score %{x:.0f} · fundamental %{customdata[1]} · technical %{customdata[2]}"
                      "<br>market weather %{customdata[3]} · suggested position %{customdata[4]}%<extra></extra>",
    ))
    for thr, label in VERDICT_BANDS[:-1]:
        fig.add_vline(x=thr, line=dict(color=INK2, width=0.8, dash="dot"), annotation_text=f"{label} ≥{thr}",
                      annotation_position="top", annotation_font=dict(size=10, color=INK2))
    _base_layout(fig, title="Watchlist ranked by Lighthouse score", height=height or int(34 * len(d) + 130), showlegend=False, hovermode="closest")
    fig.update_xaxes(range=[0, 108], title="Lighthouse score")
    return fig


def compare_prices(series: Dict[str, pd.Series], height: int = 420) -> go.Figure:
    """Normalised price paths (start = 100) of several tickers on one axis."""
    fig = go.Figure()
    palette = [SERIES["strategy"], SERIES["buy_hold"], SERIES["benchmark"], "#4a3aa7", "#d03b3b", "#fab219", "#1c5cab", "#ec835a", "#0ca30c", "#9a9892"]
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name=name, line=dict(color=palette[i % len(palette)], width=1.6),
                                 hovertemplate="%{y:.0f}<extra>" + name + "</extra>"))
    _base_layout(fig, title="Price performance, rebased to 100", height=height)
    fig.update_layout(legend=dict(y=1.08), margin=dict(t=90))
    fig.update_xaxes(rangeselector=_range_buttons(), type="date")
    fig.update_yaxes(title="value of 100 invested")
    return fig


def all_figures(result) -> Dict[str, go.Figure]:
    """Every figure the app shows for one analysis (used by the tests)."""
    figs = {"gauge": score_gauge(result.decision), "waterfall": score_waterfall(result.decision, result.profile),
            "breakdown": score_breakdown(result.decision), "dashboard": dashboard(result)}
    if result.backtest is not None:
        figs["backtest"] = backtest(result.backtest, result.regimes)
    if result.extras.get("layers") is not None:
        figs["layers"] = layers(result.extras["layers"])
    if result.extras.get("sensitivity") is not None:
        figs["sensitivity"] = sensitivity(result.extras["sensitivity"])
    if result.extras.get("walk_forward") is not None:
        figs["walk_forward"] = walk_forward(result.extras["walk_forward"])
    return figs


# --------------------------------------------------------------------------- #
# Trade desk
# --------------------------------------------------------------------------- #
def trade_chart(result, plan, last_days: int = 180, height: int = 420) -> go.Figure:
    """Recent price with the ticket's levels: entry, stop, targets (and the price bands for context)."""
    ind = result.indicators.tail(last_days)
    fig = go.Figure()
    _add_bands(fig, ind, _band_spec(result.params), 0.25)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Close", line=dict(color=SERIES["price"], width=1.8),
                             hovertemplate="close $%{y:,.2f}<extra></extra>"))
    levels = [("entry", plan.price, INK2, "solid")]
    if plan.stop is not None:
        levels.append(("stop", plan.stop, STATUS["red"], "dash"))
    for k, t in enumerate(plan.targets):
        levels.append((f"target {k + 1} ({t.r_multiple:g}R)", t.price, STATUS["green"], "dot"))
    # one layout update (fig.add_hline re-validates the whole layout on every call)
    fig.update_layout(
        shapes=[dict(type="line", xref="x domain", yref="y", x0=0, x1=1, y0=y, y1=y, line=dict(color=color, width=1.4, dash=dash))
                for _, y, color, dash in levels],
        annotations=[dict(xref="x domain", yref="y", x=1, y=y, xanchor="left", yanchor="middle", showarrow=False,
                          text=f"{label} ${y:,.2f}", font=dict(size=11, color=color)) for label, y, color, _ in levels])
    side = "LONG" if plan.side == "long" else "SHORT"
    _base_layout(fig, title=f"{plan.ticker} · {side} ticket levels · data to {plan.as_of}", height=height)
    fig.update_layout(margin=dict(r=150))
    fig.update_yaxes(title="price ($)")
    return fig


# --------------------------------------------------------------------------- #
# Portfolio builder
# --------------------------------------------------------------------------- #
def allocation_chart(table: pd.DataFrame, height: Optional[int] = None) -> go.Figure:
    """Capital weight next to risk share for every holding: where the money is vs where the risk is."""
    d = table.sort_values("target_weight")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d.index, x=d["target_weight"] * 100, orientation="h", name="share of capital",
                         marker=dict(color=SERIES["strategy"]), hovertemplate="%{y}: %{x:.1f}% of capital<extra></extra>"))
    fig.add_trace(go.Bar(y=d.index, x=d["risk_share"] * 100, orientation="h", name="share of risk",
                         marker=dict(color=SERIES["buy_hold"]), hovertemplate="%{y}: %{x:.1f}% of the risk<extra></extra>"))
    _base_layout(fig, title="Where the money is vs where the risk is", height=height or int(28 * len(d) + 140),
                 barmode="group", hovermode="closest")
    fig.update_xaxes(title="% of capital (cash excluded) · % of portfolio volatility")
    return fig


def correlation_heatmap(corr: pd.DataFrame, height: Optional[int] = None) -> go.Figure:
    n = len(corr)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1,
        colorscale=[[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]], colorbar=dict(title="ρ", thickness=12),
        text=[[f"{v:.2f}" for v in row] for row in corr.values], texttemplate="%{text}" if n <= 15 else None,
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    _base_layout(fig, title="Correlation of daily returns (lower = better diversification)", height=height or max(320, 28 * n + 120),
                 hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    return fig


def equity_curves(runs: Dict[str, object], height: int = 520) -> go.Figure:
    """Walk-forward growth of $1 (log) and drawdown for several portfolio backtests."""
    from .indicators import drawdown
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.68, 0.32],
                        subplot_titles=["Growth of $1, walk-forward (log scale)", "Drawdown from peak"])
    palette = [SERIES["strategy"], "#4a3aa7", SERIES["buy_hold"], "#d03b3b", "#fab219", "#1c5cab", "#ec835a", SERIES["benchmark"]]
    for i, (key, bt) in enumerate(runs.items()):
        color = SERIES["benchmark"] if key == "benchmark" else palette[i % len(palette)]
        eq = bt.equity
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=bt.label, line=dict(color=color, width=1.6),
                                 hovertemplate="%{y:.2f}x<extra>" + bt.label + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=eq.index, y=drawdown(eq), name=bt.label, line=dict(color=color, width=1), showlegend=False,
                                 hovertemplate="%{y:.1%}<extra>" + bt.label + "</extra>"), row=2, col=1)
    _base_layout(fig, height=height)
    fig.update_layout(legend=dict(y=1.06), margin=dict(t=80))
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.update(x=0, xanchor="left", font=dict(size=12, color=INK2))
    return fig
