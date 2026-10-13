"""
Charts (matplotlib). One consistent visual system:
  strategy = blue, buy & hold = orange, S&P 500 = aqua (fixed categorical order)
  status   = green / amber / red for beacons, never reused for series
  thin 2px lines, recessive grid, one y-axis per panel (volume gets its own panel)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
import numpy as np
import pandas as pd

try:  # headless servers / CI
    matplotlib.get_backend()
except Exception:  # noqa: BLE001
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from .palette import (DIVERGING_STOPS, GRID, INK, INK2, REGIME_FILL, SEQ_BLUE_STOPS, SERIES, STATUS,  # noqa: F401
                      SURFACE)

SEQ_BLUE = LinearSegmentedColormap.from_list("lh_blue", SEQ_BLUE_STOPS)
DIVERGING = LinearSegmentedColormap.from_list("lh_div", DIVERGING_STOPS)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "legend.fontsize": 9, "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
})


def _style(ax, title: Optional[str] = None):
    if title:
        ax.set_title(title, loc="left")
    ax.tick_params(length=0)
    return ax


def _shade_regimes(ax, regimes: Optional[pd.DataFrame], index: pd.Index):
    """Shade non-benign regimes behind a time-series axis (legend built by caller)."""
    if regimes is None:
        return
    r = regimes["regime"].reindex(index.union(regimes.index)).ffill().reindex(index)
    for label, color in REGIME_FILL.items():
        mask = (r == label).values
        if not mask.any():
            continue
        edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(index[s], index[min(e, len(index) - 1)], color=color, lw=0, zorder=0)


# --------------------------------------------------------------------------- #
def plot_dashboard(result, last_days: int = 756):
    """Price with trend/bands and trade markers, volume, RSI and MACD - all on their own axes."""
    ind = result.indicators.tail(last_days)
    p = result.params
    has_vol = ind["Volume"].notna().any()
    nrows = 4 if has_vol else 3
    heights = [3, 0.9, 1, 1] if has_vol else [3, 1, 1]
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 2.1 * nrows + 2.5), sharex=True, gridspec_kw={"height_ratios": heights})
    axes = list(np.atleast_1d(axes))
    ax = axes[0]
    _shade_regimes(ax, result.regimes, ind.index)
    ax.fill_between(ind.index, ind["band_lower"], ind["band_upper"], color=SERIES["band"], alpha=0.35, lw=0, label=f"Price bands (±{p.band_std:g}σ)")
    ax.plot(ind.index, ind["band_mid"], color=SERIES["band_mid"], lw=1.0, ls="--", label=f"Band centre (Hull MA {p.band_window})")
    ax.plot(ind.index, ind["Close"], color=SERIES["price"], lw=1.6, label="Close")
    ax.plot(ind.index, ind["sma_fast"], color=SERIES["sma_fast"], lw=1.4, label=f"SMA {p.sma_fast}")
    ax.plot(ind.index, ind["sma_slow"], color=SERIES["sma_slow"], lw=1.4, label=f"SMA {p.sma_slow}")
    if result.backtest is not None:
        for t in result.backtest.trades:
            if t.entry_date >= ind.index[0]:
                ax.scatter([t.entry_date], [t.entry_price], marker="^", s=70, color=STATUS["green"], edgecolor=SURFACE, lw=1.2, zorder=5)
            if t.exit_date >= ind.index[0] and not t.exit_reason.startswith("end"):
                ax.scatter([t.exit_date], [t.exit_price], marker="v", s=70, color=STATUS["red"], edgecolor=SURFACE, lw=1.2, zorder=5)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="^", color=STATUS["green"], ls="", ms=8), plt.Line2D([], [], marker="v", color=STATUS["red"], ls="", ms=8)]
    labels += ["strategy entry", "strategy exit"]
    for lab, col in REGIME_FILL.items():
        handles.append(Patch(color=col)); labels.append(lab)
    ax.legend(handles, labels, ncol=4, loc="upper left", fontsize=8)
    _style(ax, f"{result.ticker} · price, trend and price bands (Hull MA {p.band_window} ± {p.band_std:g}σ) · verdict {result.decision.verdict} ({result.decision.composite_score:.0f}/100)")
    ax.set_ylabel("price")
    i = 1
    if has_vol:
        axv = axes[i]; i += 1
        up = ind["Close"] >= ind["Close"].shift(1)
        axv.bar(ind.index, ind["Volume"], width=1.0, color=np.where(up, "#86b6ef", "#c3c2b7"), lw=0)
        _style(axv, "Volume"); axv.set_yticks([])
    axr = axes[i]; i += 1
    axr.plot(ind.index, ind["rsi"], color=SERIES["price"], lw=1.4)
    axr.axhline(p.rsi_overbought, color=STATUS["red"], lw=0.8, ls="--"); axr.axhline(p.rsi_oversold, color=STATUS["green"], lw=0.8, ls="--")
    axr.set_ylim(0, 100); _style(axr, f"RSI ({p.rsi_window}) · dashed = {p.rsi_overbought:.0f}/{p.rsi_oversold:.0f}")
    axm = axes[i]
    axm.bar(ind.index, ind["macd_hist"], width=1.0, color=np.where(ind["macd_hist"] >= 0, "#86b6ef", "#f1b0b0"), lw=0)
    axm.plot(ind.index, ind["macd"], color=SERIES["price"], lw=1.3, label="MACD")
    axm.plot(ind.index, ind["macd_signal"], color=SERIES["sma_fast"], lw=1.3, label="signal")
    axm.legend(loc="upper left", ncol=2); _style(axm, f"MACD ({p.macd_fast}/{p.macd_slow}/{p.macd_signal})")
    fig.tight_layout()
    return fig


def plot_score_breakdown(decision):
    """Each beacon's contribution (points) to its group score - the explainability chart."""
    beacons = [b for b in decision.beacons if not np.isnan(b.score)]
    if not beacons:
        fig, ax = plt.subplots(figsize=(8, 2)); ax.text(0.5, 0.5, "no beacons", ha="center"); return fig
    names = [f"{'F' if b.group == 'fundamental' else 'T'} · {b.name}" for b in beacons]
    vals = [b.contribution for b in beacons]
    colors = [STATUS[b.light] for b in beacons]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(beacons) + 2.2))
    y = np.arange(len(beacons))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    for yi, v, b in zip(y, vals, beacons):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, f"{v:+.1f}  ({b.value})", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    lim = max(abs(min(vals)), abs(max(vals)), 5) * 1.9
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("points added to the group score (−50 … +50 scale, weighted)")
    f = f"Fundamental {decision.fundamental_score:.0f}" if decision.fundamental_score is not None else "Fundamental n/a"
    weather = f"−{decision.regime_penalty:.0f}" if decision.regime_penalty else "0"
    _style(ax, f"Why {decision.ticker} scores {decision.composite_score:.0f}: {f} · Technical {decision.technical_score:.0f} · "
               f"weather {weather} → {decision.verdict}")
    ax.legend(handles=[Patch(color=STATUS["green"], label="bullish"), Patch(color=STATUS["amber"], label="neutral"),
                       Patch(color=STATUS["red"], label="bearish")], loc="lower left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_backtest(bt, regimes: Optional[pd.DataFrame] = None):
    """Equity curves (same scale), drawdown and exposure, with regime shading."""
    fig, (ax, axd, axe) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 0.8]})
    _shade_regimes(ax, regimes, bt.equity.index)
    ax.plot(bt.equity.index, bt.equity, color=SERIES["strategy"], label="Lighthouse strategy")
    ax.plot(bt.buy_hold.index, bt.buy_hold, color=SERIES["buy_hold"], label="Buy & hold")
    if bt.benchmark is not None:
        ax.plot(bt.benchmark.index, bt.benchmark, color=SERIES["benchmark"], label="S&P 500")
    for name, s in [("Lighthouse strategy", bt.equity), ("Buy & hold", bt.buy_hold)] + ([("S&P 500", bt.benchmark)] if bt.benchmark is not None else []):
        ax.annotate(f"{s.iloc[-1]:.2f}x", (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log scale)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x" if v in (0.5, 2, 3, 5, 20, 30, 50, 200, 300, 500) else ""))
    handles, labels = ax.get_legend_handles_labels()
    for lab, col in REGIME_FILL.items():
        handles.append(Patch(color=col)); labels.append(lab)
    ax.legend(handles, labels, ncol=3, loc="upper left", fontsize=8)
    m = bt.metrics
    _style(ax, f"{bt.ticker} · {bt.start.date()} → {bt.end.date()} · strategy CAGR {m['cagr'] * 100:.1f}%, "
               f"max DD {m['max_drawdown'] * 100:.0f}%, {m['n_trades']} trades, win rate {m['win_rate'] * 100:.0f}%")
    from .indicators import drawdown
    axd.fill_between(bt.equity.index, drawdown(bt.equity), 0, color=SERIES["strategy"], alpha=0.45, lw=0, label="strategy")
    axd.plot(bt.buy_hold.index, drawdown(bt.buy_hold), color=SERIES["buy_hold"], lw=1.2, label="buy & hold")
    axd.legend(loc="lower left", ncol=2); axd.set_ylabel("drawdown"); _style(axd, "Drawdown from peak")
    axd.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axe.fill_between(bt.exposure.index, bt.exposure, 0, color="#86b6ef", lw=0)
    axe.set_ylim(0, 1.05); axe.set_ylabel("invested"); _style(axe, "Fraction of the sleeve invested")
    axe.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    return fig


def plot_sensitivity(grid: pd.DataFrame, metric: str = "sharpe"):
    """Entry x exit threshold heatmap. A robust rule shows a plateau, not a spike."""
    piv = grid.pivot(index="exit", columns="entry", values=metric).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = piv.values.astype(float)
    im = ax.imshow(vals, cmap=SEQ_BLUE, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:.0f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{r:.0f}" for r in piv.index])
    ax.set_xlabel("entry threshold (technical score)"); ax.set_ylabel("exit threshold")
    vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 * vmax else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric)
    _style(ax, f"Sensitivity of {metric} to the entry/exit thresholds")
    fig.tight_layout()
    return fig


def plot_walk_forward(wf: dict):
    eq = wf["oos_equity"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(eq.index, eq["default"], color=SERIES["strategy"], label="Default rule")
    ax.plot(eq.index, eq["tuned"], color="#4a3aa7", label="Rule tuned in-sample")
    ax.plot(eq.index, eq["buy_hold"], color=SERIES["buy_hold"], label="Buy & hold")
    ax.legend(loc="upper left"); ax.set_ylabel("growth of $1 (out-of-sample)")
    _style(ax, f"Out-of-sample test: parameters chosen before {wf['split_date'].date()}, applied after it")
    fig.tight_layout()
    return fig


def plot_layers(layers: pd.DataFrame):
    """Strategy layers: CAGR vs max drawdown (two separate axes, same order)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(layers.index)
    y = np.arange(len(names))[::-1]
    colors = [SERIES["buy_hold"] if n == "Buy & hold" else SERIES["benchmark"] if n == "S&P 500" else SERIES["strategy"] for n in names]
    a1.barh(y, layers["cagr"] * 100, color=colors, height=0.6); a1.set_yticks(y); a1.set_yticklabels(names, fontsize=8)
    a1.set_xlabel("CAGR (%)"); _style(a1, "Return by layer")
    a2.barh(y, layers["max_drawdown"] * 100, color=colors, height=0.6); a2.set_yticks(y); a2.set_yticklabels([""] * len(names))
    a2.set_xlabel("Max drawdown (%)"); _style(a2, "Risk by layer")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_screener(df: pd.DataFrame):
    d = df.dropna(subset=["score"]).copy()
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 1.8))
    y = np.arange(len(d))[::-1]
    col = {"Strong Buy": STATUS["green"], "Buy": "#1baf7a", "Hold": STATUS["amber"], "Reduce": "#ec835a", "Sell": STATUS["red"]}
    ax.barh(y, d["score"], color=[col.get(v, STATUS["grey"]) for v in d["verdict"]], height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([f"{t}  ({v})" for t, v in zip(d["ticker"], d["verdict"])], fontsize=9)
    for yi, s in zip(y, d["score"]):
        ax.text(s + 0.8, yi, f"{s:.0f}", va="center", fontsize=8, color=INK2)
    for x, lab in [(60, "Buy ≥60"), (72, "Strong Buy ≥72"), (45, "Hold ≥45")]:
        ax.axvline(x, color=INK2, lw=0.7, ls=":"); ax.text(x, len(d) - 0.3, lab, fontsize=7, color=INK2, ha="center")
    ax.set_xlim(0, 100); ax.set_xlabel("Lighthouse score"); ax.grid(axis="y", visible=False)
    _style(ax, "Watchlist ranking")
    fig.tight_layout()
    return fig


def save_all(result, out_dir) -> List[str]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = []
    figs = {"dashboard": plot_dashboard(result), "score_breakdown": plot_score_breakdown(result.decision)}
    if result.backtest is not None:
        figs["backtest"] = plot_backtest(result.backtest, result.regimes)
    if result.extras.get("sensitivity") is not None:
        figs["sensitivity"] = plot_sensitivity(result.extras["sensitivity"])
    if result.extras.get("walk_forward") is not None:
        figs["walk_forward"] = plot_walk_forward(result.extras["walk_forward"])
    if result.extras.get("layers") is not None:
        figs["layers"] = plot_layers(result.extras["layers"])
    for name, fig in figs.items():
        p = out / f"{result.ticker}_{name}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p))
    return paths
