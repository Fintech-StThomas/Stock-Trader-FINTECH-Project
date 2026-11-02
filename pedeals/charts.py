"""
Interactive charts for the deal-sourcing tab (Plotly), in the dashboard's colour system (the values of
lighthouse/palette.py, repeated so the package also runs on its own in a notebook).

With a dozen sponsors no categorical palette stays readable, so every chart uses one blue for magnitude and puts
identity (sponsor, platform, niche) in the axis labels and the hover text; the tab shows the numbers beside each chart
as a table, so nothing is carried by colour alone.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go

BLUE, BLUE_LIGHT, INK, INK2, GRID, SURFACE = "#2a78d6", "#9ec5f4", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
FONT = dict(family="Source Sans Pro, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK)
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"], "responsive": True}


def _layout(fig: go.Figure, title: str, height: int, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15), x=0, xanchor="left"), height=height, font=FONT,
                      margin=dict(l=10, r=10, t=48, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font_size=12), showlegend=False, **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def deals_by_quarter(deals: pd.DataFrame, title: str = "Add-on acquisitions per quarter", height: int = 320) -> go.Figure:
    d = deals.assign(quarter=pd.PeriodIndex(pd.to_datetime(deals["published_date"]), freq="Q").astype(str))
    t = d.groupby("quarter").agg(n=("deal_id", "count"), sponsors=("sponsor", lambda s: ", ".join(sorted(set(s))))).reset_index()
    fig = go.Figure(go.Bar(x=t["quarter"], y=t["n"], marker=dict(color=BLUE), customdata=t[["sponsors"]].to_numpy(),
                           hovertemplate="<b>%{x}</b>: %{y} add-ons<br>by platforms of %{customdata[0]}<extra></extra>"))
    fig.update_yaxes(title_text="announcements")
    return _layout(fig, title, height)


def horizontal_counts(table: pd.DataFrame, label: str, value: str, title: str, hover: str = "", height: int = 0) -> go.Figure:
    """One-hue horizontal bars, largest first (niches, rationale tags, industries)."""
    t = table.sort_values(value)
    fig = go.Figure(go.Bar(y=t[label], x=t[value], orientation="h", marker=dict(color=BLUE),
                           text=t[value], textposition="outside", cliponaxis=False,
                           hovertemplate="<b>%{y}</b><br>%{x} " + (hover or value) + "<extra></extra>"))
    fig.update_xaxes(showgrid=False, showticklabels=False)
    return _layout(fig, title, height or max(220, 30 * len(t) + 70))


def years_held(watch: pd.DataFrame, typical: float = 4.0, height: int = 0) -> go.Figure:
    """How long each candidate has been held, with the usual 4-6 year exit window shaded."""
    t = watch.head(25).iloc[::-1]
    labels = [f"{c} ({s})" for c, s in zip(t["company"], t["sponsor"])]
    fig = go.Figure(go.Bar(y=labels, x=t["years_held"], orientation="h",
                           marker=dict(color=[BLUE if b == "from entry" else BLUE_LIGHT for b in t["years_basis"]]),
                           customdata=t[["entry_basis", "add_ons", "focus_sector"]].to_numpy(),
                           hovertemplate="<b>%{y}</b><br>%{x:.1f} years held<br>entry: %{customdata[0]}<br>"
                                         "%{customdata[1]} add-ons · %{customdata[2]}<extra></extra>"))
    fig.add_vrect(x0=typical, x1=6, fillcolor=BLUE_LIGHT, opacity=0.18, line_width=0,
                  annotation_text="typical exit window", annotation_position="top left", annotation_font_color=INK2)
    fig.update_xaxes(title_text="years since the investment (light bars: at least, entry date not stated)")
    return _layout(fig, "Sponsor-owned companies by years held, no exit announced", height or max(260, 26 * len(t) + 110))


def platform_timeline(addons: pd.DataFrame, platforms: List[str], height: int = 0) -> go.Figure:
    """Each add-on as a dot on its platform's row: who is buying in this niche, and how steadily."""
    a = addons[addons["platform"].isin(platforms)].copy()
    a["when"] = pd.to_datetime(a["published_date"])
    fig = go.Figure(go.Scatter(x=a["when"], y=a["platform"], mode="markers",
                               marker=dict(size=11, color=BLUE, line=dict(color=SURFACE, width=2)),
                               customdata=a[["target", "sponsor", "target_state", "niche"]].fillna("").to_numpy(),
                               hovertemplate="<b>%{y}</b> (%{customdata[1]}) acquired <b>%{customdata[0]}</b><br>"
                                             "%{x|%b %d, %Y} · %{customdata[2]}<br>%{customdata[3]}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(platforms)))
    return _layout(fig, "Their add-ons over time (hover a dot for the deal)", height or max(240, 30 * len(platforms) + 100))
