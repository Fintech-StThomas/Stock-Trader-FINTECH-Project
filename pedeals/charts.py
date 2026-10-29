"""
Interactive charts for the deal data (Plotly), in the dashboard's colour system (the values of lighthouse/palette.py,
repeated here so the package also runs on its own in a notebook).

Colour follows the entity and never its rank: each firm keeps its hue in every chart, assigned in a fixed order (the
four-hue order was checked for colour-blind separation). Magnitudes use one blue. Every chart has hover text and the
tab shows the numbers beside it as a table, so nothing is carried by colour alone.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from .sources import SOURCES

FIRM_ORDER: List[str] = [s.firm for s in SOURCES.values()]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]          # blue, orange, aqua, violet: fixed order
FIRM_COLOR: Dict[str, str] = {f: CATEGORICAL[i % len(CATEGORICAL)] for i, f in enumerate(FIRM_ORDER)}
BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
FONT = dict(family="Source Sans Pro, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK)
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"], "responsive": True}


def _layout(fig: go.Figure, title: str, height: int, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=15), x=0, xanchor="left"), height=height, font=FONT,
                      margin=dict(l=10, r=10, t=48, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font_size=12), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def deals_by_quarter(deals: pd.DataFrame, height: int = 360) -> go.Figure:
    """Deal announcements per quarter, one bar segment per firm (stacked; a 2px gap separates the segments)."""
    d = deals.assign(quarter=pd.PeriodIndex(pd.to_datetime(deals["published_date"]), freq="Q").astype(str))
    t = d.pivot_table(index="quarter", columns="publisher", values="deal_id", aggfunc="count", fill_value=0)
    fig = go.Figure()
    for firm in [f for f in FIRM_ORDER if f in t.columns]:
        fig.add_bar(x=t.index, y=t[firm], name=firm, marker=dict(color=FIRM_COLOR[firm], line=dict(color="#fcfcfb", width=2)),
                    hovertemplate=f"<b>{firm}</b><br>%{{x}}: %{{y}} deal announcements<extra></extra>")
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.18, x=0, traceorder="normal"))
    fig.update_yaxes(title_text="announcements")
    return _layout(fig, "Deal announcements per quarter, by publishing firm", height)


def horizontal_counts(table: pd.DataFrame, label: str, value: str, title: str, hover: str = "", height: int = 0) -> go.Figure:
    """One-hue horizontal bars, largest first (rationale tags, industries)."""
    t = table.sort_values(value)
    fig = go.Figure(go.Bar(y=t[label], x=t[value], orientation="h", marker=dict(color=BLUE),
                           text=t[value], textposition="outside", cliponaxis=False,
                           hovertemplate="<b>%{y}</b><br>%{x} " + (hover or value) + "<extra></extra>"))
    fig.update_xaxes(showgrid=False, showticklabels=False)
    return _layout(fig, title, height or max(220, 34 * len(t) + 70))


def buy_and_build_timeline(addons: pd.DataFrame, platforms: List[str], height: int = 0) -> go.Figure:
    """Each add-on as a dot on its platform's row, coloured by the sponsor: who builds, how steadily."""
    a = addons[addons["platform"].isin(platforms)].copy()
    a["when"] = pd.to_datetime(a["published_date"])
    fig = go.Figure()
    for firm in [f for f in FIRM_ORDER if f in set(a["sponsor"])]:
        g = a[a["sponsor"] == firm]
        fig.add_scatter(x=g["when"], y=g["platform"], mode="markers", name=firm,
                        marker=dict(size=11, color=FIRM_COLOR[firm], line=dict(color="#fcfcfb", width=2)),
                        customdata=g[["target", "target_industry", "rationale_tags"]].fillna("").to_numpy(),
                        hovertemplate="<b>%{y}</b> acquired <b>%{customdata[0]}</b><br>%{x|%b %d, %Y}<br>"
                                      "%{customdata[1]}<br>Stated rationale: %{customdata[2]}<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(platforms)))
    fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0))
    return _layout(fig, "Add-on acquisitions by platform (hover a dot for the deal)", height or max(260, 30 * len(platforms) + 110))
