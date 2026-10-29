"""
Analysis for the user's decisions: who is building what, which capabilities the add-ons are said to add, and how
fast platforms buy. Every function takes the deals table and returns a table (or Markdown) the app and the notebook
show as is.

What the evidence supports: what the publishing firms *announced* and the rationale they *stated*. It does not show
that an add-on improved earnings or returns (that needs operating results), and the firms' own news pages are an
incomplete, selective record of their deals (not every deal is announced, failed or small ones rarely are). Counts
here describe the collected announcements, not the PE market.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .taxonomy import RATIONALE_TAGS

DISPLAY_COLUMNS = ["published_date", "publisher", "transaction_type", "buyer", "target", "target_industry",
                   "target_state", "rationale_tags", "price_disclosure", "deal_status", "url"]


def _split(series: pd.Series) -> pd.Series:
    return series.fillna("").map(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])


# ----------------------------------------------------------------------------- summaries
def by_firm_and_type(deals: pd.DataFrame) -> pd.DataFrame:
    if deals.empty:
        return pd.DataFrame()
    t = deals.pivot_table(index="publisher", columns="transaction_type", values="deal_id", aggfunc="count", fill_value=0)
    t["Total"] = t.sum(axis=1)
    return t.sort_values("Total", ascending=False)


def by_year(deals: pd.DataFrame) -> pd.DataFrame:
    if deals.empty:
        return pd.DataFrame()
    d = deals.assign(year=deals["published_date"].str[:4])
    return d.pivot_table(index="year", columns="transaction_type", values="deal_id", aggfunc="count", fill_value=0)


def industry_mix(deals: pd.DataFrame) -> pd.DataFrame:
    if deals.empty:
        return pd.DataFrame(columns=["target_industry", "deals", "add_ons", "share"])
    g = deals.groupby("target_industry").agg(deals=("deal_id", "count"),
                                             add_ons=("transaction_type", lambda s: int((s == "Add-on acquisition").sum())))
    g["share"] = g["deals"] / g["deals"].sum()
    return g.sort_values("deals", ascending=False).reset_index()


def rationale_frequency(deals: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    """How often each stated rationale appears (share of the deals that state any rationale)."""
    d = deals.assign(tag=_split(deals["rationale_tags"])).explode("tag")
    d = d[d["tag"].notna() & (d["tag"] != "")]
    stated = deals[deals["rationale_tags"].fillna("") != ""]
    if d.empty:
        return pd.DataFrame(columns=["tag", "deals", "share_of_deals_with_rationale"])
    if by is None:
        g = d.groupby("tag")["deal_id"].nunique().rename("deals").reset_index()
        g["share_of_deals_with_rationale"] = g["deals"] / max(len(stated), 1)
        return g.sort_values("deals", ascending=False).reset_index(drop=True)
    g = d.groupby([by, "tag"])["deal_id"].nunique().unstack(fill_value=0)
    return g.reindex(columns=[c for c in RATIONALE_TAGS if c in g.columns])


def disclosure_gaps(deals: pd.DataFrame) -> pd.DataFrame:
    """How much each important field is missing: the honest size of what the announcements leave out."""
    n = max(len(deals), 1)
    checks = {
        "Transaction price": (deals["price_disclosure"] != "Disclosed").sum(),
        "Closing date": (deals["closing_date"].isin(["", "Not stated"])).sum(),
        "Ownership stake": (deals["ownership"] == "Unknown").sum(),
        "Target location (state)": (deals["target_state"].fillna("") == "").sum(),
        "Target description": (deals["target_description"].fillna("") == "").sum(),
        "Stated rationale": (deals["rationale"].fillna("") == "").sum(),
        "Deal status (completed / pending)": (deals["deal_status"] == "Unknown").sum(),
    }
    return pd.DataFrame([{"field": k, "missing": int(v), "share_missing": v / n} for k, v in checks.items()])


# ----------------------------------------------------------------------------- buy-and-build tracker
def buy_and_build(deals: pd.DataFrame) -> pd.DataFrame:
    """One row per platform that made add-ons in the collected announcements: how many announcements and companies,
    how fast (median days between announcements), into what industries and states, and the rationale stated."""
    a = deals[deals["transaction_type"] == "Add-on acquisition"].copy()
    if a.empty:
        return pd.DataFrame(columns=["platform", "sponsor", "add_ons", "companies_acquired", "first", "last", "median_days_between",
                                     "targets", "target_industries", "target_states", "rationale"])
    rows = []
    key = "platform_key" if "platform_key" in a.columns else "buyer"
    for (_, sponsor), g in a.groupby([key, "sponsor"]):
        g = g.sort_values("published_date")
        from .pipeline import norm_name
        names = {}
        for b in g["buyer"].iloc[::-1]:                   # the latest name first ("Marco Sealing Solutions / Marco Rubber & Plastics")
            names.setdefault(norm_name(b), b)
        platform = " / ".join(names.values())
        ds = [date.fromisoformat(x) for x in g["published_date"] if x]
        gaps = [(b - c).days for c, b in zip(ds, ds[1:])]
        tags = Counter(t for ts in _split(g["rationale_tags"]) for t in ts)
        rows.append({"platform": platform, "platform_key": g[key].iloc[0], "sponsor": sponsor, "add_ons": len(g),
                     "companies_acquired": int(g["n_targets"].clip(lower=1).sum()), "first": g["published_date"].iloc[0],
                     "last": g["published_date"].iloc[-1],
                     "median_days_between": float(np.median(gaps)) if gaps else math.nan,
                     "targets": "; ".join(g["target"]),
                     "target_industries": "; ".join(sorted(set(g["target_industry"]) - {"Unknown"})),
                     "target_states": "; ".join(sorted(set(x for x in g["target_state"] if x))),
                     "rationale": "; ".join(f"{t} ({n})" for t, n in tags.most_common())})
    return pd.DataFrame(rows).sort_values(["add_ons", "last"], ascending=[False, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------- finding records
def search(deals: pd.DataFrame, query: str = "", firms=None, types=None, industries=None, tags=None, states=None,
           since: Optional[str] = None, until: Optional[str] = None) -> pd.DataFrame:
    d = deals
    if firms:
        d = d[d["publisher"].isin(firms)]
    if types:
        d = d[d["transaction_type"].isin(types)]
    if industries:
        d = d[d["target_industry"].isin(industries)]
    if states:
        d = d[d["target_state"].isin(states)]
    if tags:
        d = d[_split(d["rationale_tags"]).map(lambda ts: any(t in ts for t in tags))]
    if since:
        d = d[d["published_date"] >= str(since)]
    if until:
        d = d[d["published_date"] <= str(until)]
    q = (query or "").strip()
    if q:
        words = [w for w in re.split(r"\s+", q.lower()) if w]
        hay = (d["headline"].fillna("") + " " + d["buyer"].fillna("") + " " + d["target"].fillna("") + " " +
               d["target_description"].fillna("") + " " + d["rationale"].fillna("") + " " +
               d["customer_markets"].fillna("")).str.lower()
        d = d[hay.map(lambda h: all(w in h for w in words))]
    return d.sort_values("published_date", ascending=False).reset_index(drop=True)


_TOKEN = re.compile(r"[a-z][a-z\-]{2,}")
_STOP = set("the and for with its from that this into has have are was were will been their our over more than also "
            "company companies business businesses based inc llc which who acquisition acquired acquires partner "
            "partners capital equity announced today leading provider".split())


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def similar_deals(deals: pd.DataFrame, deal_id: str, k: int = 5) -> pd.DataFrame:
    """The k deals whose target description and rationale read most alike (TF-IDF cosine similarity): 'who else
    bought something like this?'"""
    if deals.empty or deal_id not in set(deals["deal_id"]):
        return deals.iloc[0:0]
    docs = [_tokens(f"{a} {b} {c}") for a, b, c in zip(deals["target_description"], deals["rationale"], deals["target_industry"])]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    if not vocab:
        return deals.iloc[0:0]
    m = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for t, c in Counter(d).items():
            m[i, vocab[t]] = c
    idf = np.log((1 + len(docs)) / (1 + (m > 0).sum(axis=0))) + 1
    m = m * idf
    norms = np.linalg.norm(m, axis=1)
    norms[norms == 0] = 1
    m = m / norms[:, None]
    i = int(np.flatnonzero(deals["deal_id"].to_numpy() == deal_id)[0])
    sims = m @ m[i]
    out = deals.assign(similarity=sims).drop(index=deals.index[i])
    return out.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)


def new_since(deals: pd.DataFrame, since: str) -> pd.DataFrame:
    return deals[deals["published_date"] >= str(since)].sort_values("published_date", ascending=False)


# ----------------------------------------------------------------------------- Markdown outputs
def md_table(df: pd.DataFrame, index: bool = False, pct: tuple = ()) -> str:
    """A Markdown table without the optional `tabulate` dependency; columns in `pct` print as percentages."""
    if df is None or df.empty:
        return "none"
    d = df.reset_index() if index else df

    def cell(c, x):
        if c in pct and isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
            return f"{x:.0%}"
        if isinstance(x, float):
            return "—" if math.isnan(x) else f"{x:,.0f}" if abs(x) >= 100 else f"{x:.1f}"
        return str(x).replace("|", "/")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(cell(c, x) for c, x in zip(d.columns, row)) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _short(text: str, n: int = 260) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def _v(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() == "" else str(x)


def deal_card(rec) -> str:
    """One deal as an evidence card: every value next to the words it was read from."""
    r = dict(rec)
    value = f"${r['deal_value_usd']:,.0f}" if r.get("deal_value_usd") and not math.isnan(r["deal_value_usd"]) else r.get("price_disclosure")
    rows = [
        ("Publisher (PE firm)", r.get("publisher"), f"source page: {r.get('url')}"),
        ("Published", r.get("published_date"), f"{r.get('published_method')}: {r.get('published_evidence')}"),
        ("Transaction type", r.get("transaction_type"), r.get("type_evidence")),
        ("Buyer / investor", r.get("buyer"), r.get("parties_evidence")),
        ("Buyer's relationship to the publisher", r.get("buyer_role"), r.get("type_evidence") if "company" in str(r.get("buyer_role")) else ""),
        ("Target", r.get("target"), r.get("parties_evidence")),
        ("Target's business", r.get("target_description"), r.get("description_source")),
        ("Target industry (inferred)", r.get("target_industry"), f"stems matched: {r.get('industry_evidence')}"),
        ("Target's customers / end markets", r.get("customer_markets"), "from 'serving', 'used in', 'customers in' phrases"),
        ("Target location", f"{_v(r.get('target_location'))} ({_v(r.get('target_state'))})", ""),
        ("Stated rationale", r.get("rationale"), f"tags: {r.get('rationale_tags')}"),
        ("Deal status", r.get("deal_status"), r.get("status_evidence")),
        ("Closing date", r.get("closing_date"), r.get("closing_evidence") or "the publication date is not a closing date"),
        ("Price", value, r.get("price_evidence")),
        ("Ownership", r.get("ownership"), r.get("ownership_evidence")),
    ]
    lines = [f"#### {_v(r.get('headline'))}", "", "| Field | Value | Evidence |", "|---|---|---|"]
    for f, v, e in rows:
        lines.append(f"| {f} | {_v(v).replace('|', '/')} | {_v(e).replace('|', '/')[:400]} |")
    if r.get("review_flags"):
        lines += ["", f"*Review flags:* {r['review_flags']}"]
    if r.get("duplicate_of"):
        lines += ["", f"*Duplicate of deal {r['duplicate_of']}*"]
    return "\n".join(lines)


def dataset_brief(ds) -> str:
    """The dataset's one-page Markdown summary (exported with the data)."""
    u = ds.unique_deals
    lines = [f"# PE deal announcements: brief", "", f"> {ds.note}", "",
             f"Period: published {ds.start} to {ds.meta.get('latest_published', '?')} · pages collected "
             f"{str(ds.meta.get('collected_first', ''))[:10]} to {str(ds.meta.get('collected_last', ''))[:10]} · "
             f"built {ds.built_at[:10]} · fingerprint `{ds.fingerprint()}`", "",
             f"{len(ds.records)} articles from {ds.records['source'].nunique()} firms; **{len(ds.deals)} deal "
             f"announcements** in the period ({len(u)} unique deals), {int((ds.records['in_period'] & ~ds.records['is_deal']).sum())} "
             "other news items (people, funds, awards) set aside.", "",
             "## Deals by firm and type", "", md_table(by_firm_and_type(u), index=True), "",
             "## Stated rationale of add-on acquisitions", ""]
    adds = u[u["transaction_type"] == "Add-on acquisition"]
    rf = rationale_frequency(adds)
    lines += [md_table(rf, pct=("share_of_deals_with_rationale",)), "",
              "## Platforms building fastest", ""]
    bb = buy_and_build(u)
    lines += [md_table(bb[["platform", "sponsor", "add_ons", "companies_acquired", "first", "last", "median_days_between",
                              "target_industries"]].head(10)), "",
              "## What the announcements leave out", "", md_table(disclosure_gaps(u), pct=("share_missing",)), "",
              "These are announcements the firms chose to publish on their own sites: an incomplete and selective record, "
              "not a measure of the PE market. A stated rationale is the firm's claim, not evidence that the deal "
              "improved results."]
    return "\n".join(lines) + "\n"


def research_brief(deals: pd.DataFrame, industry: str, question: str = "") -> str:
    """The decision scenario: an operations lead whose platform is in `industry` asks what peer platforms bought and
    why, to shortlist capability gaps worth investigating. Returns Markdown with the evidence links."""
    a = deals[(deals["transaction_type"] == "Add-on acquisition") & (deals["target_industry"] == industry)]
    rf = rationale_frequency(a)
    lines = [f"### Add-on research brief: {industry}", ""]
    if question:
        lines += [f"*Question:* {question}", ""]
    if a.empty:
        return "\n".join(lines + ["No add-on acquisitions in this industry among the collected announcements."])
    lines += [f"{len(a)} add-on acquisitions by {a['buyer'].nunique()} platforms of {a['sponsor'].nunique()} sponsors "
              f"({a['published_date'].min()} to {a['published_date'].max()}).", "",
              "**What the buyers said the add-ons add** (share of add-ons that state any rationale):", ""]
    lines += [f"- {r.tag}: {r.deals} ({r.share_of_deals_with_rationale:.0%})" for r in rf.itertuples()]
    lines += ["", "**The deals, with the evidence:**", ""]
    for r in a.sort_values("published_date", ascending=False).itertuples():
        lines.append(f"- {r.published_date} · **{r.buyer}** ({r.sponsor}) acquired **{r.target}**"
                     f"{f' ({r.target_state})' if r.target_state else ''}: {_short(_v(r.rationale))} [source]({r.url})")
    lines += ["", "**Supported by this evidence:** which capabilities peer platforms chose to buy and how they "
              "described them. **Needs further research:** whether those add-ons improved margins, growth or "
              "returns (operating results, management interviews, lender data), deals the firms did not announce, "
              "and the prices paid (rarely disclosed)."]
    return "\n".join(lines)
