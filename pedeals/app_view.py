"""
The PE Deal Intelligence tab of the dashboard (app.py calls `render()` inside its own tab).

A separate product from the stock tabs: it reads nothing from the sidebar and keeps its state under "pe_" keys. It
runs on the saved pages by default (offline, reproducible, the collection date on screen) and can refresh one firm
from its live site.
"""
from __future__ import annotations

import html
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from . import analytics as an
from . import charts
from .collect import DEFAULT_START, PAGES_FILE, LISTINGS_FILE
from .fetch import offline_mode
from .pipeline import build_dataset
from .sources import SOURCES
from .taxonomy import RATIONALE_TAGS, TRANSACTION_TYPES

TABLE_COLUMNS = {
    "published_date": st.column_config.TextColumn("Published", help="When the firm published the announcement (not the closing date)"),
    "publisher": st.column_config.TextColumn("Publisher (PE firm)"),
    "transaction_type": st.column_config.TextColumn("Type"),
    "buyer": st.column_config.TextColumn("Buyer / investor"),
    "target": st.column_config.TextColumn("Target"),
    "target_industry": st.column_config.TextColumn("Target industry", help="Inferred from the words that describe the target's business"),
    "target_state": st.column_config.TextColumn("State", width="small"),
    "rationale_tags": st.column_config.TextColumn("Stated rationale"),
    "price_disclosure": st.column_config.TextColumn("Price"),
    "deal_status": st.column_config.TextColumn("Status"),
    "url": st.column_config.LinkColumn("Source", display_text="open"),
}


def _plotly(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config=charts.PLOTLY_CONFIG, key=key)


def _stamp() -> tuple:
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (0, 0) for p in (Path(PAGES_FILE), Path(LISTINGS_FILE)))


@st.cache_resource(show_spinner=False, max_entries=8)
def _dataset(start: str, stamp: tuple):
    return build_dataset(start=start)


def _kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f"<div class='lh-kpi' role='group' aria-label='{html.escape(label)}'><div class='lh-kpi-label'>{html.escape(label)}</div>"
                 f"<div class='lh-kpi-value'>{html.escape(value)}</div><div class='lh-kpi-sub'>{html.escape(sub)}</div></div>",
                 unsafe_allow_html=True)


def _md(text: str) -> str:
    return str(text).replace("$", "\\$")


def _refresh_controls() -> None:
    with st.expander("Refresh from the live websites"):
        st.markdown("Collection is deliberately slow: robots.txt is obeyed and each site's crawl delay (10 seconds "
                    "between pages on two of them) is respected, so one firm takes a few minutes the first time. "
                    "Pages already saved are not fetched again; only new articles are.")
        c1, c2 = st.columns([2, 1])
        firm = c1.selectbox("Firm", list(SOURCES), format_func=lambda k: SOURCES[k].firm, key="pe_refresh_firm")
        go = c2.button("Collect new pages", key="pe_refresh", disabled=offline_mode(), width="stretch",
                       help="Unavailable in offline mode" if offline_mode() else None)
        if go:
            from .collect import collect
            log = st.empty()
            lines = []

            def show(msg):
                lines.append(msg)
                log.code("\n".join(lines[-8:]))
            with st.spinner(f"Collecting {SOURCES[firm].firm} politely ..."):
                try:
                    disc = collect(start=st.session_state.get("pe_start_str", DEFAULT_START), firms=[firm], log=show)
                    st.success(f"Done: {int((disc['source'] == firm).sum())} URLs checked. The dataset below is rebuilt.")
                except Exception as exc:  # noqa: BLE001 - show the problem, keep the saved data
                    st.error(f"Collection failed ({type(exc).__name__}: {exc}); the saved pages are unchanged.")


def _filters(deals: pd.DataFrame):
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
    firms = c1.multiselect("Firms", sorted(deals["publisher"].unique()), key="pe_firms", placeholder="All firms")
    types = c2.multiselect("Transaction type", [t for t in TRANSACTION_TYPES if t in set(deals["transaction_type"])], key="pe_types",
                           placeholder="All types")
    inds = c3.multiselect("Target industry", sorted(deals["target_industry"].unique()), key="pe_inds", placeholder="All industries")
    query = c4.text_input("Search", key="pe_query", placeholder="e.g. HVAC, cybersecurity, Texas, distributor",
                          help="Every word must appear in the headline, the parties, the target's description or the rationale")
    tags = st.pills("Stated rationale", list(RATIONALE_TAGS), selection_mode="multi", key="pe_tags")
    return an.search(deals, query, firms=firms, types=types, industries=inds, tags=tags)


def _finder(view: pd.DataFrame, deals: pd.DataFrame) -> None:
    st.caption(f"{len(view)} of {len(deals)} deal announcements. Select a row for its evidence card.")
    sel = st.dataframe(view[an.DISPLAY_COLUMNS], hide_index=True, width="stretch", column_config=TABLE_COLUMNS,
                       on_select="rerun", selection_mode="single-row", key="pe_table")
    c1, c2 = st.columns(2)
    c1.download_button("Download these deals (CSV)", view.drop(columns=["body"], errors="ignore").to_csv(index=False),
                       "pe_deals.csv", "text/csv", key="pe_dl_csv", width="stretch")
    c2.download_button("Download these deals (JSON)", view.drop(columns=["body"], errors="ignore").to_json(orient="records", indent=1),
                       "pe_deals.json", "application/json", key="pe_dl_json", width="stretch")
    rows = sel.selection.rows if sel is not None and hasattr(sel, "selection") else []
    if rows:
        rec = view.iloc[rows[0]]
        st.markdown(_md(an.deal_card(rec)))
        with st.expander("Full announcement text"):
            st.markdown(_md(rec.get("body", "")))
        sim = an.similar_deals(deals, rec["deal_id"], k=5)
        if len(sim):
            st.markdown("**Similar deals** (target and rationale described in similar words)")
            st.dataframe(sim[["similarity", "published_date", "publisher", "buyer", "target", "target_industry", "url"]],
                         hide_index=True, width="stretch",
                         column_config={**TABLE_COLUMNS, "similarity": st.column_config.ProgressColumn("Similarity", format="%.2f",
                                                                                                         min_value=0, max_value=1)})


def _buy_and_build(deals: pd.DataFrame) -> None:
    bb = an.buy_and_build(deals)
    if bb.empty:
        st.info("No add-on acquisitions match the filters.")
        return
    st.markdown("Which platforms are buying, how many add-ons each announced and how fast. A short gap between add-ons "
                "suggests an integration engine worth studying; what the add-ons did to results is not in the "
                "announcements and needs operating data.")
    st.dataframe(bb.drop(columns=["platform_key"]), hide_index=True, width="stretch", column_config={
        "add_ons": st.column_config.NumberColumn("Add-on announcements"),
        "companies_acquired": st.column_config.NumberColumn("Companies acquired", help="An announcement can name several targets"),
        "median_days_between": st.column_config.NumberColumn("Median days between", format="%.0f"),
        "rationale": st.column_config.TextColumn("Stated rationale (count)", width="large")})
    top = bb[bb["add_ons"] >= 2]["platform"].tolist()[:15] or bb["platform"].tolist()[:15]
    adds = deals[deals["transaction_type"] == "Add-on acquisition"].merge(bb[["platform_key", "platform"]], on="platform_key", how="left")
    _plotly(charts.buy_and_build_timeline(adds, top), "pe_bb_timeline")


def _brief(deals: pd.DataFrame) -> None:
    st.markdown("**Decision scenario.** You run operations for a platform and are preparing the next add-on plan: which "
                "capabilities did peer sponsors' platforms in your industry buy, and how did they justify them? The "
                "brief lists the deals with their stated rationale and links, and says what still needs research.")
    adds = deals[deals["transaction_type"] == "Add-on acquisition"]
    inds = adds["target_industry"].value_counts()
    inds = [i for i in inds.index if i != "Unknown"]
    if not inds:
        st.info("No add-on acquisitions match the filters.")
        return
    c1, c2 = st.columns([1, 2])
    ind = c1.selectbox("Your platform's industry", inds, key="pe_brief_ind")
    q = c2.text_input("Your question (printed on the brief)", key="pe_brief_q",
                      value="Which capabilities are peer platforms adding, and which should we investigate?")
    a = adds[adds["target_industry"] == ind]
    left, right = st.columns([1, 1.3])
    rf = an.rationale_frequency(a)
    with left:
        if len(rf):
            _plotly(charts.horizontal_counts(rf, "tag", "deals", f"Stated rationale of {ind} add-ons", "add-ons"), "pe_brief_rf")
            st.dataframe(rf, hide_index=True, width="stretch",
                         column_config={"share_of_deals_with_rationale": st.column_config.ProgressColumn("Share", format="percent",
                                                                                                          min_value=0, max_value=1)})
    brief = an.research_brief(deals, ind, q)
    with right:
        st.markdown(_md(brief))
    st.download_button("Download the brief (Markdown)", brief, f"add_on_brief_{ind.split()[0].lower()}.md", "text/markdown",
                       key="pe_dl_brief")


def _market(deals: pd.DataFrame) -> None:
    st.caption("Counts describe the announcements these four firms published, not the PE market: firms announce "
               "selectively and small or failed deals rarely appear.")
    if deals.empty:
        st.info("No deals match the filters.")
        return
    _plotly(charts.deals_by_quarter(deals), "pe_quarter")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What each firm announced**")
        st.dataframe(an.by_firm_and_type(deals), width="stretch")
        st.markdown("**Deals by year and type**")
        st.dataframe(an.by_year(deals), width="stretch")
    with c2:
        im = an.industry_mix(deals)
        _plotly(charts.horizontal_counts(im, "target_industry", "deals", "Target industries", "deals"), "pe_ind")
    c1, c2 = st.columns(2)
    with c1:
        rf = an.rationale_frequency(deals[deals["transaction_type"] == "Add-on acquisition"])
        if len(rf):
            _plotly(charts.horizontal_counts(rf, "tag", "deals", "Why add-ons were made (stated rationale)", "add-ons"), "pe_rf")
    with c2:
        st.markdown("**What the announcements leave out**")
        st.dataframe(an.disclosure_gaps(deals), hide_index=True, width="stretch",
                     column_config={"share_missing": st.column_config.ProgressColumn("Share missing", format="percent",
                                                                                   min_value=0, max_value=1)})
        st.caption("A missing price is recorded as not stated, never as zero; the publication date is never used as a closing date.")


def _monitor(deals: pd.DataFrame) -> None:
    latest = pd.to_datetime(deals["published_date"]).max()
    default = (latest - timedelta(days=90)).date() if pd.notna(latest) else date.today() - timedelta(days=90)
    since = st.date_input("Show deals published since", value=default, key="pe_since")
    new = an.new_since(deals, since.isoformat())
    st.markdown(f"**{len(new)} deal announcements** since {since:%B %d, %Y}.")
    for r in new.itertuples():
        st.markdown(_md(f"- **{r.published_date}** · {r.publisher}: **{r.buyer}** → **{r.target}** ({r.transaction_type}; "
                        f"{r.target_industry}) · [source]({r.url})"))
    st.caption("Run `python -m pedeals collect` (or the refresh above) on a schedule and this list is the weekly digest.")


def _quality(ds) -> None:
    from .validate import evaluate
    held, dev = evaluate(ds, split="held-out"), evaluate(ds, split="development")
    st.markdown(f"**Accuracy against hand-labelled source pages.** Held-out: {held.n_labels} articles labelled after the "
                f"rules were frozen (the numbers to quote); development: {dev.n_labels} articles the rules were tuned on.")
    t = held.summary[["field", "n", "correct", "accuracy", "ci_low", "ci_high"]].merge(
        dev.summary[["field", "accuracy"]].rename(columns={"accuracy": "development"}), on="field", how="left")
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "n": "Labelled", "correct": "Correct",
        "accuracy": st.column_config.ProgressColumn("Held-out accuracy", format="percent", min_value=0, max_value=1),
        "ci_low": st.column_config.NumberColumn("95% low", format="%.2f"), "ci_high": st.column_config.NumberColumn("95% high", format="%.2f"),
        "development": st.column_config.NumberColumn("Development (in-sample)", format="%.2f")})
    if len(held.errors):
        with st.expander(f"Held-out mismatches ({len(held.errors)})"):
            st.dataframe(held.errors, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Consistency checks** (every record)")
        st.dataframe(held.consistency, hide_index=True, width="stretch")
        st.caption(f"Two builds from the same saved pages are {'identical' if held.repeatable else 'DIFFERENT'} "
                   f"(fingerprint {ds.fingerprint()}).")
    with c2:
        st.markdown("**Coverage**: what each firm's sitemap listed and what was kept")
        st.dataframe(ds.coverage.drop(columns=["hq"]), hide_index=True, width="stretch")
        if len(ds.failures):
            st.dataframe(ds.failures, hide_index=True, width="stretch")
        und = ds.records[ds.records["published_date"].fillna("") == ""]
        if len(und):
            with st.expander(f"{len(und)} articles with no publication date (left out of the period, not guessed)"):
                st.dataframe(und[["publisher", "headline", "period_note", "url"]], hide_index=True, width="stretch",
                             column_config={"url": st.column_config.LinkColumn("Source", display_text="open")})
    guide = Path(__file__).resolve().parent.parent / "docs" / "PE_DEALS.md"
    if guide.exists():
        with st.expander("Method, data dictionary and limitations"):
            st.markdown(_md(guide.read_text()))


def render() -> None:
    st.markdown("### 🤝 PE Deal Intelligence")
    st.markdown("Add-on and platform deals announced by four lower-middle-market sponsors (Tonka Bay, Goldner Hawn, "
                "Huron Capital, Align Capital Partners), read from their own news pages: who bought what, why they say "
                "they bought it, and how fast each platform builds. **Every field links to the sentence it came from.**")
    c1, c2 = st.columns([1, 3])
    start = c1.date_input("Published from", value=date.fromisoformat(DEFAULT_START), min_value=date(2005, 1, 1),
                          max_value=date.today(), key="pe_start")
    st.session_state["pe_start_str"] = start.isoformat()
    with st.spinner("Reading the saved announcements ..."):
        ds = _dataset(start.isoformat(), _stamp())
    c2.info(f"**Cached data**: pages collected {str(ds.meta.get('collected_first', ''))[:10]} to "
            f"{str(ds.meta.get('collected_last', ''))[:10]} (UTC). Announcements published {ds.start} to "
            f"{ds.meta.get('latest_published', '?')}; the live sites may have changed since.", icon="🗂️")
    _refresh_controls()
    deals = ds.unique_deals
    if deals.empty:
        st.warning("No deal announcements in this period.")
        return
    view = _filters(deals)
    k = st.columns(5)
    adds = view[view["transaction_type"] == "Add-on acquisition"]
    _kpi(k[0], "Deal announcements", f"{len(view)}", f"of {len(ds.records[ds.records['in_period']])} news items in the period")
    _kpi(k[1], "Add-on acquisitions", f"{len(adds)}", f"by {adds['platform_key'].nunique()} platforms")
    _kpi(k[2], "Platform investments", f"{int((view['transaction_type'] == 'Platform investment').sum())}",
         f"+ {int((view['transaction_type'] == 'Growth / minority investment').sum())} growth / minority")
    _kpi(k[3], "Exits", f"{int((view['transaction_type'] == 'Exit').sum())}", "sales of portfolio companies")
    disclosed = int((view["price_disclosure"] == "Disclosed").sum())
    _kpi(k[4], "Price disclosed", f"{disclosed} of {len(view)}", "the rest: not stated or withheld")

    t1, t2, t3, t4, t5, t6 = st.tabs(["🔎 Deal finder", "🏗️ Buy-and-build tracker", "🧭 Add-on research brief",
                                      "📊 Market view", "🔔 New deals", "✅ Data quality"])
    with t1:
        _finder(view, deals)
    with t2:
        _buy_and_build(view)
    with t3:
        _brief(view)
    with t4:
        _market(view)
    with t5:
        _monitor(view)
    with t6:
        _quality(ds)
    st.caption("Decision support for research, not investment advice. A stated rationale is the firm's claim, not "
               "evidence that a deal improved results.")
