"""
The PE deal-sourcing tab of the dashboard (app.py calls `render()` inside its own tab).

A separate product from the stock tabs: it reads nothing from the sidebar and keeps its state under "pe_" keys. Its
user is a deal team at a lower-middle-market sponsor sourcing in five sectors. It runs on the saved pages by default
(offline, reproducible, the collection dates on screen) and can refresh one firm from its live site.
"""
from __future__ import annotations

import html
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from . import analytics as an
from . import charts
from . import sourcing as so
from .collect import DEFAULT_START, PAGES_FILE, LISTINGS_FILE
from .fetch import offline_mode
from .pipeline import build_dataset
from .sources import SOURCES
from .taxonomy import FOCUS_SECTORS, NICHES, OUTSIDE_FOCUS, TRANSACTION_TYPES

VIEW_START = "2022-01-01"          # sourcing looks at recent activity; the ledger uses every collected year
TABLE_COLUMNS = {
    "published_date": st.column_config.TextColumn("Published", help="When the firm published the announcement (not the closing date)"),
    "publisher": st.column_config.TextColumn("Publisher (PE firm)"),
    "transaction_type": st.column_config.TextColumn("Type"),
    "buyer": st.column_config.TextColumn("Buyer / platform"),
    "target": st.column_config.TextColumn("Target"),
    "focus_sector": st.column_config.TextColumn("Sector"),
    "niche": st.column_config.TextColumn("Niche"),
    "target_state": st.column_config.TextColumn("State", width="small"),
    "rationale_tags": st.column_config.TextColumn("Stated rationale"),
    "url": st.column_config.LinkColumn("Source", display_text="open"),
}
FINDER_COLUMNS = ["published_date", "publisher", "transaction_type", "buyer", "target", "focus_sector", "niche",
                  "target_state", "rationale_tags", "url"]


def _plotly(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", config=charts.PLOTLY_CONFIG, key=key)


def _stamp() -> tuple:
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else (0, 0) for p in (Path(PAGES_FILE), Path(LISTINGS_FILE)))


@st.cache_resource(show_spinner=False, max_entries=4)
def _dataset(stamp: tuple):
    ds = build_dataset(start=DEFAULT_START)
    return ds, so.ledger(ds.records)


def _kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f"<div class='lh-kpi' role='group' aria-label='{html.escape(label)}'><div class='lh-kpi-label'>{html.escape(label)}</div>"
                 f"<div class='lh-kpi-value'>{html.escape(value)}</div><div class='lh-kpi-sub'>{html.escape(sub)}</div></div>",
                 unsafe_allow_html=True)


def _md(text: str) -> str:
    return str(text).replace("$", "\\$")


def _refresh_controls() -> None:
    with st.expander("Refresh from the live websites"):
        st.markdown("Collection is deliberately slow: robots.txt is obeyed and each site's crawl delay (10 seconds "
                    "between pages on some) is respected. Pages already saved are not fetched again; only new articles are.")
        c1, c2 = st.columns([2, 1])
        firm = c1.selectbox("Firm", list(SOURCES), format_func=lambda k: SOURCES[k].firm, key="pe_refresh_firm")
        go = c2.button("Collect new pages", key="pe_refresh", disabled=offline_mode(), width="stretch",
                       help="Unavailable in offline mode" if offline_mode() else None)
        if go:
            from .collect import collect
            log, lines = st.empty(), []

            def show(msg):
                lines.append(msg)
                log.code("\n".join(lines[-8:]))
            with st.spinner(f"Collecting {SOURCES[firm].firm} politely ..."):
                try:
                    disc = collect(start=DEFAULT_START, firms=[firm], log=show)
                    st.success(f"Done: {int((disc['source'] == firm).sum())} URLs checked. The data below is rebuilt.")
                except Exception as exc:  # noqa: BLE001 - show the problem, keep the saved data
                    st.error(f"Collection failed ({type(exc).__name__}: {exc}); the saved pages are unchanged.")


def _filters(deals: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns([1.5, 1.2, 1, 0.9, 1.3])
    sectors = c1.multiselect("Sectors", FOCUS_SECTORS + [OUTSIDE_FOCUS], default=FOCUS_SECTORS, key="pe_sectors",
                             help="The five focus sectors; add 'Outside focus' to see every other deal")
    firms = c2.multiselect("Sponsors", sorted(deals["publisher"].unique()), key="pe_firms", placeholder="All sponsors")
    since = c3.date_input("Deals since", value=date.fromisoformat(VIEW_START), min_value=date.fromisoformat(DEFAULT_START),
                          max_value=date.today(), key="pe_since_view")
    states = c4.multiselect("Target states", sorted(x for x in deals["target_state"].unique() if x), key="pe_states",
                            placeholder="All")
    query = c5.text_input("Search", key="pe_query", placeholder="e.g. HVAC, MSP, calibration, Texas")
    view = deals[deals["focus_sector"].isin(sectors or FOCUS_SECTORS)]
    view = an.search(view, query, firms=firms, states=states, since=since.isoformat())
    st.caption("⚠️ Sector and niche come from keyword rules on each target's description. On the latest held-out "
               "articles they matched the labels about half the time (Data quality tab), so treat the sector view as a "
               "first cut: open the description before relying on a company's sector, and add 'Outside focus' to "
               "see deals that may have been misplaced. Buyers, targets, deal types and dates are far more reliable.")
    return view, (sectors or FOCUS_SECTORS), since


def _radar(view: pd.DataFrame) -> None:
    st.markdown("**Which niches are being rolled up, by how many platforms, and is the pace rising?** "
                "Momentum compares the last twelve months of add-ons with the twelve before.")
    cons = so.consolidation(view, focus_only=False)
    if cons.empty:
        st.info("No add-on acquisitions match the filters.")
        return
    st.dataframe(cons, hide_index=True, width="stretch", column_config={
        "focus_sector": "Sector", "niche": "Niche", "add_ons": "Add-ons", "platforms": "Platforms", "sponsors": "Sponsors",
        "last_12m": "Last 12 months", "prior_12m": "Prior 12 months",
        "momentum": st.column_config.NumberColumn("Momentum", help="Last 12 months minus the 12 before", format="%+d"),
        "latest": "Latest", "target_states": "Where targets were (state, count)"})
    c1, c2 = st.columns(2)
    with c1:
        top = cons.head(15).assign(label=lambda t: t["niche"])
        _plotly(charts.horizontal_counts(top, "label", "add_ons", "Add-ons by niche", "add-ons"), "pe_niches")
    with c2:
        _plotly(charts.deals_by_quarter(view[view["transaction_type"] == "Add-on acquisition"]), "pe_quarter")


def _exit_watch(led: pd.DataFrame, sectors) -> None:
    st.markdown("**Sponsor-owned companies likely to be sold soon.** Lower-middle-market sponsors usually sell after "
                "four to six years; these companies have been held at least that long and no sale has been announced. "
                "They are candidates to buy from a sponsor (a secondary buyout) or to approach as an add-on seller.")
    exited = led[(led["status"] == "Exited") & (led["years_basis"] == "from entry")]["years_held"].dropna()
    c1, c2 = st.columns([1, 2])
    min_years = c1.slider("Held at least (years)", 2.0, 10.0, so.TYPICAL_HOLD_YEARS, 0.5, key="pe_min_years")
    if len(exited) >= 3:
        c2.info(f"In this data, the {len(exited)} exits with a dated entry were held a median of "
                f"{exited.median():.1f} years (range {exited.min():.1f}-{exited.max():.1f}).", icon="⏳")
    w = so.exit_watch(led, min_years, sectors)
    if w.empty:
        st.info("No sponsor-owned company in these sectors has been held that long with no exit announced.")
        return
    st.dataframe(w[["company", "sponsor", "focus_sector", "niche", "state", "years_held", "years_basis", "entry_date",
                    "entry_basis", "add_ons", "last_add_on", "signal", "entry_url"]],
                 hide_index=True, width="stretch", column_config={
                     "years_held": st.column_config.NumberColumn("Years held", format="%.1f"),
                     "years_basis": "Counted", "entry_date": "Entry", "entry_basis": "Entry from",
                     "add_ons": "Add-ons", "last_add_on": "Last add-on", "focus_sector": "Sector",
                     "entry_url": st.column_config.LinkColumn("Evidence", display_text="open")})
    _plotly(charts.years_held(w, so.TYPICAL_HOLD_YEARS), "pe_years")
    st.caption("'No exit announced' means no sale appears in the sponsor's own news; check before calling. Entry dates "
               "come from the investment announcement or a sentence such as 'ACP acquired StenTech in May of 2022'; "
               "light bars count from the first mention only, so the true hold is longer.")
    with st.expander("Full portfolio ledger (every holding found, including exits)"):
        st.dataframe(led.drop(columns=["company_key", "evidence_urls", "description"]), hide_index=True, width="stretch",
                     column_config={"years_held": st.column_config.NumberColumn("Years held", format="%.1f"),
                                    "entry_url": st.column_config.LinkColumn("Entry evidence", display_text="open"),
                                    "exit_url": st.column_config.LinkColumn("Exit evidence", display_text="open")})


def _niche_picker(view: pd.DataFrame, key: str):
    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    sector = c1.selectbox("Sector", FOCUS_SECTORS, key=f"{key}_sector")
    present = [n for n in view.loc[view["focus_sector"] == sector, "niche"].value_counts().index if n]
    niche = c2.selectbox("Niche", ["All niches"] + present, key=f"{key}_niche")
    states = c3.multiselect("Target states", sorted(x for x in view["target_state"].unique() if x), key=f"{key}_states",
                            placeholder="Anywhere")
    return sector, ("" if niche == "All niches" else niche), states


def _competitors(view: pd.DataFrame, led: pd.DataFrame) -> None:
    st.markdown("**Who would you bid against?** The platforms buying in a niche (and region), how often, and how long "
                "their sponsors have owned them.")
    sector, niche, states = _niche_picker(view, "pe_comp")
    comp = so.competing_buyers(view, led, [niche] if niche else None, [sector], states)
    if comp.empty:
        st.info("No add-on buyers match.")
        return
    st.dataframe(comp, hide_index=True, width="stretch", column_config={
        "add_ons_here": "Add-ons here", "last_add_on": "Last add-on", "target_states": "Target states",
        "platform_status": "Platform", "years_held": st.column_config.NumberColumn("Years owned", format="%.1f")})
    adds = view[(view["transaction_type"] == "Add-on acquisition") & (view["focus_sector"] == sector)]
    if niche:
        adds = adds[adds["niche"] == niche]
    adds = adds.assign(platform=adds["buyer"])
    _plotly(charts.platform_timeline(adds, comp["platform"].head(15).tolist()), "pe_comp_timeline")


def _brief(view: pd.DataFrame, led: pd.DataFrame) -> None:
    st.markdown("**The weekly sourcing memo** for a sector or niche: who is consolidating it, the companies they bought "
                "(the profile to look for), the sponsor-owned companies likely to trade, and what still needs research.")
    sector, niche, states = _niche_picker(view, "pe_brief")
    min_years = st.slider("Exit watch: held at least (years)", 2.0, 10.0, so.TYPICAL_HOLD_YEARS, 0.5, key="pe_brief_years")
    brief = so.sourcing_brief(view, led, sector, niche, states, min_years)
    st.markdown(_md(brief))
    st.download_button("Download the brief (Markdown)", brief, f"sourcing_brief_{sector.split()[0].lower()}.md", "text/markdown",
                       key="pe_dl_brief")


def _finder(view: pd.DataFrame, deals: pd.DataFrame) -> None:
    types = st.multiselect("Transaction type", [t for t in TRANSACTION_TYPES if t in set(view["transaction_type"])], key="pe_types",
                           placeholder="All types")
    v = view[view["transaction_type"].isin(types)] if types else view
    st.caption(f"{len(v)} deal announcements. Select a row for its evidence card.")
    sel = st.dataframe(v[FINDER_COLUMNS], hide_index=True, width="stretch", column_config=TABLE_COLUMNS,
                       on_select="rerun", selection_mode="single-row", key="pe_table")
    c1, c2 = st.columns(2)
    c1.download_button("Download these deals (CSV)", v.drop(columns=["body"], errors="ignore").to_csv(index=False),
                       "pe_deals.csv", "text/csv", key="pe_dl_csv", width="stretch")
    c2.download_button("Download these deals (JSON)", v.drop(columns=["body"], errors="ignore").to_json(orient="records", indent=1),
                       "pe_deals.json", "application/json", key="pe_dl_json", width="stretch")
    rows = sel.selection.rows if sel is not None and hasattr(sel, "selection") else []
    if rows:
        rec = v.iloc[rows[0]]
        st.markdown(_md(an.deal_card(rec)))
        with st.expander("Full announcement text"):
            st.markdown(_md(rec.get("body", "")))
        sim = an.similar_deals(deals, rec["deal_id"], k=5)
        if len(sim):
            st.markdown("**Similar deals** (target and rationale described in similar words): more targets like this one")
            st.dataframe(sim[["similarity", "published_date", "publisher", "buyer", "target", "niche", "target_state", "url"]],
                         hide_index=True, width="stretch",
                         column_config={**TABLE_COLUMNS, "similarity": st.column_config.ProgressColumn("Similarity", format="%.2f",
                                                                                                         min_value=0, max_value=1)})


def _monitor(view: pd.DataFrame) -> None:
    latest = pd.to_datetime(view["published_date"]).max()
    default = (latest - timedelta(days=90)).date() if pd.notna(latest) else date.today() - timedelta(days=90)
    since = st.date_input("Show deals published since", value=default, key="pe_since")
    new = an.new_since(view, since.isoformat())
    st.markdown(f"**{len(new)} deal announcements** since {since:%B %d, %Y}.")
    for r in new.itertuples():
        st.markdown(_md(f"- **{r.published_date}** · {r.publisher}: **{r.buyer}** → **{r.target}** ({r.transaction_type}; "
                        f"{r.niche or r.focus_sector}{', ' + r.target_state if r.target_state else ''}) · [source]({r.url})"))
    st.caption("Run `python -m pedeals collect` (or the refresh above) on a schedule and this list is the weekly digest.")


def _quality(ds) -> None:
    from .validate import QUOTE_SPLIT, SPLITS, evaluate, load_labels
    labels = load_labels()
    reps = {sp: evaluate(ds, labels, sp) for sp in SPLITS if (labels["split"] == sp).any()}
    held = reps[QUOTE_SPLIT] if QUOTE_SPLIT in reps else next(iter(reps.values()))
    st.markdown(f"**Accuracy against labelled source pages.** {held.split}: {held.n_labels} articles labelled after the "
                "current rules were frozen in git (the numbers to quote). Earlier rounds are shown for comparison; rules "
                "were changed after reading their mismatches, so their current scores are in-sample. "
                "Details: `data/pe_deals/validation/results.md`.")
    t = held.summary[["field", "n", "correct", "accuracy", "ci_low", "ci_high"]]
    for sp, rep in reps.items():
        if sp != held.split:
            t = t.merge(rep.summary[["field", "accuracy"]].rename(columns={"accuracy": sp}), on="field", how="left")
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "n": "Labelled", "correct": "Correct",
        "accuracy": st.column_config.ProgressColumn(f"{held.split} accuracy", format="percent", min_value=0, max_value=1),
        "ci_low": st.column_config.NumberColumn("95% low", format="%.2f"), "ci_high": st.column_config.NumberColumn("95% high", format="%.2f"),
        **{sp: st.column_config.NumberColumn(f"{sp} (now)", format="%.2f") for sp in reps if sp != held.split}})
    if len(held.errors):
        with st.expander(f"{held.split} mismatches ({len(held.errors)})"):
            st.dataframe(held.errors, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Consistency checks** (every record)")
        st.dataframe(held.consistency, hide_index=True, width="stretch")
        st.caption(f"Two builds from the same saved pages are {'identical' if held.repeatable else 'DIFFERENT'} "
                   f"(fingerprint {ds.fingerprint()}).")
    with c2:
        st.markdown("**Coverage**: what each sponsor's sitemap listed and what was kept")
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
    st.markdown("### 🎯 PE Deal Sourcing")
    st.markdown(f"For a deal team at a lower-middle-market sponsor sourcing in **{', '.join(FOCUS_SECTORS[:-1])} and "
                f"{FOCUS_SECTORS[-1]}**. Read from the announcements of {len(SOURCES)} buy-and-build sponsors: which niches "
                "are being consolidated, which sponsor-owned companies are due to trade, and who you would bid against. "
                "**Every record links to the sentence it came from.**")
    with st.spinner("Reading the saved announcements ..."):
        ds, led = _dataset(_stamp())
    st.info(f"**Cached data**: pages collected {str(ds.meta.get('collected_first', ''))[:10]} to "
            f"{str(ds.meta.get('collected_last', ''))[:10]} (UTC); announcements published {ds.start} to "
            f"{ds.meta.get('latest_published', '?')}. The live sites may have changed since.", icon="🗂️")
    _refresh_controls()
    deals = ds.unique_deals
    if deals.empty:
        st.warning("No deal announcements collected.")
        return
    view, sectors, since = _filters(deals)
    adds = view[view["transaction_type"] == "Add-on acquisition"]
    cons = so.consolidation(view, focus_only=False)
    watch = so.exit_watch(led, so.TYPICAL_HOLD_YEARS, sectors)
    k = st.columns(5)
    _kpi(k[0], "Add-on acquisitions", f"{len(adds)}", f"since {since:%b %Y} in the chosen sectors")
    _kpi(k[1], "Active platforms", f"{adds['platform_key'].nunique()}", f"of {adds['sponsor'].nunique()} sponsors")
    _kpi(k[2], "Niches being rolled up", f"{int((cons['platforms'] >= 2).sum()) if len(cons) else 0}", "two or more platforms buying")
    _kpi(k[3], "Exit watch", f"{len(watch)}", f"held {so.TYPICAL_HOLD_YEARS:g}+ years, no exit announced")
    recent = view[pd.to_datetime(view["published_date"]) >= pd.Timestamp(date.today() - timedelta(days=90))]
    _kpi(k[4], "New in 90 days", f"{len(recent)}", "deal announcements")

    tabs = st.tabs(["🎯 Sourcing radar", "⏳ Exit watch", "🏁 Competing buyers", "🧭 Sourcing brief", "🔎 Deal finder",
                    "🔔 New deals", "✅ Data quality"])
    with tabs[0]:
        _radar(view)
    with tabs[1]:
        _exit_watch(led, sectors)
    with tabs[2]:
        _competitors(view, led)
    with tabs[3]:
        _brief(view, led)
    with tabs[4]:
        _finder(view, deals)
    with tabs[5]:
        _monitor(view)
    with tabs[6]:
        _quality(ds)
    st.caption("Decision support for sourcing research, not investment advice. Announcements are what sponsors chose to "
               "publish: verify a company's ownership and status before outreach.")
