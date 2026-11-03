"""Quality and Method: how the deal data was made, how accurate it measured, and what it leaves out.

Accuracy comes from the saved validation report (`data/pe/validation.json`); everything about missing
or flagged values is computed from the dataset the explorer actually serves, so it cannot drift.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from pe.pages.explorer import PROJECT_ROOT, load_explorer_dataset

VALIDATION_PATH = PROJECT_ROOT / "data" / "pe" / "validation.json"

FIELD_LABELS = {
    "is_deal": ("Deal vs other news", "Which articles appear"),
    "news_category": ("News category", "Not shown"),
    "published_date": ("Publication date", "Publication date"),
    "transaction_type": ("Transaction type", "Type"),
    "buyer": ("Buyer", "Buyer / investor"),
    "target": ("Target", "Target"),
    "target_industry": ("Target industry", "Industry"),
    "target_state": ("Target state", "Geography"),
    "deal_status": ("Deal status", "Status in the deal card"),
    "price_disclosure": ("Price disclosure", "Disclosed value"),
}

# Extraction flags in `review_notes`, grouped into what a researcher should do about them.
FLAG_MEANINGS = [
    ("several targets", "Several targets in one announcement", "Check that the right company is shown as the target."),
    ("industry ambiguous", "Industry ambiguous", "The description fits more than one industry; read the excerpt."),
    ("buyer not found", "Buyer not named", "Often an exit where the release does not name the acquirer."),
    ("relationship inferred", "Relationship inferred", "The platform link comes from another announcement."),
    ("date known to the month only", "Date known to the month only", "Shown as the 1st of that month."),
    ("date from news listing page", "Date from the firm's news list", "The page itself carries no date."),
    ("date from dateline", "Date from the release's dateline", "Usually reliable."),
    ("date from first date in text", "Date from the first date in the text", "Check the source page if timing matters."),
]


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; well-behaved at 0% and 100% unlike the normal approximation."""
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def load_validation(path: str | Path = VALIDATION_PATH) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def accuracy_table(validation: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for f in validation["fields"]:
        label, explorer_field = FIELD_LABELS.get(f["field"], (f["field"], ""))
        lo, hi = wilson_interval(f["held_out_correct"], f["held_out_total"])
        rows.append({
            "Field": label,
            "Shown in the explorer as": explorer_field,
            "Held-out: correct": f"{f['held_out_correct']} / {f['held_out_total']}",
            "Held-out accuracy (95% interval)": f"{f['held_out_correct'] / f['held_out_total']:.0%} ({lo:.0%}–{hi:.0%})",
            "Development (in-sample)": f"{f['dev_correct'] / f['dev_total']:.0%} ({f['dev_correct']}/{f['dev_total']})",
        })
    return pd.DataFrame(rows)


def later_round_table(validation: dict[str, Any]) -> pd.DataFrame:
    later = validation["later_independent_round"]["fields"]
    rows = []
    for f in validation["fields"]:
        other = later.get(f["field"])
        if not other:
            continue
        rows.append({
            "Field": FIELD_LABELS.get(f["field"], (f["field"],))[0],
            "Shipped build, held-out": f"{f['held_out_correct'] / f['held_out_total']:.0%}",
            "Newer rules, independent labellers": f"{other['correct'] / other['total']:.0%} ({other['correct']}/{other['total']})",
        })
    return pd.DataFrame(rows)


def unknown_table(data: pd.DataFrame) -> pd.DataFrame:
    def missing(col: str) -> pd.Series:
        series = data[col]
        if col in {"rationale_tags", "source_url"}:
            return series.map(lambda v: not (isinstance(v, list) and v))
        return series.isna() | series.astype("string").str.strip().eq("")

    fields = [("buyer_or_investor", "Buyer / investor"), ("target_company", "Target"),
              ("transaction_type", "Transaction type"), ("transaction_status", "Status (completed / pending)"),
              ("target_industry", "Target industry"), ("target_geography", "Geography (target's US state)"),
              ("rationale_tags", "Rationale"), ("price", "Disclosed value"), ("evidence_excerpt", "Supporting excerpt"),
              ("source_url", "Source link")]
    total = len(data)
    rows = []
    for col, label in fields:
        n = int(missing(col).sum())
        rows.append({"Field": label, "Unknown": n, "Share unknown": f"{n / total:.0%}" if total else "–"})
    return pd.DataFrame(rows)


def flag_table(data: pd.DataFrame) -> pd.DataFrame:
    notes = data["review_notes"].fillna("").astype(str).str.casefold()
    rows = []
    for key, label, meaning in FLAG_MEANINGS:
        n = int(notes.str.contains(key, regex=False).sum())
        if n:
            rows.append({"Flag": label, "Deals": n, "What to do": meaning})
    return pd.DataFrame(rows)


def render_quality(data: Any | None = None, validation: dict[str, Any] | None = None) -> None:
    """Render the page. `data` and `validation` are injectable for tests."""
    if data is None:
        data, _ = load_explorer_dataset()
    if validation is None:
        validation = load_validation()

    st.subheader("Research you can check")
    st.write("Every deal in the explorer links to the announcement it came from and quotes the text behind it. "
             "This page explains how the records were made, how accurate the extraction measured, and what the "
             "announcements leave out, so you can decide how far to rely on them.")

    if validation is None:
        st.error("The validation report (`data/pe/validation.json`) could not be read.")
        return
    ds, labels = validation["dataset"], validation["labels"]
    if len(data) != ds["deals"]:
        st.warning(f"The explorer now serves {len(data):,} deals, but the accuracy below was measured for a build of "
                   f"{ds['deals']:,} deals. Re-run the validation before relying on these figures.")

    st.markdown("#### How the deals were produced")
    st.markdown(
        f"1. **Collected** on {ds['collected']}: the public news pages of {ds['firms']} PE firms, respecting each "
        "site's robots.txt. Opening this app never fetches anything.\n"
        "2. **Classified** by fixed rules: is the article a deal, and if so who bought whom, what kind of transaction, "
        "the target's industry and state, the stated rationale, and whether a price was disclosed. Each field keeps "
        "the sentence it was read from.\n"
        "3. **Checked** against reference labels (below) and by automatic consistency checks on every record.\n"
        "4. **Published** to the explorer with short evidence quotes only. A missing value stays *Unknown*, never "
        "zero or a guess, and repeated coverage of one deal is grouped under one deal ID.")
    st.caption(f"Source: {ds['source']}. Build fingerprint `{ds['build_fingerprint']}`; two builds from the same saved "
               "pages are identical.")

    st.markdown("#### Measured accuracy")
    st.warning(f"These labels were written by {labels['labelled_by']}. "
               + ("" if labels["verified_by_human"] else "No person has checked them yet. ")
               + "An AI checking rules it wrote can share their blind spots, so treat these figures as provisional "
               "until a team member re-labels a random sample.")
    st.dataframe(accuracy_table(validation), width="stretch", hide_index=True)
    det = validation["deal_detection_held_out"]
    st.caption(f"Held-out: {labels['held_out_articles']} articles. {labels['held_out_note']} "
               f"Deal detection on them: {det['deals_found']} deals found, {det['false_alarms']} false alarms, "
               f"{det['missed']} missed. Development: {labels['development_articles']} articles. "
               f"{labels['development_note']} Intervals are 95% Wilson intervals; with about 30 articles they are wide.")

    later = validation.get("later_independent_round")
    if later:
        st.markdown("##### A stricter test on newer rules")
        st.write(f"Later work tested newer rules against {later['articles']} articles labelled by "
                 f"{later['labelled_by']}. {later['scope']} Accuracy was noticeably lower, especially for industry "
                 "and state, so expect more errors in those fields than the shipped build's own test suggests.")
        st.dataframe(later_round_table(validation), width="stretch", hide_index=True)

    st.markdown("#### What this dataset leaves out")
    total = len(data)
    review = data["review_status"].astype("string").str.casefold()
    m1, m2, m3 = st.columns(3)
    m1.metric("Deals", f"{total:,}")
    m2.metric("Need review", f"{int(review.eq('needs review').sum()):,}")
    m3.metric("Inferred classification", f"{int(data['is_inferred'].fillna(False).astype(bool).sum()):,}")
    st.dataframe(unknown_table(data), width="stretch", hide_index=True)
    st.caption("Firms rarely publish prices or closing dates in their own announcements, so those stay unknown; "
               "*Unknown* means not stated, not zero.")

    flags = flag_table(data)
    if not flags.empty:
        st.markdown("##### Why deals are flagged")
        st.write("A deal is marked **needs review** for the first four kinds of flag below. "
                 "Date flags only say where the date came from. Each deal's flags are in its card in the explorer.")
        st.dataframe(flags, width="stretch", hide_index=True)

    with st.expander("Automatic consistency checks on every record"):
        checks = pd.DataFrame(validation["consistency_checks"]).rename(
            columns={"check": "Check", "failing": "Records failing"})
        st.dataframe(checks, width="stretch", hide_index=True)

    st.markdown("#### Known limitations")
    st.markdown(
        f"- **Coverage:** {ds['firms']} firms, and only the announcements they chose to post. This is not a sample of "
        "the PE market, and counts or charts say nothing about market activity.\n"
        "- **A snapshot:** the data is as collected on " + ds["collected"] + ". Newer announcements appear only "
        "after the collection and conversion are re-run.\n"
        "- **Self-reported rationale:** the rationale is what the firm said it wanted. It is a research lead, not "
        "evidence that the deal improved earnings or returns.\n"
        "- **Industry and geography** are the least reliable fields. Industry comes from keywords in the target's "
        "description, and geography is known only when the release names a US state or city.\n"
        "- **Publisher is not always the buyer:** a firm's page often announces an add-on bought by one of its "
        "portfolio companies, or a sale to a third party.")

    st.markdown("#### Using it well")
    st.markdown(
        "- Open the source announcement before citing a deal, and quote it rather than the extracted field.\n"
        "- Treat deals marked *needs review* or *inferred* as leads to confirm.\n"
        "- Report counts as \"announcements by these firms\", not as deals in the market.")
