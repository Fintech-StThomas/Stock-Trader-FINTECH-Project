"""Export helpers for PE Deal Intelligence."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from pe.analytics import DEAL_COLUMNS, filter_deals, load_deals

FIELD_EXPLANATIONS = {
    "deal_id": "Stable identifier for one grouped transaction.",
    "announcement_ids": "Identifiers for all source announcements associated with this deal.",
    "publisher_pe_firm": "PE firm or publisher associated with the announcement; not necessarily the direct buyer.",
    "source_url": "Original announcement URL(s) used for this record.",
    "headline": "Headline from a source announcement.",
    "publication_date": "Publication date of the available announcement; not the transaction closing date.",
    "buyer_or_investor": "Buyer or investor named in the available source evidence.",
    "target_company": "Company or asset identified as the target.",
    "transaction_type": "Classification such as acquisition, investment, exit, or unknown.",
    "transaction_status": "Status stated in the available source, such as announced or closed.",
    "target_industry": "Industry assigned to the target, not its customers.",
    "target_geography": "Geography associated with the target or transaction.",
    "rationale_tags": "Stated or inferred strategic rationale categories; consult evidence before relying on them.",
    "price": "Transaction value only when disclosed; blank means unknown or undisclosed, not zero.",
    "currency": "Currency for the disclosed transaction value.",
    "evidence_excerpt": "Short source excerpt supporting the deal details or rationale.",
    "review_status": "Current validation or human-review status.",
    "review_notes": "Extraction flags behind the review status, such as where the date came from or an ambiguous industry.",
    "is_inferred": "Whether a classification or value is inferred rather than explicitly stated.",
}


def export_frame(deals: Any) -> pd.DataFrame:
    """Create a stable, human-readable CSV frame without inventing missing values."""
    df = load_deals(deals)
    out = df.reindex(columns=DEAL_COLUMNS).copy()
    for col in ["announcement_ids", "source_url", "rationale_tags"]:
        out[col] = out[col].map(lambda value: " | ".join(value) if isinstance(value, list) else "")
    out["publication_date"] = out["publication_date"].dt.strftime("%Y-%m-%d").fillna("")
    out["price"] = out["price"].where(out["price"].notna(), "")
    out["field_explanations"] = "See the data dictionary in the Lighthouse PE Deal Intelligence README."
    return out


def export_filtered(deals: Any, filters: Mapping[str, Any] | None = None) -> bytes:
    """Return UTF-8 CSV bytes for unique deals, optionally applying explorer filters.

    Passing the unfiltered shared dataset and the same filter arguments used by the
    explorer keeps downloadable rows aligned with the charts and results table.
    Callers may also pass an already-filtered frame (for example, a saved shortlist).
    """
    selected = filter_deals(deals, **dict(filters)) if filters else load_deals(deals)
    return export_frame(selected).to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def data_dictionary_csv() -> bytes:
    frame = pd.DataFrame([{"field": key, "explanation": value} for key, value in FIELD_EXPLANATIONS.items()])
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def export_competitive_summary(deals: Any) -> bytes:
    """Descriptive summary of exactly the current filtered records, with source evidence."""
    from pe.analytics import count_by_category
    df = load_deals(deals)
    lines = ['# Lighthouse competitive research summary', '',
             'Audience: PE investment analysts monitoring competing firms.', '',
             f'Selected unique deal records: {len(df)}', '',
             'This summary describes the selected saved records, not all market or firm activity. '
             'Categories may be inferred. Announced rationale does not establish realized returns. '
             'Curated additions are flagged in review notes and excluded from historical extraction accuracy results.', '']
    known = df['publication_date'].dropna()
    lines += [f'Known publication dates: {known.min():%Y-%m-%d} to {known.max():%Y-%m-%d}' if len(known) else 'Known publication dates: Unknown',
              f'Records with unknown publication dates: {df["publication_date"].isna().sum()}', '']
    for title,col in [('Activity by firm','publisher_pe_firm'),('Target sectors','target_industry'),
                      ('Target geography','target_geography'),('Strategic rationales','rationale_tags')]:
        lines += ['## '+title,'']
        lines += [f'- {r.category}: {r.deal_count}' for r in count_by_category(df,col).itertuples()]
        lines += ['']
    lines += ['## Announcements and evidence','']
    for row in df.to_dict('records'):
        lines += [f'### {row["publisher_pe_firm"]}: {row["target_company"] or "Unknown target"}', '',
                  f'Transaction: {row["transaction_type"]} | Status: {row["transaction_status"]}',
                  f'Review: {row["review_status"]}', f'Notes: {row["review_notes"] or "None saved"}', '',
                  str(row['evidence_excerpt'] or 'No evidence excerpt saved.'), '']
        lines += [f'- Source: {url}' for url in row['source_url']]
        lines += ['']
    return '\n'.join(lines).encode('utf-8')
