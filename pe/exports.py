"""Export helpers for PE Deal Intelligence."""
from __future__ import annotations

from typing import Any
import pandas as pd

from pe.analytics import DEAL_COLUMNS, load_deals

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


def export_filtered(deals: Any) -> bytes:
    """Return UTF-8 CSV bytes for the already-filtered deal records."""
    return export_frame(deals).to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def data_dictionary_csv() -> bytes:
    frame = pd.DataFrame([{"field": key, "explanation": value} for key, value in FIELD_EXPLANATIONS.items()])
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
