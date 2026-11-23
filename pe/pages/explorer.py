"""Placeholder Deal Explorer connected to the published dataset contract."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pe.contracts import DatasetMetadata, DealRecord
from pe.styles import show_metadata, show_table


def render_page(records: tuple[DealRecord, ...], metadata: DatasetMetadata) -> None:
    st.subheader("Deal Explorer")
    show_metadata(metadata)
    if not records:
        show_table(pd.DataFrame(), empty_message="No deals have been published yet.")
        return
    frame = pd.DataFrame([r.to_dict() for r in records])
    st.caption("Example records are shown for contract testing only and are not verified transactions.")
    show_table(frame[["deal_id", "buyer", "target", "announced_date", "industry", "geography", "record_status", "source_url"]])
