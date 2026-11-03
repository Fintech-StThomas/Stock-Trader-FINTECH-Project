"""Sources and coverage placeholder connected to the shared repository."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pe.contracts import DatasetMetadata, DealRecord
from pe.styles import show_metadata, show_table


def render_page(records: tuple[DealRecord, ...], metadata: DatasetMetadata) -> None:
    st.subheader("Sources and Coverage")
    show_metadata(metadata)
    rows = [{"source": r.source_name or "Not provided", "source_url": r.source_url or "Not provided", "deal_id": r.deal_id}
            for r in records]
    show_table(pd.DataFrame(rows), empty_message="No source records have been published.")
    st.caption("Coverage dates and source counts describe the published dataset, not the full PE market.")
