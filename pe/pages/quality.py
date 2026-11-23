"""Quality and Method page: display measured validation results without inventing metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd
import streamlit as st

from pe.contracts import DatasetMetadata, QualityReport
from pe.styles import show_metadata, show_table

QUALITY_RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "quality_results.json"


def load_quality_report(path: Path | str = QUALITY_RESULTS_PATH, *, sample_size: int = 0) -> QualityReport:
    """Load team-reviewed quality evidence; absent accuracy stays unknown, never zero or guessed."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    accuracy = payload.get("measured_accuracy")
    if accuracy is not None:
        accuracy = float(accuracy)
        if not 0 <= accuracy <= 1:
            raise ValueError("measured_accuracy must be between 0 and 1")
    reviewed = int(payload.get("reviewed_sample_size", 0))
    unresolved_raw = payload.get("unresolved_errors")
    unresolved = None if unresolved_raw is None else int(unresolved_raw)
    if reviewed < 0 or (unresolved is not None and unresolved < 0):
        raise ValueError("Sample sizes and unresolved error counts cannot be negative")
    return QualityReport(
        sample_size=sample_size,
        reviewed_sample_size=reviewed,
        measured_accuracy=accuracy,
        unresolved_errors=unresolved,
        coverage_limits=tuple(payload.get("coverage_limits", ["Quality evidence has not been published."])),
        test_results=tuple(payload.get("test_results", [])),
        label_source=str(payload.get("label_source", "team-reviewed labels")),
    )


def render_quality(report: QualityReport, metadata: DatasetMetadata) -> None:
    st.subheader("Quality and Method")
    st.write("Validation is reported from team-reviewed labels and recorded automated test runs. Unknown metrics stay unknown.")
    show_metadata(metadata)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", report.sample_size)
    c2.metric("Reviewed labels", report.reviewed_sample_size)
    c3.metric("Measured accuracy", "Not measured" if report.measured_accuracy is None else f"{report.measured_accuracy:.1%}")
    c4.metric("Unresolved errors", "Not measured" if report.unresolved_errors is None else report.unresolved_errors)
    if report.measured_accuracy is None:
        st.info("Accuracy is not yet available: publish a reviewed evaluation sample and its automated results first.")
    st.markdown("#### Automated test results")
    test_frame = pd.DataFrame(report.test_results)
    show_table(test_frame, empty_message="No automated evaluation results have been published. Unit/integration test pass status is not a substitute for extraction accuracy.")
    st.markdown("#### Coverage limits and open questions")
    for item in report.coverage_limits:
        st.markdown(f"- {item}")
    st.markdown("#### Method and review rules")
    st.markdown("""
    - Preserve the original source URL and excerpt for every extracted claim where available.
    - Keep missing facts unknown; do not infer deal terms or outcomes from promotional language.
    - Deduplicate announcements before counting deals; record conflicts and unresolved errors explicitly.
    - Report accuracy only against a team-reviewed holdout sample, including the sample size and label definition.
    - Each feature owner supplies tests and reviews records; Jacob coordinates validation.
    """)


def render_page(records: Sequence, metadata: DatasetMetadata) -> None:
    """Canonical page signature used by pe.app and other PE page owners."""
    report = load_quality_report(sample_size=len(records))
    render_quality(report, metadata)
