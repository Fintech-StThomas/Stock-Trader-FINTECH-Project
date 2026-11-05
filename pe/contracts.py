"""Shared data contracts for the PE Deal Intelligence workflow.

The contract intentionally separates announcements (source-level records), deals
(grouped transactions), field-level evidence, and collection/validation manifests.
"""
from __future__ import annotations

from typing import Any, TypedDict


class Announcement(TypedDict, total=False):
    announcement_id: str
    publisher_pe_firm: str | None
    source_name: str
    source_role: str
    content_type: str
    source_url: str
    collected_at: str
    headline: str
    publication_date: str | None
    full_text: str
    content_hash: str
    collection_status: str
    is_cached: bool


class Evidence(TypedDict, total=False):
    deal_id: str
    field_name: str
    announcement_id: str
    excerpt: str
    evidence_type: str  # fact, interpretation, or inferred
    review_status: str
    extractor_version: str


class CollectionReport(TypedDict, total=False):
    run_id: str
    started_at: str
    completed_at: str
    source_count: int
    attempted: int
    successful: int
    failed: int
    cached: bool
    output_version: str | None
    failures: list[dict[str, Any]]
