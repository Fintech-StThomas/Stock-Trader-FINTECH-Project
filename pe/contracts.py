"""Shared, versioned contracts for the PE Deal Intelligence workspace.

Field names in this module are the source of truth for PE page owners. Unknown
facts must remain ``None`` rather than being inferred from an announcement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class DealRecord:
    deal_id: str
    buyer: str
    target: str
    announced_date: Optional[str] = None
    industry: Optional[str] = None
    geography: Optional[str] = None
    rationale: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    source_excerpt: Optional[str] = None
    record_status: str = "example"
    reviewer: Optional[str] = None
    reviewed_label: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DealRecord":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown deal field(s): {', '.join(sorted(unknown))}")
        for required in ("deal_id", "buyer", "target"):
            if not isinstance(value.get(required), str) or not value[required].strip():
                raise ValueError(f"Deal field '{required}' must be a non-empty string")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetMetadata:
    data_version: str
    contract_version: str
    last_updated: Optional[str]
    coverage_start: Optional[str]
    coverage_end: Optional[str]
    source_count: int
    record_count: int
    coverage_notes: str
    mode: str = "live"


@dataclass(frozen=True)
class QualityReport:
    sample_size: int
    reviewed_sample_size: int
    measured_accuracy: Optional[float]
    unresolved_errors: Optional[int]
    coverage_limits: tuple[str, ...]
    test_results: tuple[dict[str, Any], ...] = ()
    label_source: str = "team-reviewed labels"


def page_signature_reference() -> dict[str, str]:
    """Canonical signatures for page owners; all pages accept the same data shape."""
    return {
        "deal_explorer": "render_deal_explorer(records: Sequence[DealRecord], metadata: DatasetMetadata) -> None",
        "sources": "render_sources(records: Sequence[DealRecord], metadata: DatasetMetadata) -> None",
        "quality": "render_quality(report: QualityReport, metadata: DatasetMetadata) -> None",
    }
