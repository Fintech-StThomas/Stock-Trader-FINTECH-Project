"""Single-load repository for the published PE dataset and last-known-good copy."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pe.contracts import CONTRACT_VERSION, DatasetMetadata, DealRecord

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "pe" / "data" / "deals.json"
DEFAULT_LAST_KNOWN_GOOD = ROOT / "pe" / "data" / "last_known_good.json"


@dataclass(frozen=True)
class DatasetSnapshot:
    records: tuple[DealRecord, ...]
    metadata: DatasetMetadata


def _parse_payload(path: Path, mode: str) -> DatasetSnapshot:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"Dataset contract version must be {CONTRACT_VERSION}")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Dataset 'records' must be a list")
    records = tuple(DealRecord.from_mapping(row) for row in raw_records)
    ids = [row.deal_id for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contains duplicate deal_id values")
    meta = payload.get("metadata", {})
    sources = {row.source_name for row in records if row.source_name}
    metadata = DatasetMetadata(
        data_version=str(meta.get("data_version", "unversioned")),
        contract_version=CONTRACT_VERSION,
        last_updated=meta.get("last_updated"),
        coverage_start=meta.get("coverage_start"),
        coverage_end=meta.get("coverage_end"),
        source_count=len(sources),
        record_count=len(records),
        coverage_notes=str(meta.get("coverage_notes", "Coverage has not been independently assessed.")),
        mode=mode,
    )
    return DatasetSnapshot(records, metadata)


def _save_last_known_good(source: Path, destination: Path) -> None:
    """Atomically persist a successfully parsed published dataset for later outages."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        with source.open("rb") as original:
            shutil.copyfileobj(original, temp)
    os.replace(temp_path, destination)


@lru_cache(maxsize=8)
def _load_cached(published_path: str, fallback_path: str, published_mtime: Optional[int], fallback_mtime: Optional[int]) -> DatasetSnapshot:
    published, fallback = Path(published_path), Path(fallback_path)
    try:
        snapshot = _parse_payload(published, "published")
        try:
            _save_last_known_good(published, fallback)
        except OSError:
            # Read access should not fail just because the runtime is read-only.
            pass
        return snapshot
    except (OSError, ValueError, json.JSONDecodeError) as primary_error:
        try:
            cached = _parse_payload(fallback, "cached")
            return cached
        except (OSError, ValueError, json.JSONDecodeError) as fallback_error:
            raise RuntimeError(
                "PE dataset could not be loaded and no valid last-known-good copy is available. "
                f"Published error: {primary_error}; cached error: {fallback_error}"
            ) from fallback_error


def load_dataset(
    published_path: Path | str = DEFAULT_DATASET,
    fallback_path: Path | str = DEFAULT_LAST_KNOWN_GOOD,
    *,
    force_reload: bool = False,
) -> DatasetSnapshot:
    """Load once per file version; a failed publish falls back to the saved known-good copy."""
    published, fallback = Path(published_path), Path(fallback_path)
    if force_reload:
        _load_cached.cache_clear()
    published_mtime = published.stat().st_mtime_ns if published.exists() else None
    fallback_mtime = fallback.stat().st_mtime_ns if fallback.exists() else None
    return _load_cached(str(published), str(fallback), published_mtime, fallback_mtime)


def clear_repository_cache() -> None:
    """Clear the process cache, primarily for tests and explicit refresh actions."""
    _load_cached.cache_clear()


def utc_now_iso() -> str:
    """Return a timezone-aware ISO timestamp for dataset publishers."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
