"""
Disk cache for the research data: Parquet tables (and small JSON documents) under data/cache/, which is gitignored.

Bulk sources (EDGAR company facts, filing indexes, factor files, short-interest files) are streamed, filtered to what
Lighthouse uses and stored here as Parquet, so the raw downloads never sit on disk. Writes go through
`data._atomic_write`, so a reader in another thread or process sees the old file or the new one, never half of one.
Every file carries its age (the file's modification time); callers decide how old is too old.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from . import data as _data


def cache_dir() -> Path:
    """The cache root. Read from `data.CACHE_DIR` at call time, so a test that points it elsewhere moves this too."""
    return Path(_data.CACHE_DIR)


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(key)).strip("_") or "_"


def path_for(kind: str, key: str, ext: str = "parquet") -> Path:
    return cache_dir() / _safe(kind) / f"{_safe(key)}.{ext}"


def age_hours(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600.0


def read_frame(kind: str, key: str, max_age_hours: Optional[float] = None) -> Optional[pd.DataFrame]:
    """The cached table, or None when it is missing, older than `max_age_hours`, or unreadable."""
    p = path_for(kind, key)
    if not p.exists():
        return None
    if max_age_hours is not None and age_hours(p) > max_age_hours:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001 - a damaged file is simply rebuilt by the caller
        return None


def write_frame(kind: str, key: str, df: pd.DataFrame) -> Path:
    p = path_for(kind, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.attrs = {}
    _data._atomic_write(p, lambda tmp: out.to_parquet(tmp, index=True))
    return p


def read_json(kind: str, key: str, max_age_hours: Optional[float] = None) -> Optional[Any]:
    p = path_for(kind, key, "json")
    if not p.exists() or (max_age_hours is not None and age_hours(p) > max_age_hours):
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_json(kind: str, key: str, obj: Any) -> Path:
    p = path_for(kind, key, "json")
    p.parent.mkdir(parents=True, exist_ok=True)
    _data._atomic_write(p, lambda tmp: tmp.write_text(json.dumps(obj, default=str)))
    return p
