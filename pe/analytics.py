"""Data preparation, filtering, and summary analytics for PE Deal Intelligence.

This module is deliberately independent of Streamlit so the core logic can be tested
without launching the web application.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any

import pandas as pd

DEAL_COLUMNS = [
    "deal_id", "announcement_ids", "publisher_pe_firm", "source_url", "headline",
    "publication_date", "buyer_or_investor", "target_company", "transaction_type",
    "transaction_status", "target_industry", "target_geography", "rationale_tags",
    "price", "currency", "evidence_excerpt", "review_status", "review_notes", "is_inferred",
]

FILTERABLE_COLUMNS = {
    "publisher": "publisher_pe_firm",
    "transaction_type": "transaction_type",
    "target_industry": "target_industry",
    "geography": "target_geography",
}


def _as_list(value: Any) -> list[str]:
    """Convert a scalar or list-like cell to a clean list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return [str(v).strip() for v in values if v is not None and str(v).strip() and str(v).lower() != "nan"]


def load_deals(source: Any) -> pd.DataFrame:
    """Load and normalize deal records from a DataFrame, JSON path, or records.

    Duplicate rows sharing a deal_id are collapsed to one deal for counts and charts.
    Source URLs and announcement IDs from repeated coverage are preserved as lists.
    Missing values remain null rather than being silently replaced with zero.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (str, bytes, PathLike)):
        df = pd.read_json(source)
    elif isinstance(source, Iterable):
        df = pd.DataFrame(list(source))
    else:
        raise TypeError("source must be a DataFrame, JSON path, or iterable of records")

    for col in DEAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if df.empty:
        return pd.DataFrame(columns=DEAL_COLUMNS)

    df["deal_id"] = df["deal_id"].astype("string").str.strip()
    df = df[df["deal_id"].notna() & df["deal_id"].ne("")].copy()
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce", utc=True).dt.tz_convert(None)
    for col in ["price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["announcement_ids", "rationale_tags"]:
        df[col] = df[col].map(_as_list)
    df["source_url"] = df["source_url"].map(_as_list)

    if df["deal_id"].is_unique:
        # Nothing to collapse (the usual case, and every re-normalisation of already-loaded deals):
        # skip the per-group loop below, which costs about a second per call on a hundred deals.
        for col in ["announcement_ids", "source_url", "rationale_tags"]:
            df[col] = df[col].map(lambda items: list(dict.fromkeys(items)))
        return df.reset_index(drop=True).reindex(columns=DEAL_COLUMNS)

    # Collapse repeat rows for a transaction, retaining all source references.
    rows: list[dict[str, Any]] = []
    for deal_id, group in df.groupby("deal_id", sort=False, dropna=False):
        first = group.iloc[0].to_dict()
        for col in ["announcement_ids", "source_url", "rationale_tags"]:
            combined: list[str] = []
            for value in group[col]:
                for item in _as_list(value):
                    if item not in combined:
                        combined.append(item)
            first[col] = combined
        # Prefer the first known value, not an empty value from an earlier announcement.
        for col in DEAL_COLUMNS:
            if col in {"deal_id", "announcement_ids", "source_url", "rationale_tags"}:
                continue
            known = group[col].dropna()
            if len(known):
                nonempty = known[known.astype(str).str.strip().ne("")]
                if len(nonempty):
                    first[col] = nonempty.iloc[0]
        rows.append(first)
    result = pd.DataFrame(rows)
    return result.reindex(columns=DEAL_COLUMNS)


def _contains_any(cell: Any, selected: set[str]) -> bool:
    values = set(_as_list(cell))
    if not values:
        return False
    return bool(values & selected)


def filter_deals(
    deals: Any,
    *,
    search: str = "",
    publishers: Iterable[str] | None = None,
    start_date: Any = None,
    end_date: Any = None,
    transaction_types: Iterable[str] | None = None,
    industries: Iterable[str] | None = None,
    geographies: Iterable[str] | None = None,
    rationales: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return unique deals matching all supplied filters."""
    df = load_deals(deals)
    mask = pd.Series(True, index=df.index)

    for selected, col in [
        (publishers, "publisher_pe_firm"),
        (transaction_types, "transaction_type"),
        (industries, "target_industry"),
        (geographies, "target_geography"),
    ]:
        chosen = {str(v) for v in (selected or [])}
        if chosen:
            mask &= df[col].astype("string").isin(chosen).fillna(False)

    if start_date is not None:
        start = pd.Timestamp(start_date)
        mask &= df["publication_date"].notna() & (df["publication_date"] >= start)
    if end_date is not None:
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= df["publication_date"].notna() & (df["publication_date"] <= end)

    chosen_rationales = {str(v) for v in (rationales or [])}
    if chosen_rationales:
        mask &= df["rationale_tags"].map(lambda x: _contains_any(x, chosen_rationales))

    query = str(search or "").strip().casefold()
    if query:
        searchable = ["publisher_pe_firm", "headline", "buyer_or_investor", "target_company",
                      "target_industry", "target_geography", "transaction_type", "evidence_excerpt"]
        text_mask = pd.Series(False, index=df.index)
        for col in searchable:
            text_mask |= df[col].astype("string").str.casefold().str.contains(query, na=False, regex=False)
        text_mask |= df["rationale_tags"].map(lambda tags: any(query in tag.casefold() for tag in _as_list(tags)))
        mask &= text_mask

    return df.loc[mask].copy().sort_values("publication_date", ascending=False, na_position="last").reset_index(drop=True)


def filter_options(deals: Any) -> dict[str, list[str]]:
    """Return stable, sorted options for the explorer's filter controls."""
    df = load_deals(deals)
    options: dict[str, list[str]] = {}
    for key, col in FILTERABLE_COLUMNS.items():
        options[key] = sorted({str(v).strip() for v in df[col].dropna() if str(v).strip()})
    options["rationale"] = sorted({tag for tags in df["rationale_tags"] for tag in _as_list(tags)})
    return options


def summary_metrics(deals: Any) -> dict[str, int]:
    """Simple counts, always based on unique deal IDs."""
    df = load_deals(deals)
    return {
        "unique_deals": int(df["deal_id"].nunique()),
        "disclosed_values": int(df["price"].notna().sum()),
        "needs_review": int(df["review_status"].astype("string").str.casefold().isin({"pending", "needs review", "flagged"}).sum()),
        "with_evidence": int((df["evidence_excerpt"].notna() & df["evidence_excerpt"].astype("string").str.strip().ne("")).sum()),
    }


def deals_over_time(deals: Any, frequency: str = "Q") -> pd.DataFrame:
    """Count unique deals by publication period; records without dates are excluded."""
    df = load_deals(deals)
    dated = df.dropna(subset=["publication_date"]).copy()
    if dated.empty:
        return pd.DataFrame(columns=["period", "deal_count"])
    periods = dated["publication_date"].dt.to_period(frequency)
    out = periods.value_counts().sort_index().rename_axis("period").rename("deal_count").reset_index()
    out["period"] = out["period"].astype(str)
    return out


def count_by_category(deals: Any, column: str) -> pd.DataFrame:
    """Count unique deals by a scalar category or list-valued rationale category."""
    df = load_deals(deals)
    if column not in df.columns:
        raise ValueError(f"Unknown category column: {column}")
    counts: dict[str, int] = {}
    for _, row in df.iterrows():
        values = _as_list(row[column])
        if not values:
            values = ["Unknown"]
        for value in set(values):
            counts[value] = counts.get(value, 0) + 1
    return pd.DataFrame([{"category": k, "deal_count": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))])
