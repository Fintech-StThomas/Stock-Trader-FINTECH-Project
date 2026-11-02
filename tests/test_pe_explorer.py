"""Unit tests for PE Deal Explorer logic and CSV exports."""
import json

import pandas as pd

from pe.analytics import count_by_category, deals_over_time, filter_deals, load_deals, summary_metrics
from pe.exports import export_filtered


def sample_records():
    return [
        {
            "deal_id": "D1", "announcement_ids": ["A1"], "publisher_pe_firm": "Firm A",
            "source_url": ["https://example.com/a"], "headline": "Distribution acquisition",
            "publication_date": "2025-02-01", "buyer_or_investor": "Buyer A", "target_company": "Target A",
            "transaction_type": "acquisition", "transaction_status": "announced", "target_industry": "Industrials",
            "target_geography": "United States", "rationale_tags": ["distribution expansion"],
            "price": None, "currency": None, "evidence_excerpt": "Evidence for distribution expansion.",
            "review_status": "reviewed", "is_inferred": False,
        },
        {
            "deal_id": "D1", "announcement_ids": ["A2"], "publisher_pe_firm": "Firm A",
            "source_url": ["https://example.com/b"], "headline": "Duplicate coverage",
            "publication_date": "2025-02-02", "buyer_or_investor": "Buyer A", "target_company": "Target A",
            "transaction_type": "acquisition", "transaction_status": "announced", "target_industry": "Industrials",
            "target_geography": "United States", "rationale_tags": ["distribution expansion"],
            "price": None, "currency": None, "evidence_excerpt": "Another source excerpt.",
            "review_status": "reviewed", "is_inferred": False,
        },
        {
            "deal_id": "D2", "announcement_ids": ["A3"], "publisher_pe_firm": "Firm B",
            "source_url": [], "headline": "Technology investment", "publication_date": "2024-05-10",
            "buyer_or_investor": "Buyer B", "target_company": "Target B", "transaction_type": "investment",
            "transaction_status": "closed", "target_industry": "Technology", "target_geography": "Canada",
            "rationale_tags": ["technology expansion"], "price": 5000000, "currency": "USD",
            "evidence_excerpt": None, "review_status": "needs review", "is_inferred": True,
        },
    ]


def test_load_deals_deduplicates_and_preserves_sources():
    df = load_deals(sample_records())
    assert df["deal_id"].tolist() == ["D1", "D2"]
    d1 = df.loc[df.deal_id.eq("D1")].iloc[0]
    assert set(d1["source_url"]) == {"https://example.com/a", "https://example.com/b"}
    assert set(d1["announcement_ids"]) == {"A1", "A2"}


def test_filters_combine_and_rationale_matches_list_values():
    result = filter_deals(sample_records(), industries=["Industrials"], rationales=["distribution expansion"], search="target a")
    assert result["deal_id"].tolist() == ["D1"]
    assert filter_deals(sample_records(), publishers=["Firm B"], transaction_types=["acquisition"]).empty


def test_date_filter_and_missing_date_handling():
    result = filter_deals(sample_records(), start_date="2025-01-01", end_date="2025-12-31")
    assert result["deal_id"].tolist() == ["D1"]


def test_metrics_and_category_counts_use_unique_deals():
    metrics = summary_metrics(sample_records())
    assert metrics["unique_deals"] == 2
    assert metrics["disclosed_values"] == 1
    assert metrics["needs_review"] == 1
    categories = count_by_category(sample_records(), "rationale_tags").set_index("category")
    assert categories.loc["distribution expansion", "deal_count"] == 1


def test_time_series_counts_unique_deals():
    trend = deals_over_time(sample_records())
    assert int(trend["deal_count"].sum()) == 2


def test_csv_export_keeps_unknown_price_blank_and_source_links():
    csv = export_filtered(filter_deals(sample_records(), publishers=["Firm A"])).decode("utf-8-sig")
    assert "https://example.com/a" in csv and "https://example.com/b" in csv
    assert "Unknown" not in csv  # missing values remain blank, not fabricated labels
    assert "field_explanations" in csv
