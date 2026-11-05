from pe.validation import evaluate


def test_validation_metrics_include_denominators_missing_fields_and_duplicate_rows():
    reference = [
        {"deal_id": "D1", "buyer_or_investor": "Buyer", "target_company": "Target", "transaction_type": "acquisition", "publication_date": "2026-01-01"},
        {"deal_id": "D2", "buyer_or_investor": "Other", "target_company": "Asset", "transaction_type": "investment", "publication_date": "2026-02-01"},
    ]
    predicted = [
        {"deal_id": "D1", "buyer_or_investor": "Buyer", "target_company": "Target", "transaction_type": "acquisition", "publication_date": "2026-01-01", "evidence_excerpt": "quote"},
        {"deal_id": "D1", "buyer_or_investor": "Buyer", "target_company": "Target", "transaction_type": "acquisition", "publication_date": "2026-01-01"},
        {"deal_id": "D3", "buyer_or_investor": None, "target_company": "Unknown", "transaction_type": "acquisition", "publication_date": None},
    ]
    result = evaluate(predicted, reference, failed_pages=1, attempted_pages=4)
    assert result["deal_detection_precision"] == 0.5
    assert result["deal_detection_recall"] == 0.5
    assert result["duplicate_predicted_rows"] == 1
    assert result["duplicate_group_errors"] is None
    assert result["field_accuracy"]["buyer_or_investor"]["correct"] == 1
    assert result["failed_page_rate"] == 0.25
