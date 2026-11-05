"""Evaluate automated candidate records against an independently reviewed reference set.

Reference JSON should be a list of manually reviewed deal records with stable deal_id
values and, where available, evidence_excerpt. This module does not invent labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

FIELD_MAP = {"buyer_or_investor": "buyer_or_investor", "target_company": "target_company",
             "transaction_type": "transaction_type", "publication_date": "publication_date"}


def _norm(value: Any) -> str | None:
    if value is None or pd.isna(value): return None
    text = " ".join(str(value).strip().casefold().split())
    return text or None


def evaluate(predicted: list[dict[str, Any]], reference: list[dict[str, Any]], *, failed_pages: int = 0,
             attempted_pages: int = 0) -> dict[str, Any]:
    pred_ids = [str(r.get("deal_id", "")).strip() for r in predicted if str(r.get("deal_id", "")).strip()]
    ref_ids = [str(r.get("deal_id", "")).strip() for r in reference if str(r.get("deal_id", "")).strip()]
    pset, rset = set(pred_ids), set(ref_ids)
    tp = len(pset & rset)
    field_accuracy = {}
    for field in FIELD_MAP:
        compared = correct = missing_pred = 0
        ref_by_id = {str(r.get("deal_id")): r for r in reference}
        pred_by_id = {str(r.get("deal_id")): r for r in predicted}
        for deal_id in sorted(pset & rset):
            expected = _norm(ref_by_id[deal_id].get(field))
            actual = _norm(pred_by_id[deal_id].get(field))
            compared += 1
            if actual is None and expected is not None: missing_pred += 1
            if actual == expected: correct += 1
        field_accuracy[field] = {"correct": correct, "compared": compared,
                                 "accuracy": correct / compared if compared else None,
                                 "missing_prediction_when_reference_known": missing_pred}
    evidence_total = sum(bool(_norm(r.get("evidence_excerpt"))) for r in predicted)
    review_status_counts: dict[str, int] = {}
    for row in predicted:
        status = str(row.get("review_status") or "unknown")
        review_status_counts[status] = review_status_counts.get(status, 0) + 1
    return {
        "reference_set_size": len(rset), "predicted_unique_deals": len(pset), "true_positive_deals": tp,
        "false_positive_deals": len(pset - rset), "false_negative_deals": len(rset - pset),
        "deal_detection_precision": tp / len(pset) if pset else None,
        "deal_detection_recall": tp / len(rset) if rset else None,
        "field_accuracy": field_accuracy,
        "duplicate_predicted_rows": len(pred_ids) - len(pset),
        "duplicate_group_errors": None,  # repeated IDs do not measure incorrect cross-article group assignments
        "records_with_evidence": evidence_total, "predicted_records": len(predicted),
        "review_status_counts": review_status_counts,
        "failed_pages": failed_pages, "attempted_pages": attempted_pages,
        "failed_page_rate": failed_pages / attempted_pages if attempted_pages else None,
        "limitations": "Metrics require independently reviewed reference labels and valid matching deal IDs. Duplicate rows are counted; transaction-group accuracy is not measured by this evaluator.",
    }


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict): payload = payload.get("deals", [])
    if not isinstance(payload, list): raise ValueError(f"Expected a JSON list in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PE deal extraction against reviewed reference records")
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--collection-report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/pe/validation_results.json"))
    args = parser.parse_args()
    report = {}
    if args.collection_report and args.collection_report.exists():
        report = json.loads(args.collection_report.read_text(encoding="utf-8"))
    result = evaluate(_load(args.predicted), _load(args.reference), failed_pages=int(report.get("failed", 0)), attempted_pages=int(report.get("attempted", 0)))
    result["reference_file"] = str(args.reference)
    result["predicted_file"] = str(args.predicted)
    result["evaluated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
