"""Convert the PE deal collector's output into the Deal Explorer's dataset.

Source: `data/pe_deals/output/deals.json`, written by the collector and rule-based extractor
(`python -m pedeals`, on the `claude/dashboard-functionality-yi0loj` branch). Each source record
is one deal announcement from a PE firm's own news page, with the quoted evidence behind each field.

    python scripts/build_pe_explorer_dataset.py path/to/deals.json [data/pe/deals.json]

Only short evidence quotes are kept, not article bodies. Unknown values stay null so the
explorer shows "Unknown" instead of inventing a value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "pe" / "deals.json"

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

# Flags that question who did the deal or what the target is. Date-source flags are informational.
NEEDS_REVIEW_FLAGS = ("buyer not found", "industry ambiguous", "relationship inferred", "several targets")
UNKNOWN = {"", "unknown", "not stated", "none"}


def _known(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in UNKNOWN else text


def _evidence(rec: dict) -> str | None:
    parts = [_known(rec.get("parties_evidence")), _known(rec.get("rationale"))]
    parts = [p for i, p in enumerate(parts) if p and p not in parts[:i]]
    return " … ".join(parts) or None


def convert(rec: dict) -> dict:
    flags = [f.strip() for f in str(rec.get("review_flags") or "").split(";") if f.strip()]
    needs_review = any(f.startswith(NEEDS_REVIEW_FLAGS) for f in flags)
    price = rec.get("deal_value_usd")
    return {
        "data_status": "collected",
        "deal_id": rec["deal_id"],
        "announcement_ids": [rec["deal_id"]],
        "publisher_pe_firm": rec["publisher"],
        "source_url": [rec["url"]],
        "headline": rec["headline"],
        "publication_date": rec["published_date"],
        "buyer_or_investor": _known(rec.get("buyer")),
        "target_company": _known(rec.get("target")),
        "transaction_type": _known(rec.get("transaction_type")),
        "transaction_status": _known(rec.get("deal_status")),
        "target_industry": _known(rec.get("target_industry")),
        "target_geography": US_STATES.get(str(rec.get("target_state") or "").strip()),
        "rationale_tags": [t.strip() for t in str(rec.get("rationale_tags") or "").split(";") if t.strip()],
        "price": price if isinstance(price, (int, float)) and price > 0 else None,
        "currency": "USD" if isinstance(price, (int, float)) and price > 0 else None,
        "evidence_excerpt": _evidence(rec),
        "review_status": "needs review" if needs_review else "machine-extracted",
        "review_notes": "; ".join(flags) or None,
        "is_inferred": bool(rec.get("industry_ambiguous")) or any("inferred" in f for f in flags),
    }


def build(source: Path, out: Path = DEFAULT_OUT) -> list[dict]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload["deals"] if isinstance(payload, dict) else payload
    deals = [convert(r) for r in records
             if r.get("is_deal") and r.get("in_period", True) and not r.get("duplicate_of")]
    deals.sort(key=lambda d: (d["publication_date"] or "", d["deal_id"]), reverse=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(deals, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return deals


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    written = build(Path(sys.argv[1]), target)
    print(f"wrote {len(written)} deals to {target}")
