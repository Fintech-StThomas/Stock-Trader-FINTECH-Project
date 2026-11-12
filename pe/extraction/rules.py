"""Transparent rule-based PE announcement extraction with review flags.

This offline baseline intentionally prefers unknown + review over fabricated values.
It is not a substitute for human review or a validated model extraction pipeline.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

EXTRACTOR_VERSION = "rules-1.0"
DEAL_TERMS = ("acquire", "acquisition", "buyout", "investment", "invests", "invested", "sale to", "sold to", "divest", "exit", "merger", "merges", "purchase")
NON_DEAL_TERMS = ("quarterly results", "market outlook", "appoints", "fundraising", "raises a fund", "portfolio update", "thought leadership")
RATIONALE_RULES = {
    "distribution expansion": ("distribution", "channel", "retail network", "route to market"),
    "product expansion": ("product portfolio", "new products", "product offering", "complementary products"),
    "technology expansion": ("technology", "software", "automation", "digital", "data platform"),
    "geographic expansion": ("geographic", "new markets", "international expansion", "regional markets", "enter new markets"),
}


def _excerpt(text: str, terms: tuple[str, ...], limit: int = 280) -> str | None:
    lower = text.casefold()
    for term in terms:
        at = lower.find(term)
        if at >= 0:
            left = max(0, text.rfind(".", 0, at) + 1)
            right = text.find(".", at)
            right = len(text) if right < 0 else right + 1
            snippet = " ".join(text[left:right].split()).strip()
            return snippet[:limit].strip() or None
    return None


def _deal_id(buyer: str | None, target: str | None, headline: str, date: str | None) -> str:
    key = ("|".join([buyer.casefold(), target.casefold(), (date or "unknown")[:7]]) if buyer and target else "|".join([(buyer or "unknown").casefold(), (target or "unknown").casefold(), (date or "unknown")[:7], headline.casefold()[:70]]))
    return "DEAL-" + hashlib.sha256(key.encode()).hexdigest()[:14].upper()


def extract(announcements: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return deals, evidence, and a review queue from normalized Announcement records."""
    deals: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    for ann in announcements:
        headline = str(ann.get("headline") or "")
        text = str(ann.get("full_text") or "")
        source_role = str(ann.get("source_role") or "deal_news")
        if source_role in {"market_context", "legal_regulatory"}:
            review_queue.append({"announcement_id": ann.get("announcement_id"), "classification": "context", "review_status": "context only; not a transaction record", "reason": f"Source role is {source_role}; retain for market/legal analysis, not deal counting."})
            continue
        combined = f"{headline}. {text}".strip()
        lower = combined.casefold()
        hits = [term for term in DEAL_TERMS if term in lower]
        non_deal = any(term in lower for term in NON_DEAL_TERMS) and not hits
        if non_deal:
            review_queue.append({"announcement_id": ann.get("announcement_id"), "classification": "non-deal", "review_status": "rule candidate; confirm manually", "reason": "Non-deal indicator found; no transaction indicator."})
            continue
        if not hits:
            review_queue.append({"announcement_id": ann.get("announcement_id"), "classification": "uncertain", "review_status": "needs review", "reason": "No clear transaction indicator found."})
            continue

        date = ann.get("publication_date")
        if date:
            try: date = datetime.fromisoformat(str(date).replace("Z", "+00:00")).date().isoformat()
            except ValueError: date = None
        transaction_type = "investment" if any(x in lower for x in ("investment", "invests", "invested")) else "exit" if any(x in lower for x in ("exit", "divest", "sold its stake")) else "acquisition" if any(x in lower for x in ("acquire", "acquisition", "buyout", "purchase", "merger")) else None
        # Named-party extraction is deliberately conservative: only populate when explicit phrases fit.
        buyer = target = None
        patterns = [r"(?P<buyer>[A-Z][A-Za-z0-9&.,' -]{1,70}?)\s+(?:has agreed to acquire|agreed to acquire|acquires|acquired|will acquire|to acquire)\s+(?P<target>[A-Z][A-Za-z0-9&.,' -]{1,70}?)(?:[.,;]|\s+for\s+\$)",
                    r"(?P<buyer>[A-Z][A-Za-z0-9&.,' -]{1,70}?)\s+(?:makes an investment in|invests in|invested in)\s+(?P<target>[A-Z][A-Za-z0-9&.,' -]{1,70}?)(?:[.,;])"]
        for pattern in patterns:
            match = re.search(pattern, combined)
            if match:
                buyer, target = (match.group("buyer").strip(" .,"), match.group("target").strip(" .,"))
                break
        rationale_tags = [tag for tag, terms in RATIONALE_RULES.items() if any(term in lower for term in terms)]
        rationale_excerpt = _excerpt(combined, tuple(term for terms in RATIONALE_RULES.values() for term in terms))
        transaction_excerpt = _excerpt(combined, DEAL_TERMS)
        deal_id = _deal_id(buyer, target, headline, date)
        uncertain = not buyer or not target or not date or not rationale_tags
        status = "needs review" if uncertain else "unreviewed extraction"
        record = {"deal_id": deal_id, "announcement_ids": [ann.get("announcement_id")] if ann.get("announcement_id") else [],
            "data_status": "demo" if ann.get("data_status") == "demo" or ann.get("collection_status") == "demo" else "unreviewed extraction", "publisher_pe_firm": ann.get("publisher_pe_firm"), "source_name": ann.get("source_name"),
            "source_role": source_role, "content_type": ann.get("content_type"), "source_url": [ann.get("source_url")] if ann.get("source_url") else [],
            "headline": headline or None, "publication_date": date, "buyer_or_investor": buyer, "target_company": target,
            "transaction_type": transaction_type, "transaction_status": "announced", "target_industry": None,
            "target_geography": None, "rationale_tags": rationale_tags, "price": None, "currency": None,
            "evidence_excerpt": rationale_excerpt or transaction_excerpt, "review_status": status,
            "is_inferred": bool(rationale_tags), "extractor_version": EXTRACTOR_VERSION}
        deals.append(record)
        if transaction_excerpt:
            evidence.append({"deal_id": deal_id, "field_name": "transaction_type", "announcement_id": ann.get("announcement_id"), "excerpt": transaction_excerpt, "evidence_type": "fact", "review_status": "unreviewed", "extractor_version": EXTRACTOR_VERSION})
        if rationale_excerpt:
            evidence.append({"deal_id": deal_id, "field_name": "rationale_tags", "announcement_id": ann.get("announcement_id"), "excerpt": rationale_excerpt, "evidence_type": "interpretation", "review_status": "unreviewed", "extractor_version": EXTRACTOR_VERSION})
        if uncertain:
            review_queue.append({"announcement_id": ann.get("announcement_id"), "deal_id": deal_id, "classification": "possible-deal", "review_status": "needs review", "reason": "One or more key fields could not be extracted confidently."})
    return {"deals": deals, "evidence": evidence, "review_queue": review_queue, "extractor_version": EXTRACTOR_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat()}
