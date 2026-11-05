"""Polite, bounded collector for explicitly configured public announcement URLs.

No sources are enabled by default. Team members must review a site's terms and add
approved URLs to data/pe/source_config.json before running a collection.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urljoin

import requests
from lxml import html

from pe.contracts import Announcement, CollectionReport
from pe.collectors.http import PoliteClient
from pe.collectors.config import canonical_url
from pe.storage import DEFAULT_PRIVATE

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = DEFAULT_PRIVATE / "research"
CONFIG_PATH = ROOT / "data/pe/source_config.json"
CACHE_ROOT = DATA_ROOT / "cache"


def clean_text(page: str) -> tuple[str, str]:
    """Extract title and readable text from HTML, removing navigation and scripts."""
    tree = html.fromstring(page)
    title = " ".join((tree.xpath("//title/text()") or [""])[0].split())
    for node in tree.xpath("//script|//style|//nav|//footer|//header|//aside|//form|//noscript"):
        node.drop_tree()
    main = tree.xpath("//main|//article|//*[@role='main']")
    root = main[0] if main else tree
    text = "\n".join(line.strip() for line in root.text_content().splitlines() if line.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def load_source_config(path: str | Path = CONFIG_PATH) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    sources = payload.get("sources", []) if isinstance(payload, dict) else []
    return [s for s in sources if isinstance(s, dict) and s.get("enabled") and (s.get("urls") or s.get("pagination"))]


def _source_urls(source: dict[str, Any]) -> list[str]:
    """Expand explicit URLs and a bounded page-number template for site pagination."""
    urls = [str(url).strip() for url in source.get("urls", []) if str(url).strip()]
    pagination = source.get("pagination") or {}
    template = pagination.get("url_template")
    if template:
        start = max(1, int(pagination.get("start", 1)))
        end = max(start, int(pagination.get("end", start)))
        max_pages = min(100, max(1, int(pagination.get("max_pages", 20))))
        end = min(end, start + max_pages - 1)
        urls.extend(str(template).format(page=page) for page in range(start, end + 1))
    return list(dict.fromkeys(urls))


def collect(source_config: list[dict[str, Any]] | None = None, *, cached: bool = False,
            timeout: float = 12, retries: int = 2, delay_seconds: float = 1.0,
            session: requests.Session | None = None) -> tuple[list[Announcement], CollectionReport]:
    """Fetch explicitly configured URLs or load cache, returning announcements and a report."""
    if session is None and os.getenv("LIGHTHOUSE_OFFLINE", "").lower() in {"1", "true", "yes"}:
        cached = True
    started = datetime.now(timezone.utc)
    sources = source_config if source_config is not None else load_source_config()
    run_id = started.strftime("collect-%Y%m%dT%H%M%S%fZ")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    report: CollectionReport = {"run_id": run_id, "started_at": started.isoformat(), "source_count": len(sources),
                                "attempted": 0, "successful": 0, "failed": 0, "cached": cached, "failures": []}
    announcements: list[Announcement] = []
    client = session or requests.Session()
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    transports = {}
    for source in sources:
        if not cached and session is None and source.get('access_reviewed') is not True:
            report['failures'].append({'source': source.get('name'), 'reason': 'Access review pending; no live requests attempted'})
            report['failed'] += 1
            continue
        publisher = source.get("publisher_pe_firm")
        source_name = str(source.get("name") or publisher or "Unknown")
        source_role = str(source.get("role") or "deal_news")
        source_type = str(source.get("content_type") or "news")
        allowed_hosts = set(source.get("allowed_hosts") or [])
        queue = _source_urls(source)
        queued = set(queue)
        index = 0
        while index < len(queue):
            url = str(queue[index]).strip()
            index += 1
            if source.get('open_access_only') and not urlparse(url).path.startswith('/open-access/'):
                report['failed'] += 1
                report['failures'].append({'url':url,'reason':'Outside configured open-access section'})
                continue
            if not url or url in seen_urls: continue
            seen_urls.add(url)
            host = urlparse(url).hostname or ""
            if allowed_hosts and host not in allowed_hosts:
                report["failed"] += 1; report["failures"].append({"url": url, "reason": "host not in configured allowlist"}); continue
            cache_file = CACHE_ROOT / (hashlib.sha256(url.encode()).hexdigest() + ".json")
            payload = None
            if cached and cache_file.exists():
                try: payload = json.loads(cache_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError): payload = None
            elif not cached:
                report["attempted"] += 1
                last_error = "request failed"
                for attempt in range(max(0, retries) + 1):
                    try:
                        if session is not None:  # explicit injected transport for offline unit tests
                            response = client.get(url, timeout=timeout, headers={"User-Agent": "Lighthouse-PE-Research/1.0"})
                        else:
                            canonical_url(url)
                            origin = urlparse(url).hostname
                            if origin not in transports:
                                def permits(target, host=origin):
                                    try:
                                        target = canonical_url(target)
                                        return urlparse(target).hostname == host
                                    except ValueError:
                                        return False
                                cfg = SimpleNamespace(enabled=True, access_status='internal_research', listing_url=url,
                                    delay_seconds=max(1.0,float(source.get('delay_seconds',delay_seconds))),
                                    timeout_seconds=timeout, retries=0, terms_url=source.get('terms_url',''),
                                    access_reviewed_on=source.get('access_reviewed_on',''), allows=permits)
                                transport=PoliteClient(cfg)
                                transport.prepare()
                                transports[origin]=transport
                            raw, final_url = transports[origin].get(url)
                            response=SimpleNamespace(text=raw, headers={'content-type':'text/html'}, raise_for_status=lambda:None)
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "")
                        if "html" not in content_type and not response.text.lstrip().startswith("<"):
                            raise ValueError("response does not appear to be HTML")
                        title, text = clean_text(response.text)
                        parsed = html.fromstring(response.text)
                        # Listing pages can discover article links, bounded by the source's
                        # host allowlist and optional path keywords. Never crawl off-domain.
                        if url in _source_urls(source) and source.get("discover_articles", False):
                            path_terms = [str(x).casefold() for x in source.get("article_path_contains", [])]
                            for href in parsed.xpath("//a[@href]/@href"):
                                candidate = urljoin(url, str(href)).split("#", 1)[0]
                                parsed_url = urlparse(candidate)
                                if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname not in allowed_hosts:
                                    continue
                                if source.get('open_access_only') and not parsed_url.path.startswith('/open-access/'):
                                    continue
                                if candidate == url or candidate in queued:
                                    continue
                                if len([part for part in parsed_url.path.split("/") if part]) < 2:
                                    continue
                                if path_terms and not any(term in parsed_url.path.casefold() for term in path_terms):
                                    continue
                                if any(x in parsed_url.path.casefold() for x in ("/tag/", "/author/", "/page/", "/category/")):
                                    continue
                                if len(queue) >= min(100,max(1,int(source.get("max_articles", 20)))) + len(_source_urls(source)):
                                    break
                                queued.add(candidate); queue.append(candidate)
                        date_values = parsed.xpath("//meta[@property='article:published_time']/@content | //meta[@name='date']/@content | //time[@datetime]/@datetime")
                        publication_date = None
                        if date_values:
                            try: publication_date = datetime.fromisoformat(str(date_values[0]).replace("Z", "+00:00")).date().isoformat()
                            except ValueError: publication_date = None
                        discovered = [u for u in queue if u not in _source_urls(source)] if url in _source_urls(source) else []
                        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        payload = {"is_listing": bool(url in _source_urls(source) and source.get("discover_articles")), "discovered_urls": discovered, "publisher_pe_firm": publisher, "source_name": source_name,
                                   "source_role": source_role, "content_type": source_type,
                                   "source_url": url, "headline": title,
                                   "publication_date": publication_date, "full_text": text, "content_hash": digest,
                                   "collected_at": datetime.now(timezone.utc).isoformat(), "collection_status": "success", "is_cached": False}
                        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                        break
                    except Exception as exc:  # record per-page failure; never abort a run
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt < retries: time.sleep(min(delay_seconds * (attempt + 1), 5))
                if payload is None:
                    report["failed"] += 1; report["failures"].append({"url": url, "reason": last_error}); continue
                if delay_seconds > 0: time.sleep(max(1, delay_seconds))
            if not payload:
                report["failed"] += 1; report["failures"].append({"url": url, "reason": "no cached record"}); continue
            for found in payload.get('discovered_urls', []):
                if found not in queued and urlparse(found).hostname in allowed_hosts and len(queue) < min(100,int(source.get('max_articles',20))) + len(_source_urls(source)):
                    queued.add(found);queue.append(found)
            if payload.get('is_listing'):
                continue  # a newsroom landing page is not a transaction announcement
            digest = str(payload.get("content_hash") or hashlib.sha256(str(payload.get("full_text", "")).encode()).hexdigest())
            if digest in seen_hashes: continue
            seen_hashes.add(digest)
            announcement_id = "ANN-" + hashlib.sha256(url.encode()).hexdigest()[:16]
            announcements.append({"announcement_id": announcement_id, "publisher_pe_firm": payload.get("publisher_pe_firm", publisher),
                "source_name": payload.get("source_name", source_name), "source_role": payload.get("source_role", source_role),
                "content_type": payload.get("content_type", source_type), "source_url": url, "collected_at": payload.get("collected_at", started.isoformat()),
                "headline": payload.get("headline", ""), "publication_date": payload.get("publication_date"),
                "full_text": payload.get("full_text", ""), "content_hash": digest,
                "collection_status": payload.get("collection_status", "success"), "is_cached": bool(cached or payload.get("is_cached"))})
            report["successful"] += 1
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["output_version"] = started.strftime("announcements-%Y%m%dT%H%M%S%fZ")
    output = DATA_ROOT / f"{report['output_version']}.json"
    output.write_text(json.dumps(announcements, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA_ROOT / "collection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return announcements, report
