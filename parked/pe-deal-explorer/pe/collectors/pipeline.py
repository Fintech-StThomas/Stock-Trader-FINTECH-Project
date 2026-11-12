"""Bounded, repeatable announcement collection; never classify transactions here."""
from collections import deque
from copy import deepcopy
from lxml.etree import LxmlError
from .http import PoliteClient, CollectionError
from .parsing import parse_listing, parse_announcement
from .config import canonical_url
from pe.storage import CollectionStore, utc_now, run_id


def collect(source_config, store=None, *, client=None, cached=False):
    """Return (Announcement dictionaries, collection report).

    Cached mode performs no network access and preserves original collection timestamps.
    The CLI holds a store-wide lock; direct callers must likewise serialize writes.
    """
    cfg = source_config
    store = store or CollectionStore()
    if cached:
        records, report = store.load(cfg.source_id)
        if report is None:
            raise CollectionError(f'No successful saved collection for {cfg.source_id}')
        report = deepcopy(report); report['mode'] = 'cached'
        return records, report
    started = utc_now()
    report = {
        'schema_version': 1, 'collector_version': '1.0', 'run_id': run_id(), 'source_id': cfg.source_id,
        'publisher_pe_firm': cfg.publisher_pe_firm, 'started_at': started, 'mode': 'live',
        'requested_period': {'start': cfg.start_date, 'end': cfg.end_date},
        'limits': {'max_pages': cfg.max_pages, 'max_articles': cfg.max_articles},
        'counts': {'attempted_pages': 0, 'successful_pages': 0, 'failed_pages': 0,
                   'listing_pages': 0, 'article_pages': 0, 'discovered_urls': 0,
                   'duplicate_links': 0, 'duplicate_content': 0, 'outside_period': 0,
                   'unknown_dates': 0, 'announcements': 0},
        'errors': [], 'pages': [], 'raw_inputs': [], 'duplicates': [], 'truncated': False, 'stop_reasons': [],
        'coverage_note': 'Announcements from the configured public listing only. Not a complete market or deal count; missing dates remain unverified.'}
    counts = report['counts']
    records, by_hash = [], {}
    previous, _ = store.load(cfg.source_id)
    prior_hash = {row['content_hash']: row for row in previous}
    transport = client or PoliteClient(cfg)

    def fail(url, stage, exc):
        report['errors'].append({'url': url, 'stage': stage, 'message': str(exc)[:240]})

    def fetch_page(url, kind):
        counts['attempted_pages'] += 1
        try:
            raw, final_url = transport.get(url)
            if not cfg.allows(final_url):
                raise CollectionError('Response URL outside configured source')
            raw_path = store.save_raw(cfg.source_id, raw)
            report['raw_inputs'].append({'requested_url': url, 'final_url': final_url, 'raw_path': raw_path})
            return raw, final_url, raw_path
        except (CollectionError, ValueError) as exc:
            counts['failed_pages'] += 1
            fail(url, kind, exc)
            report['pages'].append({'url': url, 'kind': kind, 'status': 'failed'})
            return None

    try:
        transport.prepare()
        report['access_check'] = transport.access
    except (CollectionError, ValueError) as exc:
        fail(cfg.listing_url, 'access', exc)
    else:
        queue, visited, article_urls = deque([cfg.listing_url]), set(), []
        article_seen = set()
        while queue and len(visited) < cfg.max_pages:
            url = queue.popleft()
            if url in visited:
                report['stop_reasons'].append('Repeated pagination link')
                report['truncated'] = True
                continue
            visited.add(url)
            result = fetch_page(url, 'listing')
            if not result:
                continue
            raw, final, raw_path = result
            try:
                links, next_pages = parse_listing(raw, final, cfg)
                if not links:
                    raise ValueError('No article links found; layout or access may have changed')
            except (ValueError, LxmlError) as exc:
                counts['failed_pages'] += 1; fail(url, 'listing_parse', exc)
                report['pages'].append({'url': url, 'kind': 'listing', 'status': 'failed'})
                continue
            counts['successful_pages'] += 1; counts['listing_pages'] += 1
            report['pages'].append({'url': final, 'kind': 'listing', 'status': 'success'})
            for link in links:
                if link in article_seen:
                    counts['duplicate_links'] += 1
                else:
                    article_seen.add(link); article_urls.append(link)
            for link in next_pages:
                if link in visited:
                    report['stop_reasons'].append('Repeated pagination link'); report['truncated'] = True
                elif link not in queue:
                    queue.append(link)
        if queue:
            report['stop_reasons'].append('Listing-page limit reached'); report['truncated'] = True
        counts['discovered_urls'] = len(article_urls)
        if len(article_urls) > cfg.max_articles:
            report['stop_reasons'].append('Article limit reached'); report['truncated'] = True
        for url in article_urls[:cfg.max_articles]:
            result = fetch_page(url, 'article')
            if not result:
                continue
            raw, final, raw_path = result
            try:
                row = parse_announcement(raw, final, cfg, utc_now())
            except (ValueError, LxmlError) as exc:
                counts['failed_pages'] += 1; fail(url, 'article_parse', exc)
                report['pages'].append({'url': url, 'kind': 'article', 'status': 'failed'})
                continue
            counts['successful_pages'] += 1; counts['article_pages'] += 1
            report['pages'].append({'url': final, 'kind': 'article', 'status': 'success'})
            if row['within_requested_period'] is False:
                counts['outside_period'] += 1
                continue
            row['raw_path'] = raw_path
            row['source_urls'] = list(dict.fromkeys([final, url]))
            h = row['content_hash']
            if h in by_hash:
                original = by_hash[h]
                original['source_urls'] = list(dict.fromkeys(original['source_urls'] + row['source_urls']))
                report['duplicates'].append({'source_url': url, 'announcement_id': original['announcement_id'],
                                             'raw_path': raw_path, 'content_hash': h})
                counts['duplicate_content'] += 1
                continue
            if h in prior_hash:
                old = prior_hash[h]
                row['announcement_id'] = old['announcement_id']
                row['source_url'] = old['source_url']
                row['source_urls'] = list(dict.fromkeys(old.get('source_urls', [old['source_url']]) + row['source_urls']))
            by_hash[h] = row; records.append(row)
        counts['unknown_dates'] = sum(row['publication_date'] is None for row in records)
    counts['announcements'] = len(records)
    known = sorted(row['publication_date'] for row in records if row['publication_date'])
    report['observed_period'] = {'start': known[0] if known else None, 'end': known[-1] if known else None}
    report['status'] = 'partial' if report['errors'] and records else 'failed' if report['errors'] else 'success' if records else 'empty'
    report['finished_at'] = utc_now()
    store.save_run(cfg, records, report)
    return records, report
