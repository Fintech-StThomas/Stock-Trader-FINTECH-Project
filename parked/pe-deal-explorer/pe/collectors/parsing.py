"""Parse publisher HTML as untrusted data, never execute embedded JavaScript."""
import hashlib
import json
import re
from datetime import date, datetime
from urllib.parse import urljoin, urlsplit
from lxml import html, etree
from .config import canonical_url


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def text(node):
    return ' '.join((' '.join(node.itertext()) if hasattr(node, 'itertext') else str(node)).split())


def publication_date(value):
    value = ' '.join(value.split())
    # Only parse a date within the dedicated publication-date field, never the article body.
    for pattern, fmt in [(r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),
                         (r'[A-Z][a-z]+ \d{1,2}, \d{4}', '%B %d, %Y')]:
        match = re.search(pattern, value)
        if match:
            try:
                return datetime.strptime(match.group(), fmt).date().isoformat()
            except ValueError:
                pass
    return None


def parse_listing(raw, url, config):
    root = html.fromstring(raw)
    links = root.xpath('//a/@href')
    if config.listing_kind == 'riverside_embedded':
        # Riverside embeds its current listing in rcCurrentsData.push objects.
        # Read only JSON string URL values; no eval, browser execution or body extraction.
        for script in root.xpath('//script/text()'):
            if 'rcCurrentsData.push' in script:
                for match in re.finditer(r'\burl\s*:\s*("(?:\\.|[^"\\])*")', script):
                    links.append(json.loads(match.group(1)))
    articles = []
    for href in links:
        try:
            target = canonical_url(urljoin(url, href))
        except ValueError:
            continue
        p = urlsplit(target)
        if (config.allows(target) and p.path.startswith(config.article_prefix)
                and p.path.rstrip('/') != urlsplit(config.listing_url).path.rstrip('/')
                and not p.path.endswith(('.pdf', '.xml', '.jpg', '.png'))):
            articles.append(target)
    next_pages = []
    for href in root.xpath(config.next_xpath):
        target = canonical_url(urljoin(url, str(href)))
        if config.allows(target):
            next_pages.append(target)
    return articles, list(dict.fromkeys(next_pages))


def parse_announcement(raw, url, config, collected_at):
    root = html.fromstring(raw)
    titles, bodies = root.xpath(config.title_xpath), root.xpath(config.body_xpath)
    if not titles or not bodies:
        raise ValueError('Article layout missing headline or body; review the source adapter')
    title = text(titles[0])
    dates = root.xpath(config.date_xpath)
    published = publication_date(text(dates[0])) if dates else None
    body = bodies[0]
    for junk in body.xpath('.//script|.//style|.//nav|.//footer|.//form|.//noscript'):
        junk.drop_tree()
    paragraphs = [text(n) for n in body.xpath('.//p[not(ancestor::li)]|.//li[not(.//li)]|.//h2|.//h3') if text(n)]
    full_text = '\n\n'.join(paragraphs) if paragraphs else text(body)
    if not title or len(full_text) < 120:
        raise ValueError('Article text too short; refusing a likely empty/error page')
    source_url = canonical_url(url)
    return {
        'announcement_id': 'ann_' + digest(config.source_id + ':' + source_url)[:24],
        'source_id': config.source_id, 'publisher_pe_firm': config.publisher_pe_firm,
        'source_url': source_url, 'source_urls': [source_url], 'headline': title,
        'publication_date': published, 'collected_at': collected_at,
        'full_text': full_text, 'content_hash': digest(' '.join(full_text.split())),
        'collection_status': 'success', 'is_cached': False,
        'date_status': 'known' if published else 'unknown',
        'within_requested_period': None if published is None else config.start_date <= published <= config.end_date,
    }
