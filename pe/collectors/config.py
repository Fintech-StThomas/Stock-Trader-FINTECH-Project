"""Source configuration owned by collection; announcement dictionaries follow the team plan."""
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, unquote
import json
import re

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'sources.json'


def canonical_url(url):
    p = urlsplit(url)
    if p.scheme != 'https' or not p.hostname or p.username or p.password or p.port not in (None, 443):
        raise ValueError('Only ordinary HTTPS source URLs are accepted')
    decoded_path = unquote(p.path)
    if any(part in {'.', '..'} for part in decoded_path.split('/')) or '\\' in decoded_path:
        raise ValueError('Unsafe source path')
    if any(ord(c) < 32 for c in url):
        raise ValueError('Control characters in URL')
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in {'fbclid', 'gclid'}]
    return urlunsplit(('https', p.hostname.lower(), p.path or '/', urlencode(sorted(query)), ''))


@dataclass
class SourceConfig:
    source_id: str
    publisher_pe_firm: str
    listing_url: str
    article_prefix: str
    body_xpath: str
    date_xpath: str
    next_xpath: str = '//a[@rel="next"]/@href'
    title_xpath: str = '//h1'
    listing_kind: str = 'html'
    enabled: bool = False
    access_status: str = 'needs_review'
    terms_url: str = ''
    access_note: str = ''
    access_reviewed_on: str = ''
    delay_seconds: float = 2.0
    timeout_seconds: float = 20.0
    retries: int = 2
    max_pages: int = 3
    max_articles: int = 20
    start_date: str = '2026-01-01'
    end_date: str = '2026-11-03'

    def __post_init__(self):
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,50}', self.source_id):
            raise ValueError('Invalid source ID')
        self.listing_url = canonical_url(self.listing_url)
        if not self.article_prefix.startswith('/') or self.article_prefix == '/':
            raise ValueError('An explicit article path prefix is required')
        if date.fromisoformat(self.start_date) > date.fromisoformat(self.end_date):
            raise ValueError('Start date must not follow end date')
        if not 1 <= self.delay_seconds <= 60 or not 1 <= self.timeout_seconds <= 60:
            raise ValueError('Delay and timeout must be bounded')
        if not 0 <= self.retries <= 3 or not 1 <= self.max_pages <= 30 or not 1 <= self.max_articles <= 100:
            raise ValueError('Collection limits exceed the supported range')
        if self.listing_kind not in {'html', 'riverside_embedded'}:
            raise ValueError('Unknown listing adapter')

    def allows(self, url):
        try:
            p, base = urlsplit(canonical_url(url)), urlsplit(self.listing_url)
        except ValueError:
            return False
        return p.hostname == base.hostname and (
            p.path.rstrip('/') == base.path.rstrip('/') or p.path.startswith(self.article_prefix))

    def to_dict(self):
        return asdict(self)


def load_sources(path=CONFIG_PATH):
    configs = [SourceConfig(**row) for row in json.loads(Path(path).read_text())]
    if len({s.source_id for s in configs}) != len(configs):
        raise ValueError('Source IDs must be unique')
    return configs
