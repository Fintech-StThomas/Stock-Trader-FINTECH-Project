"""PE-specific transport adapted from Lighthouse's rate/retry helper.

Reuse its per-host limiter and retry status policy, but never its SEC identity or
unrestricted automatic redirects. Each redirect is scope/robots checked before following.
"""
import os
import time
from urllib.parse import urlsplit, urljoin
from urllib.robotparser import RobotFileParser
import requests
from lighthouse.http import _RateLimiter, RETRY_STATUS, PROJECT_URL
from .config import canonical_url

USER_AGENT = f'LighthouseResearch/0.1 (+{PROJECT_URL})'
_limiter = _RateLimiter()


class CollectionError(RuntimeError):
    pass


class PoliteClient:
    def __init__(self, config, session=None):
        self.config = config
        self.session = session or requests.Session()
        self.delay = config.delay_seconds
        self.robots = None
        self.access = {}

    def _request(self, url):
        if os.getenv('LIGHTHOUSE_OFFLINE', '').lower() in {'1', 'true', 'yes'}:
            raise CollectionError('Offline mode: live requests are disabled')
        cfg = self.config
        for attempt in range(cfg.retries + 1):
            _limiter.wait(urlsplit(url).hostname, 1 / self.delay)
            try:
                with self.session.get(url, timeout=cfg.timeout_seconds, headers={'User-Agent': USER_AGENT},
                                      allow_redirects=False, stream=True) as response:
                    status = response.status_code
                    if status in RETRY_STATUS:
                        retry_after = response.headers.get('Retry-After', '')
                        if attempt == cfg.retries:
                            raise CollectionError(f'HTTP {status}; retry limit reached')
                        if retry_after.isdigit() and int(retry_after) > 60:
                            raise CollectionError('Server requests a longer pause; defer refresh')
                        time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                        continue
                    if status in {301, 302, 303, 307, 308}:
                        return status, '', response.headers.get('Location', '')
                    if not 200 <= status < 300:
                        raise CollectionError(f'HTTP {status}')
                    chunks, size = [], 0
                    for chunk in response.iter_content(chunk_size=65536):
                        size += len(chunk)
                        if size > 3_000_000:
                            raise CollectionError('Page exceeds the 3 MB input limit')
                        chunks.append(chunk)
                    return status, b''.join(chunks).decode(response.encoding or 'utf-8', errors='replace'), ''
            except requests.RequestException as exc:
                if attempt == cfg.retries:
                    raise CollectionError(f'{type(exc).__name__}; retry limit reached') from exc
                time.sleep(2 ** attempt)
        raise CollectionError('Request failed')

    def prepare(self):
        cfg = self.config
        if not cfg.enabled or cfg.access_status != 'internal_research':
            raise CollectionError('Source disabled or access review incomplete')
        origin = 'https://' + urlsplit(cfg.listing_url).netloc
        # Fail closed if robots cannot be checked, including unexpected redirects.
        status, raw, redirect = self._request(origin + '/robots.txt')
        if redirect or '<html' in raw.lower():
            raise CollectionError('Could not verify robots.txt; refresh stopped')
        robots = RobotFileParser(); robots.parse(raw.splitlines()); self.robots = robots
        self.delay = max(self.delay, float(robots.crawl_delay(USER_AGENT) or 0))
        rate = robots.request_rate(USER_AGENT)
        if rate:
            self.delay = max(self.delay, rate.seconds / rate.requests)
        if self.delay > 60:
            raise CollectionError('Source requests a longer interval; schedule a dedicated collection')
        self.access = {'robots_url': origin + '/robots.txt', 'robots_text': raw,
                       'delay_seconds': self.delay, 'terms_url': cfg.terms_url,
                       'access_reviewed_on': cfg.access_reviewed_on}

    def get(self, url):
        if self.robots is None:
            raise CollectionError('Run access checks before collecting pages')
        for _ in range(4):
            url = canonical_url(url)
            if not self.config.allows(url) or not self.robots.can_fetch(USER_AGENT, url):
                raise CollectionError('URL outside configured scope or disallowed by robots.txt')
            status, raw, redirect = self._request(url)
            if not redirect:
                return raw, url
            url = urljoin(url, redirect)
        raise CollectionError('Redirect limit reached')
