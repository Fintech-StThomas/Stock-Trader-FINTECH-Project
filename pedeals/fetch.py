"""
Polite HTTP for the PE firms' websites.

* robots.txt is read once per host and obeyed: a disallowed page is never requested, and the host's Crawl-delay (10 s
  for several of these sites) is the minimum gap between two requests to it; hosts without one get DEFAULT_DELAY.
* The User-Agent names the tool and links to the project, so a site owner can see who is reading and why.
* Failures are returned, not raised: the collector records every failed page with its reason (coverage report).
* Offline mode (LIGHTHOUSE_OFFLINE=1 or PEDEALS_OFFLINE=1) refuses every request, so tests never touch the network.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
from urllib import robotparser
from urllib.parse import urlparse

USER_AGENT = "Lighthouse-PE-Deals/1.0 (+https://github.com/lotainc0/Stock-Trader-FINTECH-Project)"
DEFAULT_DELAY = 5.0          # seconds between requests to one host that states no Crawl-delay
TIMEOUT = 30.0


def offline_mode() -> bool:
    return any(os.environ.get(v, "").strip().lower() in {"1", "true", "yes"}
               for v in ("LIGHTHOUSE_OFFLINE", "PEDEALS_OFFLINE"))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Fetched:
    url: str
    final_url: str
    status: Optional[int]       # None: no answer (offline, robots, network error)
    text: str
    fetched_at: str             # UTC, ISO 8601
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300 and not self.error


class PoliteFetcher:
    """One fetcher per run; safe to share between threads (one thread per host is the intended use)."""

    def __init__(self, user_agent: str = USER_AGENT, default_delay: float = DEFAULT_DELAY, retries: int = 2):
        self.user_agent = user_agent
        self.default_delay = default_delay
        self.retries = retries
        self._robots: Dict[str, Optional[robotparser.RobotFileParser]] = {}
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._session = None

    # ------------------------------------------------------------------ robots.txt
    def _robots_for(self, url: str) -> Optional[robotparser.RobotFileParser]:
        p = urlparse(url)
        host = f"{p.scheme}://{p.netloc}"
        with self._lock:
            if host in self._robots:
                return self._robots[host]
        rp: Optional[robotparser.RobotFileParser] = robotparser.RobotFileParser()
        res = self._request(f"{host}/robots.txt", check_robots=False)
        if res.status is not None and 200 <= res.status < 300:
            rp.parse(res.text.splitlines())
        elif res.status in (401, 403):
            rp.parse(["User-agent: *", "Disallow: /"])     # robots.txt itself refused: treat the site as closed
        else:
            rp = None                                       # no robots.txt: everything allowed, default delay
        with self._lock:
            self._robots[host] = rp
        return rp

    def allowed(self, url: str) -> bool:
        rp = self._robots_for(url)
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    def delay_for(self, url: str) -> float:
        rp = self._robots_for(url)
        stated = rp.crawl_delay(self.user_agent) if rp is not None else None
        return max(float(stated or 0), self.default_delay)

    # ------------------------------------------------------------------ requests
    def _wait(self, host: str, delay: float) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._last.get(host, -1e9) + delay)
            self._last[host] = slot
        if slot > now:
            time.sleep(slot - now)

    def _request(self, url: str, check_robots: bool = True) -> Fetched:
        if offline_mode():
            return Fetched(url, url, None, "", utc_now(), "offline mode: not fetched")
        if check_robots and not self.allowed(url):
            return Fetched(url, url, None, "", utc_now(), "disallowed by robots.txt")
        import requests
        if self._session is None:
            self._session = requests.Session()
        host = urlparse(url).netloc
        delay = self.delay_for(url) if check_robots else self.default_delay
        err = ""
        for attempt in range(self.retries + 1):
            self._wait(host, delay)
            try:
                r = self._session.get(url, headers={"User-Agent": self.user_agent}, timeout=TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - network trouble: retry, then report
                err = f"{type(exc).__name__}: {str(exc)[:160]}"
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                err = f"HTTP {r.status_code}"
                continue
            r.encoding = r.encoding or r.apparent_encoding
            return Fetched(url, r.url, r.status_code, r.text, utc_now(),
                           "" if r.ok else f"HTTP {r.status_code}")
        return Fetched(url, url, None, "", utc_now(), err or "no answer")

    def get(self, url: str) -> Fetched:
        return self._request(url)
