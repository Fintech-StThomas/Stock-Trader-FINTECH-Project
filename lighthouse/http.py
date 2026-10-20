"""
Polite HTTP for the free data sources: one place that knows each host's rules.

* Every request carries the project's contact e-mail (CONTACT_EMAIL, lota.inc0@gmail.com; LIGHTHOUSE_SEC_EMAIL
  overrides it) in the standard `From` header, so no host ever has to guess who is calling.
* SEC EDGAR asks for a descriptive User-Agent with a contact e-mail and at most 10 requests per second; it answers
  403 "Request Rate Threshold Exceeded" (or 429) when a client goes faster. The User-Agent carries the contact e-mail
  (SEC_USER_AGENT replaces the whole string), so EDGAR, the analyst memo and the fundamental score always load.
* FRED's CDN stalls on custom or browser-like User-Agents and answers requests' default in under a second, so FRED
  (and the other public CSV hosts) keep the library default User-Agent (with the `From` header).
* Wikipedia asks for a User-Agent that identifies the tool and a way to reach its operator: the project URL and the
  contact e-mail.

Every request is rate-limited per host (thread-safe, so parallel loaders share the budget), retried with exponential
backoff on throttling and server errors (honouring Retry-After), and refused outright in offline mode
(LIGHTHOUSE_OFFLINE=1), so tests can never reach the network by accident.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

from .data import DataUnavailable, offline_mode

PROJECT_URL = "https://github.com/lotainc0/Stock-Trader-FINTECH-Project"
DEFAULT_CONTACT_EMAIL = "lota.inc0@gmail.com"


def contact_email() -> str:
    """The e-mail every request carries: LIGHTHOUSE_SEC_EMAIL when it holds an address, else the project's."""
    email = os.environ.get("LIGHTHOUSE_SEC_EMAIL", "").strip()
    return email if "@" in email else DEFAULT_CONTACT_EMAIL


def project_user_agent() -> str:
    return f"Lighthouse/1.3 (+{PROJECT_URL}; {contact_email()})"


def contact_headers() -> Dict[str, str]:
    """Headers for requests made outside `fetch` (the Stooq and FRED CSVs): the contact e-mail, nothing else."""
    return {"From": contact_email()}

RETRY_STATUS = {429, 500, 502, 503, 504}


class ConfigurationError(RuntimeError):
    """A setting the user must provide is missing (kept for callers; the SEC contact e-mail now has a default)."""


class HTTPStatusError(DataUnavailable):
    """The host answered, with a status that retrying does not change (404, 403 without a rate-limit message, ...):
    for most hosts a definite "there is no such file". A timeout, a dropped connection or a throttling or server
    error that lasted through every retry raises a plain DataUnavailable instead (a transient failure)."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = int(status)


def sec_user_agent() -> str:
    """The User-Agent the SEC requires: a name and a contact e-mail (never missing: see `contact_email`)."""
    custom = os.environ.get("SEC_USER_AGENT", "").strip()
    if custom and "@" in custom:
        return custom
    return f"Lighthouse research tool {contact_email()}"


@dataclass
class HostPolicy:
    """How to talk to one host: requests per second and the User-Agent to send (None: the library default)."""
    rate: float
    user_agent: Optional[Callable[[], str]] = None
    accept_encoding: str = "gzip, deflate"


# Host suffix -> policy. The SEC's published limit is 10 requests/second; 8 leaves room for clock jitter. Several
# processes downloading at once must split it: LIGHTHOUSE_SEC_RATE lowers this process's share (never above 8).
SEC_RATE = min(float(os.environ.get("LIGHTHOUSE_SEC_RATE", "8") or 8), 8.0)
POLICIES: Dict[str, HostPolicy] = {
    "sec.gov": HostPolicy(rate=SEC_RATE, user_agent=sec_user_agent),
    "wikipedia.org": HostPolicy(rate=2.0, user_agent=project_user_agent),
    "finra.org": HostPolicy(rate=4.0, user_agent=project_user_agent),
    "dartmouth.edu": HostPolicy(rate=2.0, user_agent=None),
    "stlouisfed.org": HostPolicy(rate=2.0, user_agent=None),
    "githubusercontent.com": HostPolicy(rate=4.0, user_agent=None),
    "nd.edu": HostPolicy(rate=1.0, user_agent=project_user_agent),
    "alpaca.markets": HostPolicy(rate=3.0, user_agent=None),     # paper trading: 200 requests/minute per key
    "localhost": HostPolicy(rate=50.0, user_agent=None),         # a local Ollama server (optional text reading)
}
DEFAULT_POLICY = HostPolicy(rate=4.0, user_agent=None)


def policy_for(url: str) -> HostPolicy:
    host = (urlparse(url).hostname or "").lower()
    for suffix, pol in POLICIES.items():
        if host == suffix or host.endswith("." + suffix):
            return pol
    return DEFAULT_POLICY


class _RateLimiter:
    """At most `rate` calls per second per host, shared by every thread."""

    def __init__(self) -> None:
        self._next: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str, rate: float) -> None:
        interval = 1.0 / max(rate, 1e-6)
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next.get(host, 0.0))
            self._next[host] = slot + interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


_limiter = _RateLimiter()
_session_local = threading.local()


def _session():
    import requests
    s = getattr(_session_local, "session", None)
    if s is None:
        s = requests.Session()
        _session_local.session = s
    return s


def _is_throttle(resp) -> bool:
    if resp.status_code in RETRY_STATUS:
        return True
    # the SEC answers 403 with this text when a client exceeds 10 requests/second
    return resp.status_code == 403 and "rate threshold" in (resp.text or "")[:2000].lower()


def fetch(url: str, params: Optional[dict] = None, *, timeout: float = 30.0, retries: int = 4, stream: bool = False,
          headers: Optional[dict] = None, method: str = "GET", json_body: Optional[dict] = None):
    """GET (or POST) `url` under its host's policy and return the `requests.Response` (status 2xx).

    Raises DataUnavailable offline or after the retries, HTTPStatusError (a DataUnavailable with the `status`) for a
    non-retryable HTTP error."""
    if offline_mode():
        raise DataUnavailable(f"offline mode (LIGHTHOUSE_OFFLINE=1): not fetching {url}")
    pol = policy_for(url)
    host = (urlparse(url).hostname or "").lower()
    h = {"Accept-Encoding": pol.accept_encoding, **contact_headers()}
    if pol.user_agent is not None:
        h["User-Agent"] = pol.user_agent()
    h.update(headers or {})
    last_err = ""
    for attempt in range(retries + 1):
        _limiter.wait(host, pol.rate)
        try:
            resp = _session().request(method, url, params=params, headers=h, timeout=timeout, stream=stream,
                                      json=json_body)
        except Exception as exc:  # noqa: BLE001 - network trouble: retry, then report
            last_err = f"{type(exc).__name__}: {str(exc)[:160]}"
        else:
            if 200 <= resp.status_code < 300:
                return resp
            if not _is_throttle(resp):
                raise HTTPStatusError(f"HTTP {resp.status_code} from {host} for {url}: {(resp.text or '')[:160]}",
                                      resp.status_code)
            last_err = f"HTTP {resp.status_code} (throttled)"
            retry_after = resp.headers.get("Retry-After")
            resp.close()
            if retry_after and retry_after.isdigit() and attempt < retries:
                time.sleep(min(float(retry_after), 60.0))
                continue
        if attempt < retries:
            time.sleep(min(2.0 ** attempt, 30.0))       # 1, 2, 4, 8 s
    raise DataUnavailable(f"{host}: gave up after {retries + 1} attempts ({last_err}) for {url}")
