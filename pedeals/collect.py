"""
Collection: find every news article a firm lists in its sitemap and save the pages.

    python -m pedeals collect --start 2023-01-01            # every firm, polite (a few minutes per firm)
    python -m pedeals collect --firm tonkabay --refresh     # one firm, re-fetching pages already saved

The saved pages (data/pe_deals/cache/pages.jsonl.gz) are the program's input: parsing and extraction always run on
them, so a run is reproducible and still works when a site changes or goes down. Every line carries the URL, the
time it was fetched (UTC) and the HTTP status; scripts, styles and inline images are stripped from the HTML to keep
the file small, the text is untouched. Two sites print no date on their article pages; their news listing pages,
which do, are saved too (listings.jsonl.gz) and give those articles their dates. data/pe_deals/cache/discovery.csv lists every URL the sitemaps offered and
what happened to it (fetched, already saved, outside the period, disallowed, failed), the coverage record.

The period filter uses the sitemap's <lastmod>, the date the page was last modified: a page modified before the start
date was necessarily published before it, so skipping it is safe. Pages modified later are fetched and filtered by
the publication date read from the page itself (pedeals.parse), since sites re-save old pages (a site migration)
and <lastmod> says nothing about when a release was first published.
"""
from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .fetch import PoliteFetcher, offline_mode, utc_now
from .sources import SOURCES, Source

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "pe_deals" / "cache"
PAGES_FILE = CACHE_DIR / "pages.jsonl.gz"
LISTINGS_FILE = CACHE_DIR / "listings.jsonl.gz"      # news listing pages (the dates of articles that show none)
DISCOVERY_FILE = CACHE_DIR / "discovery.csv"
MAX_LISTING_PAGES = 15
DEFAULT_START = "2023-01-01"

_STRIP = re.compile(r"<(script|style|svg|noscript|iframe)\b.*?</\1\s*>", re.S | re.I)
_DATA_URI = re.compile(r"""(src|srcset|href)=(["'])data:[^"']*\2""", re.I)


def slim_html(html: str) -> str:
    return _DATA_URI.sub(r"\1=\2\2", _STRIP.sub("", html or ""))


# ----------------------------------------------------------------------------- the saved pages
def load_pages(path: Path = PAGES_FILE) -> Dict[str, dict]:
    """url -> saved page record. Empty when nothing was saved yet."""
    out: Dict[str, dict] = {}
    if Path(path).exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out[rec["url"]] = rec
    return out


def save_pages(pages: Dict[str, dict], path: Path = PAGES_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: the same pages give the same bytes, so an unchanged cache never shows up as a change in git
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for url in sorted(pages):
            gz.write((json.dumps(pages[url], ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))


# ----------------------------------------------------------------------------- discovery
def parse_sitemap(xml: str) -> List[dict]:
    """[{url, lastmod}] from a sitemap (or the child sitemaps of a sitemap index, flagged `index`)."""
    rows = []
    for block in re.findall(r"<(?:url|sitemap)>(.*?)</(?:url|sitemap)>", xml or "", re.S):
        loc = re.search(r"<loc>\s*(.*?)\s*</loc>", block, re.S)
        if not loc:
            continue
        mod = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", block, re.S)
        rows.append({"url": loc.group(1).replace("&amp;", "&"), "lastmod": mod.group(1)[:10] if mod else ""})
    return rows


def discover(source: Source, fetcher: PoliteFetcher) -> tuple:
    """(article rows [{url, lastmod}], problems [str]) from the source's sitemaps."""
    rows, problems = [], []
    for sm in source.sitemaps:
        res = fetcher.get(sm)
        if not res.ok:
            problems.append(f"{sm}: {res.error}")
            continue
        rows += [r for r in parse_sitemap(res.text) if source.is_article(r["url"])]
    seen, out = set(), []
    for r in rows:
        if r["url"] not in seen:
            seen.add(r["url"])
            out.append(r)
    return out, problems


# ----------------------------------------------------------------------------- collection
def collect_source(source: Source, fetcher: PoliteFetcher, pages: Dict[str, dict], start: str,
                   refresh: bool = False, max_pages: Optional[int] = None, log=print) -> List[dict]:
    """Fetch the source's articles into `pages` (in place); returns its discovery rows."""
    found, problems = discover(source, fetcher)
    for p in problems:
        log(f"[{source.key}] sitemap problem: {p}")
    log(f"[{source.key}] {len(found)} article URLs in the sitemap; crawl delay {fetcher.delay_for(source.homepage):.0f}s")
    disc, fetched = [], 0
    for r in found:
        row = {"source": source.key, "firm": source.firm, "url": r["url"], "lastmod": r["lastmod"], "checked_at": utc_now()}
        old = pages.get(r["url"])
        if r["lastmod"] and r["lastmod"] < start:
            row["outcome"] = "skipped: modified before the period start"
        elif old is not None and old.get("status") == 200 and not refresh:
            row["outcome"] = "already saved"
        elif not fetcher.allowed(r["url"]):
            row["outcome"] = "skipped: disallowed by robots.txt"
        elif max_pages is not None and fetched >= max_pages:
            row["outcome"] = "skipped: page limit for this run"
        else:
            res = fetcher.get(r["url"])
            fetched += 1
            if res.ok:
                pages[r["url"]] = {"source": source.key, "url": r["url"], "final_url": res.final_url, "status": res.status,
                                   "fetched_at": res.fetched_at, "lastmod": r["lastmod"], "html": slim_html(res.text)}
                row["outcome"] = "fetched"
            else:
                row["outcome"] = f"failed: {res.error}"
                if old is None:      # keep a record of the failure; never overwrite a good saved copy with one
                    pages[r["url"]] = {"source": source.key, "url": r["url"], "final_url": res.final_url,
                                       "status": res.status, "fetched_at": res.fetched_at, "lastmod": r["lastmod"],
                                       "html": "", "error": res.error}
            if fetched % 10 == 0:
                log(f"[{source.key}] {fetched} pages fetched ...")
        disc.append(row)
    log(f"[{source.key}] done: {fetched} fetched")
    return disc


def collect_listings(source: Source, fetcher: PoliteFetcher, listings: Dict[str, dict], start: str, log=print) -> int:
    """Fetch the source's news listing pages into `listings` (in place); returns how many were fetched."""
    from .parse import listing_dates
    n = 0
    years = range(int(start[:4]), date.today().year + 1)
    for template in source.listings:
        for year in (years if "{year}" in template else [None]):
            for page in (range(1, MAX_LISTING_PAGES + 1) if "{page}" in template else [None]):
                url = template.format(year=year, page=page)
                res = fetcher.get(url)
                n += 1
                if not res.ok:
                    if page in (None, 1):
                        log(f"[{source.key}] listing {url}: {res.error}")
                    break
                found = listing_dates(res.text, url, source.is_article)
                listings[url] = {"source": source.key, "url": url, "final_url": res.final_url, "status": res.status,
                                 "fetched_at": res.fetched_at, "html": slim_html(res.text)}
                if not found:
                    break
    log(f"[{source.key}] {n} listing pages fetched")
    return n


def collect(start: str = DEFAULT_START, firms: Optional[Iterable[str]] = None, refresh: bool = False,
            max_pages: Optional[int] = None, pages_file: Path = PAGES_FILE, discovery_file: Path = DISCOVERY_FILE,
            listings_file: Path = LISTINGS_FILE, log=print) -> pd.DataFrame:
    """Collect every firm (in parallel: one thread per firm, each polite to its own host) and save the pages.
    Returns the discovery / coverage table."""
    if offline_mode():
        raise RuntimeError("offline mode: unset LIGHTHOUSE_OFFLINE / PEDEALS_OFFLINE to collect live pages")
    start = str(date.fromisoformat(str(start)))
    keys = list(firms) if firms else list(SOURCES)
    unknown = [k for k in keys if k not in SOURCES]
    if unknown:
        raise KeyError(f"unknown firm(s) {unknown}; choose from {sorted(SOURCES)}")
    pages = load_pages(pages_file)
    listings = load_pages(listings_file)
    fetcher = PoliteFetcher()

    def one(k: str) -> List[dict]:
        rows = collect_source(SOURCES[k], fetcher, pages, start, refresh, max_pages, log)
        collect_listings(SOURCES[k], fetcher, listings, start, log)
        return rows
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        parts = list(pool.map(one, keys))
    save_pages(pages, pages_file)
    save_pages(listings, listings_file)
    disc = pd.DataFrame([r for part in parts for r in part])
    if Path(discovery_file).exists():          # keep the other firms' rows when only some were collected
        old = pd.read_csv(discovery_file, dtype=str).fillna("")
        disc = pd.concat([old[~old["source"].isin(keys)], disc], ignore_index=True)
    disc = disc.sort_values(["source", "url"]).reset_index(drop=True)
    disc.to_csv(discovery_file, index=False)
    return disc
