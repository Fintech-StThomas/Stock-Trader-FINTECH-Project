"""
The PE firms whose news pages are collected, and why these.

Scope (docs/PE_DEALS.md §2): lower-middle-market sponsors in the Midwest and the central US that buy and build
industrial manufacturing, value-added distribution and business / IT services companies, the peer set of the
product's user (a portfolio operations manager at a sponsor of that kind). Each publishes its deal announcements on
its own site and lists them in an XML sitemap, which is how the collector finds every article without crawling menus.

Firms were kept only if their robots.txt allows the news pages to generic crawlers; sites that block automated or AI
agents by name were left out even where a generic rule would allow them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Source:
    key: str                 # short id used in file names and the `source` column
    firm: str                # the publishing PE firm
    homepage: str
    hq: str                  # the firm's home office (from its site), for context only
    sitemaps: Tuple[str, ...]     # sitemaps that list the news articles
    include: str             # regex an article URL must match (listing pages and other sections do not)
    exclude: str = r"$^"     # regex of URLs to leave out even when `include` matches
    # news listing pages that show each article's date, for sites whose article pages carry none; "{year}" and
    # "{page}" are filled in (every year of the period, pages 1, 2, ... until one lists no article)
    listings: Tuple[str, ...] = ()

    def is_article(self, url: str) -> bool:
        return bool(re.search(self.include, url)) and not re.search(self.exclude, url)


SOURCES: Dict[str, Source] = {s.key: s for s in [
    Source("tonkabay", "Tonka Bay Equity Partners", "https://tonkabayequity.com/", "Minnetonka, MN",
           ("https://tonkabayequity.com/wp-sitemap-posts-news-1.xml",),
           r"^https://tonkabayequity\.com/news/[^/]+/?$"),
    Source("goldnerhawn", "Goldner Hawn", "https://goldnerhawn.com/", "Minneapolis, MN",
           ("https://goldnerhawn.com/wp-sitemap-posts-news-1.xml",),
           r"^https://goldnerhawn\.com/news/[^/]+/?$",
           listings=("https://goldnerhawn.com/news/",)),           # its article pages show no date
    Source("huron", "Huron Capital", "https://www.huroncapital.com/", "Detroit, MI",
           ("https://www.huroncapital.com/news-sitemap.xml",),
           r"^https://www\.huroncapital\.com/news/(?!news-category/)[^/]+/?$",
           # its article pages show no date; the news page shows the month of its five latest items (robots.txt
           # disallows the paginated archive, so older undated articles stay undated)
           listings=("https://www.huroncapital.com/news/",)),
    Source("align", "Align Capital Partners", "https://aligncp.com/", "Dallas, TX and Cleveland, OH",
           ("https://aligncp.com/post-sitemap.xml",),
           r"^https://aligncp\.com/news/[^/]+/?$",
           exclude=r"/news/align-hiring-"),          # job postings, published as news posts
]}

# How a firm refers to itself in its own releases, beyond its full name (used to tell the sponsor from the buyer).
FIRM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "tonkabay": ("Tonka Bay Equity Partners", "Tonka Bay Equity", "Tonka Bay"),
    "goldnerhawn": ("Goldner Hawn Johnson & Morrison", "Goldner Hawn"),
    "huron": ("Huron Capital Partners", "Huron Capital", "Huron"),
    "align": ("Align Capital Partners", "Align Capital", "Align", "ACP"),
}
