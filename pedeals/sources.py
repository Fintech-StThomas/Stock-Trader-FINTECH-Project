"""
The PE firms whose news pages are collected, and why these.

Scope (docs/PE_DEALS.md §2): lower-middle-market buy-and-build sponsors active in the five focus sectors (building &
facility services, IT services & cybersecurity, tech-enabled business services, industrial manufacturing, value-added
distribution), the competitors and counterparties of the product's user (a deal team sourcing in those sectors). Each publishes its deal announcements on
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
    # Added November 2026 for the deal-sourcing focus, after vetting (robots.txt without AI-crawler opt-outs, a
    # sitemap of the news, dated article pages): docs/PE_DEALS.md §2 lists the firms considered and why some were left out.
    Source("fortpoint", "Fort Point Capital", "https://fortpointcapital.com/", "Boston, MA",
           ("https://fortpointcapital.com/sitemap.xml",),
           r"^https://fortpointcapital\.com/news/[a-z0-9-]+/?$", exclude=r"/news/20\d\d-(?:year-)?review/?$"),
    Source("compass", "Compass Group Equity Partners", "https://www.cgep.com/", "St. Louis, MO",
           ("https://www.cgep.com/post-sitemap.xml",),
           r"^https://www\.cgep\.com/(?!blog/)[a-z0-9-]+/$", exclude=r"thesis-thursday|cornhole|meet-"),
    Source("shoreview", "ShoreView Industries", "https://www.shoreview.com/", "Minneapolis, MN",
           ("https://www.shoreview.com/sitemap.xml",),
           r"^https://www\.shoreview\.com/(?!(?:team|about|news|contact|privacy-policy|terms-and-conditions|portfolio|"
           r"casestudies|testimonial)(?:/|$))[a-z0-9-]+/$"),
    Source("cid", "CID Capital", "https://cidcap.com/", "Carmel, IN",
           ("https://cidcap.com/wp-sitemap-posts-news_release-1.xml",),
           r"^https://cidcap\.com/news/[a-z0-9-]+/$", exclude=r"/news/(?:newsinsight-1-headline-goes-here|587-2|435)/$",
           listings=("https://cidcap.com/news/", "https://cidcap.com/news/page/{page}/")),
    Source("masonwells", "Mason Wells", "https://www.masonwells.com/", "Milwaukee, WI",
           ("https://www.masonwells.com/sitemap.xml",),
           r"^https://www\.masonwells\.com/(?!(?:terms-of-use|contact|news|portfolio|criteria|approach|about|privacy-policy|"
           r"companies)/?$)[a-z0-9-]+/$", exclude=r"/mason-wells-newsletter-"),
    Source("gryphon", "Gryphon Investors", "https://www.gryphon-inv.com/", "San Francisco, CA",
           ("https://www.gryphon-inv.com/news-sitemap.xml",),
           r"^https://www\.gryphon-inv\.com/news/[^/]+/$"),
    Source("gridiron", "Gridiron Capital", "https://gridironcapital.com/", "New Canaan, CT",
           ("https://gridironcapital.com/post-sitemap.xml",),
           r"^https://gridironcapital\.com/(?!news-and-insights/)[^/]+/$"),
    Source("oridian", "Oridian Capital Partners", "https://oridiancapital.com/", "Washington, DC",
           ("https://oridiancapital.com/post-sitemap.xml",),
           r"^https://oridiancapital\.com/(?!news/$)[^/]+/$"),
    Source("azalea", "Azalea Capital", "https://azaleacapital.com/", "Greenville, SC",
           ("https://azaleacapital.com/post-sitemap.xml",),
           r"^https://azaleacapital\.com/(?!news/$)[^/]+/$"),
    Source("montage", "Montage Partners", "https://www.montagepartners.com/", "Scottsdale, AZ",
           ("https://www.montagepartners.com/post-sitemap.xml",),
           r"^https://www\.montagepartners\.com/[a-z0-9-]+/$"),
    Source("pikestreet", "Pike Street Capital Partners", "https://www.pikestreetcapital.com/", "Seattle, WA",
           ("https://www.pikestreetcapital.com/sitemap.xml",),
           r"^https://www\.pikestreetcapital\.com/news/[a-z0-9-]+$"),
    Source("resilience", "Resilience Capital Partners", "https://resiliencecapital.com/", "Cleveland, OH",
           ("https://resiliencecapital.com/news-sitemap.xml",),
           r"^https://resiliencecapital\.com/news-posts/[a-z0-9-]+/$"),
]}

# How a firm refers to itself in its own releases, beyond its full name (used to tell the sponsor from the buyer).
FIRM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "tonkabay": ("Tonka Bay Equity Partners", "Tonka Bay Equity", "Tonka Bay"),
    "goldnerhawn": ("Goldner Hawn Johnson & Morrison", "Goldner Hawn"),
    "huron": ("Huron Capital Partners", "Huron Capital", "Huron Flex Equity", "Huron Flex", "Huron"),
    "align": ("Align Capital Partners", "Align Capital", "Align", "ACP"),
    "fortpoint": ("Fort Point Capital", "Fort Point"),
    "compass": ("Compass Group Equity Partners", "Compass Group", "CGEP"),
    "shoreview": ("ShoreView Industries", "ShoreView Capital Partners", "ShoreView"),
    "cid": ("CID Capital", "CID"),
    "masonwells": ("Mason Wells",),
    "gryphon": ("Gryphon Investors", "Gryphon"),
    "gridiron": ("Gridiron Capital", "Gridiron"),
    "oridian": ("Oridian Capital Partners", "Oridian Capital", "Oridian", "HCI Equity Partners", "HCI Equity", "HCI"),
    "azalea": ("Azalea Capital", "Azalea"),
    "montage": ("Montage Partners", "Montage"),
    "pikestreet": ("Pike Street Capital Partners", "Pike Street Capital", "Pike Street"),
    "resilience": ("Resilience Capital Partners", "Resilience Capital", "Resilience"),
}
