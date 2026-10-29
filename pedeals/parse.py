"""
Parsing: a saved HTML page -> the article's headline, publication date, body text and boilerplate.

Generic on purpose (the firms' sites are different WordPress themes): the article body is the page container holding
the most paragraph text once navigation, headers, footers and forms are removed. Each output keeps how it was found
(`published_method`, `published_evidence`) so a reviewer can check it against the page.

Publication date, in order of trust: the page's structured metadata (article:published_time, JSON-LD datePublished,
a <time datetime>), then a press-release dateline in the opening text ("MINNETONKA, MN (July 7, 2026) —"), then the
first written date near the top of the body. It is the date the firm *published* the announcement: it says nothing
about when the transaction closed, which is a separate field (see extract.py). No date found -> None, never a guess.

Boilerplate: the "About <company>" paragraphs at the end of a release describe the sponsor and the companies in
general terms. They are kept apart from the announcement's own text (`about` sections), so that, for example, the
sponsor's sector focus is never read as the target's industry.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
_MONTH_RE = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
DATE_RE = re.compile(_MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I)
# "CITY, ST (Month D, YYYY) —", "CITY, ST – Month D, YYYY –", "CITY, Month D, YYYY /PRNewswire/ --"
DATELINE_RE = re.compile(r"^\s*([A-Z][A-Za-z.'\- ]{1,40}(?:,\s*[A-Za-z.\- ]{2,25})?)\s*[,(\-–—:]*\s*\(?\s*"
                         + _MONTH_RE + r"\s+(\d{1,2}),?\s+(\d{4})\s*\)?\s*(?:/[A-Za-z ]+/)?\s*[-–—:]+", re.S)
MONTH_YEAR_RE = re.compile(r"\b" + _MONTH_RE + r"\s+(\d{4})\b", re.I)
GENERIC_TITLE = re.compile(r"^\s*(?:recent |latest |in the )?(?:news|press|press releases|media|blog|insights|updates)\.?\s*$", re.I)
ABOUT_RE = re.compile(r"^\s*About\s+(.{2,80}?)\s*(?:\n|:|$)", re.I)
END_MARKERS = re.compile(r"^\s*(?:#{3}|-\s*\d+\s*-|view all news|back to news|media contacts?:?|contacts?:|for more information,? contact|"
                         r"share this|related news|recent news|previous post|next post)\b", re.I)


@dataclass
class Article:
    url: str
    title: str = ""
    published_date: Optional[str] = None     # ISO date
    published_method: str = "not found"      # metadata / dateline / first date in text / not found
    published_evidence: str = ""
    published_precision: str = ""            # day / month ("" when no date)
    modified_date: Optional[str] = None      # article:modified_time: an upper bound on the publication date, never the date
    dateline_location: str = ""              # where the release was issued (the publisher's city), not the target's
    body: str = ""                           # the announcement's own text, paragraphs separated by blank lines
    about: List[Tuple[str, str]] = field(default_factory=list)   # [(name, text)] boilerplate sections
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["about"] = json.dumps(self.about, ensure_ascii=False)
        return d


def to_date(month: str, day: str, year: str) -> Optional[date]:
    key = month.lower().rstrip(".")
    try:
        return date(int(year), MONTHS.get(key) or MONTHS[key[:3]], int(day))
    except (KeyError, ValueError):
        return None


def first_date(text: str) -> Optional[Tuple[date, str]]:
    m = DATE_RE.search(text or "")
    if not m:
        return None
    d = to_date(m.group(1), m.group(2), m.group(3))
    return (d, m.group(0)) if d else None


def _iso(value: str) -> Optional[date]:
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        try:
            return date.fromisoformat(value.strip()[:10])
        except (ValueError, AttributeError):
            return None


def _meta_date(soup) -> Optional[Tuple[date, str]]:
    for attr, name in (("property", "article:published_time"), ("name", "article:published_time"),
                       ("itemprop", "datePublished"), ("name", "date"), ("name", "pubdate")):
        tag = soup.find("meta", attrs={attr: name})
        if tag and tag.get("content") and _iso(tag["content"]):
            return _iso(tag["content"]), f'<meta {attr}="{name}" content="{tag["content"]}">'
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack += node
            elif isinstance(node, dict):
                if node.get("datePublished") and _iso(str(node["datePublished"])):
                    return _iso(str(node["datePublished"])), f'JSON-LD datePublished "{node["datePublished"]}"'
                stack += [v for v in node.values() if isinstance(v, (list, dict))]
    t = soup.find("time", attrs={"datetime": True})
    if t and _iso(t["datetime"]):
        return _iso(t["datetime"]), f'<time datetime="{t["datetime"]}">'
    return None


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" +([,.;:!?)\]’”]|'s\b)", r"\1", text)     # get_text(" ") puts a space before punctuation after a link
    text = re.sub(r"([(\[“]) +", r"\1", text)
    return text.strip()


def _paragraphs(node) -> List[str]:
    for br in node.find_all("br"):
        br.replace_with("\n")
    out = []
    blocks = node.find_all(["p", "li", "h2", "h3", "h4", "h5", "blockquote"])
    for b in blocks:
        if b.find_parent(["p", "li", "blockquote"]) is not None and b.name != "li":
            continue           # a <p> inside a <li>/<blockquote>: its parent already holds the text
        if b.name == "li" and b.find_parent("li") is not None:
            continue
        t = _clean(b.get_text(" "))
        if t:
            out.append(t)
    if not out:                # a theme that puts text in bare <div>s
        out = [p for p in (_clean(x) for x in node.get_text("\n").split("\n\n")) if p]
    return out


def _main_node(soup):
    for tag in soup.find_all(["nav", "header", "footer", "form", "aside", "button"]):
        tag.decompose()
    for sel in ("[class*=menu]", "[class*=breadcrumb]", "[class*=share]", "[class*=sidebar]", "[id*=sidebar]",
                "[class*=related]", "[class*=cookie]", "[class*=newsletter]", "[class*=footer]", "[class*=header]"):
        for tag in soup.select(sel):
            if tag.name not in ("html", "body", "main", "article"):
                tag.decompose()
    # the deepest container holding at least 90% of the page's paragraph text: containers that pass nest inside one
    # another (two disjoint ones cannot both hold most of the text), so the last one in document order is the deepest
    root = soup.body or soup
    total = sum(len(p.get_text(" ", strip=True)) for p in root.find_all("p"))
    best = root
    for c in root.find_all(["article", "main", "section", "div"]):
        if total and sum(len(p.get_text(" ", strip=True)) for p in c.find_all("p")) >= 0.9 * total:
            best = c
    return best


def _title(soup) -> str:
    for h1 in soup.find_all("h1"):           # skip the logo in a header and section titles such as "Recent News."
        if h1.find_parent(["header", "nav"]) is None and h1.get_text(strip=True) and not GENERIC_TITLE.match(h1.get_text(" ")):
            return _clean(h1.get_text(" "))
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return re.split(r"\s+[|–—-]\s+", og["content"].strip())[0]
    t = soup.find("title")
    return re.split(r"\s+[|–—-]\s+", t.get_text(strip=True))[0] if t else ""


def parse_article(html: str, url: str = "") -> Article:
    art = Article(url=url)
    if not html:
        art.error = "empty page"
        return art
    soup = BeautifulSoup(html, "lxml")
    meta = _meta_date(soup)
    mod = soup.find("meta", attrs={"property": "article:modified_time"})
    if mod and mod.get("content") and _iso(mod["content"]):
        art.modified_date = _iso(mod["content"]).isoformat()
    art.title = _title(soup)
    main = _main_node(soup)
    paras = _paragraphs(main)
    if paras and art.title and paras[0].strip().lower() == art.title.strip().lower():
        paras = paras[1:]

    body, about, current = [], [], None
    for p in paras:
        if END_MARKERS.match(p) and not about:
            break
        m = ABOUT_RE.match(p)
        if m and len(m.group(1).split()) <= 8:
            name = m.group(1).strip().rstrip(":")
            rest = p[m.end():].strip()
            current = [name, rest]
            about.append(current)
        elif current is not None:
            if END_MARKERS.match(p):
                break
            current[1] = (current[1] + "\n\n" + p).strip()
        else:
            body.append(p)
    art.body = "\n\n".join(body)
    art.about = [(n, t) for n, t in about]

    lead = art.body[:400]
    dl = DATELINE_RE.match(lead)
    if meta:
        art.published_date, art.published_method, art.published_evidence = meta[0].isoformat(), "page metadata", meta[1]
    elif dl and to_date(dl.group(2), dl.group(3), dl.group(4)):
        d = to_date(dl.group(2), dl.group(3), dl.group(4))
        art.published_date, art.published_method, art.published_evidence = d.isoformat(), "dateline", dl.group(0).strip()
    else:
        found = first_date(lead) or first_date(_clean(main.get_text(" "))[:600])
        if found:
            art.published_date, art.published_method, art.published_evidence = found[0].isoformat(), "first date in text", found[1]
    if dl:
        art.dateline_location = dl.group(1).strip(" ,(-–—")
    if art.published_date:
        art.published_precision = "day"
    if not art.body:
        art.error = "no article text found"
    return art


def listing_dates(html: str, base_url: str, is_article) -> List[dict]:
    """[{url, listed_date, precision, evidence}] from a news listing page: each article link with the date printed
    next to it. The date is read from the smallest block around the link that holds a date and links to no other
    article, so a neighbour's date is never borrowed. 'July 2026' gives the first of the month, precision 'month'."""
    from urllib.parse import urljoin
    soup = BeautifulSoup(html or "", "lxml")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"])
        if not is_article(url) or url in seen:
            continue
        node = a
        for _ in range(6):
            node = node.parent
            if node is None:
                break
            links = {urljoin(base_url, x["href"]) for x in node.find_all("a", href=True)}
            if len({u for u in links if is_article(u)}) > 1:
                break
            text = _clean(node.get_text(" "))
            full = DATE_RE.search(text)
            if full and to_date(full.group(1), full.group(2), full.group(3)):
                d = to_date(full.group(1), full.group(2), full.group(3))
                out.append({"url": url, "listed_date": d.isoformat(), "precision": "day", "evidence": full.group(0)})
                seen.add(url)
                break
            my = MONTH_YEAR_RE.search(text)
            if my and to_date(my.group(1), "1", my.group(2)):
                d = to_date(my.group(1), "1", my.group(2))
                out.append({"url": url, "listed_date": d.isoformat(), "precision": "month", "evidence": my.group(0)})
                seen.add(url)
                break
    return out
