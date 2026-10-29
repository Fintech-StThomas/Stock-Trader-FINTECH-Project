"""
Extraction: a parsed article -> one deal record, every important field with the sentence it came from.

Rules, not a model: each field is read by patterns a reviewer can follow (and argue with), its evidence is kept, and a
value the text does not support is left unknown rather than guessed. The rules work in this order:

1. **Is it a deal?** The headline first: a non-deal signal (promoted, joins, named, award, podcast, fund close ...)
   decides unless the headline also has a clear deal verb (acquires, invests in, sale of, joins forces, add-on ...).
   A headline with neither, or only a weak word (forms, launches, platform, exits), is decided by the opening text.
   The headline goes first because releases about people or funds often *mention* past deals.
2. **Who bought what** comes from the release's lead sentence, which follows a formula even when the headline is
   playful ("Marco Expands into Brewery End Market with Add-on of Jet Gasket & Seal"): *[sponsor's] [platform]
   [Buyer] (“Short name”), [a description], has acquired [Target] (“Short name”), [a description of the target]*.
   The sentence is split at its deal verb; the buyer is the last company name before it, the target the first
   after it; parenthesised short names are kept as aliases. Headline patterns ("X acquires Y", "Firm announces the
   sale of Y to Z") are the fallback. A name that is the publisher's own (Align Capital Partners, ACP) is the
   publisher; a buyer introduced with the sponsor's name or as its portfolio / partner / platform company is a
   company the publisher owns.
3. **Transaction type** follows from the verb and from who the buyer is (the publisher -> platform investment; a
   company it owns -> add-on), unless the text names it (sale -> exit, combination -> merger, minority or growth
   investment, dividend recapitalization).
4. **The target's business** is read from the words that describe the target (the appositive after its name in the
   lead sentence, "ATD designs and manufactures ...", or its own "About" section), never from the sponsor's
   boilerplate. Its industry is the taxonomy category with the most stems in that description; what the target's
   *customers* do (used in industrial applications, serving hospitals) goes to `customer_markets` instead.
5. **Rationale, price, status.** The sentences that say why (expands, broadens, adds, strengthens ...), tagged with
   the capability they name; a price only if a dollar amount is tied to the transaction (otherwise "Not disclosed
   (stated)" when the release says so, else "Not stated", never zero); the status the verbs support (has acquired
   -> completed, definitive agreement / will acquire -> announced). The publication date is never a closing date.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple

from .parse import Article, DATE_RE, to_date
from .sources import FIRM_ALIASES, SOURCES
from .taxonomy import AMBIGUOUS_CITIES, CITY_STATE, END_MARKETS, INDUSTRIES, RATIONALE_TAGS, STATE_ABBR, US_STATES, region_of

UNKNOWN = "Unknown"

# ----------------------------------------------------------------------------- 1. vocabulary: is it a deal?
STRONG_DEAL = re.compile(r"\b(acqui\w*|invest(?:s|ed|ing)? in|investment in|sale of|sells?|sold|exits|exited|merge[sd]?|merger|"
                         r"recapitali[sz]\w*|join(?:s|ed)? forces|partners? with|partnership with|add-?ons?|"
                         r"combin(?:e|es|ed|ation)|divest\w*)\b", re.I)
WEAK_DEAL = re.compile(r"\b(exit|forms?|backs|launch\w*|platform|adds?|unite[sd]?|purchases?|buys|"
                       r"majority|minority|realiz\w*)\b", re.I)
NON_DEAL = {
    "Recognition / media": re.compile(r"\b(award\w*|honou?r\w*|rising star|ranked|ranking|recogni[sz]\w*|list|podcast|"
                                      r"featured|interview\w*|webinar|panel|conference|speaks?|article|op-ed|video|"
                                      r"top \d+|best places|finalist|winner|manager of the year|named to)\b", re.I),
    "Personnel": re.compile(r"\b(promot\w*|joins?|hires?|hired|recruit\w*|named|appoint\w*|welcomes?|elevat\w*|"
                            r"retire\w*|steps? down|succession|as (?:its )?(?:new )?(?:ceo|president|cfo|chair\w*)|"
                            r"announces .{0,30}\b(?:ceo|president|cfo)\b)", re.I),
    "Fund / firm": re.compile(r"\b(fund [ivx\d]+|new .{0,30}funds?|closes? .{0,40}funds?|final close|hard cap|fundrais\w*|"
                              r"capital commitments|continuation (?:vehicle|fund)|opens? .{0,20}office|anniversary|"
                              r"rebrand\w*|new website|celebrat\w*|independent sponsors?)\b", re.I),
}
# a launch of a strategy or platform initiative is firm news unless the opening text names an acquisition
LAUNCH_HEADLINE = re.compile(r"\b(launch\w*|establish\w*|introduc\w*|unveil\w*)\b", re.I)
OTHER_HEADLINE = re.compile(r"\b(hiring|job posting|position|pre-mba|seeks|careers?|internship|small is beautiful|"
                            r"are you ready|how to|tips|lessons|insights?|perspective|outlook|community|donat\w*|"
                            r"volunteer\w*)\b", re.I)
REVIEW_HEADLINE = re.compile(r"\b(reflects on|year[- ]in[- ]review|looks? back)\b", re.I)
DEAL_BODY = re.compile(
    r"\b(has (?:been )?acquired|acquired|(?:the |its )?acquisitions? of|purchases? of|closing of|"
    r"(?:made|completed) an? (?:\w+ )?(?:\w+ )?investment in|invested in|has partnered with|recapitali[sz](?:ation of|ed)|"
    r"sale of|has sold|merged with|joined forces|strategic combination|definitive agreement to (?:acquire|sell|merge))\b", re.I)

# ----------------------------------------------------------------------------- 2. the lead sentence
# the deal verb of a lead sentence; the earliest one in the sentence is the main one
LEAD_VERBS: List[Tuple[str, re.Pattern]] = [
    ("exit_passive", re.compile(r"\b(?:has been|was|have been|has agreed to be|agreed to be|will be|(?:definitive\s+)?agreement\s+to\s+be)"
                                r"\s+acquired by\b", re.I)),
    ("exit_agree", re.compile(r"\b(?:definitive\s+)?agreement\s+to\s+sell\b|\bagreed\s+to\s+sell\b", re.I)),
    ("exit", re.compile(r"\b(?:(?:has\s+)?(?:announced|announces|announce|completed|completes|closed|finalized)\s+)?(?:today\s+)?"
                        r"(?:the\s+|a\s+)?(?:successful\s+|majority\s+)?(?:sale|divestiture)\s+of\b"
                        r"(?:\s+its\s+(?:investment|interest|stake)\s+in\b)?", re.I)),
    ("exit", re.compile(r"\b(?:has\s+)?(?:sold|exited)\s+(?:its\s+(?:investment|interest|stake)\s+in\b)?", re.I)),
    ("merger", re.compile(r"\b(?:announced\s+(?:today\s+)?)?(?:a\s+)?(?:strategic\s+)?(?:combination|merger)\s+with\b|"
                          r"\b(?:has\s+)?(?:merged|combined)\s+with\b", re.I)),
    ("recap", re.compile(r"\b(?:completed\s+)?(?:the\s+|a\s+)?(?:dividend\s+)?recapitali[sz]ation\s+of\b|"
                         r"\b(?:has\s+)?recapitali[sz]ed\b", re.I)),
    ("invest", re.compile(r"\b(?:(?:has\s+)?(?:made|completed|announced?|announces)\s+)?(?:an?\s+|its\s+)?"
                          r"(?:strategic\s+|growth\s+|capital\s+|minority\s+|majority\s+|significant\s+|equity\s+)*"
                          r"investment\s+in\b|\b(?:has\s+)?invested\s+in\b|\b(?:has\s+)?partnered\s+with\b|"
                          r"\b(?:its\s+|a\s+|new\s+)*(?:strategic\s+)?partnership\s+with\b", re.I)),
    ("buy", re.compile(r"\b(?:has\s+|have\s+)?(?:recently\s+)?acquired\b(?!\s+by)|"
                       r"\b(?:(?:has\s+)?(?:led|completed|announced|announces|announce|closed|finalized)\s+)?(?:today\s+)?"
                       r"(?:the\s+|its\s+|an?\s+|two\s+|three\s+)?(?:recent\s+|strategic\s+|successive\s+)*"
                       r"(?:acquisitions?|purchases?)\s+of\b|\bclosing\s+of\b|\bacquisitions,?\s+including\b", re.I)),
]
NAME_TOKEN = r"(?:[A-Z0-9][\w.'’&/+\-]*|[a-z]{1,3}[A-Z][\w.'’&\-]*)"
NAME_SUFFIX = r"(?:,?\s+(?:Inc|LLC|L\.L\.C|Ltd|Corp|LP|L\.P|PLLC|PC)\b\.?)"
NAME_SPAN = re.compile(NAME_TOKEN + r"(?:(?:\s+(?:&|of|de|du|la|and)\s+|\s+)" + NAME_TOKEN + r")*" + NAME_SUFFIX + "?")
NAME_SPAN_NO_AND = re.compile(NAME_TOKEN + r"(?:(?:\s+(?:&|of|de|du|la)\s+|\s+)" + NAME_TOKEN + r")*" + NAME_SUFFIX + "?")
NOT_A_NAME = re.compile(r"^(?:The|A|An|This|These|That|Its|Their|Our|Leading|Based|Headquartered|Founded|Media|Today|"
                        r"Since|With|In|On|At|As|For|By|Over|After|Both|Each|CEO|President|Chief|LLC|Inc|Company|"
                        r"Acquisition|Acquisitions|Investment|Partnership|Transaction|Combined|Together|Additionally|"
                        r"North|South|East|West|United States|U\.S\.?)$")
ALIAS_RE = re.compile(r"\(\s*(?:the\s+|or\s+)?[“\"]([^”\"]{1,40})[”\"](?:\s*(?:or|,|and)+\s*,?\s*(?:the\s+)?[“\"][^”\"]{1,40}[”\"])*\s*\)")
NAMED_PAREN = re.compile(r"(" + NAME_SPAN_NO_AND.pattern + r")\s*\(([^()]{1,120})\)")
LOCATION_IN_NAME = re.compile(r"(?:(?:[A-Z][\w.']+\s?){1,3},\s*)?(?:[A-Z][\w.']+[\s-]?){1,3}-based\s+")
MONTH_SPAN = re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:\s+\d{1,2},?)?(?:\s+\d{4})?$|^Q[1-4](?:\s+\d{4})?$|^\d{4}$")
DATELINES = [
    re.compile(r"^.{0,100}?" + DATE_RE.pattern + r"\s*\)?\s*(?:/[A-Za-z ]+/)?\s*(?:\||[—–]+|--|-|:)\s*", re.I),
    re.compile(r"^[A-Z][A-Za-z.,' ]{0,40}[—–-]+\s*\((?:[A-Z ]*WIRE|[A-Z][a-z]+ \d{1,2}, \d{4})\)\s*[—–-]+\s*"),
    re.compile(r"^[A-Z][A-Za-z.]+(?:\s[A-Z][A-Za-z.]+)*,\s*[A-Z][A-Za-z.]{1,10}\s*[—–]+\s*"),
]
PLURAL = re.compile(r"\b(acquisitions|purchases|two|three|four|five|both|several|separate transaction)\b", re.I)
PUBLISHER_SUBJECT = re.compile(r"\b(its (?:portfolio|partner|platform) company|portfolio company|partner company|platform|"
                               r"backed|owned)\b", re.I)

# ----------------------------------------------------------------------------- 3-5. type, status, price
MINORITY_WORDS = re.compile(r"\b(minority (?:investment|stake|interest|recapitalization)|growth (?:equity|investment|capital)|"
                            r"structured (?:equity|capital)|non-control|mezzanine|subordinated debt)\b", re.I)
CONTROL_WORDS = re.compile(r"\b(acquisition of|acquired|recapitali[sz]\w*|majority)\b", re.I)
DIVIDEND_RECAP = re.compile(r"\bdividend recap\w*", re.I)
ADDON_WORDS = re.compile(r"\badd-?ons?\b", re.I)
COMPLETED = re.compile(r"\b(exits|exited|has (?:been )?(?:acquired|completed|closed|invested|partnered|sold|merged|joined|recapitali[sz]ed)|"
                       r"(?:completed|completes|closed|closes|finalized) (?:the |its |an? |two )?(?:successive )?"
                       r"(?:acquisitions?|sale|investment|growth investment|recapitali[sz]ation|merger|transaction|"
                       r"combination|purchases?)|recently (?:acquired|completed|invested|partnered|reached)|"
                       r"have (?:joined|merged|combined)|has made an? (?:\w+ )?(?:\w+ )?investment|recent acquisitions? of|"
                       r"completed its first acquisition|"
                       r"(?:led|announced (?:that it|that|it) (?:has )?)(?:acquired|completed|sold|invested)|"
                       r"invested in|acquired\b(?! by))", re.I)
PENDING = re.compile(r"\b(definitive agreement|has agreed to|have agreed to|agreed to (?:acquire|sell|merge|be acquired)|"
                     r"will acquire|to be acquired|expected to close|subject to (?:customary|regulatory)|"
                     r"letter of intent|pending (?:regulatory )?approval)\b", re.I)
UNDISCLOSED = re.compile(r"\b((?:financial |the )?terms(?: of the (?:transaction|deal|acquisition|investment|sale|agreement))?"
                         r"(?: were| was| are| have)? (?:not|n't) (?:been )?(?:disclosed|released|made public)|undisclosed|"
                         r"not disclosed|did not disclose|were not released)\b", re.I)
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million|thousand|bn|mm|m|b|k)?\b", re.I)
VALUE_CONTEXT = re.compile(r"\b(purchase price|valued at|valuation|transaction value|enterprise value|for (?:approximately |about )?\$|"
                           r"consideration|deal (?:is )?worth|in a deal)\b", re.I)
RATIONALE_VERBS = re.compile(r"\b(expand\w*|broaden\w*|extend\w*|add(?:s|ed|ing)?|strengthen\w*|enhanc\w*|complement\w*|"
                             r"accelerat\w*|enabl\w*|position\w*|diversif\w*|deepen\w*|bolster\w*|establish\w*|enter\w*|"
                             r"increas\w*|build(?:s|ing)? on|creat\w* (?:a|one of)|grow(?:s|th|ing)?|scale|capabilit\w*)\b", re.I)
DESCRIBE_VERBS = re.compile(r"\b(is (?:a|an|the|one of)|are (?:a|an)|manufactur\w*|provid\w*|distribut\w*|design\w*|deliver\w*|"
                            r"supplier|specializ\w*|offers?|serves?|produc\w*|operat\w*|founded in|based in|headquartered)\b", re.I)
_STATE_ALT = "|".join(sorted([re.escape(n) for n in US_STATES.values()] + [re.escape(a) for a in STATE_ABBR] + list(US_STATES),
                             key=len, reverse=True))
_PLACE = r"(?:[A-Z][A-Za-z.'\-]+\s?){1,4}?"
LOCATION = re.compile(r"(?:\b(?:[Bb]ased|[Hh]eadquartered|[Ll]ocated|[Hh]eadquarters) (?:in|out of|near)\s+(?:the\s+)?"
                      r"(?:(?:central|northern|southern|eastern|western|north|south|east|west|greater|downtown|suburban|metro)\s+)?|"
                      r"\b(?:of|in)\s+(?=[A-Z][a-z]+,\s*(?:" + _STATE_ALT + r")\b))"
                      r"(" + _PLACE + r"(?:,\s*(?:" + _STATE_ALT + r"))?)(?=[\s,.;)]|$)")
LOCATION_PREFIX = re.compile(r"\b((?:[A-Z][A-Za-z.'\-]+\s?){1,3}(?:,\s*(?:" + _STATE_ALT + r"))?)-based\b")

# headline patterns: the fallback when the lead sentence gives no target
TRAIL_NOISE = re.compile(r"\s*(?:,\s+(?:a|an|the|its|which|who|based|headquartered|located|adding|fueling)\b.*|\s+[-–—]\s+.*|"
                         r"\s*\(.*?\)\s*$|\s+(?:in|into|after|to (?:expand|broaden|strengthen|grow|create|form|enhance|"
                         r"accelerate|add|extend|enter|build|boost|deepen|further|advance)|for (?:its|their)|from|"
                         r"as (?:a|an|its)|with (?:the|its)|through|expanding|strengthening|broadening|adding|creating|"
                         r"enhancing|bolstering|extending)\b.*|[.!:;,]+$)", re.I)
HEADLINE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("merger", re.compile(r"^(?P<buyer>.+?)\s+(?:and|&)\s+(?P<target>.+?)\s+(?:join(?:s|ed)? forces|merge|combine|unite|"
                          r"announce (?:a |their )?(?:merger|combination|partnership))\b", re.I)),
    ("exit", re.compile(r"^(?P<seller>.+?)\s+(?:announces|completes|closes|finalizes)?\s*(?:the\s+|a\s+)?(?:successful\s+|majority\s+)?"
                        r"(?:sale|divestiture|exit|realization) of (?:its (?:investment|interest) in\s+)?(?P<target>.+?)"
                        r"(?:\s+to\s+(?P<buyer>.+))?$", re.I)),
    ("exit", re.compile(r"^(?P<seller>.+?)\s+(?:sells|sold|exits|exited|divests|realizes)\s+(?:its\s+)?(?:investment in\s+|"
                        r"interest in\s+|stake in\s+)?(?P<target>.+?)(?:\s+to\s+(?P<buyer>.+))?$", re.I)),
    ("exit_passive", re.compile(r"^(?P<target>.+?)\s+(?:is |has been |to be |agrees to be )?acquired by\s+(?P<buyer>.+)$", re.I)),
    ("buy", re.compile(r"^(?P<buyer>.+?)\s+(?:announces|completes|closes|finalizes|has completed|completed)?\s*"
                       r"(?:the\s+|its\s+|an?\s+)?(?:strategic\s+|add-on\s+)?(?:acquisitions?|purchases?)\s+of\s+(?P<target>.+)$", re.I)),
    ("buy", re.compile(r"^(?P<buyer>.+?)\s+(?:acquires?|has acquired|acquired|buys|to acquire|purchases|adds)\s+(?P<target>.+)$", re.I)),
    ("invest", re.compile(r"^(?P<buyer>.+?)\s+(?:announces|completes|makes|closes)?\s*(?:an?\s+|its\s+|new\s+)?"
                          r"(?:strategic\s+|majority\s+|minority\s+|growth\s+|capital\s+|platform\s+|new\s+)*investments?\s+in\s+(?P<target>.+)$", re.I)),
    ("invest", re.compile(r"^(?P<buyer>.+?)\s+(?:invests|has invested|invested|partners|has partnered|partnered|backs|"
                          r"recapitalizes)\s+(?:in\s+|with\s+)?(?P<target>.+)$", re.I)),
]
HEAD_NOISE = re.compile(r"^(?:the\s+|its\s+|an?\s+|announces\s+|today\s+|portfolio company\s+|platform company\s+|"
                        r"partner company\s+)+", re.I)
GENERIC_NAMES = re.compile(r"^(?:two|three|four|several|multiple|new|a|an|the)?\s*(?:companies|businesses|acquisitions?|"
                           r"add-?ons?|platforms?|investments?|firms?|providers?|leaders?|partnership)\b", re.I)
ABBREVIATIONS = re.compile(r"(?:\b(?:Inc|Co|Corp|Ltd|St|Mr|Ms|Mrs|Dr|Jr|Sr|No|Mt|Ft|Ave|Blvd|vs|etc|approx|"
                           r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)|"
                           r"\b(?:[A-Z]\.){1,3}|\b[A-Z]|" + "|".join(re.escape(a[:-1]) for a in STATE_ABBR) + r")\.$")


# ----------------------------------------------------------------------------- helpers
def sentences(text: str) -> List[str]:
    """Sentences, paragraph by paragraph; no split after an abbreviation (Inc., Co., U.S., Minn., Jan., an initial)."""
    out = []
    for para in (text or "").split("\n\n"):
        para = re.sub(r"\s+", " ", para).strip()
        pieces = re.split(r"(?:(?<=[.!?])|(?<=[.!?][\"”’]))\s+(?=[\"“A-Z0-9])", para)
        merged: List[str] = []
        for piece in pieces:
            if merged and ABBREVIATIONS.search(merged[-1]):
                merged[-1] = merged[-1] + " " + piece
            else:
                merged.append(piece)
        out += [m.strip() for m in merged if m.strip()]
    return out


def stems_in(text: str, stems: List[str]) -> List[str]:
    low = (text or "").lower()
    return [s for s in stems if re.search(r"(?<![a-z])" + re.escape(s), low)]


def strip_dateline(s: str) -> str:
    for _ in range(3):
        for rx in DATELINES:
            m = rx.match(s)
            if m and m.end() < len(s):
                s = s[m.end():]
                break
        else:
            break
    return s.strip()


def _aliases(source: str) -> Tuple[str, ...]:
    al = FIRM_ALIASES.get(source, ())
    firm = SOURCES[source].firm if source in SOURCES else ""
    return tuple(sorted({a for a in (*al, firm) if a}, key=len, reverse=True))


def is_publisher(name: str, source: str) -> bool:
    """The publishing firm under any of its names ('Align Capital Partners', 'ACP', 'Goldner Hawn, LP', 'Huron')."""
    n = re.sub(r"[’']s?$", "", (name or "").strip()).lower()
    n = re.sub(r",?\s+(?:llc|lp|l\.p\.|inc\.?)$", "", n)
    return any(n == a.lower() or re.fullmatch(re.escape(a.lower()) + r"(?: equity| capital| partners| llc| lp)*", n)
               for a in _aliases(source))


def clean_name(name: str) -> str:
    if not name:
        return ""
    n = re.sub(r"[“”\"]", "", name).strip()
    n = HEAD_NOISE.sub("", n).strip()
    prev = None
    while prev != n:
        prev = n
        n = TRAIL_NOISE.sub("", n).strip()
    n = re.sub(r"^(?:portfolio company|partner company|platform company|platform)\s*,?\s*", "", n, flags=re.I)
    n = re.sub(r"\s*['’]s$", "", n)
    return n.strip(" ,.;:-–—")


def strip_publisher_prefix(name: str, source: str) -> Tuple[str, bool]:
    """'Goldner Hawn Portfolio Company Cyber Advisors' -> ('Cyber Advisors', True); "Huron Capital's Platform Sunland
    Asphalt" -> ('Sunland Asphalt', True); 'ACP-Backed E Source' -> ('E Source', True)."""
    for a in _aliases(source):
        m = re.match(re.escape(a) + r"(?:\s+(?:equity|capital|partners))*\s*(?:['’]s?\s+|[-–]\s*backed\s+|\s+)(?P<rest>.+)$", name or "", re.I)
        if m:
            rest = re.sub(r"^(?:[\w®&\- ]{0,40}?\s)?(?:portfolio company|partner company|platform company|platform|company|"
                          r"affiliate)\s*,?\s*", "", m.group("rest").strip(), flags=re.I)
            rest = re.sub(r"^ExecFactor\s*®?\s*", "", rest).strip(" ,")
            if not rest or re.match(r"(?:announces|completes|closes|acquires|invests|today|forms|partners|exits|sells)\b", rest, re.I):
                return name, False
            return rest, True
    return name, False


def name_spans(text: str, allow_and: bool = True) -> List[str]:
    """Capitalised name spans ('Marco Rubber & Plastics', 'Advanced Tooling Design, Inc', '1-800 HANSONS', 'eDot')."""
    rx = NAME_SPAN if allow_and else NAME_SPAN_NO_AND
    out = []
    for m in rx.finditer(text or ""):
        span = m.group(0).strip(" ,.")
        if re.match(r"^(?:A|An)\s+", span):
            span = re.sub(r"^(?:A|An)\s+", "", span)
        span = re.sub(r"[’']s?$", "", span)
        if span and not NOT_A_NAME.match(span) and not MONTH_SPAN.match(span):
            out.append(span)
    return out


def mentions(text: str, name: str, aliases: Tuple[str, ...] = ()) -> bool:
    for n in (name, *aliases):
        n = (n or "").split(",")[0].strip()
        if len(n) >= 2 and re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", text or ""):
            return True
    first = (name or "").split()[0] if (name or "").split() else ""
    common = {"American", "National", "Global", "United", "General", "First", "Premier", "Advanced", "International",
              "Southern", "Northern", "Western", "Eastern", "Central", "Pacific", "Atlantic", "Midwest", "Superior",
              "Independent", "Professional", "Custom", "Standard", "Total", "Systems", "Applied", "Integrated"}
    long_enough = len(first) >= 5 or (len(first) >= 3 and first.isupper())
    return (long_enough and first[0].isupper() and first not in common
            and re.search(r"\b" + re.escape(first) + r"\b", text or "") is not None)


def money_value(text: str) -> Optional[float]:
    m = MONEY.search(text or "")
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    mult = {"billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
    return v * mult.get((m.group(2) or "").lower(), 1.0)


def deal_id(url: str) -> str:
    return hashlib.sha1(url.rstrip("/").lower().encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------- 1. is it a deal?
def classify_news(title: str, body: str, source: str = "") -> Tuple[str, str]:
    """(category, evidence). Headline first; the opening text when the headline is not clear."""
    head = title or ""
    strong = STRONG_DEAL.search(head)
    aliases = "|".join(re.escape(a) for a in _aliases(source)) or r"$^"
    m = re.match(rf"^(?:{aliases})\s+(?:closes|completes|celebrates|reaches|marks)\s+(?:its\s+)?\d+\w*\s+"
                 r"(?:acquisition|investment|transaction|deal)s?\b", head, re.I)
    if m:
        return "Fund / firm", f'headline: "{m.group(0)}" (a firm milestone, no single deal)'
    m = REVIEW_HEADLINE.search(head)
    if m:
        return "Other", f'headline: "{m.group(0)}" (a review of past activity, no single deal)'
    for cat, rx in NON_DEAL.items():
        m = rx.search(head)
        if m and not strong:
            return cat, f'headline: "{m.group(0)}"'
    m = OTHER_HEADLINE.search(head)
    if m and not strong:
        return "Other", f'headline: "{m.group(0)}"'
    m = LAUNCH_HEADLINE.search(head)
    if m and not strong and not DEAL_BODY.search(" ".join(sentences(body)[:5])):
        return "Fund / firm", f'headline: "{m.group(0)}" and no acquisition in the opening text'
    if strong:
        return "Deal", f'headline: "{strong.group(0)}"'
    lead = " ".join(sentences(body)[:5])
    m = DEAL_BODY.search(lead)
    if m:
        weak = WEAK_DEAL.search(head)
        return "Deal", f'opening text: "{m.group(0)}"' + (f' (headline: "{weak.group(0)}")' if weak else "")
    return "Other", "no deal language in the headline or the opening text"


# ----------------------------------------------------------------------------- 2. who bought what
def _verb(sentence: str):
    best = None
    for kind, rx in LEAD_VERBS:
        m = rx.search(sentence)
        if m and (best is None or m.start() < best[1].start()):
            best = (kind, m)
    return best


def _plain(text: str) -> str:
    return re.sub(r"\s*\([^()]*\)", "", text)          # parentheticals out


def lead_parse(art: Article, source: str) -> dict:
    """Buyer / target / seller from the first sentence (of the first six) that has a deal verb."""
    out = {"kind": "", "buyer": "", "targets": [], "seller": "", "subject": "", "sentence": "", "aliases": {},
           "target_tail": ""}
    sents = sentences(art.body)[:6]
    earlier_protagonist, earlier_subject = "", ""
    for i, raw in enumerate(sents):
        s = strip_dateline(raw)
        if len(s.split()) < 5:
            continue
        found = _verb(s)
        if not found:
            for am in NAMED_PAREN.finditer(s):
                if re.search(r"[“\"]Company[”\"]", am.group(2)) and not earlier_protagonist:
                    earlier_protagonist = am.group(1).strip()
                    earlier_subject = _plain(s[:am.end()])
            continue
        kind, m = found
        aliases, protagonists = {}, []
        for am in NAMED_PAREN.finditer(s):
            q = re.search(r"[“\"]([^”\"]{1,40})[”\"]", am.group(2))
            if q:
                aliases[am.group(1).strip()] = q.group(1).strip(" ,")
            if re.search(r"[“\"]Company[”\"]", am.group(2)):          # the release's protagonist: (the “Company”)
                protagonists.append(re.sub(r"^(?:[A-Z]\w*\s+){0,6}?(?:platform|company|provider|firm)\s+", "", am.group(1).strip()))
        subject, obj = _plain(s[:m.start()]), LOCATION_IN_NAME.sub("", _plain(s[m.end():]))
        out.update(kind=kind, sentence=raw, subject=subject, aliases=aliases)

        # the subject: the company the release calls "the Company" if it is before the verb, else the last company
        # name before the verb once a description after ", a ..." is cut off
        cut = re.split(r",\s+(?:a|an|the|another|one of|which|who)\b", subject, maxsplit=1)[0]
        spans = name_spans(cut)
        named = spans[-1] if spans else ""
        for p in protagonists:
            if p in subject:
                named = name_spans(p)[-1] if name_spans(p) else p
                break
        if not named and earlier_protagonist and kind not in ("exit", "exit_passive", "exit_agree"):
            named = name_spans(earlier_protagonist)[-1] if name_spans(earlier_protagonist) else earlier_protagonist
            out["subject"] = subject = earlier_subject
        if kind == "exit_passive":
            out["targets"] = [named] if named else []
            spans_o = name_spans(obj)
            out["buyer"] = spans_o[0] if spans_o else ""
            return out
        if kind == "exit_agree":            # "... has signed a definitive agreement to sell X to Y"
            kind = "exit"
            out["kind"] = kind
        if kind == "exit":
            out["seller"] = named
            tgt_part, _, buyer_part = obj.partition(" to ")
            spans_t = name_spans(tgt_part)
            out["targets"] = spans_t[:1]
            buyer_part = re.sub(r"^\s*(?:an?\s+)?(?:affiliates? of|(?:private equity )?funds? (?:affiliated with|managed by|advised by))\s+",
                                "", buyer_part, flags=re.I)
            spans_b = name_spans(buyer_part)
            out["buyer"] = spans_b[0] if spans_b else ""
            out["target_tail"] = tgt_part
            return out
        out["buyer"] = named
        obj = re.sub(r"\band,?\s+in a separate transaction,?\s+", " and ", obj, flags=re.I)
        obj = re.sub(r"^\s*,\s*(?:in partnership with|together with|alongside|along with)[^,]*,\s*", " ", obj, flags=re.I)
        plural = bool(PLURAL.search(s) or PLURAL.search(art.title or "")) or ", and " in obj.split(".")[0]
        head = re.split(r",\s+(?:a|an|the|another|one of|which|who|based|headquartered|located|from|to|in)\b|"
                        r"\s+(?:to|from|for|in|as|with)\s+(?=[a-z])", obj, maxsplit=1)[0]
        spans_t = name_spans(head, allow_and=not plural)
        targets = [t for t in spans_t if not is_publisher(t, source)] if plural else spans_t[:1]
        two = re.match(r"\s*(" + NAME_SPAN_NO_AND.pattern + r")\s+and\s+(" + NAME_SPAN_NO_AND.pattern + r")", head)
        if not plural and two and re.search(NAME_SUFFIX + "$", two.group(1)):
            targets = [two.group(1).strip(" ,."), two.group(2).strip(" ,.")]
        # "has acquired three providers of outsourced appellate services": the next sentences name them
        if not targets and re.match(r"\s*(?:two|three|four|five|several|multiple)\b", obj, re.I):
            later = " ".join(sents[i + 1:i + 3])
            named_later = [am.group(1).strip() for am in re.finditer(r"(" + NAME_SPAN.pattern + r")\s*" + ALIAS_RE.pattern, later)]
            named_later = [n for n in named_later if not is_publisher(n, source) and n != out["buyer"]
                           and not NOT_A_NAME.match(n)]
            targets = list(dict.fromkeys(named_later))
        out["targets"] = targets
        out["target_tail"] = obj
        if targets:
            return out
    return out


def headline_parse(title: str, source: str) -> dict:
    for kind, rx in HEADLINE_PATTERNS:
        m = rx.search(title or "")
        if not m:
            continue
        g = m.groupdict()
        buyer, owned = strip_publisher_prefix(clean_name(g.get("buyer") or ""), source)
        seller = clean_name(g.get("seller") or "")
        target_raw = g.get("target") or ""
        plural = bool(PLURAL.search(title))
        parts = re.split(r"\s*,\s*and\s+|\s*,\s*|\s+and\s+" if plural else r"\s*,\s*", target_raw)
        targets = [clean_name(p) for p in parts if p.strip()]
        targets = [t for t in targets if t and not GENERIC_NAMES.match(t) and not is_publisher(t, source)]
        # a descriptor in front of the name: "Stencil Manufacturer Photo Etch Technology", "Managed Service Provider eDot"
        targets = [re.sub(r"^(?:[A-Z][\w&\-]*\s+){0,4}?(?:Manufacturer|Provider|Firm|Distributor|Advisor|Company|Leader)\s+(?=[A-Z])",
                          "", t) for t in targets]
        return {"kind": kind, "buyer": buyer, "owned": owned, "seller": seller, "targets": targets}
    return {"kind": "", "buyer": "", "owned": False, "seller": "", "targets": []}


def _owned_by_publisher(name: str, text: str, source: str) -> Tuple[bool, str]:
    """Does the text say the publisher owns `name`? ('Standard Locknut, a Tonka Bay partner company', 'its portfolio
    company Cyber Advisors', 'a Huron Capital portfolio company')."""
    if not name:
        return False, ""
    aliases = "|".join(re.escape(a) for a in _aliases(source))
    n = re.escape(name.split(",")[0])
    pats = [rf"{n}[^.]{{0,160}}?\ban?\s+(?:{aliases})(?:['’]s?)?\s+(?:partner|portfolio|platform)\b",
            rf"(?:{aliases})(?:['’]s?)?\s+(?:[\w\-]+\s+){{0,6}}?(?:portfolio company|partner company|platform(?: company)?|provider|firm|company),?\s+{n}",
            rf"(?:{aliases})[-–\s]+backed\s+{n}",
            rf"its\s+(?:[\w\-]+\s+){{0,4}}?(?:portfolio|partner|platform)\s+(?:company,?\s+)?{n}",
            rf"{n}[^.]{{0,80}}?\b(?:backed by|owned by|a portfolio company of)\s+(?:{aliases})"]
    for s in sentences(text)[:8]:
        s2 = _plain(s)
        for p in pats:
            if re.search(p, s2, re.I):
                return True, s
    return False, ""


def parties(art: Article, source: str) -> dict:
    lp = lead_parse(art, source)
    hp = headline_parse(art.title, source)
    kind = lp["kind"] or hp["kind"]
    if hp["kind"] in ("exit", "exit_passive") and kind not in ("exit", "exit_passive"):
        kind = "exit"                                  # "Announces Sale of SEAM Group to ABB": the headline says who sells
        lp = dict(lp, buyer=hp["buyer"] or "", targets=hp["targets"] or lp["targets"], seller=hp["seller"])
    targets = lp["targets"] or hp["targets"]
    evidence = lp["sentence"] if lp["targets"] else ""
    if not lp["targets"] and hp["targets"]:
        evidence = f'headline: "{art.title}"' + (f" / {lp['sentence']}" if lp["sentence"] else "")
    buyer = lp["buyer"] or hp["buyer"]
    owned_prefix = False
    if buyer and not is_publisher(buyer, source):
        buyer, owned_prefix = strip_publisher_prefix(buyer, source)
    seller = lp["seller"] or hp["seller"]
    # a buyer the publisher owns: the sponsor is named in the lead sentence's subject, or elsewhere in the text
    owned, owned_ev = False, ""
    if buyer and not is_publisher(buyer, source) and kind not in ("exit", "exit_passive"):
        aliases = "|".join(re.escape(a) for a in _aliases(source))
        if lp["subject"] and (re.search(rf"\b(?:{aliases})\b", lp["subject"]) or PUBLISHER_SUBJECT.search(lp["subject"])):
            owned, owned_ev = True, lp["sentence"]
        elif hp["owned"] or owned_prefix:
            owned, owned_ev = True, f'headline: "{art.title}"'
        else:
            owned, owned_ev = _owned_by_publisher(buyer, art.body, source)
    if kind == "exit_passive":
        kind = "exit"
    return {"kind": kind, "buyer": buyer, "targets": [t for t in targets if t], "seller": seller,
            "parties_evidence": evidence, "buyer_owned": owned, "owned_evidence": owned_ev,
            "aliases": lp["aliases"], "target_tail": lp["target_tail"], "lead_sentence": lp["sentence"]}


# ----------------------------------------------------------------------------- 3. transaction type
def transaction_type(p: dict, art: Article, source: str) -> Tuple[str, str]:
    title = art.title or ""
    lead = p.get("lead_sentence") or " ".join(sentences(art.body)[:2])
    text = title + " " + lead
    if DIVIDEND_RECAP.search(text):
        return "Dividend recapitalization", f'"{DIVIDEND_RECAP.search(text).group(0)}"'
    if p["kind"] == "exit":
        return "Exit", "the publisher sells: " + (lead[:220] if lead else f'headline "{title}"')
    if p["kind"] == "merger":
        return "Merger / combination", f'"combination / merger with": {lead[:220]}'
    if p["buyer"] and is_publisher(p["buyer"], source):
        m = MINORITY_WORDS.search(text)
        if m and not CONTROL_WORDS.search(lead):
            return "Growth / minority investment", f'"{m.group(0)}"'
        m = ADDON_WORDS.search(lead)
        if m:
            return "Add-on acquisition", f'"{m.group(0)}" (the firm calls the deal an add-on)'
        return "Platform investment", f"the buyer is the publishing firm ({SOURCES[source].firm}): {lead[:220]}"
    if p["buyer_owned"]:
        return "Add-on acquisition", f"the buyer is a {SOURCES[source].firm} company: {p['owned_evidence'][:220]}"
    if p["buyer"]:
        return "Unclear", "the buyer is not the publisher and the text does not say the publisher owns it"
    return "Unclear", "buyer not identified"


# ----------------------------------------------------------------------------- 4. the target's business
def _short_names(first: str, aliases: Dict[str, str]) -> Tuple[str, ...]:
    key = first.split(",")[0]
    return tuple(v for k, v in aliases.items() if first and (k.startswith(key) or first.startswith(k.split(",")[0])))


def describe_target(targets: List[str], aliases: Dict[str, str], tail: str, art: Article, source: str,
                    lead: str = "") -> Tuple[str, str]:
    """(description, where it came from) for the first named target."""
    if not targets:
        return "", ""
    first = targets[0]
    short = _short_names(first, aliases)
    # the appositive after the target's name in the lead sentence: "... acquired ATD, a designer of ..."
    m = re.search(r",\s+((?:a|an|the|another|one of)\s+.+)$", tail or "")
    if m and len(m.group(1).split()) >= 4 and not re.match(r"(?:a|an)\s+(?:affiliate|fund)", m.group(1)):
        return m.group(1).rstrip(". "), "the lead sentence"
    publisher = "|".join(re.escape(a) for a in _aliases(source))
    cands: List[Tuple[str, str]] = []
    sents = sentences(art.body)
    for i, s in enumerate(sents):
        s2 = strip_dateline(s)
        if s == lead or s2.startswith(("“", '"')) or re.match(rf"\s*(?:{publisher})\b", s2):
            continue
        named = mentions(s2, first, short)
        # "Both firms are based in ...", "The company provides ...": the sentence right after one about the target
        follows = i > 0 and mentions(sents[i - 1], first, short) and re.match(
            r"(?:Both (?:firms|companies)|The (?:company|firm|business)|It|They|The three companies)\b", s2)
        if (named or follows) and DESCRIBE_VERBS.search(s2) and not re.search(r"\b(?:acquisition|investment) (?:in|of|for)\b", s2[:80]):
            cands.append((s2, "announcement text"))
    for name, txt in art.about:
        if mentions(name, first, short) or mentions(first, name):
            cands.append((" ".join(sentences(txt)[:2]), f'"About {name}" section'))
    for c in cands:                                   # the first that names a business the taxonomy knows
        if industry_of(c[0])[0] != UNKNOWN:
            return c
    if cands:
        return cands[0]
    # "has acquired three providers of outsourced appellate services": the deal sentence's own words
    if tail and not mentions(tail[:60], first, short) and len(tail.split()) >= 3:
        return tail.strip(" .,"), "the lead sentence (what was acquired)"
    return "", ""


def industry_of(description: str) -> Tuple[str, List[str], bool]:
    """(category, matched stems, ambiguous). Only the description's business clause counts: in 'a manufacturer of
    bearings used in industrial applications' the business is manufacturing; what follows 'used in' / 'serving' is
    the customers' industry and goes to customer_markets."""
    if not description:
        return UNKNOWN, [], False
    business = re.split(r"\b(?:used (?:in|by)|utilized|serving|serves|for (?:the )?customers|customers (?:in|across|such as)|"
                        r"end markets?|clients (?:in|across|such as)|across (?:the|a)|to (?:help|support|comply))\b",
                        description, maxsplit=1, flags=re.I)[0]
    scores: Dict[str, float] = {}
    hits: Dict[str, List[str]] = {}
    first: Dict[str, int] = {}
    low = business.lower()
    for cat, stems in INDUSTRIES.items():
        h = stems_in(business, stems)
        if h:
            scores[cat] = len(h)
            hits[cat] = h
            first[cat] = min(re.search(r"(?<![a-z])" + re.escape(x), low).start() for x in h)
    if not scores:
        return UNKNOWN, [], False
    head = min(first, key=first.get)             # "a distributor of oil seals": the head noun names the business
    scores[head] += 1
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], first[kv[0]]))
    ambiguous = len(ranked) > 1 and ranked[1][1] >= ranked[0][1]
    return ranked[0][0], hits[ranked[0][0]], ambiguous


def customer_markets(description: str, body: str) -> List[str]:
    tail = ""
    m = re.search(r"\b(?:used (?:in|by)|utilized|serving|serves|for (?:the )?customers|customers (?:in|across|including|such as)|"
                  r"end markets?|clients (?:in|across|including|such as)|across (?:the|a))\b(.*)", description or "", re.I)
    if m:
        tail = m.group(1)
    for s in sentences(body):
        mm = re.search(r"\b(?:serv(?:es|ing) (?:\w+ )?(?:customers|clients) (?:in|across)|end[- ]markets? (?:including|such as|like)|"
                       r"customers (?:include|in|across))\b(.*)", s, re.I)
        if mm:
            tail += " " + mm.group(1)
    return [cat for cat, stems in END_MARKETS.items() if stems_in(tail, stems)]


def location_of(targets: List[str], aliases: Dict[str, str], description: str, body: str) -> Tuple[str, str]:
    """(location text, US state code) of the target, from the sentences about it only (the dateline is the
    publisher's city, never the target's). "X-based" counts only when it qualifies the target ("a Houston-based
    distributor", not "acquired Southern California-based Verdant" in a sentence about another company)."""
    first = targets[0] if targets else ""
    short = _short_names(first, aliases)
    sents = sentences(body)
    cands = [description]
    for i, s in enumerate(sents):
        if first and (mentions(s, first, short) or (i > 0 and mentions(sents[i - 1], first, short)
                                                   and re.match(r"(?:Both (?:firms|companies)|The (?:company|firm|business)|It)\b", s))):
            cands.append(s)
    names = "|".join(re.escape(n.split(",")[0]) for n in (first, *short) if n)
    generic = r"(?:manufacturer|provider|distributor|company|firm|business|supplier|contractor|consultancy|consulting|maker|developer|operator|" + (names or "$^") + ")"
    for s in cands:
        if not s:
            continue
        s2 = strip_dateline(s)
        m = LOCATION.search(s2)
        if not m:
            m = next((x for x in LOCATION_PREFIX.finditer(s2)
                      if re.match(r"\s*(?:[\w&\-]+\s+){0,4}?" + generic, s2[x.end():], re.I)), None)
        if m:
            return m.group(1).strip(" ,"), state_code(m.group(1))
    m = re.search(r"\bin\s+(?:the\s+)?(?:state of\s+)?(" + "|".join(US_STATES.values()) + r")\b", description or "")
    if m:                                             # "a provider of underground infrastructure services in Texas"
        return m.group(1), state_code(m.group(1))
    return "", ""


def state_code(loc: str) -> str:
    """The US state of a location text; "" when it names none, several, or a city shared by two states."""
    if any(c in loc for c in AMBIGUOUS_CITIES) and not re.search(r",\s*\S", loc):
        return ""
    found = [code for code, name in US_STATES.items() if re.search(r"\b" + name + r"\b", loc)]
    if re.search(r"\bKansas City\b", loc):
        found = [c for c in found if c != "KS"]
    if len(set(found)) > 1:
        return ""
    if found:
        return found[0]
    for city, code in CITY_STATE.items():
        if re.search(r"\b" + re.escape(city) + r"\b", loc):
            return code
    for code, name in US_STATES.items():
        if re.search(r"\b" + name + r"\b", loc):
            return code
    for abbr, code in STATE_ABBR.items():
        if abbr in loc:
            return code
    m = re.search(r"(?:,\s*|^)([A-Z]{2})\b", loc)
    if m and m.group(1) in US_STATES:
        return m.group(1)
    return ""


# ----------------------------------------------------------------------------- 5. rationale, price, status
def rationale(art: Article, targets: List[str]) -> Tuple[str, List[str]]:
    picked = []
    first = targets[0] if targets else ""
    for s in sentences(art.body):
        if RATIONALE_VERBS.search(s) and (re.search(r"\b(acqui\w*|partnership|combination|transaction|investment|"
                                                     r"together|addition|merger|join\w*|this|will)\b", s, re.I)
                                          or (first and mentions(s, first))):
            if not UNDISCLOSED.search(s) and len(s) > 40:
                picked.append(s)
        if len(picked) == 2:
            break
    text = " ".join(picked)
    tags = [tag for tag, stems in RATIONALE_TAGS.items() if stems_in(text, stems)]
    return text, tags


def price(art: Article) -> Tuple[str, Optional[float], str]:
    """(disclosure, value in USD or None, evidence). A missing price is never zero."""
    for s in sentences(art.body):
        ctx = VALUE_CONTEXT.search(s)
        if ctx and MONEY.search(s[ctx.start():]) and not re.search(r"\bfunds?\b|revenue|sales of|commitments|EBITDA", s, re.I):
            return "Disclosed", money_value(s[ctx.start():]), s
    for s in sentences(art.body):
        if UNDISCLOSED.search(s):
            return "Not disclosed (stated)", None, s
    return "Not stated", None, ""


def status(art: Article, lead: str = "") -> Tuple[str, str]:
    """From the headline, the lead deal sentence and the sentence after it only: later sentences tell the history
    ("Jeff and Scott acquired ISPN in 2020") and would make every announcement look completed."""
    sents = sentences(art.body)
    i = sents.index(lead) if lead in sents else 0
    sents = [strip_dateline(x) for x in sents[i:i + 2]]
    for s in [art.title] + sents:
        if PENDING.search(s or ""):
            return "Announced / pending", s
    for s in [art.title] + sents:
        if COMPLETED.search(s or ""):
            return "Completed", s
    return UNKNOWN, ""


def closing_date(art: Article) -> Tuple[str, str]:
    """Only a date the text ties to the closing ("closed on March 3, 2025"); otherwise unknown."""
    for s in sentences(art.body):
        m = re.search(r"\b(?:closed|completed|finalized|effective)\s+(?:on\s+|as of\s+)?(" + DATE_RE.pattern + ")", s, re.I)
        if m:
            d = DATE_RE.search(m.group(1))
            dd = to_date(d.group(1), d.group(2), d.group(3)) if d else None
            if dd:
                return dd.isoformat(), s
    return "", ""


def ownership(art: Article) -> Tuple[str, str]:
    for s in [art.title] + sentences(art.body)[:6]:
        m = re.search(r"\b(majority|minority|100%|wholly[- ]owned|controlling)\b", s or "", re.I)
        if m:
            w = m.group(1).lower()
            return ("Minority" if w == "minority" else "100%" if w in ("100%", "wholly-owned", "wholly owned")
                    else "Majority / control"), s
    return UNKNOWN, ""


# ----------------------------------------------------------------------------- the record
BLANK = {"transaction_type": "", "type_evidence": "", "deal_status": "", "status_evidence": "", "buyer": "",
         "buyer_role": "", "sponsor": "", "seller": "", "target": "", "n_targets": 0, "parties_evidence": "",
         "target_description": "", "description_source": "", "target_industry": "", "industry_evidence": "",
         "industry_ambiguous": False, "customer_markets": "", "target_location": "", "target_state": "",
         "target_region": "", "rationale": "", "rationale_tags": "", "price_disclosure": "", "deal_value_usd": None,
         "price_evidence": "", "ownership": "", "ownership_evidence": "", "closing_date": "", "closing_evidence": "",
         "review_flags": ""}


def extract(art: Article, source: str, collected_at: str = "", url: str = "") -> dict:
    url = url or art.url
    category, cat_ev = classify_news(art.title, art.body, source)
    firm = SOURCES[source].firm if source in SOURCES else source
    rec = {
        "deal_id": deal_id(url), "source": source, "publisher": firm,
        "url": url, "collected_at": collected_at, "headline": art.title,
        "published_date": art.published_date or "", "published_method": art.published_method,
        "published_evidence": art.published_evidence, "dateline_location": art.dateline_location,
        "news_category": category, "is_deal": category == "Deal", "category_evidence": cat_ev,
    }
    rec.update(BLANK)
    if category != "Deal":
        return rec

    p = parties(art, source)
    ttype, t_ev = transaction_type(p, art, source)
    st, st_ev = status(art, p["lead_sentence"])
    buyer = p["buyer"]
    if ttype == "Exit":
        buyer_role = "Third party (acquirer)" if buyer and not is_publisher(buyer, source) else UNKNOWN
        sponsor, seller = firm, (firm if (not p["seller"] or is_publisher(p["seller"], source)) else p["seller"])
    elif buyer and is_publisher(buyer, source):
        buyer, buyer_role, sponsor, seller = firm, "Publisher (the PE firm itself)", firm, ""
    elif buyer and p["buyer_owned"]:
        buyer_role, sponsor, seller = f"{firm} portfolio company", firm, ""
    elif buyer:
        buyer_role, sponsor, seller = "Not stated as publisher-owned", firm, ""
    else:
        buyer_role, sponsor, seller = UNKNOWN, firm, ""

    targets = p["targets"]
    desc, desc_src = describe_target(targets, p["aliases"], p["target_tail"], art, source, p["lead_sentence"])
    ind, ind_hits, ambiguous = industry_of(desc)
    loc, stc = location_of(targets, p["aliases"], desc, art.body)
    why, tags = rationale(art, targets)
    disclosure, value, price_ev = price(art)
    own, own_ev = ownership(art)
    cdate, c_ev = closing_date(art)

    flags = []
    if not buyer:
        flags.append("buyer not found")
    if not targets:
        flags.append("target not found")
    if len(targets) > 1:
        flags.append("several targets")
    if ttype == "Unclear":
        flags.append("type unclear")
    if not desc:
        flags.append("no target description")
    if ambiguous:
        flags.append("industry ambiguous")
    if not art.published_date:
        flags.append("no publication date")
    elif art.published_method != "page metadata":
        flags.append(f"date from {art.published_method}")

    rec.update({
        "transaction_type": ttype, "type_evidence": t_ev, "deal_status": st, "status_evidence": st_ev,
        "buyer": buyer or UNKNOWN, "buyer_role": buyer_role, "sponsor": sponsor, "seller": seller,
        "target": "; ".join(targets) or UNKNOWN, "n_targets": len(targets), "parties_evidence": p["parties_evidence"],
        "target_description": desc, "description_source": desc_src, "target_industry": ind,
        "industry_evidence": ", ".join(ind_hits), "industry_ambiguous": ambiguous,
        "customer_markets": "; ".join(customer_markets(desc, art.body)), "target_location": loc or UNKNOWN,
        "target_state": stc, "target_region": region_of(stc) if stc else UNKNOWN,
        "rationale": why, "rationale_tags": "; ".join(tags), "price_disclosure": disclosure, "deal_value_usd": value,
        "price_evidence": price_ev, "ownership": own, "ownership_evidence": own_ev, "closing_date": cdate or "Not stated",
        "closing_evidence": c_ev, "review_flags": "; ".join(flags),
    })
    return rec
