"""
Deterministic text features from SEC filings: the same document always gives the same numbers, so they can be
backtested (an LLM's reading of a filing cannot: see docs/DATA_AND_ML.md).

Cleaning     `html_to_text` drops scripts, styles and the hidden inline-XBRL header, strips tags and entities;
             `tokens` keeps words of two or more letters, upper-cased (the Loughran-McDonald convention)
Sections     `risk_factors` finds Item 1A (Risk Factors) in a 10-K: the longest span from an "Item 1A" heading to the
             next "Item 1B" / "Item 2" heading, so the table of contents is not mistaken for the section
Tone         the share of words in each Loughran-McDonald (2011) list: negative, positive, uncertainty, litigious,
             constraining, strong and weak modal. The master dictionary (University of Notre Dame SRAF) is free for
             academic research; commercial use needs a licence from its authors. Lighthouse never redistributes it:
             `lm_dictionary` downloads it into your cache (data/cache, gitignored) the first time it is needed.
Change       cosine similarity of this filing's word counts with the same company's previous filing of the same form
             (Cohen, Malloy & Nguyen 2020, "Lazy Prices": firms that change their 10-K's language later underperform)
Readability  the Gunning fog index (Li 2008): 0.4 x (words per sentence + 100 x share of words of 3+ syllables)
"""
from __future__ import annotations

import html as _html
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

LM_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
LM_PAGE = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
LM_CATEGORIES = ("negative", "positive", "uncertainty", "litigious", "constraining", "strong_modal", "weak_modal")
LM_CITATION = ("Loughran, T. and McDonald, B. (2011), When is a Liability not a Liability? Textual Analysis, "
               "Dictionaries, and 10-Ks, Journal of Finance 66(1); master dictionary from " + LM_PAGE)

_DROP = re.compile(r"<(script|style|ix:header|head)\b.*?</\1\s*>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z]+")
_SENT = re.compile(r"[.!?]+(?:\s|$)")


def html_to_text(doc: str) -> str:
    """Plain text from an EDGAR HTML (or plain-text) filing, line breaks kept at block boundaries."""
    s = _DROP.sub(" ", doc)
    s = re.sub(r"<\s*(br|/p|/div|/tr|/li|/h\d)\b[^>]*>", "\n", s, flags=re.I)
    s = _TAG.sub(" ", s)
    s = _html.unescape(s).replace("\xa0", " ")
    s = _WS.sub(" ", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def tokens(text: str) -> List[str]:
    return [w.upper() for w in _WORD.findall(text)]


_ITEM_1A = re.compile(r"item\s*1a\.?\s*[\-–—:]?\s*risk\s+factors", re.I)
_ITEM_END = re.compile(r"item\s*(1b|2)\.?\s*[\-–—:]?\s*(unresolved|properties)", re.I)


def risk_factors(text: str) -> str:
    """The Item 1A section of a 10-K (empty when it cannot be found)."""
    best = ""
    for m in _ITEM_1A.finditer(text):
        end = _ITEM_END.search(text, m.end())
        span = text[m.end(): end.start() if end else min(len(text), m.end() + 400_000)]
        if len(span) > len(best):
            best = span
    return best if len(best) > 2_000 else ""         # a table-of-contents line is not the section


@dataclass
class LMDictionary:
    words: Dict[str, Set[str]]                       # category -> words
    syllables: Dict[str, int] = field(default_factory=dict)
    source: str = ""


def parse_lm_csv(path_or_buffer) -> LMDictionary:
    """The master dictionary CSV: a word is in a category when its flag (the year it was added) is positive; a
    negative year means it was removed."""
    df = pd.read_csv(path_or_buffer, keep_default_na=False, dtype={"Word": str})
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    words = {}
    for cat in LM_CATEGORIES:
        c = cols.get(cat)
        if c is None:
            continue
        flag = pd.to_numeric(df[c], errors="coerce").fillna(0)
        words[cat] = set(df.loc[flag > 0, "Word"].str.upper())
    syl = {}
    if "syllables" in cols:
        s = pd.to_numeric(df[cols["syllables"]], errors="coerce")
        syl = dict(zip(df["Word"].str.upper(), s.fillna(0).astype(int)))
    return LMDictionary(words, syl, str(path_or_buffer))


def lm_dictionary(refresh: bool = False) -> LMDictionary:
    """The Loughran-McDonald master dictionary, downloaded once into the cache (see the module docstring on its terms)."""
    import io
    from . import store
    from .data import DataUnavailable, offline_mode
    path = store.path_for("text", "lm_master_dictionary").with_suffix(".csv")
    if path.exists() and not refresh:
        return parse_lm_csv(path)
    if offline_mode():
        raise DataUnavailable("the Loughran-McDonald dictionary is not in the cache (offline)")
    from .http import fetch
    body = fetch(LM_URL, timeout=120).content
    if not body.startswith(b"Word,"):
        raise DataUnavailable("the Loughran-McDonald dictionary download did not return the CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    return parse_lm_csv(io.BytesIO(body))


def tone(words: Iterable[str], lm: LMDictionary) -> Dict[str, float]:
    """Share of the words in each list, and the word count."""
    c = Counter(words)
    n = sum(c.values())
    out = {"words": float(n)}
    for cat, vocab in lm.words.items():
        out[cat] = float(sum(v for w, v in c.items() if w in vocab) / n) if n else float("nan")
    return out


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    common = set(a) & set(b)
    num = float(sum(a[w] * b[w] for w in common))
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else float("nan")


def _syllables(word: str, lm: Optional[LMDictionary]) -> int:
    if lm is not None and lm.syllables.get(word):
        return int(lm.syllables[word])
    groups = re.findall(r"[AEIOUY]+", word)
    n = len(groups) - (1 if word.endswith("E") and len(groups) > 1 else 0)
    return max(n, 1)


def fog(text: str, lm: Optional[LMDictionary] = None) -> float:
    """Gunning fog index of the text (higher = harder to read)."""
    ws = tokens(text)
    sentences = max(len(_SENT.findall(text)), 1)
    if not ws:
        return float("nan")
    complex_share = sum(_syllables(w, lm) >= 3 for w in ws) / len(ws)
    return 0.4 * (len(ws) / sentences + 100.0 * complex_share)


def features(doc_html: str, lm: LMDictionary, previous: Optional[Dict[str, Counter]] = None) -> tuple:
    """(features, counts) for one filing. `previous` is the counts of the company's last filing of the same form
    ({'full': Counter, 'risk': Counter}); the counts returned are what the next filing compares itself with."""
    text = html_to_text(doc_html)
    full = tokens(text)
    risk_text = risk_factors(text)
    risk = tokens(risk_text)
    cf, cr = Counter(full), Counter(risk)
    out = {f"lm_{k}": v for k, v in tone(full, lm).items() if k != "words"}
    out["words"] = float(len(full))
    out["risk_words"] = float(len(risk))
    out["risk_uncertainty"] = tone(risk, lm).get("uncertainty", float("nan")) if risk else float("nan")
    out["fog"] = fog(text[:2_000_000], lm)
    prev = previous or {}
    out["similarity"] = cosine(cf, prev.get("full", Counter()))
    out["risk_similarity"] = cosine(cr, prev.get("risk", Counter())) if risk else float("nan")
    pw = sum(prev.get("full", Counter()).values())
    out["length_change"] = float(np.log(len(full) / pw)) if pw and full else float("nan")
    return out, {"full": cf, "risk": cr}


__all__ = ["html_to_text", "tokens", "risk_factors", "LMDictionary", "parse_lm_csv", "lm_dictionary", "tone", "cosine",
           "fog", "features", "LM_CATEGORIES", "LM_CITATION", "LM_URL"]
