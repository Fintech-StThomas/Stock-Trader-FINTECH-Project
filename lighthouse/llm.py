"""
Optional language-model reading of filings: off by default, local only, and checked.

What it is for   questions the deterministic features cannot answer ("what changed in the risk factors?", "argue
                 against this memo"). The model reads numbered passages from the company's own filing and a table of
                 numbers from structured data (XBRL, prices), and must cite a passage for every sentence.
Checks           every answer is verified before it is shown: sentences without a [n] citation, citations to passages
                 that were not given, and numbers that appear in neither the passages nor the numbers table are listed
                 next to the answer (`LLMAnswer.problems`). A failed check is shown, never silently fixed.
Red team         `red_team(memo)` asks for the strongest case against the memo's stance, under the same rules.
Providers        `ollama` (a local server, default http://localhost:11434, free) when LIGHTHOUSE_LLM=ollama;
                 `anthropic` is a stub that refuses to run (a paid API: outside this project's $0 budget).
Limits           an LLM's judgment cannot be backtested: the model has read text written after any past date, so its
                 view of 2015 is contaminated by what happened next, and the same prompt can give different answers.
                 Nothing it writes feeds a score, a verdict, a ticket or a backtest (docs/DATA_AND_ML.md).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

PROVIDER_ENV = "LIGHTHOUSE_LLM"
OLLAMA_HOST_ENV = "OLLAMA_HOST"
OLLAMA_MODEL_ENV = "LIGHTHOUSE_OLLAMA_MODEL"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

RULES = ("Answer only from the numbered passages and the numbers table below. End every sentence with the passage "
         "number(s) it rests on, like [2] or [1][3]. Use only numbers that appear in the passages or the table; do not "
         "compute new ones. If the passages do not answer the question, say so. Be brief.")


class LLMDisabled(RuntimeError):
    """The language-model layer is off (the default) or the provider is not available."""


@dataclass
class LLMAnswer:
    question: str
    text: str
    model: str
    passages: List[str]
    prompt_sha: str
    problems: List[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return not self.problems

    def cited(self) -> List[int]:
        return sorted({int(x) for x in re.findall(r"\[(\d+)\]", self.text)})


# --------------------------------------------------------------------------- #
# Passages and checks (deterministic: testable without a model)
# --------------------------------------------------------------------------- #
def split_passages(text: str, max_chars: int = 900) -> List[str]:
    """Paragraph-sized passages (long paragraphs split at sentence ends)."""
    out: List[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if len(para) < 40:
            continue
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            out.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            out.append(para)
    return out


def select_passages(passages: Sequence[str], question: str, k: int = 8) -> List[str]:
    """The k passages sharing the most (rarer) words with the question, in document order."""
    from math import log
    from .text import tokens
    q = set(tokens(question)) - {"THE", "AND", "OF", "TO", "IN", "WHAT", "IS", "ARE", "FOR", "ON", "A", "AN", "IT"}
    docs = [set(tokens(p)) for p in passages]
    n = len(docs)
    df = {w: sum(w in d for d in docs) for w in q}
    score = [sum(log((n + 1) / (df[w] + 0.5)) for w in q if w in d) for d in docs]
    top = sorted(range(n), key=lambda i: (-score[i], i))[:k]
    return [passages[i] for i in sorted(top)]


_NUM = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def _numbers(text: str) -> set:
    out = set()
    for m in _NUM.findall(text):
        v = m.replace("$", "").replace(",", "").replace("%", "").lstrip("+")
        if v.strip("-") and not re.fullmatch(r"-?\d{1}", v):         # single digits (list markers, years of text) pass
            out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def check_answer(text: str, passages: Sequence[str], numbers: Dict[str, object]) -> List[str]:
    """Problems with an answer: uncited sentences, citations to missing passages, numbers with no source."""
    problems = []
    body = re.sub(r"\s+", " ", text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body) if len(s.strip()) > 20]
    for s in sentences:
        if not re.search(r"\[\d+\]", s):
            problems.append(f"no citation: \"{s[:90]}\"")
    for c in {int(x) for x in re.findall(r"\[(\d+)\]", text)}:
        if not 1 <= c <= len(passages):
            problems.append(f"cites passage [{c}], which was not given")
    allowed = set()
    for p in passages:
        allowed |= _numbers(p)
    for v in numbers.values():
        allowed |= _numbers(str(v))
    stripped = re.sub(r"\[\d+\]", " ", text)
    for n in sorted(_numbers(stripped) - allowed):
        problems.append(f"number {n} is in neither the passages nor the numbers table")
    return problems


def build_prompt(question: str, passages: Sequence[str], numbers: Dict[str, object], role: str = "") -> str:
    lines = [role or "You are a careful equity analyst reading a company's SEC filing.", RULES, "", "PASSAGES"]
    lines += [f"[{i}] {p}" for i, p in enumerate(passages, 1)]
    lines += ["", "NUMBERS (from structured data)"]
    lines += [f"- {k}: {v}" for k, v in numbers.items()]
    lines += ["", f"QUESTION: {question}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def provider() -> str:
    return (os.environ.get(PROVIDER_ENV) or "off").strip().lower()


def _ollama(prompt: str, model: Optional[str] = None, host: Optional[str] = None, timeout: float = 300.0) -> tuple:
    from .data import DataUnavailable
    from .http import fetch
    host = (host or os.environ.get(OLLAMA_HOST_ENV) or DEFAULT_OLLAMA_HOST).rstrip("/")
    model = model or os.environ.get(OLLAMA_MODEL_ENV) or DEFAULT_OLLAMA_MODEL
    try:
        r = fetch(f"{host}/api/generate", method="POST", timeout=timeout, retries=1,
                  json_body={"model": model, "prompt": prompt, "stream": False,
                             "options": {"temperature": 0, "seed": 7}})
    except DataUnavailable as exc:
        raise LLMDisabled(f"no Ollama server at {host} ({exc}); start one with `ollama serve` and `ollama pull {model}`") \
            from exc
    return str(r.json().get("response", "")).strip(), f"ollama:{model}"


def _anthropic(prompt: str, **_) -> tuple:
    raise LLMDisabled("the Anthropic adapter is a stub: a paid API is outside this project's $0 budget. "
                      "Use LIGHTHOUSE_LLM=ollama with a local model instead.")


PROVIDERS = {"ollama": _ollama, "anthropic": _anthropic}


def ask(question: str, passages: Sequence[str], numbers: Optional[Dict[str, object]] = None, role: str = "",
        **kw) -> LLMAnswer:
    """One checked answer from the configured provider. Raises LLMDisabled when the layer is off."""
    name = provider()
    if name in ("", "off", "0", "false", "none"):
        raise LLMDisabled(f"the language-model layer is off; set {PROVIDER_ENV}=ollama to use a local model")
    if name not in PROVIDERS:
        raise LLMDisabled(f"unknown provider {name!r}; choose one of {sorted(PROVIDERS)}")
    numbers = numbers or {}
    prompt = build_prompt(question, passages, numbers, role)
    text, model = PROVIDERS[name](prompt, **kw)
    return LLMAnswer(question, text, model, list(passages), hashlib.sha256(prompt.encode()).hexdigest()[:16],
                     check_answer(text, passages, numbers))


def memo_numbers(memo) -> Dict[str, object]:
    """The numbers table for a memo: TTM lines, ratios, the DCF values (all from structured data)."""
    f, r = memo.financials, memo.financials.ratios()
    out: Dict[str, object] = {"price": f"${f.price:,.2f}", "revenue (TTM)": f"${f.get('revenue') / 1e9:,.1f}bn",
                              "operating margin": f"{r['operating_margin'] * 100:.1f}%",
                              "revenue growth": f"{r['revenue_growth'] * 100:.1f}%"}
    if memo.valuation is not None:
        v = memo.valuation
        out.update({f"{k} case value": f"${c.per_share:,.2f}" for k, c in v.cases.items()})
        out["weighted value"] = f"${v.target:,.2f}"
        if v.implied_growth is not None:
            out["growth implied by the price"] = f"{v.implied_growth * 100:.1f}%"
    return out


def red_team(memo, filing_text: str, k: int = 8, **kw) -> LLMAnswer:
    """The strongest case against the memo's stance, from the filing's own words."""
    passages = select_passages(split_passages(filing_text), "risk decline competition loss debt litigation "
                               "uncertain regulation customers margin", k)
    q = (f"The memo's stance on {memo.ticker} is {memo.stance.upper()}: {memo.thesis} Give the three strongest "
         f"reasons, from these passages, that this stance is wrong.")
    return ask(q, passages, memo_numbers(memo), role="You are a skeptical reviewer whose job is to find what the "
               "analyst missed.", **kw)


__all__ = ["LLMDisabled", "LLMAnswer", "split_passages", "select_passages", "check_answer", "build_prompt", "ask",
           "red_team", "memo_numbers", "provider", "PROVIDERS"]
