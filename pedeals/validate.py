"""
Validation: how far can the user trust the records, and where do the rules fail.

Three kinds of evidence, kept apart (docs/PE_DEALS.md §6):

1. **Reference labels** (data/pe_deals/validation/reference_labels.csv): a sample of articles labelled by reading
   the source page, field by field ("none" = the text gives no value, so the program must leave it unknown; a
   blank cell is not scored). Accuracy per field is the share of
   labelled articles where the program's value matches the label (names compared after normalisation: case,
   punctuation and Inc./LLC suffixes ignored; one name containing the other counts). Every mismatch is listed for
   investigation. This is the validation evidence; the review flags the program prints are not.
2. **Consistency checks** on every record, no labels needed: a date in the future, a publisher recorded as its own
   add-on buyer, a price of zero, a deal without its source URL or evidence, and so on. Each should find nothing.
3. **Repeatability**: building the dataset twice from the same saved pages must give identical records.

Labels are never used to correct the outputs: a field the rules get wrong stays wrong in the dataset (and is
counted here), so the accuracy reported is the accuracy the user gets.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .collect import ROOT
from .pipeline import canonical_url, norm_name

LABELS_FILE = ROOT / "data" / "pe_deals" / "validation" / "reference_labels.csv"
RESULTS_FILE = ROOT / "data" / "pe_deals" / "validation" / "results.md"
# label column -> (program column, how to compare); deal-only fields are scored on articles labelled as deals
FIELDS = {
    "is_deal": ("is_deal", "bool"),
    "news_category": ("news_category", "exact"),
    "published_date": ("published_date", "exact"),
    "transaction_type": ("transaction_type", "exact"),
    "buyer": ("buyer", "name"),
    "target": ("target", "name"),
    "target_industry": ("target_industry", "exact"),
    "target_state": ("target_state", "exact"),
    "deal_status": ("deal_status", "exact"),
    "price_disclosure": ("price_disclosure", "exact"),
}
DEAL_ONLY = {"transaction_type", "buyer", "target", "target_industry", "target_state", "deal_status", "price_disclosure"}


def _bool(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def names_match(a: str, b: str) -> bool:
    """Same company name after normalisation; one containing the other counts ('Jones Bearing' ~ 'Jones Bearing Co.').
    Several names (';') match when every labelled name is found."""
    la = [norm_name(x) for x in str(a or "").split(";") if norm_name(x)]
    lb = [norm_name(x) for x in str(b or "").split(";") if norm_name(x)]
    if not la and not lb:
        return True
    if not la or not lb:
        return False
    return all(any(x == y or (len(x) >= 4 and len(y) >= 4 and (x in y or y in x)) for y in lb) for x in la)


def _empty(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() in ("", "Unknown")


def compare(label, value, how: str) -> bool:
    """A label of "none" says the text gives no value: the program must leave the field empty / Unknown."""
    if how == "bool":
        return _bool(label) == bool(value)
    if str(label).strip().lower() == "none":
        return _empty(value)
    if how == "name":
        return names_match(label, value)
    lab = "" if label is None or (isinstance(label, float) and math.isnan(label)) else str(label).strip()
    val = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()
    return lab.lower() == val.lower()


@dataclass
class ValidationReport:
    summary: pd.DataFrame                 # per field: labelled n, correct, accuracy
    classification: Dict[str, float]      # deal vs non-deal precision / recall / F1
    type_confusion: pd.DataFrame
    errors: pd.DataFrame                  # every mismatch, for investigation
    consistency: pd.DataFrame             # per check: records failing
    repeatable: bool
    n_labels: int
    notes: List[str] = field(default_factory=list)
    split: str = "held-out"

    def markdown(self) -> str:
        c = self.classification
        lines = [f"## Validation against {self.n_labels} reference labels ({self.split})", "",
                 "| Field | Labelled | Correct | Accuracy (95% interval) |", "|---|---|---|---|"]
        for r in self.summary.itertuples():
            lines.append(f"| {r.field} | {r.n} | {r.correct} | {r.accuracy:.0%} ({r.ci_low:.0%}-{r.ci_high:.0%}) |")
        lines += ["", f"Deal vs other news: precision {c['precision']:.0%}, recall {c['recall']:.0%}, F1 {c['f1']:.0%} "
                      f"({c['tp']} deals found, {c['fp']} false alarms, {c['fn']} missed).", "",
                  "### Consistency checks (every record)", "", "| Check | Records failing |", "|---|---|"]
        lines += [f"| {r.check} | {r.failing} |" for r in self.consistency.itertuples()]
        lines += ["", f"Repeatability: two builds from the same saved pages are {'identical' if self.repeatable else 'DIFFERENT'}."]
        if len(self.errors):
            lines += ["", f"### Mismatches ({len(self.errors)})", "", "| Article | Field | Label | Program |", "|---|---|---|---|"]
            for r in self.errors.itertuples():
                lines.append(f"| {str(r.headline)[:60]} | {r.field} | {str(r.label)[:50]} | {str(r.program)[:50]} |")
        lines += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval of an accuracy of k out of n (honest about small samples)."""
    if not n:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_labels(path: Path = LABELS_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["url", *FIELDS])
    return pd.read_csv(path, dtype=str).fillna("")


def consistency_checks(ds) -> pd.DataFrame:
    r, d = ds.records, ds.deals
    today = date.today().isoformat()
    checks = {
        "publication date after the collection date": (r["published_date"].fillna("") > r["collected_at"].fillna("").str[:10]) & (r["published_date"].fillna("") != ""),
        "publication date in the future": r["published_date"].fillna("") > today,
        "deal without a source URL": d["url"].fillna("") == "",
        "deal without category evidence": d["category_evidence"].fillna("") == "",
        "publisher recorded as an add-on buyer": (d["transaction_type"] == "Add-on acquisition") & d.apply(
            lambda x: norm_name(x["buyer"]) == norm_name(x["publisher"]), axis=1),
        "platform investment not by the publisher": (d["transaction_type"] == "Platform investment") & d.apply(
            lambda x: norm_name(x["buyer"]) != norm_name(x["publisher"]), axis=1),
        "buyer and target the same": d.apply(lambda x: x["target"] != "Unknown" and norm_name(x["buyer"]) == norm_name(x["target"]), axis=1),
        "price recorded as zero": d["deal_value_usd"].fillna(-1) == 0,
        "price disclosed without a value": (d["price_disclosure"] == "Disclosed") & d["deal_value_usd"].isna(),
        "closing date before the publication date by > 1 year": d.apply(
            lambda x: bool(re.match(r"\d{4}-", str(x["closing_date"]))) and x["closing_date"] < str(int(x["published_date"][:4]) - 1), axis=1),
        "duplicate deal ids": r["deal_id"].duplicated(),
        "deal with a type but no buyer and no target": (d["buyer"] == "Unknown") & (d["target"] == "Unknown"),
    }
    return pd.DataFrame([{"check": k, "failing": int(pd.Series(v).sum())} for k, v in checks.items()])


def evaluate(ds, labels: pd.DataFrame = None, split: str = "held-out") -> ValidationReport:
    """Score the dataset against the labels of one split: "held-out" (labelled after the rules were frozen: the
    accuracy to quote), "development" (the labels the rules were tuned on: in-sample, optimistic) or "all"."""
    labels = load_labels() if labels is None else labels
    if split != "all" and "split" in labels.columns:
        labels = labels[labels["split"] == split]
    rec = ds.records.assign(_cu=ds.records["url"].map(canonical_url)).set_index("_cu")
    rows, errors = [], []
    matched = 0
    for lab in labels.itertuples(index=False):
        lab = lab._asdict()
        cu = canonical_url(lab["url"])
        if cu not in rec.index:
            continue
        matched += 1
        prog = rec.loc[cu]
        is_deal_label = _bool(lab.get("is_deal"))
        for f, (col, how) in FIELDS.items():
            if f not in lab or (f in DEAL_ONLY and not is_deal_label):
                continue
            if f != "is_deal" and str(lab[f]).strip() == "":
                continue        # the labeller left it blank: not scored
            ok = compare(lab[f], prog[col], how)
            rows.append({"field": f, "ok": ok})
            if not ok:
                errors.append({"headline": prog["headline"], "url": lab["url"], "field": f, "label": lab[f], "program": prog[col]})
    res = pd.DataFrame(rows, columns=["field", "ok"])
    summary = (res.groupby("field", sort=False)["ok"].agg(n="count", correct="sum").reset_index()
               if len(res) else pd.DataFrame(columns=["field", "n", "correct"]))
    summary["accuracy"] = summary["correct"] / summary["n"].where(summary["n"] > 0)
    ci = [wilson(int(k), int(n)) for k, n in zip(summary["correct"], summary["n"])]
    summary["ci_low"], summary["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    summary = summary.set_index("field").reindex([f for f in FIELDS if f in set(summary["field"])]).reset_index()

    lab_d = labels.assign(_cu=labels["url"].map(canonical_url))
    lab_d = lab_d[lab_d["_cu"].isin(rec.index)]
    y = lab_d["is_deal"].map(_bool).to_numpy()
    p = rec.loc[lab_d["_cu"], "is_deal"].astype(bool).to_numpy()
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * prec * recall / (prec + recall) if prec and recall and not math.isnan(prec + recall) else math.nan

    deal_lab = lab_d[lab_d["is_deal"].map(_bool)]
    conf = pd.crosstab(deal_lab["transaction_type"].rename("label"),
                       rec.loc[deal_lab["_cu"], "transaction_type"].rename("program").to_numpy()) if len(deal_lab) else pd.DataFrame()

    from .pipeline import build_dataset
    again = build_dataset(start=ds.start, pages_file=Path(ds.pages_file))
    notes = [f"{len(labels) - matched} labelled URLs are not in this dataset (outside the period or not collected)."
             if len(labels) - matched else "Every labelled URL is in the dataset.",
             "Labels were made by reading each source page; see data/pe_deals/validation/README.md for who labelled "
             "what and how. The intervals are 95% Wilson intervals: with 30-50 labelled deals they are wide."]
    return ValidationReport(summary=summary, classification={"precision": prec, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
                            type_confusion=conf, errors=pd.DataFrame(errors, columns=["headline", "url", "field", "label", "program"]),
                            consistency=consistency_checks(ds), repeatable=again.fingerprint() == ds.fingerprint(),
                            n_labels=matched, notes=notes, split=split)


def full_report(ds, labels: pd.DataFrame = None) -> str:
    """The validation write-up: held-out accuracy (the number to quote) next to the development accuracy, the
    consistency checks, repeatability and every mismatch."""
    held, dev = evaluate(ds, labels, "held-out"), evaluate(ds, labels, "development")
    acc = lambda r, f: r.summary.set_index("field").loc[f] if f in set(r.summary["field"]) else None  # noqa: E731
    lines = ["# PE Deal Intelligence: validation", "",
             f"Reference labels: {held.n_labels} held-out articles (labelled after the extraction rules were frozen in git) "
             f"and {dev.n_labels} development articles (the rules were tuned on these, so their accuracy is in-sample).", "",
             "| Field | Held-out: correct / labelled | Held-out accuracy (95% interval) | Development accuracy |", "|---|---|---|---|"]
    for f in FIELDS:
        h, d = acc(held, f), acc(dev, f)
        if h is None:
            continue
        lines.append(f"| {f} | {int(h['correct'])} / {int(h['n'])} | {h['accuracy']:.0%} ({h['ci_low']:.0%}-{h['ci_high']:.0%}) | "
                     + (f"{d['accuracy']:.0%} ({int(d['correct'])}/{int(d['n'])})" if d is not None else "—") + " |")
    c = held.classification
    lines += ["", f"Deal vs other news (held-out): precision {c['precision']:.0%}, recall {c['recall']:.0%} "
                  f"({c['tp']} deals found, {c['fp']} false alarms, {c['fn']} missed).", "",
              "## Consistency checks (every record, no labels needed)", "", "| Check | Records failing |", "|---|---|"]
    lines += [f"| {r.check} | {r.failing} |" for r in held.consistency.itertuples()]
    lines += ["", f"Repeatability: two builds from the same saved pages are {'identical' if held.repeatable else 'DIFFERENT'} "
                  f"(fingerprint `{ds.fingerprint()}`)."]
    for name, rep in (("held-out", held), ("development", dev)):
        if len(rep.errors):
            lines += ["", f"## Mismatches, {name} ({len(rep.errors)})", "", "| Article | Field | Label | Program |", "|---|---|---|---|"]
            lines += [f"| {str(r.headline)[:70]} | {r.field} | {str(r.label)[:60]} | {str(r.program)[:60]} |" for r in rep.errors.itertuples()]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in held.notes]
    return "\n".join(lines) + "\n"
