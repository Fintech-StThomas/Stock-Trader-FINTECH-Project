"""
The repeatable workflow: saved pages -> parsed articles -> deal records -> de-duplicated, period-filtered dataset.

    from pedeals import build_dataset
    ds = build_dataset()            # runs on data/pe_deals/cache/pages.jsonl.gz: offline, same result every time
    ds.deals                        # deal announcements in the period, one row each, duplicates marked
    ds.export()                     # CSV / JSON / Markdown into data/pe_deals/output/

Duplicates. The same URL is only ever stored once (URLs are normalised: lower-case host, no query, no trailing
slash). The same announcement published twice by one firm (same headline) and the same deal announced by two firms
(same target, published within DUPLICATE_WINDOW days of each other) are kept but marked: `duplicate_of` names the
first record, and the analysis uses `unique_deals`.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

from .collect import DEFAULT_START, DISCOVERY_FILE, LISTINGS_FILE, PAGES_FILE, ROOT, load_pages
from .extract import extract
from .parse import Article, listing_dates, parse_article
from .sources import SOURCES

OUTPUT_DIR = ROOT / "data" / "pe_deals" / "output"
DUPLICATE_WINDOW = 120       # days: the same target announced twice within this window is one deal
CACHE_NOTE = ("CACHED DATA: built from pages saved by `python -m pedeals collect` (fetch time in `collected_at`); "
              "the live sites may have changed since.")


def canonical_url(url: str) -> str:
    p = urlparse((url or "").strip())
    return f"{p.scheme.lower() or 'https'}://{p.netloc.lower()}{p.path.rstrip('/') or '/'}"


def norm_name(name: str) -> str:
    n = (name or "").lower()
    n = re.sub(r"[“”\"'’.,()]", " ", n)
    n = re.sub(r"\b(inc|llc|ltd|corp|corporation|company|co|lp|holdings|group|the)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


@dataclass
class DealDataset:
    records: pd.DataFrame                  # every collected article (deal or not), extraction columns filled for deals
    coverage: pd.DataFrame                 # per firm: what the sitemaps listed, what was fetched, parsed, kept
    failures: pd.DataFrame                 # pages that could not be fetched or parsed, with the reason
    start: str
    built_at: str
    pages_file: str
    note: str = CACHE_NOTE
    meta: Dict = field(default_factory=dict)

    @property
    def deals(self) -> pd.DataFrame:
        """Deal announcements published in the period (duplicates marked, not removed)."""
        r = self.records
        return r[r["is_deal"] & r["in_period"]].reset_index(drop=True)

    @property
    def unique_deals(self) -> pd.DataFrame:
        d = self.deals
        return d[d["duplicate_of"] == ""].reset_index(drop=True)

    def fingerprint(self) -> str:
        """A hash of the extracted records: two runs on the same saved pages must give the same one."""
        cols = [c for c in self.records.columns if c not in ("collected_at",)]
        blob = self.records[cols].sort_values("url").to_csv(index=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def summary_text(self) -> str:
        d, u = self.deals, self.unique_deals
        lines = [f"PE Deal Intelligence dataset ({self.note})",
                 f"period: published {self.start} to {self.meta.get('latest_published', '?')}; built {self.built_at}",
                 f"{len(self.records)} articles parsed from {self.records['source'].nunique()} firms; {len(d)} deal "
                 f"announcements in the period, {len(u)} unique deals, {len(self.failures)} failed pages"]
        if len(u):
            lines.append(u.groupby(["publisher", "transaction_type"]).size().rename("deals").to_string())
        return "\n".join(lines)

    def export(self, out: Optional[str] = None) -> List[str]:
        out_dir = Path(out) if out else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for name, df in (("deals", self.deals), ("all_articles", self.records.drop(columns=["body"], errors="ignore")),
                         ("coverage", self.coverage), ("failures", self.failures)):
            p = out_dir / f"{name}.csv"
            df.to_csv(p, index=False)
            written.append(str(p))
        p = out_dir / "deals.json"
        payload = {"note": self.note, "period_start": self.start, "built_at": self.built_at, "fingerprint": self.fingerprint(),
                   "deals": json.loads(self.deals.to_json(orient="records"))}
        p.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        written.append(str(p))
        from .analytics import dataset_brief
        p = out_dir / "deal_brief.md"
        p.write_text(dataset_brief(self))
        written.append(str(p))
        return written


def _names_match(a: str, b: str) -> bool:
    x, y = norm_name(a), norm_name(b)
    return bool(x) and bool(y) and x != "unknown" and y != "unknown" and (x == y or (min(len(x), len(y)) >= 6 and (x in y or y in x)))


def infer_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """A buyer the release does not call the publisher's company, but which the publisher's other announcements show
    it owned at the time: the publisher invested in it on or before this date, it made an add-on for the publisher,
    or the publisher sold it on or after this date. The deal becomes an add-on, flagged as inferred, with the other
    announcement as the evidence."""
    priority = {"Platform investment": 0, "Growth / minority investment": 0, "Add-on acquisition": 1, "Exit": 2}
    owned_rows = df[df["is_deal"] & df["transaction_type"].isin(list(priority))]
    # the clearest evidence first: the publisher's own investment, then dated add-ons, then a later sale
    owned_rows = owned_rows.assign(_p=owned_rows["transaction_type"].map(priority),
                                   _undated=owned_rows["published_date"].fillna("") == "").sort_values(["_p", "_undated", "published_date"])
    for i, r in df[df["is_deal"] & (df["transaction_type"] == "Unclear") & (df["buyer"] != "Unknown")].iterrows():
        for o in owned_rows[owned_rows["source"] == r["source"]].itertuples():
            if o.transaction_type == "Add-on acquisition":
                ok = _names_match(r["buyer"], o.buyer)
            elif o.transaction_type == "Exit":
                ok = any(_names_match(r["buyer"], t) for t in o.target.split(";")) and o.published_date >= r["published_date"]
            else:
                ok = any(_names_match(r["buyer"], t) for t in o.target.split(";")) and o.published_date <= r["published_date"]
            if ok:
                df.at[i, "transaction_type"] = "Add-on acquisition"
                df.at[i, "buyer_role"] = f"{r['publisher']} portfolio company (from another announcement)"
                df.at[i, "type_evidence"] = (f"{r['buyer']} appears as a {r['publisher']} company in \"{o.headline}\" "
                                             f"({o.published_date}, {o.url})")
                flags = [f for f in r["review_flags"].split("; ") if f and f != "type unclear"]
                df.at[i, "review_flags"] = "; ".join(flags + ["relationship inferred from another announcement"])
                break
    return df


def platform_keys(df: pd.DataFrame) -> pd.Series:
    """One key per buying company within a sponsor: its short name when the release gives one (so "Marco Rubber &
    Plastics" and, after its rename, "Marco Sealing Solutions" are one platform), else its name without suffixes;
    names that contain one another ("Counsel Press" / "Counsel Press Inc") share the shorter key."""
    keys = [f"{src}|{norm_name(short) or norm_name(b)}" for src, short, b in zip(df["source"], df["buyer_short"].fillna(""), df["buyer"].fillna(""))]
    by_src: Dict[str, List[str]] = {}
    for k in set(keys):
        by_src.setdefault(k.split("|")[0], []).append(k.split("|", 1)[1])
    out = []
    for k in keys:
        src, name = k.split("|", 1)
        shorter = [n for n in by_src[src] if n and n != name and len(n) >= 4 and (name.startswith(n + " ") or name == n)]
        out.append(f"{src}|{min(shorter, key=len) if shorter else name}")
    return pd.Series(out, index=df.index)


def _mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["published_date", "url"]).reset_index(drop=True)
    df["duplicate_of"] = ""
    df["also_reported_by"] = ""
    seen_head: Dict[tuple, str] = {}
    seen_target: Dict[str, List[tuple]] = {}
    for i, r in df.iterrows():
        if not r["is_deal"]:
            continue
        hk = (r["source"], norm_name(r["headline"]))
        if hk in seen_head:
            df.at[i, "duplicate_of"] = seen_head[hk]
            continue
        seen_head[hk] = r["deal_id"]
        tk = norm_name(r["target"])
        if not tk or tk == "unknown" or not r["published_date"]:
            continue
        d = date.fromisoformat(r["published_date"])
        for j, (d0, did, src) in enumerate(seen_target.get(tk, [])):
            if abs((d - d0).days) <= DUPLICATE_WINDOW:
                df.at[i, "duplicate_of"] = did
                k = df.index[df["deal_id"] == did][0]
                df.at[k, "also_reported_by"] = "; ".join(x for x in [df.at[k, "also_reported_by"], r["publisher"]] if x)
                break
        else:
            seen_target.setdefault(tk, []).append((d, r["deal_id"], r["source"]))
    return df


def listed_dates(listings_file: Path = LISTINGS_FILE) -> Dict[str, dict]:
    """canonical article URL -> the date its firm's news listing page prints next to it."""
    out: Dict[str, dict] = {}
    for url, pg in load_pages(listings_file).items():
        src = SOURCES.get(pg.get("source", ""))
        if src is None or not pg.get("html"):
            continue
        for d in listing_dates(pg["html"], url, src.is_article):
            d["listing_url"] = url
            out.setdefault(canonical_url(d["url"]), d)
    return out


def apply_listing_date(art: Article, listed: Optional[dict]) -> None:
    """Order of trust: page metadata > a dateline in the release > the listing's full date > the first date in the
    text > the listing's month. A month-only date is stored as the first of the month, precision 'month', flagged."""
    if not listed:
        return
    if listed["precision"] == "day" and art.published_method in ("not found", "first date in text"):
        art.published_precision = "day"
    elif listed["precision"] == "month" and art.published_method == "not found":
        art.published_precision = "month"
    else:
        return
    art.published_date, art.published_method = listed["listed_date"], "news listing page"
    art.published_evidence = f'"{listed["evidence"]}" next to the link on {listed["listing_url"]}'


def build_dataset(start: str = DEFAULT_START, pages_file: Path = PAGES_FILE, discovery_file: Path = DISCOVERY_FILE,
                  end: Optional[str] = None, listings_file: Path = LISTINGS_FILE) -> DealDataset:
    pages = load_pages(pages_file)
    listed = listed_dates(listings_file)
    rows, failures = [], []
    seen: Dict[str, str] = {}
    for url in sorted(pages):
        pg = pages[url]
        cu = canonical_url(url)
        if cu in seen:
            failures.append({"source": pg.get("source"), "url": url, "stage": "dedupe", "reason": f"same page as {seen[cu]}"})
            continue
        seen[cu] = url
        if pg.get("status") != 200 or not pg.get("html"):
            failures.append({"source": pg.get("source"), "url": url, "stage": "fetch",
                             "reason": pg.get("error") or f"HTTP {pg.get('status')}", "fetched_at": pg.get("fetched_at")})
            continue
        try:
            art = parse_article(pg["html"], url)
            apply_listing_date(art, listed.get(cu))
            if art.error:
                failures.append({"source": pg["source"], "url": url, "stage": "parse", "reason": art.error,
                                 "fetched_at": pg.get("fetched_at")})
                continue
            rec = extract(art, pg["source"], collected_at=pg.get("fetched_at", ""), url=url)
        except Exception as exc:  # noqa: BLE001 - one bad page never stops the run; it is reported
            failures.append({"source": pg.get("source"), "url": url, "stage": "extract", "reason": f"{type(exc).__name__}: {exc}",
                             "fetched_at": pg.get("fetched_at")})
            continue
        rec["published_precision"] = art.published_precision
        rec["modified_date"] = art.modified_date or ""
        if art.published_precision == "month":
            rec["review_flags"] = "; ".join(x for x in [rec["review_flags"], "date known to the month only"] if x)
        rec["body"] = art.body
        rec["about_sections"] = "; ".join(n for n, _ in art.about)
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["deal_id", "source", "publisher", "url", "published_date", "is_deal", "headline",
                                   "target", "transaction_type", "body"])
    end = end or date.today().isoformat()
    pub = df["published_date"].fillna("")
    df["in_period"] = (pub >= start) & (pub <= end) & (pub != "")
    mod = df["modified_date"].fillna("") if "modified_date" in df else pd.Series("", index=df.index)
    df["period_note"] = ["" if ok else ("published before the period" if p and p < start else
                                        "dated after the build date" if p else
                                        f"no publication date found; last modified {m}, so published before the period" if m and m < start else
                                        f"no publication date found (last modified {m})" if m else "no publication date found")
                         for ok, p, m in zip(df["in_period"], pub, mod)]
    df = infer_ownership(df)
    df = _mark_duplicates(df)
    df["platform_key"] = platform_keys(df)

    fails = pd.DataFrame(failures, columns=["source", "url", "stage", "reason", "fetched_at"])
    coverage = _coverage(df, fails, discovery_file, start)
    latest = df.loc[df["in_period"], "published_date"].max() if len(df) else ""
    return DealDataset(records=df, coverage=coverage, failures=fails, start=start,
                       built_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(), pages_file=str(pages_file),
                       meta={"latest_published": latest if isinstance(latest, str) else "",
                             "collected_first": df["collected_at"].min() if len(df) else "",
                             "collected_last": df["collected_at"].max() if len(df) else ""})


def _coverage(df: pd.DataFrame, fails: pd.DataFrame, discovery_file: Path, start: str) -> pd.DataFrame:
    disc = pd.read_csv(discovery_file, dtype=str).fillna("") if Path(discovery_file).exists() else pd.DataFrame(
        columns=["source", "outcome"])
    rows = []
    for key, src in SOURCES.items():
        d = disc[disc["source"] == key]
        r = df[df["source"] == key]
        f = fails[fails["source"] == key]
        rows.append({
            "source": key, "firm": src.firm, "hq": src.hq,
            "sitemap_urls": len(d),
            "skipped_before_period": int(d["outcome"].str.startswith("skipped: modified before").sum()) if len(d) else 0,
            "skipped_robots": int(d["outcome"].str.contains("robots").sum()) if len(d) else 0,
            "pages_parsed": len(r), "failed_pages": len(f),
            "outside_period": int((~r["in_period"]).sum()) if len(r) else 0,
            "articles_in_period": int(r["in_period"].sum()) if len(r) else 0,
            "deals_in_period": int((r["in_period"] & r["is_deal"]).sum()) if len(r) else 0,
            "non_deal_news_in_period": int((r["in_period"] & ~r["is_deal"]).sum()) if len(r) else 0,
            "duplicates_marked": int((r["in_period"] & r["is_deal"] & (r["duplicate_of"] != "")).sum()) if len(r) else 0,
            "earliest_in_period": r.loc[r["in_period"], "published_date"].min() if len(r) and r["in_period"].any() else "",
            "latest_in_period": r.loc[r["in_period"], "published_date"].max() if len(r) and r["in_period"].any() else "",
            "collected_at": r["collected_at"].max() if len(r) else "",
        })
    return pd.DataFrame(rows)
