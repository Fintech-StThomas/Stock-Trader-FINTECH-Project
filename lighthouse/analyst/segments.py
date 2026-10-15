"""
Business segments from a filing's XBRL instance: revenue and operating income by reportable segment, and revenue by
product or service and by geography, as the company tagged them in its 10-K (or 10-Q).

Why the instance: the SEC's companyfacts API keeps only facts without dimensions, so the segment numbers (facts on
contexts with a us-gaap:StatementBusinessSegmentsAxis member) are not in it. Each filing folder on
www.sec.gov/Archives/edgar/data/{cik}/{accession}/ holds the instance: for inline XBRL filings (from 2019) the
extracted document `*_htm.xml`, before that `{prefix}-{date}.xml`.

Views      segment    us-gaap:StatementBusinessSegmentsAxis, alone or with srt:ConsolidationItemsAxis =
                      us-gaap:OperatingSegmentsMember (how ASU 2023-07 filings tag the segment note)
           product    srt:ProductOrServiceAxis (us-gaap:ProductOrServiceAxis before the 2018 taxonomy)
           geography  srt:StatementGeographicalAxis (likewise us-gaap: before 2018)
           A context with any other dimension (a segment x product cell, an elimination, a typed member) is left out,
           so a view never counts the same dollars twice. A company may tag two breakdowns on one axis (Microsoft:
           product lines, and product vs service), so a view need not add up to the total; the `total` rows hold the
           consolidated figure of the same concept and period.
Concepts   revenue: the first of REVENUE_CONCEPTS that has values in the view; operating income: OperatingIncomeLoss,
           else SegmentReportingInformationOperatingIncomeLoss (older filings). US-dollar facts with a duration only.
Timing     a filing can be used from the session after its filing date (`usable_from`); `segments_for` takes the
           latest 10-K usable on the date asked, so a memo dated before a 10-K never shows that 10-K's segments.
Labels     the company's terse label, else its standard label without " [Member]", from its label linkbase (`*_lab.xml`,
           or the schema `.xsd` when the filer embeds its linkbases there, as Microsoft's 2026 10-K does); a member
           without one is spelled out from its name ('IntelligentCloudMember' -> 'Intelligent Cloud'). One extra
           request per filing; if it fails the names are spelled out and a note says so.
Cache      the parsed table, never the downloaded files: data/cache/edgar_segments/{cik}_{accession}.parquet (a filed
           document does not change, so it never expires).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .. import store

VIEW_AXES = {"segment": ("us-gaap:StatementBusinessSegmentsAxis",),
             "product": ("srt:ProductOrServiceAxis", "us-gaap:ProductOrServiceAxis"),
             "geography": ("srt:StatementGeographicalAxis", "us-gaap:StatementGeographicalAxis")}
CONSOLIDATION_AXES = ("srt:ConsolidationItemsAxis", "us-gaap:ConsolidationItemsAxis")
OPERATING_SEGMENTS = "us-gaap:OperatingSegmentsMember"
REVENUE_CONCEPTS = ["us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:Revenues",
                    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
                    "us-gaap:RevenuesNetOfInterestExpense", "us-gaap:SalesRevenueNet", "us-gaap:SalesRevenueGoodsNet",
                    "us-gaap:SalesRevenueServicesNet"]
OPERATING_INCOME_CONCEPTS = ["us-gaap:OperatingIncomeLoss", "us-gaap:SegmentReportingInformationOperatingIncomeLoss"]
METRIC = {**{c: "revenue" for c in REVENUE_CONCEPTS}, **{c: "operating_income" for c in OPERATING_INCOME_CONCEPTS}}
COLUMNS = ["view", "axis", "member", "label", "metric", "concept", "start", "end", "months", "value"]
FORMS = ("10-K",)

XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"
XLINK = "http://www.w3.org/1999/xlink"
LINK = "http://www.xbrl.org/2003/linkbase"
XSI_NIL = "{http://www.w3.org/2001/XMLSchema-instance}nil"
# standard taxonomies change namespace every year (http://fasb.org/us-gaap/2025): matched by family, not by year
_FAMILIES = {"fasb.org/us-gaap/": "us-gaap", "fasb.org/srt/": "srt", "xbrl.sec.gov/dei/": "dei",
             "xbrl.sec.gov/country/": "country", "xbrl.org/2003/iso4217": "iso4217"}
_LABEL_ROLES = ["http://www.xbrl.org/2003/role/terseLabel", "http://www.xbrl.org/2003/role/label"]


def _parser():
    """lxml with entities, DTDs and network access off: filings are untrusted input."""
    from lxml import etree
    return etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=True,
                           remove_comments=True, remove_pis=True)


def _family(uri: Optional[str], prefix: Optional[str]) -> str:
    for k, v in _FAMILIES.items():
        if uri and k in uri:
            return v
    return prefix or ""


def _canon(qname: str, nsmap: Dict[Optional[str], str]) -> str:
    """'us-gaap:Revenues' with the document's own prefix resolved: the standard taxonomies by family, a company's
    extension by its prefix."""
    qname = (qname or "").strip()
    prefix, _, local = qname.rpartition(":")
    return f"{_family(nsmap.get(prefix or None), prefix)}:{local}"


def humanize(member: str) -> str:
    """A readable name from a member's QName when the filing gives no label: 'msft:XBOXMember' -> 'XBOX',
    'us-gaap:NonUsMember' -> 'Non US', 'country:US' -> 'US'."""
    local = member.rpartition(":")[2]
    local = re.sub(r"Member$", "", local) or local
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", local).split()
    small = {"And", "Of", "The", "For", "In", "On", "To"}
    out = [w if i == 0 or w not in small else w.lower() for i, w in enumerate(words)]
    return " ".join("US" if w == "Us" else w for w in out)


def parse_instance(xml: bytes) -> pd.DataFrame:
    """Every segment, product and geography revenue / operating-income fact of one XBRL instance, plus the
    consolidated (`total`) values of the same concepts, one row per view, member, concept and period (COLUMNS)."""
    from lxml import etree
    root = etree.fromstring(xml, _parser())
    usd = set()
    for u in root.iter(f"{{{XBRLI}}}unit"):
        m = u.findall(f"{{{XBRLI}}}measure")
        if len(m) == 1 and _canon(m[0].text, u.nsmap) == "iso4217:USD":
            usd.add(u.get("id"))
    axes = {a: v for v, aa in VIEW_AXES.items() for a in aa}
    ctx = {}
    for c in root.iter(f"{{{XBRLI}}}context"):
        p = c.find(f"{{{XBRLI}}}period")
        start, end = p.findtext(f"{{{XBRLI}}}startDate"), p.findtext(f"{{{XBRLI}}}endDate")
        if not start or not end:
            continue                                            # an instant: a balance, not a flow
        if any(True for _ in c.iter(f"{{{XBRLDI}}}typedMember")):
            continue
        dims = {_canon(m.get("dimension"), m.nsmap): _canon(m.text, m.nsmap)
                for m in c.iter(f"{{{XBRLDI}}}explicitMember")}
        # "operating segments" next to a segment member is the ASU 2023-07 tagging of the same cell; alone it is the sum
        # of the segments (before corporate costs), not the consolidated total
        if len(dims) > 1:
            for a in CONSOLIDATION_AXES:
                if dims.get(a) == OPERATING_SEGMENTS:
                    dims.pop(a)
        if not dims:
            view, axis, member = "total", "", ""
        elif len(dims) == 1 and next(iter(dims)) in axes:
            axis, member = next(iter(dims.items()))
            view = axes[axis]
        else:
            continue
        ctx[c.get("id")] = (view, axis, member, start.strip(), end.strip())
    rows = []
    for el in root:
        if not isinstance(el.tag, str) or el.get("contextRef") not in ctx or el.get("unitRef") not in usd:
            continue
        if el.get(XSI_NIL) == "true":
            continue
        q = etree.QName(el)
        concept = f"{_family(q.namespace, el.prefix)}:{q.localname}"
        if concept not in METRIC:
            continue
        try:
            value = float((el.text or "").strip())
        except ValueError:
            continue
        view, axis, member, start, end = ctx[el.get("contextRef")]
        rows.append((view, axis, member, METRIC[concept], concept, start, end, value))
    df = pd.DataFrame(rows, columns=["view", "axis", "member", "metric", "concept", "start", "end", "value"])
    df["start"], df["end"] = pd.to_datetime(df["start"]), pd.to_datetime(df["end"])
    df = df.drop_duplicates(["view", "member", "concept", "start", "end"]).reset_index(drop=True)
    df["months"] = ((df["end"] - df["start"]).dt.days / 30.44).round().astype(int)
    # one revenue concept and one operating-income concept per view (the first in the preference order it uses)
    keep = []
    for view, g in df[df["view"] != "total"].groupby("view"):
        for concepts in (REVENUE_CONCEPTS, OPERATING_INCOME_CONCEPTS):
            have = [c for c in concepts if (g["concept"] == c).any()]
            if have:
                keep.append(g[g["concept"] == have[0]])
    used = set(pd.concat(keep)["concept"]) if keep else set()
    keep.append(df[(df["view"] == "total") & df["concept"].isin(used)])
    out = pd.concat(keep, ignore_index=True)
    out.insert(3, "label", [humanize(m) if m else "Total" for m in out["member"]])
    return out[COLUMNS].sort_values(["view", "metric", "member", "end"]).reset_index(drop=True)


def parse_labels(xml: bytes) -> Dict[str, str]:
    """{member QName: label} from a label linkbase, or from a schema that embeds one: the terse label, else the
    standard label without its ' [Member]' suffix."""
    from lxml import etree
    root = etree.fromstring(xml, _parser())
    found: Dict[str, Dict[str, str]] = {}
    for link in root.iter(f"{{{LINK}}}labelLink"):
        locs, res = {}, {}
        for el in link:
            if not isinstance(el.tag, str):
                continue
            lab = el.get(f"{{{XLINK}}}label")
            if el.tag == f"{{{LINK}}}loc":
                frag = (el.get(f"{{{XLINK}}}href") or "").rpartition("#")[2]
                prefix, _, local = frag.partition("_")
                if local:
                    locs[lab] = f"{prefix}:{local}"
            elif el.tag == f"{{{LINK}}}label":
                res.setdefault(lab, {})[el.get(f"{{{XLINK}}}role")] = (el.text or "").strip()
        for el in link.iter(f"{{{LINK}}}labelArc"):
            concept = locs.get(el.get(f"{{{XLINK}}}from"))
            if concept:
                found.setdefault(concept, {}).update(res.get(el.get(f"{{{XLINK}}}to"), {}))
    out = {}
    for concept, roles in found.items():
        for r in _LABEL_ROLES:
            if roles.get(r):
                out[concept] = re.sub(r"\s*\[Member\]$", "", roles[r])
                break
    return out


def instance_document(names: Iterable[str]) -> Optional[str]:
    """The XBRL instance among a filing folder's file names: the inline filing's extracted `*_htm.xml`, else the one
    .xml that is not a linkbase or the filing summary."""
    names = [str(n) for n in names]
    inline = [n for n in names if n.lower().endswith("_htm.xml")]
    if inline:
        return inline[0]
    skip = re.compile(r"(_(cal|def|lab|pre)\.xml|^FilingSummary\.xml|^MetaLinks)$", re.I)
    plain = [n for n in names if n.lower().endswith(".xml") and not skip.search(n)]
    return plain[0] if len(plain) == 1 else None


def label_document(names: Iterable[str]) -> Optional[str]:
    names = [str(n) for n in names]
    lab = [n for n in names if n.lower().endswith("_lab.xml")]
    xsd = [n for n in names if n.lower().endswith(".xsd")]
    return (lab or xsd or [None])[0]


def next_session(day) -> pd.Timestamp:
    """The first trading session after `day`: when a filing made on `day` can first be traded on."""
    from ..data import is_trading_day
    d = pd.Timestamp(day).normalize() + pd.Timedelta(days=1)
    while not is_trading_day(d):
        d += pd.Timedelta(days=1)
    return d


@dataclass
class SegmentReport:
    """The segment tables of one filing, with the dates that make them point-in-time."""
    ticker: str
    cik: int
    accn: str
    form: str
    filed: pd.Timestamp
    usable_from: pd.Timestamp
    table: pd.DataFrame
    notes: List[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{self.accn.replace('-', '')}/"

    def views(self) -> List[str]:
        return [v for v in VIEW_AXES if (self.table["view"] == v).any()]

    def annual(self, view: str, metric: str = "revenue") -> pd.DataFrame:
        """members x fiscal-year ends (12-month periods), largest in the latest year first; for a 10-Q, use
        `periods(view, metric, months=3)`."""
        return self.periods(view, metric, 12)

    def periods(self, view: str, metric: str = "revenue", months: int = 12) -> pd.DataFrame:
        tab = self.table
        t = tab[(tab["view"] == view) & (tab["metric"] == metric) & (tab["months"] == months)]
        if t.empty:
            return pd.DataFrame()
        w = t.pivot_table(index="label", columns="end", values="value", aggfunc="first").sort_index(axis=1)
        return w.sort_values(w.columns[-1], ascending=False)

    def total(self, view: str, metric: str = "revenue", months: int = 12) -> pd.Series:
        """The consolidated figure of the concept the view uses, by period end."""
        used = self.table[(self.table["view"] == view) & (self.table["metric"] == metric)]["concept"].unique()
        t = self.table[(self.table["view"] == "total") & self.table["concept"].isin(used) &
                       (self.table["months"] == months)]
        return t.set_index("end")["value"].sort_index()


def _fetch_bytes(url: str) -> bytes:
    from ..http import fetch
    return fetch(url, timeout=120).content


def _usable(filings: pd.DataFrame, as_of: pd.Timestamp, forms: Iterable[str]) -> pd.DataFrame:
    f = filings[filings["form"].isin(list(forms))].copy()
    if "is_xbrl" in f:
        f = f[pd.to_numeric(f["is_xbrl"], errors="coerce").fillna(1) > 0]
    f = f.dropna(subset=["filed"])
    f["usable_from"] = [next_session(d) for d in f["filed"]]
    return f[f["usable_from"] <= as_of].sort_values("filed")


def segments_for(ticker: str, as_of=None, source: str = "auto", forms: Iterable[str] = FORMS,
                 filings: Optional[pd.DataFrame] = None, cik: Optional[int] = None,
                 refresh: bool = False) -> Optional[SegmentReport]:
    """The segment tables of the latest filing among `forms` usable on `as_of` (default: today), or None when there
    is none: no such filing, no XBRL instance, no segment facts, or (offline, or from the sample pack, which holds no
    instances) a filing whose tables are not cached yet. Downloads: the filing index, the instance and the label
    document, three requests under the SEC's rate limit."""
    from ..data import DataUnavailable, offline_mode
    from ..edgar import company_filings, filing_documents
    t = ticker.upper().strip()
    if filings is None or cik is None:
        profile, filings = company_filings(t, source=source)
        cik = cik if cik is not None else profile.get("cik")
    if filings is None or filings.empty or not cik:
        return None
    day = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    usable = _usable(filings, day, forms)
    if usable.empty:
        return None
    last = usable.iloc[-1]
    accn = str(last["accn"])
    key = f"{int(cik)}_{accn.replace('-', '')}"
    notes: List[str] = []
    table = None if refresh else store.read_frame("edgar_segments", key)
    if table is None:
        if source == "sample" or offline_mode():
            return None
        docs = filing_documents(int(cik), accn)
        names = list(docs["name"]) if "name" in docs else []
        inst = instance_document(names)
        if inst is None:
            return None
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/"
        table = parse_instance(_fetch_bytes(base + inst))
        lab = label_document(names)
        labels: Dict[str, str] = {}
        if lab is not None and len(table):
            try:
                labels = parse_labels(_fetch_bytes(base + lab))
            except (DataUnavailable, ValueError) as exc:
                notes.append(f"labels not read ({type(exc).__name__}); member names are spelled out")
        if labels:
            table["label"] = [labels.get(m, lbl) for m, lbl in zip(table["member"], table["label"])]
        store.write_frame("edgar_segments", key, table)
    if table.empty or not (table["view"] != "total").any():
        return None
    for c in ("start", "end"):
        table[c] = pd.to_datetime(table[c])
    return SegmentReport(ticker=t, cik=int(cik), accn=accn, form=str(last["form"]), filed=pd.Timestamp(last["filed"]),
                         usable_from=pd.Timestamp(last["usable_from"]), table=table, notes=notes)


# --------------------------------------------------------------------------- #
# The memo section
# --------------------------------------------------------------------------- #
def _bn(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"${x / 1e9:,.1f}bn"


def _pc(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.1f}%"


def _share(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.0f}%"


def view_table(rep: SegmentReport, view: str) -> pd.DataFrame:
    """The memo's table for one view: latest-year revenue, growth on the year before, share of the consolidated
    revenue, and (segments) operating income and margin."""
    rev = rep.annual(view, "revenue")
    if rev.empty:
        return pd.DataFrame()
    y = rev.columns[-1]
    prev = rev.columns[-2] if len(rev.columns) > 1 else None
    tot = rep.total(view, "revenue")
    total = float(tot.get(y, np.nan)) if len(tot) else np.nan
    out = pd.DataFrame(index=rev.index)
    out.index.name = {"segment": "segment", "product": "product or service", "geography": "region"}.get(view, view)
    out[f"revenue FY{y.year}"] = rev[y].map(_bn)
    out["growth"] = (rev[y] / rev[prev] - 1.0).map(_pc) if prev is not None else "n/a"
    out["share of total"] = (rev[y] / total).map(_share) if np.isfinite(total) and total else "n/a"
    oi = rep.annual(view, "operating_income")
    if not oi.empty and y in oi.columns:
        o = oi[y].reindex(rev.index)
        out[f"operating income FY{y.year}"] = o.map(_bn)
        out["margin"] = (o / rev[y]).map(_share)
    return out


def memo_lines(rep: Optional[SegmentReport]) -> List[str]:
    """Markdown lines of the memo's segment section; none when there are no segment tables."""
    if rep is None or rep.table is None or rep.table.empty:
        return []
    from ..report import _md_table
    titles = {"segment": "By reportable segment", "product": "Revenue by product or service",
              "geography": "Revenue by geography"}
    blocks = []
    for v in rep.views():
        tab = view_table(rep, v)
        if not tab.empty:
            blocks += [f"**{titles[v]}**\n", _md_table(tab) + "\n"]
    if not blocks:
        return []
    end = rep.table.loc[rep.table["months"] == 12, "end"].max()
    head = (f"## Business segments ({rep.form} filed {rep.filed.date()}"
            + (f", fiscal year to {end.date()}" if pd.notna(end) else "") + ")")
    foot = (f"- From the filing's XBRL instance ({rep.url}), usable from {rep.usable_from.date()}. A company can tag "
            "two breakdowns on one axis (product lines and product vs service, say), so a view need not add up to "
            "100%; shares are of the consolidated revenue.")
    return [head] + blocks + [foot] + [f"- Note: {n}" for n in rep.notes] + [""]


__all__ = ["SegmentReport", "segments_for", "parse_instance", "parse_labels", "instance_document", "label_document",
           "humanize", "next_session", "view_table", "memo_lines", "VIEW_AXES", "REVENUE_CONCEPTS",
           "OPERATING_INCOME_CONCEPTS"]
