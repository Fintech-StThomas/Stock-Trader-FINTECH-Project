"""
Deal sourcing: turn the announcement record into the three questions a deal team asks every week.

1. **Exit watch** — which sponsor-owned companies have been held long enough that a sale is likely soon? The
   portfolio ledger (`ledger`) rebuilds each sponsor's holdings from its own announcements: an investment
   announcement or a sentence that dates the investment ("ACP acquired StenTech in May of 2022") opens a holding,
   add-ons show it is still owned, a sale announcement closes it. Lower-middle-market sponsors typically hold four to
   six years, so a holding past four years with no announced exit is a candidate for a sponsor-to-sponsor deal.
2. **Consolidation map** (`consolidation`) — which niches of the five focus sectors are being rolled up, by how many
   platforms, and is the pace rising or falling?
3. **Competing buyers** (`competing_buyers`) — who else is buying in a niche (and a region): the platforms you would
   bid against for a target, with their cadence and where they buy.

What the evidence supports: what the sponsors announced. "No exit announced" is not proof a company is still held
(sales are not always announced on the seller's site), and an entry date found in passing is the sponsor's own
statement. Every row keeps the announcement it came from.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from .pipeline import norm_name
from .taxonomy import FOCUS_SECTORS, OUTSIDE_FOCUS

TYPICAL_HOLD_YEARS = 4.0          # LMM sponsors usually sell after four to six years
MAX_PLAUSIBLE_HOLD = 10.0         # longer holds are rare; past this the sale was most likely never announced
STALE_YEARS = 4.0                 # no mention for this long: probably sold quietly
_SPONSOR_LIKE = re.compile(r"\b(?:Capital|Equity|Investments?|Management|Advisors|Fund)\b(?:\s+(?:Partners|Group|LLC|LP|L\.P\.))?$")


def _d(s) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


class _Names:
    """Company names of one sponsor, merged when one starts with the other ("Counsel Press" / "Counsel Press Holdings")
    or when the releases gave them the same short name ("Marco Rubber & Plastics" and, after a rename, "Marco Sealing
    Solutions", both "Marco")."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.latest: Dict[str, tuple] = {}

    def find(self, k: str) -> str:
        self.parent.setdefault(k, k)
        while self.parent[k] != k:
            k = self.parent[k]
        return k

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def key(self, name: str, short: str = "", when: str = "") -> str:
        k = norm_name(name)
        if not k or k == "unknown":
            return ""
        self.find(k)
        for other in [o for o in self.parent if not o.startswith("short:") and o != k]:
            if min(len(other), len(k)) >= 5 and (other.startswith(k + " ") or k.startswith(other + " ")):
                self.union(k, other)
        if short and len(norm_name(short)) >= 3:
            self.union(k, "short:" + norm_name(short))
        if " " not in k and 3 <= len(k) <= 8:                # an acronym or one-word name ("AKS", "ISPN") is a short name
            self.union(k, "short:" + k)
        if when >= self.latest.get(k, ("", ""))[0]:
            self.latest[k] = (when, name)
        return k

    def lookup(self, name: str) -> str:
        """The holding a name refers to, if the sponsor's deal announcements already know it; "" otherwise."""
        k = norm_name(name)
        if not k:
            return ""
        if k in self.parent:
            return self.find(k)
        if "short:" + k in self.parent:
            return self.find("short:" + k)
        for other in [o for o in self.parent if not o.startswith("short:")]:
            if min(len(other), len(k)) >= 5 and (other.startswith(k + " ") or k.startswith(other + " ")):
                return self.find(other)
        return ""

    def display(self, root: str) -> str:
        members = [(w, n) for k, (w, n) in self.latest.items() if self.find(k) == root]
        return max(members)[1] if members else root


def ledger(records: pd.DataFrame, as_of: Optional[date] = None) -> pd.DataFrame:
    """Each sponsor's holdings as its own announcements describe them (all collected years, not only the period)."""
    as_of = as_of or date.today()
    events = []          # (source, key, kind, date, detail, url)
    names: Dict[str, _Names] = defaultdict(_Names)
    meta: Dict[tuple, dict] = {}
    r = records.sort_values("published_date")
    for x in r.itertuples():
        nm = names[x.source]
        when = x.published_date or ""
        if x.is_deal and x.transaction_type in ("Platform investment", "Growth / minority investment") and x.target != "Unknown":
            for i, t in enumerate(x.target.split("; ")):
                k = nm.key(t, x.target_short if i == 0 else "", when)
                events.append((x.source, k, "entry_announced", when, x.transaction_type, x.url))
                meta.setdefault((x.source, k), {}).update({"focus_sector": x.focus_sector, "niche": x.niche,
                                                           "state": x.target_state, "description": x.target_description})
        elif x.is_deal and x.transaction_type == "Add-on acquisition" and x.buyer != "Unknown":
            k = nm.key(x.buyer, x.buyer_short, when)
            events.append((x.source, k, "add_on", when, x.target, x.url))
            m = meta.setdefault((x.source, k), {})
            m.setdefault("addon_sectors", Counter())[x.focus_sector] += 1
            if x.niche:
                m.setdefault("addon_niches", Counter())[x.niche] += 1
        elif x.is_deal and x.transaction_type == "Exit" and x.target != "Unknown":
            k = nm.key(x.target.split("; ")[0], x.target_short, when)
            events.append((x.source, k, "exit", when, x.buyer, x.url))
            meta.setdefault((x.source, k), {}).update({k2: v for k2, v in (("focus_sector", x.focus_sector), ("niche", x.niche),
                                                                           ("state", x.target_state)) if v})
    # a rename ties the old and new names to one holding ("Custom Veterinary Services" ... sold as "CompletePet")
    for x in r.itertuples():
        for rn in json.loads(getattr(x, "renames", "") or "[]") if isinstance(getattr(x, "renames", ""), str) else []:
            a, b = names[x.source].lookup(rn["old"]), names[x.source].lookup(rn["new"])
            if a and b:
                names[x.source].union(a, b)
            elif a or b:
                names[x.source].union(a or b, names[x.source].key(rn["new"] if a else rn["old"], "", x.published_date or ""))
    # a sentence dating an investment counts only for a company the sponsor's deal announcements know, so people
    # ("partnered with Dr. ... in 2008") and companies of other sponsors never become holdings
    for x in r.itertuples():
        for e in json.loads(x.entry_statements or "[]") if isinstance(x.entry_statements, str) else []:
            k = names[x.source].lookup(e["company"])
            if k:
                stated = f"{e['year']:04d}-{(e['month'] or 1):02d}-01"
                events.append((x.source, k, "entry_stated", stated, e["evidence"][:300], x.url))
    rows = []
    by: Dict[tuple, list] = defaultdict(list)
    for src, k, kind, when, detail, url in events:
        if k:
            by[(src, names[src].find(k))].append((kind, when, detail, url))
    for (src, k), ev in by.items():
        if not k:
            continue
        nm = names[src]
        entries = sorted([e for e in ev if e[0] in ("entry_announced", "entry_stated") and e[1]], key=lambda e: e[1])
        adds = sorted([e for e in ev if e[0] == "add_on" and e[1]], key=lambda e: e[1])
        exits = sorted([e for e in ev if e[0] == "exit" and e[1]], key=lambda e: e[1])
        dated = sorted(e[1] for e in ev if e[1])
        first_seen = dated[0] if dated else ""
        mentioned = sorted(e[1] for e in ev if e[1] and e[0] != "entry_stated")      # articles about it, not the dates they cite
        last_seen = max(mentioned + [e[1] for e in ev if e[0] == "entry_stated"]) if dated else ""
        entry = entries[0] if entries else None
        entry_date = entry[1] if entry else ""
        start = _d(entry_date) or _d(first_seen)
        exit_ = exits[-1] if exits else None
        end = _d(exit_[1]) if exit_ else as_of
        years = (end - start).days / 365.25 if start and end and end > start else math.nan
        stale = bool(not exit_ and last_seen and (as_of - _d(last_seen)).days / 365.25 >= STALE_YEARS)
        m = {}
        for kk in [kk for kk in meta if kk[0] == src and kk[1] and names[src].find(kk[1]) == k]:
            m.update({a: b for a, b in meta[kk].items() if b and not a.startswith("addon_")})
            for c in ("addon_sectors", "addon_niches"):
                if c in meta[kk]:
                    m.setdefault(c, Counter()).update(meta[kk][c])
        sector = m.get("focus_sector") or (m["addon_sectors"].most_common(1)[0][0] if m.get("addon_sectors") else "")
        rows.append({
            "source": src, "company": nm.display(k), "company_key": f"{src}|{k}",
            "status": "Exited" if exit_ else "No exit announced",
            "entry_date": entry_date, "entry_basis": ("investment announced" if entry and entry[0] == "entry_announced" else
                                                      "date stated in a release" if entry else
                                                      f"not found; first seen {first_seen}" if first_seen else "not found"),
            "entry_evidence": entry[2] if entry and entry[0] == "entry_stated" else "",
            "entry_url": entry[3] if entry else "",
            "years_held": years, "years_basis": "from entry" if entry else "at least (from first mention)",
            "last_seen": last_seen, "stale": stale,
            "add_ons": len(adds), "last_add_on": adds[-1][1] if adds else "", "exit_date": exit_[1] if exit_ else "",
            "exit_buyer": exit_[2] if exit_ else "", "exit_url": exit_[3] if exit_ else "",
            "focus_sector": sector or OUTSIDE_FOCUS,
            "niche": m.get("niche") or (m["addon_niches"].most_common(1)[0][0] if m.get("addon_niches") else ""),
            "state": m.get("state", ""),
            "description": m.get("description", ""),
            "evidence_urls": "; ".join(dict.fromkeys(e[3] for e in ev)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # a "holding" named like a sponsor ("Salt Creek Capital") is a co-investor read as a target: drop it
    out = out[~out["company"].map(lambda c: bool(_SPONSOR_LIKE.search(c)))]
    # "Company Vertical Supply Group": a release's "the Company" glued to the name
    out["company"] = out["company"].str.replace(r"^(?:[Tt]he )?Company (?=[A-Z])", "", regex=True)
    from .sources import SOURCES
    out.insert(0, "sponsor", out["source"].map(lambda s: SOURCES[s].firm if s in SOURCES else s))
    return out.sort_values(["status", "years_held"], ascending=[False, False]).reset_index(drop=True)


def exit_watch(led: pd.DataFrame, min_years: float = TYPICAL_HOLD_YEARS, sectors: Optional[List[str]] = None,
               include_stale: bool = False, max_years: float = MAX_PLAUSIBLE_HOLD) -> pd.DataFrame:
    """Companies with no announced exit held at least `min_years`: candidates for a sale in the next year or two.
    Left out by default: holdings not mentioned for STALE_YEARS (probably sold without an announcement on the
    sponsor's site) and holds longer than `max_years` (same reason)."""
    if led.empty:
        return led
    w = led[(led["status"] == "No exit announced") & (led["years_held"] >= min_years)]
    if not include_stale:
        w = w[~w["stale"] & (w["years_held"] <= max_years)]
    if sectors:
        w = w[w["focus_sector"].isin(sectors)]
    w = w.assign(signal=[("strong" if y >= 5 else "watch") if b == "from entry" else "watch (entry not dated)"
                         for y, b in zip(w["years_held"], w["years_basis"])])
    return w.sort_values("years_held", ascending=False).reset_index(drop=True)


def consolidation(deals: pd.DataFrame, as_of: Optional[date] = None, focus_only: bool = True) -> pd.DataFrame:
    """Add-on activity per focus sector and niche: how many add-ons, by how many platforms and sponsors, and whether
    the last twelve months were busier than the twelve before."""
    as_of = as_of or date.today()
    a = deals[deals["transaction_type"] == "Add-on acquisition"].copy()
    if focus_only:
        a = a[a["focus_sector"].isin(FOCUS_SECTORS)]
    if a.empty:
        return pd.DataFrame(columns=["focus_sector", "niche", "add_ons", "platforms", "sponsors", "last_12m", "prior_12m",
                                     "momentum", "latest", "target_states"])
    d = pd.to_datetime(a["published_date"]).dt.date
    a["recent"] = [(as_of - x).days <= 365 for x in d]
    a["prior"] = [365 < (as_of - x).days <= 730 for x in d]
    g = a.groupby(["focus_sector", "niche"])
    out = pd.DataFrame({
        "add_ons": g.size(), "platforms": g["platform_key"].nunique(), "sponsors": g["sponsor"].nunique(),
        "last_12m": g["recent"].sum().astype(int), "prior_12m": g["prior"].sum().astype(int), "latest": g["published_date"].max(),
        "target_states": g["target_state"].agg(lambda s: ", ".join(f"{k} ({v})" for k, v in Counter(x for x in s if x).most_common(5))),
    }).reset_index()
    out["momentum"] = out["last_12m"] - out["prior_12m"]
    return out.sort_values(["add_ons", "platforms"], ascending=False).reset_index(drop=True)


def competing_buyers(deals: pd.DataFrame, led: pd.DataFrame, niches: Optional[List[str]] = None,
                     sectors: Optional[List[str]] = None, states: Optional[List[str]] = None) -> pd.DataFrame:
    """The platforms buying in a niche (and optionally a set of states): who you would bid against for a target."""
    a = deals[(deals["transaction_type"] == "Add-on acquisition") & ~deals["buyer"].fillna("Unknown").isin(["", "Unknown"])]
    if sectors:
        a = a[a["focus_sector"].isin(sectors)]
    if niches:
        a = a[a["niche"].isin(niches)]
    if states:
        a = a[a["target_state"].isin(states)]
    if a.empty:
        return pd.DataFrame(columns=["platform", "sponsor", "add_ons_here", "last_add_on", "target_states", "niches",
                                     "platform_status", "years_held"])
    status = {}
    if not led.empty:
        for r in led.itertuples():
            status[(r.source, norm_name(r.company))] = (r.status, r.years_held)
    rows = []
    for (key, sponsor), g in a.groupby(["platform_key", "sponsor"]):
        name = g.sort_values("published_date")["buyer"].iloc[-1]
        st = next((v for (s, n), v in status.items() if s == g["source"].iloc[0] and n and (n in norm_name(name) or norm_name(name) in n)),
                  ("", math.nan))
        rows.append({"platform": name, "sponsor": sponsor, "add_ons_here": len(g), "last_add_on": g["published_date"].max(),
                     "target_states": ", ".join(sorted(set(x for x in g["target_state"] if x))),
                     "niches": "; ".join(sorted(set(g["niche"]))), "platform_status": st[0], "years_held": st[1]})
    return pd.DataFrame(rows).sort_values(["add_ons_here", "last_add_on"], ascending=False).reset_index(drop=True)


def sourcing_brief(deals: pd.DataFrame, led: pd.DataFrame, sector: str, niche: str = "", states: Optional[List[str]] = None,
                   min_years: float = TYPICAL_HOLD_YEARS) -> str:
    """The weekly sourcing memo for one sector (and niche): who is consolidating it, the companies they bought (the
    archetype to look for), the sponsor-owned companies likely to trade, and what still needs research."""
    a = deals[(deals["focus_sector"] == sector) & (deals["transaction_type"] == "Add-on acquisition")]
    if niche:
        a = a[a["niche"] == niche]
    scope = f"{sector}{' / ' + niche if niche else ''}{' in ' + ', '.join(states) if states else ''}"
    lines = [f"### Sourcing brief: {scope}", ""]
    comp = competing_buyers(deals, led, [niche] if niche else None, [sector], states)
    lines += [f"**Who is consolidating it** ({len(comp)} platforms, {int(comp['add_ons_here'].sum()) if len(comp) else 0} add-ons "
              "in the collected announcements):", ""]
    lines += [f"- **{r.platform}** ({r.sponsor}): {r.add_ons_here} add-on(s), latest {r.last_add_on}"
              f"{f'; buys in {r.target_states}' if r.target_states else ''}" for r in comp.head(10).itertuples()] or ["- none"]
    recent = a.sort_values("published_date", ascending=False)
    if states:
        recent = recent[recent["target_state"].isin(states)]
    lines += ["", "**What they bought** (the target profile to look for):", ""]
    for r in recent.head(12).itertuples():
        desc = (r.target_description or "").strip()
        lines.append(f"- {r.published_date} · **{r.target}**{f' ({r.target_state})' if r.target_state else ''} → {r.buyer}: "
                     f"{desc[:180] + ('…' if len(desc) > 180 else '')} [source]({r.url})")
    if recent.empty:
        lines.append("- none")
    w = exit_watch(led, min_years, [sector])
    if niche and len(w):
        w = w[(w["niche"] == niche) | (w["niche"] == "")]
    lines += ["", f"**Sponsor-owned companies in this sector held {min_years:g}+ years with no exit announced** "
                  "(possible sponsor-to-sponsor deals):", ""]
    lines += [f"- **{r.company}** ({r.sponsor}): {r.years_held:.1f} years "
              f"({'since ' + r.entry_date[:7] if r.entry_date else r.entry_basis}), {r.add_ons} add-on(s) [evidence]({r.entry_url or r.evidence_urls.split('; ')[0]})"
              for r in w.head(10).itertuples()] or ["- none in the collected announcements"]
    lines += ["", "**Supported by this evidence:** who announced which acquisitions, where, and when, and how long each "
                  "sponsor says it has owned its platforms. **Needs further research:** companies nobody has bought yet "
                  "(a company database or industry association lists), whether 'no exit announced' companies are "
                  "really still held, deal sizes and prices, and the sponsors' fund life (an older fund sells sooner)."]
    return "\n".join(lines)
