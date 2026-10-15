"""Business segments from XBRL instances (lighthouse/analyst/segments.py), on recorded excerpts of real filings
(tests/fixtures/segments, recorded October 2026). Each excerpt keeps the real root element, schema reference, units,
contexts and facts unchanged; facts the tests do not need were dropped, and with them the contexts nobody refers to:

* msft_10k_2026_segments_excerpt.xml: Microsoft's 10-K for fiscal 2026 (accession 0001193125-26-323660, filed 29 July
  2026, msft-20260630_htm.xml): revenue and operating income on the segment axis, revenue on the product and
  geography axes, the consolidated totals, and facts that must be ignored (segment cost of revenue, segment goodwill
  at an instant, share counts, document facts).
* aapl_10k_2025_segments_excerpt.xml: Apple's 10-K for fiscal 2025 (0000320193-25-000079, filed 31 October 2025),
  which tags its segment note the ASU 2023-07 way (segment member + ConsolidationItemsAxis = OperatingSegmentsMember)
  and its corporate costs on ConsolidationItemsAxis = CorporateNonSegmentMember.
* msft_10k_2026_labels_excerpt.xml: the label linkbase in Microsoft's schema (msft-20260630.xsd), the standard and
  terse labels of the members above.

Nothing here touches the network: the filing index and the downloads are replaced by functions serving these files."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lighthouse import data
from lighthouse.analyst import segments as SG
from lighthouse.data import DataUnavailable

FIX = Path(__file__).parent / "fixtures" / "segments"
MSFT_CIK, MSFT_ACCN = 789019, "0001193125-26-323660"
FY24, FY25, FY26 = pd.Timestamp("2024-06-30"), pd.Timestamp("2025-06-30"), pd.Timestamp("2026-06-30")
# the file names in the filing's index.json (October 2026), without the R*.htm pages and exhibits
MSFT_NAMES = ["0001193125-26-323660-index-headers.html", "0001193125-26-323660-index.html", "0001193125-26-323660.txt",
              "0001193125-26-323660-xbrl.zip", "FilingSummary.xml", "MetaLinks.json", "msft-20260630.htm",
              "msft-20260630.xsd", "msft-20260630_htm.xml", "msft-ex21.htm", "R1.htm", "report.css", "Show.js"]
# MSFT's 10-Ks of 2024-2026, as EDGAR's submissions list them
MSFT_FILINGS = pd.DataFrame({
    "accn": ["0000950170-24-087843", "0000950170-25-100235", MSFT_ACCN, "0001193125-25-256321"],
    "form": ["10-K", "10-K", "10-K", "10-Q"],
    "filed": pd.to_datetime(["2024-07-30", "2025-07-30", "2026-07-29", "2025-10-29"]),
    "doc": ["msft-20240630.htm", "msft-20250630.htm", "msft-20260630.htm", "msft-20250930.htm"],
    "is_xbrl": [1, 1, 1, 1]})


@pytest.fixture(scope="module")
def msft():
    return SG.parse_instance((FIX / "msft_10k_2026_segments_excerpt.xml").read_bytes())


@pytest.fixture(scope="module")
def aapl():
    return SG.parse_instance((FIX / "aapl_10k_2025_segments_excerpt.xml").read_bytes())


def _report(table, ticker="MSFT"):
    return SG.SegmentReport(ticker, MSFT_CIK, MSFT_ACCN, "10-K", pd.Timestamp("2026-07-29"), pd.Timestamp("2026-07-30"),
                            table)


def test_msft_segments_from_the_instance(msft):
    t = msft
    assert list(t.columns) == SG.COLUMNS
    assert set(t["view"]) == {"segment", "product", "geography", "total"}
    assert set(t["concept"]) == {"us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                                 "us-gaap:OperatingIncomeLoss"}            # cost of revenue, goodwill, shares: ignored
    assert (t["months"] == 12).all() and set(t["end"]) == {FY24, FY25, FY26}
    rep = _report(t)
    rev, oi = rep.annual("segment", "revenue"), rep.annual("segment", "operating_income")
    assert rev.loc["Intelligent Cloud", FY26] == 137_791e6 and oi.loc["Intelligent Cloud", FY26] == 56_972e6
    assert rev.loc["Productivity and Business Processes", FY24] == 106_820e6
    assert list(rev.index) == ["Productivity and Business Processes", "Intelligent Cloud", "More Personal Computing"]
    # the three segments are the whole company, every year, for revenue and operating income
    pd.testing.assert_series_equal(rev.sum(), rep.total("segment", "revenue"), check_names=False)
    pd.testing.assert_series_equal(oi.sum(), rep.total("segment", "operating_income"), check_names=False)
    assert rep.total("segment", "revenue")[FY26] == 331_839e6
    # geography: the United States and the rest add up to the total too
    geo = rep.annual("geography")
    assert geo.sum()[FY26] == 331_839e6 and len(geo) == 2
    # the product axis carries two breakdowns (product lines; product vs service): each adds up on its own
    prod = rep.annual("product")
    assert prod.loc[["Product", "Service Other"], FY26].sum() == 331_839e6     # us-gaap members, spelled out
    assert prod.drop(["Product", "Service Other"])[FY26].sum() == 331_839e6
    assert prod.loc["XBOX", FY26] == 21_790e6


def test_labels_from_the_label_linkbase(msft):
    labels = SG.parse_labels((FIX / "msft_10k_2026_labels_excerpt.xml").read_bytes())
    assert labels["msft:LinkedInCorporationMember"] == "LinkedIn"                       # the terse label wins
    assert labels["msft:IntelligentCloudMember"] == "Intelligent Cloud"                 # ' [Member]' removed
    assert labels["us-gaap:NonUsMember"] == "Other Countries"                           # a standard member relabelled
    # without labels a member is spelled out from its name
    assert SG.humanize("msft:MicrosoftThreeSixFiveConsumerProductsAndCloudServicesMember") == \
        "Microsoft Three Six Five Consumer Products and Cloud Services"
    assert SG.humanize("us-gaap:NonUsMember") == "Non US" and SG.humanize("country:US") == "US"
    assert SG.humanize("msft:XBOXMember") == "XBOX"


def test_asu_2023_07_tagging_and_corporate_costs(aapl):
    """Apple tags each segment cell with ConsolidationItemsAxis = OperatingSegmentsMember as well; its corporate costs
    sit on CorporateNonSegmentMember alone, which is neither a segment nor the consolidated total."""
    rep = _report(aapl, "AAPL")
    y = pd.Timestamp("2025-09-27")
    rev, oi = rep.annual("segment"), rep.annual("segment", "operating_income")
    assert len(rev) == 5 and rev.loc["Americas Segment", y] == 178_353e6
    assert rev.sum()[y] == rep.total("segment")[y] == 416_161e6
    # segment operating income before corporate R&D and G&A ($42.6bn): the total is the consolidated figure
    assert oi.sum()[y] == 175_677e6 and rep.total("segment", "operating_income")[y] == 133_050e6
    assert oi.sum()[y] - 42_627e6 == rep.total("segment", "operating_income")[y]
    assert set(aapl["view"]) == {"segment", "total"}


def test_instance_and_label_documents_in_a_filing_folder():
    assert SG.instance_document(MSFT_NAMES) == "msft-20260630_htm.xml"
    assert SG.label_document(MSFT_NAMES) == "msft-20260630.xsd"                         # the linkbases moved into it
    # a filing from before inline XBRL (MSFT's 10-K of August 2017, as its index lists the files)
    old = ["FilingSummary.xml", "Financial_Report.xlsx", "msft-10k_20170630.htm", "msft-20170630.xml",
           "msft-20170630.xsd", "msft-20170630_cal.xml", "msft-20170630_def.xml", "msft-20170630_lab.xml",
           "msft-20170630_pre.xml"]
    assert SG.instance_document(old) == "msft-20170630.xml"
    assert SG.label_document(old) == "msft-20170630_lab.xml"
    assert SG.instance_document(["FilingSummary.xml", "a.htm"]) is None


def _serve(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    from lighthouse import edgar

    def docs(cik, accn):
        calls.append(("index", accn))
        if accn != MSFT_ACCN:
            raise DataUnavailable(f"not recorded: {accn}")
        return pd.DataFrame({"name": MSFT_NAMES})

    def fetch(url):
        calls.append(("get", url.rsplit("/", 1)[-1]))
        files = {"msft-20260630_htm.xml": "msft_10k_2026_segments_excerpt.xml",
                 "msft-20260630.xsd": "msft_10k_2026_labels_excerpt.xml"}
        return (FIX / files[url.rsplit("/", 1)[-1]]).read_bytes()
    monkeypatch.setattr(edgar, "filing_documents", docs)
    monkeypatch.setattr(SG, "_fetch_bytes", fetch)
    return calls


def test_segments_for_is_point_in_time_and_cached(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, tmp_path)
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")                # the downloads are the stand-ins above
    rep = SG.segments_for("MSFT", "2026-07-30", filings=MSFT_FILINGS, cik=MSFT_CIK)
    assert rep.accn == MSFT_ACCN and rep.filed == pd.Timestamp("2026-07-29")
    assert rep.usable_from == pd.Timestamp("2026-07-30")                    # the session after the filing date
    assert calls == [("index", MSFT_ACCN), ("get", "msft-20260630_htm.xml"), ("get", "msft-20260630.xsd")]
    assert rep.annual("product").loc["LinkedIn", FY26] == 19_817e6          # the company's own label
    assert (tmp_path / "edgar_segments" / f"{MSFT_CIK}_{MSFT_ACCN.replace('-', '')}.parquet").exists()
    # the parsed table is cached: no download the second time, and offline it is still there
    again = SG.segments_for("MSFT", "2026-10-14", filings=MSFT_FILINGS, cik=MSFT_CIK)
    assert len(calls) == 3
    for c in ("view", "member", "label", "concept", "value", "months"):
        assert again.table[c].tolist() == rep.table[c].tolist()
    assert list(again.table["end"]) == list(rep.table["end"])
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "1")
    assert SG.segments_for("MSFT", "2026-10-14", filings=MSFT_FILINGS, cik=MSFT_CIK) is not None
    # on the filing date itself the 2026 10-K is not usable yet: the 2025 10-K is the one asked for (here: not
    # cached, offline, so no table rather than a look-ahead)
    assert SG.segments_for("MSFT", "2026-07-29", filings=MSFT_FILINGS, cik=MSFT_CIK) is None
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    with pytest.raises(DataUnavailable, match="0000950170-25-100235"):
        SG.segments_for("MSFT", "2026-07-29", filings=MSFT_FILINGS, cik=MSFT_CIK)
    # the sample pack has no instances: nothing is downloaded
    n = len(calls)
    assert SG.segments_for("MSFT", "2025-08-15", source="sample", filings=MSFT_FILINGS, cik=MSFT_CIK) is None
    assert len(calls) == n
    # before the first 10-K in the list: nothing
    assert SG.segments_for("MSFT", "2024-07-30", filings=MSFT_FILINGS, cik=MSFT_CIK) is None


def test_memo_section(msft):
    rep = _report(msft)
    lines = SG.memo_lines(rep)
    text = "\n".join(lines)
    assert lines[0] == "## Business segments (10-K filed 2026-07-29, fiscal year to 2026-06-30)"
    assert "| Intelligent Cloud | $137.8bn | +29.7% | 42% | $57.0bn | 41% |" in text
    assert "**Revenue by product or service**" in text and "**Revenue by geography**" in text
    seg = SG.view_table(rep, "segment")
    assert list(seg.columns) == ["revenue FY2026", "growth", "share of total", "operating income FY2026", "margin"]
    assert "operating income FY2026" not in SG.view_table(rep, "geography")
    # no segments, no section
    assert SG.memo_lines(None) == []
    assert SG.memo_lines(_report(msft[msft["view"] == "total"])) == []


def test_the_memo_shows_segments_only_when_there_are_some(msft):
    from lighthouse.analyst import memo as memo_mod
    m = memo_mod.build("MSFT", as_of="2016-11-15", source="sample", peers=False, risk_free=0.005, equity_premium=0.06)
    assert m.segments is None                                   # the sample pack holds no XBRL instances
    assert "## Business segments" not in m.markdown()
    assert not any("segment" in n for n in m.notes)              # failing soft: not even a note
    m.segments = _report(msft)
    text = m.markdown()
    assert "## Business segments" in text
    assert text.index("## The numbers") < text.index("## Business segments") < text.index("## Valuation")
    assert np.isfinite(m.financials.price)
