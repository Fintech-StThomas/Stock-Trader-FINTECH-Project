"""PE Deal Intelligence (pedeals): parsing, extraction rules on hard cases, the offline pipeline, validation, the tab."""
import math
from pathlib import Path

import pandas as pd
import pytest

from pedeals import build_dataset
from pedeals import analytics as an
from pedeals.collect import load_pages, parse_sitemap, slim_html
from pedeals.extract import classify_news, extract, location_of, sentences, state_code
from pedeals.fetch import PoliteFetcher
from pedeals.parse import Article, listing_dates, parse_article
from pedeals.sources import SOURCES
from pedeals.validate import evaluate, wilson

JONES = "https://tonkabayequity.com/news/standard-locknut-acquires-jones-bearing/"


@pytest.fixture(scope="module")
def ds():
    return build_dataset()


def art(title, body, about=(), url="https://example.com/news/x/", date="2025-01-01"):
    return Article(url=url, title=title, published_date=date, published_method="page metadata", body=body, about=list(about))


# ----------------------------------------------------------------------------- the brief's example, from the saved page
def test_the_briefs_example_is_read_as_the_brief_describes_it():
    page = load_pages()[JONES]
    a = parse_article(page["html"], JONES)
    assert a.title == "Standard Locknut acquires Jones Bearing"
    assert a.published_date == "2026-07-07" and a.published_method == "dateline"
    assert a.dateline_location == "MINNETONKA, MN"                      # the publisher's city, not the target's
    assert [n for n, _ in a.about] == ["Tonka Bay Equity Partners"]       # sponsor boilerplate kept apart
    r = extract(a, "tonkabay", url=JONES)
    assert r["publisher"] == "Tonka Bay Equity Partners" and r["buyer"] == "Standard Locknut"
    assert r["buyer_role"] == "Tonka Bay Equity Partners portfolio company"     # the publisher is not the buyer
    assert r["transaction_type"] == "Add-on acquisition" and r["target"] == "Jones Bearing"
    assert r["target_state"] == "AL" and r["target_industry"] == "Industrial manufacturing"
    assert "Industrial / OEM" in r["customer_markets"]                    # the customers' industry, a separate field
    assert {"Product / service expansion", "Customers / channels"} <= set(r["rationale_tags"].split("; "))
    assert r["price_disclosure"] == "Not stated" and r["deal_value_usd"] is None      # missing price is not zero
    assert r["closing_date"] == "Not stated"                                          # nor a closing date


# ----------------------------------------------------------------------------- is it a deal? (the hard headlines)
@pytest.mark.parametrize("title,body,expected", [
    ("Shane Slominski Interviewed on “Poised for Exit” Podcast", "He discussed exits.", "Recognition / media"),
    ("Huron Capital Named to Inc.’s 2023 List of Founder-Friendly Investors", "", "Recognition / media"),
    ("Align Capital Partners Closes 100th Acquisition", "ACP recently closed its 100th acquisition.", "Fund / firm"),
    ("Goldner Hawn Closes New $364 Million Private Equity Fund", "The fund will acquire companies.", "Fund / firm"),
    ("Align Capital Partners Closes Continuation Vehicle for Proceed", "ACP acquired Counsel Press in 2023.", "Fund / firm"),
    ("ACP Job Posting", "Work on add-on acquisitions.", "Other"),
    ("Kam Kielhorn Promoted to Partner", "He sourced several acquisitions.", "Personnel"),
    ("Huron Capital Exits Pacific Shoring After Five-Year Partnership", "", "Deal"),
    ("Align Makes a Fast Connection with ISPN Network Services",
     "CLEVELAND and DALLAS, December 18, 2024 | Align Capital Partners announced today the acquisition of ISPN Network Services.", "Deal"),
    ("Huron Capital Launches Enterprise Application Services Platform",
     "Huron is pleased to announce the launch of its latest initiative in partnership with an executive.", "Fund / firm"),
])
def test_news_classification(title, body, expected):
    assert classify_news(title, body, "huron" if "Huron" in title else "align")[0] == expected


# ----------------------------------------------------------------------------- who bought what
def test_playful_headline_is_resolved_from_the_lead_sentence():
    a = art("Marco Expands into Brewery End Market with Add-on of Jet Gasket & Seal",
            "CLEVELAND and DALLAS, February 28, 2023 | Align Capital Partners’ (“ACP”) specialty supplier Marco Rubber & "
            "Plastics (“Marco” or the “Company”) has acquired Jet Gasket & Seal (“Jet”), a specialty distributor of seals, "
            "gaskets and related products. Headquartered in Las Vegas, Nevada, Jet is a market-leading provider of O-rings.")
    r = extract(a, "align")
    assert (r["buyer"], r["target"], r["transaction_type"]) == ("Marco Rubber & Plastics", "Jet Gasket & Seal", "Add-on acquisition")
    assert r["buyer_short"] == "Marco" and r["target_industry"] == "Value-added distribution" and r["target_state"] == "NV"


def test_exit_to_an_affiliate_and_a_pending_sale():
    a = art("Align Capital Partners Completes Sale of Action Elevator",
            "Align Capital Partners (“ACP”) announced today the sale of Action Elevator (“Action” or the “Company”) to an "
            "affiliate of H.I.G. Capital (“H.I.G.”). Headquartered near Baltimore, Maryland, Action is a provider of elevator services.")
    r = extract(a, "align")
    assert (r["transaction_type"], r["buyer"], r["target"], r["seller"]) == ("Exit", "H.I.G. Capital", "Action Elevator", "Align Capital Partners")
    assert r["deal_status"] == "Completed" and r["target_state"] == "MD"
    p = art("Align Capital Partners Announces Sale of SEAM Group to ABB",
            "Align Capital Partners (“ACP”) announced today that its portfolio company SEAM Group (the “Company”) has entered "
            "into a definitive agreement to be acquired by ABB. The transaction is subject to regulatory approval.")
    r = extract(p, "align")
    assert (r["transaction_type"], r["buyer"], r["target"], r["deal_status"]) == ("Exit", "ABB", "SEAM Group", "Announced / pending")


def test_platform_investment_growth_investment_and_merger():
    r = extract(art("Tonka Bay Invests in LightMart",
                    "MINNETONKA, MN (January 7, 2025) — Tonka Bay Equity Partners LLC (“Tonka Bay”) invested in LightMart "
                    "Holdings LLC (“LightMart”), a value-added distributor of outdoor lighting products."), "tonkabay")
    assert (r["transaction_type"], r["buyer"], r["buyer_role"]) == ("Platform investment", "Tonka Bay Equity Partners", "Publisher (the PE firm itself)")
    r = extract(art("ACP Secures New Platform with Investment in Global Guardian",
                    "Align Capital Partners (“ACP”) is pleased to announce it has made a growth capital investment in Global "
                    "Guardian (or the “Company”), a provider of security and travel-related services."), "align")
    assert r["transaction_type"] == "Growth / minority investment"
    r = extract(art("MKB Company and Western Green Combine",
                    "Align Capital Partners’ (“ACP”) erosion control products platform MKB Company (the “Company” or “MKB”) "
                    "announced today a strategic combination with Western Green, an innovator in engineered erosion control."), "align")
    assert (r["transaction_type"], r["buyer"], r["target"]) == ("Merger / combination", "MKB Company", "Western Green")


def test_several_targets_and_names_given_in_the_next_sentence():
    r = extract(art("Trystar Completes the Acquisitions of Macromatic and Primax Technologies",
                    "Trystar, a Goldner Hawn portfolio company, recently completed the purchases of Macromatic and Primax "
                    "Technologies (“Primax”)."), "goldnerhawn")
    assert r["target"] == "Macromatic; Primax Technologies" and r["n_targets"] == 2
    r = extract(art("ACP-Backed Counsel Press Announces Three Strategic Add-ons",
                    "Align Capital Partners’ (“ACP”) outsourced legal services platform Counsel Press, Inc. (“Counsel Press” or "
                    "the “Company”) has acquired three providers of outsourced appellate services. Since partnering with ACP, "
                    "the Company acquired Cockle Legal Briefs (“Cockle”) in March, Becker Gallagher Briefs and Records (“Becker "
                    "Gallagher”) in May and added Lantagne Legal Printing (“Lantagne”) in September."), "align")
    assert r["target"].split("; ") == ["Cockle Legal Briefs", "Becker Gallagher Briefs and Records", "Lantagne Legal Printing"]


def test_locations_only_from_sentences_about_the_target():
    assert state_code("Kansas City") == "" and state_code("Miami") == "FL" and state_code("Grand Rapids, Mich.") == "MI"
    assert state_code("Virginia, Maryland") == ""                       # several states: unknown, not the first one
    # "X-based" describing another company does not locate the target
    loc = location_of(["Strata Landscape Services"], {}, "", "Strata’s team founded the Company when they acquired "
                      "Southern California-based Verdant Landscaping in 2024.")
    assert loc == ("", "")
    assert location_of(["Jenco"], {}, "", "Based in Scottsdale, Arizona, Jenco is an electrical contractor.")[1] == "AZ"


def test_sentences_do_not_split_after_abbreviations():
    assert sentences("Acme, Inc. acquired Beta Co. on Monday. It is based in St. Paul, Minn. and grows.") == [
        "Acme, Inc. acquired Beta Co. on Monday.", "It is based in St. Paul, Minn. and grows."]


def test_price_is_disclosed_only_with_a_dollar_amount_tied_to_the_deal():
    r = extract(art("X Acquires Y", "X Holdings has acquired Yco, a provider of testing services. The purchase price was "
                    "$42.5 million. X manages a $300 million fund."), "align")
    assert r["price_disclosure"] == "Disclosed" and math.isclose(r["deal_value_usd"], 42.5e6)
    r = extract(art("X Acquires Y", "X Holdings has acquired Yco. The terms of the transaction were not disclosed."), "align")
    assert r["price_disclosure"] == "Not disclosed (stated)" and r["deal_value_usd"] is None


# ----------------------------------------------------------------------------- collection helpers (no network)
def test_sitemap_listing_and_slimming():
    xml = ("<urlset><url><loc>https://goldnerhawn.com/news/a/</loc><lastmod>2024-05-01T10:00:00+00:00</lastmod></url>"
           "<url><loc>https://goldnerhawn.com/news/</loc></url></urlset>")
    rows = parse_sitemap(xml)
    assert rows[0] == {"url": "https://goldnerhawn.com/news/a/", "lastmod": "2024-05-01"}
    assert [r["url"] for r in rows if SOURCES["goldnerhawn"].is_article(r["url"])] == ["https://goldnerhawn.com/news/a/"]
    html = ("<div class='item'><h3><a href='/news/a/'>A</a></h3><p>October 13, 2026 | <a href='/news/a/'>Read More</a></p></div>"
            "<div class='item'><h3><a href='/news/b/'>B</a></h3><p>July 2026</p></div>")
    d = listing_dates(html, "https://goldnerhawn.com/news/", SOURCES["goldnerhawn"].is_article)
    assert [(x["listed_date"], x["precision"]) for x in d] == [("2026-10-13", "day"), ("2026-07-01", "month")]
    assert "<script" not in slim_html("<p>x</p><script>alert(1)</script>")


def test_fetcher_is_offline_in_tests_and_obeys_robots():
    f = PoliteFetcher()
    assert f.get("https://tonkabayequity.com/news/").error.startswith("offline mode")
    from urllib import robotparser
    rp = robotparser.RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /private/", "Crawl-delay: 10"])
    f._robots["https://example.com"] = rp
    assert f.allowed("https://example.com/news/a/") and not f.allowed("https://example.com/private/x")
    assert f.delay_for("https://example.com/news/a/") == 10


# ----------------------------------------------------------------------------- the pipeline on the saved pages
def test_pipeline_is_offline_complete_and_repeatable(ds):
    assert len(ds.records) > 150 and ds.failures.empty
    d = ds.deals
    assert len(d) >= 100 and set(d["publisher"]) == {s.firm for s in SOURCES.values()}
    assert (d["published_date"] >= ds.start).all() and (d["url"] != "").all()
    assert not (d["deal_value_usd"] == 0).any()
    undated = ds.records[ds.records["published_date"] == ""]
    assert len(undated) and not undated["in_period"].any()         # undated pages are reported, never placed in the period
    assert ds.fingerprint() == build_dataset().fingerprint()
    assert set(ds.coverage["source"]) == set(SOURCES) and (ds.coverage["pages_parsed"] > 0).all()


def test_ownership_inferred_from_another_announcement(ds):
    r = ds.records[ds.records["headline"] == "Corporate Technologies Acquires NuMSP"].iloc[0]
    assert r["transaction_type"] == "Add-on acquisition"
    assert "Tonka Bay Invests in Corporate Technologies" in r["type_evidence"]
    assert "relationship inferred" in r["review_flags"]


def test_analytics(ds):
    d = ds.unique_deals
    bb = an.buy_and_build(d)
    marco = bb[bb["platform"].str.contains("Marco")]
    assert len(marco) == 1 and "Marco Sealing Solutions" in marco["platform"].iloc[0]     # one platform across its rename
    sim = an.similar_deals(d, d["deal_id"].iloc[0], k=3)
    assert len(sim) == 3 and sim["similarity"].is_monotonic_decreasing
    brief = an.research_brief(d, "IT services & cybersecurity")
    assert "Needs further research" in brief and "[source](" in brief
    gaps = an.disclosure_gaps(d).set_index("field")
    assert gaps.loc["Transaction price", "share_missing"] > 0.9
    assert "CACHED DATA" in an.dataset_brief(ds)
    assert an.search(d, "cybersecurity", firms=["Goldner Hawn"])["publisher"].eq("Goldner Hawn").all()


def test_held_out_accuracy_does_not_regress(ds):
    rep = evaluate(ds, split="held-out")
    acc = rep.summary.set_index("field")["accuracy"]
    assert rep.n_labels >= 35
    assert acc["is_deal"] >= 0.95 and acc["transaction_type"] >= 0.9 and acc["buyer"] >= 0.9 and acc["target"] >= 0.9
    assert acc["target_industry"] >= 0.75 and acc["published_date"] >= 0.95
    assert (rep.consistency["failing"] == 0).all() and rep.repeatable
    lo, hi = wilson(26, 30)
    assert 0.6 < lo < 0.87 < hi < 1


# ----------------------------------------------------------------------------- the dashboard tab
def test_pe_tab_renders_in_the_dashboard():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    assert "🤝 PE deals" in labels
    assert {"🔎 Deal finder", "🏗️ Buy-and-build tracker", "🧭 Add-on research brief", "✅ Data quality"} <= set(labels)
    text = " ".join(str(m.value) for m in at.markdown)
    assert "PE Deal Intelligence" in text and "Decision scenario" in text
    assert any("Cached data" in str(i.value) for i in at.info)
