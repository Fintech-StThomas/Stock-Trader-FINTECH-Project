"""Offline contract, collection, extraction and repository integration tests."""
import json
from pathlib import Path

from pe.collectors.core import clean_text, collect
from pe.extraction.rules import extract
from pe.repository import load_dataset, publish_dataset


def test_clean_text_removes_navigation_and_extracts_title():
    title, text = clean_text('<html><head><title>Deal News</title></head><body><nav>Menu</nav><main><article><p>Acme acquired Beta Systems.</p></article></main><footer>Copyright</footer></body></html>')
    assert title == "Deal News"
    assert "Acme acquired Beta Systems." in text
    assert "Menu" not in text and "Copyright" not in text


def test_pagination_expansion_is_bounded_and_deduplicated():
    from pe.collectors.core import _source_urls
    urls = _source_urls({"urls": ["https://example.com/page/1"], "pagination": {
        "url_template": "https://example.com/page/{page}", "start": 1, "end": 500, "max_pages": 3}})
    assert urls == ["https://example.com/page/1", "https://example.com/page/2", "https://example.com/page/3"]

def test_collector_records_failure_without_aborting_and_deduplicates_urls(tmp_path, monkeypatch):
    import pe.collectors.core as core
    monkeypatch.setattr(core, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(core, "DATA_ROOT", tmp_path)
    class BrokenSession:
        def get(self, *args, **kwargs): raise TimeoutError("timed out")
    records, report = collect([{"name": "Test", "enabled": True, "urls": ["https://example.com/a", "https://example.com/a"]}], retries=0, delay_seconds=0, session=BrokenSession())
    assert records == []
    assert report["attempted"] == 1
    assert report["failed"] == 1
    assert len(report["failures"]) == 1



def test_cached_collector_never_uses_network_and_loads_saved_page(tmp_path, monkeypatch):
    import hashlib
    import pe.collectors.core as core
    monkeypatch.setattr(core, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(core, "DATA_ROOT", tmp_path)
    url = "https://example.com/deal"
    (tmp_path / "cache").mkdir()
    text = "Acme acquired Beta Systems."
    digest = hashlib.sha256(text.encode()).hexdigest()
    cache_file = tmp_path / "cache" / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    cache_file.write_text(json.dumps({"publisher_pe_firm": "Example PE", "source_url": url,
        "headline": "Acme acquired Beta Systems", "publication_date": "2026-01-01",
        "full_text": text, "content_hash": digest, "collected_at": "2026-01-02T00:00:00+00:00",
        "collection_status": "success", "is_cached": True}))
    class ForbiddenSession:
        def get(self, *args, **kwargs): raise AssertionError("cached mode must not access the network")
    records, report = collect([{"name": "Example PE", "enabled": True, "urls": [url]}], cached=True, session=ForbiddenSession(), delay_seconds=0)
    assert len(records) == 1 and records[0]["is_cached"] is True
    assert records[0]["publication_date"] == "2026-01-01"
    assert report["attempted"] == 0 and report["successful"] == 1

def test_extraction_flags_non_deal_and_uncertain_and_preserves_evidence():
    result = extract([
        {"announcement_id": "A1", "headline": "Acme acquired Beta Systems", "full_text": "Acme acquired Beta Systems. The deal expands distribution into new markets.", "source_url": "https://example.com/a", "publisher_pe_firm": "Firm"},
        {"announcement_id": "A2", "headline": "Quarterly results", "full_text": "The company reported quarterly results."},
        {"announcement_id": "A3", "headline": "Company update", "full_text": "A leadership team met today."},
    ])
    assert len(result["deals"]) == 1
    deal = result["deals"][0]
    assert deal["buyer_or_investor"] == "Acme"
    assert deal["target_company"] == "Beta Systems"
    assert "distribution expansion" in deal["rationale_tags"]
    assert any("distribution" in e["excerpt"].lower() for e in result["evidence"])
    assert {x["classification"] for x in result["review_queue"]} == {"non-deal", "uncertain", "possible-deal"}


def test_repository_falls_back_to_demo_and_publish_round_trip(tmp_path, monkeypatch):
    import pe.repository as repo
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps([{"deal_id": "D0", "data_status": "demo", "target_company": "Example"}]))
    monkeypatch.setattr(repo, "DEFAULT_SAMPLE", sample)
    monkeypatch.setattr(repo, "DEFAULT_REAL", tmp_path / "missing-real.json")
    monkeypatch.setattr(repo, "PUBLISHED_ROOT", tmp_path / "published")
    monkeypatch.setattr(repo, "MANIFEST_PATH", tmp_path / "manifest.json")
    fallback = repo.load_dataset()
    assert fallback["metadata"]["is_demo"] is True
    result = repo.publish_dataset([{"deal_id": "D1", "target_company": "Target"}], version="test-v1", metadata={"is_demo": False})
    assert result["deal_count"] == 1
    loaded = repo.load_dataset()
    assert loaded["metadata"]["version"] == "test-v1"
    assert loaded["deals"]["deal_id"].tolist() == ["D1"]


def test_market_and_legal_context_is_not_counted_as_a_deal():
    result = extract([
        {"announcement_id": "CTX1", "headline": "Private equity market outlook", "full_text": "Deal activity and fundraising trends.", "source_role": "market_context"},
        {"announcement_id": "CTX2", "headline": "New M&A regulatory analysis", "full_text": "A discussion of transaction structures.", "source_role": "legal_regulatory"},
    ])
    assert result["deals"] == []
    assert len(result["review_queue"]) == 2
    assert {item["classification"] for item in result["review_queue"]} == {"context"}


def test_source_configuration_has_all_requested_publishers():
    from pe.collectors.core import CONFIG_PATH
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    names = {source["name"] for source in config["sources"]}
    assert len(names) == 10
    assert "Bain — Global Private Equity Report" in names
    assert "Alternatives Watch — Open Access" in names
    assert all(source.get("allowed_hosts") and source.get("urls") for source in config["sources"])


def test_repository_preserves_real_records_and_handles_corrupt_current(tmp_path,monkeypatch):
    import pe.repository as repo
    monkeypatch.setattr(repo,'PUBLISHED_ROOT',tmp_path/'published')
    monkeypatch.setattr(repo,'MANIFEST_PATH',tmp_path/'manifest.json')
    assert len(repo.load_dataset()['records'])==119
    repo.publish_dataset([{'deal_id':'D1','buyer_or_investor':'Buyer','target_company':'Target','review_notes':'Keep this note','source_role':'firm_primary'}],version='v1',evidence=[{'deal_id':'D1','excerpt':'Original evidence'}])
    repo.publish_dataset([{'deal_id':'D2','target_company':'Other'}],version='v2')
    (tmp_path/'published/v2.json').write_text('corrupt')
    snapshot=repo.load_dataset()
    assert snapshot['metadata']['version']=='v1' and snapshot['metadata']['fallback']
    assert snapshot['records'][0]['review_notes']=='Keep this note'
    assert snapshot['records'][0]['source_role']=='firm_primary'
    assert snapshot['evidence'][0]['excerpt']=='Original evidence'


def test_publish_rejects_overwrites_empty_data_and_unsafe_versions(tmp_path,monkeypatch):
    import pytest
    import pe.repository as repo
    monkeypatch.setattr(repo,'PUBLISHED_ROOT',tmp_path/'published')
    monkeypatch.setattr(repo,'MANIFEST_PATH',tmp_path/'manifest.json')
    with pytest.raises(ValueError):repo.publish_dataset([],version='empty')
    with pytest.raises(ValueError):repo.publish_dataset([{'deal_id':'D1'}],version='../escape')
    repo.publish_dataset([{'deal_id':'D1'}],version='v1')
    with pytest.raises(ValueError):repo.publish_dataset([{'deal_id':'D2'}],version='v1')


def test_new_research_source_live_review_gate_makes_no_requests(tmp_path,monkeypatch):
    import pe.collectors.core as core
    monkeypatch.setattr(core,'CACHE_ROOT',tmp_path/'cache')
    monkeypatch.setattr(core,'DATA_ROOT',tmp_path)
    monkeypatch.delenv('LIGHTHOUSE_OFFLINE',raising=False)
    def forbidden(*a,**kw):raise AssertionError('Unreviewed source must not request the network')
    monkeypatch.setattr('requests.Session.get',forbidden)
    records,report=core.collect([{'name':'Unreviewed','enabled':True,'urls':['https://example.com/news']}])
    assert not records and report['attempted']==0 and report['failed']==1


def test_saved_deal_detail_page_is_available():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('from pe.repository import load_dataset\nfrom pe.pages.deal_detail import render_deal_detail\nimport streamlit as st\ns = load_dataset()\nst.session_state["pe_selected_deal_id"] = s["records"][0]["deal_id"]\nrender_deal_detail(s["deals"],s["evidence"])').run()
    assert not app.exception
    assert any(s.value=='Deal Detail' for s in app.subheader)
    assert any('→' in s.value for s in app.title)


def test_deal_detail_works_without_prior_selection_and_can_switch():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('from pe.repository import load_dataset\nfrom pe.pages.deal_detail import render_deal_detail\ns=load_dataset()\nrender_deal_detail(s["deals"],s["evidence"])').run()
    assert not app.exception and not any('Inspect saved deal' in x.value for x in app.info)
    picker=app.selectbox(key='pe_detail_choice')
    assert len(picker.options)==119
    first=app.session_state['pe_selected_deal_id']
    picker.select(picker.options[-1]).run()
    assert not app.exception and app.session_state['pe_selected_deal_id']!=first
    assert any('→' in x.value for x in app.title)
