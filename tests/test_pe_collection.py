"""Offline acceptance tests for Nikhil's collection and coverage handoff."""
from dataclasses import replace
import json
from pathlib import Path
import requests
import pytest
from pe.collectors import SourceConfig, collect, load_sources
from pe.collectors.config import canonical_url
from pe.collectors.parsing import parse_announcement, parse_listing
from pe.collectors.http import PoliteClient, CollectionError
from pe.storage import CollectionStore, write_json

BASE = 'https://firm.example'
ARTICLE = '<html><nav>unrelated menu</nav><h1>Example acquisition</h1><time>October 2, 2026</time><article><p>' + ('A synthetic announcement for testing describes a proposed transaction and its stated rationale. ' * 3) + '</p><nav>remove this</nav><script>bad()</script></article></html>'


@pytest.fixture
def cfg():
    return SourceConfig('firm', 'Example Firm', BASE+'/news/', '/news/', '//article', '//time',
                        enabled=True, access_status='internal_research', delay_seconds=1)


class FakeClient:
    access = {'robots_text': 'User-agent: *\nAllow: /', 'delay_seconds': 1}
    def __init__(self, pages):
        self.pages = pages; self.calls = []
    def prepare(self):
        pass
    def get(self, url):
        self.calls.append(url)
        result = self.pages[url]
        if isinstance(result, Exception):
            raise result
        return result, url


def client_for(*names):
    links = ''.join(f'<a href="/news/{n}">item</a>' for n in names)
    pages = {BASE+'/news/': links}
    pages.update({BASE+'/news/'+n: ARTICLE for n in names})
    return FakeClient(pages)


def test_repeat_run_stable_ids_and_no_repeated_articles(cfg, tmp_path):
    store = CollectionStore(tmp_path)
    client = client_for('a', 'a?utm_source=x', 'b')
    rows, report = collect(cfg, store, client=client)
    assert len(rows) == 1
    assert report['counts']['duplicate_links'] == 1
    assert report['counts']['duplicate_content'] == 1
    assert len(rows[0]['source_urls']) == 2
    again, second = collect(cfg, store, client=client_for('b', 'a'))
    assert len(again) == 1 and again[0]['announcement_id'] == rows[0]['announcement_id']
    assert again[0]['source_url'] == rows[0]['source_url']
    assert report['run_id'] != second['run_id']
    assert (tmp_path / rows[0]['raw_path']).exists()
    assert report['counts']['attempted_pages'] == report['counts']['successful_pages'] + report['counts']['failed_pages']


def test_pagination_and_loop_are_bounded(cfg, tmp_path):
    fake = client_for('a')
    fake.pages[BASE+'/news/'] += '<a rel="next" href="?page=2">Next</a>'
    fake.pages[BASE+'/news/?page=2'] = '<a href="/news/b">b</a><a rel="next" href="/news/">loop</a>'
    fake.pages[BASE+'/news/b'] = ARTICLE.replace('proposed', 'completed')
    rows, report = collect(cfg, CollectionStore(tmp_path), client=fake)
    assert len(rows) == 2
    assert report['counts']['listing_pages'] == 2
    assert report['truncated'] and 'Repeated pagination link' in report['stop_reasons']
    assert len(fake.calls) == 4


def test_failed_refresh_preserves_last_good_and_records_errors(cfg, tmp_path):
    store = CollectionStore(tmp_path)
    first, good = collect(cfg, store, client=client_for('a'))
    fake = client_for('a', 'b'); fake.pages[BASE+'/news/b'] = CollectionError('Timeout; retry limit reached')
    rows, partial = collect(cfg, store, client=fake)
    assert partial['status'] == 'partial' and len(rows) == 1
    assert partial['counts']['failed_pages'] == 1
    assert store.load('firm')[1]['run_id'] == good['run_id']
    coverage = store.coverage([cfg])['sources'][0]
    assert coverage['latest_attempt']['status'] == 'partial'
    assert coverage['published_run_id'] == good['run_id']
    assert 'full_text' not in json.dumps(coverage) and 'raw_path' not in json.dumps(coverage)


def test_cached_mode_uses_no_client_and_retains_provenance(cfg, tmp_path):
    store = CollectionStore(tmp_path)
    rows, report = collect(cfg, store, client=client_for('a'))
    class NoNetwork:
        def prepare(self):
            pytest.fail('Cached collection must never request the network')
    cached, replay = collect(cfg, store, client=NoNetwork(), cached=True)
    assert cached[0]['is_cached'] is True and replay['mode'] == 'cached'
    assert cached[0]['collected_at'] == rows[0]['collected_at']
    assert cached[0]['full_text'] == rows[0]['full_text']
    assert store.load('firm')[1]['mode'] == 'live'


def test_missing_date_is_unknown_and_outside_window_excluded(cfg, tmp_path):
    fake = client_for('a', 'b')
    fake.pages[BASE+'/news/a'] = ARTICLE.replace('<time>October 2, 2026</time>', '')
    fake.pages[BASE+'/news/b'] = ARTICLE.replace('October 2, 2026', 'October 2, 2025')
    rows, report = collect(cfg, CollectionStore(tmp_path), client=fake)
    assert len(rows) == 1 and rows[0]['publication_date'] is None
    assert rows[0]['within_requested_period'] is None
    assert report['counts']['unknown_dates'] == 1 and report['counts']['outside_period'] == 1


def test_limits_disclose_incomplete_coverage(cfg, tmp_path):
    cfg = replace(cfg, max_articles=1)
    rows, report = collect(cfg, CollectionStore(tmp_path), client=client_for('a', 'b'))
    assert len(rows) == 1 and report['truncated']
    assert report['counts']['discovered_urls'] == 2
    assert 'Article limit reached' in report['stop_reasons']


def test_listing_layout_failure_not_silently_successful(cfg, tmp_path):
    rows, report = collect(cfg, CollectionStore(tmp_path), client=FakeClient({BASE+'/news/':'<html>Access denied</html>'}))
    assert not rows and report['status'] == 'failed'
    assert report['errors'][0]['stage'] == 'listing_parse'


def test_article_layout_failure_does_not_crash_run(cfg, tmp_path):
    fake = client_for('a', 'b');fake.pages[BASE+'/news/a'] = '<html>Not an article</html>'
    rows, report = collect(cfg, CollectionStore(tmp_path), client=fake)
    assert len(rows) == 1 and report['status'] == 'partial'
    assert report['counts']['attempted_pages'] == 3
    assert report['counts']['successful_pages'] == 2


def test_html_cleaning_does_not_use_body_dates(cfg):
    row = parse_announcement(ARTICLE.replace('<time>October 2, 2026</time>', ''), BASE+'/news/a', cfg, '2026-11-03T00:00:00Z')
    assert row['publication_date'] is None
    assert 'unrelated menu' not in row['full_text'] and 'remove this' not in row['full_text']
    assert 'bad()' not in row['full_text']


def test_riverside_embedded_links_read_without_executing_script(cfg):
    cfg = replace(cfg, listing_kind='riverside_embedded')
    raw = '<script>rcCurrentsData.push({url:"https://firm.example/news/a",body:"not executed"});evil()</script>'
    assert parse_listing(raw, cfg.listing_url, cfg)[0] == [BASE+'/news/a']


def test_multiple_firms_do_not_collapse_shared_text(cfg, tmp_path):
    store = CollectionStore(tmp_path)
    one, _ = collect(cfg, store, client=client_for('a'))
    second = replace(cfg, source_id='other', publisher_pe_firm='Other Firm')
    two, _ = collect(second, store, client=client_for('a'))
    assert one[0]['announcement_id'] != two[0]['announcement_id']
    assert len(store.coverage([cfg, second])['sources']) == 2


def test_lock_and_unsafe_paths(cfg, tmp_path):
    store = CollectionStore(tmp_path)
    with store.lock():
        with pytest.raises(RuntimeError):
            with store.lock():
                pass
    with pytest.raises(ValueError):
        store.source_dir('../escape')
    with pytest.raises(ValueError):
        store.load('firm', '../escape')
    with pytest.raises(ValueError):
        replace(cfg, start_date='2027-01-01')


class Response:
    def __init__(self, status=200, raw='User-agent: *\nAllow: /', **headers):
        self.status_code=status;self.headers=headers;self.encoding='utf-8';self.raw=raw
    def __enter__(self):return self
    def __exit__(self, *args):pass
    def iter_content(self, chunk_size):yield self.raw.encode()


class Session:
    def __init__(self, responses):self.responses=list(responses);self.calls=[]
    def get(self, url, **kwargs):
        self.calls.append((url,kwargs));r=self.responses.pop(0)
        if isinstance(r,Exception):raise r
        return r


@pytest.fixture
def online(monkeypatch):
    monkeypatch.delenv('LIGHTHOUSE_OFFLINE',raising=False)
    monkeypatch.setattr('pe.collectors.http._limiter.wait',lambda *args:None)
    monkeypatch.setattr('pe.collectors.http.time.sleep',lambda *args:None)


def test_offline_guard(cfg):
    session=Session([])
    with pytest.raises(CollectionError,match='Offline'):
        PoliteClient(cfg,session).prepare()
    assert session.calls == []


def test_robots_deny_and_delay(cfg, online):
    session=Session([Response(raw='User-agent: *\nDisallow: /news/blocked\nCrawl-delay: 10')])
    client=PoliteClient(cfg,session);client.prepare()
    assert client.delay == 10
    with pytest.raises(CollectionError,match='disallowed'):
        client.get(BASE+'/news/blocked')
    assert len(session.calls)==1


def test_redirect_cannot_escape_source(cfg, online):
    session=Session([Response(),Response(302,Location='https://other.example/private')])
    client=PoliteClient(cfg,session);client.prepare()
    with pytest.raises(CollectionError,match='outside'):
        client.get(BASE+'/news/a')
    assert len(session.calls)==2
    assert session.calls[-1][1]['allow_redirects'] is False


def test_timeouts_retry_bounded_and_use_project_not_sec_identity(cfg, online):
    session=Session([requests.Timeout(),requests.Timeout(),requests.Timeout()])
    with pytest.raises(CollectionError,match='retry limit'):
        PoliteClient(cfg,session).prepare()
    assert len(session.calls)==3
    assert 'LighthouseResearch' in session.calls[0][1]['headers']['User-Agent']
    assert '@' not in session.calls[0][1]['headers']['User-Agent']


def test_retry_after_and_403_not_bypassed(cfg, online):
    session=Session([Response(429,**{'Retry-After':'120'})])
    with pytest.raises(CollectionError,match='longer pause'):
        PoliteClient(cfg,session).prepare()
    assert len(session.calls)==1
    session=Session([Response(403)])
    with pytest.raises(CollectionError,match='403'):
        PoliteClient(cfg,session).prepare()
    assert len(session.calls)==1


def test_disabled_source_never_requests(cfg, online):
    session=Session([])
    with pytest.raises(CollectionError,match='disabled'):
        PoliteClient(replace(cfg,enabled=False),session).prepare()
    assert not session.calls


def test_sources_screen_is_read_only(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import pe.pages.sources as page
    monkeypatch.setattr(page,'DEFAULT_COVERAGE',tmp_path/'missing.json')
    def no_network(*args,**kwargs):pytest.fail('Public coverage must not collect')
    monkeypatch.setattr(PoliteClient,'prepare',no_network)
    app=AppTest.from_string('from pe.pages.sources import render_sources\nrender_sources()').run()
    assert not app.exception
    assert not app.button and not app.text_area
    assert any('cached coverage' in value.value for value in app.info)
    assert any('not enabled' in value.value for value in app.warning)


def test_sources_failed_attempt_displays_fallback(cfg,tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import pe.pages.sources as page
    store=CollectionStore(tmp_path/'private')
    collect(cfg,store,client=client_for('a'))
    bad=client_for('a');bad.pages[BASE+'/news/a']=CollectionError('Timeout')
    collect(cfg,store,client=bad)
    target=tmp_path/'coverage.json';write_json(target,store.coverage([cfg]))
    monkeypatch.setattr(page,'DEFAULT_COVERAGE',target)
    app=AppTest.from_string('from pe.pages.sources import render_sources\nrender_sources()').run()
    assert not app.exception
    assert any('previous successful snapshot' in value.value for value in app.warning)
    assert app.metric[1].value=='1'


def test_unsafe_urls_and_tracking(cfg):
    assert canonical_url(BASE+'/news/a?utm_medium=test#top') == BASE+'/news/a'
    for url in [BASE+'/news/../admin', BASE+'/news/%2e%2e/admin', BASE+'/news/%5cadmin',
                'http://firm.example/news/a', 'https://user:pass@firm.example/news/a']:
        with pytest.raises(ValueError):
            canonical_url(url)


def test_all_failed_and_empty_runs_keep_snapshot(cfg,tmp_path):
    store=CollectionStore(tmp_path)
    _,good=collect(cfg,store,client=client_for('a'))
    bad=client_for('a');bad.pages[BASE+'/news/a']=CollectionError('Timeout')
    _,failed=collect(cfg,store,client=bad)
    _,empty=collect(replace(cfg,start_date='2026-11-01'),store,client=client_for('a'))
    assert failed['status']=='failed' and empty['status']=='empty'
    assert store.load('firm')[1]['run_id']==good['run_id']


def test_page_cap_and_raw_provenance(cfg,tmp_path):
    fake=client_for('a');fake.pages[BASE+'/news/']+='<a rel="next" href="?page=2">next</a>'
    rows,report=collect(replace(cfg,max_pages=1),CollectionStore(tmp_path),client=fake)
    assert report['truncated'] and 'Listing-page limit reached' in report['stop_reasons']
    assert len(report['raw_inputs'])==2
    assert all((tmp_path/p['raw_path']).exists() for p in report['raw_inputs'])
    assert 'raw_inputs' not in CollectionStore.public_report(report)


def test_line_breaks_and_nested_list_paragraphs(cfg):
    raw=ARTICLE.replace('proposed transaction','proposed<br/>transaction')
    row=parse_announcement(raw,BASE+'/news/a',cfg,'2026-11-03T00:00:00Z')
    assert 'proposed transaction' in row['full_text'] and 'proposedtransaction' not in row['full_text']
    raw=ARTICLE.replace('</article>','<ul><li><p>Unique item</p></li></ul></article>')
    row=parse_announcement(raw,BASE+'/news/a',cfg,'2026-11-03T00:00:00Z')
    assert row['full_text'].count('Unique item')==1
