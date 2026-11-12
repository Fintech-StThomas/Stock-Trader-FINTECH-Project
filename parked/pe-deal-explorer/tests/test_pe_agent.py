import json
from datetime import date
from dataclasses import replace
import pytest
import requests
from pe.agent import AgentSettings, ResearchError, FIELDS, ground_results, research, settings

TODAY=date(2026,11,4)
URL='https://firm.example/news/deal'
SAVED={'deal_id':'D1','publisher_pe_firm':'Sponsor','buyer_or_investor':'Buyer','target_company':'Target',
       'source_url':[URL],'publication_date':'2026-10-10','transaction_type':'Acquisition',
       'transaction_status':'Announced','target_industry':'Technology','target_geography':None,
       'rationale_tags':['Product expansion'],'price':None,'evidence_excerpt':'Buyer agreed to acquire Target.',
       'review_status':'needs review','review_notes':None}


def row(saved='D1',**values):
    return dict(saved_deal_id=saved,**{f:values.get(f) for f in FIELDS},evidence=values.get('evidence',[]))


def payload(*rows):
    return {'interpretation':'Technology acquisitions in the last 90 days.','limitations':['Selected sources only.'],'results':list(rows)}


def test_saved_facts_cannot_be_rewritten_and_unknowns_are_preserved():
    p=payload(row(buyer='Wrong buyer',ownership_stake='100%',listing_status='private',deal_value='$2bn'))
    r=ground_results(p,[SAVED],set(),TODAY)['results'][0]
    assert r['buyer']=='Buyer' and r['transaction_status']=='Announced'
    assert r['ownership_stake'] is None and r['listing_status'] is None and r['deal_value'] is None


def test_fake_ids_and_uncited_web_records_are_dropped():
    p=payload(row('invented'),row(None,buyer='B',target='T'))
    assert ground_results(p,[SAVED],set(),TODAY)['results']==[]


def test_cited_web_facts_and_dated_listing_status():
    values={'buyer':'Buyer','target':'Target','ownership_stake':'60%','listing_status':'public','listing_status_as_of':'2026-11-03'}
    evidence=[{'field':f,'quote':v,'url':URL} for f,v in values.items()]
    r=ground_results(payload(row(None,**values,evidence=evidence)),[],{URL},TODAY)['results'][0]
    assert r['ownership_stake']=='60%' and r['listing_status']=='public'
    assert r['origin']=='Web research' and r['review_status']=='needs review'
    p=payload(row(None,**dict(values,listing_status_as_of=None),evidence=evidence))
    assert ground_results(p,[],{URL},TODAY)['results'][0]['listing_status'] is None


def test_future_dates_and_valuation_are_not_transaction_value():
    evidence=[{'field':'deal_value','quote':'at a valuation above $1 billion','url':URL},
              {'field':'publication_date','quote':'2027-01-01','url':URL}]
    r=ground_results(payload(row(deal_value='$1bn',publication_date='2027-01-01',evidence=evidence)),[SAVED],{URL},TODAY)['results'][0]
    assert r['deal_value'] is None
    # The model cannot replace the original saved publication date.
    assert r['publication_date']=='2026-10-10'


def test_bad_schema_raises_safe_error():
    with pytest.raises(ResearchError,match='verified'):
        ground_results({'results':[]},[],set())


class Response:
    status_code=200
    def __init__(self,p):self.p=p
    def json(self):return self.p


class Session:
    def __init__(self,response):self.response=response;self.calls=[]
    def post(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        if isinstance(self.response,Exception):raise self.response
        return self.response


def answer(p):return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(p)}]}]}


@pytest.fixture
def online(monkeypatch):
    monkeypatch.delenv('LIGHTHOUSE_OFFLINE',raising=False)
    monkeypatch.setattr('pe.agent._used',0)


def test_request_uses_server_key_schema_no_storage_and_no_key_in_input(online):
    session=Session(Response(answer(payload(row()))))
    r=research('recent tech acquisitions',[SAVED],AgentSettings('secret-key'),session=session,today=TODAY)
    args,kwargs=session.calls[0]
    assert args[0]=='https://api.openai.com/v1/responses'
    assert kwargs['headers']['Authorization']=='Bearer secret-key'
    body=kwargs['json']
    assert body['store'] is False and body['text']['format']['strict'] is True
    assert body['tools'][0]['type']=='web_search'
    assert 'secret-key' not in body['input'] and '2026-11-04' in body['input']
    assert r['results'][0]['target']=='Target'


def test_missing_key_offline_and_limits_never_request(monkeypatch,online):
    session=Session(None)
    with pytest.raises(ResearchError,match='not connected'):
        research('tech',[],AgentSettings(''),session=session)
    monkeypatch.setenv('LIGHTHOUSE_OFFLINE','1')
    with pytest.raises(ResearchError,match='offline'):
        research('tech',[],AgentSettings('key'),session=session)
    assert not session.calls


def test_provider_errors_do_not_expose_credentials(online):
    response=Response({'error':'secret-key'});response.status_code=401
    with pytest.raises(ResearchError) as exc:
        research('tech',[],AgentSettings('secret-key'),session=Session(response))
    assert 'secret-key' not in str(exc.value)
    with pytest.raises(ResearchError,match='reached'):
        research('tech',[],AgentSettings('secret-key',daily_limit=1),session=Session(response))


def test_failed_or_incomplete_output_is_not_a_result(online):
    with pytest.raises(ResearchError,match='complete result'):
        research('tech',[],AgentSettings('key'),session=Session(Response({'status':'incomplete'})))
    with pytest.raises(ResearchError,match='reached'):
        research('tech',[],AgentSettings('key'),session=Session(requests.Timeout()))


def test_search_ui_one_input_no_filters_no_network_without_key(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import pe.pages.agent_search as page
    monkeypatch.setattr(page,'settings',lambda *a:AgentSettings(''))
    app=AppTest.from_string('from pe.pages.agent_search import render_agent_search\nrender_agent_search()').run()
    assert not app.exception and len(app.text_area)==1 and not app.sidebar.text_area
    assert not app.sidebar.multiselect and not app.sidebar.date_input
    app.text_area(key='pe_ai_question').set_value('Recent technology acquisitions')
    app.button(key='FormSubmitter:pe_ai_search_form-Search deals').click().run()
    assert not app.exception and any('not connected' in x.value for x in app.error)


def test_search_results_persist_without_repeated_api_calls(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import pe.pages.agent_search as page
    monkeypatch.setattr(page,'settings',lambda *a:AgentSettings('key'))
    calls=[]
    def fake(*args):
        calls.append(args)
        r=ground_results(payload(row()),[SAVED],set(),TODAY)
        return dict(r,model='fake',web_search_enabled=False,query=args[0])
    monkeypatch.setattr(page,'research',fake)
    app=AppTest.from_string('from pe.pages.agent_search import render_agent_search\nrender_agent_search()').run()
    app.text_area(key='pe_ai_question').set_value('tech')
    app.button(key='FormSubmitter:pe_ai_search_form-Search deals').click().run()
    assert not app.exception and len(calls)==1
    app.run();assert len(calls)==1
    assert any('Buyer → Target' in x.value for x in app.subheader)
    assert any('Not disclosed' in str(x.value) for x in app.dataframe)


def test_saved_url_cannot_launder_invented_evidence():
    candidate=row(ownership_stake='100%',evidence=[{'field':'ownership_stake','quote':'all shares were acquired','url':URL}])
    r=ground_results(payload(candidate),[SAVED],set(),TODAY)['results'][0]
    assert r['ownership_stake'] is None


def test_api_source_metadata_supports_web_results(online):
    values={'buyer':'Buyer','target':'Target'}
    ev=[{'field':f,'quote':v,'url':URL} for f,v in values.items()]
    raw=answer(payload(row(None,**values,evidence=ev)))
    raw['output'].insert(0,{'type':'web_search_call','action':{'sources':[{'url':URL}]}})
    result=research('technology acquisitions',[],AgentSettings('key'),session=Session(Response(raw)),today=TODAY)
    assert len(result['results'])==1 and result['results'][0]['source_urls']==[URL]


def test_invalid_source_schemes_are_not_cited():
    values={'buyer':'Buyer','target':'Target'}
    ev=[{'field':f,'quote':v,'url':'javascript:evil()'} for f,v in values.items()]
    assert not ground_results(payload(row(None,**values,evidence=ev)),[],{'javascript:evil()'},TODAY)['results']


def test_missing_saved_fields_can_be_enriched_only_from_cited_web_evidence():
    p=payload(row(geography='Canada',evidence=[{'field':'geography','quote':'based in Canada','url':URL}]))
    r=ground_results(p,[SAVED],{URL},TODAY)['results'][0]
    assert r['geography']=='Canada' and 'geography' in r['enriched_fields']
    assert r['saved_record']['buyer_or_investor']=='Buyer'
