import csv
import io
import json
import zipfile
from datetime import date
from pe.retrieval import retrieve_records
from pe.research_exports import research_csv,articles_csv,evidence_csv,research_bundle


def result():
    return {'query':'tech acquisitions','interpretation':'technology deals','searched_at':'2026-11-04',
            'model':'test','limitations':['Selected sources only.'],
            'retrieval':{'records_scanned':350,'candidates_supplied':80},
            'articles_consulted':[{'url':'https://example.com/a','title':'Actual source title'},
                                  {'url':'https://example.com/context','title':'Context only'}],
            'results':[{'deal_id':'D1','headline':'Original announcement headline','firm':'Sponsor','buyer':'Buyer','target':'Target',
                'publication_date':'2026-10-20','transaction_type':'Acquisition','transaction_status':'Announced',
                'ownership_stake':None,'deal_value':None,'listing_status':None,'listing_status_as_of':None,
                'industry':'Technology','geography':None,'rationale':'Product expansion','origin':'Saved record',
                'review_status':'needs review','review_notes':'Check closing date',
                'saved_evidence':'Saved exact source excerpt.','saved_source_urls':['https://example.com/a'],
                'source_urls':['https://example.com/a','https://example.com/b'],
                'articles':[{'url':'https://example.com/a','title':'Original announcement headline'},
                            {'url':'https://example.com/b','title':'Additional source'}],
                'evidence':[{'field':'rationale','quote':'New product capability','url':'https://example.com/a'},
                            {'field':'rationale','quote':'Further product evidence','url':'https://example.com/b'}]}]}


def read(data):return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def test_all_facts_evidence_notes_and_articles_survive_csv():
    r=read(research_csv(result()))[0]
    assert r['saved_evidence']=='Saved exact source excerpt.'
    assert r['review_notes']=='Check closing date'
    assert r['headline']=='Original announcement headline'
    assert 'https://example.com/b' in r['source_urls']
    assert 'Additional source' in r['article_titles']
    assert len(json.loads(r['field_evidence']))==2
    assert 'Further product evidence' in r['rationale_evidence']
    assert json.loads(r['full_result_json'])==result()['results'][0]
    assert r['ownership_stake']==''


def test_articles_distinguish_consulted_pages_from_supporting_evidence():
    rows={r['article_url']:r for r in read(articles_csv(result()))}
    assert len(rows)==3
    assert rows['https://example.com/context']['source_kind']=='Consulted during web research'
    assert rows['https://example.com/context']['targets']==''
    assert rows['https://example.com/a']['source_kind']=='Supporting source'
    assert 'Saved exact source excerpt.' in rows['https://example.com/a']['supporting_quotes']
    assert len(read(evidence_csv(result())))==3


def test_zip_has_complete_independently_readable_contents():
    with zipfile.ZipFile(io.BytesIO(research_bundle(result()))) as z:
        assert set(z.namelist())=={'deals.csv','articles.csv','evidence.csv','research.json','README.txt'}
        assert json.loads(z.read('research.json'))==result()
        assert len(read(z.read('articles.csv')))==3


def test_csv_escapes_formulas_but_json_preserves_original():
    r=result();r['results'][0]['target']='=HYPERLINK("bad")'
    row=read(research_csv(r))[0]
    assert row['target'].startswith("'=")
    assert json.loads(row['full_result_json'])['target'].startswith('=')


def test_retrieval_finds_relevant_record_beyond_first_200():
    data=[{'deal_id':f'D{i}','target_company':f'Steel company {i}','target_industry':'Industrials','publication_date':'2026-10-01'} for i in range(350)]
    data.append({'deal_id':'AFTER350','target_company':'Cybersecurity Software','target_industry':'Technology','transaction_type':'Acquisition','publication_date':'2026-10-30'})
    selected,scope=retrieve_records('recent tech acquisitions',data,today=date(2026,11,4),limit=20)
    assert selected[0]['deal_id']=='AFTER350'
    assert scope['records_scanned']==351 and scope['candidates_supplied']==20
    assert len({x['deal_id'] for x in selected})==20


def test_suggestion_fills_centered_search_without_calling_ai(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from pe.agent import AgentSettings
    import pe.pages.agent_search as page
    monkeypatch.setattr(page,'settings',lambda *a:AgentSettings(''))
    monkeypatch.setattr(page,'research',lambda *a:(_ for _ in ()).throw(AssertionError('Suggestion must not call paid API')))
    app=AppTest.from_string('from pe.pages.agent_search import render_agent_search\nrender_agent_search()').run()
    assert not app.exception and not app.sidebar.text_input and not app.sidebar.text_area
    app.button(key='pe_prompt_0').click().run()
    assert app.text_area(key='pe_ai_question').value==page.SUGGESTIONS[0][1]
