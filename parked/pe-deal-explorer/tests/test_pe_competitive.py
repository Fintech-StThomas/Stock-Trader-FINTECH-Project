import json
from pathlib import Path
from pe.published import published_records, published_inventory, BASE_DEALS, CURATED_DEALS
from pe.pages.explorer import load_explorer_dataset
from pe.analytics import filter_deals
from pe.exports import export_filtered, export_competitive_summary


def test_curated_additions_are_traceable_and_not_historical_validation():
    base=json.loads(BASE_DEALS.read_text());extra=json.loads(CURATED_DEALS.read_text())
    assert len(base)==115 and len(extra)==4
    assert {x['publisher_pe_firm'] for x in extra}=={'Advent International','Bain Capital'}
    assert all(x['data_status']=='curated' and x['review_status']=='needs review' and x['source_url'] and x['evidence_excerpt'] for x in extra)
    assert all(x['price'] is None for x in extra)
    kahua=next(x for x in extra if x['target_company']=='Kahua')
    hotel=next(x for x in extra if x['target_company']=='The Lake Como EDITION')
    assert hotel['buyer_or_investor']=='Sansiri Capital' and hotel['transaction_type']=='Exit'
    assert kahua['publication_date'] is None and hotel['publication_date'] is None


def test_full_competitive_flow_and_export():
    df,demo=load_explorer_dataset()
    assert len(df)==119 and not demo
    selected=filter_deals(df,publishers=['Advent International'],industries=['Healthcare'],geographies=['New Zealand and Australia'],rationales=['Geographic reach'],start_date='2026-08-01',end_date='2026-09-01')
    assert len(selected)==1 and selected.iloc[0]['target_company']=='NZCR Group'
    csv=export_filtered(selected).decode('utf-8-sig')
    summary=export_competitive_summary(selected).decode()
    assert 'NZCR Group' in csv and 'https://www.adventinternational.com/' in csv
    assert 'NZCR Group' in summary and 'Vermeg' not in summary and 'Kahua' not in summary
    assert 'Advent International: 1' in summary and 'Selected unique deal records: 1' in summary


def test_inventory_matches_combined_data():
    records=published_records();inventory=published_inventory(records)
    assert sum(x['Published deal records'] for x in inventory)==119
    assert sum(x['Curated samples'] for x in inventory)==4
    assert len(inventory)==6


def test_repeated_supplemental_urls_do_not_inflate_counts(tmp_path):
    extra=json.loads(CURATED_DEALS.read_text())
    base=tmp_path/'base.json';curated=tmp_path/'curated.json'
    base.write_text(json.dumps(extra[:1]));curated.write_text(json.dumps(extra+extra))
    assert len(published_records(base,curated))==4


def test_unknown_dates_stay_visible_until_explicit_date_filter():
    df,_=load_explorer_dataset()
    assert len(filter_deals(df,publishers=['Bain Capital']))==2
    assert filter_deals(df,publishers=['Bain Capital'],start_date='2026-01-01',end_date='2026-12-31').empty


def test_ui_firm_choice_evidence_and_summary():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('from pe.pages.explorer import render_legacy_explorer\nrender_legacy_explorer()').run(timeout=30)
    firms=app.sidebar.multiselect(key='pe_explorer_publishers')
    assert 'Bain Capital' in firms.options and 'Advent International' in firms.options
    firms.select('Advent International').run()
    assert not app.exception and int(app.metric[0].value)==2
    assert any('Curated source-linked sample' in x.value for x in app.caption)
