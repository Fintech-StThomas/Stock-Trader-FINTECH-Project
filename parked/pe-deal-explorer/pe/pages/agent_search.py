"""One-question PE research interface. Public visitors never enter an API key."""
import json
import time
import pandas as pd
import streamlit as st
from pe.agent import settings, research, ResearchError
from pe.published import published_records


SUGGESTIONS = [
 ('Recent tech acquisitions', 'Find up to five private equity acquisitions in the technology industry announced in the last 90 days. For each, show the PE sponsor, direct buyer, target company, announcement date, announced or completed status, disclosed ownership percentage and transaction value, and public/private listing status with the date of its evidence. Include the stated acquisition rationale, supporting excerpts and original source links. Mark missing facts as Not disclosed or Not verified. Do not confuse company valuation with the amount paid or assume an announced take-private deal has completed.'),
 ('Compare Advent and Bain', 'Compare available Advent International and Bain Capital transaction announcements from the last 12 months. Show sectors, targets, transaction types and stated rationales with sources. Explain coverage gaps instead of treating sample counts as total firm activity.'),
 ('Healthcare add-on deals', 'Find healthcare add-on acquisitions announced in the last six months. Identify the direct portfolio-company buyer, PE sponsor, target, geography and stated acquisition rationale. Link each supporting announcement.'),
 ('PE exits and buyers', 'Find recent private equity exits. Show the selling sponsor, actual buyer, target, announced/completed status and disclosed deal value. Keep the seller separate from the buyer and cite the evidence.'),
 ('Minority technology investments', 'Find minority growth investments in technology companies from the last 12 months. Show any disclosed stake and transaction amount, keeping company valuations separate. Include dates, evidence and source links.'),
 ('Take-private status', 'Find take-private transactions announced in the last 90 days. Distinguish signed agreements from completed acquisitions and verified delistings. Show dated exchange or company evidence for public/private status and leave unsupported status unknown.'),
]


def use_prompt(prompt):
    st.session_state['pe_ai_question']=prompt


def inspect_saved_deal(deal_id):
    """Callbacks run before navigation widgets are built on the next render."""
    st.session_state['pe_selected_deal_id']=deal_id
    st.session_state['pe_page']='detail'
    st.query_params['pe_page']='detail'


def render_agent_search(records=None):
    from pe.repository import load_dataset
    records=load_dataset()["records"] if records is None else records
    try:
        secret_values=dict(st.secrets)
    except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError):
        secret_values={}
    config=settings(secret_values)
    st.markdown("""<style>
    .st-key-pe_search_center {max-width: 1000px; margin-left: auto; margin-right: auto; padding-top: 1.5rem;}
    .st-key-pe_search_center textarea {font-size: 1.12rem; line-height: 1.6; border-radius: 18px; padding: 18px;}
    .st-key-pe_search_center [data-testid="stTextArea"] > div {border-radius: 18px;}
    .st-key-pe_search_center [data-testid="stForm"] {padding: 0; border: none;}
    .pe-research-heading {text-align:center; font-size:2rem; font-weight:650; margin-bottom:0.3rem;}
    .pe-research-intro {text-align:center; color:#5c6874; margin-bottom:1.4rem;}
    </style>""",unsafe_allow_html=True)
    with st.container(key='pe_search_center'):
        st.markdown('<div class="pe-research-heading">Research private equity deals</div>',unsafe_allow_html=True)
        st.markdown('<div class="pe-research-intro">Ask about acquisitions, ownership, deal values, and the evidence behind them.</div>',unsafe_allow_html=True)
        with st.form('pe_ai_search_form',border=False):
            question=st.text_area('What would you like to research?',key='pe_ai_question',max_chars=1200,height=140,
                placeholder='Find recent PE acquisitions in the technology industry…',label_visibility='collapsed')
            submitted=st.form_submit_button('Search deals',type='primary',width='stretch')
        st.caption('Try a suggested prompt — click to fill the search box, then edit or search.')
        for start in (0,3):
            cols=st.columns(3)
            for i in range(start,start+3):
                cols[i-start].button(SUGGESTIONS[i][0],key=f'pe_prompt_{i}',on_click=use_prompt,args=(SUGGESTIONS[i][1],),width='stretch',help=SUGGESTIONS[i][1])
        with st.expander('See the detailed acquisition prompt'):
            st.write(SUGGESTIONS[0][1])
            st.button('Use this detailed prompt',key='pe_prompt_detailed',on_click=use_prompt,args=(SUGGESTIONS[0][1],))
        firms=len({r.get('publisher_pe_firm') for r in records if r.get('publisher_pe_firm')})
        st.caption(f'{len(records):,} saved deal records · {firms} firms · '+('Web research enabled' if config.web_search else 'Saved data research')+'. Search questions are sent to OpenAI.')
    if submitted:
        last=st.session_state.get('pe_ai_last_request',0)
        if time.monotonic()-last<15:
            st.warning('Please wait a few seconds before starting another search.')
        elif st.session_state.get('pe_ai_session_requests',0)>=20:
            st.warning('This session has reached its AI search allowance.')
        else:
            st.session_state.pop('pe_ai_result',None)
            try:
                if config.api_key:
                    st.session_state['pe_ai_last_request']=time.monotonic()
                    st.session_state['pe_ai_session_requests']=st.session_state.get('pe_ai_session_requests',0)+1
                with st.spinner('Researching deal announcements and checking available evidence…'):
                    st.session_state['pe_ai_result']=research(question,records,config)
            except ResearchError as exc:
                st.error(str(exc))
    result=st.session_state.get('pe_ai_result')
    if not result:
        if not config.api_key:
            st.info('AI search is ready for setup. Your website owner must connect an OpenAI API project before live searches can run.')
        st.write('Results will show buyer and target, transaction status, ownership stake, disclosed deal value, and dated public/private listing evidence.')
        st.caption('Missing information stays “Not disclosed” or “Not verified.” An announced acquisition does not establish completed ownership or delisting.')
        return
    st.markdown('#### Your research')
    st.write(result['query']);st.write(result['interpretation'])
    st.caption(f"Searched {result['searched_at']} · {'Web search enabled' if result['web_search_enabled'] else 'Saved dataset only'} · Up to 8 results. “Recent” defaults to 90 days unless you specify a period.")
    scope=result.get('retrieval')
    if scope:
        st.caption(f"Saved-data retrieval: searched all {scope['records_scanned']:,} records and supplied the {scope['candidates_supplied']:,} highest-ranked candidates for AI review.")
    for limitation in result['limitations']:st.caption(limitation)
    if not result['results']:
        st.info('No supported matching transactions were found. Try another period, sector, or firm.')
    for row in result['results']:
        with st.container(border=True):
            st.subheader(f"{row['buyer']} → {row['target']}")
            st.caption(f"{row['origin']} · {row['review_status']} · Publisher / sponsor: {row['firm'] or 'Not verified'}")
            table=[{'Fact':label,'Available information':row.get(field) or unknown} for field,label,unknown in [
                ('publication_date','Publication date','Not verified'),('transaction_type','Transaction type','Not verified'),
                ('transaction_status','Transaction status','Not verified'),('industry','Target industry','Not verified'),
                ('geography','Target geography','Not verified'),('ownership_stake','Ownership stake acquired','Not disclosed'),
                ('deal_value','Disclosed transaction value','Not disclosed'),('listing_status','Public / private listing status','Not verified'),
                ('listing_status_as_of','Listing evidence as of','Not verified'),('rationale','Stated / classified rationale','Not disclosed')]]
            st.dataframe(table,hide_index=True,width='stretch')
            if row['saved_evidence']:
                st.markdown('**Saved supporting evidence**');st.write(row['saved_evidence'])
            if row.get('review_notes'):st.caption(row['review_notes'])
            for ev in row['evidence']:
                st.markdown(f"**{ev['field'].replace('_',' ').capitalize()} evidence**")
                st.write(ev['quote']);st.link_button('Open supporting source',ev['url'])
            if row['deal_id']:
                st.button('Inspect saved deal',key='pe_ai_detail_'+row['deal_id'],
                          on_click=inspect_saved_deal,args=(row['deal_id'],))
            if row['source_urls']:
                st.markdown('**Source articles**')
                titles={a['url']:a.get('title') for a in row.get('articles',[])}
                for url in row['source_urls']:
                    st.link_button(titles.get(url) or url,url)
    st.caption('AI-extracted web facts require review. Listing status is dated evidence, not a guarantee of current exchange status. Coverage is selective.')
    st.markdown('#### Download this research')
    from pe.research_exports import research_csv, articles_csv, research_bundle
    cols=st.columns(3)
    cols[0].download_button('All deal information (CSV)',research_csv(result),file_name='lighthouse_ai_deals.csv',mime='text/csv',key='pe_ai_csv',width='stretch')
    cols[1].download_button('Articles and sources (CSV)',articles_csv(result),file_name='lighthouse_ai_articles.csv',mime='text/csv',key='pe_ai_articles_csv',width='stretch')
    cols[2].download_button('Complete research pack (ZIP)',research_bundle(result),file_name='lighthouse_ai_research.zip',mime='application/zip',key='pe_ai_bundle',width='stretch')
    st.download_button('Full research JSON',json.dumps(result,ensure_ascii=False,indent=2),file_name='lighthouse_ai_research.json',mime='application/json',key='pe_ai_download')
    st.caption('Downloads include all returned deal facts, saved excerpts, review notes, field evidence, article titles, URLs, and search details. Source links open the original full articles.')
