"""Complete, normalized downloads for AI research facts and their source articles."""
import io
import json
import zipfile
import pandas as pd

FACTS=['deal_id','headline','firm','buyer','target','publication_date','transaction_type','transaction_status',
       'ownership_stake','deal_value','listing_status','listing_status_as_of','industry','geography','rationale',
       'origin','review_status','review_notes','saved_evidence']


def _json(value):return json.dumps(value,ensure_ascii=False,default=str)


def _cell(value):
    if isinstance(value,(dict,list,tuple)):value=_json(value)
    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):
        return "'"+value  # prevent spreadsheet formulas from external/model text
    return value


def _csv(rows,columns=None):
    frame=pd.DataFrame(rows,columns=columns)
    for column in frame.columns:frame[column]=frame[column].map(_cell)
    return frame.to_csv(index=False).encode('utf-8-sig')


def article_rows(result):
    """One row per article/source and its association with returned transactions."""
    rows={}
    for article in result.get('articles_consulted',[]):
        url=article['url']
        rows[url]={'article_url':url,'article_title':article.get('title') or '', 'source_kind':'Consulted during web research',
                   'deal_ids':[],'targets':[],'supported_fields':[],'supporting_quotes':[]}
    for index,deal in enumerate(result.get('results',[]),1):
        ident=deal.get('deal_id') or f'web-result-{index}'
        titles={a['url']:a.get('title') or '' for a in deal.get('articles',[])}
        for url in deal.get('source_urls',[]):
            row=rows.setdefault(url,{'article_url':url,'article_title':'','source_kind':'Supporting source',
                                    'deal_ids':[],'targets':[],'supported_fields':[],'supporting_quotes':[]})
            row['source_kind']='Supporting source'
            if titles.get(url):row['article_title']=titles[url]
            row['deal_ids'].append(ident);row['targets'].append(deal.get('target') or '')
            if deal.get('saved_evidence') and url in deal.get('saved_source_urls',deal.get('source_urls',[])):
                row['supporting_quotes'].append(deal['saved_evidence'])
            for ev in deal.get('evidence',[]):
                if ev['url']==url:
                    row['supported_fields'].append(ev['field']);row['supporting_quotes'].append(ev['quote'])
    for row in rows.values():
        for key in ['deal_ids','targets','supported_fields','supporting_quotes']:
            row[key]=' | '.join(dict.fromkeys(str(x) for x in row[key] if x))
    return list(rows.values())


def research_csv(result):
    rows=[]
    for index,row in enumerate(result.get('results',[]),1):
        evidence=row.get('evidence',[])
        records={**{k:row.get(k) for k in FACTS},'result_number':index,
                 'source_urls':' | '.join(row.get('source_urls',[])),
                 'article_titles':' | '.join(a.get('title') or a['url'] for a in row.get('articles',[])),
                 'field_evidence':_json(evidence),'full_result_json':_json(row),
                 'query':result.get('query'),'search_interpretation':result.get('interpretation'),
                 'searched_at':result.get('searched_at'),'model':result.get('model'),
                 'limitations':' | '.join(result.get('limitations',[])),
                 'retrieval_scope':_json(result.get('retrieval',{})),
                 'articles_consulted':_json(result.get('articles_consulted',[])),
                 'research_context_json':_json({k:v for k,v in result.items() if k!='results'})}
        for name,value in (row.get('saved_record') or {}).items():
            records['saved_'+name]=value
        # Separate evidence columns are easy to read in Excel, with no lost multi-source quotes.
        for field in dict.fromkeys(e['field'] for e in evidence):
            records[f'{field}_evidence']=' | '.join(e['quote'] for e in evidence if e['field']==field)
            records[f'{field}_sources']=' | '.join(dict.fromkeys(e['url'] for e in evidence if e['field']==field))
        rows.append(records)
    return _csv(rows) if rows else _csv([],FACTS+['source_urls','query'])


def articles_csv(result):
    return _csv(article_rows(result),['article_url','article_title','source_kind','deal_ids','targets','supported_fields','supporting_quotes'])


def evidence_csv(result):
    rows=[]
    for i,deal in enumerate(result.get('results',[]),1):
        ident=deal.get('deal_id') or f'web-result-{i}'
        if deal.get('saved_evidence'):
            rows.append({'deal_id':ident,'target':deal.get('target'),'field':'saved_evidence','quote':deal['saved_evidence'],
                         'source_url':' | '.join(deal.get('saved_source_urls',deal.get('source_urls',[]))),'origin':'Saved excerpt'})
        for ev in deal.get('evidence',[]):
            rows.append({'deal_id':ident,'target':deal.get('target'),'field':ev['field'],'quote':ev['quote'],'source_url':ev['url'],'origin':'AI-selected source evidence'})
    return _csv(rows,['deal_id','target','field','quote','source_url','origin'])


def research_bundle(result):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('deals.csv',research_csv(result))
        z.writestr('articles.csv',articles_csv(result))
        z.writestr('evidence.csv',evidence_csv(result))
        z.writestr('research.json',_json(result))
        z.writestr('README.txt','Lighthouse research export\n\nDeals contains all returned result facts, saved evidence, notes, article links, per-field evidence and the complete result JSON.\nArticles lists all returned supporting sources plus other sources consulted by web search, marked separately. A consulted URL is not proof of a deal.\nEvidence has one row per field quotation or saved excerpt.\nBlank facts mean unknown/not disclosed. Full article bodies are not part of this export; source URLs open the original articles.\nCSV cells that could run a spreadsheet formula are escaped with an apostrophe. The JSON preserves original values.\n\nQuery: '+str(result.get('query',''))+'\nInterpretation: '+str(result.get('interpretation',''))+'\nLimitations: '+'; '.join(result.get('limitations',[])))
    return buffer.getvalue()
