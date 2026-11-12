"""Deterministic lexical retrieval across the full saved collection before an AI call."""
from collections import Counter
from datetime import date, timedelta
import math
import re

STOP=set('a an the in of for to and or by with show find me give please tell about recent latest last days months years pe private equity acquisitions acquisition investments investment deals deal industry sector announced completed which what is are was were it its our'.split())
ALIASES={'tech':'technology','technologies':'technology','software':'technology',
         'acquisition':'acquire','acquisitions':'acquire','acquired':'acquire','acquires':'acquire',
         'buyout':'acquire','buyouts':'acquire','purchased':'acquire',
         'investments':'invest','investment':'invest','invested':'invest','invests':'invest',
         'health':'healthcare','medical':'healthcare','healthtech':'healthcare',
         'exits':'exit','exited':'exit','divestment':'exit','sold':'exit','divest':'exit',
         'usa':'unitedstates','us':'unitedstates','cyber':'cybersecurity'}
FIELDS=['publisher_pe_firm','buyer_or_investor','target_company','headline','transaction_type',
        'transaction_status','target_industry','target_geography','rationale_tags','evidence_excerpt']


def tokens(value):
    return [ALIASES.get(t,t) for t in re.findall(r'[\w]+',str(value).casefold())]


def retrieve_records(query, records, *, today=None, limit=80):
    """Rank all records, not an initial slice. Candidate retrieval is not final matching.

    Dates/transaction types are extra ranking hints. The model must still check the
    full question and reject nonmatches. Unknown dates remain eligible for review.
    """
    today=today or date.today()
    terms=set(tokens(query))-{ALIASES.get(t,t) for t in STOP}
    raw=tokens(query)
    # Preserve intent words even when they are common query boilerplate.
    for term in ('acquire','invest','exit'):
        if term in raw:terms.add(term)
    documents=[]
    for row in records:
        parts=[]
        for field in FIELDS:
            value=row.get(field)
            if isinstance(value,list):value=' '.join(map(str,value))
            parts.extend(tokens(value or ''))
            if field in {'publisher_pe_firm','target_company','target_industry'}:parts.extend(tokens(value or ''))
        documents.append(Counter(parts))
    n=len(records)
    freq=Counter(t for doc in documents for t in doc if t in terms)
    average=sum(sum(doc.values()) for doc in documents)/max(1,n)
    scores=[]
    query_tokens=set(tokens(query))
    topic=set()
    if re.search(r'\b(tech|technology|software|cybersecurity)\b',query,re.I):topic={'technology','cybersecurity'}
    elif re.search(r'\b(healthcare|medical|clinical)\b',query,re.I):topic={'healthcare','clinical'}
    recent=bool(re.search(r'\b(recent|latest|last|past)\b',query,re.I))
    for i,(row,doc) in enumerate(zip(records,documents)):
        length=sum(doc.values());score=0.0
        for term in terms:
            f=doc.get(term,0)
            if f:
                idf=math.log(1+(n-freq[term]+0.5)/(freq[term]+0.5))
                score+=idf*f*2.2/(f+1.2*(0.25+0.75*length/max(1,average)))
        industry=set(tokens(row.get('target_industry') or ''))
        if topic and industry.intersection(topic):score+=10
        publisher=set(tokens(row.get('publisher_pe_firm') or ''))
        if publisher and publisher.issubset(query_tokens):score+=8
        try:
            published=date.fromisoformat(str(row.get('publication_date'))[:10])
            ordinal=published.toordinal()
            if recent and 0<=(today-published).days<=90:score+=2
        except (ValueError,TypeError):ordinal=0
        scores.append((score,ordinal,str(row.get('deal_id','')),i))
    scores.sort(key=lambda x:(-x[0],-x[1],x[2]))
    selected=[records[x[3]] for x in scores[:max(1,limit)]]
    return selected,{'records_scanned':n,'candidates_supplied':len(selected),'candidate_limit':limit,
                     'method':'lexical relevance across all saved records; date/type hints; AI checks final matches',
                     'candidate_selection_limited':n>len(selected)}
