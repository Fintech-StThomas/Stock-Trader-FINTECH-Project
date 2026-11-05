"""Versioned reviewed deal snapshots, with previous-version and real-data fallback."""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pe.analytics import load_deals
from pe.published import BASE_DEALS, CURATED_DEALS, published_records
from pe.storage import write_json

ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT=ROOT/'data/pe'
PUBLISHED_ROOT=DATA_ROOT/'published'
MANIFEST_PATH=DATA_ROOT/'manifest.json'
DEFAULT_SAMPLE=DATA_ROOT/'sample_deals.json'  # optional explicitly labelled demo, never overwrites real data
DEFAULT_REAL=BASE_DEALS


def _read_json(path,default):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def _version(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}',value):
        raise ValueError('Use a version containing letters, numbers, hyphens or underscores')
    return value


def load_dataset(version=None):
    manifest=_read_json(MANIFEST_PATH,{})
    current=version or manifest.get('current_version')
    candidates=[current] if version else [current,manifest.get('previous_version')]
    for candidate in dict.fromkeys(v for v in candidates if v):
        try: candidate=_version(candidate)
        except ValueError:
            if version: raise
            continue
        path=PUBLISHED_ROOT/f'{candidate}.json'
        payload=_read_json(path,None)
        if isinstance(payload,dict) and isinstance(payload.get('deals'),list) and payload['deals']:
            return {'deals':load_deals(payload['deals']), 'records':payload['deals'], 'evidence':payload.get('evidence',[]),
                    'metadata':{**payload.get('metadata',{}),'version':candidate,'source':str(path),
                                'fallback':candidate!=current,'is_demo':bool(payload.get('metadata',{}).get('is_demo'))}}
    if Path(DEFAULT_REAL).exists():
        records=published_records(DEFAULT_REAL,CURATED_DEALS)
        return {'deals':load_deals(records),'records':records,'evidence':[],
                'metadata':{'version':'bundled-real-snapshot','source':str(DEFAULT_REAL),'is_demo':False,
                            'fallback':bool(current),'current_version_requested':current}}
    sample=_read_json(DEFAULT_SAMPLE,[])
    records=sample.get('deals',[]) if isinstance(sample,dict) else sample
    return {'deals':load_deals(records),'records':records,'evidence':[],
            'metadata':{'version':'demo-sample' if records else 'empty','source':str(DEFAULT_SAMPLE),
                        'is_demo':any(r.get('data_status')=='demo' for r in records),'fallback':bool(current)}}


def publish_dataset(deals,*,version=None,metadata=None,evidence=None):
    if not deals:raise ValueError('An empty dataset cannot replace the last successful publication')
    stamp=datetime.now(timezone.utc)
    version=_version(version or stamp.strftime('pe-%Y%m%dT%H%M%S')+'-'+str(stamp.microsecond))
    target=PUBLISHED_ROOT/f'{version}.json'
    if target.exists():raise ValueError('Version already exists; choose a new version to preserve history')
    frame=load_deals(deals)
    if frame.empty: raise ValueError("No valid deal IDs remain; publication refused")
    records=json.loads(frame.to_json(orient='records',date_format='iso'))
    for record in records:
        if record.get('publication_date'):record['publication_date']=record['publication_date'][:10]
    payload={'metadata':{'version':version,'published_at':stamp.isoformat(),**(metadata or {})},
             'deals':records,'evidence':evidence or []}
    write_json(target,payload)
    manifest=_read_json(MANIFEST_PATH,{})
    previous=manifest.get('current_version')
    manifest.update({'current_version':version,'previous_version':previous,'last_successful_update':stamp.isoformat(),
                     'deal_count':len(frame),'coverage':(metadata or {}).get('coverage',{})})
    if (metadata or {}).get('validation') is not None:manifest['validation']=metadata['validation']
    write_json(MANIFEST_PATH,manifest)
    return {'version':version,'path':str(target),'deal_count':len(frame)}
