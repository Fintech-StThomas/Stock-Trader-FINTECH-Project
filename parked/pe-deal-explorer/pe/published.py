"""Shared read-only published records; keep curated samples separate from tested extraction output."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_DEALS = ROOT / 'data/pe/deals.json'
CURATED_DEALS = ROOT / 'data/pe/curated_deals.json'


def published_records(base_path=BASE_DEALS, curated_path=CURATED_DEALS):
    records = json.loads(Path(base_path).read_text()) if Path(base_path).exists() else []
    if curated_path and Path(curated_path).exists():
        supplemental = json.loads(Path(curated_path).read_text())
        ids = {row['deal_id'] for row in records}
        urls = {url for row in records for url in row.get('source_url', [])}
        for row in supplemental:
            if row['deal_id'] not in ids and not urls.intersection(row.get('source_url', [])):
                records.append(row); ids.add(row['deal_id']); urls.update(row.get('source_url', []))
    return records


def published_inventory(records):
    groups = {}
    for row in records:
        firm = row['publisher_pe_firm']
        group = groups.setdefault(firm, {'PE firm':firm,'Published deal records':0,'Curated samples':0,'Known dates':[]})
        group['Published deal records'] += 1
        group['Curated samples'] += row.get('data_status') == 'curated'
        if row.get('publication_date'):
            group['Known dates'].append(row['publication_date'])
    for group in groups.values():
        dates = sorted(group.pop('Known dates'))
        group['Observed publication dates'] = f'{dates[0]} to {dates[-1]}' if dates else 'Unknown — review pending'
    return sorted(groups.values(),key=lambda row:row['PE firm'])
