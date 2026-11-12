"""Team-only command; intentionally not exposed as a Streamlit action."""
import argparse
from dataclasses import replace
import json
from .config import CONFIG_PATH, load_sources
from .pipeline import collect
from .http import CollectionError
from pe.storage import CollectionStore, DEFAULT_PRIVATE, DEFAULT_COVERAGE, write_json


def main():
    p = argparse.ArgumentParser(description='Collect PE announcements or load a saved offline snapshot.')
    p.add_argument('--config', default=str(CONFIG_PATH))
    p.add_argument('--source', action='append', help='Repeat for multiple source IDs; defaults to all enabled sources')
    p.add_argument('--start'); p.add_argument('--end')
    p.add_argument('--max-pages', type=int); p.add_argument('--max-articles', type=int)
    p.add_argument('--private-dir', default=str(DEFAULT_PRIVATE))
    p.add_argument('--coverage-output', default=str(DEFAULT_COVERAGE))
    p.add_argument('--research-sources', action='store_true', help='Use the additional research-publisher collector')
    p.add_argument('--cached', action='store_true', help='Read last successful snapshot without requests or writes')
    args = p.parse_args()
    if args.research_sources:
        if args.source or args.start or args.end or args.max_pages or args.max_articles:
            p.error('Research-publisher mode uses its own bounded source configuration, not firm/date overrides')
        from pe.collectors.core import collect as collect_research, load_source_config, CONFIG_PATH as RESEARCH_CONFIG
        config_path = RESEARCH_CONFIG if args.config == str(CONFIG_PATH) else args.config
        rows, report = collect_research(load_source_config(config_path), cached=args.cached)
        print(json.dumps({'announcements': len(rows), 'report': report}, indent=2))
        return 1 if report.get('failed') else 0
    configs = load_sources(args.config)
    ids = set(args.source or [s.source_id for s in configs if s.enabled])
    unknown = ids - {s.source_id for s in configs}
    if unknown:
        p.error('Unknown sources: ' + ', '.join(sorted(unknown)))
    selected = [s for s in configs if s.source_id in ids]
    if not selected:
        p.error('No enabled sources selected')
    overrides = {k: v for k, v in {'start_date': args.start, 'end_date': args.end,
                 'max_pages': args.max_pages, 'max_articles': args.max_articles}.items() if v is not None}
    if args.cached and overrides:
        p.error('Cached mode replays saved coverage; date and collection limits cannot be overridden')
    selected = [replace(s, **overrides) for s in selected]
    store = CollectionStore(args.private_dir)
    exit_code = 0

    def run():
        nonlocal exit_code
        for config in selected:
            try:
                rows, report = collect(config, store, cached=args.cached)
                print(json.dumps({'source': config.source_id, 'mode': report['mode'], 'status': report['status'],
                                  'announcements': len(rows), 'run_id': report['run_id'],
                                  'truncated': report['truncated'], 'errors': report['errors']}))
                if report['status'] in {'failed', 'partial', 'empty'}:
                    exit_code = 1
            except CollectionError as exc:
                print(json.dumps({'source': config.source_id, 'error': str(exc)})); exit_code = 1
    if args.cached:
        run()
    else:
        with store.lock():
            run()
            write_json(args.coverage_output, store.coverage(configs))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
