"""Immutable private run archives and atomic last-known-good pointers.

This is collection storage, not Jacob's future shared deal repository. The public
coverage export is explicitly whitelisted and contains no raw HTML or article text.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE = ROOT / 'data' / 'pe' / 'private'
DEFAULT_COVERAGE = ROOT / 'data' / 'pe' / 'coverage.json'


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.writing-', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class CollectionStore:
    def __init__(self, root=DEFAULT_PRIVATE):
        self.root = Path(root)

    @contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / '.refresh.lock'
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError('A refresh is already running (or a stale lock needs team review)') from exc
        try:
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            yield
        finally:
            path.unlink(missing_ok=True)

    def source_dir(self, source_id):
        import re
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,50}', source_id):
            raise ValueError('Invalid source ID')
        return self.root / source_id

    def read_pointer(self, source_id):
        p = self.source_dir(source_id) / 'index.json'
        return json.loads(p.read_text()) if p.exists() else {}

    def load(self, source_id, version=None):
        version = version or self.read_pointer(source_id).get('last_good_run')
        if not version:
            return [], None
        if Path(version).name != version or not version.startswith('run_'):
            raise ValueError('Invalid run version')
        p = self.source_dir(source_id) / 'runs' / version
        records = json.loads((p / 'announcements.json').read_text())
        for row in records:
            row['is_cached'] = True
        return records, json.loads((p / 'manifest.json').read_text())

    def save_raw(self, source_id, raw):
        from pe.collectors.parsing import digest
        content_hash = digest(raw)
        relative = Path(source_id) / 'raw' / (content_hash + '.html')
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=target.parent, prefix='.raw-')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return relative.as_posix()

    def save_run(self, config, records, report):
        version = report['run_id']
        dest = self.source_dir(config.source_id) / 'runs' / version
        dest.mkdir(parents=True, exist_ok=False)
        write_json(dest / 'announcements.json', records)
        write_json(dest / 'manifest.json', report)
        pointer = self.read_pointer(config.source_id)
        pointer['last_attempt_run'] = version
        # Partial/failed runs are retained for diagnosis, never promoted.
        if report['status'] == 'success' and records:
            pointer['last_good_run'] = version
        write_json(self.source_dir(config.source_id) / 'index.json', pointer)

    def coverage(self, configs):
        result = {'schema_version': 1, 'generated_at': utc_now(), 'delivery_mode': 'cached', 'sources': []}
        for config in configs:
            pointer = self.read_pointer(config.source_id)
            current = self.load(config.source_id)[1]
            attempted = self.load(config.source_id, pointer['last_attempt_run'])[1] if pointer.get('last_attempt_run') else None
            entry = {k: v for k, v in config.to_dict().items() if k in {
                'source_id', 'publisher_pe_firm', 'listing_url', 'enabled', 'access_status', 'terms_url',
                'access_note', 'access_reviewed_on'}}
            entry.update({'last_successful_run': current['finished_at'] if current else None,
                          'published_run_id': current['run_id'] if current else None,
                          'published_coverage': self.public_report(current),
                          'latest_attempt': self.public_report(attempted)})
            result['sources'].append(entry)
        return result

    @staticmethod
    def public_report(report):
        if not report:
            return None
        return {k: deepcopy(v) for k, v in report.items() if k in {
            'run_id', 'started_at', 'finished_at', 'mode', 'status', 'requested_period', 'observed_period',
            'counts', 'limits', 'truncated', 'stop_reasons', 'coverage_note', 'errors', 'pages'}}


def run_id():
    return 'run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8]
