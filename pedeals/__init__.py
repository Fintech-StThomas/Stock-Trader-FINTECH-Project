"""
PE Deal Intelligence - public private-equity deal announcements turned into an evidence-backed deal record.

A separate product that lives in the Lighthouse dashboard (its own tab) and shares nothing with the stock engine but
the app. For a portfolio operations manager at a lower-middle-market sponsor: which add-on acquisitions are peer
sponsors' platforms making, what capabilities do they say each one adds, and how fast do they build.

Public API
----------
>>> from pedeals import build_dataset
>>> ds = build_dataset()                    # the saved (cached) pages: runs offline, reproducible
>>> ds.deals.head()                         # one row per deal announcement, every field with its evidence
>>> ds.coverage                             # what was collected, skipped and why, per firm
>>> from pedeals.validate import evaluate
>>> evaluate(ds).summary                    # accuracy against the hand-labelled reference set

>>> from pedeals.collect import collect     # live collection (polite: robots.txt and crawl delays)
>>> collect(start="2023-01-01")             # refreshes data/pe_deals/cache/pages.jsonl.gz
"""
import os as _os

_DIR = _os.path.dirname(_os.path.abspath(__file__))


def _source_stamp() -> tuple:
    """(path, mtime, size) of every .py file in the package."""
    out = []
    for root, dirs, files in _os.walk(_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                st = _os.stat(_os.path.join(root, f))
                out.append((_os.path.relpath(_os.path.join(root, f), _DIR), st.st_mtime_ns, st.st_size))
    return tuple(sorted(out))


_IMPORTED_STAMP = _source_stamp()


def source_changed() -> bool:
    """True when the package's files on disk are no longer the ones this process imported (see app.py)."""
    return _source_stamp() != _IMPORTED_STAMP


from .pipeline import build_dataset, DealDataset  # noqa: E402

__version__ = "1.0.0"
__all__ = ["build_dataset", "DealDataset"]
