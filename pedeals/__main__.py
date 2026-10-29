"""
Command line for PE Deal Intelligence.

    python -m pedeals collect [--start 2023-01-01] [--firm tonkabay ...] [--refresh] [--max-pages N]
    python -m pedeals build   [--out data/pe_deals/output]      # parse + extract the saved pages, export the dataset
    python -m pedeals validate                                  # accuracy against the reference labels, consistency checks
    python -m pedeals new-since 2026-09-01                      # deals published since a date (monitoring)
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pedeals", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="fetch the firms' news pages (polite) into the saved cache")
    c.add_argument("--start", default=None, help="first publication date of interest (default 2023-01-01)")
    c.add_argument("--firm", action="append", help="firm key (repeatable); default: every firm")
    c.add_argument("--refresh", action="store_true", help="re-fetch pages already saved")
    c.add_argument("--max-pages", type=int, default=None, help="at most this many page requests per firm")
    b = sub.add_parser("build", help="parse and extract the saved pages; export CSV / JSON / Markdown")
    b.add_argument("--out", default=None)
    b.add_argument("--start", default=None)
    sub.add_parser("validate", help="score the extraction against the reference labels")
    n = sub.add_parser("new-since", help="deal announcements published on or after a date")
    n.add_argument("date")
    a = ap.parse_args(argv)

    from .collect import DEFAULT_START
    if a.cmd == "collect":
        from .collect import collect
        disc = collect(start=a.start or DEFAULT_START, firms=a.firm, refresh=a.refresh, max_pages=a.max_pages)
        print(disc.groupby(["source", "outcome"]).size().to_string())
        return 0
    from .pipeline import build_dataset
    ds = build_dataset(start=getattr(a, "start", None) or DEFAULT_START)
    if a.cmd == "build":
        paths = ds.export(a.out)
        print(ds.summary_text())
        print("\n".join(f"wrote {p}" for p in paths))
    elif a.cmd == "validate":
        from .validate import full_report, RESULTS_FILE
        text = full_report(ds)
        RESULTS_FILE.write_text(text)
        print(text)
        print(f"wrote {RESULTS_FILE}")
    elif a.cmd == "new-since":
        d = ds.deals[ds.deals["published_date"] >= a.date]
        cols = ["published_date", "publisher", "buyer", "target", "transaction_type", "target_industry", "url"]
        print(d[cols].to_string(index=False) if len(d) else f"no deal announcements published since {a.date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
