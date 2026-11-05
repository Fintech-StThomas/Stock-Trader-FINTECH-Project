"""Explicit publishing gate for a reviewed deal dataset.

Usage: python -m pe.publish data/pe/reviewed_deals.json --version pe-v1
Only records marked reviewed are accepted unless --allow-unreviewed is used for a
clearly labelled internal demo. Publishing is never triggered by a web request.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pe.repository import publish_dataset

parser = argparse.ArgumentParser(description="Publish a versioned PE deal dataset")
parser.add_argument("input", type=Path, help="JSON list of normalized deal records")
parser.add_argument("--version", help="version ID (defaults to a UTC timestamp)")
parser.add_argument("--allow-unreviewed", action="store_true", help="allow unreviewed records; published metadata will identify this")
parser.add_argument("--validation-results", type=Path, help="optional independently reviewed validation-results JSON")
args = parser.parse_args()
try:
    deals = json.loads(args.input.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Cannot read deal JSON: {exc}")
evidence = deals.get("evidence", []) if isinstance(deals, dict) else []
if isinstance(deals, dict): deals = deals.get("deals", [])
if not isinstance(deals, list): raise SystemExit("Input must be a JSON list or an object with a deals list")
unreviewed = [d for d in deals if str(d.get("review_status", "")).casefold() not in {"reviewed", "approved", "validated"}]
if unreviewed and not args.allow_unreviewed:
    raise SystemExit(f"Refusing to publish {len(unreviewed)} unreviewed record(s). Review them first or use --allow-unreviewed only for a labelled demo.")
metadata = {"is_demo": bool(args.allow_unreviewed), "unreviewed_records": len(unreviewed)}
if args.validation_results:
    try: metadata["validation"] = json.loads(args.validation_results.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise SystemExit(f"Cannot read validation results: {exc}")
result = publish_dataset(deals, version=args.version, metadata=metadata, evidence=evidence)
print(json.dumps(result, indent=2))
