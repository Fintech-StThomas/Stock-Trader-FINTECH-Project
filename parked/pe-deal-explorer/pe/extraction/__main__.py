"""Run the offline rule-based extractor against saved announcement records."""
import argparse
import json
from pathlib import Path
from pe.extraction.rules import extract

from pe.storage import DEFAULT_PRIVATE
root = DEFAULT_PRIVATE / "research"
parser = argparse.ArgumentParser(description="Extract candidate PE deals from saved announcements")
parser.add_argument("--input", type=Path, help="announcement JSON file; defaults to the latest collected file")
parser.add_argument("--output", type=Path, default=root / "extraction_output.json")
args = parser.parse_args()
if args.input:
    source = args.input
else:
    files = sorted(root.glob("announcements-*.json"))
    if not files:
        raise SystemExit("No saved announcements found. Run the collector or pass --input data/pe/demo_announcements.json.")
    source = files[-1]
try:
    announcements = json.loads(source.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Cannot read announcements: {exc}")
if not isinstance(announcements, list):
    raise SystemExit("Announcement input must be a JSON list")
result = extract(announcements)
result["input_file"] = str(source)
result["is_demo"] = any(a.get("data_status") == "demo" or a.get("collection_status") == "demo" for a in announcements if isinstance(a, dict))
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {args.output}: {len(result['deals'])} candidate deals, {len(result['review_queue'])} review items; demo={result['is_demo']}.")
