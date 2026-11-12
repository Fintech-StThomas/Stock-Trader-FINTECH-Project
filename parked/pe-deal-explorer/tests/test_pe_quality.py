"""Quality and Method page: interval maths, the validation file's link to the shipped data, and rendering."""
import json
from pathlib import Path

import pytest

from pe.analytics import load_deals
from pe.pages.quality import flag_table, load_validation, unknown_table, wilson_interval

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("correct,total,lo,hi", [(26, 30, 70, 95), (30, 30, 89, 100), (38, 38, 91, 100), (29, 30, 83, 99)])
def test_wilson_interval_reproduces_the_validation_report(correct, total, lo, hi):
    low, high = wilson_interval(correct, total)
    assert (round(low * 100), round(high * 100)) == (lo, hi)


def test_validation_file_describes_the_shipped_dataset():
    validation = load_validation()
    deals = json.loads((ROOT / "data" / "pe" / "deals.json").read_text(encoding="utf-8"))
    assert validation["dataset"]["deals"] == len(deals), "re-run the validation for the new dataset"
    assert validation["labels"]["verified_by_human"] is False  # the page must keep saying so until a person checks
    for field in validation["fields"]:
        assert 0 < field["held_out_total"] and field["held_out_correct"] <= field["held_out_total"]
        assert field["dev_correct"] <= field["dev_total"]


def test_unknown_and_flag_tables_count_from_the_data():
    data = load_deals([
        {"deal_id": "a", "buyer_or_investor": None, "target_company": "T", "source_url": ["https://x"],
         "rationale_tags": [], "review_notes": "buyer not found; date from dateline", "review_status": "needs review"},
        {"deal_id": "b", "buyer_or_investor": "B", "target_company": "U", "source_url": ["https://y"],
         "rationale_tags": ["Geographic reach"], "review_notes": None, "review_status": "machine-extracted"},
    ])
    unknown = unknown_table(data).set_index("Field")
    assert unknown.loc["Buyer / investor", "Unknown"] == 1
    assert unknown.loc["Rationale", "Unknown"] == 1
    assert unknown.loc["Source link", "Unknown"] == 0
    assert unknown.loc["Disclosed value", "Share unknown"] == "100%"
    flags = flag_table(data).set_index("Flag")
    assert flags.loc["Buyer not named", "Deals"] == 1
    assert flags.loc["Date from the release's dateline", "Deals"] == 1


def test_archived_quality_renderer_is_still_available_for_team_review():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string("from pe.pages.quality import render_quality\nrender_quality()", default_timeout=60)
    at.run()
    assert not at.exception
    assert any(s.value == "Research you can check" for s in at.subheader)
    assert any("No person has checked them yet" in w.value for w in at.warning)
    assert not any("could not be read" in e.value for e in at.error)
    assert len(at.dataframe) >= 3
