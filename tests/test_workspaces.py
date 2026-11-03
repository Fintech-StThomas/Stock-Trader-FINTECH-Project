"""Exercise real app routing, including direct PE links, the PE sidebar and switching back."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def assert_pe_only(at):
    assert not at.exception
    assert any(t.value == "PE Deal Intelligence" for t in at.title)
    # PE navigation lives in the sidebar; the stock workspace's tabs and sidebar controls are absent.
    assert at.sidebar.radio(key="pe_page").options == [
        "🔎  Deal Explorer", "🗂️  Sources and Coverage", "✅  Quality and Method"]
    assert "📍 Analyse a stock" not in [t.label for t in at.tabs]
    assert not [w for w in at.sidebar.text_input if w.key == "ticker"]
    assert not at.sidebar.slider
    assert not at.sidebar.title


def test_switch_replaces_stock_workspace_and_can_return():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.sidebar.text_input(key="ticker")
    at.radio(key="workspace_choice").set_value("pe").run()
    assert_pe_only(at)
    assert at.query_params["workspace"] in ("pe", ["pe"])
    at.run()  # selection persists across reruns
    assert_pe_only(at)
    at.radio(key="workspace_choice").set_value("stock").run()
    assert not at.exception
    assert at.sidebar.text_input(key="ticker")
    assert "📍 Analyse a stock" in [t.label for t in at.tabs]
    assert not [r for r in at.sidebar.radio if r.key == "pe_page"]
    assert at.query_params["workspace"] in ("stock", ["stock"])
    assert "pe_page" not in at.query_params


def test_pe_deep_link_does_not_call_stock_analysis(monkeypatch):
    import lighthouse

    def forbidden(*args, **kwargs):
        pytest.fail("PE mode must not invoke stock analysis")

    monkeypatch.setattr(lighthouse, "analyze", forbidden)
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["workspace"] = "pe"
    at.run()
    assert_pe_only(at)


def test_unknown_workspace_defaults_to_stock():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["workspace"] = "unknown"
    at.run()
    assert not at.exception
    assert at.radio(key="workspace_choice").value == "stock"
    assert at.sidebar.text_input(key="ticker")


def test_pe_sidebar_switches_sections_and_deep_links():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["workspace"] = "pe"
    at.run()
    assert_pe_only(at)
    assert at.radio(key="pe_page").value == "explorer"
    assert at.sidebar.text_input(key="pe_explorer_search")
    assert not any("Demo dataset" in w.value for w in at.warning)
    assert any(c.value.startswith("Source: ") and "deal announcements" in c.value for c in at.caption)

    at.radio(key="pe_page").set_value("sources").run()
    assert_pe_only(at)
    assert at.query_params["pe_page"] in ("sources", ["sources"])
    assert any(s.value == "Know where the information comes from" for s in at.subheader)
    assert not [w for w in at.sidebar.text_input if w.key == "pe_explorer_search"]

    linked = AppTest.from_file(APP, default_timeout=60)
    linked.query_params["workspace"] = "pe"
    linked.query_params["pe_page"] = "quality"
    linked.run()
    assert_pe_only(linked)
    assert linked.radio(key="pe_page").value == "quality"


def test_pe_sidebar_filters_narrow_results_and_clear():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["workspace"] = "pe"
    at.run()
    total = int(at.metric[0].value)
    assert total > 1
    industry = at.sidebar.multiselect(key="pe_explorer_industries")
    industry.select(industry.options[0]).run()
    assert not at.exception
    narrowed = int(at.metric[0].value)
    assert 0 < narrowed < total
    assert any("1 filter active" in c.value for c in at.caption)

    at.sidebar.button(key="pe_explorer_clear").click().run()
    assert not at.exception
    assert int(at.metric[0].value) == total
    assert at.sidebar.multiselect(key="pe_explorer_industries").value == []


def test_pe_rationale_shortlist_add_and_remove():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["workspace"] = "pe"
    at.run()
    rationale = at.multiselect(key="pe_explorer_shortlist_rationales")
    rationale.select(rationale.options[0]).run()
    candidates = at.multiselect(key="pe_explorer_shortlist_add_selection")
    assert len(candidates.options) >= 2
    first, second = candidates.options[:2]  # display labels such as "Buyer → Target (deal_id)"
    candidates.select(first).select(second).run()
    first, second = (label.rsplit("(", 1)[1].rstrip(")") for label in (first, second))
    at.button(key="pe_explorer_shortlist_add").click().run()
    assert not at.exception
    assert at.session_state["pe_explorer_shortlist_ids"] == [first, second]
    assert any("Shortlist now contains 2 unique deal(s)." in s.value for s in at.success)

    remove = at.multiselect(key="pe_explorer_shortlist_remove_selection")
    remove.select(next(o for o in remove.options if o.endswith(f"({first})"))).run()
    at.button(key="pe_explorer_shortlist_remove").click().run()
    assert not at.exception
    assert at.session_state["pe_explorer_shortlist_ids"] == [second]
    assert at.session_state["pe_selected_deal_id"]  # handoff for the Deal Detail page
