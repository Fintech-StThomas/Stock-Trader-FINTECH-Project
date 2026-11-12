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
        "🔎  AI Deal Research", "🧾  Deal Detail"]
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
    assert at.text_area(key="pe_ai_question")
    assert not any("Demo dataset" in w.value for w in at.warning)
    assert not at.sidebar.multiselect and not at.sidebar.date_input

    at.radio(key="pe_page").set_value("detail").run()
    assert_pe_only(at)
    assert at.query_params["pe_page"] in ("detail", ["detail"])
    assert at.selectbox(key="pe_detail_choice")

    linked = AppTest.from_file(APP, default_timeout=60)
    linked.query_params["workspace"] = "pe"
    linked.query_params["pe_page"] = "sources"
    linked.run()
    assert_pe_only(linked)
    assert linked.radio(key="pe_page").value == "explorer"
    assert linked.query_params["pe_page"] in ("explorer", ["explorer"])


def test_ai_sidebar_replaces_manual_filters():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["workspace"] = "pe"
    at.run()
    assert_pe_only(at)
    assert not at.sidebar.text_input and len(at.text_area) == 1
    assert not at.sidebar.multiselect and not at.sidebar.date_input
    assert any("Research private equity deals" in x.value for x in at.markdown)


def test_pe_rationale_shortlist_add_and_remove():
    at = AppTest.from_string("from pe.pages.explorer import render_legacy_explorer\nrender_legacy_explorer()", default_timeout=60).run()
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


def test_inspect_saved_deal_callback_navigates_without_widget_error():
    from pe.repository import load_dataset
    from pe.agent import FIELDS, ground_results
    records=load_dataset()['records']
    selected=records[0]['deal_id']
    payload={'interpretation':'A saved deal','limitations':[], 'results':[{'saved_deal_id':selected,**{f:None for f in FIELDS},'evidence':[]}]}
    result=ground_results(payload,records,set())
    result.update(query='Show a saved deal',model='test',web_search_enabled=False)
    at=AppTest.from_file(APP,default_timeout=60)
    at.query_params['workspace']='pe'
    at.session_state['pe_ai_result']=result
    at.run()
    assert not at.exception
    at.button(key='pe_ai_detail_'+selected).click().run()
    assert not at.exception
    assert at.radio(key='pe_page').value=='detail'
    assert at.session_state['pe_selected_deal_id']==selected
    assert at.selectbox(key='pe_detail_choice').value==selected
