"""Exercise real app routing, including direct PE links and switching back."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def assert_pe_only(at):
    assert not at.exception
    assert any(t.value == "PE Deal Intelligence" for t in at.title)
    assert [t.label for t in at.tabs] == ["Deal Explorer", "Sources and Coverage", "Quality and Method"]
    assert not at.sidebar.text_input
    assert not at.sidebar.slider
    assert not at.sidebar.button
    assert not at.sidebar.title
    assert any("not yet connected" in i.value for i in at.info)


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
    assert "Deal Explorer" not in [t.label for t in at.tabs]
    assert at.query_params["workspace"] in ("stock", ["stock"])


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
