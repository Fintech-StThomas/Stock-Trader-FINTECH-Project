"""Smoke tests of the Streamlit app (through Streamlit's AppTest harness, no browser) and of its interactive charts."""
import sys
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from lighthouse import analyze, InvestorProfile  # noqa: E402
from lighthouse import interactive as ix  # noqa: E402
from lighthouse.screener import screen  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_an_analysis_offline():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    at.sidebar.text_input(key="ticker").set_value("MSFT")
    at.sidebar.selectbox(key="source").set_value("sample")
    at.sidebar.button(key="analyse").click()
    at.run()
    assert not at.exception, at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "Lighthouse score" in text and "Position plan" in text
    assert any("Strong Buy" in str(m.value) or "Buy" in str(m.value) for m in at.markdown)
    # the analysis carries its analyst memo (from the bundled SEC filings), next to the verdict and outside the score
    assert "Analyst memo (fundamentals)" in text and "Investment memo" in text and "What would prove this wrong" in text
    assert "Fundamental beacons" in text                       # the memo is where the fundamental beacons live now
    assert "aria-label='Fundamental score'" not in text      # no fundamental score card next to the verdict
    labels = [t.label for t in at.tabs]          # every tab, the Analyse tab's own sub-tabs included, in page order
    assert labels[0] == "📍 Analyse a stock" and labels[1:6] == ["🎯 Plan & ticket", "🧑‍💼 Analyst memo", "📈 Technicals",
                                                                 "🧪 Backtest", "🗂️ Data"]
    assert {"🎯 Find trades", "🧺 Portfolio", "📓 Journal & theses", "ℹ️ How it works"} <= set(labels)
    assert not any("Research" in str(t.label) for t in at.tabs)   # research is evidence, not a tab


def test_app_without_filings_keeps_the_verdict_and_says_why_there_is_no_memo():
    """An index (or an ETF, or a company that does not file with the SEC) has no memo; the rule's verdict stands."""
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.text_input(key="ticker").set_value("^GSPC")
    at.sidebar.selectbox(key="source").set_value("sample")
    at.sidebar.button(key="analyse").click()
    at.run()
    assert not at.exception, at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "Lighthouse score" in text and "Analyst memo (fundamentals)" in text and "Investment memo" not in text
    assert any("No analyst memo for ^GSPC" in str(w.value) and "no SEC filings" in str(w.value) for w in at.warning)


def test_app_reports_a_bad_ticker_kindly():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.text_input(key="ticker").set_value("NOPE_NOT_A_TICKER")
    at.sidebar.selectbox(key="source").set_value("sample")
    at.sidebar.button(key="analyse").click()
    at.run()
    assert not at.exception, at.exception
    assert any("Could not load data" in str(e.value) for e in at.error)


def test_interactive_figures_build_for_a_full_analysis():
    r = analyze("MSFT", InvestorProfile(horizon="long", risk="balanced", max_loss_pct=2, portfolio_value=25_000), source="sample", years=6)
    r.evaluate()
    figs = ix.all_figures(r)
    for name in ("gauge", "waterfall", "breakdown", "dashboard", "backtest", "layers", "sensitivity", "walk_forward"):
        assert name in figs, name
        assert len(figs[name].data) >= 1, name
        figs[name].to_json()   # must serialise for the browser
    # the beacon chart carries its explanation in the hover data
    bar = figs["breakdown"].data[0]
    assert bar.customdata is not None and len(bar.customdata[0]) == 5
    # the dashboard shows the backtested trades on the price panel
    names = {t.name for t in figs["dashboard"].data}
    assert "Close" in names and "strategy entry" in names


def test_watchlist_figures_build():
    df = screen(["MSFT", "AAPL", "JNJ"], years=5, source="sample", progress=False)
    fig = ix.screener(df)
    assert len(fig.data) == 1 and len(fig.data[0].x) == 3
    r = analyze("MSFT", source="sample", years=5)
    cmp_ = ix.compare_prices({"MSFT": r.prices["Close"], "S&P 500": r.benchmark["Close"]})
    assert len(cmp_.data) == 2
    assert abs(cmp_.data[0].y[0] - 100) < 1e-9


def _offline_app():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    at.sidebar.selectbox(key="source").set_value("sample")
    at.sidebar.number_input(key="cash").set_value(5000)
    at.sidebar.number_input(key="holdings").set_value(20000)
    at.run()
    return at


def test_app_trade_desk_offline():
    at = _offline_app()
    at.button(key="desk_go").click()
    at.run()
    assert not at.exception, at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "BUY " in text and "SELL SHORT" not in text       # long tickets only
    assert "Ranked by Lighthouse score" in text               # the same scan ranks the universe ...
    assert any({"ticker", "verdict", "score", "technical"} <= set(getattr(df.value, "columns", [])) for df in at.dataframe)
    assert "Trade ideas funded from your cash" in text        # ... and funds its best setups
    assert "The plan" in text and "stop order" in text
    assert "When to get out" in text                          # every ticket says where to take the loss and the profit
    exits = [t.value for t in at.table if "Exit" in getattr(t.value, "columns", [])]
    trend = {"Stop-loss (stop order)", "Trend exit (review)", "Signal exit (review)"}
    assert exits and trend <= set(exits[0]["Exit"])            # long-term buys: the backtest's review-day exits
    assert not any("Position monitor" in str(t.label) for t in at.tabs)   # removed: the tickets carry the exits


def test_app_portfolio_builder_offline():
    at = _offline_app()
    at.button(key="pf_go").click()
    at.run()
    assert not at.exception, at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "Orders for your cash" in text and "Walk-forward test" in text
    at.radio(key="pf_mode").set_value("Build for a goal: Lighthouse chooses")
    at.run()
    at.button(key="pf_goal_go").click()
    at.run()
    assert not at.exception, at.exception
    assert any("Balanced risk and reward" in str(m.value) for m in at.markdown)


def test_trade_and_portfolio_figures_build():
    from lighthouse.tradeplan import TradeSettings, build_trade_plan
    from lighthouse import portfolio as pf
    r = analyze("MSFT", InvestorProfile.from_cash(5_000, 20_000), source="sample", years=6, run_backtest_=False)
    fig = ix.trade_chart(r, build_trade_plan(r, "long", with_stats=False))
    assert any(t.name == "Close" for t in fig.data) and len(fig.layout.shapes) == 2   # entry, stop ...
    assert any(t.name and t.name.startswith("trend exit") for t in fig.data)          # ... and the trend exit's path
    fig = ix.trade_chart(r, build_trade_plan(r, "long", TradeSettings(long_exit="targets"), with_stats=False))
    assert len(fig.layout.shapes) >= 3                                                # entry, stop, targets
    fig.to_json()
    port = pf.build_portfolio(["AAPL", "JNJ", "XOM", "KO", "PG"], "risk_parity", capital=10_000, source="sample", years=6)
    for f in (ix.allocation_chart(port.table()), ix.correlation_heatmap(port.risk["correlation"]), ix.equity_curves(port.comparison_runs)):
        assert len(f.data) >= 1
        f.to_json()


def test_app_saves_the_memo_thesis_to_the_journal_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_JOURNAL", str(tmp_path / "journal.sqlite"))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    at.sidebar.selectbox(key="source").set_value("sample")
    at.run()
    text = " ".join(str(m.value) for m in at.info)
    assert "No journal yet" in text or "No journal at" in text           # nothing saved: no file is created
    assert not (tmp_path / "journal.sqlite").exists()
    at.sidebar.text_input(key="ticker").set_value("MSFT")
    at.sidebar.button(key="analyse").click()
    at.run()
    assert not at.exception, at.exception
    md = " ".join(str(m.value) for m in at.markdown)
    assert "Investment memo" in md and "What would prove this wrong" in md
    at.button(key="memo_save").click()
    at.run()
    assert not at.exception, at.exception
    assert (tmp_path / "journal.sqlite").exists()
    assert any("Thesis 1 saved" in str(x.value) for x in at.success)


def test_app_journal_tab_shows_the_watch_list_as_watch_only(tmp_path, monkeypatch):
    """A journal with three scored month ends of a watch signal: the Journal tab shows its forward score, labelled
    watch-only (not a survivor, no weight)."""
    import numpy as np
    import pandas as pd
    from lighthouse.journal import Journal
    from lighthouse.research import watch as W
    path = tmp_path / "journal.sqlite"
    j = Journal(path)
    rng = np.random.default_rng(1)
    names = [f"T{k:02d}" for k in range(35)]
    for d in ("2026-07-31", "2026-08-31", "2026-09-30", "2026-10-30"):
        at = (pd.Timestamp(d) + pd.Timedelta(hours=22)).tz_localize("UTC").isoformat(timespec="seconds")
        j.append_watch_rows([{"recorded_at": at, "as_of": d, "signal": "fcf_yield", "ticker": t, "value": float(v),
                              "tradable": 1, "cost": 0.002, "config_hash": W.WATCH_SIGNALS["fcf_yield"].config_hash,
                              "data_version": "{}", "code_version": "test", "run_id": "r"}
                             for t, v in zip(names, rng.normal(size=len(names)))])
        if d != "2026-10-30":
            j.append_watch_outcomes([{"recorded_at": at, "as_of": d, "end_date": "x", "ticker": t, "ret_exclude": float(r),
                                      "ret_pessimistic": float(r), "data_version": "{}", "code_version": "test",
                                      "run_id": "r"} for t, r in zip(names, rng.normal(0.01, 0.05, len(names)))])
    monkeypatch.setenv("LIGHTHOUSE_JOURNAL", str(path))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    md = " ".join(str(m.value) for m in at.markdown)
    assert "Watch-only research signals (not survivors, no weight)" in md
    assert "**fcf_yield** (forward rule): collecting: 3 forward month(s)" in md
    shown = [df.value for df in at.dataframe if {"rule", "months", "ic_nw_t"} <= set(getattr(df.value, "columns", []))]
    assert len(shown) == 1 and set(shown[0]["months"]) == {3} and set(shown[0]["signal"]) == {"fcf_yield"}


def _lighthouse_modules() -> dict:
    return {n: m for n, m in sys.modules.items() if n == "lighthouse" or n.startswith("lighthouse.")}


def test_source_stamp_covers_the_subpackages():
    """A push that only changes lighthouse/analyst or lighthouse/research must also count as new code."""
    import lighthouse
    names = {name for name, _, _ in lighthouse._source_stamp()}
    assert {"engine.py", "analyst/memo.py", "research/harness.py"} <= {n.replace("\\", "/") for n in names}


def test_app_keeps_lighthouse_while_its_files_are_unchanged():
    import lighthouse
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    assert sys.modules["lighthouse"] is lighthouse          # no reload (and no cache wipe) on an ordinary rerun


@pytest.mark.parametrize("old_code", ["files changed", "imported before the check existed"])
def test_app_reloads_lighthouse_deployed_under_a_running_server(monkeypatch, old_code):
    """Streamlit Community Cloud pulls a push without restarting Python: the old modules must not outlive their files."""
    import lighthouse
    before = _lighthouse_modules()
    if old_code == "files changed":
        monkeypatch.setattr(lighthouse, "_IMPORTED_STAMP", ())
    else:
        monkeypatch.delattr(lighthouse, "source_changed")
    try:
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        assert not at.exception, at.exception
        assert sys.modules["lighthouse"] is not lighthouse
        assert not sys.modules["lighthouse"].source_changed()
    finally:   # put this process's modules back so later tests keep using one set of classes
        for n in _lighthouse_modules():
            del sys.modules[n]
        sys.modules.update(before)
