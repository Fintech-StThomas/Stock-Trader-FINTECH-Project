import pandas as pd
import pytest

from lighthouse import analyze, screen, InvestorProfile
from lighthouse.data import load_prices, clean_prices, sample_pack_tickers, DataUnavailable
from lighthouse.report import build_report


def test_clean_prices_reports_changes():
    idx = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    df = pd.DataFrame({"Open": [10, 10, None, 12, 13], "High": [9, 9, 11, 12, 14], "Low": [8, 8, 10, 11, 12],
                       "Close": [10, 10.5, 11, None, 13], "Volume": [1, 1, 1, 1, 1]}, index=idx)
    clean, rep = clean_prices(df, "T", "test")
    assert rep.duplicates_removed == 1 and rep.rows_dropped == 1
    assert (clean["High"] >= clean[["Open", "Close"]].max(axis=1)).all()
    assert clean.index.is_monotonic_increasing and clean.index.is_unique


def test_sample_pack_available():
    assert "MSFT" in sample_pack_tickers() and "^GSPC" in sample_pack_tickers()
    df = load_prices("MSFT", years=5, source="sample")
    assert df.attrs["source"] == "sample" and len(df) > 1000
    assert df.index[-1] - df.index[0] < pd.Timedelta(days=365.25 * 6.5)   # `years` window applied to offline data
    with pytest.raises(DataUnavailable):
        load_prices("NOPE_TICKER", source="sample")


def test_end_to_end_analysis_and_report(tmp_path):
    prof = InvestorProfile(horizon="long", risk="conservative", max_loss_pct=1.5, portfolio_value=50_000)
    r = analyze("MSFT", prof, years=8, source="sample")
    d = r.decision
    assert d.verdict in {"Strong Buy", "Buy", "Hold", "Reduce", "Sell"}
    assert 0 <= d.composite_score <= 100
    assert d.composite_score == pytest.approx(max(0.0, min(100.0, d.technical_score - d.regime_penalty)))
    assert all(b.group == "technical" for b in d.beacons)
    assert d.plan.position_pct <= prof.max_position_pct
    assert r.backtest is not None and r.backtest.metrics["n_trades"] >= 1
    assert "Same exposure, never traded" in r.backtest.summary().index
    md = build_report(r)
    for needle in ["Verdict", "Technical beacons", "Market weather", "Position plan", "Same exposure, never traded",
                   "analyst memo", "Limitations"]:
        assert needle in md
    r.evaluate(layers=True, sensitivity=False, walk_forward_test=False)
    files = r.save(tmp_path, charts=True)
    assert any(f.endswith("_report.md") for f in files) and any(f.endswith("_charts.html") for f in files)


def test_short_horizon_profile_changes_rule():
    r = analyze("MSFT", InvestorProfile(horizon="short"), years=6, source="sample")
    assert r.params.atr_stop and r.params.decision_frequency == "weekly"


def test_as_of_time_travel():
    r = analyze("MSFT", years=6, source="sample", as_of="2009-03-09")
    assert r.decision.as_of == "2009-03-09"
    assert r.decision.regime is not None and r.decision.regime.is_bear


def test_screener_handles_failures():
    df = screen(["MSFT", "AAPL", "BADTICKER"], years=4, source="sample", progress=False)
    assert list(df["ticker"][:2]) != ["BADTICKER", "BADTICKER"]
    assert df.loc[df["ticker"] == "BADTICKER", "error"].iloc[0] != ""
    assert df["score"].dropna().is_monotonic_decreasing


def test_the_rule_reads_prices_only(monkeypatch):
    """The verdict, the ticket and the backtest need no fundamentals: an analysis never opens the SEC filings."""
    from lighthouse import edgar, pit

    def refuse(*a, **k):
        raise AssertionError("the rule must not read SEC filings")
    monkeypatch.setattr(pit, "load_statements", refuse)
    monkeypatch.setattr(edgar, "company_facts", refuse)
    r = analyze("AAPL", years=8, source="sample")
    assert r.decision.verdict and r.backtest is not None
