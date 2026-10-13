import numpy as np
import pandas as pd
import pytest

from lighthouse.config import InvestorProfile, StrategyParams, verdict_from_score
from lighthouse.decision import composite_score, size_position, conviction_from_score, stop_level
from lighthouse.regime import regime_table, latest_regime, regime_adjustments
from lighthouse.backtest import run_backtest, compute_metrics, strategy_layers, sensitivity_grid, walk_forward, regime_breakdown


def test_profile_validation():
    with pytest.raises(ValueError):
        InvestorProfile(horizon="medium")
    with pytest.raises(ValueError):
        InvestorProfile(risk="yolo")
    p = InvestorProfile(horizon="Short", risk="Aggressive")
    assert p.technical_weight == 0.7 and p.max_position_pct == 35


def test_verdict_bands():
    assert verdict_from_score(80) == "Strong Buy"
    assert verdict_from_score(72) == "Strong Buy"
    assert verdict_from_score(65) == "Buy"
    assert verdict_from_score(50) == "Hold"
    assert verdict_from_score(40) == "Reduce"
    assert verdict_from_score(10) == "Sell"


def test_composite_blend_and_penalty():
    prof = InvestorProfile(horizon="long")
    assert composite_score(80, 60, prof) == pytest.approx(0.6 * 80 + 0.4 * 60)
    assert composite_score(None, 60, prof) == 60
    assert composite_score(80, 60, prof, regime_penalty=8) == pytest.approx(72 - 8)
    assert composite_score(0, 0, prof, regime_penalty=20) == 0


def test_position_sizing_respects_every_cap():
    prof = InvestorProfile(risk="balanced", max_loss_pct=1.0, portfolio_value=100_000)
    plan = size_position(price=100, atr=3, realized_vol=0.40, score=85, verdict="Strong Buy", profile=prof,
                         params=StrategyParams(atr_stop=True))
    assert 0 < plan.position_pct <= prof.max_position_pct
    # max-loss rule: position x stop distance <= max loss
    assert plan.position_value * plan.stop_pct <= prof.portfolio_value * prof.max_loss_pct / 100 + 1e-6
    assert plan.shares == int(plan.position_value // 100)
    hold = size_position(100, 3, 0.2, 50, "Hold", prof)
    assert hold.position_pct == 0
    sell = size_position(100, 3, 0.2, 10, "Sell", prof)
    assert sell.position_pct == 0 and sell.shares == 0


def test_stop_level_long_uses_sma_and_short_uses_atr():
    prof_l = InvestorProfile(horizon="long")
    lvl, label, kind = stop_level(100, 2, 90, prof_l, StrategyParams.for_horizon("long"))
    assert kind == "sma" and lvl == pytest.approx(90 * 0.92)
    prof_s = InvestorProfile(horizon="short", risk="balanced")
    lvl, label, kind = stop_level(100, 2, 90, prof_s, StrategyParams.for_horizon("short"))
    assert kind == "atr" and lvl == pytest.approx(100 - 2.5 * 2)


def test_conviction_scaling():
    assert conviction_from_score(45) == 0 and conviction_from_score(80) == 1
    assert 0 < conviction_from_score(60) < 1


def test_regime_table_and_adjustments(benchmark):
    rt = regime_table(benchmark)
    assert set(rt["regime"].dropna().unique()) <= {"Bull / Calm", "Bull / Turbulent", "Bear / Calm", "Bear / Turbulent"}
    snap = latest_regime(rt)
    assert snap is not None and snap.label in rt["regime"].dropna().unique()
    adj = regime_adjustments(True, True)
    assert adj["block_entries"] and adj["size_multiplier"] == 0.5 and adj["score_penalty"] == 11


def test_backtest_buy_hold_matches_price_ratio(prices, benchmark):
    res = run_backtest(prices, benchmark)
    assert res.buy_hold.iloc[-1] == pytest.approx(prices["Close"].loc[res.end] / prices["Close"].loc[res.start])
    assert res.equity.iloc[0] == pytest.approx(1.0)
    assert (res.exposure.between(0, 1 + 1e-9)).all()
    m = res.metrics
    assert set(["cagr", "sharpe", "max_drawdown", "n_trades", "win_rate", "exposure"]) <= set(m)
    assert -1 <= m["max_drawdown"] <= 0


def test_backtest_never_trades_before_signal_and_pays_costs(prices, benchmark):
    p = StrategyParams(cost_bps=50, slippage_bps=50)
    res = run_backtest(prices, benchmark, p, sizing="full")
    for t in res.trades:
        assert t.exit_date > t.entry_date
        assert t.holding_days >= 1
    # every trade must open strictly after the first date with a defined score
    assert all(t.entry_date > res.start for t in res.trades)
    cheap = run_backtest(prices, benchmark, StrategyParams(cost_bps=0, slippage_bps=0), sizing="full")
    if res.trades:
        assert res.equity.iloc[-1] < cheap.equity.iloc[-1]


def test_backtest_has_no_lookahead(prices, benchmark):
    """Decisions up to date t must be identical whether or not data after t exists."""
    full = run_backtest(prices, benchmark, sizing="full")
    cut_date = full.equity.index[-150]
    part = run_backtest(prices[prices.index <= cut_date], benchmark[benchmark.index <= cut_date], sizing="full")
    common = part.equity.index[:-1]        # last day of the truncated run force-closes the open trade
    assert np.allclose(full.equity.loc[common].values, part.equity.loc[common].values)


def test_regime_filter_blocks_bear_turbulent_entries(prices, benchmark):
    """No trade may be opened on the day after a Bear/Turbulent review day (entries are blocked there)."""
    p = StrategyParams(entry_score=0, exit_score=-1, trend_gate=False, exit_below_sma=None, decision_frequency="daily")
    res = run_backtest(prices, benchmark, p, sizing="full", use_regime=True)
    sig = res.signals
    blocked_days = sig[(sig["bull"] == 0) & (sig["turbulent"] == 1) & (sig["regime_known"] == 1)].index
    entries = {t.entry_date for t in res.trades}
    for d in blocked_days:
        i = sig.index.get_loc(d)
        if i + 1 < len(sig.index):
            assert sig.index[i + 1] not in entries


def test_evaluation_helpers_run(prices, benchmark):
    p = StrategyParams()
    prof = InvestorProfile()
    layers = strategy_layers(prices, benchmark, p, prof)
    assert "Buy & hold" in layers.index and "S&P 500" in layers.index
    grid = sensitivity_grid(prices, benchmark, p, prof, entry_values=(55, 60), exit_values=(35, 45))
    assert len(grid) == 4 and {"entry", "exit", "sharpe"} <= set(grid.columns)
    wf = walk_forward(prices, benchmark, p, prof, entry_values=(55, 60), exit_values=(35, 45))
    assert "Buy & hold" in wf["out_of_sample"].index
    rb = regime_breakdown(run_backtest(prices, benchmark, p, prof))
    assert "strategy_ann_return" in rb.columns


# --------------------------------------------------------------------------- risk-free rate (3-month T-bill)
def test_sharpe_uses_the_daily_t_bill_rate():
    idx = pd.bdate_range("2020-01-01", periods=600)
    r = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, len(idx)), index=idx)
    eq = (1 + r).cumprod()
    fixed = compute_metrics(eq, 0.02)
    flat = compute_metrics(eq, pd.Series(0.02, index=idx[::5]))           # published weekly, carried forward
    assert flat["sharpe"] == pytest.approx(fixed["sharpe"]) and flat["sortino"] == pytest.approx(fixed["sortino"])
    rates = pd.Series(np.where(idx < idx[300], 0.0, 0.05), index=idx)     # 0% for the first half, 5% after
    m = compute_metrics(eq, rates)
    rd = eq.pct_change().dropna()
    by_hand = (rd - rates.reindex(rd.index) / 252).mean() * 252 / (rd.std(ddof=0) * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(by_hand)
    early = pd.Series([0.03], index=[idx[100]])                            # dates before the first value take it
    assert compute_metrics(eq, early)["sharpe"] == pytest.approx(compute_metrics(eq, 0.03)["sharpe"])


def test_backtest_reports_its_risk_free_rate(prices, benchmark):
    fixed = run_backtest(prices, benchmark)
    tbill = pd.Series(0.045, index=prices.index[::7])
    bt = run_backtest(prices, benchmark, risk_free=tbill)
    assert [t.__dict__ for t in bt.trades] == [t.__dict__ for t in fixed.trades]     # no trade changes
    assert "fixed 2% a year" in fixed.options["risk_free"] and "FRED DTB3" in bt.options["risk_free"]
    assert bt.metrics["sharpe"] < fixed.metrics["sharpe"]                             # 4.5% is a higher bar than 2%
    assert bt.summary().loc["Buy & hold", "sharpe"] < fixed.summary().loc["Buy & hold", "sharpe"]
    layers = strategy_layers(prices, benchmark, StrategyParams(), InvestorProfile(), risk_free=tbill)
    assert layers.loc["Buy & hold", "sharpe"] == pytest.approx(bt.summary().loc["Buy & hold", "sharpe"])


def test_risk_free_loader_parses_fred_and_stays_offline(tmp_path, monkeypatch):
    from lighthouse import data
    s = data._parse_fred_csv("observation_date,DTB3\n2024-01-02,5.20\n2024-01-03,.\n2024-01-04,5.18\n")
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04"] and s.iloc[0] == pytest.approx(0.052)
    assert data.load_risk_free("auto") is None                       # LIGHTHOUSE_OFFLINE=1 in the tests
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    assert data.load_risk_free("sample") is None and data.load_risk_free("cache") is None
    s.to_csv(tmp_path / "riskfree_DTB3.csv", header=True)
    got = data.load_risk_free("cache")
    assert got is not None and got.iloc[-1] == pytest.approx(0.0518)
