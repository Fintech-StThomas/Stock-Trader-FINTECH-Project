"""Portfolio estimators, optimisers, constraints, risk report, share allocation and the walk-forward test."""
import numpy as np
import pandas as pd
import pytest

from lighthouse.portfolio import (METHODS, GOALS, PortfolioConstraints, ledoit_wolf, bayes_stein_means, estimate, optimize,
                                  risk_contributions, to_shares, vol_target_scale, backtest_weights, load_universe,
                                  build_portfolio, goal_portfolio, selection_panels, select, risk_report, _hrp)
from lighthouse.universe import available_universe, SECTORS, LARGE_CAP_UNIVERSE

BASKET = ["AAPL", "JNJ", "XOM", "JPM", "KO", "PG", "UNH", "WMT", "HD", "CVX", "MRK", "PEP"]


@pytest.fixture(scope="module")
def uni():
    return load_universe(BASKET, years=10, source="sample")


@pytest.fixture(scope="module")
def window(uni):
    return uni.returns.iloc[-252:]


def _rand_returns(t=300, n=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0004, 0.01, (t, n)) @ (np.eye(n) + 0.3 * rng.normal(size=(n, n)))


def test_ledoit_wolf_is_a_valid_shrunk_covariance():
    x = _rand_returns()
    cov, shrink = ledoit_wolf(x)
    assert 0 <= shrink <= 1 and np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    sample = np.cov(x, rowvar=False, ddof=0)
    target = np.trace(sample) / x.shape[1] * np.eye(x.shape[1])
    assert np.allclose(cov, shrink * target + (1 - shrink) * sample)


def test_ledoit_wolf_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.covariance")
    x = _rand_returns(seed=3)
    ours, s1 = ledoit_wolf(x)
    theirs, s2 = sk.ledoit_wolf(x)
    assert np.allclose(ours, theirs) and s1 == pytest.approx(s2)


def test_bayes_stein_shrinks_toward_the_min_variance_mean():
    x = _rand_returns(seed=5)
    cov, _ = ledoit_wolf(x)
    mu_bs, phi = bayes_stein_means(x, cov)
    mu = x.mean(axis=0)
    inv = np.linalg.inv(cov)
    one = np.ones(len(mu))
    mu0 = one @ inv @ mu / (one @ inv @ one)
    assert 0 < phi <= 1
    assert np.allclose(mu_bs, (1 - phi) * mu + phi * mu0)


@pytest.mark.parametrize("method", list(METHODS))
def test_every_method_is_fully_invested_and_respects_caps(window, method):
    cons = PortfolioConstraints(max_weight=0.25)
    w, _ = optimize(window, method, cons)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all() and (w <= 0.25 + 1e-4).all()


def test_method_properties(window):
    est = estimate(window)
    free = PortfolioConstraints(max_weight=1.0)
    w_rp, _ = optimize(window, "risk_parity", free, est=est)
    assert np.allclose(risk_contributions(w_rp.values, est.cov), 1 / len(w_rp), atol=1e-4)     # equal risk contributions
    w_mv, _ = optimize(window, "min_variance", free, est=est)
    w_ew, _ = optimize(window, "equal_weight", free, est=est)
    assert w_mv.values @ est.cov @ w_mv.values <= w_ew.values @ est.cov @ w_ew.values
    w_iv, _ = optimize(window, "inverse_vol", free, est=est)
    assert np.allclose(w_iv.values * est.vol, (w_iv.values * est.vol)[0])                    # weight x vol is constant
    w_ms, _ = optimize(window, "max_sharpe", free, risk_free_rate=0.02, est=est)
    sr = lambda w: (w @ est.mu_bs - 0.02) / np.sqrt(w @ est.cov @ w)                          # noqa: E731
    assert sr(w_ms.values) >= max(sr(w_ew.values), sr(w_mv.values), sr(w_rp.values)) - 1e-9


def test_hrp_on_two_uncorrelated_identical_clusters():
    block = np.full((2, 2), 0.5) + 0.5 * np.eye(2)
    cov = np.block([[block, np.zeros((2, 2))], [np.zeros((2, 2)), block]]) * 0.04
    assert np.allclose(_hrp(cov), 0.25)


def test_infeasible_caps_are_reported(window):
    with pytest.raises(ValueError, match="cannot add up"):
        optimize(window.iloc[:, :3], "equal_weight", PortfolioConstraints(max_weight=0.25))
    sectors = ["Energy"] * 12
    with pytest.raises(ValueError, match="sector cap"):
        optimize(window, "min_variance", PortfolioConstraints(max_weight=0.25, max_sector_weight=0.5), sectors)


def test_sector_cap_is_respected(uni, window):
    sectors = [uni.sectors[t] for t in window.columns]
    w, _ = optimize(window, "min_variance", PortfolioConstraints(max_weight=0.25, max_sector_weight=0.30), sectors)
    assert pd.Series(w.values, index=sectors).groupby(level=0).sum().max() <= 0.30 + 1e-4


def test_to_shares_never_overspends_and_keeps_the_cash_buffer():
    w = pd.Series({"A": 0.4, "B": 0.3, "C": 0.1})            # 20% deliberate cash buffer
    px = pd.Series({"A": 95.0, "B": 33.0, "C": 410.0})
    table, left = to_shares(w, px, 10_000)
    assert table["value"].sum() + left == pytest.approx(10_000)
    assert table["value"].sum() <= 0.8 * 10_000 + 1e-9
    assert (table["shares"] == np.floor(table["shares"])).all()
    assert (abs(table["value"] - w * 10_000) <= px + 1e-9).all()     # within one share of the target


def test_volatility_target_holds_cash(window):
    est = estimate(window)
    w = np.full(window.shape[1], 1 / window.shape[1])
    k = vol_target_scale(w, est.cov, 0.08)
    assert 0 < k < 1 and np.sqrt((w * k) @ est.cov @ (w * k)) == pytest.approx(0.08)
    assert vol_target_scale(w, est.cov, None) == 1.0


def test_risk_report_scales_with_the_cash_buffer(uni, window):
    w = pd.Series(1 / window.shape[1], index=window.columns)
    full = risk_report(w, window)
    half = risk_report(w * 0.5, window)
    assert half["expected_vol"] == pytest.approx(0.5 * full["expected_vol"])
    assert half["var95_1d"] == pytest.approx(0.5 * full["var95_1d"])
    assert half["cash_weight"] == pytest.approx(0.5)
    assert full["risk_contributions"].sum() == pytest.approx(1.0)
    assert full["es95_1d"] >= full["var95_1d"] > 0


def test_walk_forward_has_no_lookahead(uni):
    rets = uni.returns
    seen = []

    def fn(t):
        seen.append(t)
        w, _ = optimize(rets.iloc[t - 251: t + 1], "min_variance", PortfolioConstraints(max_weight=0.3))
        return w.values, 1.0
    full = backtest_weights(rets, fn, 252, "monthly", 10.0)
    cut = full.equity.index[len(full.equity) // 2]
    part = backtest_weights(rets[rets.index <= cut], fn, 252, "monthly", 10.0)
    common = part.equity.index[:-1]
    assert np.allclose(full.equity.loc[common].values, part.equity.loc[common].values)
    assert full.equity.iloc[0] == pytest.approx(1.0)


def test_walk_forward_single_asset_tracks_its_price(uni):
    rets = uni.returns[["KO"]]
    bt = backtest_weights(rets, lambda t: (np.array([1.0]), 1.0), 252, "monthly", 0.0)
    px = uni.closes["KO"].reindex(bt.equity.index)
    # invested from the close of the day after the first decision
    start = bt.equity.index[1]
    ratio = bt.equity / bt.equity.loc[start]
    assert np.allclose(ratio.loc[start:].values, (px / px.loc[start]).loc[start:].values)


def test_load_universe_drops_stale_series():
    u = load_universe(["MSFT", "AAPL", "KO"], years=8, source="sample")
    assert "MSFT" in u.excluded and set(u.tickers) == {"AAPL", "KO"}


def test_build_portfolio_end_to_end(uni):
    r = build_portfolio(BASKET, "risk_parity", PortfolioConstraints(max_weight=0.2, target_vol=0.12), capital=25_000,
                        source="sample", universe=uni)
    assert r.allocation["value"].sum() + r.cash_left == pytest.approx(25_000)
    assert r.risk["expected_vol"] <= 0.12 + 1e-6 and r.cash_weight > 0
    assert set(r.comparison.index) >= {METHODS[m] for m in METHODS} | {"S&P 500 (buy & hold)"}
    assert {"var95_1d_usd", "beta", "stress", "correlation"} <= set(r.risk)


def test_goal_portfolio_selects_by_its_rules():
    uni_all = load_universe(available_universe("sample"), years=10, source="sample")
    panels = selection_panels(uni_all)
    t = len(uni_all.closes) - 1
    for key in ("growth", "balanced"):
        g = GOALS[key]
        picked = select(panels, t, g, uni_all.sectors, max_per_sector=3)
        assert 2 <= len(picked) <= g.n_stocks
        assert pd.Series([uni_all.sectors[x] for x in picked]).value_counts().max() <= 3
        if g.require_uptrend:
            assert (panels["uptrend"].iloc[t][picked] == 1.0).all()
        if g.min_technical is not None:
            assert (panels["technical"].iloc[t][picked] >= g.min_technical).all()
    r = goal_portfolio("balanced", capital=10_000, source="sample", universe=uni_all)
    assert r.backtest is not None and "Equal weight, whole universe" in r.comparison.index
    assert r.weights.sum() <= 1 + 1e-9 and list(r.selection.index) == list(r.weights.index)


def test_universe_labels_cover_every_ticker():
    assert all(t in SECTORS for t in LARGE_CAP_UNIVERSE)
    assert all(t in SECTORS for t in available_universe("sample"))


def test_goal_keeps_holdings_while_they_qualify():
    from lighthouse.portfolio import Goal
    uni_all = load_universe(available_universe("sample"), years=10, source="sample")
    panels = selection_panels(uni_all)
    t = len(uni_all.closes) - 1
    g = Goal(**{**GOALS["balanced"].__dict__, "n_stocks": 4})
    fresh = select(panels, t, g, uni_all.sectors)
    tech = panels["technical"].iloc[t]
    ok = tech[tech >= g.min_technical].sort_values()
    low, failing = ok.index[0], tech[tech < g.min_technical].index
    assert low not in fresh                                        # too weak to be picked from scratch ...
    kept = select(panels, t, g, uni_all.sectors, held=[low])
    assert kept[0] == low and set(kept[1:]) <= set(fresh)          # ... but kept while it still qualifies
    if len(failing):
        assert failing[0] not in select(panels, t, g, uni_all.sectors, held=[failing[0]])   # sold once it fails the filter
    narrow = Goal(**{**g.__dict__, "keep_rank": 1.0})              # a buffer as wide as the goal: only the top n stay
    assert select(panels, t, narrow, uni_all.sectors, held=[low]) == fresh
    assert GOALS["preservation"].keep_rank is None                 # preservation re-picks from scratch
    pres = GOALS["preservation"]
    assert select(panels, t, pres, uni_all.sectors, held=[low]) == select(panels, t, pres, uni_all.sectors)

    r = goal_portfolio("balanced", capital=10_000, source="sample", universe=uni_all, held=[low, "ZZZZ"])
    assert low in r.weights.index and any(n.startswith("Kept 1 of the 2 stocks you hold") for n in r.notes)
    rows = list(r.comparison.index)
    kept_row = next(x for x in rows if "holdings kept while they qualify" in x)
    scratch_row = next(x for x in rows if "re-selected from scratch" in x)
    assert r.comparison.loc[kept_row, "turnover_annual"] < r.comparison.loc[scratch_row, "turnover_annual"]


def test_walk_forward_passes_the_drifted_holdings(uni):
    rets = uni.returns[["KO", "PG"]]
    seen = []

    def fn(t, w_now):
        seen.append(w_now)
        return np.array([0.5, 0.5]), 1.0
    backtest_weights(rets, fn, 252, "monthly", 0.0, pass_holdings=True)
    assert np.allclose(seen[0], 0.0) and seen[1].sum() == pytest.approx(1.0) and not np.allclose(seen[1], 0.5)
