"""The research harness: statistics on a synthetic panel with a planted signal (the only way to know the right answer),
delisting rules, the hold-out guard, and causality of every signal on the sample pack's real prices."""
import numpy as np
import pandas as pd
import pytest

from lighthouse import membership as mb
from lighthouse.data import load_prices
from lighthouse.research import harness as H
from lighthouse.research import signals as S


def _universe_frames(n=60, months=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=21 * months)
    frames = {}
    for k in range(n):
        r = rng.normal(0.0004, 0.015, len(idx))
        c = 50 * np.exp(np.cumsum(r))
        frames[f"S{k:02d}"] = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=idx)
    iv = pd.DataFrame({"ticker": list(frames), "start": idx[0], "end": pd.NaT})
    return mb.Universe("synthetic", iv), frames


def test_a_planted_signal_is_found_and_noise_is_not():
    uni, frames = _universe_frames()
    panel = H.build_panel(uni, frames, start="2010-01-01")
    fwd = panel.fwd["exclude"]
    rng = np.random.default_rng(1)
    planted = fwd + rng.normal(0, fwd.std().mean(), fwd.shape)            # knows next month's return (a cheat)
    noise = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    period = ("2010-01-01", None)
    good = H.evaluate(panel, planted, "planted", "exclude", period, min_names=20)
    bad = H.evaluate(panel, noise, "noise", "exclude", period, min_names=20)
    assert good.summary["ic_mean"] > 0.3 and good.summary["ic_nw_t"] > 5 and good.summary["monotonic"] > 0.8
    assert abs(bad.summary["ic_nw_t"]) < 3
    assert good.summary["spread_net_ann"] < good.summary["spread_gross_ann"]   # costs and borrow are charged
    assert 0 <= good.summary["top_turnover"] <= 1


def test_delisting_rules_in_the_panel():
    uni, frames = _universe_frames(n=40, months=24)
    dead = frames["S00"]
    frames["S00"] = dead.iloc[: 21 * 10 + 5]                     # stops trading early in month 11
    panel = H.build_panel(uni, frames, start="2010-01-01")
    k = panel.dates.searchsorted(frames["S00"].index[-1]) - 1
    d = panel.dates[k]
    assert panel.fwd["pessimistic"].loc[d, "S00"] == -1.0
    last = frames["S00"]["Close"].iloc[-1]
    assert panel.fwd["exclude"].loc[d, "S00"] == pytest.approx(last / panel.prices.loc[d, "S00"] - 1)
    assert panel.fwd["exclude"].iloc[k + 1:]["S00"].isna().all()      # gone afterwards
    assert (panel.coverage().iloc[k + 1:] < 1).all()                    # a member without prices lowers coverage


def test_statistics():
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 0.05, 400)
    m, t, nw = H.t_stats(x)
    assert t == pytest.approx(m / (x.std(ddof=1) / np.sqrt(400))) and abs(nw / t - 1) < 0.2   # iid: NW ~ plain
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = 0.002 + 0.5 * X["a"] - 0.3 * X["b"] + rng.normal(0, 0.01, 300)
    reg = H.ols_nw(pd.Series(y), X)
    assert reg["coef"]["a"] == pytest.approx(0.5, abs=0.01) and reg["coef"]["b"] == pytest.approx(-0.3, abs=0.01)
    one = H.deflated_sharpe(0.2, 120, 0.0, 3.0, 1, 0.1)
    many = H.deflated_sharpe(0.2, 120, 0.0, 3.0, 100, 0.1)
    assert one > 0.95 and many < one                                   # more trials, a higher bar


def test_holdout_is_evaluated_once(tmp_path):
    uni, frames = _universe_frames(n=40, months=150)                 # 2010 to 2022: crosses into the hold-out
    panel = H.build_panel(uni, frames, start="2010-01-01")
    log = H.ExperimentLog(tmp_path)
    sig = panel.prices.pct_change(3)                                  # 40 names: above the 30-name minimum
    res = H.evaluate_holdout(panel, sig, "mom3", {"signal": "mom3"}, log, {"v": 1})
    assert set(res) == {"exclude", "pessimistic"} and len(log.holdouts()) == 2
    with pytest.raises(H.HoldoutUsed):
        H.evaluate_holdout(panel, sig, "mom3", {"signal": "mom3"}, log, {"v": 1})
    H.evaluate_holdout(panel, sig, "mom3", {"signal": "mom3"}, log, {"v": 1}, allow_repeat=True)
    assert log.holdouts()["holdout_id"].nunique() == 2 and log.trials() == 1
    assert "deflated_sharpe" in res["exclude"].summary


@pytest.mark.parametrize("ticker", ["MSFT", "XOM"])
def test_every_price_signal_is_causal(ticker):
    df = load_prices(ticker, source="sample", years=None)
    df.attrs = {}
    full = S.product_signals(df)
    cut = df.index[len(df) * 2 // 3]
    part_df = df[df.index <= cut]
    part = S.product_signals(part_df)
    for k in full:
        a, b = full[k][full[k].index <= cut], part[k]
        pd.testing.assert_series_equal(a, b, check_names=False, obj=k)


def test_month_end_sampling_uses_the_last_value_on_or_before():
    idx = pd.bdate_range("2020-01-01", "2020-04-30")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    me = H.month_ends(idx)
    m = H.sample_at_month_ends({"A": s}, me)
    assert m.loc[me[0], "A"] == s.loc[:"2020-01-31"].iloc[-1]


def test_fundamental_signals_on_real_filings_and_causal(monkeypatch):
    from lighthouse import edgar
    from lighthouse.research import fundamental_signals as FS
    df = load_prices("AAPL", source="sample", years=None)
    df.attrs = {}
    full = FS.daily_values("AAPL", df, source="sample")
    assert set(FS.FUNDAMENTAL_SIGNALS) <= set(full)
    d = pd.Timestamp("2015-01-02")
    roic, ey = full["roic"].loc[d], full["earnings_yield"].loc[d]
    assert 0.2 < roic < 0.6 and 0.05 < ey < 0.1                      # AAPL early 2015: ROIC ~ 30-40%, P/E ~ 15
    assert full["sue"].dropna().between(-50, 50).all()
    # a split is not an issue of shares: AAPL's 4-for-1 split in August 2020 leaves net issuance near zero
    assert abs(full["net_issuance"].loc["2021-03-01"]) < 0.1
    # nor a change in market value: the cover page of 30 July 2020 counts pre-split shares, the price after 31 August
    # is post-split; the count is restated, so the value moves only with the price (+3.4% that day) and agrees with
    # the memo's (analyst.financials) and membership's market cap
    mc = full["market_cap"]
    assert mc.loc["2020-08-31"] / mc.loc["2020-08-28"] == pytest.approx(df["Close"].loc["2020-08-31"] / df["Close"].loc["2020-08-28"],
                                                                        rel=1e-9)
    assert 1.8e12 < mc.loc["2020-09-15"] < 2.2e12
    assert abs(full["net_issuance"].loc["2020-10-15"]) < 0.1
    # truncating the filings at a date leaves every earlier value unchanged
    facts = edgar.company_facts(ticker="AAPL", source="sample")
    cut = pd.Timestamp("2016-06-30")
    monkeypatch.setattr(edgar, "company_facts", lambda **kw: facts[facts["filed"] <= cut])
    part = FS.daily_values("AAPL", df, source="sample")
    for k in ("roic", "sue", "earnings_yield", "piotroski", "net_issuance", "accruals"):
        a, b = full[k][full[k].index <= cut], part[k][part[k].index <= cut]
        pd.testing.assert_series_equal(a, b, check_names=False, obj=k)


def test_coverage_counts_members_without_any_price():
    uni, frames = _universe_frames(n=40, months=24)
    iv = pd.concat([uni.intervals, pd.DataFrame({"ticker": [f"DEAD{k}" for k in range(10)], "start": uni.intervals["start"].iloc[0],
                                                 "end": pd.NaT})], ignore_index=True)
    panel = H.build_panel(mb.Universe("with unpriced members", iv), frames, start="2010-01-01")
    assert panel.coverage().iloc[3] == pytest.approx(40 / 50)        # 10 members never had a price
    res = H.evaluate(panel, panel.prices.pct_change(3), "mom3", "exclude", ("2010-01-01", None), min_names=20)
    assert res.summary["coverage"] <= 0.8 + 1e-9


def test_series_without_volume_are_not_tested_and_costs_are_capped():
    """A stale feed or another company's data under a reused ticker shows as days without volume (CPWR in the Yahoo
    cache: zero volume and a $2 price for four years while it was an S&P 500 member)."""
    uni, frames = _universe_frames(n=40, months=24)
    stale = frames["S01"].copy()
    stale.loc[stale.index[21 * 6]:, "Volume"] = 0.0                     # no trades from month 7 on
    frames["S01"] = stale
    panel = H.build_panel(uni, frames, start="2010-01-01")
    assert panel.tradable.loc[panel.dates[3], "S01"] and not panel.tradable.loc[panel.dates[8], "S01"]
    assert not panel.investable.loc[panel.dates[8], "S01"]
    assert panel.coverage().iloc[8] == pytest.approx(39 / 40)
    sig = panel.prices.pct_change(3)
    res = H.evaluate(panel, sig, "mom3", "exclude", ("2010-01-01", None), min_names=20)
    assert res.summary["coverage"] < 1.0
    costs = H.cost_panel(frames, panel.dates)
    assert float(costs.max().max()) <= H.MAX_COST
    flat = H.cost_panel(frames, panel.dates, fixed_bps=10.0)
    assert np.allclose(flat.to_numpy(), 0.001)


def test_experiment_log_columns_stay_aligned(tmp_path):
    """Tuning rows and hold-out rows (which carry the trial count and deflated Sharpe ratios) share one fixed header;
    a statistic added later lands in the `extra` column instead of shifting every field after it."""
    log = H.ExperimentLog(tmp_path)
    base = {"signal": "x", "rule": "exclude", "ic_mean": 0.01}
    log.append(base, {"signal": "x"}, {"v": 1}, rule="exclude", period="tuning")
    log.append({**base, "trials": 3, "deflated_sharpe": 0.5, "long_only_deflated_sharpe": 0.4},
               {"signal": "x"}, {"v": 1}, holdout=True, rule="exclude", period="holdout", holdout_id="h1")
    log.append({**base, "brand_new_stat": 7.0}, {"signal": "x"}, {"v": 1}, rule="exclude", period="tuning")
    r = log.rows()
    assert list(r.columns) == H.LOG_COLUMNS and len(r) == 3
    assert r["ic_mean"].tolist() == [0.01, 0.01, 0.01]
    assert r.loc[1, "deflated_sharpe"] == 0.5 and pd.isna(r.loc[0, "deflated_sharpe"])
    assert '"brand_new_stat": 7.0' in r.loc[2, "extra"]
    assert len(log.holdouts()) == 1 and log.trials() == 1
    with pytest.raises(H.HoldoutUsed):
        log.check_holdout("x", {"signal": "x"})
    (tmp_path / "experiments.csv").write_text("other,columns\n1,2\n")
    with pytest.raises(RuntimeError, match="misalign"):
        log.append(base, {"signal": "x"}, {"v": 1})


def test_trial_sharpe_dispersion_comes_from_every_logged_configuration(tmp_path):
    """The deflated Sharpe ratio's variance of trials: a run that evaluates one signal is judged against every
    configuration in the log (latest tuning row of each configuration and rule), not against itself."""
    log = H.ExperimentLog(tmp_path)
    assert log.trial_sharpe_std(default=0.1) == 0.1                      # nothing logged yet
    sharpes = {"a": (0.6, 0.3), "b": (-1.2, -0.4), "c": (0.1, 0.9)}
    for name, (ls, lo) in sharpes.items():
        for rule in ("exclude", "pessimistic"):
            log.append({"signal": name, "rule": rule, "spread_net_sharpe": ls, "long_only_net_sharpe": lo},
                       {"signal": name}, {"v": 1}, rule=rule, period="tuning")
    # a re-run of "a" with corrected data replaces its earlier rows; a hold-out row never counts
    for rule in ("exclude", "pessimistic"):
        log.append({"signal": "a", "rule": rule, "spread_net_sharpe": 2.0, "long_only_net_sharpe": 1.0},
                   {"signal": "a"}, {"v": 2}, rule=rule, period="tuning")
    log.append({"signal": "c", "rule": "exclude", "spread_net_sharpe": 50.0, "long_only_net_sharpe": 50.0,
                "trials": 3}, {"signal": "c"}, {"v": 1}, holdout=True, rule="exclude", period="holdout",
               holdout_id="h")
    want = pd.Series([2.0, 1.0, -1.2, -0.4, 0.1, 0.9] * 2) / np.sqrt(12)
    assert log.trial_sharpe_std() == pytest.approx(want.std(ddof=1))
