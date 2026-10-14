"""The research harness: statistics on a synthetic panel with a planted signal (the only way to know the right answer),
delisting rules, the hold-out guard, and causality of every signal on the sample pack's real prices."""
import numpy as np
import pandas as pd
import pytest

from lighthouse import membership as mb
from lighthouse.data import load_prices, sample_pack_tickers
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
    full = {**S.product_signals(df), **{k: v.fn(df) for k, v in S.PRICE_SIGNALS.items()}}
    cut = df.index[len(df) * 2 // 3]
    part_df = df[df.index <= cut]
    part = {**S.product_signals(part_df), **{k: v.fn(part_df) for k, v in S.PRICE_SIGNALS.items()}}
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
    # truncating the filings at a date leaves every earlier value unchanged
    facts = edgar.company_facts(ticker="AAPL", source="sample")
    cut = pd.Timestamp("2016-06-30")
    monkeypatch.setattr(edgar, "company_facts", lambda **kw: facts[facts["filed"] <= cut])
    part = FS.daily_values("AAPL", df, source="sample")
    for k in ("roic", "sue", "earnings_yield", "piotroski", "net_issuance", "accruals"):
        a, b = full[k][full[k].index <= cut], part[k][part[k].index <= cut]
        pd.testing.assert_series_equal(a, b, check_names=False, obj=k)


def test_residual_momentum_finds_a_planted_residual_drift(monkeypatch):
    from lighthouse.research import panel_signals as PS
    rng = np.random.default_rng(5)
    months = pd.period_range("2010-01", periods=72, freq="M")
    fac = pd.DataFrame({"Mkt-RF": rng.normal(0.008, 0.04, 72), "SMB": rng.normal(0, 0.02, 72),
                        "HML": rng.normal(0, 0.02, 72), "RF": 0.001}, index=months.to_timestamp("M"))
    monkeypatch.setattr(PS.ff, "load_factors", lambda freq="monthly": fac.copy())
    idx = pd.bdate_range("2010-01-01", "2015-12-31")
    me = H.month_ends(idx)
    prices = {}
    for k in range(20):
        # residual momentum is relative to the stock's own 36-month alpha: S00 beats its factor exposure only lately
        drift = np.where(np.arange(72) >= 58, 0.03, 0.0) if k == 0 else 0.0
        r = 0.001 + 1.0 * fac["Mkt-RF"].to_numpy() + drift + rng.normal(0, 0.01, 72)
        prices[f"S{k:02d}"] = pd.Series(100 * np.cumprod(1 + r), index=me[:72])
    panel = H.Panel(dates=me[:72], prices=pd.DataFrame(prices), members=pd.DataFrame(True, index=me[:72], columns=list(prices)),
                    fwd={}, last_price=pd.Series(dtype="datetime64[ns]"), data_end=me[71])
    rm = PS.residual_momentum(panel)
    last = rm.iloc[-1].dropna()
    assert last.idxmax() == "S00" and len(last) == 20


def test_coverage_counts_members_without_any_price():
    uni, frames = _universe_frames(n=40, months=24)
    iv = pd.concat([uni.intervals, pd.DataFrame({"ticker": [f"DEAD{k}" for k in range(10)], "start": uni.intervals["start"].iloc[0],
                                                 "end": pd.NaT})], ignore_index=True)
    panel = H.build_panel(mb.Universe("with unpriced members", iv), frames, start="2010-01-01")
    assert panel.coverage().iloc[3] == pytest.approx(40 / 50)        # 10 members never had a price
    res = H.evaluate(panel, panel.prices.pct_change(3), "mom3", "exclude", ("2010-01-01", None), min_names=20)
    assert res.summary["coverage"] <= 0.8 + 1e-9
