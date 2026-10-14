"""Dividends, splits, total returns and the delisting rules."""
import numpy as np
import pandas as pd
import pytest

from lighthouse import total_return as tr


def _actions(rows):
    df = pd.DataFrame(rows, columns=["Date", "dividend", "split"]).set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df


def test_price_only_close_inverts_the_total_return_adjustment():
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(np.linspace(100, 130, 60), index=idx)                      # split-adjusted, price-only
    acts = _actions([(idx[10], 1.0, 1.0), (idx[40], 1.5, 1.0)])
    # Yahoo's adjustment: multiply every earlier close by (1 - D / previous close) at each ex-date
    adj = close.copy()
    for d in acts.index:
        k = idx.get_loc(d)
        adj.iloc[:k] *= 1.0 - acts.loc[d, "dividend"] / close.iloc[k - 1]
    np.testing.assert_allclose(tr.price_only_close(adj, acts).to_numpy(), close.to_numpy(), rtol=1e-12)
    # the total-return index grows by price plus the dividends reinvested
    tri = tr.total_return_index(close, acts)
    assert tri.iloc[0] == 1.0 and tri.iloc[-1] > close.iloc[-1] / close.iloc[0]
    assert tri.iloc[-1] == pytest.approx(adj.iloc[-1] / adj.iloc[0], rel=2e-3)


def test_split_factors_and_units():
    acts = _actions([("2014-06-09", 0.0, 7.0), ("2020-08-31", 0.0, 4.0), ("2021-02-05", 0.205, 1.0)])
    f = tr.split_factor_after(["2013-01-02", "2015-01-02", "2021-01-04"], acts)
    np.testing.assert_allclose(f, [28.0, 4.0, 1.0])
    np.testing.assert_allclose(tr.split_factor_after(["2013-01-02"], acts, end="2019-12-31"), [7.0])
    # a series adjusted at the end of 2015 does not know the 2020 split: its dividends are 4x today's figures
    u = tr.to_units_of(_actions([("2015-05-14", 0.13, 1.0), ("2020-08-31", 0.0, 4.0)]), "2015-12-31")
    assert u["dividend"].iloc[0] == pytest.approx(0.52) and len(u) == 1
    s = pd.Series([10.0, 10.0], index=pd.to_datetime(["2013-01-02", "2021-01-04"]))
    np.testing.assert_allclose(tr.as_traded_close(s, acts).to_numpy(), [280.0, 10.0])


def test_delisting_rules():
    idx = pd.bdate_range("2020-01-01", periods=30)
    px = pd.DataFrame({"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0, "Volume": 1e6}, index=idx)
    end = idx[-1] + pd.Timedelta(days=200)
    assert tr.apply_delisting(px, "exclude", end) is px
    worst = tr.apply_delisting(px, "pessimistic", end)
    assert len(worst) == len(px) + 1 and worst["Close"].iloc[-1] < 1e-6 and "pessimistic" in worst.attrs["delisting"]
    assert tr.apply_delisting(px, "pessimistic", idx[-1]) is px                # still trading: nothing to apply
    assert tr.delisting_return("pessimistic") == -1.0 and tr.delisting_return("exclude") is None
    with pytest.raises(ValueError):
        tr.apply_delisting(px, "optimistic", end)


def test_sample_pack_actions_are_recorded():
    a = tr.load_actions("AAPL", source="sample")
    assert set(a.columns) == {"dividend", "split"}
    assert pd.Timestamp("2020-08-31") in a.index and a.loc["2020-08-31", "split"] == 4.0
    assert tr.load_actions("NOPE", source="sample").empty
