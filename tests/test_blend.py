"""The sector/size-neutral blend (lighthouse/research/blend.py): neutral to the industry and size it is told about,
equal-weight across signals, cross-sectional only (no look-ahead)."""
import numpy as np
import pandas as pd
import pytest

from lighthouse.research import blend as B


def _panel(months=24, n=90, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-31", periods=months, freq="ME")
    tickers = [f"T{k:02d}" for k in range(n)]
    industries = {t: ["Banks", "Softw", "Utils", "Oil"][k % 4] for k, t in enumerate(tickers)}
    effect = {"Banks": 2.0, "Softw": -1.0, "Utils": 0.5, "Oil": -1.5}
    mcap = pd.DataFrame(np.exp(rng.normal(24, 1.0, (months, n))), index=dates, columns=tickers)
    noise = pd.DataFrame(rng.normal(size=(months, n)), index=dates, columns=tickers)
    ind_part = pd.DataFrame([[effect[industries[t]] for t in tickers]] * months, index=dates, columns=tickers)
    return dates, tickers, industries, mcap, noise, ind_part


def test_industry_and_size_are_removed():
    dates, tickers, industries, mcap, noise, ind_part = _panel()
    raw = ind_part + 0.8 * np.log(mcap) + noise
    z = B.neutral_scores(raw, industries, mcap)
    groups = pd.Series(industries)
    for d in dates[:5]:
        row = z.loc[d]
        assert row.mean() == pytest.approx(0, abs=1e-9) and row.std(ddof=1) == pytest.approx(1, abs=1e-9)
        assert row.groupby(groups).mean().abs().max() < 1e-9          # no industry left
        assert abs(row.corr(np.log(mcap.loc[d]))) < 1e-9               # no size left
        assert row.corr(noise.loc[d]) > 0.9                            # what is left is the stock-specific part
    # the raw signal was mostly industry and size
    assert raw.loc[dates[0]].groupby(groups).mean().abs().max() > 1


def test_small_industries_are_pooled_and_missing_size_gets_the_median():
    dates, tickers, industries, mcap, noise, _ = _panel(months=3)
    ind = dict(industries)
    ind["T00"], ind["T01"] = "Rare", None          # one name alone in its industry, one with no SIC code
    mc = mcap.copy()
    mc.loc[:, "T02"] = np.nan
    z = B.neutral_scores(noise, ind, mc)
    assert z.loc[dates[0], ["T00", "T01", "T02"]].notna().all()


def test_mask_limits_the_cross_section():
    dates, tickers, industries, mcap, noise, _ = _panel(months=3)
    mask = pd.DataFrame(True, index=dates, columns=tickers)
    mask.loc[:, tickers[:10]] = False
    z = B.neutral_scores(noise, industries, mcap, mask)
    assert z[tickers[:10]].isna().all().all() and z[tickers[10:]].notna().all().all()
    full = B.neutral_scores(noise[tickers[10:]], industries, mcap[tickers[10:]])
    pd.testing.assert_frame_equal(z[tickers[10:]], full)


def test_blend_is_an_equal_weight_mean_with_a_minimum_share():
    dates, tickers, industries, mcap, noise, _ = _panel(months=4)
    rng = np.random.default_rng(11)
    other = pd.DataFrame(rng.normal(size=noise.shape), index=dates, columns=tickers)
    other.loc[:, "T05"] = np.nan                       # one name lacks the second signal
    b = B.blend({"a": noise, "b": other}, industries, mcap)
    za, zb = B.neutral_scores(noise, industries, mcap), B.neutral_scores(other, industries, mcap)
    assert b.loc[dates[0], "T03"] == pytest.approx((za.loc[dates[0], "T03"] + zb.loc[dates[0], "T03"]) / 2)
    assert b.loc[dates[0], "T05"] == pytest.approx(za.loc[dates[0], "T05"])          # 1 of 2 = half: kept
    strict = B.blend({"a": noise, "b": other}, industries, mcap, min_share=1.0)
    assert np.isnan(strict.loc[dates[0], "T05"])
    with pytest.raises(ValueError):
        B.blend({}, industries, mcap)


def test_blend_has_no_look_ahead():
    dates, tickers, industries, mcap, noise, ind_part = _panel()
    raw = ind_part + noise
    full = B.blend({"a": raw, "b": noise}, industries, mcap)
    cut = dates[11]
    trunc = B.blend({"a": raw.loc[:cut], "b": noise.loc[:cut]}, industries, mcap.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], trunc)
    # changing a later month changes nothing before it
    later = raw.copy()
    later.loc[dates[12]:] = later.loc[dates[12]:] * -3
    pd.testing.assert_frame_equal(B.blend({"a": later, "b": noise}, industries, mcap).loc[:cut], full.loc[:cut])


def test_thin_months_are_left_out():
    dates, tickers, industries, mcap, noise, _ = _panel(months=2, n=20)
    assert B.neutral_scores(noise, industries, mcap).isna().all().all()   # 20 names < 30
