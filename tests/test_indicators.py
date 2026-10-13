import numpy as np
import pandas as pd

from lighthouse import indicators as ind
from lighthouse.technicals import compute_indicators, technical_score, beacon_scores, latest_beacons
from lighthouse.config import StrategyParams


def test_sma_matches_pandas(prices):
    s = ind.sma(prices["Close"], 20)
    assert np.allclose(s.dropna(), prices["Close"].rolling(20).mean().dropna())


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(1, 2, 60))
    assert ind.rsi(up, 14).dropna().max() <= 100
    assert ind.rsi(up, 14).iloc[-1] == 100.0
    down = pd.Series(np.linspace(2, 1, 60))
    assert ind.rsi(down, 14).iloc[-1] < 1e-9
    noisy = pd.Series(np.sin(np.arange(300) / 5) + 10)
    r = ind.rsi(noisy, 14).dropna()
    assert r.between(0, 100).all()


def test_bollinger_pct_b_consistency(prices):
    mid, up, lo, pct_b, bw = ind.bollinger(prices["Close"], 20, 2.0)
    valid = pct_b.dropna().index
    inside = ((prices.loc[valid, "Close"] >= lo[valid]) & (prices.loc[valid, "Close"] <= up[valid]))
    assert inside.mean() > 0.8          # most closes sit inside the 2-sigma bands (random walks drift outside more often)
    assert np.allclose(((up - lo) / 2 + lo)[valid], mid[valid])


def test_atr_positive(prices):
    a = ind.atr(prices["High"], prices["Low"], prices["Close"], 14).dropna()
    assert (a > 0).all()


def test_max_drawdown_known_value():
    eq = pd.Series([1, 2, 1, 1.5, 3], index=pd.date_range("2020", periods=5))
    assert ind.max_drawdown(eq) == -0.5


def test_indicators_are_causal(prices):
    """Truncating future data must not change any indicator or score value in the past."""
    full = compute_indicators(prices)
    cut = compute_indicators(prices.iloc[:-100])
    cols = [c for c in cut.columns if cut[c].dtype.kind == "f"]
    a, b = full.loc[cut.index, cols], cut[cols]
    assert np.allclose(a.fillna(-999).values, b.fillna(-999).values)
    sf, sc = technical_score(full), technical_score(cut)
    assert np.allclose(sf.loc[sc.index].fillna(-1).values, sc.fillna(-1).values)


def test_technical_score_range_and_warmup(prices):
    sc = technical_score(compute_indicators(prices))
    assert sc.dropna().between(0, 100).all()
    assert sc.iloc[:199].isna().all()   # 200-day SMA / 52-week high warm-up
    assert sc.iloc[-1] == sc.iloc[-1]   # defined at the end


def test_beacon_scores_in_unit_interval(prices):
    bs = beacon_scores(compute_indicators(prices))
    assert bs.dropna().abs().max().max() <= 1.0 + 1e-9


def test_missing_volume_excludes_volume_beacon(prices):
    p = prices.copy()
    p["Volume"] = np.nan
    ind_df = compute_indicators(p)
    bs = beacon_scores(ind_df)
    assert bs["volume"].isna().all()
    sc = technical_score(ind_df)
    assert sc.dropna().between(0, 100).all()
    beacons = latest_beacons(ind_df)
    vol = [b for b in beacons if b.key == "volume"][0]
    assert np.isnan(vol.score) and vol.light == "grey"


def test_latest_beacons_weights_sum_to_one(prices):
    beacons = latest_beacons(compute_indicators(prices), "long")
    assert abs(sum(b.weight for b in beacons) - 1.0) < 1e-9
    beacons = latest_beacons(compute_indicators(prices), "short")
    assert abs(sum(b.weight for b in beacons) - 1.0) < 1e-9


def test_trend_beacon_sign_in_uptrend():
    up = pd.DataFrame({"Close": np.linspace(50, 150, 400)})
    up["Open"] = up["Close"]; up["High"] = up["Close"] * 1.01; up["Low"] = up["Close"] * 0.99; up["Volume"] = 1e6
    up.index = pd.bdate_range("2020-01-01", periods=400)
    bs = beacon_scores(compute_indicators(up))
    assert bs["trend"].iloc[-1] > 0.9 and bs["breakout"].iloc[-1] > 0.9


def test_wma_and_hull_ma():
    s = pd.Series(np.random.default_rng(1).normal(100, 5, 80))
    w = np.arange(1, 21)
    ref = s.rolling(20).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)
    assert np.allclose(ind.wma(s, 20).dropna(), ref.dropna()) and ind.wma(s, 20).isna().sum() == 19
    ramp = pd.Series(np.arange(200, dtype=float))             # +1 a session
    sma_lag = (ramp - ind.sma(ramp, 20)).dropna()
    hull_lag = (ramp - ind.hull_ma(ramp, 20)).dropna()
    assert np.allclose(sma_lag, 9.5)                          # (20 - 1) / 2 sessions behind
    assert np.allclose(hull_lag, 2 / 3)                       # the Hull MA trails a straight line by 2/3 of a session


def test_lowlag_bands_follow_the_price(prices):
    close = prices["Close"]
    mid, up, lo, pct_b = ind.lowlag_bands(close, 20, 2.0)
    valid = pct_b.dropna().index
    assert (up[valid] >= mid[valid]).all() and (lo[valid] <= mid[valid]).all()
    assert np.allclose(pct_b[valid], ((close - lo) / (up - lo))[valid])
    gap_low = (close - mid).abs().div(close)[valid].mean()
    gap_sma = (close - ind.sma(close, 20)).abs().div(close)[valid].mean()
    assert gap_low < 0.75 * gap_sma                           # the centre sits much closer to the price
    full = compute_indicators(prices)
    assert {"ll_mid", "ll_upper", "ll_lower", "ll_pct_b"} <= set(full.columns)
