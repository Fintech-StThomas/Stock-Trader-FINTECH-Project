"""The speed-ups must not change any number: the indicator memo, parallel ticker loading and one-shot chart shapes."""
import threading

import numpy as np
import pytest

from lighthouse import memo
from lighthouse.config import StrategyParams
from lighthouse.parallel import map_tickers
from lighthouse.regime import regime_table
from lighthouse.technicals import beacon_scores, compute_indicators, technical_score


def test_memo_returns_equal_independent_copies(prices):
    memo.clear()
    a = compute_indicators(prices)
    b = compute_indicators(prices)
    assert a is not b and a.equals(b)
    a.loc[a.index[-1], "rsi"] = -1.0                 # a caller scribbling on its copy ...
    assert compute_indicators(prices)["rsi"].iloc[-1] == b["rsi"].iloc[-1] != -1.0   # ... never reaches the cache


def test_memo_keys_on_content_and_on_parameters_that_matter(prices):
    memo.clear()
    base = compute_indicators(prices)
    other = prices.copy()
    other.loc[other.index[-1], "Close"] *= 1.01      # same shape and dates, different data
    assert not np.isclose(compute_indicators(other)["Close"].iloc[-1], base["Close"].iloc[-1])
    swapped = prices.copy()
    swapped.iloc[[-1, -2], swapped.columns.get_loc("Close")] = prices["Close"].iloc[[-2, -1]].to_numpy()   # same sums
    assert memo.frame_fingerprint(swapped) != memo.frame_fingerprint(prices)
    faster = compute_indicators(prices, StrategyParams(band_alpha=0.5))
    assert not np.allclose(faster["band_mid"].dropna().tail(50), base["band_mid"].dropna().tail(50))
    wider = compute_indicators(prices, StrategyParams(band_window=30))
    assert not np.allclose(wider["band_upper"].dropna().tail(50), base["band_upper"].dropna().tail(50))
    # trading thresholds never change an indicator, so a sensitivity grid reuses one computation
    k1 = memo.params_key(StrategyParams(entry_score=55, exit_score=30, cost_bps=20))
    assert k1 == memo.params_key(StrategyParams())
    assert memo.params_key(StrategyParams(rsi_window=10)) != memo.params_key(StrategyParams())


def test_memoised_scores_match_a_fresh_computation(prices, benchmark):
    p = StrategyParams()
    memo.clear()
    cold = (technical_score(compute_indicators(prices, p), "long", p), beacon_scores(compute_indicators(prices, p), p),
            regime_table(benchmark, p))
    warm = (technical_score(compute_indicators(prices, p), "long", p), beacon_scores(compute_indicators(prices, p), p),
            regime_table(benchmark, p))
    for x, y in zip(cold, warm):
        assert x.equals(y)
    assert not technical_score(compute_indicators(prices, p), "short", p).equals(cold[0])   # the horizon is in the key


def test_map_tickers_keeps_order_and_isolates_failures():
    seen = set()

    def fn(t):
        seen.add(threading.current_thread().name)
        if t == "BAD":
            raise ValueError("no such ticker")
        return t.lower()
    out, errors = map_tickers(fn, ["msft", "BAD", "aapl", "MSFT", " "], max_workers=4)
    assert list(out) == ["MSFT", "AAPL"] and out["AAPL"] == "aapl"          # upper-cased, de-duplicated, in order
    assert errors == {"BAD": "ValueError: no such ticker"}
    progress = []
    map_tickers(fn, ["A", "B", "C"], max_workers=1, on_done=lambda t, k, n: progress.append((t, k, n)))
    assert progress == [("A", 1, 3), ("B", 2, 3), ("C", 3, 3)]


def test_charts_draw_every_level_and_regime_in_one_layout():
    pytest.importorskip("plotly")
    from lighthouse import analyze, InvestorProfile
    from lighthouse import interactive as ix
    from lighthouse.tradeplan import build_trade_plan
    r = analyze("AAPL", InvestorProfile.from_cash(5_000, 0), source="sample", years=10, run_backtest_=False)
    pl = build_trade_plan(r, "long", with_stats=False)
    fig = ix.trade_chart(r, pl)
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert sorted(round(s.y0, 6) for s in lines) == sorted(round(y, 6) for y in [pl.price, pl.stop] + [t.price for t in pl.targets])
    assert {a.text.split(" $")[0] for a in fig.layout.annotations} >= {"entry", "stop"}
    full = ix.dashboard(r, last_days=len(r.indicators))          # the full history used to take seconds to shade
    rects = [s for s in full.layout.shapes if s.type == "rect"]
    assert len(rects) > 10 and {(s.xref, s.yref) for s in rects} == {("x", "y domain")}
