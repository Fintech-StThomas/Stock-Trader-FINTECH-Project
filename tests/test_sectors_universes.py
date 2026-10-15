"""Sector labels outside the curated list, the 'Unknown' exemption from sector limits, and the pre-built universes."""
import json
from datetime import datetime

import pandas as pd
import pytest

from conftest import make_prices
import lighthouse.data as data
from lighthouse.portfolio import (GOALS, PortfolioConstraints, PriceUniverse, goal_portfolio, optimize, select,
                                  selection_panels, sector_room)
from lighthouse.universe import (SECTORS, UNIVERSES, UNKNOWN_SECTOR, get_universe, sectors_for, universe_names,
                                 SAMPLE_UNIVERSE_NAME)

CUSTOM = [f"ZZ{c}" for c in "ABCDEFGHIJKL"]          # 12 symbols that are not in the curated list


def _synthetic_universe(tickers, sectors=None) -> PriceUniverse:
    frames = {t: make_prices(n=900, seed=i + 1, drift=0.0002 + 0.00005 * i, vol=0.01 + 0.002 * i) for i, t in enumerate(tickers)}
    closes = pd.concat({t: f["Close"] for t, f in frames.items()}, axis=1).dropna()
    return PriceUniverse(closes=closes, frames=frames, sectors=sectors or sectors_for(tickers, "sample"), benchmark=None,
                         excluded={}, sources={t: "synthetic" for t in tickers})


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    return tmp_path


def test_sector_comes_from_the_curated_list_then_the_sector_cache(tmp_cache):
    (tmp_cache / "fundamentals").mkdir(parents=True)
    (tmp_cache / "fundamentals" / "_sectors.json").write_text(
        json.dumps({"ZZB": {"sector": "Utilities", "as_of": datetime.now().isoformat()}}))
    got = sectors_for(["AAPL", "zza", "ZZB"], source="sample")
    assert got == {"AAPL": "Technology", "ZZA": UNKNOWN_SECTOR, "ZZB": "Utilities"}


def test_live_sector_lookup_is_cached(tmp_cache, monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    asked = []

    def fake(t):
        asked.append(t)
        return {"ZZA": ("Healthcare", True), "ZZC": (None, False)}.get(t, (None, True))   # ZZB: an ETF; ZZC: a thin answer
    monkeypatch.setattr(data, "_sector_from_yfinance", fake)
    got = sectors_for(["ZZA", "ZZB", "ZZC", "MSFT"], source="auto")
    assert got == {"ZZA": "Healthcare", "ZZB": UNKNOWN_SECTOR, "ZZC": UNKNOWN_SECTOR, "MSFT": "Technology"}
    assert sorted(asked) == ["ZZA", "ZZB", "ZZC"]      # curated tickers never hit the network
    cached = json.loads((tmp_cache / "fundamentals" / "_sectors.json").read_text())
    assert cached["ZZA"]["sector"] == "Healthcare" and cached["ZZB"]["sector"] is None
    assert "ZZC" not in cached                         # a stock that came back without a sector is asked again
    asked.clear()
    assert sectors_for(["ZZA", "ZZB", "ZZC"], source="auto")["ZZA"] == "Healthcare" and asked == ["ZZC"]


def test_offline_lookup_never_calls_the_network(tmp_cache, monkeypatch):
    monkeypatch.setattr(data, "_sector_from_yfinance", lambda t: pytest.fail("network used offline"))
    assert sectors_for(["ZZA"], source="auto") == {"ZZA": UNKNOWN_SECTOR}


def test_unknown_sectors_do_not_cap_a_goal_portfolio_at_three_stocks(tmp_cache):
    uni = _synthetic_universe(CUSTOM)
    assert set(uni.sectors.values()) == {UNKNOWN_SECTOR}
    panels = selection_panels(uni)
    g = GOALS["preservation"]                         # ranks by low volatility: no score filter
    picked = select(panels, len(uni.closes) - 1, g, uni.sectors, max_per_sector=3)
    assert len(picked) == g.n_stocks == 10            # was 3: every unknown ticker shared one "Unknown" sector
    r = goal_portfolio("preservation", capital=10_000, universe=uni, run_backtest=False)
    assert len(r.weights) == 10
    assert not any("sector cap was dropped" in n for n in r.notes)   # the 35% sector cap still applies
    assert any("No sector found" in n for n in r.notes)


def test_sector_caps_apply_to_known_sectors_only(tmp_cache):
    tickers = CUSTOM[:8]
    sectors = ["Energy"] * 4 + [UNKNOWN_SECTOR] * 4
    cons = PortfolioConstraints(max_weight=0.25, max_sector_weight=0.30)
    cons.check_feasible(tickers, sectors)             # 30% energy + 4 x 25% unknown: feasible
    assert sector_room(sectors, 0.30, 0.25) == pytest.approx(1.30)
    with pytest.raises(ValueError, match="sector cap"):
        cons.check_feasible(tickers, ["Energy"] * 8)  # one real sector at 30%: still infeasible
    uni = _synthetic_universe(tickers, dict(zip(tickers, sectors)))
    w, _ = optimize(uni.returns.iloc[-252:], "equal_weight", cons, sectors)
    assert w.sum() == pytest.approx(1.0)
    assert w.iloc[:4].sum() <= 0.30 + 1e-4 and (w <= 0.25 + 1e-4).all()
    w_all_unknown, _ = optimize(uni.returns.iloc[-252:], "min_variance", cons, [UNKNOWN_SECTOR] * 8)
    assert w_all_unknown.sum() == pytest.approx(1.0) and (w_all_unknown <= 0.25 + 1e-4).all()


def test_prebuilt_universes():
    for name, tickers in UNIVERSES.items():
        assert len(tickers) >= 6 and len(set(tickers)) == len(tickers), name
        stocks = [t for t in tickers if not name.endswith("ETFs")]
        assert all(t in SECTORS for t in stocks), name           # every stock universe is labelled
    assert get_universe("semiconductors") == UNIVERSES["Semiconductors"]
    assert universe_names("sample") == [SAMPLE_UNIVERSE_NAME]
    assert "MSFT" in get_universe(SAMPLE_UNIVERSE_NAME) and "^GSPC" not in get_universe(SAMPLE_UNIVERSE_NAME)
    with pytest.raises(ValueError, match="unknown universe"):
        get_universe("no such list")


def test_a_single_sector_universe_still_fills_the_goal(tmp_cache):
    tickers = CUSTOM[:12]
    uni = _synthetic_universe(tickers, {t: "Technology" for t in tickers})
    r = goal_portfolio("preservation", capital=10_000, universe=uni, run_backtest=False)
    assert len(r.weights) == GOALS["preservation"].n_stocks          # not 3: the per-sector limit is raised to 10
    assert any("has 1 sector" in n and "up to 10 stocks per sector" in n for n in r.notes)
    assert any("sector cap was dropped" in n for n in r.notes)        # one sector cannot hold 100% under 35%
    mixed = _synthetic_universe(tickers, {t: ("Energy" if i < 6 else "Utilities") for i, t in enumerate(tickers)})
    r2 = goal_portfolio("preservation", capital=10_000, universe=mixed, run_backtest=False)
    assert len(r2.weights) == 10 and any("up to 5 stocks" in n for n in r2.notes)
    # stocks without a sector are never limited: 15 tech + 10 unknown fill 10 without raising the limit
    big = [f"T{i:02d}" for i in range(15)] + [f"U{i:02d}" for i in range(10)]
    mixed2 = _synthetic_universe(big, {t: ("Technology" if t.startswith("T") else UNKNOWN_SECTOR) for t in big})
    r3 = goal_portfolio("preservation", capital=10_000, universe=mixed2, run_backtest=False)
    assert not any("stocks per sector are allowed" in n for n in r3.notes)
    assert sum(mixed2.sectors[t] == "Technology" for t in r3.weights.index) <= 3
