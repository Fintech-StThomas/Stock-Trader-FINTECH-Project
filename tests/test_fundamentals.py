import numpy as np
import pandas as pd

from lighthouse.fundamentals import (FundamentalSnapshot, StatementHistory, fundamental_beacons, fundamental_score,
                                     point_in_time_scores, score_multiple_vs_norm, score_band)
from lighthouse.config import StrategyParams


def snap(**kw):
    base = dict(ticker="X", name="X Corp", sector="Technology", price=100, market_cap=1e10, pe_trailing=20, pe_forward=18,
                peg=1.0, pb=5, ev_ebitda=15, roe=0.25, operating_margin=0.30, debt_to_equity=0.5, current_ratio=2,
                fcf_yield=0.05, revenue_growth=0.15, eps_growth=0.20)
    base.update(kw)
    return FundamentalSnapshot(**base)


def test_scoring_helpers():
    assert score_multiple_vs_norm(10, 20) == 1.0          # half the norm -> cheap
    assert score_multiple_vs_norm(30, 20) == -1.0         # 1.5x the norm -> expensive
    assert score_multiple_vs_norm(20, 20) == 0.0
    assert score_multiple_vs_norm(-5, 20) == -0.75        # negative earnings
    assert score_multiple_vs_norm(None, 20) is None
    assert score_band(0.25, 0.10, 0.15) == 1.0
    assert score_band(-0.05, 0.10, 0.15) == -1.0


def test_strong_company_scores_high_and_weak_scores_low():
    strong = fundamental_score(fundamental_beacons(snap()))
    weak = fundamental_score(fundamental_beacons(snap(pe_trailing=60, peg=4, pb=15, ev_ebitda=40, roe=-0.1,
                                                      operating_margin=-0.05, debt_to_equity=3, fcf_yield=-0.02,
                                                      revenue_growth=-0.2, eps_growth=-0.5)))
    assert strong > 75 and weak < 25


def test_group_weights_sum_to_one_and_missing_groups_renormalise():
    beacons = fundamental_beacons(snap())
    assert abs(sum(b.weight for b in beacons) - 1.0) < 1e-9
    only_quality = fundamental_beacons(snap(pe_trailing=None, pe_forward=None, peg=None, pb=None, ev_ebitda=None,
                                            revenue_growth=None, eps_growth=None))
    assert all(b.key.startswith("quality") for b in only_quality)
    assert fundamental_score(only_quality) is not None


def test_unavailable_snapshot_gives_none():
    assert fundamental_score(fundamental_beacons(FundamentalSnapshot(ticker="Z"))) is None
    assert not FundamentalSnapshot(ticker="Z").available


def test_financials_skip_leverage_beacon():
    b = fundamental_beacons(snap(sector="Financial Services", debt_to_equity=5))
    assert not any(x.key.endswith("leverage") for x in b)


def test_default_beacons_keep_the_sector_norms_and_wording():
    """Without `norms` the beacons are the sector-norm ones the decision and the backtest were tested with."""
    b = {x.key: x for x in fundamental_beacons(snap())}
    assert b["valuation:pe"].value == "P/E 20.0 vs Technology ~28"
    assert b["valuation:pb"].value == "P/B 5.00 vs ~7.0"
    assert b["valuation:ev_ebitda"].value == "EV/EBITDA 15.0 vs ~20"
    assert b["quality:margin"].value == "op. margin 30.0% vs ~22.0%"
    assert "typical Technology multiple (~28)" in b["valuation:pe"].rationale
    assert not any("peer" in x.value or "Norm:" in x.rationale for x in b.values())
    same = fundamental_beacons(snap(), norms=None)
    assert [vars(x) for x in same] == [vars(x) for x in b.values()]


def test_peer_norms_replace_the_sector_norms_and_say_which():
    norms = {"pe": 40.0, "pb": None, "ev_ebitda": 25.0, "op_margin": 0.35, "label": "peer median",
             "source": "median of 6 peers (4-digit SIC 7372): A, B, C, D, E, F"}
    base = {x.key: x for x in fundamental_beacons(snap())}
    peer = {x.key: x for x in fundamental_beacons(snap(), norms=norms)}
    assert peer["valuation:pe"].score > base["valuation:pe"].score         # P/E 20 is cheap against a 40x median
    assert peer["valuation:pe"].value == "P/E 20.0 vs peer median ~40"
    assert "median of 6 peers (4-digit SIC 7372)" in peer["valuation:pe"].rationale
    assert peer["valuation:ev_ebitda"].value == "EV/EBITDA 15.0 vs peer median ~25"
    assert peer["quality:margin"].score < base["quality:margin"].score     # a 30% margin against 35% at the peers
    assert "vs peer median" in peer["quality:margin"].value
    # a peer median that is missing keeps the sector norm, and the sector wording
    assert vars(peer["valuation:pb"]) == vars(base["valuation:pb"])
    # beacons on fixed bands are untouched (same scores, texts and weights)
    for k in ("valuation:peg", "quality:roe", "quality:fcf", "growth:rev_growth", "growth:eps_growth"):
        assert vars(peer[k]) == vars(base[k])
    # norms equal to the sector's score exactly like the default
    from lighthouse.fundamentals import SECTOR_NORMS
    eq = fundamental_beacons(snap(), norms={**SECTOR_NORMS["Technology"], "label": "peer median", "source": "x"})
    assert [x.score for x in eq] == [x.score for x in base.values()]


def test_peer_norms_never_turn_ev_ebitda_on_for_financials():
    b = fundamental_beacons(snap(sector="Financial Services"),
                            norms={"pe": 11.0, "ev_ebitda": 9.0, "label": "peer median", "source": "x"})
    assert not any(x.key.endswith("ev_ebitda") for x in b)
    assert any(x.value == "P/E 20.0 vs peer median ~11" for x in b)


def test_point_in_time_scores_respect_reporting_lag(prices):
    annual = pd.DataFrame({"revenue": [100, 120], "net_income": [10, 15], "eps": [1.0, 1.5], "operating_income": [20, 28],
                           "equity": [50, 60], "total_debt": [20, 20]},
                          index=pd.to_datetime(["2016-12-31", "2017-12-31"]))
    hist = StatementHistory(annual=annual)
    p = StrategyParams(reporting_lag_days=90)
    s = point_in_time_scores(prices, hist, "Technology", p)
    first_known = pd.Timestamp("2016-12-31") + pd.Timedelta(days=90)
    assert s[s.index < first_known].isna().all()
    assert s[s.index >= first_known].notna().all()
    assert s.dropna().between(0, 100).all()
    # valuation moves daily with price: scores are not constant within a statement window
    window = s[(s.index >= first_known) & (s.index < pd.Timestamp("2017-12-31") + pd.Timedelta(days=90))]
    assert window.std() > 0
