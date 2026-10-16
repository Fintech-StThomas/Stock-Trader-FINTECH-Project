"""Survivorship-aware universes on recorded excerpts of the real sources (tests/fixtures/universe, recorded October 2026):
fja05680/sp500 snapshots and Wikipedia's S&P 500 pages."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lighthouse import membership as mb
from lighthouse.research.cohort import summarize

FIX = Path(__file__).parent / "fixtures" / "universe"


@pytest.fixture(scope="module")
def snaps():
    return mb.parse_fja_snapshots((FIX / "fja_sp500_snapshots_excerpt.csv").read_text())


def test_snapshots_become_membership_intervals(snaps):
    assert snaps["date"].min() == pd.Timestamp("1996-01-02")
    assert (snaps.groupby("date").size() > 480).all()                     # about 500 members on every date
    assert "BRK-B" in set(snaps["ticker"]) and not any("." in t for t in snaps["ticker"])   # one spelling of share classes
    iv = mb.snapshots_to_intervals(snaps)
    u = mb.Universe("test", iv)
    first, last = snaps["date"].min(), snaps["date"].max()
    assert set(u.members_on(first)) == set(snaps.loc[snaps["date"] == first, "ticker"])
    assert set(u.members_on(last)) == set(snaps.loc[snaps["date"] == last, "ticker"])
    gone = set(snaps.loc[snaps["date"] == first, "ticker"]) - set(snaps.loc[snaps["date"] == last, "ticker"])
    assert len(gone) > 300                                                 # most 1996 members are no longer in the index
    t = sorted(gone)[0]
    row = iv[iv["ticker"] == t].iloc[0]
    assert pd.notna(row["end"]) and row["end"] < last


def test_change_log_brings_membership_forward(snaps):
    changes = mb.parse_wiki_changes((FIX / "wikipedia_sp500_changes_excerpt.html").read_text())
    assert {"date", "added", "removed"} <= set(changes.columns) and changes["date"].is_monotonic_increasing
    twlo = changes[changes["added"] == "TWLO"].iloc[0]
    assert twlo["removed"] == "WBD" and twlo["date"] == pd.Timestamp("2026-10-06")
    iv = mb.snapshots_to_intervals(snaps)
    iv2, notes = mb.apply_changes(iv, changes, snaps["date"].max())
    u = mb.Universe("test", iv2)
    on = set(u.members_on("2026-10-07"))
    assert "TWLO" in on and "WBD" not in on and "CTVA" not in on and "VYLR" in on
    assert "WBD" in u.members_on("2026-10-05")
    cur = mb.parse_wiki_current((FIX / "wikipedia_sp500_current_excerpt.html").read_text())
    assert {"ticker", "gics_sector", "cik"} <= set(cur.columns) and cur.loc[cur.ticker == "MMM", "cik"].iloc[0] == 66740


def test_coverage_counts_member_days_with_prices():
    iv = pd.DataFrame({"ticker": ["A", "B", "C"], "start": pd.to_datetime(["2020-01-01"] * 3),
                       "end": [pd.NaT, pd.Timestamp("2020-06-30"), pd.NaT]})
    u = mb.Universe("t", iv)
    priced = {"A": pd.bdate_range("2019-01-01", "2020-12-31"), "B": pd.bdate_range("2020-01-01", "2020-03-31")}
    cov = u.coverage(priced, pd.to_datetime(["2020-02-28", "2020-05-29", "2020-09-30"]))
    assert cov["members"].tolist() == [3, 3, 2]
    assert cov["with_prices"].tolist() == [2, 1, 1]                        # B's prices stop in March, C has none
    assert cov["coverage"].iloc[0] == pytest.approx(2 / 3)


def test_cohort_bounds_count_untested_members_as_lost():
    df = pd.DataFrame({"ticker": ["A", "B"] * 2, "rule": ["exclude"] * 2 + ["pessimistic"] * 2, "delisted": False,
                       "years": 10.0, "bh_cagr": [0.10, 0.0] * 2, "lighthouse_cagr": [0.05, 0.0] * 2,
                       "signals_cagr": [0.07, 0.0] * 2, "bh_sharpe": 0.5, "bh_mdd": -0.5, "signals_sharpe": 0.4,
                       "lighthouse_vol": 0.1, "lighthouse_sharpe": 0.3, "lighthouse_mdd": -0.2, "conservative_sharpe": 0.2,
                       "same_cagr": [0.04, 0.0] * 2, "same_sharpe": [0.35, 0.25] * 2, "same_mdd": -0.25, "trades": 5})
    s = summarize(df, members=4)
    w = (1.1 ** 10 + 1.0) / 2
    assert s.loc["exclude", "cohort_bh_cagr"] == pytest.approx(w ** 0.1 - 1)
    assert s.loc["exclude", "cohort_same_cagr"] == pytest.approx(((1.04 ** 10 + 1.0) / 2) ** 0.1 - 1)
    assert s.loc["exclude", "beat_same_sharpe"] == 1 and s.loc["exclude", "lower_mdd_than_same"] == 2
    assert s.loc["pessimistic", "cohort_bh_cagr"] == pytest.approx((w / 2) ** 0.1 - 1)    # 2 of 4 members worth nothing
    assert s.loc["pessimistic", "coverage"] == 0.5


def test_pre_registered_timing_test_is_a_paired_sign_test():
    """METHODOLOGY 8d: per stock, the full rule against the same exposure (CAGR, drawdown) and the plain trend rule
    (Sharpe); a hypothesis holds in a cohort when the rule wins on more than half the stocks with p < 0.05."""
    from scipy.stats import binomtest
    from lighthouse.research.cohort import paired_sign_test, timing_test
    n = 40
    rng = np.random.default_rng(3)
    base = rng.normal(0.08, 0.03, n)
    df = pd.DataFrame({"ticker": [f"T{k}" for k in range(n)], "rule": "exclude",
                       "lighthouse_cagr": base - 0.01, "same_cagr": base,                         # timing loses return
                       "lighthouse_mdd": -0.25 + rng.normal(0, 0.01, n), "same_mdd": -0.40,       # ... and cuts drawdown
                       "lighthouse_sharpe": 0.4 + np.r_[np.full(22, 0.01), np.full(18, -0.01)],   # 22 of 40: not enough
                       "trend_sharpe": 0.4})
    t = timing_test(df).set_index("hypothesis")
    h1, h2, h3 = t.index
    assert t.loc[h1, "wins"] == 0 and not t.loc[h1, "holds"]
    assert t.loc[h2, "wins"] == n and t.loc[h2, "holds"] and t.loc[h2, "p_value"] < 1e-9
    assert t.loc[h3, "wins"] == 22 and not t.loc[h3, "holds"]
    assert t.loc[h3, "p_value"] == pytest.approx(binomtest(22, 40, 0.5).pvalue)
    # ties and missing values are left out of the count
    r = paired_sign_test(pd.Series([1.0, 2.0, np.nan, 3.0]), pd.Series([1.0, 1.0, 1.0, 1.0]))
    assert (r["wins"], r["n"]) == (2, 2)
    none = timing_test(df, rule="pessimistic")                          # no rows under that rule: nothing holds
    assert (none["n"] == 0).all() and not none["holds"].any()


def test_static_universe_says_it_is_biased():
    u = mb.static_universe("today's large caps", ["aapl", "brk.b"])
    assert u.members_on("2001-01-02") == ["AAPL", "BRK-B"] and "survivorship" in u.notes[0]


def test_market_cap_from_cover_shares_and_as_traded_price():
    """AAPL on 2 January 2015: about 5.8 billion shares (10-K cover, October 2014) x $109.33 as traded = ~$640bn."""
    px = pd.read_csv("data/sample/prices/AAPL.csv", index_col=0, parse_dates=True)["Close"]
    from lighthouse.total_return import load_actions, price_only_close, to_units_of
    split_adj = price_only_close(px, to_units_of(load_actions("AAPL", source="sample"), px.index[-1]))
    mc = mb.market_cap_on("2015-01-02", ["AAPL", "NOPE"], {"AAPL": split_adj}, source="sample")
    assert 5.5e11 < mc["AAPL"] < 7.0e11 and np.isnan(mc["NOPE"])
    u = mb.market_cap_universe(["AAPL"], ["2015-01-02", "2016-01-04"], {"AAPL": split_adj}, top_n=1, source="sample")
    assert u.members_on("2015-06-01") == ["AAPL"]


def test_market_cap_after_a_split_uses_the_restated_count():
    """AAPL on 15 September 2020, after the 4-for-1 split of 31 August and before the next 10-K: the latest cover page
    (July 2020) counts pre-split shares; restated, the value is ~$1.97tn (unrestated it was a quarter of that), the
    same as the memo's and the research signals' market value."""
    px = pd.read_csv("data/sample/prices/AAPL.csv", index_col=0, parse_dates=True)["Close"]
    from lighthouse.data import load_prices
    from lighthouse.research import fundamental_signals as FS
    from lighthouse.total_return import load_actions, price_only_close, to_units_of
    split_adj = price_only_close(px, to_units_of(load_actions("AAPL", source="sample"), px.index[-1]))
    mc = mb.market_cap_on("2020-09-15", ["AAPL"], {"AAPL": split_adj}, source="sample")["AAPL"]
    assert 1.8e12 < mc < 2.2e12
    df = load_prices("AAPL", source="sample", years=None)
    df.attrs = {}
    assert FS.daily_values("AAPL", df, source="sample")["market_cap"].loc["2020-09-15"] == pytest.approx(mc, rel=1e-9)


def test_delisting_names_match_edgar_filers(monkeypatch):
    import lighthouse.edgar as edgar
    fake = pd.DataFrame({"cik": [1, 2], "name": ["LEHMAN BROTHERS HOLDINGS INC", "WASHINGTON MUTUAL, INC."],
                         "form": ["25-NSE", "15-12B"], "filed": pd.to_datetime(["2008-09-17", "2008-10-01"]),
                         "accn": ["a", "b"]})
    monkeypatch.setattr(edgar, "delistings", lambda start_year=2001, end_year=None, refresh=False: fake)
    d = mb.delisting_dates({"LEH": "Lehman Brothers Holdings Inc.", "WM": "Washington Mutual", "XYZ": "Nobody Corp"})
    assert d.set_index("ticker").loc["LEH", "form"] == "25-NSE" and set(d["ticker"]) == {"LEH", "WM"}


def test_a_blank_ticker_in_the_change_log_adds_no_member():
    """pandas 3 stores a blank text cell as NaN, which is truthy: a pure removal must not add a member without a ticker."""
    iv = pd.DataFrame({"ticker": ["AAA", "BBB"], "start": pd.Timestamp("2020-01-01"), "end": pd.NaT})
    changes = pd.DataFrame({"date": pd.to_datetime(["2021-03-01", "2021-04-01"]), "added": [np.nan, "CCC"],
                            "removed": ["BBB", float("nan")]})
    out, notes = mb.apply_changes(iv, changes, pd.Timestamp("2020-12-31"))
    assert out["ticker"].notna().all() and set(out["ticker"]) == {"AAA", "BBB", "CCC"}
    assert mb.Universe("t", out).members_on("2021-05-01") == ["AAA", "CCC"] and not notes
