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
                       "trades": 5})
    s = summarize(df, members=4)
    w = (1.1 ** 10 + 1.0) / 2
    assert s.loc["exclude", "cohort_bh_cagr"] == pytest.approx(w ** 0.1 - 1)
    assert s.loc["pessimistic", "cohort_bh_cagr"] == pytest.approx((w / 2) ** 0.1 - 1)    # 2 of 4 members worth nothing
    assert s.loc["pessimistic", "coverage"] == 0.5


def test_static_universe_says_it_is_biased():
    u = mb.static_universe("today's large caps", ["aapl", "brk.b"])
    assert u.members_on("2001-01-02") == ["AAPL", "BRK-B"] and "survivorship" in u.notes[0]
