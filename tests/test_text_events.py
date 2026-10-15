"""Deterministic text features and the event signals (earnings reaction, 10-K language), offline. The word lists here
are small hand-made ones for the tests, not the Loughran-McDonald dictionary (which Lighthouse never redistributes)."""
import io
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from lighthouse import text
from lighthouse.research import event_signals as ev

LM_CSV = """Word,Seq_num,Word Count,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining,Syllables
LOSS,1,10,2009,0,0,0,0,0,0,1
MAY,2,10,0,0,2009,0,0,2009,0,1
UNCERTAIN,3,10,0,0,2009,0,0,0,0,3
LAWSUIT,4,10,0,0,0,2009,0,0,0,2
GAIN,5,10,0,2009,0,0,0,0,0,1
COULD,6,10,0,0,-2012,0,0,0,0,1
"""


@pytest.fixture(scope="module")
def lm():
    return text.parse_lm_csv(io.StringIO(LM_CSV))


def test_lm_csv_flags_and_removed_words(lm):
    assert lm.words["negative"] == {"LOSS"}
    assert lm.words["uncertainty"] == {"MAY", "UNCERTAIN"}            # COULD was removed (negative year)
    assert lm.syllables["UNCERTAIN"] == 3


def test_html_to_text_drops_hidden_xbrl_and_scripts():
    doc = ("<html><head><title>x</title></head><body><ix:header><ix:hidden>SECRET 123</ix:hidden></ix:header>"
           "<script>var a=1;</script><p>Net&nbsp;sales &amp; revenue</p><div>grew</div></body></html>")
    t = text.html_to_text(doc)
    assert "SECRET" not in t and "var a" not in t
    assert "Net sales & revenue" in t and "grew" in t
    assert text.tokens("It's a 10-K, Item 1A.") == ["IT", "ITEM"]


def test_risk_factors_skips_the_table_of_contents():
    body = "We may lose money. " * 200
    doc = ("Table of contents Item 1A. Risk Factors 12 Item 1B. Unresolved Staff Comments 20 Item 2. Properties 21 "
           f"... Item 1A. Risk Factors {body} Item 1B. Unresolved Staff Comments None. Item 2. Properties Offices.")
    rf = text.risk_factors(doc)
    assert rf.strip().startswith("We may lose money") and "Unresolved" not in rf
    assert text.risk_factors("Item 1A. Risk Factors short. Item 2. Properties x") == ""


def test_tone_and_similarity(lm):
    words = text.tokens("We may record a loss. The lawsuit is uncertain. A gain may follow.")
    t = text.tone(words, lm)
    assert t["words"] == len(words)
    assert t["uncertainty"] == pytest.approx(3 / len(words))          # MAY twice, UNCERTAIN once
    assert t["litigious"] == pytest.approx(1 / len(words))
    a = Counter({"A": 2, "B": 1})
    assert text.cosine(a, a) == pytest.approx(1.0)
    assert text.cosine(a, Counter({"C": 1})) == 0.0
    assert np.isnan(text.cosine(a, Counter()))


def test_features_compare_with_the_previous_filing(lm):
    risk = "Item 1A. Risk Factors " + "We may face a lawsuit and a loss. " * 120 + " Item 2. Properties offices."
    f1, c1 = text.features("<p>" + risk + "</p>", lm)
    assert np.isnan(f1["similarity"])                                  # nothing to compare with yet
    f2, _ = text.features("<p>" + risk + "</p>", lm, c1)
    assert f2["similarity"] == pytest.approx(1.0) and f2["risk_similarity"] == pytest.approx(1.0)
    assert f2["length_change"] == pytest.approx(0.0)
    f3, _ = text.features("<p>" + risk.replace("lawsuit", "gain") + " New risks appear everywhere.</p>", lm, c1)
    assert f3["similarity"] < 1.0 and f3["lm_litigious"] < f2["lm_litigious"]


def test_fog_index_counts_long_words_and_sentences(lm):
    easy = text.fog("The cat sat. The dog ran. We ate.")
    hard = text.fog("Uncertain regulatory considerations necessitate comprehensive evaluation.")
    assert easy < 5 < hard


# --------------------------------------------------------------------------- #
# Earnings announcement return
# --------------------------------------------------------------------------- #
def prices(n=300, seed=3):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)


def test_announcement_return_is_the_three_session_excess_return():
    close = prices()
    market = close.pct_change() * 0.0                                   # a flat market: the excess is the raw return
    d = close.index[100]
    jump = close.copy()
    jump.iloc[100:] *= 1.08                                             # +8% on the release session
    out = ev.announcement_returns(jump, market, pd.Series([d]))
    assert list(out.index) == [close.index[102]]                        # usable the session after the window
    expect = jump.iloc[101] / jump.iloc[98] - 1.0
    assert out.iloc[0] == pytest.approx(expect)
    # a release filed on a Saturday belongs to the Monday session
    sat = d + pd.Timedelta(days=(5 - d.weekday()) % 7)
    out2 = ev.announcement_returns(jump, market, pd.Series([sat]))
    assert out2.index[0] == close.index[close.index.searchsorted(sat) + 2]


def test_event_values_are_causal_and_expire():
    close = prices()
    market = pd.Series(0.0, index=close.index)
    dates = pd.Series([close.index[50], close.index[150]])
    full = ev.hold_daily(ev.announcement_returns(close, market, dates), close.index, ev.EAR_MAX_AGE_DAYS)
    cut = close.index[120]
    part = ev.hold_daily(ev.announcement_returns(close[:cut], market[:cut], dates[dates <= cut]),
                         close[:cut].index, ev.EAR_MAX_AGE_DAYS)
    pd.testing.assert_series_equal(full[:cut], part, check_names=False)
    assert full.loc[:close.index[51]].isna().all()                      # not known before the window closes
    assert np.isnan(full.loc[close.index[52] + pd.Timedelta(days=ev.EAR_MAX_AGE_DAYS + 3):close.index[149]]).all()


def test_text_workers_split_this_runs_sec_rate(monkeypatch, tmp_path):
    """LIGHTHOUSE_SEC_RATE gives a run a share of the SEC's 10 requests a second (other processes download at the same
    time); the 10-K text workers must split that share, not 8 a second, between them."""
    from types import SimpleNamespace

    from lighthouse import data, http
    from lighthouse.research import harness as H
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(http, "SEC_RATE", 1.0)                     # as with LIGHTHOUSE_SEC_RATE=1
    monkeypatch.setattr(http.POLICIES["sec.gov"], "rate", 1.0)
    seen = {}

    class Pool:                                                     # records what the workers would be given
        def __init__(self, max_workers, initializer, initargs):
            seen.update(workers=max_workers, initializer=initializer, initargs=initargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, jobs, chunksize=1):
            return [(t, None, None) for t, _, _ in jobs]

    monkeypatch.setattr(ev, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(text, "lm_dictionary", lambda: None)
    idx = pd.bdate_range("2020-01-01", periods=60)
    frames = {"AAA": pd.DataFrame({"Close": 1.0}, index=idx), "BBB": pd.DataFrame({"Close": 2.0}, index=idx)}
    ev.text_monthly(frames, SimpleNamespace(dates=H.month_ends(idx)), refresh=True, processes=3)
    assert seen["workers"] == 3 and seen["initializer"] is ev._worker_init
    assert seen["workers"] * seen["initargs"][0] == pytest.approx(1.0)       # 1 request a second in total, not 8
    # without LIGHTHOUSE_SEC_RATE the run has 8 a second, split the same way
    monkeypatch.setattr(http, "SEC_RATE", 8.0)
    monkeypatch.setattr(http.POLICIES["sec.gov"], "rate", 8.0)
    assert 4 * ev.worker_sec_rate(4) == pytest.approx(8.0) and ev.worker_sec_rate(0) == pytest.approx(8.0)
    # a policy lowered at run time is respected too
    monkeypatch.setattr(http.POLICIES["sec.gov"], "rate", 2.0)
    assert ev.worker_sec_rate(2) == pytest.approx(1.0)
    # the initializer installs the share as the worker's SEC policy
    ev._worker_init(0.25)
    assert http.POLICIES["sec.gov"].rate == 0.25


# --------------------------------------------------------------------------- #
# 10-K documents of a successor issuer
# --------------------------------------------------------------------------- #
def test_text_history_reads_each_10k_under_the_registrant_that_filed_it(lm, monkeypatch):
    # a successor issuer's list holds its predecessor's 10-Ks (edgar.submissions merges them), which sit under the
    # predecessor's CIK: ExxonMobil Holdings Corp (2115436, 2026) <- Exxon Mobil Corp (34088). Synthetic filings.
    from lighthouse import edgar
    filings = pd.DataFrame({"accn": ["a-2024", "a-2025", "a-2027"], "form": ["10-K", "10-K", "10-K"],
                            "filed": pd.to_datetime(["2024-02-20", "2025-02-19", "2027-02-17"]),
                            "doc": ["k24.htm", "k25.htm", "k27.htm"], "cik": [34088, 34088, 2115436]})
    monkeypatch.setattr(edgar, "cik_for", lambda t: 2115436)
    monkeypatch.setattr(edgar, "company_filings", lambda t, source="auto", refresh=False: ({}, filings))
    asked = []
    monkeypatch.setattr(edgar, "fetch_document", lambda cik, accn, doc, cache=True: asked.append((cik, accn)) or
                        "We expect a loss. Our risk factors are uncertain. Revenue may decline.")
    monkeypatch.setattr(ev.store, "read_frame", lambda kind, key: None)
    monkeypatch.setattr(ev.store, "write_frame", lambda kind, key, df: None)
    h = ev.text_history("XOM", lm)
    assert asked == [(34088, "a-2024"), (34088, "a-2025"), (2115436, "a-2027")] and len(h) == 3
