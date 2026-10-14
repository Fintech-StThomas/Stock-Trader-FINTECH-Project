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


# --------------------------------------------------------------------------- #
# 13F breadth (recorded excerpts of the SEC's 13F data sets and fails-to-deliver files, October 2026)
# --------------------------------------------------------------------------- #
from pathlib import Path  # noqa: E402

from lighthouse.research import institutions as inst  # noqa: E402

FIX13 = Path(__file__).parent / "fixtures" / "sec13f"
AAPL, MSFT, XOM = "037833100", "594918104", "30231G102"


def test_13f_parser_keeps_on_time_original_share_holdings():
    hold, filers = inst.parse_13f_zip((FIX13 / "2013q3_form13f_excerpt.zip").read_bytes(), [AAPL, MSFT, XOM])
    # six filings in the excerpt: four original 13F-HRs filed by 14 August, one filed late, one amendment
    assert len(filers) == 4 and (filers["period"] == pd.Timestamp("2013-06-30")).all()
    assert set(hold["cusip"]) == {AAPL, MSFT, XOM}                       # a lower-case CUSIP is matched too
    per = hold.groupby("cusip")["cik"].nunique()
    assert (per == 3).all()                                             # the fourth filer holds only options
    assert hold["cik"].isin(filers["cik"]).all()
    # the 2013 Q2 data set has only filings made after the 45-day deadline (structured 13Fs began in late May 2013)
    h2, f2 = inst.parse_13f_zip((FIX13 / "2013q2_form13f_excerpt.zip").read_bytes(), [AAPL, MSFT, XOM])
    assert h2.empty and f2.empty


def test_ftd_parser_and_cusip_mapping():
    ftd = inst.parse_ftd_zip((FIX13 / "cnsfails202608a_excerpt.zip").read_bytes())
    assert len(ftd) > 10 and (ftd["cusip"].str.len() == 9).all()
    assert ftd["symbol"].str.fullmatch(r"[A-Z0-9]+").all()
    q1, q2 = pd.Timestamp("2015-03-31"), pd.Timestamp("2022-03-31")
    obs = {pd.Timestamp("2015-04-01"): pd.DataFrame({"cusip": ["111111111"], "symbol": ["ABC"]}),
           pd.Timestamp("2021-10-01"): pd.DataFrame({"cusip": ["222222222"], "symbol": ["ABC"]})}
    m = inst.cusip_for_quarters(obs, ["ABC", "BRK-B"], [q1, q2, pd.Timestamp("2018-06-30")])
    assert m[("ABC", q1)] == "111111111" and m[("ABC", q2)] == "222222222"     # a reused ticker keeps its own CUSIP
    # mid-2018 is more than two years from both observations: left missing rather than guessed
    assert ("ABC", pd.Timestamp("2018-06-30")) not in m and ("BRK-B", q1) not in m


def test_breadth_change_counts_only_managers_filing_both_quarters():
    p0, p1 = pd.Timestamp("2014-03-31"), pd.Timestamp("2014-06-30")
    filers = pd.DataFrame({"period": [p0] * 4 + [p1] * 4, "cik": [1, 2, 3, 4, 1, 2, 3, 5]})
    hold = pd.DataFrame({"period": [p0, p0, p1, p1, p1, p1],
                         "cik": [1, 4, 1, 2, 3, 5],
                         "cusip": ["X", "X", "X", "X", "X", "X"]})
    bc = inst.breadth_change(hold, filers)
    # managers in both quarters: 1, 2, 3. Holders among them: 1 before, 3 after -> +2 / 3
    assert bc.loc[p1, "X"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# The optional language-model layer (no model is called: the provider is replaced by a canned reply)
# --------------------------------------------------------------------------- #
from lighthouse import llm  # noqa: E402

PASSAGES = ["Our revenue grew 12% in fiscal 2025, driven by cloud services.",
            "We face intense competition, and a loss of large customers could reduce margins.",
            "We had $4.5 billion of debt due in 2027."]


def test_llm_layer_is_off_by_default(monkeypatch):
    monkeypatch.delenv(llm.PROVIDER_ENV, raising=False)
    with pytest.raises(llm.LLMDisabled):
        llm.ask("What grew?", PASSAGES)
    monkeypatch.setenv(llm.PROVIDER_ENV, "anthropic")
    with pytest.raises(llm.LLMDisabled, match="stub"):
        llm.ask("What grew?", PASSAGES)


def test_answers_are_checked_for_citations_and_numbers(monkeypatch):
    monkeypatch.setenv(llm.PROVIDER_ENV, "ollama")
    good = "Revenue grew 12% on cloud services [1]. Competition could reduce margins [2]. Debt of $4.5 billion is due [3]."
    monkeypatch.setitem(llm.PROVIDERS, "ollama", lambda prompt, **kw: (good, "ollama:test"))
    a = llm.ask("What happened?", PASSAGES, {"price": "$83.87"})
    assert a.verified and a.cited() == [1, 2, 3] and a.model == "ollama:test"
    bad = "Revenue grew 15% this year [1]. Margins will collapse next year. Debt is large [7]."
    monkeypatch.setitem(llm.PROVIDERS, "ollama", lambda prompt, **kw: (bad, "ollama:test"))
    b = llm.ask("What happened?", PASSAGES)
    assert not b.verified
    text = " | ".join(b.problems)
    assert "number 15" in text and "no citation" in text and "[7]" in text


def test_prompt_and_passage_selection_are_deterministic():
    doc = "\n".join(["Paragraph about weather and offices in many places around the world. " * 2,
                     "Litigation risk: we are defendants in several lawsuits that could cause losses. " * 2,
                     "Our debt agreements contain covenants that restrict our operations. " * 2])
    ps = llm.split_passages(doc)
    assert len(ps) == 3
    sel = llm.select_passages(ps, "What litigation and lawsuits does it face?", k=1)
    assert sel == [ps[1]]
    p1 = llm.build_prompt("Q?", sel, {"price": "$10"})
    assert p1 == llm.build_prompt("Q?", sel, {"price": "$10"}) and "[1] Litigation" in p1
