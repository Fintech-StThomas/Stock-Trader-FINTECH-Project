"""SEC EDGAR client and point-in-time statements, on recorded real responses (tests/fixtures/edgar, recorded October
2026) and the sample pack's recorded XBRL facts. Nothing here touches the network."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lighthouse import edgar, http, pit
from lighthouse.data import DataUnavailable
from lighthouse.fundamentals import StatementHistory, point_in_time_scores
from lighthouse.total_return import load_actions

FIX = Path(__file__).parent / "fixtures" / "edgar"


@pytest.fixture(scope="module")
def msft_facts():
    return edgar.flatten_company_facts(json.loads((FIX / "msft_companyfacts_2016_2018.json").read_text()))


def test_flatten_company_facts_keeps_filing_dates(msft_facts):
    f = msft_facts
    assert set(edgar.FACT_COLUMNS) <= set(f.columns)
    assert f["filed"].notna().all() and f["end"].notna().all()
    ni = f[(f.concept == "NetIncomeLoss") & (f.start == "2016-07-01") & (f.end == "2017-06-30")].sort_values("filed")
    # FY2017 net income: $21,204m in the 10-K of 2 August 2017, restated to $25,489m (ASC 606) a year later
    assert ni["val"].iloc[0] == 21_204_000_000 and ni["filed"].iloc[0] == pd.Timestamp("2017-08-02")
    assert ni["val"].iloc[-1] == 25_489_000_000 and ni["filed"].iloc[-1] > pd.Timestamp("2018-08-01")
    only = edgar.flatten_company_facts(json.loads((FIX / "msft_companyfacts_2016_2018.json").read_text()), ["NetIncomeLoss"])
    assert set(only.concept) == {"NetIncomeLoss"}


def test_quarters_from_year_to_date_values_and_first_reported_values(msft_facts):
    h = pit.build_statements(msft_facts)
    q, a = h.quarterly, h.annual
    fy17, fy18 = pd.Timestamp("2017-06-30"), pd.Timestamp("2018-06-30")
    # the fourth quarter is the 10-K's full year minus the 10-Q's nine months, dated by the 10-K
    assert q.loc[fy17, "net_income"] == pytest.approx(21_204e6 - (4_690e6 + 5_200e6 + 4_801e6))
    assert q.loc[fy17, "filed"] == pd.Timestamp("2017-08-02")
    # cash flow is reported year to date: the fourth quarter is the year minus nine months
    assert q.loc[fy18, "operating_cash_flow"] == pytest.approx(43_884e6 - 32_466e6)
    assert q.loc[fy18, "filed"] == pd.Timestamp("2018-08-03")
    # the restatement is not used: the year keeps the number public at its 10-K
    assert a.loc[fy17, "net_income"] == 21_204e6 and a.loc[fy17, "filed"] == pd.Timestamp("2017-08-02")
    # a value whose only source is a later filing's comparative column was not public at the row's filing: left out
    assert np.isnan(q.loc[pd.Timestamp("2016-12-31"), "operating_cash_flow"])
    # every row carries the date it became public, never before its period ended
    assert (pd.to_datetime(q["filed"]) > q.index).all() and (pd.to_datetime(a["filed"]) > a.index).all()
    assert h.has_filing_dates and "cover_shares" in a


def test_statements_are_point_in_time_under_truncation():
    """A later filing (a restatement, a comparative column) never changes what was known earlier: building from the
    facts filed up to D gives exactly the rows public at D of the full build."""
    facts = edgar.company_facts(ticker="AAPL", source="sample")
    full = pit.build_statements(facts)
    for cut in ("2012-12-31", "2016-06-30", "2020-03-31"):
        part = pit.build_statements(facts[facts.filed <= cut])
        for name in ("annual", "quarterly"):
            known = pit.as_of(getattr(full, name), cut)
            got = getattr(part, name)
            cols = [c for c in known.columns if c in got.columns and c != "operating_income_is_proxy"]
            pd.testing.assert_frame_equal(known[cols], got.loc[known.index, cols], check_dtype=False)


def test_value_first_public_wins_across_concepts():
    """JNJ reported equity under 'including non-controlling interests' in its 10-K and under the plain concept only
    in later 10-Qs; the annual row takes the value that was public at the 10-K."""
    h = pit.build_statements(edgar.company_facts(ticker="JNJ", source="sample"))
    a = h.annual
    assert a["equity"].tail(5).notna().all()


def test_parse_submissions_and_earnings_dates():
    payload = json.loads((FIX / "msft_submissions_2017_2018.json").read_text())
    prof, f = edgar.parse_submissions(payload)
    assert prof["cik"] == 789019 and prof["sic"] == "7372" and prof["fiscalYearEnd"] == "0630"
    assert {"10-K", "10-Q", "8-K"} >= set(f["form"]) and f["filed"].is_monotonic_increasing
    dates = edgar.earnings_release_dates(f)
    assert len(dates) >= 6                                          # quarterly earnings 8-Ks under Item 2.02
    assert pd.Timestamp("2017-07-20") in set(dates)                 # FY2017 Q4 results


def test_form_index_parser_stops_after_the_wanted_forms():
    lines = (FIX / "form_idx_2015q2_snippet.idx").read_text().splitlines()
    df = edgar.parse_form_index_lines(lines, edgar.DELISTING_FORMS)
    assert set(df["form"]) <= set(edgar.DELISTING_FORMS) and len(df) >= 6
    assert df["cik"].dtype.kind == "i" and df["filed"].notna().all() and df["accn"].notna().all()
    assert (df["filed"].dt.to_period("Q") == pd.Period("2015Q2")).all()


def test_bulk_statement_data_set_parser():
    blob = (FIX / "fsds_2017q3_msft_subset.zip").read_bytes()
    f = edgar.parse_fsds_zip(blob)
    rev = f[(f.concept == "SalesRevenueNet") & (f.end == "2017-06-30")].set_index("start")["val"]
    assert rev[pd.Timestamp("2016-07-01")] == 89_950e6 and rev[pd.Timestamp("2017-04-01")] == 23_317e6
    assert (f["filed"] == pd.Timestamp("2017-08-02")).all() and (f["form"] == "10-K").all()
    h = pit.build_statements(f)
    assert h.annual.loc[pd.Timestamp("2017-06-30"), "revenue"] == 89_950e6
    assert edgar.parse_fsds_zip(blob, ciks=[1]).empty


def test_http_rules(monkeypatch):
    monkeypatch.delenv("LIGHTHOUSE_SEC_EMAIL", raising=False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(http.ConfigurationError):
        http.sec_user_agent()
    monkeypatch.setenv("LIGHTHOUSE_SEC_EMAIL", "analyst@example.com")
    assert http.sec_user_agent().endswith("analyst@example.com")
    assert http.policy_for("https://data.sec.gov/x").rate <= 10
    assert http.policy_for("https://fred.stlouisfed.org/x").user_agent is None    # FRED keeps the library default
    with pytest.raises(DataUnavailable):                                        # offline: never reaches the network
        http.fetch("https://data.sec.gov/submissions/CIK0000789019.json")


def test_http_retries_on_throttling(monkeypatch):
    calls = []

    class R:
        def __init__(self, code, text=""):
            self.status_code, self.text, self.headers = code, text, {}

        def close(self):
            pass

    class S:
        def request(self, *a, **k):
            calls.append(k["headers"].get("User-Agent"))
            return R(403, "Request Rate Threshold Exceeded") if len(calls) < 3 else R(200, "ok")

    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    monkeypatch.setenv("LIGHTHOUSE_SEC_EMAIL", "analyst@example.com")
    monkeypatch.setattr(http, "_session", lambda: S())
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.fetch("https://data.sec.gov/x").text == "ok" and len(calls) == 3
    assert all("analyst@example.com" in c for c in calls)
    calls.clear()

    class Bad(S):
        def request(self, *a, **k):
            calls.append(1)
            return R(404, "nope")
    monkeypatch.setattr(http, "_session", lambda: Bad())
    with pytest.raises(DataUnavailable):
        http.fetch("https://data.sec.gov/y")
    assert len(calls) == 1                                    # a 404 is not retried


def test_pit_scores_use_filing_dates_and_are_causal():
    from lighthouse import analyze, InvestorProfile
    r = analyze("AAPL", InvestorProfile(), source="sample", years=40, run_backtest_=False)
    a = r.pit_statements.annual
    pit_s = r.pit_scores
    first_filed = pd.to_datetime(a["filed"]).min()
    assert pit_s[pit_s.index <= first_filed].isna().all()            # nothing before the first filing is known
    assert pit_s[pit_s.index > first_filed + pd.Timedelta(days=5)].notna().any()
    # removing every statement filed after D leaves every score up to D unchanged
    for cut in ("2014-03-31", "2019-06-28"):
        trunc = StatementHistory(annual=a[pd.to_datetime(a["filed"]) <= cut], source="test")
        s2 = point_in_time_scores(r.prices, trunc, "Technology", r.params, actions=r.actions)
        s1 = point_in_time_scores(r.prices, r.pit_statements, "Technology", r.params, actions=r.actions)
        m = s1.index <= pd.Timestamp(cut)
        pd.testing.assert_series_equal(s1[m], s2[m])


def test_pe_uses_eps_in_the_price_series_units():
    """AAPL split 7:1 (2014) and 4:1 (2020) after its FY2012 10-K; the sample's prices are split-adjusted, so the as-
    reported EPS of $44.15 must be divided by 28. P/E in January 2013 was about 12."""
    from lighthouse import total_return as tr
    facts = edgar.company_facts(ticker="AAPL", source="sample")
    a = pit.build_statements(facts).annual
    row = a.loc[pd.Timestamp("2012-09-29")]
    acts = load_actions("AAPL", source="sample")
    f = tr.split_factor_after([row["filed"]], acts, "2022-12-28")[0]
    assert row["eps"] == pytest.approx(44.15) and f == pytest.approx(28.0)
    px = pd.read_csv("data/sample/prices/AAPL.csv", index_col=0, parse_dates=True)["Close"]
    po = tr.price_only_close(px, tr.to_units_of(acts, px.index[-1]))
    pe = po[po.index >= "2013-01-02"].iloc[0] / (row["eps"] / f)
    assert 10 < pe < 15


def test_predecessor_ciks_cover_a_reorganised_registrant():
    assert 34088 in edgar.PREDECESSOR_CIKS[2115436]             # ExxonMobil Holdings Corp <- Exxon Mobil Corp
    h = pit.build_statements(edgar.company_facts(ticker="XOM", source="sample"))
    assert h.annual.index.min() < pd.Timestamp("2010-01-01") and len(h.annual) > 15
