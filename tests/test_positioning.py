"""Positioning signals (lighthouse/research/positioning.py) on recorded excerpts of the real sources (tests/fixtures/
positioning, recorded October 2026). Every row is copied unchanged from the downloaded file; only rows were dropped:

* FINRA Reg SHO daily short-sale volume: the consolidated files (CNMS) of the last five sessions of December 2023 and of
  2 January 2024, and the per-facility files of 2 January 2015 (FNSQ, FNYX, FNRA; FNQC did not exist yet and FINRA
  answers 403), 21 symbols each; the trailer line (the record count) is rewritten to the rows kept.
* The SEC's Insider Transactions Data Set for 2020 Q1 (form345.zip): all eight tables with their real headers, only the
  rows of 24 filings by insiders of Exxon Mobil, Chevron and JPMorgan.
* FINRA consolidated short interest (API, CSV) at the 30 November and 29 December 2023 settlement dates, 14 rows each.

The network is never touched: `positioning._get` is replaced by a function that serves these files, and the cache
points at a fresh temporary folder."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lighthouse import data, store
from lighthouse.data import DataUnavailable
from lighthouse.research import positioning as P

FIX = Path(__file__).parent / "fixtures" / "positioning"
TICKERS = ["AAPL", "AMD", "BRK-B", "BF-B", "MSFT", "XOM", "ZZZZ"]


class _Resp:
    def __init__(self, body: bytes, status: int = 200):
        self.content, self.status_code = body, status
        self.text = body.decode("utf-8")


def _serve(monkeypatch, tmp_path, until=None):
    """Answer FINRA and SEC URLs from the fixtures, as the hosts do: a missing daily file is HTTP 403 (DataUnavailable),
    a date without short interest is HTTP 204 with no body. `until`: pretend nothing after this date exists yet."""
    calls = []
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    cut = pd.Timestamp(until) if until is not None else None

    def fake(url, **kw):
        body = kw.get("json_body")
        calls.append(url if body is None else (url, body["compareFilters"][0]["fieldValue"], body["offset"]))
        if url == P.FINRA_SI_API:
            day = kw["json_body"]["compareFilters"][0]["fieldValue"]
            f = FIX / f"finra_si_{day.replace('-', '')}_excerpt.csv"
            if kw["json_body"]["offset"] or not f.exists() or (cut is not None and pd.Timestamp(day) > cut):
                return _Resp(b"", 204)
            return _Resp(f.read_bytes())
        name = url.rsplit("/", 1)[-1]
        f = FIX / name
        if not f.exists() or (cut is not None and pd.Timestamp(name[-12:-4]) > cut):
            raise DataUnavailable(f"HTTP 403 from cdn.finra.org for {url}: <Error><Code>AccessDenied</Code>")
        return _Resp(f.read_bytes())

    monkeypatch.setattr(P, "_get", fake)
    return calls


# --------------------------------------------------------------------------- #
# FINRA daily short-sale volume
# --------------------------------------------------------------------------- #
def test_short_volume_day_reads_the_consolidated_file(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, tmp_path)
    sv = P.short_volume_day("2024-01-02")
    assert calls == [P.FINRA_SHVOL.format(name="CNMS", date="20240102")]      # one file since August 2018
    assert len(sv) == 21 and sv.index.notna().all()                            # the record-count trailer is no symbol
    assert sv.loc["AAPL", "ShortVolume"] == 9_684_282 and sv.loc["AAPL", "TotalVolume"] == 30_746_748
    # FINRA writes class shares 'BRK/B': they must reach the panel's 'BRK-B' (they were missed before October 2026)
    assert sv.loc["BRK-B", "ShortVolume"] == 305_959 and sv.loc["BF-B", "TotalVolume"] > 0
    assert "ACHR-WS" in sv.index and not any("/" in s for s in sv.index)
    again = P.short_volume_day("2024-01-02")                                   # cached: no second download
    assert len(calls) == 1 and again.equals(sv)


def test_a_session_cached_by_the_old_parser_is_normalised_on_read(monkeypatch, tmp_path):
    # the old short_volume_day summed by raw symbol and renamed '.' to '-' afterwards: 'XYZ.A' and 'XYZ-A' became two
    # rows labelled 'XYZ-A' (reindexing then failed with "duplicate labels"), and 'BRK/B' was kept as written.
    # Here the 29 December 2023 AAPL row is split in two rows under one label, as such a file holds them.
    calls = _serve(monkeypatch, tmp_path)
    old = pd.DataFrame({"ShortVolume": [4_000_000, 914_869, 278_574, 217_379],
                        "TotalVolume": [10_000_000, 4_012_980, 925_693, 323_293]},
                       index=pd.Index(["AAPL", "AAPL", "BRK/B", "BF/B"], name="Symbol"))
    path = store.write_frame("finra_shvol", "20231229", old)
    before = path.read_bytes()
    sv = P.short_volume_day("2023-12-29")
    assert calls == []                                                          # read from the cache, not downloaded
    assert sv.index.is_unique and list(sv.index) == ["AAPL", "BF-B", "BRK-B"]
    assert sv.loc["AAPL"].tolist() == [4_914_869, 14_012_980]                   # the file's AAPL row
    assert path.read_bytes() == before                                          # the cache file is not rewritten
    sig = P.short_volume_signal(pd.DatetimeIndex(["2023-12-29"]), ["AAPL", "BRK-B"])
    assert np.isfinite(sig.loc["2023-12-29", "BRK-B"])                          # no "duplicate labels" error


def test_short_volume_day_before_august_2018_sums_the_facilities(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, tmp_path)
    sv = P.short_volume_day("2015-01-02")
    asked = [c.rsplit("/", 1)[-1][:4] for c in calls]
    assert asked == ["FNSQ", "FNYX", "FNQC", "FNRA"]                           # FNQC (403) is skipped
    # AAPL on the Nasdaq Carteret TRF, the NYSE TRF and the ADF
    assert sv.loc["AAPL", "ShortVolume"] == 4_537_230 + 1_120_622 + 196_866
    assert sv.loc["AAPL", "TotalVolume"] == 17_474_580 + 2_937_726 + 533_853
    assert sv.loc["BRK-B", "ShortVolume"] == 301_100 + 13_140 + 7_756
    assert sv.loc["BF-A", "TotalVolume"] == 927                                # only on one facility's file
    # a session with no file at all (the host is down, or offline) is None and is not cached
    assert P.short_volume_day("2015-01-05") is None
    assert not (tmp_path / "finra_shvol" / "20150105.parquet").exists()


def test_short_volume_signal_uses_the_last_five_sessions_and_is_causal(monkeypatch, tmp_path):
    dates = pd.DatetimeIndex(["2023-12-29", "2024-01-31"])
    calls = _serve(monkeypatch, tmp_path / "full")
    full = P.short_volume_signal(dates, TICKERS)
    days = sorted({c.rsplit("shvol", 1)[-1][:8] for c in calls})
    # December's five sessions skip Christmas Day; January's last five (25-31) are not in the excerpt
    assert days[:5] == ["20231222", "20231226", "20231227", "20231228", "20231229"]
    assert "20231225" not in days and "20231221" not in days and "20240102" not in days
    short = 6_472_070 + 3_852_742 + 6_315_121 + 3_891_667 + 4_914_869
    total = 13_573_593 + 11_354_301 + 18_507_950 + 12_447_595 + 14_012_980
    assert full.loc["2023-12-29", "AAPL"] == pytest.approx(-short / total)
    assert np.isfinite(full.loc["2023-12-29", "BRK-B"]) and np.isnan(full.loc["2023-12-29", "ZZZZ"])
    assert full.loc["2024-01-31"].isna().all()                                  # no file for those sessions
    assert (full.loc["2023-12-29"].dropna() < 0).all() and (full.loc["2023-12-29"].dropna() > -1).all()
    # truncation: with nothing published after 29 December the December value is the same
    _serve(monkeypatch, tmp_path / "cut", until="2023-12-29")
    part = P.short_volume_signal(dates, TICKERS)
    pd.testing.assert_series_equal(full.loc["2023-12-29"], part.loc["2023-12-29"])
    # before FINRA's files start (August 2009) the signal is missing, without a download
    calls = _serve(monkeypatch, tmp_path / "early")
    early = P.short_volume_signal(pd.DatetimeIndex(["2009-06-30"]), TICKERS)
    assert early.isna().all().all() and calls == []


# --------------------------------------------------------------------------- #
# SEC Form 4: insider transactions
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trades():
    return P.parse_form345((FIX / "form345_2020q1_excerpt.zip").read_bytes())


def test_parse_form345_keeps_open_market_trades_in_common_stock(trades):
    t = trades
    assert set(t["code"]) == {"P", "S"}                                         # A, D, F, G and M are not trades
    n = t.groupby(["symbol", "code"]).size()
    assert n[("XOM", "P")] == 10 and ("XOM", "S") not in n                     # four buyers, ten purchase rows
    assert n[("CVX", "P")] == 5 and n[("CVX", "S")] == 5
    # JPMorgan: one director bought common stock; an officer's purchase of Series HH preferred stock is not counted
    assert n[("JPM", "P")] == 1 and n[("JPM", "S")] == 2
    assert set(t["issuer_cik"]) == {34088, 93410, 19617}
    hobson = t[(t["symbol"] == "JPM") & (t["code"] == "P")].iloc[0]
    assert hobson["filed"] == pd.Timestamp("2020-01-16") and hobson["owner"] == "0001169896"
    assert hobson["relationship"] == "Director"
    assert hobson["value"] == pytest.approx(14_600 * 136.39)                    # purchases are positive
    wirth = t[(t["symbol"] == "CVX") & (t["code"] == "S") & (t["filed"] == "2020-01-08")]
    assert len(wirth) == 4 and wirth["value"].sum() == pytest.approx(-(6_777 * 121.43 + 26_973 * 120.98
                                                                       + 26_731 * 120.98 + 7_019 * 121.43))


def test_issuer_symbols_as_filed_are_cleaned():
    # values seen in the ISSUERTRADINGSYMBOL field of the 2020 Q1 data set
    raw = pd.Series(["NYSE: SCS", "(CALX)", '"OMEX"', "?NHF?", "LEN, LEN-B", "VIACA,VIAC", "WSO; WSOB", "BRK.A",
                     "N O G", "ASX:CRN", np.nan, "(NYSE:FBC)"])
    exploded = P._issuer_symbols(raw).explode()
    sym = P._norm_symbols(exploded.astype(object))
    by_row = sym.groupby(level=0).apply(list)
    assert by_row[0] == ["SCS"] and by_row[1] == ["CALX"] and by_row[2] == ["OMEX"] and by_row[3] == ["NHF"]
    assert by_row[4] == ["LEN", "LEN-B"] and by_row[5] == ["VIACA", "VIAC"] and by_row[6] == ["WSO", "WSOB"]
    assert by_row[7] == ["BRK-A"]
    assert by_row[8] == ["N O G"]                       # Northern Oil & Gas: never split into N, O (Realty Income), G
    assert by_row[9] == ["ASX:CRN"]                     # a foreign listing never matches a US ticker
    assert pd.isna(by_row[10][0]) and by_row[11] == ["FBC"]


MONTH_ENDS = pd.DatetimeIndex(["2019-12-31", "2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30", "2020-05-29",
                               "2020-06-30"])


def test_insider_signals_count_distinct_buyers_by_filing_date_over_90_days(trades):
    tick = ["XOM", "CVX", "JPM", "MSFT"]
    s = P.insider_signals(trades, MONTH_ENDS, tick)
    b = s["insider_buyers"]
    assert b.loc["2019-12-31"].isna().all()                                     # before the first filing in the data
    assert b.loc["2020-01-31"].tolist() == [0, 0, 1, 0]                         # Hobson (JPM); Wirth (CVX) only sold
    assert b.loc["2020-02-28"].tolist() == [1, 1, 1, 0]                         # Woods (XOM), Frank (CVX, 3 filings)
    assert b.loc["2020-03-31"].tolist() == [4, 2, 1, 0]                         # + Weldon, Duffin, Swiger; Moorman
    assert b.loc["2020-04-30"].tolist() == [4, 2, 0, 0]                         # Hobson (16 Jan) left the window
    assert b.loc["2020-05-29"].tolist() == [3, 2, 0, 0]                         # Woods (11 Feb) left it
    assert b.loc["2020-06-30"].tolist() == [0, 0, 0, 0]
    # the window is (d - 90 days, d], by filing date: Woods filed on 11 February 2020
    edge = pd.DatetimeIndex([pd.Timestamp("2020-02-11") + pd.Timedelta(days=89),
                             pd.Timestamp("2020-02-11") + pd.Timedelta(days=90)])
    e = P.insider_signals(trades, edge, ["XOM"])["insider_buyers"]["XOM"]
    assert e.iloc[0] == 3 + 1 and e.iloc[1] == 3
    # Weldon bought on 6 and 9 March and filed on 13 March: he counts from the filing date, not the trade date
    mid = P.insider_signals(trades, pd.DatetimeIndex(["2020-03-12", "2020-03-13"]), ["XOM"])["insider_buyers"]["XOM"]
    assert mid.tolist() == [1, 2]
    # without market values the net-purchase signal is missing rather than a raw dollar amount
    assert s["insider_net"].isna().all().all()


def test_insider_net_is_net_purchases_over_market_value(trades):
    tick = ["XOM", "CVX", "JPM"]
    cap = pd.DataFrame(1e9, index=MONTH_ENDS, columns=tick)
    cap.loc["2020-03-31", "JPM"] = 0.0                                          # a missing market value
    net = P.insider_signals(trades, MONTH_ENDS, tick, cap)["insider_net"]
    woods = 475 * 59.86 + 500 * 59.85 + 466 * 59.86 + 466 * 59.89 + 485 * 59.86 + 466 * 59.87
    xom = woods + 4_590.98 * 43.56 + 4_180 * 48.09 + 30_000 * 36.41 + 30_000 * 33.80
    assert net.loc["2020-03-31", "XOM"] == pytest.approx(xom / 1e9)
    cvx_buys = 500 * 109.85 + 400 * 104.61 + 500 * 91.79 + 500 * 95.98 + 6_551 * 91.43
    cvx_sales = 6_777 * 121.43 + 26_973 * 120.98 + 26_731 * 120.98 + 7_019 * 121.43 + 4_900 * 111.71
    assert net.loc["2020-03-31", "CVX"] == pytest.approx((cvx_buys - cvx_sales) / 1e9)
    assert np.isnan(net.loc["2020-03-31", "JPM"])
    assert net.loc["2020-01-31", "JPM"] == pytest.approx((14_600 * 136.39 - 62_085 * 131.94) / 1e9)


@pytest.mark.parametrize("cut", ["2020-01-31", "2020-02-28", "2020-03-13"])
def test_insider_signals_are_causal(trades, cut):
    tick = ["XOM", "CVX", "JPM"]
    cap = pd.DataFrame(1e9, index=MONTH_ENDS, columns=tick)
    full = P.insider_signals(trades, MONTH_ENDS, tick, cap)
    part = P.insider_signals(trades[trades["filed"] <= cut], MONTH_ENDS, tick, cap)
    for k in full:
        pd.testing.assert_frame_equal(full[k].loc[:cut], part[k].loc[:cut], obj=k)


# --------------------------------------------------------------------------- #
# FINRA short interest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("month, settles", [("2023-11-15", "2023-11-30"), ("2023-12-29", "2023-12-29"),
                                            ("2024-02-10", "2024-02-29"), ("2024-03-01", "2024-03-28"),
                                            ("2022-12-31", "2022-12-30")])
def test_settlement_dates(month, settles):
    # the last trading day of the month: 30-31 December 2023 is a weekend, 29 March 2024 Good Friday
    assert P.settlement_dates(pd.Timestamp(month)) == pd.Timestamp(settles)


def test_short_interest_on_parses_and_caches(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, tmp_path)
    si = P.short_interest_on("2023-12-29")
    assert calls == [(P.FINRA_SI_API, "2023-12-29", 0)]                         # 14 rows < 5,000: one page
    assert list(si.columns) == ["short", "adv", "days_to_cover"] and len(si) == 14
    assert si.loc["AAPL"].tolist() == [108_220_157, 42_416_650, 2.55]
    assert si.loc["AMD", "days_to_cover"] == 1.0                                # FINRA floors 0.95 at 1.00
    assert si.loc["BRKB", "short"] == 7_114_791                                 # no class separator in this source
    P.short_interest_on("2023-12-29")
    assert len(calls) == 1
    # a date that is not a settlement date: HTTP 204, an empty table, cached once it is old
    assert P.short_interest_on("2023-12-28").empty
    assert (tmp_path / "finra_si" / "2023-12-28.parquet").exists()
    # ... but not for a date FINRA may not have published yet
    soon = pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
    assert P.short_interest_on(soon).empty
    assert not (tmp_path / "finra_si" / f"{soon.date()}.parquet").exists()


def test_days_to_cover_is_used_a_month_later_and_is_causal(monkeypatch, tmp_path):
    dates = pd.DatetimeIndex(["2023-11-30", "2023-12-29", "2024-01-31"])
    tick = ["AAPL", "AMD", "BRK-B", "BF-B", "BFA", "BFACU", "MSFT"]
    _serve(monkeypatch, tmp_path / "full")
    full = P.days_to_cover_signal(dates, tick)
    assert full.loc["2023-11-30"].isna().all()                                  # nothing published before it
    # December's value is November's settlement (published in December); January's is December's
    assert full.loc["2023-12-29", "AAPL"] == pytest.approx(-110_653_413 / 42_011_310)
    assert full.loc["2024-01-31", "AAPL"] == pytest.approx(-108_220_157 / 42_416_650)
    # short / ADV, not FINRA's floored figure: AMD's 0.95 is not tied with every other stock under 1.00
    assert full.loc["2024-01-31", "AMD"] == pytest.approx(-49_153_022 / 51_851_451)
    assert full.loc["2024-01-31", "BRK-B"] == pytest.approx(-7_114_791 / 3_020_750)   # matched as 'BRKB'
    assert full.loc["2024-01-31", "BF-B"] == pytest.approx(-7_237_812 / 1_533_360)
    assert np.isnan(full.loc["2024-01-31", "BFACU"])                             # zero volume: no ratio (not 999.99)
    assert np.isnan(full.loc["2024-01-31", "MSFT"])                              # not in the excerpt
    # truncation: run on 29 December, when FINRA had published the settlements up to mid-December only (it
    # publishes ~7 business days after each one): December's value is unchanged
    _serve(monkeypatch, tmp_path / "cut", until="2023-12-15")
    part = P.days_to_cover_signal(dates, tick)
    pd.testing.assert_frame_equal(full.loc[:"2023-12-29"], part.loc[:"2023-12-29"])
    # before the start date nothing is requested and nothing is known
    calls = _serve(monkeypatch, tmp_path / "early")
    early = P.days_to_cover_signal(pd.DatetimeIndex(["2017-11-30", "2017-12-29"]), tick)
    assert early.isna().all().all() and calls == []
