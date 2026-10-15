"""FINRA short volume (lighthouse/research/positioning.py) on recorded excerpts of the real sources (tests/fixtures/
positioning, recorded October 2026). Every row is copied unchanged from the downloaded file; only rows were dropped:

* FINRA Reg SHO daily short-sale volume: the consolidated files (CNMS) of the last five sessions of December 2023 and of
  2 January 2024, and the per-facility files of 2 January 2015 (FNSQ, FNYX, FNRA; FNQC did not exist yet and FINRA
  answers 403), 21 symbols each; the trailer line (the record count) is rewritten to the rows kept.

The network is never touched: `positioning._get` is replaced by a function that serves these files, and the cache
points at a fresh temporary folder."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lighthouse import data, store
from lighthouse.data import DataUnavailable
from lighthouse.http import HTTPStatusError
from lighthouse.research import positioning as P

FIX = Path(__file__).parent / "fixtures" / "positioning"
TICKERS = ["AAPL", "AMD", "BRK-B", "BF-B", "MSFT", "XOM", "ZZZZ"]


class _Resp:
    def __init__(self, body: bytes, status: int = 200):
        self.content, self.status_code = body, status
        self.text = body.decode("utf-8")


def _serve(monkeypatch, tmp_path, until=None):
    """Answer FINRA URLs from the fixtures, as the host does: a missing daily file is HTTP 403 (HTTPStatusError, as
    `http.fetch` raises it). `until`: pretend nothing after this date exists yet."""
    calls = []
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    cut = pd.Timestamp(until) if until is not None else None

    def fake(url, **kw):
        calls.append(url)
        name = url.rsplit("/", 1)[-1]
        f = FIX / name
        if not f.exists() or (cut is not None and pd.Timestamp(name[-12:-4]) > cut):
            raise HTTPStatusError(f"HTTP 403 from cdn.finra.org for {url}: <Error><Code>AccessDenied</Code>", 403)
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


def _failing_once(monkeypatch, failures: dict):
    """Wrap the fixture server: the first request for each named file fails with the given error, later ones are
    served as usual."""
    served = P._get

    def fake(url, **kw):
        name = url.rsplit("/", 1)[-1]
        if name in failures:
            raise failures.pop(name)
        return served(url, **kw)
    monkeypatch.setattr(P, "_get", fake)


def test_a_session_is_cached_only_when_every_facility_answered(monkeypatch, tmp_path):
    # 2 January 2015: FNSQ, FNYX and FNRA published; FNQC did not exist yet and FINRA answers 403 ("no such file").
    # Before October 2026 a timeout on one facility (a plain DataUnavailable: no answer through every retry) was
    # skipped like a 403 and the session cached without it, for good.
    calls = _serve(monkeypatch, tmp_path)
    timeout = DataUnavailable("cdn.finra.org: gave up after 5 attempts (ReadTimeout: read timed out) for the file")
    _failing_once(monkeypatch, {"FNYXshvol20150102.txt": timeout})
    part = P.short_volume_day("2015-01-02")
    assert part.loc["AAPL", "ShortVolume"] == 4_537_230 + 196_866              # FNSQ and FNRA: usable in this run ...
    assert store.read_frame("finra_shvol", "20150102") is None                   # ... but not cached
    full = P.short_volume_day("2015-01-02")                                      # asked again: every facility answers
    assert full.loc["AAPL", "ShortVolume"] == 4_537_230 + 1_120_622 + 196_866
    assert store.read_json("finra_shvol", "20150102") == {
        "facilities": {"FNSQ": "published", "FNYX": "published", "FNQC": "not published", "FNRA": "published"},
        "rows": len(full)}
    n = len(calls)
    assert P.short_volume_day("2015-01-02").equals(full) and len(calls) == n       # now from the cache
    # an HTTP error other than "no such file" (401 here) is a failure too, not an absent facility
    _serve(monkeypatch, tmp_path / "auth")
    _failing_once(monkeypatch, {"FNRAshvol20150102.txt": HTTPStatusError("HTTP 401 from cdn.finra.org", 401)})
    assert P.short_volume_day("2015-01-02").loc["AAPL", "ShortVolume"] == 4_537_230 + 1_120_622
    assert store.read_frame("finra_shvol", "20150102") is None


def test_a_signal_built_on_a_partial_session_is_not_cached(monkeypatch, tmp_path):
    from types import SimpleNamespace
    _serve(monkeypatch, tmp_path)
    _failing_once(monkeypatch, {"FNSQshvol20150102.txt": DataUnavailable("cdn.finra.org: gave up after 5 attempts")})
    # 9 January 2015: the window is 2 and 5-8 January; the excerpt holds 2 January only (the rest answer 403)
    panel = SimpleNamespace(dates=pd.DatetimeIndex(["2015-01-09"]), prices=pd.DataFrame(columns=["AAPL", "MSFT"]))
    first = P.monthly(None, panel)["short_volume"]
    assert first.attrs["incomplete_sessions"] == ["2015-01-02"]
    assert first.loc["2015-01-09", "AAPL"] == pytest.approx(-(1_120_622 + 196_866) / (2_937_726 + 533_853))
    assert store.read_frame("research_signals", "short_volume") is None          # not cached as complete
    assert store.read_json("research", "short_volume_incomplete") == ["2015-01-02"]
    again = P.monthly(None, panel)["short_volume"]                               # the next run downloads it again
    assert again.attrs["incomplete_sessions"] == [] and store.read_json("research", "short_volume_incomplete") == []
    assert again.loc["2015-01-09", "AAPL"] == pytest.approx(-(4_537_230 + 1_120_622 + 196_866)
                                                            / (17_474_580 + 2_937_726 + 533_853))
    pd.testing.assert_frame_equal(store.read_frame("research_signals", "short_volume"), again, check_freq=False,
                                  check_index_type=False)


def test_an_early_file_without_a_header_line_is_read(monkeypatch, tmp_path):
    # 22 November 2010: the Nasdaq file starts with its first row (no header), the NYSE file has one, the ADF file
    # has a header and no rows, and the Chicago facility did not exist yet (403)
    _serve(monkeypatch, tmp_path)
    day = P.short_volume_day("2010-11-22")
    assert day.loc["AAPL", "ShortVolume"] == 2_055_344 + 42_559
    assert day.loc["AAPL", "TotalVolume"] == 4_611_502 + 92_786
    assert day.loc["MSFT", "TotalVolume"] == 11_483_645 + 3_055_503
    assert len(day) == 5 and "5" not in day.index                                # the trailer line is no symbol
    with pytest.raises(ValueError):
        P._read_shvol("20101122|AAPL|1\n")


def test_short_volume_signal_uses_the_last_five_sessions_and_is_causal(monkeypatch, tmp_path):
    dates = pd.DatetimeIndex(["2023-12-29", "2024-01-31"])
    calls = _serve(monkeypatch, tmp_path / "full")
    full = P.short_volume_signal(dates, TICKERS)
    days = sorted({c.rsplit("shvol", 1)[-1][:8] for c in calls})
    # December's signal reads the five sessions before the month's last (29 December: FINRA posts that file after
    # the close), skipping Christmas Day; 21 December is not in the excerpt (403), so four sessions are summed.
    # January's sessions (24-30) are not in the excerpt either.
    assert days[:5] == ["20231221", "20231222", "20231226", "20231227", "20231228"]
    assert "20231225" not in days and "20231220" not in days and "20231229" not in days and "20240102" not in days
    short = 6_472_070 + 3_852_742 + 6_315_121 + 3_891_667
    total = 13_573_593 + 11_354_301 + 18_507_950 + 12_447_595
    assert full.loc["2023-12-29", "AAPL"] == pytest.approx(-short / total)
    assert np.isfinite(full.loc["2023-12-29", "BRK-B"]) and np.isnan(full.loc["2023-12-29", "ZZZZ"])
    assert full.loc["2024-01-31"].isna().all()                                  # no file for those sessions
    assert (full.loc["2023-12-29"].dropna() < 0).all() and (full.loc["2023-12-29"].dropna() > -1).all()
    # truncation: with nothing published after 28 December (the month-end session's file not out yet) the December
    # value is the same
    _serve(monkeypatch, tmp_path / "cut", until="2023-12-28")
    part = P.short_volume_signal(dates, TICKERS)
    pd.testing.assert_series_equal(full.loc["2023-12-29"], part.loc["2023-12-29"])
    # before FINRA's files start (August 2009) the signal is missing, without a download
    calls = _serve(monkeypatch, tmp_path / "early")
    early = P.short_volume_signal(pd.DatetimeIndex(["2009-06-30"]), TICKERS)
    assert early.isna().all().all() and calls == []


def test_the_short_volume_window_skips_one_off_closures(monkeypatch, tmp_path):
    # Hurricane Sandy closed the market on 29 and 30 October 2012. October's signal reads the five sessions before
    # the 31st: 22-26 October. Before October 2026 the calendar knew only the regular holidays, asked for the two
    # closed days (no file: 403) and summed three sessions. Synthetic files (one facility, two symbols), one a day.
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    vols = {"20121022": (100, 1_000), "20121023": (200, 1_000), "20121024": (300, 1_000), "20121025": (400, 1_000),
            "20121026": (500, 1_000), "20121019": (900, 1_000)}
    calls = []

    def fake(url, **kw):
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        day = name[-12:-4]
        if not name.startswith("FNSQ") or day not in vols:
            raise HTTPStatusError(f"HTTP 403 from cdn.finra.org for {url}: <Error><Code>AccessDenied</Code>", 403)
        short, total = vols[day]
        body = (f"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n{day}|AAA|{short}|0|{total}|Q\n"
                f"{day}|BBB|1|0|10|Q\n2\n")
        return _Resp(body.encode())
    monkeypatch.setattr(P, "_get", fake)
    assert not data.is_trading_day("2012-10-29") and not data.is_trading_day("2012-10-30")
    sig = P.short_volume_signal(pd.DatetimeIndex(["2012-10-31"]), ["AAA", "BBB"])
    asked = sorted({c[-12:-4] for c in calls})
    assert asked == ["20121022", "20121023", "20121024", "20121025", "20121026"]   # five real sessions, no closure
    assert sig.loc["2012-10-31", "AAA"] == pytest.approx(-(100 + 200 + 300 + 400 + 500) / 5_000)
