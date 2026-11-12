"""The faster SEC readers give the same numbers as the slow ones: the statement cache, the concurrent and incremental
10-K reader, the compressed document cache, the bulk-archive warmers and the fog index from word counts."""
import io
import json
import re
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lighthouse import data, edgar, pit, store, text
from lighthouse.research import event_signals as ev

FIX = Path(__file__).parent / "fixtures" / "edgar"
LM_CSV = ("Word,Negative,Positive,Uncertainty,Litigious,Constraining,Strong_Modal,Weak_Modal,Syllables\n"
          "LOSS,2009,0,0,0,0,0,0,1\nUNCERTAIN,0,0,2009,0,0,0,0,3\nDECLINE,2009,0,0,0,0,0,0,2\nMAY,0,0,0,0,0,0,2009,1\n")


@pytest.fixture
def lm():
    return text.parse_lm_csv(io.StringIO(LM_CSV))


# --------------------------------------------------------------------------- #
# Statements: built once, read back identical, rebuilt when the facts or the code change
# --------------------------------------------------------------------------- #
def test_cached_statements_are_the_built_statements(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    facts = edgar.company_facts(ticker="MSFT", source="sample")
    built = pit.build_statements(facts)
    calls = []
    real = pit.build_statements
    monkeypatch.setattr(pit, "build_statements", lambda f, *a, **k: calls.append(1) or real(f, *a, **k))
    first = pit.cached_statements("MSFT", facts)
    again = pit.cached_statements("MSFT", facts)
    assert len(calls) == 1                                          # the second read came from the disk
    for h in (first, again):
        pd.testing.assert_frame_equal(h.annual, built.annual)
        pd.testing.assert_frame_equal(h.quarterly, built.quarterly)
    assert again.source == built.source
    # other facts under the same key (a new filing arrived): rebuilt, never the old tables
    fewer = facts[facts["filed"] < facts["filed"].max()].reset_index(drop=True)
    got = pit.cached_statements("MSFT", fewer)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(got.quarterly, real(fewer).quarterly)
    # a change to the module's rules: rebuilt
    pit.cached_statements("MSFT", fewer)
    assert len(calls) == 2
    monkeypatch.setattr(pit, "_code_stamp", lambda: "another-version")
    pit.cached_statements("MSFT", fewer)
    assert len(calls) == 3
    pit.cached_statements("MSFT", fewer, refresh=True)
    assert len(calls) == 4


def test_a_half_written_statement_cache_is_rebuilt(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    facts = edgar.company_facts(ticker="KO", source="sample")
    older = facts[facts["filed"] < "2020-01-01"].reset_index(drop=True)
    pit.cached_statements("KO", older)
    annual_of_older = store.path_for("edgar_statements", "KO_annual").read_bytes()
    want = pit.cached_statements("KO", facts)
    store.path_for("edgar_statements", "KO_annual").write_bytes(annual_of_older)      # one file of each write
    got = pit.cached_statements("KO", facts)
    pd.testing.assert_frame_equal(got.annual, want.annual)


def test_item_values_reads_the_table_or_its_groups():
    facts = edgar.flatten_company_facts(json.loads((FIX / "jpm_companyfacts_2021_2022.json").read_text()))
    orig = pit.original_facts(facts)
    groups = pit.concept_groups(orig)
    for item in pit.LINE_ITEMS + pit.INDUSTRY_ITEMS:
        for a, b in zip(pit.item_values(orig, item), pit.item_values(groups, item)):
            pd.testing.assert_frame_equal(a, b)


def test_quarters_from_year_to_date_values():
    # 3, 6 and 9 months and the year from one start; the 10-Q of the second quarter was filed late
    f = pd.DataFrame({"start": pd.to_datetime(["2020-01-01"] * 4 + ["2021-01-01"]),
                      "end": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31"]),
                      "val": [10.0, 25.0, 45.0, 70.0, 12.0],
                      "filed": pd.to_datetime(["2020-05-01", "2020-11-20", "2020-11-01", "2021-02-15", "2021-05-01"])})
    q, a = pit.flow_values(f)
    assert list(q["val"]) == [10.0, 15.0, 20.0, 25.0, 12.0]
    assert q.loc["2020-06-30", "filed"] == pd.Timestamp("2020-11-20")          # the later of the two filings
    assert q.loc["2020-09-30", "filed"] == pd.Timestamp("2020-11-20")
    assert list(a["val"]) == [70.0]


# --------------------------------------------------------------------------- #
# Documents: fetched several at a time, in order; the cache is compressed
# --------------------------------------------------------------------------- #
def test_fetch_documents_keeps_the_order_and_overlaps_the_waiting(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    live, peak = [0], [0]
    lock = threading.Lock()

    def slow(cik, accn, doc, cache=True):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05 if accn != "a0" else 0.15)               # the first one arrives last
        with lock:
            live[0] -= 1
        return f"{cik}/{accn}/{doc}/{cache}"
    monkeypatch.setattr(edgar, "fetch_document", slow)
    jobs = [(1, f"a{i}", "k.htm") for i in range(8)]
    t = time.perf_counter()
    got = list(edgar.fetch_documents(jobs, threads=3, cache=False))
    took = time.perf_counter() - t
    assert got == [f"1/a{i}/k.htm/False" for i in range(8)]
    assert 2 <= peak[0] <= 3 and took < 0.45                     # one at a time would take 0.5 s


def test_fetch_documents_raises_at_the_failed_document(monkeypatch):
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")

    def fetch(cik, accn, doc, cache=True):
        if accn == "a2":
            raise data.DataUnavailable("gone")
        return accn
    monkeypatch.setattr(edgar, "fetch_document", fetch)
    it = edgar.fetch_documents([(1, f"a{i}", "k.htm") for i in range(5)], threads=2)
    assert [next(it), next(it)] == ["a0", "a1"]
    with pytest.raises(data.DataUnavailable):
        next(it)


def test_the_document_cache_is_compressed_and_reads_older_copies(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    body = "<html>" + "Revenue may decline. " * 5000 + "é</html>"
    calls = []

    class R:
        text = body
    monkeypatch.setattr(edgar, "_get", lambda url, **kw: calls.append(url) or R())
    assert edgar.fetch_document(7, "0001-23-000004", "k.htm") == body
    assert edgar.fetch_document(7, "0001-23-000004", "k.htm") == body and len(calls) == 1
    p = store.path_for("edgar_docs", "7_000123000004_k.htm", "txt.gz")
    assert p.exists() and p.stat().st_size < len(body) / 20
    store.write_json("edgar_docs", "7_000123000005_old.htm", "stored before the cache was compressed")
    assert edgar.fetch_document(7, "0001-23-000005", "old.htm") == "stored before the cache was compressed"
    assert len(calls) == 1
    assert edgar.fetch_document(7, "0001-23-000004", "k.htm", cache=False) == body and len(calls) == 2


# --------------------------------------------------------------------------- #
# 10-K text: only new filings are read
# --------------------------------------------------------------------------- #
DOCS = {"a1": "Item 1A. Risk Factors " + "We expect a loss. " * 300 + " Item 1B. Unresolved Staff Comments",
        "a2": "Item 1A. Risk Factors " + "We expect a loss. Revenue may decline. " * 200 + " Item 1B. Unresolved",
        "a3": "Item 1A. Risk Factors " + "Results are uncertain. Revenue may decline. " * 250 + " Item 1B. Unresolved",
        "a4": "Item 1A. Risk Factors " + "Results are uncertain and may decline. " * 260 + " Item 1B. Unresolved"}
FILINGS = pd.DataFrame({"accn": ["a1", "a2", "x", "a3", "a4"],
                        "filed": pd.to_datetime(["2014-02-14", "2015-02-13", "2015-05-01", "2016-02-29", "2017-02-10"]),
                        "form": ["10-K", "10-K", "10-Q", "10-K", "10-K"],
                        "doc": ["k1.htm", "k2.htm", "q.htm", "k3.htm", "k4.htm"]})


def _serve(monkeypatch, until):
    f = FILINGS[FILINGS["filed"] <= pd.Timestamp(until)]
    monkeypatch.setattr(edgar, "cik_for", lambda t: 1234)
    monkeypatch.setattr(edgar, "company_filings", lambda t, source="auto", refresh=False: ({}, f.copy()))
    fetched = []
    monkeypatch.setattr(edgar, "fetch_document", lambda cik, accn, doc, cache=True: fetched.append(accn) or DOCS[accn])
    return fetched


@pytest.mark.parametrize("threads", [1, 3])
def test_text_history_reads_only_new_filings_and_gives_the_full_run(lm, monkeypatch, tmp_path, threads):
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")                 # the downloads are the stand-ins
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "all")
    _serve(monkeypatch, "2017-12-31")
    full = ev.text_history("ABC", lm, threads=threads)            # every 10-K in one run
    assert list(full["accn"]) == ["a1", "a2", "a3", "a4"]

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "grown")
    fetched = _serve(monkeypatch, "2015-12-31")
    first = ev.text_history("ABC", lm, threads=threads)
    assert sorted(fetched) == ["a1", "a2"] and list(first["accn"]) == ["a1", "a2"]
    del fetched[:]
    assert ev.text_history("ABC", lm, threads=threads).equals(first) and fetched == []      # nothing new: no reading
    fetched = _serve(monkeypatch, "2017-12-31")
    grown = ev.text_history("ABC", lm, threads=threads)
    assert sorted(fetched) == ["a2", "a3", "a4"]                  # the two new ones and the one a3 is compared with
    pd.testing.assert_frame_equal(grown, full)
    del fetched[:]
    again = ev.text_history("ABC", lm, refresh=True, threads=threads)
    assert sorted(fetched) == ["a1", "a2", "a3", "a4"]
    pd.testing.assert_frame_equal(again, full)


def test_text_history_offline_returns_the_cache(lm, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    _serve(monkeypatch, "2015-12-31")
    first = ev.text_history("ABC", lm)
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "1")
    fetched = _serve(monkeypatch, "2017-12-31")
    assert ev.text_history("ABC", lm).equals(first) and fetched == []


# --------------------------------------------------------------------------- #
# Text features from the counts
# --------------------------------------------------------------------------- #
def _slow_fog(txt, lm):
    """The fog index as it was computed word by word."""
    ws = text.tokens(txt)
    if not ws:
        return float("nan")
    sentences = max(len(re.findall(r"[.!?]+(?:\s|$)", txt)), 1)
    complex_share = sum(text._syllables(w, lm) >= 3 for w in ws) / len(ws)
    return 0.4 * (len(ws) / sentences + 100.0 * complex_share)


def test_fog_and_features_from_counts_match_the_word_by_word_values(lm, monkeypatch):
    doc = "<p>" + DOCS["a3"] + " Extraordinary circumstances notwithstanding. Is it so? Yes!</p>"
    assert text.fog(text.html_to_text(doc), lm) == _slow_fog(text.html_to_text(doc), lm)
    assert text.fog(text.html_to_text(doc)) == _slow_fog(text.html_to_text(doc), None)
    assert np.isnan(text.fog("12 34."))
    feats, counts = text.features(doc, lm)
    plain = text.html_to_text(doc)
    assert feats["fog"] == _slow_fog(plain, lm)
    assert counts == text.word_counts(doc) == {"full": Counter(text.tokens(plain)),
                                               "risk": Counter(text.tokens(text.risk_factors(plain)))}
    for k, v in text.tone(text.tokens(plain), lm).items():
        if k != "words":
            assert feats[f"lm_{k}"] == v
    # a filing longer than the fog window: the index reads its first part only, the rest of the features all of it
    monkeypatch.setattr(text, "FOG_CHARS", 500)
    long_feats, _ = text.features(doc, lm)
    assert long_feats["fog"] == _slow_fog(plain[:500], lm) and long_feats["words"] == feats["words"]


# --------------------------------------------------------------------------- #
# The nightly archives fill the same caches the per-company readers use
# --------------------------------------------------------------------------- #
class _Archive:
    def __init__(self, files):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, payload in files.items():
                zf.writestr(name, json.dumps(payload))
        self.blob, self.calls = buf.getvalue(), 0

    def __call__(self, url, **kw):
        self.calls += 1
        blob = self.blob

        class R:
            def iter_content(self, chunk_size=1):
                for i in range(0, len(blob), 4096):
                    yield blob[i:i + 4096]

            def close(self):
                pass
        return R()


def test_warm_company_facts_writes_what_company_facts_writes(monkeypatch, tmp_path):
    payload = json.loads((FIX / "msft_companyfacts_2016_2018.json").read_text())
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "one")
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")

    class One:
        def json(self):
            return payload
    monkeypatch.setattr(edgar, "_get", lambda url, **kw: One())
    one_by_one = edgar.company_facts(cik=789019)

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "bulk")
    srv = _Archive({"CIK0000789019.json": payload})
    monkeypatch.setattr(edgar, "_get", srv)
    assert edgar.warm_company_facts([789019, 789019, 42]) == [789019]          # 42 is not in the archive
    assert srv.calls == 1
    pd.testing.assert_frame_equal(edgar.company_facts(cik=789019), one_by_one)  # a fresh cache hit:
    assert srv.calls == 1                                                       # ... no request
    assert edgar.warm_company_facts([789019]) == [] and srv.calls == 1          # already fresh
    assert edgar.warm_company_facts([42]) == [] and srv.calls == 1              # the day's archive is reused
    assert edgar.warm_company_facts([789019], concepts=["SomeNewConcept"]) == [789019]
    meta = store.read_json("edgar_facts", "0000789019")
    assert "SomeNewConcept" in meta["concepts"] and "NetIncomeLoss" in meta["concepts"]


def test_warm_submissions_writes_what_submissions_writes(monkeypatch, tmp_path):
    payload = json.loads((FIX / "msft_submissions_2017_2018.json").read_text())
    recent = payload["filings"]["recent"]
    n = len(recent["accessionNumber"])
    page = {k: v[n // 2:] for k, v in recent.items()}
    first = dict(payload, filings={"recent": {k: v[:n // 2] for k, v in recent.items()},
                                   "files": [{"name": "CIK0000789019-submissions-001.json"}]})
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    srv = _Archive({"CIK0000789019.json": first, "CIK0000789019-submissions-001.json": page})
    monkeypatch.setattr(edgar, "_get", srv)
    assert edgar.warm_submissions([789019]) == [789019]
    prof, filings = edgar.submissions(789019)
    assert srv.calls == 1                                         # read from the cache the archive filled
    want_prof, want = edgar.parse_submissions(payload)
    assert prof == json.loads(json.dumps(want_prof, default=str)) and len(filings) == len(want) == n
    assert list(filings["accn"]) == list(want["accn"])
    assert edgar.warm_submissions([789019]) == []


def test_a_bulk_download_that_breaks_keeps_the_old_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("LIGHTHOUSE_OFFLINE", "0")
    srv = _Archive({"CIK0000000001.json": {}})
    monkeypatch.setattr(edgar, "_get", srv)
    p = edgar._bulk_archive("x.zip", "https://www.sec.gov/x.zip")
    good = p.read_bytes()

    class Broken:
        def iter_content(self, chunk_size=1):
            yield b"half"
            raise OSError("connection reset")

        def close(self):
            pass
    monkeypatch.setattr(edgar, "_get", lambda url, **kw: Broken())
    assert edgar._bulk_archive("x.zip", "https://www.sec.gov/x.zip", refresh=True) == p and p.read_bytes() == good
    assert [f.name for f in p.parent.iterdir()] == ["x.zip"]      # no half-written file left behind


# --------------------------------------------------------------------------- #
# Polite HTTP: the User-Agent names this version and this repository
# --------------------------------------------------------------------------- #
def test_the_project_user_agent_carries_the_package_version():
    import lighthouse
    from lighthouse import http
    ua = http.policy_for("https://en.wikipedia.org/wiki/S%26P_500").user_agent()
    assert ua == f"Lighthouse/{lighthouse.__version__} (+{http.PROJECT_URL})" and "Fintech-StThomas" in ua
    assert http.policy_for("http://localhost:11434/api") is http.DEFAULT_POLICY
