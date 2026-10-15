"""The single-stock analyst toolkit and the thesis tracker, offline: synthetic companies where the right answer is known,
and the sample pack's recorded SEC filings for MSFT."""
import numpy as np
import pandas as pd
import pytest

from lighthouse import edgar, pit
from lighthouse.analyst import checks, memo as memo_mod, thesis
from lighthouse.analyst.financials import Financials, financials_as_of
from lighthouse.analyst.model import Assumptions, project
from lighthouse.analyst.peers import choose_peers, compare, peer_norms
from lighthouse.analyst.valuation import DiscountRate, dcf, reverse_dcf, value, valuation_bands
from lighthouse.data import load_prices
from lighthouse.total_return import load_actions


def company(price=100.0, shares=1e9, revenue=50e9, margin=0.20, growth=0.08, debt=10e9, cash=5e9, interest=0.5e9,
            ni=None, ocf=None, ticker="SYN", **bs):
    ni = revenue * margin * 0.79 if ni is None else ni
    ttm = {"revenue": revenue, "operating_income": revenue * margin, "net_income": ni,
           "operating_cash_flow": ni * 1.2 if ocf is None else ocf, "capex": revenue * 0.05, "d_and_a": revenue * 0.04,
           "sbc": revenue * 0.01, "interest_expense": interest, "income_tax": revenue * margin * 0.21,
           "pretax_income": revenue * margin, "gross_profit": revenue * 0.5, "dividends_paid": ni * 0.3,
           "buybacks": ni * 0.2}
    prev = {k: v / (1 + growth) for k, v in ttm.items()}
    sheet = {"total_debt": debt, "cash": cash, "equity": 30e9, "total_assets": 80e9, "current_assets": 20e9,
             "current_liabilities": 12e9, "receivables": 6e9, "inventory": 3e9, "accounts_payable": 4e9, "ppe": 20e9,
             "cover_shares": shares}
    sheet.update(bs)
    return Financials(ticker=ticker, as_of=pd.Timestamp("2020-06-30"), price=price, shares=shares, ttm=ttm,
                      ttm_prev=prev, bs=sheet, annual=pd.DataFrame(), quarterly=pd.DataFrame())


RATE = DiscountRate(risk_free=0.02, beta=1.0, equity_premium=0.06, cost_of_equity=0.08, cost_of_debt=0.04,
                    tax_rate=0.21, equity_weight=0.9, wacc=0.08)


def test_projection_balances_every_year():
    f = company()
    p = project(Assumptions.from_financials(f, years=5))
    assert len(p) == 5
    assert np.allclose(p["balance_check"], 0.0, atol=1.0)          # dollars, on a $50bn company
    # stock-based pay is a cost: the valuation's cash flow is lower than the cash view by exactly that amount
    assert np.allclose(p["fcff"] - p["fcff_after_sbc"], p["sbc"])


def test_dcf_falls_with_the_discount_rate_and_rises_with_growth():
    f = company()
    a = Assumptions.from_financials(f, years=5)
    assert dcf(a, f, 0.07).per_share > dcf(a, f, 0.09).per_share
    assert dcf(a.scaled(0.03), f, 0.08).per_share > dcf(a, f, 0.08).per_share
    with pytest.raises(ValueError):
        dcf(a, f, 0.02, terminal_growth=0.025)


def test_terminal_value_reinvests_for_growth_not_last_years_capex():
    f = company()
    a = Assumptions.from_financials(f, years=5)
    rate, g = 0.08, 0.025
    r = dcf(a, f, rate, g)
    p = r.projection
    roic = f.ratios()["roic"]
    assert r.ronic == pytest.approx(max(roic, rate))
    # by hand: NOPAT of year 6 = EBIT of year 5 x (1 + g) x (1 - tax); reinvestment g / RONIC of it
    nopat6 = p["operating_income"].iloc[-1] * (1 + g) * (1 - a.tax_rate)
    tv = nopat6 * (1 - g / r.ronic) / (rate - g)
    disc = (1 + rate) ** -np.arange(1, 6)
    ev = float((p["fcff_after_sbc"].to_numpy() * disc).sum()) + tv * disc[-1]
    assert r.enterprise_value == pytest.approx(ev)
    assert r.per_share == pytest.approx((ev - f.net_debt) / f.shares)
    # a build-out (capex 35% of revenue in the projection years) costs those years' cash, not the terminal value
    heavy = dcf(Assumptions(**{**a.to_dict(), "capex_pct": 0.35}), f, rate, g)
    pv_tv = lambda x: x.terminal_share * x.enterprise_value                     # noqa: E731
    assert pv_tv(heavy) == pytest.approx(pv_tv(r)) and heavy.enterprise_value < r.enterprise_value
    # new capital earns at least its cost: a low-ROIC company's growth adds nothing, and never destroys value
    weak = company(margin=0.02)
    assert dcf(Assumptions.from_financials(weak, years=5), weak, rate, g).ronic == pytest.approx(rate)
    # an explicit RONIC: the higher it is, the less of the profit growth consumes
    assert dcf(a, f, rate, g, ronic=0.30).per_share > dcf(a, f, rate, g, ronic=0.10).per_share
    # a loss in the last year is capitalised as it is (nothing to reinvest)
    loss = Assumptions(**{**a.to_dict(), "operating_margin": [-0.05] * 5})
    pl = project(loss)
    want = pl["operating_income"].iloc[-1] * (1 + g) * (1 - loss.tax_rate) / (rate - g)
    got = dcf(loss, f, rate, g)
    assert got.terminal_share * got.enterprise_value == pytest.approx(want * disc[-1])


def test_reverse_dcf_reproduces_the_price():
    f = company(price=120.0)
    base = Assumptions.from_financials(f, years=5)
    g, m = reverse_dcf(f, base, RATE.wacc)
    assert g is not None and m is not None
    at_g = dcf(Assumptions(**{**base.to_dict(), "growth": [g] * 5, "operating_margin": [base.operating_margin[0]] * 5}),
               f, RATE.wacc)
    assert at_g.per_share == pytest.approx(120.0, rel=1e-4)
    at_m = dcf(Assumptions(**{**base.to_dict(), "operating_margin": [m] * 5}), f, RATE.wacc)
    assert at_m.per_share == pytest.approx(120.0, rel=1e-4)


def test_valuation_cases_order_and_target_is_the_weighted_mean():
    f = company()
    close = pd.Series(100.0, index=pd.bdate_range("2015-01-01", "2020-06-30"))
    v = value(f, close, rate=RATE)
    assert v.cases["bear"].per_share < v.cases["base"].per_share < v.cases["bull"].per_share
    assert v.target == pytest.approx(0.25 * v.cases["bear"].per_share + 0.5 * v.cases["base"].per_share
                                     + 0.25 * v.cases["bull"].per_share)
    assert v.table().loc["base", "weight"] == 0.5


def test_net_debt_counts_short_term_investments():
    f = company(debt=10e9, cash=2e9, short_investments=6e9)
    assert f.liquid_assets == pytest.approx(8e9)
    assert f.net_debt == pytest.approx(2e9)
    g = company(debt=10e9, cash=2e9, cash_and_investments=9e9)     # some companies report only the combined line
    assert g.net_debt == pytest.approx(1e9)


def test_valuation_bands_percentile():
    s = pd.Series(np.linspace(0.01, 0.10, 2600), index=pd.bdate_range("2010-01-01", periods=2600))
    b = valuation_bands({"earnings_yield": s})
    assert b.loc["earnings_yield", "percentile"] == pytest.approx(1.0)      # today is the highest yield: cheapest
    assert b.loc["earnings_yield", "p10"] < b.loc["earnings_yield", "median"] < b.loc["earnings_yield", "p90"]


def test_peers_by_sic_then_size():
    sics = {"A": "7372", "B": "7372", "C": "7372", "D": "7371", "E": "7370", "F": "2834", "G": "7372"}
    caps = {"A": 100.0, "B": 90.0, "C": 10.0, "D": 95.0, "E": 50.0, "F": 100.0, "G": 500.0}
    peers, level = choose_peers("A", sics, caps, n=3, min_peers=3)
    assert level == "4-digit SIC 7372" and peers == ["B", "G", "C"]           # closest in log market value first
    peers, level = choose_peers("A", sics, caps, n=3, min_peers=4)
    assert level.startswith("3-digit") and "F" not in peers and peers[0] in ("B", "D")
    assert choose_peers("X", sics, caps) == ([], "no SIC code or market value")


def test_peer_comparison_premium_and_norms():
    target = company(price=150.0, ticker="T")
    peers = {f"P{i}": company(price=80.0 + 5 * i, ticker=f"P{i}", growth=0.04 + 0.01 * i) for i in range(6)}
    cmp = compare(target, peers, "4-digit SIC 7372")
    assert cmp.premium["pe"] > 0                                   # same earnings, higher price
    assert "premium" in cmp.explanation[0]
    assert cmp.predicted_ev_ebitda is not None
    norms = peer_norms(cmp)
    assert norms["pe"] == pytest.approx(cmp.table.loc["peer median", "pe"])
    assert "6 peers" in norms["source"]


def test_fundamental_view_scores_the_memo_numbers_against_peer_medians():
    target = company(price=150.0, ticker="T")
    dear = {f"P{i}": company(price=400.0 + 20 * i, ticker=f"P{i}") for i in range(6)}     # same business, dearer
    cmp = compare(target, dear, "4-digit SIC 7372")
    snap = memo_mod.fundamental_snapshot(target, "Technology")
    r = target.ratios()
    assert snap.pe_trailing == pytest.approx(r["pe"]) and snap.ev_ebitda == pytest.approx(r["ev_ebitda"])
    assert snap.pe_forward is None and snap.peg is None                  # no estimates in free data
    view = memo_mod.fundamental_view(target, cmp, "Technology")
    assert view.peer_based and "6 peers (4-digit SIC 7372)" in view.source
    pe = next(b for b in view.beacons if b.key == "valuation:pe")
    assert "vs peer median" in pe.value and "6 peers" in pe.rationale
    assert view.score > view.sector_score                               # cheap against its peers, not the sector
    none = memo_mod.fundamental_view(target, None, "Technology", why_no_peers="peer search off")
    assert not none.peer_based and "peer search off" in none.source and none.score == none.sector_score
    loss = memo_mod.fundamental_snapshot(company(ni=-1e9))
    assert loss.pe_trailing == -1.0                                     # the snapshot's "negative earnings"


def test_memo_peer_norms_are_opt_in(msft, monkeypatch):
    from lighthouse.report import _fundamental_view_lines
    kw = dict(as_of="2016-11-15", source="sample", risk_free=0.005, equity_premium=0.06)
    plain = memo_mod.build("MSFT", peers=False, **kw)
    assert plain.fundamental_view is None and "## Fundamental beacons" not in plain.markdown()
    off = memo_mod.build("MSFT", peers=False, peer_norms=True, **kw)
    assert not off.fundamental_view.peer_based and "peer search off" in off.fundamental_view.source
    text = off.markdown()
    assert "## Fundamental beacons against the sector norms" in text
    # the section is added; nothing else in the memo changes
    assert text.replace("\n".join(_fundamental_view_lines(off.fundamental_view)) + "\n", "") == plain.markdown()
    assert (off.stance, off.thesis, off.criteria) == (plain.stance, plain.thesis, plain.criteria)
    # with peers (the offline sample pack has no filing lists to find them, so the search is stood in for)
    peers = {f"P{i}": company(price=400.0 + 20 * i, ticker=f"P{i}") for i in range(6)}
    monkeypatch.setattr(memo_mod, "find_peers", lambda t, f, sic, cands, source="auto", n=8:
                        compare(f, peers, "4-digit SIC 7372"))
    m = memo_mod.build("MSFT", peers=True, peer_norms=True, **kw)
    assert m.fundamental_view.peer_based and m.fundamental_view.sector == "Technology"
    md = m.markdown()
    assert "## Fundamental beacons against the peer medians" in md and "vs peer median" in md
    assert "against the sector norms for Technology" in md
    assert m.rule_verdict == plain.rule_verdict


def test_memo_cli_peer_norms_flag(capsys):
    from lighthouse.__main__ import main
    assert main(["memo", "MSFT", "--source", "sample", "--as-of", "2016-11-15", "--no-signals", "--no-peers",
                 "--peer-norms"]) == 0
    out, err = capsys.readouterr()
    assert "## Fundamental beacons against the sector norms" in out and "--peer-norms needs the peer search" in err


def test_red_flags():
    # net income far above cash flow, thin interest cover, a big maturity against little cash
    f = company(ni=8e9, ocf=2e9, interest=6e9, debt_due_1y=20e9, cash=1e9)   # $10bn operating income
    q = checks.earnings_quality(f)
    assert any(x.severity == "red" and "Accruals" in x.text for x in q)
    assert any("cash flow" in x.text and x.severity == "amber" for x in q)
    b = checks.balance_sheet(f)
    assert any(x.severity == "red" and "covers interest" in x.text for x in b)
    assert any(x.severity == "red" and "falls due" in x.text for x in b)
    clean = checks.balance_sheet(company())
    assert all(x.severity != "red" for x in clean)


def test_amended_filings_flagged_and_catalysts_from_last_year():
    filings = pd.DataFrame({
        "filed": pd.to_datetime(["2019-07-25", "2019-10-24", "2020-01-30", "2020-02-10", "2020-04-28", "2019-10-24",
                                 "2019-08-01"]),
        "form": ["10-K", "10-Q", "10-Q", "10-K/A", "10-Q", "8-K", "DEF 14A"],
        "items": ["", "", "", "", "", "2.02,9.01", ""]})
    f = company()
    assert any("amended" in x.text for x in checks.earnings_quality(f, filings))
    cal = checks.catalysts(filings, "2020-06-30", fiscal_year_end="0630")
    got = dict(zip(cal["event"], cal["expected"].astype(str)))
    assert got["annual report (10-K)"] == "2020-07-27"             # 2019-07-25 + 1 year = a Saturday: next Monday
    assert got["earnings release (8-K Item 2.02)"] == "2020-10-26"   # 2020-10-24 is a Saturday
    assert got["fiscal year end"] == "2021-06-30"
    assert list(cal["expected"]) == sorted(cal["expected"])


def test_capital_allocation_sums_the_years():
    f = company()
    f.annual = pd.DataFrame({"operating_cash_flow": [10.0, 12.0], "capex": [2.0, 2.0], "dividends_paid": [3.0, 3.0],
                             "buybacks": [1.0, 3.0], "cover_shares": [100.0, 95.0]},
                            index=pd.to_datetime(["2018-06-30", "2019-06-30"]))
    c = checks.capital_allocation(f)
    assert c["fcf"] == 18.0 and c["dividends"] == 6.0 and c["returned_share_of_fcf"] == pytest.approx(10 / 18)
    assert c["share_change"] == pytest.approx(-0.05)


def test_conviction_is_bounded_and_never_raises_size():
    assert thesis.conviction_size(100, 1.0) == 100
    assert thesis.conviction_size(100, 0.0) == 50
    assert thesis.conviction_size(100, 7.0) == 100


# --------------------------------------------------------------------------- #
# MSFT from the sample pack
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def msft():
    px = load_prices("MSFT", years=None, source="sample")
    return px, pit.load_statements("MSFT", source="sample"), load_actions("MSFT", source="sample")


def test_financials_use_only_filings_public_on_the_date(msft):
    """The numbers on date D are the same whether or not later filings exist (no look-ahead)."""
    px, st, acts = msft
    facts = edgar.company_facts(ticker="MSFT", source="sample")
    for cut in ("2014-06-30", "2016-11-15"):
        full = financials_as_of("MSFT", st, px, acts, cut)
        part = financials_as_of("MSFT", pit.build_statements(facts[facts.filed <= cut]), px, acts, cut)
        assert full.last_filed <= pd.Timestamp(cut)
        for k in ("revenue", "net_income", "operating_cash_flow"):
            assert full.ttm[k] == pytest.approx(part.ttm[k])
        assert full.bs["equity"] == pytest.approx(part.bs["equity"])
        assert full.shares == pytest.approx(part.shares)


def test_msft_memo_end_to_end_offline(msft, tmp_path):
    px, st, acts = msft
    m = memo_mod.build("MSFT", as_of="2016-11-15", source="sample", peers=False, risk_free=0.005, equity_premium=0.06)
    assert m.as_of == pd.Timestamp("2016-11-15")
    assert m.stance in ("buy", "hold", "avoid")
    assert m.valuation is not None and m.valuation.implied_growth is not None
    assert m.criteria and any(c.metric == "price" for c in m.criteria)
    text = m.markdown()
    for section in ("## Recommendation", "## Valuation", "## What would prove this wrong", "## Pre-mortem",
                    "## Sources and notes"):
        assert section in text
    # the thesis tracker: open in November 2016, check a year later (new 10-Qs and a 10-K have been filed), close
    book = thesis.ThesisBook(path=tmp_path / "j.sqlite")
    tid = thesis.open_thesis(m, book)
    sts = thesis.check_theses(book, source="sample", date="2017-11-10")
    assert len(sts) == 1 and sts[0].new_filing is not None and sts[0].price_now > sts[0].price_then
    again = thesis.check_theses(book, source="sample", date="2017-11-10")
    assert again[0].new_filing is None                              # the same filing is not news twice
    cov = thesis.coverage(book, date="2017-11-10", prices_now=False)
    assert cov.loc[tid, "filings_since"] == 1 and cov.loc[tid, "stale"]
    lines = thesis.close_thesis(tid, "test", book, source="sample", date="2017-11-10")
    assert any("Revenue grew" in x for x in lines) and any("P/E" in x for x in lines)
    # growth per year of reported quarters: the trailing sums known on the two dates, over their quarter ends
    f_close = financials_as_of("MSFT", st, px, acts, "2017-11-10")
    q = f_close.quarterly
    end0 = q.index[pd.to_datetime(q["filed"]) <= "2016-11-15"].max()
    end1 = q.index.max()
    g = (f_close.get("revenue") / m.financials.get("revenue")) ** (365.25 / (end1 - end0).days) - 1.0
    rev_line = next(x for x in lines if x.startswith("Revenue grew"))
    assert f"Revenue grew {g:+.1%} a year (trailing twelve months to {pd.Timestamp(end0).date()}" in rev_line
    assert thesis.coverage(book, date="2017-11-10", prices_now=False).loc[tid, "status"] == "closed"
    assert book.open_ids() == []
    with pytest.raises(Exception):                                  # append-only
        with book._conn() as c:
            c.execute("DELETE FROM theses")
