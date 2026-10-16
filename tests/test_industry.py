"""Industry metrics (banks, REITs, software, insurers) on small trimmed copies of real SEC companyfacts (recorded
October 2026; tests/fixtures/edgar): JPM 2021-2022 and USB 2025 (banks), O 2024-2025 (REIT), CRM 2024-2026 (software),
TRV 2024-2025 (insurer). Nothing here touches the network."""
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from lighthouse import edgar, pit
from lighthouse.analyst import industry, memo as memo_mod
from lighthouse.analyst.industry import industry_for_sic, industry_metrics

FIX = Path(__file__).parent / "fixtures" / "edgar"


def facts(name: str) -> pd.DataFrame:
    return edgar.flatten_company_facts(json.loads((FIX / f"{name}.json").read_text()))


@pytest.fixture(scope="module")
def jpm():
    return facts("jpm_companyfacts_2021_2022")


def test_industry_from_sic_code():
    assert industry_for_sic("6021") == "bank" and industry_for_sic(6022) == "bank" and industry_for_sic("6036") == "bank"
    assert industry_for_sic("6798") == "reit"
    assert industry_for_sic("7372") == "software" and industry_for_sic("7370") == "software"
    assert industry_for_sic("6331") == "insurer" and industry_for_sic("6324") == "insurer"
    assert industry_for_sic("6411") is None                    # insurance brokers do not underwrite
    assert industry_for_sic("3571") is None and industry_for_sic(None) is None and industry_for_sic("") is None
    assert industry_metrics(pd.DataFrame(), "3571", "2022-12-28") is None


def test_industry_concepts_are_in_the_facts_cache_list_but_not_in_the_statements(jpm):
    assert {"InterestIncomeExpenseNet", "Deposits", "RevenueRemainingPerformanceObligation", "PremiumsEarnedNet",
            "GainLossOnSaleOfProperties"} <= set(pit.CONCEPTS)
    assert not set(pit.INDUSTRY) & set(pit.ITEMS)              # the generic statements keep their columns
    # facts that carry the industry concepts build exactly the statements the generic concepts alone build
    generic = {c for li in pit.LINE_ITEMS for c in li.concepts} | set(pit.COVER_CONCEPTS)
    a, b = pit.build_statements(jpm[jpm["concept"].isin(generic)]), pit.build_statements(jpm)
    pd.testing.assert_frame_equal(a.annual, b.annual)
    pd.testing.assert_frame_equal(a.quarterly, b.quarterly)


def test_bank_metrics_jpm(jpm):
    v = industry_metrics(jpm, "6021", "2022-12-28", market_cap=390e9)
    assert v.industry == "bank" and v.as_of == pd.Timestamp("2022-12-28")
    nii = v.get("net_interest_income")
    # twelve months to 2022-09-30: Q4 2021 (the 10-K's year minus the 10-Q's nine months) + three 2022 quarters
    assert nii.value == pytest.approx((52_311 - 38_710 + 13_872 + 15_128 + 17_518) * 1e6)
    assert nii.period == "12 months to 2022-09-30 (four quarters)" and nii.filed == pd.Timestamp("2022-11-03")
    assert v.get("nii_to_assets").value == pytest.approx(60_119 / ((3_773_884 + 3_757_576) / 2.0))
    nie = (71_343 - 53_455) + 19_191 + 18_749 + 19_178
    nonii = (69_338 - 53_682) + 16_845 + 15_587 + 15_198
    assert v.get("efficiency_ratio").value == pytest.approx(nie / (60_119 + nonii))
    ld = v.get("loans_to_deposits")
    assert ld.value == pytest.approx(1_068_073 / 2_408_615) and ld.period == "at 2022-09-30"
    # JPM tagged no intangible-assets concept in 2021-2022 (its first such facts came in 2024): no tangible book
    tbv = v.get("tbv_per_share")
    assert tbv.value is None and tbv.filed is None and "intangible assets" in tbv.note
    assert v.get("price_to_tbv").value is None
    # the 10-K for 2022 (filed 2023-02-21) moves every flow to the fiscal year
    later = industry_metrics(jpm, "6021", "2023-03-01")
    assert later.get("net_interest_income").value == pytest.approx(66_710e6)
    assert later.get("net_interest_income").period == "12 months to 2022-12-31 (fiscal year)"


def test_tangible_book_value_usb():
    usb = facts("usb_companyfacts_2025")
    v = industry_metrics(usb, "6021", "2025-12-01", market_cap=100e9)
    tbv = v.get("tbv_per_share")
    # (equity - preferred - goodwill - other intangibles) / the cover-page count of the same 10-Q
    assert tbv.value == pytest.approx((63_340 - 6_808 - 12_634 - 5_152) * 1e6 / 1_554_434_067)
    assert tbv.period == "at 2025-09-30" and tbv.filed == pd.Timestamp("2025-11-05")
    assert v.get("price_to_tbv").value == pytest.approx(100e9 / ((63_340 - 6_808 - 12_634 - 5_152) * 1e6))
    # USB tags CommonStockSharesOutstanding rounded (1.6bn); the exact cover-page count of the 10-K is used instead
    after_10k = industry_metrics(usb, "6021", "2026-03-02").get("tbv_per_share")
    assert after_10k.value == pytest.approx((65_193 - 6_808 - 12_635 - 4_904) * 1e6 / 1_553_695_799)
    # a missing component makes the metric unknown (never a guess)
    no_gw = usb[usb["concept"] != "Goodwill"]
    gone = industry_metrics(no_gw, "6021", "2026-03-02").get("tbv_per_share")
    assert gone.value is None and "goodwill" in gone.note


def test_tangible_book_counts_preferred_as_zero_only_when_never_tagged():
    usb = facts("usb_companyfacts_2025")
    pref = usb["concept"].isin(pit.INDUSTRY["preferred_equity"].concepts)
    # a bank whose filings never tag preferred stock under any preferred concept: preferred counted as 0, said so
    none = industry_metrics(usb[~pref], "6021", "2025-12-01", market_cap=100e9)
    tbv = none.get("tbv_per_share")
    assert tbv.value == pytest.approx((63_340 - 0 - 12_634 - 5_152) * 1e6 / 1_554_434_067)
    assert "counted as 0" in tbv.note and "counted as 0" in none.get("price_to_tbv").note
    assert none.get("price_to_tbv").value == pytest.approx(100e9 / ((63_340 - 12_634 - 5_152) * 1e6))
    # tagged in the 10-Q but not in the later 10-K: the 10-K's tangible book stays unknown (never a guess)
    gap = usb[~(pref & (usb["end"] == pd.Timestamp("2025-12-31")))]
    late = industry_metrics(gap, "6021", "2026-03-02").get("tbv_per_share")
    assert late.value is None and "preferred stock" in late.note and "counted as 0" not in late.note
    # with preferred tagged, nothing changes and no note is added
    assert industry_metrics(usb, "6021", "2025-12-01").get("tbv_per_share").note == ""


def test_reit_ffo_realty_income():
    o = facts("o_companyfacts_2024_2025")
    v = industry_metrics(o, "6798", "2026-03-02", market_cap=50e9)
    ffo = v.get("ffo")
    # NAREIT's definition: net income + D&A - gains on property sales + real-estate impairments (fiscal 2025)
    assert ffo.value == pytest.approx((1_058_590 + 2_524_200 - 177_640 + 471_335) * 1e3)
    assert ffo.period == "12 months to 2025-12-31 (fiscal year)" and ffo.filed == pd.Timestamp("2026-02-25")
    assert "NAREIT" in ffo.formula and ffo.note == ""
    assert v.get("ffo_payout").value == pytest.approx(2_920_895 / (1_058_590 + 2_524_200 - 177_640 + 471_335))
    assert v.get("p_ffo").value == pytest.approx(50e9 / ffo.value)
    assert v.get("re_impairments").value == pytest.approx(471_335e3)
    # before the 10-K: four quarters, the fourth quarter of 2024 from the 10-K minus nine months, dividends from
    # year-to-date cash flows
    q = industry_metrics(o, "6798", "2025-12-01")
    ni = (860_772 - 661_160) + 249_815 + 196_919 + 315_771
    da = (2_395_644 - 1_788_973) + 608_935 + 647_849 + 631_981
    gains = (117_275 - 92_290) + 22_537 + 38_566 + 49_107
    imp = (425_833 - 282_867) + 116_589 + 143_363 + 86_972
    assert q.get("ffo").value == pytest.approx((ni + da - gains + imp) * 1e3)
    assert q.get("re_impairments").value == pytest.approx(imp * 1e3)
    divs = (2_691_719 - 1_999_858) + 2_177_133
    assert q.get("ffo_payout").value == pytest.approx(divs / (ni + da - gains + imp))
    # no impairment tagged for the period: nothing added back, and the metric says so
    no_imp = industry_metrics(o[o["concept"] != "ImpairmentOfRealEstate"], "6798", "2026-03-02")
    assert no_imp.get("ffo").value == pytest.approx((1_058_590 + 2_524_200 - 177_640) * 1e3)
    assert "none added back" in no_imp.get("ffo").note and no_imp.get("re_impairments").value is None
    # a REIT that tags no gains on property sales under a standard concept (PLD since 2020, SPG): no FFO
    no_gains = industry_metrics(o[o["concept"] != "GainLossOnSaleOfProperties"], "6798", "2026-03-02")
    assert no_gains.get("ffo").value is None and "real_estate_gains" in no_gains.get("ffo").note
    assert no_gains.get("p_ffo").value is None and no_gains.get("ffo_payout").value is None


def test_software_rpo_salesforce():
    crm = facts("crm_companyfacts_2024_2026")
    v = industry_metrics(crm, "7372", "2026-03-31")
    assert v.get("rpo").value == pytest.approx(72.4e9) and v.get("rpo").period == "at 2026-01-31"
    rev = crm[(crm.concept == "RevenueFromContractWithCustomerExcludingAssessedTax") & (crm.end == "2026-01-31")
              & (crm.start == "2025-02-01")]["val"].iloc[0]
    assert v.get("rpo_to_revenue").value == pytest.approx(72.4e9 / rev)
    rpo_prev = crm[(crm.concept == "RevenueRemainingPerformanceObligation") & (crm.end == "2025-01-31")]["val"].iloc[0]
    assert v.get("rpo_growth").value == pytest.approx(72.4e9 / rpo_prev - 1.0)
    dr = v.get("deferred_revenue_growth")
    cur = crm[crm.concept == "ContractWithCustomerLiabilityCurrent"].set_index("end")["val"]
    assert dr.value == pytest.approx(cur[pd.Timestamp("2026-01-31")] / cur[pd.Timestamp("2025-01-31")] - 1.0)
    assert "current portion" in dr.formula                      # CRM tags only the current part of its contract liability


def test_insurer_loss_ratio_travelers():
    trv = facts("trv_companyfacts_2024_2025")
    v = industry_metrics(trv, "6331", "2026-03-02")
    prem = trv[(trv.concept == "PremiumsEarnedNet") & (trv.start == "2025-01-01") & (trv.end == "2025-12-31")]["val"].iloc[0]
    loss = trv[(trv.concept == "PolicyholderBenefitsAndClaimsIncurredNet") & (trv.start == "2025-01-01")
               & (trv.end == "2025-12-31")]["val"].iloc[0]
    assert v.get("premiums_earned").value == pytest.approx(prem)
    assert v.get("loss_ratio").value == pytest.approx(loss / prem)
    assert v.get("premium_growth").value is not None
    # premiums tagged in the 10-K only (Allstate since 2018): the latest fiscal year, both numbers over it
    annual_only = trv[~((trv.concept == "PremiumsEarnedNet") & (trv.form == "10-Q"))]
    a = industry_metrics(annual_only, "6331", "2025-12-01")
    assert a.get("loss_ratio").period == "12 months to 2024-12-31 (fiscal year)"
    # losses not tagged (Allstate after the first quarter of 2024): premiums alone, no loss ratio
    no_loss = industry_metrics(trv[trv.concept != "PolicyholderBenefitsAndClaimsIncurredNet"], "6331", "2026-03-02")
    assert no_loss.get("loss_ratio").value is None and "losses_and_benefits" in no_loss.get("loss_ratio").note
    assert no_loss.get("premiums_earned").value == pytest.approx(prem)


CASES = [("jpm_companyfacts_2021_2022", "6021", ["2022-02-23", "2022-08-01", "2022-11-03", "2022-12-28"]),
         ("usb_companyfacts_2025", "6021", ["2025-11-05", "2025-11-06", "2026-02-24"]),
         ("o_companyfacts_2024_2025", "6798", ["2025-05-07", "2025-08-08", "2025-12-01"]),
         ("crm_companyfacts_2024_2026", "7372", ["2025-06-01", "2025-12-15"]),
         ("trv_companyfacts_2024_2025", "6331", ["2025-07-18", "2025-12-01"])]


def _values(view):
    return [(m.key, m.value, m.period, m.filed) for m in view.metrics]


@pytest.mark.parametrize("name,sic,dates", CASES)
def test_industry_metrics_are_point_in_time_under_truncation(name, sic, dates):
    """The metrics on date D are the same when every fact filed on or after D is removed, or replaced by anything:
    nothing filed from D on is used (a value is usable from the session after its filing)."""
    f = facts(name)
    for d in dates:
        full = industry_metrics(f, sic, d, market_cap=80e9)
        cut = industry_metrics(f[f["filed"] < pd.Timestamp(d)], sic, d, market_cap=80e9)
        assert _values(full) == _values(cut)
        later = f["filed"] >= pd.Timestamp(d)
        noisy = f.copy()
        noisy.loc[later, "val"] = noisy.loc[later, "val"] * 7.0 + 1.0
        assert _values(industry_metrics(noisy, sic, d, market_cap=80e9)) == _values(full)
        assert all(m.filed is None or m.filed < pd.Timestamp(d) for m in full.metrics)


def test_a_filing_is_used_from_the_next_session(jpm):
    on_day = industry_metrics(jpm, "6021", "2022-11-03").get("net_interest_income")
    next_day = industry_metrics(jpm, "6021", "2022-11-04").get("net_interest_income")
    assert on_day.period == "12 months to 2022-06-30 (four quarters)" and on_day.filed == pd.Timestamp("2022-08-03")
    assert next_day.period == "12 months to 2022-09-30 (four quarters)" and next_day.filed == pd.Timestamp("2022-11-03")


def test_memo_shows_the_industry_metrics(monkeypatch, jpm):
    """JPM from the sample pack (prices to 2022-12-28): the pack records its profile (SIC 6021) and the bank concepts
    since October 2026, so the memo carries the bank section offline; adding the fixture's facts changes neither the
    section's inputs nor the generic numbers; a company outside the four industries gets none."""
    # the modules memo.build imports from at call time: the app tests reload the package when a source file changed
    # during the run, so the ones bound when this file was collected may no longer be the live ones
    edgar_now = importlib.import_module("lighthouse.edgar")
    memo_now = importlib.import_module("lighthouse.analyst.memo")
    real_facts = edgar_now.company_facts
    plain = memo_now.build("JPM", source="sample", peers=False, risk_free=0.03, equity_premium=0.06)
    assert plain.industry is not None and plain.industry.industry == "bank" and plain.industry.sic == "6021"
    assert "## Industry metrics: bank (SIC 6021 National Commercial Banks)" in plain.markdown()

    def with_bank_concepts(cik=None, ticker=None, concepts=None, refresh=False, source="auto"):
        f = real_facts(cik=cik, ticker=ticker, concepts=concepts, refresh=refresh, source=source)
        if ticker != "JPM":
            return f
        # the fixture repeats some of the sample pack's facts (same filing, same value): one copy of each, as EDGAR
        both = pd.concat([f, jpm], ignore_index=True)
        return both.drop_duplicates(["taxonomy", "concept", "unit", "start", "end", "accn", "val"]).reset_index(drop=True)

    monkeypatch.setattr(edgar_now, "company_facts", with_bank_concepts)
    monkeypatch.setattr(edgar_now, "company_filings", lambda t, source="auto", refresh=False: (
        {"name": "JPMORGAN CHASE & CO", "sic": "6021", "sicDescription": "National Commercial Banks",
         "fiscalYearEnd": "1231"}, None))
    m = memo_now.build("JPM", source="sample", peers=False, risk_free=0.03, equity_premium=0.06)
    assert m.as_of == pd.Timestamp("2022-12-28") and m.industry is not None and m.industry.industry == "bank"
    assert m.industry.get("efficiency_ratio").value == pytest.approx(
        industry_metrics(jpm, "6021", "2022-12-28").get("efficiency_ratio").value)
    text = m.markdown()
    assert "## Industry metrics: bank (SIC 6021 National Commercial Banks)" in text
    assert "Efficiency ratio" in text and "2022-11-03" in text and "intangible assets" in text
    # the generic numbers are the sample pack's, untouched by the extra concepts
    pd.testing.assert_series_equal(pd.Series(m.financials.ratios()), pd.Series(plain.financials.ratios()))
    assert m.stance == plain.stance and m.financials.ratios()["pe"] == pytest.approx(10.25, abs=0.01)
    # the sector the opt-in beacons would use comes from the SIC code when the curated list has none
    assert memo_now._memo_sector("ZZZZ", None, "6021") == "Financial Services"
    monkeypatch.setattr(edgar_now, "company_filings", lambda t, source="auto", refresh=False: (
        {"name": t, "sic": "3571"}, None))
    assert memo_now.build("JPM", source="sample", peers=False, risk_free=0.03, equity_premium=0.06).industry is None


def test_memo_without_a_sic_code_has_no_industry_section(monkeypatch):
    # MSFT's recorded profile (SIC 7372) makes it a software company: RPO and deferred revenue are shown
    m = memo_mod.build("MSFT", as_of="2016-11-15", source="sample", peers=False, risk_free=0.005, equity_premium=0.06)
    assert m.industry is not None and m.industry.industry == "software"
    assert "## Industry metrics: software" in m.markdown()
    # the same company with a profile that has no SIC code gets no section, and no error
    edgar_now = importlib.import_module("lighthouse.edgar")
    memo_now = importlib.import_module("lighthouse.analyst.memo")
    monkeypatch.setattr(edgar_now, "company_filings", lambda t, source="auto", refresh=False: ({"name": t}, None))
    bare = memo_now.build("MSFT", as_of="2016-11-15", source="sample", peers=False, risk_free=0.005,
                          equity_premium=0.06)
    assert bare.industry is None and "## Industry metrics" not in bare.markdown()


def test_metric_text_formats():
    mk = industry.IndustryMetric
    assert mk("a", "A", 2.5e9, "money", "p", None, "f").text() == "$2.50bn"
    assert mk("a", "A", 0.523, "ratio", "p", None, "f").text() == "52.3%"
    assert mk("a", "A", 1.75, "multiple", "p", None, "f").text() == "1.75x"
    assert mk("a", "A", 24.93, "per_share", "p", None, "f").text() == "$24.93"
    assert mk("a", "A", None, "ratio", "n/a", None, "f").text() == "n/a"
