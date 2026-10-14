"""
Record the offline sample pack's point-in-time fundamentals and corporate actions (provenance for data/sample/).

For every stock in the sample pack:
  data/sample/edgar/<TICKER>.parquet    its SEC EDGAR XBRL facts (companyfacts API), filtered to the concepts in
                                        lighthouse.pit.CONCEPTS and to 10-K / 10-Q filings, each with its filing date
  data/sample/actions/<TICKER>.csv      its dividends and splits by ex-date from Yahoo Finance (today's share units)

The SEC's data is public; the request carries the contact e-mail in LIGHTHOUSE_SEC_EMAIL, as the SEC requires.

Run:  LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/build_edgar_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from lighthouse import edgar, total_return  # noqa: E402
from lighthouse.data import SAMPLE_DIR, sample_pack_tickers  # noqa: E402


def main() -> int:
    tickers = [t for t in sample_pack_tickers() if not t.startswith("^")]
    (SAMPLE_DIR / "edgar").mkdir(parents=True, exist_ok=True)
    (SAMPLE_DIR / "actions").mkdir(parents=True, exist_ok=True)
    for t in tickers:
        facts = edgar.company_facts(ticker=t, refresh=True)
        facts = facts[facts["form"].isin(edgar.PERIODIC_FORMS) | facts["concept"].isin(["EntityCommonStockSharesOutstanding",
                                                                                         "EntityPublicFloat"])]
        facts = facts.drop(columns=[c for c in ("frame", "cik") if c in facts.columns]).reset_index(drop=True)
        facts.to_parquet(edgar.sample_facts_path(t), index=False, compression="zstd")
        acts = total_return.load_actions(t, refresh=True)
        acts.to_csv(total_return.sample_actions_path(t))
        print(f"{t}: {len(facts):,} facts filed {facts['filed'].min().date()} to {facts['filed'].max().date()}; "
              f"{int((acts['dividend'] > 0).sum())} dividends, {int((acts['split'] != 1).sum())} splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
