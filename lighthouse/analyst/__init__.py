"""
The single-stock analyst toolkit: what an equity analyst builds before writing a recommendation, from free data.

  financials  the company's latest point-in-time numbers (TTM flows, balance sheet, per-share) as of a date
  model       a small linked three-statement projection (income statement, balance sheet, cash flow) from a few
              assumptions, 3-5 years out
  valuation   discounted cash flow under bull / base / bear cases, a reverse DCF (the growth today's price implies),
              historical valuation bands, a probability-weighted target and the upside-to-downside ratio
  peers       real peers by SIC code and market value, a comparison table and the premium or discount explained
  checks      earnings-quality red flags, balance-sheet risk (debt coming due, interest cover, liquidity),
              the capital-allocation record, a catalyst calendar from past filing dates
  memo        the investment memo: thesis, supporting points, valuation, risks, what would prove it wrong, pre-mortem

Numbers come from SEC filings (dated by filing), prices from the price feed, rates from FRED, the equity premium from
the Ken French library. Nothing here uses analyst estimates (none are free): "consensus" is what the price implies.
"""
