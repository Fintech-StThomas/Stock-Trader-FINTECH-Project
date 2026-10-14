"""
Lighthouse research: cross-stock tests of whether a signal has an edge, on survivorship-aware universes.

  panel     load many price series politely (cache first, retries for throttling, remembered misses) and measure
            their coverage of a universe
  cohort    the per-stock rule test on a fixed cohort (the members of an index on one date, followed to today or to
            their delisting), under both delisting rules
  harness   monthly cross-sectional tests: information coefficients, quintile portfolios, costs, factor regressions,
            tuning / hold-out split, deflated Sharpe ratio (Phase 3)

Every published number comes from a script in scripts/ and is logged in reports/research/.
"""
