"""
The one-call pipeline: data -> fundamentals -> technicals -> regime -> decision -> backtest.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest, strategy_layers, sensitivity_grid, walk_forward, regime_breakdown
from .config import InvestorProfile, StrategyParams
from .data import load_prices, load_benchmark, load_fundamentals, align, DataUnavailable
from .decision import Decision, make_decision
from .fundamentals import FundamentalSnapshot, StatementHistory, fundamental_beacons, fundamental_score, point_in_time_scores
from .regime import regime_table, latest_regime, RegimeSnapshot
from .technicals import compute_indicators, technical_score, latest_beacons


@dataclass
class AnalysisResult:
    ticker: str
    profile: InvestorProfile
    params: StrategyParams
    prices: pd.DataFrame
    indicators: pd.DataFrame
    technical_scores: pd.Series
    benchmark: Optional[pd.DataFrame]
    regimes: Optional[pd.DataFrame]
    fundamentals: FundamentalSnapshot
    statements: StatementHistory
    pit_scores: pd.Series
    decision: Decision
    backtest: Optional[BacktestResult]
    warnings: List[str] = field(default_factory=list)
    extras: Dict[str, object] = field(default_factory=dict)   # layers / sensitivity / walk-forward when computed

    # ---- convenience -------------------------------------------------------------
    @property
    def verdict(self) -> str:
        return self.decision.verdict

    @property
    def score(self) -> float:
        return self.decision.composite_score

    def report(self, include_backtest: bool = True) -> str:
        from .report import build_report
        return build_report(self, include_backtest=include_backtest)

    def evaluate(self, layers: bool = True, sensitivity: bool = True, walk_forward_test: bool = True) -> Dict[str, object]:
        """Run the deeper (slower) evaluations and cache them in `extras`."""
        bt_kwargs = dict(prices=self.prices, benchmark=self.benchmark, params=self.params, profile=self.profile,
                         fundamentals_pit=self.pit_scores)
        if layers and "layers" not in self.extras:
            self.extras["layers"] = strategy_layers(**bt_kwargs)
        if sensitivity and "sensitivity" not in self.extras:
            self.extras["sensitivity"] = sensitivity_grid(**bt_kwargs)
        if walk_forward_test and "walk_forward" not in self.extras:
            self.extras["walk_forward"] = walk_forward(**bt_kwargs)
        if self.backtest is not None and "regime_breakdown" not in self.extras:
            self.extras["regime_breakdown"] = regime_breakdown(self.backtest)
        return self.extras

    def save(self, out_dir: str, charts: bool = True) -> List[str]:
        """Write the report, tables and charts to a folder. Returns the paths written."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        (out / f"{self.ticker}_report.md").write_text(self.report())
        written.append(str(out / f"{self.ticker}_report.md"))
        if self.backtest is not None:
            self.backtest.summary().to_csv(out / f"{self.ticker}_backtest_summary.csv")
            self.backtest.trades_frame().to_csv(out / f"{self.ticker}_trades.csv", index=False)
            written += [str(out / f"{self.ticker}_backtest_summary.csv"), str(out / f"{self.ticker}_trades.csv")]
        if charts:
            from . import charts as ch
            written += ch.save_all(self, out)
        return written


def analyze(ticker: str, profile: Optional[InvestorProfile] = None, params: Optional[StrategyParams] = None,
            years: float = 10, benchmark: str = "^GSPC", source: str = "auto", run_backtest_: bool = True,
            fundamentals_overrides: Optional[Dict[str, float]] = None, fundamentals_fixture: Optional[str] = None,
            as_of=None) -> AnalysisResult:
    """Analyse one ticker end-to-end and return everything the product shows.

    Parameters
    ----------
    ticker       : e.g. "MSFT"
    profile      : the investor's horizon / risk tolerance / max loss / account value and cash available
    years        : history to analyse and backtest
    benchmark    : market index for regime detection and comparison
    source       : "auto" (live with offline fallbacks), "yfinance", "stooq", "cache", "sample"
    as_of        : optional date - "what would Lighthouse have said on this day?"
    """
    prof = profile or InvestorProfile()
    p = params or StrategyParams.for_horizon(prof.horizon)
    ticker = ticker.upper().strip()
    warnings: List[str] = []

    prices = load_prices(ticker, years=years, end=as_of, source=source)
    rep = prices.attrs.get("cleaning")
    if rep is not None:
        warnings += rep.notes
    if len(prices) < p.min_history:
        raise DataUnavailable(f"{ticker}: only {len(prices)} trading days available; need at least {p.min_history} "
                              f"for the 200-day trend and 52-week high indicators")

    bench = None
    try:
        # same calendar window as the stock (offline sample series end on different dates)
        bench = load_benchmark(benchmark, years=None, start=prices.index[0], end=prices.index[-1], source=source)
        prices, bench = align(prices, bench)
    except DataUnavailable as exc:
        warnings.append(f"Benchmark {benchmark} unavailable ({exc}); market-regime rules disabled.")
        bench = None
    prices.attrs["ticker"] = ticker

    ind = compute_indicators(prices, p)
    tscores = technical_score(ind, prof.horizon, p)
    t_beacons = latest_beacons(ind, prof.horizon, p)
    last = ind.iloc[-1]
    t_now = float(tscores.iloc[-1])
    if not np.isfinite(t_now):
        raise DataUnavailable(f"{ticker}: indicators are not fully defined on the last date; need more history")

    snap, hist = load_fundamentals(ticker, source=source, last_price=float(last["Close"]),
                                   overrides=fundamentals_overrides, fixture_path=fundamentals_fixture)
    warnings += snap.notes
    if snap.source.startswith("offline") or "fixture" in snap.source:
        warnings.append(f"Fundamentals come from an offline fixture ({snap.source}); values are approximate and dated {snap.as_of}.")
    f_beacons = fundamental_beacons(snap, hist, p)
    f_now = fundamental_score(f_beacons)

    pit = point_in_time_scores(prices, hist, snap.sector, p)
    if as_of is not None and pit.notna().any():
        # time-travel mode: use the point-in-time fundamentals rather than today's snapshot
        pit_now = pit.iloc[-1]
        if np.isfinite(pit_now):
            f_now = float(pit_now)
            warnings.append("as_of mode: fundamental score taken from point-in-time statements (snapshot ratios are current, not historical).")

    regimes = regime_table(bench, p) if bench is not None else None
    regime_now: Optional[RegimeSnapshot] = latest_regime(regimes, benchmark) if regimes is not None else None

    decision = make_decision(ticker, prices.index[-1], float(last["Close"]), float(last["atr"]), float(last["sizing_vol"]),
                             f_now, t_now, f_beacons + t_beacons, regime_now, prof, p, warnings,
                             sma_slow=float(last["sma_slow"]))

    bt = None
    if run_backtest_:
        try:
            bt = run_backtest(prices, bench, p, prof, pit)
        except ValueError as exc:
            warnings.append(f"Backtest skipped: {exc}")

    return AnalysisResult(ticker=ticker, profile=prof, params=p, prices=prices, indicators=ind, technical_scores=tscores,
                          benchmark=bench, regimes=regimes, fundamentals=snap, statements=hist, pit_scores=pit,
                          decision=decision, backtest=bt, warnings=list(dict.fromkeys(warnings)))
