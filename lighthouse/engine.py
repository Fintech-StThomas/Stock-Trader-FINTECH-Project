"""
The one-call pipeline: prices -> fundamentals (SEC filings) -> technicals -> regime -> decision -> backtest. The
fundamental score is built from the same point-in-time numbers as the analyst memo (analyst.financials_as_of), and
the backtest gates its entries on the score each statement gave on the day it was public.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest, strategy_layers, sensitivity_grid, walk_forward, regime_breakdown
from .config import InvestorProfile, StrategyParams
from .data import EXCHANGE_TZ, Quote, load_prices, load_benchmark, load_quote, load_risk_free, align, DataUnavailable
from .decision import Decision, make_decision
from .fundamentals import FundamentalSnapshot, StatementHistory, fundamental_beacons, fundamental_score, point_in_time_scores
from .parallel import map_tickers
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
    decision: Decision
    backtest: Optional[BacktestResult]
    fundamentals: Optional[FundamentalSnapshot] = None   # today's SEC numbers behind the fundamental beacons
    statements: Optional[StatementHistory] = None        # SEC EDGAR statements with filing dates (None: unavailable)
    pit_scores: Optional[pd.Series] = None               # the point-in-time fundamental score the backtest gates on
    warnings: List[str] = field(default_factory=list)
    extras: Dict[str, object] = field(default_factory=dict)   # layers / sensitivity / walk-forward when computed
    quote: Optional[Quote] = None              # the live price the orders are priced from (None: the last close)
    partial_bar: Optional[dict] = None         # today's unfinished session, left out of the signals
    risk_free: Optional[pd.Series] = None      # daily 3-month T-bill rate behind the Sharpe ratios (None: a fixed 2%)
    name: str = ""                             # the company's name when the price feed gives one (else the ticker)

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
                         fundamentals_pit=self.pit_scores, risk_free=self.risk_free)
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
        """Write the report, tables and charts (one interactive HTML file) to a folder. Returns the paths written."""
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
            from .interactive import all_figures
            figs = all_figures(self)
            parts = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                     for i, f in enumerate(figs.values())]
            path = out / f"{self.ticker}_charts.html"
            path.write_text("<!doctype html><html><head><meta charset='utf-8'><title>"
                            f"{self.ticker} charts</title></head><body>" + "\n".join(parts) + "</body></html>")
            written.append(str(path))
        return written


def analyze(ticker: str, profile: Optional[InvestorProfile] = None, params: Optional[StrategyParams] = None,
            years: float = 10, benchmark: str = "^GSPC", source: str = "auto", run_backtest_: bool = True,
            as_of=None, live_quote: bool = True, refresh: bool = False, keep_partial: bool = False) -> AnalysisResult:
    """Analyse one ticker end-to-end and return everything the product shows.

    Parameters
    ----------
    ticker       : e.g. "MSFT"
    profile      : the investor's horizon / risk tolerance / max loss / account value and cash available
    years        : history to analyse and backtest
    benchmark    : market index for regime detection and comparison
    source       : "auto" (live with offline fallbacks), "yfinance", "stooq", "cache", "sample"
    as_of        : optional date - "what would Lighthouse have said on this day?"
    live_quote   : price "buy now" orders from the feed's latest quote (with its timestamp) instead of the last close.
                   The signals always use finished sessions: a session still trading is left out of the indicators.
    refresh      : download prices again even if the cache is fresh
    keep_partial : keep today's unfinished bar in the series (the position replay needs today's range; the signals
                   and the backtest were designed on finished sessions, so the default leaves it out)
    """
    prof = profile or InvestorProfile()
    p = params or StrategyParams.for_horizon(prof.horizon)
    ticker = ticker.upper().strip()
    warnings: List[str] = []

    prices = load_prices(ticker, years=years, end=as_of, source=source, refresh=refresh, keep_partial=keep_partial)
    data_attrs = dict(prices.attrs)          # source, cleaning report, quote: put back on the result at the end
    rep = data_attrs.get("cleaning")
    if rep is not None:
        warnings += rep.notes
    partial = data_attrs.get("partial_bar")
    quote = None
    if live_quote and as_of is None:
        quote = data_attrs.get("quote")
        # a separate quote only for prices that are current (a fresh cache); after a fallback to old or offline data
        # it would cost another network timeout, and pair today's price with stale signals
        if quote is None and data_attrs.get("source") in {"yfinance", "stooq", "cache"}:
            quote = load_quote(ticker, source)
    if len(prices) < p.min_history:
        raise DataUnavailable(f"{ticker}: only {len(prices)} trading days available; need at least {p.min_history} "
                              f"for the 200-day trend and 52-week high indicators")

    bench = None
    try:
        # same calendar window as the stock (offline sample series end on different dates)
        bench = load_benchmark(benchmark, years=None, start=prices.index[0], end=prices.index[-1], source=source,
                               keep_partial=keep_partial)
        prices, bench = align(prices, bench)
        bench.attrs = {}
    except DataUnavailable as exc:
        warnings.append(f"Benchmark {benchmark} unavailable ({exc}); market-regime rules disabled.")
        bench = None
    # pandas deep-copies attrs into every derived object: only the name travels through the computations
    prices.attrs = {"ticker": ticker}

    ind = compute_indicators(prices, p)
    tscores = technical_score(ind, prof.horizon, p)
    t_beacons = latest_beacons(ind, prof.horizon, p)
    last = ind.iloc[-1]
    t_now = float(tscores.iloc[-1])
    if not np.isfinite(t_now):
        raise DataUnavailable(f"{ticker}: indicators are not fully defined on the last date; need more history")

    snap, hist, pit = _fundamentals(ticker, prices, source, data_attrs.get("source"), p, warnings)
    f_beacons = fundamental_beacons(snap) if snap is not None else []
    f_now = fundamental_score(f_beacons) if f_beacons else None

    regimes = regime_table(bench, p) if bench is not None else None
    regime_now: Optional[RegimeSnapshot] = latest_regime(regimes, benchmark) if regimes is not None else None

    close = float(last["Close"])
    order_price = close
    if quote is not None:
        q_day = quote.timestamp.tz_convert(EXCHANGE_TZ).date()
        move = quote.price / close - 1.0
        if q_day < prices.index[-1].date():
            quote = None                      # older than the data: nothing to add
        elif abs(move) > 0.25:
            warnings.append(f"The live quote ({quote.describe()}) is {move * 100:+.0f}% from the {prices.index[-1].date()} close: "
                            "check for a split or a bad tick. Orders are priced from the close instead.")
            quote = None
        else:
            order_price = float(quote.price)
            if q_day > prices.index[-1].date() or abs(move) > 0.0005:
                warnings.append(f"Orders are priced from the live quote {quote.describe()}, {move * 100:+.1f}% vs the "
                                f"{prices.index[-1].date()} close; the signals use finished sessions through {prices.index[-1].date()}. "
                                f"{quote.note}")

    decision = make_decision(ticker, prices.index[-1], order_price, float(last["atr"]), float(last["sizing_vol"]),
                             f_now, t_now, f_beacons + t_beacons, regime_now, prof, p, warnings,
                             sma_slow=float(last["sma_slow"]))

    # the T-bill rate for the Sharpe ratios, when the prices are current (live or a fresh cache); offline and sample
    # data keep the fixed StrategyParams.risk_free_rate so their published numbers stay reproducible
    risk_free = load_risk_free(source) if data_attrs.get("source") in {"yfinance", "stooq", "cache"} else None
    bt = None
    if run_backtest_:
        try:
            bt = run_backtest(prices, bench, p, prof, pit, risk_free=risk_free)
        except ValueError as exc:
            warnings.append(f"Backtest skipped: {exc}")

    prices.attrs = {**data_attrs, "ticker": ticker}
    name = data_attrs.get("name") or ticker
    return AnalysisResult(ticker=ticker, profile=prof, params=p, prices=prices, indicators=ind, technical_scores=tscores,
                          benchmark=bench, regimes=regimes, decision=decision, backtest=bt,
                          warnings=list(dict.fromkeys(warnings)), quote=quote, partial_bar=partial, risk_free=risk_free,
                          name=name, fundamentals=snap, statements=hist, pit_scores=pit)


def _fundamentals(ticker: str, prices: pd.DataFrame, source: str, price_source: Optional[str], params: StrategyParams,
                  warnings: List[str]) -> Tuple[Optional[FundamentalSnapshot], Optional[StatementHistory], Optional[pd.Series]]:
    """(today's fundamental snapshot, the statements, the point-in-time score series) from SEC EDGAR, or Nones when
    the company has no filings to read (an ETF, a foreign issuer, no SEC contact e-mail or no connection). The snapshot
    is the analyst memo's (financials_as_of on the last price, so a past `as_of` sees only the filings made by then)."""
    from .analyst.financials import financials_as_of
    from .analyst.memo import fundamental_snapshot
    from .pit import load_statements
    from .total_return import load_actions
    from .universe import UNKNOWN_SECTOR, sector_of
    src = "sample" if price_source == "sample" else source
    hist = load_statements(ticker, source=src)
    if hist.quarterly is None or hist.quarterly.empty or not hist.has_filing_dates:
        warnings += [f"{ticker}: {n}" for n in hist.notes] or [f"{ticker}: no SEC filings to score"]
        return None, None, None
    actions = load_actions(ticker, source=src)
    acts = None if actions.empty else actions
    sector = sector_of(ticker, None)
    sector = None if sector == UNKNOWN_SECTOR else sector
    try:
        snap = fundamental_snapshot(financials_as_of(ticker, hist, prices, acts), sector)
    except ValueError as exc:                 # no quarter filed by the last price
        warnings.append(f"{ticker}: no fundamental snapshot ({exc})")
        snap = None
    if acts is None:
        warnings.append(f"{ticker}: no dividend or split history, so EDGAR's per-share numbers cannot be put in the "
                        "price series' units; the backtest's P/E is left out.")
    pit = point_in_time_scores(prices, hist, sector, params, actions=acts)
    return snap, hist, pit


def analyze_many(tickers: Iterable[str], profile: Optional[InvestorProfile] = None, params: Optional[StrategyParams] = None,
                 max_workers: Optional[int] = None, **kwargs) -> Tuple[Dict[str, AnalysisResult], Dict[str, str]]:
    """`analyze` for several tickers (parallel when live). Returns ({ticker: result}, {ticker: error})."""
    return map_tickers(lambda t: analyze(t, profile, params, **kwargs), tickers, max_workers, kwargs.get("source", "auto"))
