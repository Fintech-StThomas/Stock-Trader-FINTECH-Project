"""Command-line interface.

  python -m lighthouse analyze MSFT --horizon long --risk balanced --max-loss 2 --cash 5000 --holdings 20000 --out reports
  python -m lighthouse screen AAPL MSFT NVDA JPM --risk conservative
  python -m lighthouse backtest MSFT --years 15 --evaluate
  python -m lighthouse trade MSFT AAPL JNJ --side long --cash 5000 --holdings 20000      # buy-now tickets for your cash
  python -m lighthouse monitor MSFT --side long --entry 73.9 --shares 30 --date 2017-09-01 --source sample
  python -m lighthouse portfolio AAPL JNJ XOM JPM KO PG --method risk_parity --cash 10000 --target-vol 0.12
  python -m lighthouse portfolio --goal balanced --cash 10000                           # Lighthouse picks the stocks
  python -m lighthouse trade --universe semiconductors --cash 5000                      # a pre-built universe
  python -m lighthouse universes                                                         # list the pre-built universes
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .config import InvestorProfile, StrategyParams
from .engine import analyze
from .report import format_metrics
from .screener import screen


def _profile(a) -> InvestorProfile:
    if a.cash is not None:
        return InvestorProfile.from_cash(a.cash, a.holdings, horizon=a.horizon, risk=a.risk, max_loss_pct=a.max_loss)
    # legacy --portfolio: the whole account is treated as cash available to invest
    return InvestorProfile(horizon=a.horizon, risk=a.risk, max_loss_pct=a.max_loss, portfolio_value=a.portfolio + a.holdings,
                           cash_available=a.portfolio)


def _params(a) -> StrategyParams:
    # the horizon chooses the rule preset (short = weekly review + ATR stop); --params overrides on top of it
    p = StrategyParams.for_horizon(a.horizon)
    if getattr(a, "params", None):
        p = p.copy(**json.loads(a.params))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lighthouse", description="Lighthouse - explainable stock decisions")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--horizon", choices=["long", "short"], default="long")
        sp.add_argument("--risk", choices=["conservative", "balanced", "aggressive"], default="balanced")
        sp.add_argument("--max-loss", type=float, default=2.0, help="max loss per position, %% of the total account")
        sp.add_argument("--cash", type=float, default=None, help="cash available to invest now ($)")
        sp.add_argument("--holdings", type=float, default=0.0, help="value of stocks you already hold ($); account = cash + holdings")
        sp.add_argument("--portfolio", type=float, default=10_000, help="legacy alias for --cash (used when --cash is not given)")
        sp.add_argument("--years", type=float, default=10)
        sp.add_argument("--source", default="auto", help="auto | yfinance | stooq | cache | sample")
        sp.add_argument("--params", help='JSON overrides, e.g. \'{"entry_score": 65}\'')

    a1 = sub.add_parser("analyze", help="full analysis of one ticker"); common(a1)
    a1.add_argument("ticker"); a1.add_argument("--out", help="folder for report + charts")
    a1.add_argument("--evaluate", action="store_true", help="also run layer attribution, sensitivity and walk-forward tests")
    a1.add_argument("--as-of", help="YYYY-MM-DD: what would Lighthouse have said on that day?")
    a1.add_argument("--fundamentals-json", help="JSON fixture/override file for fundamentals")

    def universe_opt(sp):
        sp.add_argument("--universe", help="add a pre-built universe's tickers (see: python -m lighthouse universes)")

    a2 = sub.add_parser("screen", help="rank a watchlist"); common(a2); universe_opt(a2)
    a2.add_argument("tickers", nargs="*"); a2.add_argument("--csv", help="write the ranking to this file")

    a3 = sub.add_parser("backtest", help="backtest the rule on one ticker"); common(a3)
    a3.add_argument("ticker"); a3.add_argument("--evaluate", action="store_true")

    def trade_opts(sp):
        sp.add_argument("--side", choices=["long", "short"], default="long")
        sp.add_argument("--targets", type=float, nargs=2, default=[2.0, 3.0], metavar=("R1", "R2"), help="take-profit levels in R")
        sp.add_argument("--partial", type=float, default=0.5, help="share of the position sold at target 1")
        sp.add_argument("--time-stop", type=int, default=None, help="sessions (default: 40 short horizon / 252 long)")

    a4 = sub.add_parser("trade", help="trade tickets for your cash across one or more tickers"); common(a4); trade_opts(a4)
    universe_opt(a4)
    a4.add_argument("tickers", nargs="*")
    a4.add_argument("--max-ideas", type=int, default=5)
    a4.add_argument("--max-total-risk", type=float, default=6.0, help="cap on all ideas' planned loss, %% of the account")

    a5 = sub.add_parser("monitor", help="replay an open position against its plan (exits taken, P&L, stop now)"); common(a5); trade_opts(a5)
    a5.add_argument("ticker"); a5.add_argument("--entry", type=float, required=True); a5.add_argument("--shares", type=int, required=True)
    a5.add_argument("--date", required=True, help="entry date YYYY-MM-DD"); a5.add_argument("--stop", type=float)

    a6 = sub.add_parser("portfolio", help="allocate a basket, or build one for a goal, with a risk report"); common(a6)
    universe_opt(a6)
    a6.add_argument("tickers", nargs="*")
    a6.add_argument("--method", default="risk_parity", help="equal_weight | inverse_vol | risk_parity | min_variance | max_sharpe | hrp")
    a6.add_argument("--goal", help="preservation | balanced | growth | diversified (Lighthouse picks the stocks)")
    a6.add_argument("--max-weight", type=float, default=None); a6.add_argument("--max-sector", type=float, default=None)
    a6.add_argument("--target-vol", type=float, default=None, help="annual, e.g. 0.12; holds cash above it")
    a6.add_argument("--n", type=int, default=None, help="number of stocks for --goal")

    sub.add_parser("universes", help="list the pre-built universes")

    a = ap.parse_args(argv)
    pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)

    from .universe import get_universe, universe_names
    if a.cmd == "universes":
        for name in universe_names("auto"):
            tick = get_universe(name)
            print(f"{name} ({len(tick)}): {' '.join(tick)}")
        return 0
    if getattr(a, "universe", None):
        try:
            a.tickers = list(dict.fromkeys([t.upper() for t in a.tickers] + get_universe(a.universe, a.source)))
        except ValueError as exc:
            print(exc, file=sys.stderr); return 1
    if a.cmd in {"trade", "screen"} and not a.tickers:
        print("give tickers, or --universe NAME", file=sys.stderr); return 1

    if a.cmd in {"analyze", "backtest"}:
        res = analyze(a.ticker, _profile(a), _params(a), years=a.years, source=a.source,
                      as_of=getattr(a, "as_of", None), fundamentals_fixture=getattr(a, "fundamentals_json", None))
        if a.evaluate:
            print("running layer attribution, sensitivity grid and walk-forward test ...", file=sys.stderr)
            res.evaluate()
        if a.cmd == "analyze":
            print(res.report())
            if a.out:
                for pth in res.save(a.out):
                    print(f"wrote {pth}", file=sys.stderr)
        else:
            if res.backtest is None:
                print("backtest unavailable"); return 1
            print(format_metrics(res.backtest.summary()).to_string())
            if "layers" in res.extras:
                print("\nLayer attribution\n" + format_metrics(res.extras["layers"]).to_string())
            if "walk_forward" in res.extras:
                wf = res.extras["walk_forward"]
                print(f"\nOut-of-sample after {wf['split_date'].date()}\n" + format_metrics(wf["out_of_sample"]).to_string())
        return 0

    if a.cmd in {"trade", "monitor"}:
        from .tradeplan import TradeSettings, allocate_ideas, ideas_table, monitor_position, _signed_usd
        s = TradeSettings(reward_multiples=tuple(sorted(a.targets)), partial_exit=a.partial, time_stop_days=a.time_stop,
                          **({"max_ideas": a.max_ideas, "max_total_risk_pct": a.max_total_risk} if a.cmd == "trade" else {}))
        prof = _profile(a)
        if a.cmd == "monitor":
            res = analyze(a.ticker, prof, _params(a), years=a.years, source=a.source, run_backtest_=False)
            ms = monitor_position(res, a.side, a.entry, a.shares, a.date, stop=a.stop, settings=s)
            print(f"{ms.ticker} {ms.side} · {ms.status}\n{ms.action}")
            for day, n, px, why in ms.exits:
                print(f"  {day}: {why}, {n} shares at ${px:,.2f}")
            print(f"last ${ms.last_price:,.2f} · P&L {_signed_usd(ms.pnl)} ({ms.pnl_pct:+.1f}%, {ms.r_multiple:+.2f}R): "
                  f"realised {_signed_usd(ms.realised_pnl)}, unrealised {_signed_usd(ms.unrealised_pnl)} on {ms.shares_open} open shares · "
                  f"stop now ${ms.current_stop:,.2f} ({ms.stop_note})")
            for x in ms.alerts:
                print("! " + x)
            return 0
        results = []
        for t in a.tickers:
            try:
                results.append(analyze(t, prof, _params(a), years=a.years, source=a.source, run_backtest_=False))
            except Exception as exc:  # noqa: BLE001
                print(f"{t}: {type(exc).__name__}: {exc}", file=sys.stderr)
        funded, watch, table = allocate_ideas(results, a.side, s, with_stats=True)
        print(prof.describe())
        for pl in funded:
            print(f"\n=== {pl.side.upper()} {pl.ticker}: {pl.shares} shares ===")
            print("\n".join(f"{i}. {x}" for i, x in enumerate(pl.management_steps(), 1)))
            print("when to get out:")
            for x in pl.exit_plan():
                lvl = f" at ${x.price:,.2f} ({x.move_pct:+.1f}%)" if x.price is not None else ""
                res_ = f" -> {_signed_usd(x.pnl)} ({x.r:+.2f}R) if the trade ends here" if x.pnl is not None else ""
                print(f"  - {x.name}{lvl}{res_}" + (f" · history: {x.history}" if x.history else ""))
            if pl.stats:
                print(f"history: {pl.stats.n_trades} trades, win {pl.stats.win_rate:.0%}, {pl.stats.avg_r:+.2f}R per trade")
            for w in pl.warnings[:2]:
                print("! " + w)
        if not funded:
            print(f"no actionable {a.side} setup in {', '.join(a.tickers)}")
        if len(table):
            print("\n" + table[["ticker", "status", "score", "shares", "cost", "stop", "target1", "target2", "risk_usd"]].to_string(index=False))
        return 0

    if a.cmd == "portfolio":
        from . import portfolio as pf
        cash = a.cash if a.cash is not None else a.portfolio
        if a.goal:
            g = pf.GOALS[a.goal]
            cons = pf.PortfolioConstraints(max_weight=a.max_weight or g.max_weight, max_sector_weight=a.max_sector or g.max_sector_weight,
                                           target_vol=a.target_vol if a.target_vol is not None else g.target_vol)
            res = pf.goal_portfolio(a.goal, a.tickers or None, capital=cash, years=a.years, source=a.source, n_stocks=a.n, constraints=cons)
        else:
            if not a.tickers:
                print("give tickers, or --goal"); return 1
            cons = pf.PortfolioConstraints(max_weight=a.max_weight or 0.25, max_sector_weight=a.max_sector, target_vol=a.target_vol)
            res = pf.build_portfolio(a.tickers, a.method, cons, capital=cash, years=a.years, source=a.source)
        print(f"{res.label} · ${res.capital:,.0f} · prices as of {res.as_of}")
        for n in res.notes:
            print("- " + n)
        print(res.table().round(4).to_string())
        r = res.risk
        print(f"expected vol {r['expected_vol']:.1%} · 1-day 95% VaR ${r.get('var95_1d_usd', 0):,.0f} · ES ${r.get('es95_1d_usd', 0):,.0f} · "
              f"beta {r.get('beta', float('nan')):.2f} · cash left ${res.cash_left:,.0f}")
        if res.comparison is not None:
            print("\nWalk-forward test\n" + format_metrics(res.comparison).to_string())
        return 0

    if a.cmd == "screen":
        df = screen(a.tickers, _profile(a), _params(a), years=a.years, source=a.source)
        show = df[["ticker", "name", "verdict", "score", "fundamental", "technical", "regime", "position_pct", "price", "error"]]
        print(show.to_string())
        if a.csv:
            df.to_csv(a.csv)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
