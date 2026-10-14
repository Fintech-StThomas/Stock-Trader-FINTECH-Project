"""
Run signals through the research harness on the point-in-time S&P 500 (lighthouse.research.harness): monthly rank ICs,
quintile spreads after costs, Fama-French alphas, under both delisting rules, tuning period first, hold-out once.

Pre-registered survival rule (written before any hold-out was evaluated, October 2026). A signal survives, and may be
given weight in lighthouse/decision.py, only if under BOTH delisting rules:
  tuning 2009-2017    IC Newey-West t >= 2.0, and the net return of its better use (long-only top quintile minus the
                      equal-weight universe, or long-short top minus bottom quintile) is positive
  hold-out 2018-      IC mean > 0, the net return of that same use > 0, and its deflated Sharpe ratio >= 0.90 given
                      every configuration tried so far (reports/research/experiments.csv)
Everything is reported, failures included (reports/research/signal_results.csv, *_holdout.csv).

Run:  LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py              # tuning period only
      LIGHTHOUSE_SEC_EMAIL=you@example.com python scripts/research_signals.py --holdout    # then the hold-out, once
Live data (Yahoo prices cached in data/cache, SEC filings, Ken French factors); the price panel is cached as Parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lighthouse import factors as ff  # noqa: E402
from lighthouse import membership, store  # noqa: E402
from lighthouse.research import harness as H  # noqa: E402
from lighthouse.research import panel as P  # noqa: E402
from lighthouse.research import signals as S  # noqa: E402
from lighthouse.total_return import DELISTING_RULES  # noqa: E402

OUT = ROOT / "reports" / "research"
START = "2007-01-01"


def survival(tune: dict, hold: dict) -> dict:
    """The pre-registered rule (module docstring). tune / hold: {rule: summary}."""
    use = "long_only" if all(tune[r]["long_only_net_ann"] >= tune[r]["spread_net_ann"] for r in DELISTING_RULES) else "long_short"
    key = "long_only_net_ann" if use == "long_only" else "spread_net_ann"
    dsr_key = "long_only_deflated_sharpe" if use == "long_only" else "deflated_sharpe"
    t_ok = all(tune[r]["ic_nw_t"] >= 2.0 and tune[r][key] > 0 for r in DELISTING_RULES)
    if not hold:
        return {"use": use, "tuning_pass": t_ok, "holdout_pass": None, "survives": None}
    h_ok = all(hold[r]["ic_mean"] > 0 and hold[r][key] > 0 and hold[r].get(dsr_key, 0) >= 0.90 for r in DELISTING_RULES)
    return {"use": use, "tuning_pass": t_ok, "holdout_pass": h_ok, "survives": bool(t_ok and h_ok)}


def load_inputs(refresh_signals: bool = False):
    uni = membership.sp500()
    tickers = uni.ever_members("2008-01-01")
    frames, missing = P.load_frames(tickers, start=START)
    frames = {t: df for t, df in frames.items() if len(df) > 260}
    panel = H.build_panel(uni, frames, start="2008-01-01")
    return uni, frames, missing, panel


def _no_signal(hit, frames) -> set:
    """Tickers too short for any signal (they are absent from a cached table on purpose)."""
    return {t for t, df in frames.items() if len(df) < 260}


def monthly_signals(frames, panel, names=None, refresh=False) -> dict:
    """{signal: month-end x ticker} for the product and price signals, cached as Parquet."""
    names = names or (S.PRODUCT_SIGNALS + list(S.PRICE_SIGNALS))
    out, todo = {}, []
    for n in names:
        hit = None if refresh else store.read_frame("research_signals", n)
        if hit is not None and hit.index.equals(panel.dates) and set(frames) <= set(hit.columns) | _no_signal(hit, frames):
            out[n] = hit
        else:
            todo.append(n)
    if todo:
        daily = S.daily_signals(frames, todo)
        for n in todo:
            m = H.sample_at_month_ends(daily[n], panel.dates)
            store.write_frame("research_signals", n, m)
            out[n] = m
    return out


def fundamental_signal(frames, panel, refresh=False) -> pd.DataFrame:
    """The product's point-in-time fundamental score at month ends (SEC filing dates; Yahoo-sector norms where the
    curated sector list knows the stock, the default norms otherwise)."""
    hit = None if refresh else store.read_frame("research_signals", "fund_pit")
    if hit is not None and hit.index.equals(panel.dates):
        return hit
    from lighthouse.fundamentals import point_in_time_scores
    from lighthouse.pit import load_statements
    from lighthouse.total_return import load_actions
    from lighthouse.universe import SECTORS
    from lighthouse.config import StrategyParams
    from lighthouse.parallel import map_tickers

    def one(t):
        h = load_statements(t)
        if not h.has_filing_dates:
            return None
        acts = load_actions(t, max_age_hours=None)
        return point_in_time_scores(frames[t], h, SECTORS.get(t), StrategyParams(), actions=None if acts.empty else acts)
    got, _ = map_tickers(one, list(frames), max_workers=8)
    daily = {t: v for t, v in got.items() if v is not None}
    m = H.sample_at_month_ends(daily, panel.dates)
    store.write_frame("research_signals", "fund_pit", m)
    return m


def data_version(uni, frames, missing, panel) -> dict:
    return {"universe": uni.name, "universe_as_of": str(uni.as_of.date()) if uni.as_of is not None else None,
            "sources": uni.sources, "tickers_priced": len(frames), "tickers_missing": len(missing),
            "prices_end": str(panel.data_end.date()), "months": len(panel.dates),
            "factors": ff.CITATION}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--holdout", action="store_true", help="evaluate the hold-out (once per signal; logged)")
    ap.add_argument("--signals", nargs="*", default=None)
    ap.add_argument("--refresh-signals", action="store_true")
    ap.add_argument("--allow-repeat", action="store_true", help="re-evaluate a hold-out (counts as another trial)")
    a = ap.parse_args(argv)
    uni, frames, missing, panel = load_inputs()
    cov = panel.coverage()
    print(f"{uni.summary()}; {len(frames)} priced, {len(missing)} without prices; "
          f"coverage {cov[cov.index >= '2009-01-01'].mean():.0%} of member-months 2009-{panel.dates[-1].year}")
    sigs = monthly_signals(frames, panel, refresh=a.refresh_signals)
    if os.environ.get("LIGHTHOUSE_SEC_EMAIL") or os.environ.get("SEC_USER_AGENT"):
        sigs["fund_pit"] = fundamental_signal(frames, panel, a.refresh_signals)
        tech = sigs["tech_long"]
        f = sigs["fund_pit"].reindex_like(tech)
        sigs["composite_long"] = (0.6 * f + 0.4 * tech).where(f.notna(), tech)
    from lighthouse.research import fundamental_signals as FS
    from lighthouse.research import panel_signals as PS
    from lighthouse.research import positioning as PO
    fs = FS.monthly(frames, panel, refresh=a.refresh_signals)
    mcap = fs.pop("market_cap")
    sigs.update(fs)
    sigs["size"] = -np.log(mcap.where(mcap > 0))                     # small minus big (Banz 1981)
    sigs.update(PS.monthly(frames, panel, refresh=a.refresh_signals))
    sigs.update(PO.monthly(frames, panel, market_cap=mcap, refresh=a.refresh_signals))
    for mod in (FS, PS, PO):
        S.DESCRIPTIONS.update(mod.DESCRIPTIONS)
    S.DESCRIPTIONS["size"] = "small size (minus log market value)"
    if a.signals:
        sigs = {k: v for k, v in sigs.items() if k in a.signals}
    costs = H.cost_panel(frames, panel.dates)
    factors = ff.load_factors("monthly")
    log = H.ExperimentLog(OUT)
    dv = data_version(uni, frames, missing, panel)
    rows, tune_by = [], {}
    for name, sig in sigs.items():
        cfg = {"signal": name, "universe": "sp500_pit", "quantiles": 5, "costs": "cs_spread+sqrt_impact@1m/100",
               "borrow": 0.005, "description": S.DESCRIPTIONS.get(name, "")}
        tune_by[name] = {}
        for rule in DELISTING_RULES:
            res = H.evaluate(panel, sig, name, rule, H.TUNING, costs, factors)
            log.append(res.summary, cfg, dv, rule=rule, period="tuning")
            tune_by[name][rule] = res.summary
            rows.append({**res.summary, "period": "tuning"})
    tab = pd.DataFrame(rows)
    cols = ["signal", "rule", "months", "coverage", "ic_mean", "ic_nw_t", "spread_net_ann", "spread_net_sharpe",
            "long_only_net_ann", "long_only_net_nw_t", "alpha_ann", "alpha_nw_t", "top_turnover", "monotonic"]
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "signal_results_tuning.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[cols].round(3).to_string(index=False))
    if not a.holdout:
        return 0
    # the hold-out: once per signal, both rules; trials = every configuration in the log
    trial_sr = []
    for name, by in tune_by.items():
        for r, s in by.items():
            for k in ("spread_net_sharpe", "long_only_net_sharpe"):
                if np.isfinite(s.get(k, np.nan)):
                    trial_sr.append(s[k] / np.sqrt(12))
    sr_std = float(np.std(trial_sr, ddof=1)) if len(trial_sr) > 2 else 0.1
    n_trials = log.trials()
    hrows, verdicts = [], []
    for name, sig in sigs.items():
        cfg = {"signal": name, "universe": "sp500_pit", "quantiles": 5, "costs": "cs_spread+sqrt_impact@1m/100",
               "borrow": 0.005, "description": S.DESCRIPTIONS.get(name, "")}
        try:
            res = H.evaluate_holdout(panel, sig, name, cfg, log, dv, costs, factors, allow_repeat=a.allow_repeat,
                                     n_trials=n_trials, sr_std=sr_std)
        except H.HoldoutUsed as exc:
            print(f"{name}: {exc}")
            continue
        hold = {r: v.summary for r, v in res.items()}
        v = survival(tune_by[name], hold)
        verdicts.append({"signal": name, **v})
        hrows += [{**hold[r], "period": "holdout"} for r in DELISTING_RULES]
    if hrows:
        ht = pd.DataFrame(hrows)
        ht.to_csv(OUT / "signal_results_holdout.csv", index=False)
        pd.DataFrame(verdicts).to_csv(OUT / "signal_verdicts.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print("\nHOLD-OUT\n" + ht[cols + ["deflated_sharpe", "long_only_deflated_sharpe"]].round(3).to_string(index=False))
            print("\n" + pd.DataFrame(verdicts).to_string(index=False))
        print(f"\n{n_trials} configurations tried; Sharpe std across trials {sr_std:.3f} (monthly)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
