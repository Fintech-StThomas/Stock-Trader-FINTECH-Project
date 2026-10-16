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
Live data (Yahoo prices cached in data/cache, SEC filings, FINRA short volume, Ken French factors); the price panel is
cached as Parquet. The 10-K text signals read every 10-K since 2006 (not stored; --no-text skips them) and need the
Loughran-McDonald dictionary (downloaded to the cache; free for academic research, see lighthouse/text.py). Every
signal computed is cached under data/cache/research_signals, so a second run evaluates without downloading.

The signals: the product's own (technical score and beacons: lighthouse.research.signals), the fundamental library
(SEC filings by filing date), FINRA short volume, the earnings reaction and the 10-K text features. The signals that
failed and are not on the watch list were removed in October 2026; their logged rows stay in experiments.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from lighthouse import factors as ff  # noqa: E402
from lighthouse import membership, store  # noqa: E402
from lighthouse.research import harness as H  # noqa: E402
from lighthouse.research import panel as P  # noqa: E402
from lighthouse.research import signals as S  # noqa: E402
from lighthouse.total_return import DELISTING_RULES  # noqa: E402

OUT = ROOT / "reports" / "research"
START = "2007-01-01"
FLAT_BPS = 10.0      # cost sensitivity: one cost per side for every stock (large-cap trading cost studies: 5-20 bps)


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
    """{signal: month-end x ticker} for the product's signals, cached as Parquet."""
    names = names or list(S.PRODUCT_SIGNALS)
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


def _merge(path: Path, new: pd.DataFrame, keys) -> pd.DataFrame:
    """Write `new` into the results file, replacing only its own rows (earlier runs' signals stay)."""
    if path.exists():
        old = pd.read_csv(path)
        mine = set(map(tuple, new[keys].astype(str).to_numpy()))
        old = old[[tuple(r) not in mine for r in old[keys].astype(str).to_numpy()]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


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
    ap.add_argument("--no-text", action="store_true", help="skip the 10-K text signals (reads every 10-K since 2006)")
    ap.add_argument("--product-only", action="store_true", help="only the signals the product uses (the technical "
                    "score and its beacons): no SEC or FINRA data")
    a = ap.parse_args(argv)
    uni, frames, missing, panel = load_inputs()
    cov = panel.coverage()
    print(f"{uni.summary()}; {len(frames)} priced, {len(missing)} without prices; "
          f"coverage {cov[cov.index >= '2009-01-01'].mean():.0%} of member-months 2009-{panel.dates[-1].year}")
    product = [n for n in S.PRODUCT_SIGNALS if a.signals is None or n in a.signals]
    sigs = monthly_signals(frames, panel, product, refresh=a.refresh_signals) if product else {}
    if not a.product_only:
        sigs.update(library_signals(frames, panel, a))
    return evaluate_all(a, uni, frames, missing, panel, sigs)


def _wanted(a, names) -> bool:
    """Is any of `names` asked for (`--signals`; none given: all are)?"""
    return a.signals is None or bool(set(a.signals) & set(names))


def library_signals(frames, panel, a) -> dict:
    """The signal library: fundamental (SEC filings), FINRA short volume, the earnings reaction and the 10-K text.
    With --signals, only the families that hold a requested signal are computed (no FINRA or 10-K download for a
    fundamental signal)."""
    from lighthouse.research import event_signals as EV
    from lighthouse.research import fundamental_signals as FS
    from lighthouse.research import positioning as PO
    sigs = {}
    if _wanted(a, FS.FUNDAMENTAL_SIGNALS):
        sigs.update(FS.monthly(frames, panel, refresh=a.refresh_signals))
        sigs.pop("market_cap")
    if _wanted(a, PO.DESCRIPTIONS):
        sigs.update(PO.monthly(frames, panel, refresh=a.refresh_signals))
    if _wanted(a, EV.DESCRIPTIONS):
        sigs.update(EV.monthly(frames, panel, refresh=a.refresh_signals, text=not a.no_text))
    for mod in (FS, PO, EV):
        S.DESCRIPTIONS.update(mod.DESCRIPTIONS)
    return sigs


def config_for(name: str) -> dict:
    """The logged configuration of a signal: harness.signal_config plus what changed its values since its first
    trial (fundamental_signals.config_extra: the share-count method)."""
    from lighthouse.research import fundamental_signals as FS
    return H.signal_config(name, S.DESCRIPTIONS.get(name, ""), **FS.config_extra(name))


def evaluate_all(a, uni, frames, missing, panel, sigs) -> int:
    if a.signals:
        sigs = {k: v for k, v in sigs.items() if k in a.signals}
    costs = H.cost_panel(frames, panel.dates)
    flat = H.cost_panel(frames, panel.dates, fixed_bps=FLAT_BPS)          # sensitivity only: not in the survival rule
    deaths = int((panel.fwd["pessimistic"] == -1.0).sum().sum())
    print(f"{deaths} priced members stopped trading before the data end (the pessimistic rule's -100% months); "
          f"{int((~panel.tradable & panel.members & panel.prices.notna()).sum().sum())} member-months dropped for "
          f"trading on fewer than {H.MIN_TRADED_DAYS} of 21 sessions")
    factors = ff.load_factors("monthly")
    log = H.ExperimentLog(OUT)
    dv = data_version(uni, frames, missing, panel)
    rows, tune_by = [], {}
    for name, sig in sigs.items():
        cfg = config_for(name)
        tune_by[name] = {}
        for rule in DELISTING_RULES:
            res = H.evaluate(panel, sig, name, rule, H.TUNING, costs, factors)
            log.append(res.summary, cfg, dv, rule=rule, period="tuning")
            tune_by[name][rule] = res.summary
            alt = H.evaluate(panel, sig, name, rule, H.TUNING, flat, None).summary
            rows.append({**res.summary, "period": "tuning", f"spread_net_ann_{FLAT_BPS:g}bps": alt["spread_net_ann"],
                         f"long_only_net_ann_{FLAT_BPS:g}bps": alt["long_only_net_ann"]})
    tab = pd.DataFrame(rows)
    cols = ["signal", "rule", "months", "coverage", "ic_mean", "ic_nw_t", "spread_gross_ann", "spread_net_ann",
            f"spread_net_ann_{FLAT_BPS:g}bps", "long_only_net_ann", f"long_only_net_ann_{FLAT_BPS:g}bps",
            "long_only_net_nw_t", "alpha_ann", "alpha_nw_t", "top_turnover", "monotonic"]
    OUT.mkdir(parents=True, exist_ok=True)
    tab = _merge(OUT / "signal_results_tuning.csv", tab, ["signal", "rule"])
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[cols].round(3).to_string(index=False))
    if not a.holdout:
        return 0
    # the hold-out: once per signal, both rules; trials = every configuration in the log
    sr_std = log.trial_sharpe_std()          # every configuration in the log, not only this run's signals
    n_trials = log.trials()
    hrows, verdicts = [], []
    for name, sig in sigs.items():
        cfg = config_for(name)
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
        _merge(OUT / "signal_results_holdout.csv", ht, ["signal", "rule"])
        _merge(OUT / "signal_verdicts.csv", pd.DataFrame(verdicts), ["signal"])
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            hc = [c for c in cols if c in ht.columns] + ["deflated_sharpe", "long_only_deflated_sharpe"]
            print("\nHOLD-OUT\n" + ht[hc].round(3).to_string(index=False))
            print("\n" + pd.DataFrame(verdicts).to_string(index=False))
        print(f"\n{n_trials} configurations tried; Sharpe std across trials {sr_std:.3f} (monthly)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
