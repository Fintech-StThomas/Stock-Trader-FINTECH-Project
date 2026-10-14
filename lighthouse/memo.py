"""
A small in-process memo for the pure, frame-in / frame-out computations that the engine repeats: indicators, beacon
scores and the regime table. A deep evaluation runs ~80 backtests on the same prices that differ only in their trading
thresholds; without the memo each of them recomputed every indicator from scratch.

Keys are content fingerprints (shape, first/last date, column sums), never object ids, and every hit returns a copy,
so a caller that changes its frame afterwards can neither see nor corrupt another caller's result.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import asdict
from typing import Callable, Hashable, Iterable, Optional, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")

# StrategyParams fields that only steer trading rules, sizing, costs or evaluation. They never change an indicator, a
# beacon score or the regime table, so they are left out of the keys (a sensitivity grid varies exactly these).
# A field NOT listed here is part of the key: a new parameter can only cost a cache miss, never a stale result.
RULE_ONLY_FIELDS = frozenset({
    "rebalance_threshold", "entry_score", "exit_score", "trend_gate", "exit_below_sma", "atr_stop",
    "decision_frequency", "score_smoothing", "trend_exit", "max_entry_distance", "bear_entry_penalty",
    "bear_size_multiplier", "block_entries_in_bear_turbulent", "fundamental_gate_min", "reporting_lag_days",
    "bear_score_penalty", "turbulent_score_penalty", "cost_bps", "slippage_bps", "risk_free_rate",
    "cost_model", "order_value", "resting_stops", "cash_yield", "short_borrow_fee",
})

_MAX_ENTRIES = 96
_store: "OrderedDict[Hashable, object]" = OrderedDict()
_lock = threading.Lock()


def frame_fingerprint(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> tuple:
    """Shape, first and last date, and per numeric column its sum and its position-weighted sum (so reordered values
    also change the key): identical data gives an identical key."""
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    num = df.select_dtypes(include="number")
    sums: tuple = ()
    if num.shape[1]:
        x = num.to_numpy(dtype=float)
        w = np.arange(1, len(x) + 1, dtype=float)[:, None]
        sums = tuple(float(v) for v in np.concatenate([np.nansum(x, axis=0), np.nansum(x * w, axis=0)]))
    ends = (df.index[0], df.index[-1]) if len(df) else (None, None)
    return (df.shape, tuple(map(str, num.columns)), ends, sums)


def params_key(params) -> tuple:
    return tuple(sorted((k, v) for k, v in asdict(params).items() if k not in RULE_ONLY_FIELDS))


def memoize(key: Hashable, build: Callable[[], T]) -> T:
    """Return a copy of the cached value for `key`, building (and caching) it on a miss."""
    with _lock:
        hit = _store.get(key)
        if hit is not None:
            _store.move_to_end(key)
    if hit is None:
        hit = build()
        with _lock:
            _store[key] = hit
            _store.move_to_end(key)
            while len(_store) > _MAX_ENTRIES:
                _store.popitem(last=False)
    return hit.copy() if hasattr(hit, "copy") else hit


def clear() -> None:
    with _lock:
        _store.clear()
