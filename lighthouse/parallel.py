"""
Run one function over many tickers, in parallel threads when the data comes over the network. Downloads overlap, so a
live scan of a large universe finishes several times sooner; offline work is pure CPU, where Python threads only add
contention, so it stays sequential.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, Optional, Tuple

from .data import offline_mode


def default_workers(n: int, source: str = "auto") -> int:
    """Threads for loading several tickers: downloads overlap, so live loads use up to 8; offline work is pure CPU
    (and Python's GIL), where threads only add contention."""
    if n <= 1 or offline_mode() or source in {"sample", "cache"}:
        return 1
    return min(8, n)


def map_tickers(fn: Callable[[str], object], tickers: Iterable[str], max_workers: Optional[int] = None,
                source: str = "auto", on_done: Optional[Callable[[str, int, int], None]] = None,
                initializer: Optional[Callable[[], None]] = None) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Run `fn(ticker)` for every ticker, in parallel threads when live. Returns ({ticker: result}, {ticker: error}) in
    the tickers' order; one failure never stops the others. `on_done(ticker, k, n)` reports progress (always from the
    calling thread); `initializer` runs once in each worker thread (the web app attaches its session context there)."""
    order = list(dict.fromkeys(t.upper().strip() for t in tickers if t and t.strip()))
    workers = max_workers or default_workers(len(order), source)
    out: Dict[str, object] = {}
    errors: Dict[str, str] = {}

    def run(t: str):
        try:
            return t, fn(t), None
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the batch
            return t, None, f"{type(exc).__name__}: {str(exc)[:160]}"

    if workers <= 1:
        results = []
        for k, t in enumerate(order, 1):
            results.append(run(t))
            if on_done:
                on_done(t, k, len(order))
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers, initializer=initializer) as pool:
            futures = [pool.submit(run, t) for t in order]
            for k, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if on_done:
                    on_done(results[-1][0], k, len(order))
    by = {t: (r, e) for t, r, e in results}
    for t in order:
        r, e = by[t]
        if e is None:
            out[t] = r
        else:
            errors[t] = e
    return out, errors

