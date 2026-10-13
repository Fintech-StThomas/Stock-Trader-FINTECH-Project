"""
Measure how far behind the price each candidate band centre runs, and how rough it is, on the offline sample pack.

For every stock in the pack (20 S&P 500 names) the last 15 years of log closes are filtered by each candidate, and:

  best lag     the shift k (sessions, refined between whole sessions) for which the centre on day t sits closest (RMSE)
               to the close of day t - k: "the line shows where price was k sessions ago"
  turn delay   sessions from a price swing high or low (the extreme of +/- 10 sessions) until the centre turns too
  half a jump  sessions until the centre covers half of a one-off jump in price (0 = on the day of the jump)
  overshoot    how far past a one-off jump the centre swings before it settles (share of the jump)
  in a trend   sessions the centre trails a price rising in a straight line
  roughness    standard deviation of the centre's daily change / that of the close (1 = as jumpy as the price itself)

The first two are averages over the 20 stocks; the jump and trend columns come from synthetic prices.

Run:  LIGHTHOUSE_OFFLINE=1 python scripts/band_lag_evaluation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LIGHTHOUSE_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lighthouse import indicators as ind  # noqa: E402
from lighthouse.config import StrategyParams  # noqa: E402
from lighthouse.data import _from_sample, sample_pack_tickers  # noqa: E402

P = StrategyParams()


def _linreg_end(s: pd.Series, n: int) -> pd.Series:
    """End point of a least-squares line through the last n closes."""
    x = np.arange(n, dtype=float)
    w = np.vander(x, 2)[-1] @ np.linalg.pinv(np.vander(x, 2))
    return s.rolling(n, min_periods=n).apply(lambda v: float(w @ v), raw=True)


CANDIDATES = {
    "SMA 20 (Bollinger)": lambda c: ind.sma(c, 20),
    "Hull MA 20 (until Oct 2026)": lambda c: ind.hull_ma(c, 20),
    "Hull MA 10": lambda c: ind.hull_ma(c, 10),
    "Linear-regression end 20": lambda c: _linreg_end(c, 20),
    "EMA 5": lambda c: ind.ema(c, 5),
    "Holt 0.3 / 0.1": lambda c: ind.holt_trend(c, 0.3, 0.1),
    f"Holt {P.band_alpha:g} / {P.band_beta:g} (now)": lambda c: ind.holt_trend(c, P.band_alpha, P.band_beta),
    "Holt 0.5 / 0.1": lambda c: ind.holt_trend(c, 0.5, 0.1),
}


def best_lag(mid: pd.Series, close: pd.Series, kmax: int = 15) -> float:
    rmse = np.array([np.sqrt(((mid - close.shift(k)).dropna() ** 2).mean()) for k in range(kmax + 1)])
    k = int(np.argmin(rmse))
    if 0 < k < kmax:                                       # parabola through the three points around the minimum
        a, b, c = rmse[k - 1:k + 2]
        if a - 2 * b + c > 0:
            return k + 0.5 * (a - c) / (a - 2 * b + c)
    return float(k)


def turn_delay(mid: pd.Series, close: pd.Series, w: int = 10, maxd: int = 15) -> float:
    c, m, delays = close.to_numpy(), mid.to_numpy(), []
    for i in range(w, len(c) - w - maxd):
        win = c[i - w:i + w + 1]
        peak, trough = c[i] == win.max(), c[i] == win.min()
        if not (peak or trough):
            continue
        d = next((j for j in range(maxd) if (peak and m[i + j + 1] < m[i + j]) or (trough and m[i + j + 1] > m[i + j])), maxd)
        delays.append(d)
    return float(np.mean(delays))


def step_response(f) -> tuple:
    s = f(pd.Series(np.r_[np.zeros(150), np.ones(150)])).to_numpy()[150:]
    ramp = pd.Series(np.arange(300, dtype=float))
    return int(np.argmax(s >= 0.5)), float(s.max() - 1.0), float((ramp - f(ramp)).iloc[-1])


def build() -> pd.DataFrame:
    closes = {t: np.log(_from_sample(t)["Close"].astype(float)).iloc[-252 * 15:]
              for t in sample_pack_tickers() if not t.startswith("^")}
    rows = []
    for name, f in CANDIDATES.items():
        lags, turns, rough = [], [], []
        for c in closes.values():
            m, cc = f(c).iloc[60:], c.iloc[60:]
            lags.append(best_lag(m, cc))
            turns.append(turn_delay(m, cc))
            rough.append(m.diff().std() / c.diff().std())
        half, over, trend = step_response(f)
        rows.append({"centre": name, "best lag": np.mean(lags), "turn delay": np.mean(turns), "half a jump": half,
                     "overshoot": over, "in a trend": trend, "roughness": np.mean(rough)})
    return pd.DataFrame(rows).set_index("centre")


def main() -> int:
    df = build()
    print(f"{len(sample_pack_tickers()) - 1} sample stocks, last 15 years each (sessions; roughness 1 = the price itself)\n")
    print(df.to_string(formatters={"best lag": "{:.1f}".format, "turn delay": "{:.1f}".format,
                                   "overshoot": "{:.0%}".format,
                                   "in a trend": lambda v: f"{abs(v):.1f}", "roughness": "{:.2f}".format}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
