import os
import numpy as np
import pandas as pd
import pytest

os.environ["LIGHTHOUSE_OFFLINE"] = "1"
# The suite builds trade tickets on the bundled sample prices to test their mechanics, which the live-mode rule refuses
# without the override; tests/test_live_mode.py removes it to test the refusal itself.
os.environ["LIGHTHOUSE_ALLOW_STALE_DATA"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")


def make_prices(n: int = 900, seed: int = 7, drift: float = 0.0004, vol: float = 0.018, start="2015-01-01") -> pd.DataFrame:
    """Synthetic but realistic OHLCV series (geometric random walk with a regime shift)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    r[n // 2: n // 2 + 120] -= 0.004          # a bear leg in the middle
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol_ = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol_}, index=idx)
    df.index.name = "Date"
    df.attrs["ticker"] = "TEST"
    return df


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def benchmark():
    return make_prices(seed=11, drift=0.0003, vol=0.011)
