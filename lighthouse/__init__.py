"""
Lighthouse - an explainable stock decision engine for self-directed investors.

Public API
----------
>>> from lighthouse import analyze, InvestorProfile
>>> result = analyze("MSFT", InvestorProfile(horizon="long", risk="balanced", max_loss_pct=2))
>>> print(result.decision.verdict, result.decision.composite_score)
>>> print(result.report())           # plain-English explanation
>>> result.backtest.summary()        # honest, point-in-time evaluation

>>> from lighthouse.tradeplan import build_trade_plan            # buy / short-now ticket for your cash
>>> from lighthouse import portfolio                              # basket allocation, goal portfolios, risk
"""
import os as _os

_DIR = _os.path.dirname(_os.path.abspath(__file__))


def _source_stamp() -> tuple:
    return tuple(sorted((e.name, e.stat().st_mtime_ns, e.stat().st_size) for e in _os.scandir(_DIR) if e.name.endswith(".py")))


_IMPORTED_STAMP = _source_stamp()   # the files this process loaded


def source_changed() -> bool:
    """True when the package's files on disk are no longer the ones this process imported: new code landed under a
    running server (Streamlit Community Cloud pulls every push without restarting Python)."""
    return _source_stamp() != _IMPORTED_STAMP


from .config import InvestorProfile, StrategyParams, VERDICT_BANDS  # noqa: E402
from .engine import analyze, AnalysisResult  # noqa: E402
from .screener import screen  # noqa: E402

__version__ = "1.2.0"
__all__ = ["analyze", "AnalysisResult", "screen", "InvestorProfile", "StrategyParams", "VERDICT_BANDS"]
