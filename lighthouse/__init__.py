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
from .config import InvestorProfile, StrategyParams, VERDICT_BANDS
from .engine import analyze, AnalysisResult
from .screener import screen

__version__ = "1.2.0"
__all__ = ["analyze", "AnalysisResult", "screen", "InvestorProfile", "StrategyParams", "VERDICT_BANDS"]
