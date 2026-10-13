"""
Ticker universes and sector labels for the portfolio builder and the trade desk.

`LARGE_CAP_UNIVERSE` is a curated list of liquid US large caps spread across all 11 sectors (Yahoo Finance's sector
names, which `fundamentals.SECTOR_NORMS` also uses). It is a convenience starting point, not an index: membership is
a point-in-time choice and therefore survivorship-biased when used for backtests. Edit it freely.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .data import offline_mode, sample_pack_tickers

SECTORS: Dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AVGO": "Technology", "ORCL": "Technology",
    "CRM": "Technology", "ADBE": "Technology", "AMD": "Technology", "CSCO": "Technology", "ACN": "Technology",
    "INTU": "Technology", "TXN": "Technology", "QCOM": "Technology", "IBM": "Technology", "NOW": "Technology",
    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "TMUS": "Communication Services", "VZ": "Communication Services",
    "T": "Communication Services", "CMCSA": "Communication Services",
    # Consumer Cyclical
    "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", "HD": "Consumer Cyclical", "MCD": "Consumer Cyclical",
    "NKE": "Consumer Cyclical", "LOW": "Consumer Cyclical", "SBUX": "Consumer Cyclical", "BKNG": "Consumer Cyclical",
    "TJX": "Consumer Cyclical", "BBY": "Consumer Cyclical",
    # Consumer Defensive
    "WMT": "Consumer Defensive", "PG": "Consumer Defensive", "KO": "Consumer Defensive", "PEP": "Consumer Defensive",
    "COST": "Consumer Defensive", "PM": "Consumer Defensive", "MDLZ": "Consumer Defensive", "CL": "Consumer Defensive",
    # Healthcare
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare", "MRK": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "PFE": "Healthcare", "AMGN": "Healthcare", "ISRG": "Healthcare",
    # Financial Services
    "JPM": "Financial Services", "V": "Financial Services", "MA": "Financial Services", "BAC": "Financial Services",
    "WFC": "Financial Services", "GS": "Financial Services", "MS": "Financial Services", "BRK-B": "Financial Services",
    "AXP": "Financial Services", "SPGI": "Financial Services", "BLK": "Financial Services",
    # Industrials
    "GE": "Industrials", "CAT": "Industrials", "RTX": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "BA": "Industrials", "DE": "Industrials", "LMT": "Industrials", "UPS": "Industrials", "ETN": "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy", "SLB": "Energy", "RRC": "Energy",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate",
    # Basic Materials
    "LIN": "Basic Materials", "SHW": "Basic Materials", "APD": "Basic Materials", "FCX": "Basic Materials",
}

# RRC (a mid cap) and BBY are labelled because they are in the offline sample pack; they are not in the large-cap list.
LARGE_CAP_UNIVERSE: List[str] = [t for t in SECTORS if t not in {"RRC", "BBY"}]


def sector_of(ticker: str, fallback: Optional[str] = None) -> str:
    return SECTORS.get(ticker.upper(), fallback or "Unknown")


def available_universe(source: str = "auto") -> List[str]:
    """'All available tickers': the bundled sample pack when offline, the curated large-cap list otherwise."""
    if source == "sample" or offline_mode():
        return [t for t in sample_pack_tickers() if not t.startswith("^")]
    return list(LARGE_CAP_UNIVERSE)
