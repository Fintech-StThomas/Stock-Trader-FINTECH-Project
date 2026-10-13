"""
Ticker universes and sector labels for the portfolio builder and the trade desk.

`LARGE_CAP_UNIVERSE` is a curated list of liquid US large caps spread across all 11 sectors (Yahoo Finance's sector
names, which `fundamentals.SECTOR_NORMS` also uses). It is a convenience starting point, not an index: membership is
a point-in-time choice and therefore survivorship-biased when used for backtests. Edit it freely.

`UNIVERSES` holds the pre-built lists the app and the CLI offer (sectors, themes, ETFs). A ticker that is not in
`SECTORS` gets the sector its fundamentals snapshot reports (see `sectors_for`); one with no sector at all (ETFs,
unknown symbols, no connection) is labelled `UNKNOWN_SECTOR` and is never counted against a per-sector limit.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .data import load_sectors, offline_mode, sample_pack_tickers

UNKNOWN_SECTOR = "Unknown"

# The large-cap universe (labels in Yahoo's naming).
_LARGE_CAPS: Dict[str, str] = {
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

# Extra labels for the themed universes below (not part of the large-cap list).
_EXTRA: Dict[str, str] = {
    "INTC": "Technology", "MU": "Technology", "AMAT": "Technology", "LRCX": "Technology", "KLAC": "Technology",
    "ADI": "Technology", "MRVL": "Technology",
    "C": "Financial Services", "SCHW": "Financial Services", "PNC": "Financial Services", "USB": "Financial Services",
    "BMY": "Healthcare", "GILD": "Healthcare", "MDT": "Healthcare", "VRTX": "Healthcare", "CVS": "Healthcare",
    "OXY": "Energy", "PSX": "Energy", "MPC": "Energy", "WMB": "Energy", "KMI": "Energy",
    "AEP": "Utilities", "D": "Utilities", "SRE": "Utilities", "EXC": "Utilities", "XEL": "Utilities",
    "O": "Real Estate", "SPG": "Real Estate", "PSA": "Real Estate", "CCI": "Real Estate", "WELL": "Real Estate",
    "NEM": "Basic Materials", "ECL": "Basic Materials", "NUE": "Basic Materials",
}

SECTORS: Dict[str, str] = {**_LARGE_CAPS, **_EXTRA}

# RRC (a mid cap) and BBY are labelled because they are in the offline sample pack; they are not in the large-cap list.
LARGE_CAP_UNIVERSE: List[str] = [t for t in _LARGE_CAPS if t not in {"RRC", "BBY"}]


def _in_sector(*sectors: str) -> List[str]:
    return [t for t, s in SECTORS.items() if s in sectors and t not in {"RRC", "BBY"}]


# Pre-built universes: name -> tickers. ETFs have no sector, so per-sector limits never apply to them.
UNIVERSES: Dict[str, List[str]] = {
    "Mega caps": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "LLY", "JPM", "V"],
    "Dividend & defensive": ["JNJ", "KO", "PG", "PEP", "WMT", "UNH", "MCD", "CL", "MDLZ", "PM", "DUK", "SO"],
    "Technology": _in_sector("Technology"),
    "Semiconductors": ["NVDA", "AVGO", "AMD", "QCOM", "TXN", "INTC", "MU", "AMAT", "LRCX", "KLAC", "ADI", "MRVL"],
    "Communication services": _in_sector("Communication Services"),
    "Consumer": _in_sector("Consumer Cyclical", "Consumer Defensive"),
    "Healthcare": _in_sector("Healthcare"),
    "Financials": _in_sector("Financial Services"),
    "Industrials": _in_sector("Industrials"),
    "Energy & materials": _in_sector("Energy", "Basic Materials"),
    "Utilities & real estate": _in_sector("Utilities", "Real Estate"),
    "Sector ETFs": ["XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLB", "XLC"],
    "Index & bond ETFs": ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD"],
    "Large caps (all sectors)": list(LARGE_CAP_UNIVERSE),
}
SAMPLE_UNIVERSE_NAME = "Sample pack (offline)"


def sector_of(ticker: str, fallback: Optional[str] = None) -> str:
    """The curated label, else `fallback`, else UNKNOWN_SECTOR. No network: see `sectors_for` for the full lookup."""
    return SECTORS.get(ticker.upper(), fallback or UNKNOWN_SECTOR)


def sectors_for(tickers: Iterable[str], source: str = "auto") -> Dict[str, str]:
    """Sector per ticker: the curated label first, then the sector in the ticker's fundamentals snapshot (cached, or
    one Yahoo `info` request per ticker when live), else UNKNOWN_SECTOR."""
    tickers = [t.upper().strip() for t in tickers if t and t.strip()]
    missing = [t for t in tickers if t not in SECTORS]
    looked_up = load_sectors(missing, source=source) if missing else {}
    return {t: SECTORS.get(t) or looked_up.get(t) or UNKNOWN_SECTOR for t in tickers}


def universe_names(source: str = "auto") -> List[str]:
    """The pre-built universes on offer: the sample pack only when offline (live universes cannot load)."""
    if source == "sample" or offline_mode():
        return [SAMPLE_UNIVERSE_NAME]
    return [*UNIVERSES, SAMPLE_UNIVERSE_NAME]


def get_universe(name: str, source: str = "auto") -> List[str]:
    """Tickers of a pre-built universe (case-insensitive name). The sample pack is always available."""
    if name.strip().lower() == SAMPLE_UNIVERSE_NAME.lower():
        return [t for t in sample_pack_tickers() if not t.startswith("^")]
    for k, v in UNIVERSES.items():
        if k.lower() == name.strip().lower():
            return list(v)
    raise ValueError(f"unknown universe {name!r}; choose one of: {', '.join([*UNIVERSES, SAMPLE_UNIVERSE_NAME])}")


def available_universe(source: str = "auto") -> List[str]:
    """'All available tickers': the bundled sample pack when offline, the curated large-cap list otherwise."""
    if source == "sample" or offline_mode():
        return [t for t in sample_pack_tickers() if not t.startswith("^")]
    return list(LARGE_CAP_UNIVERSE)
