"""
The colour system shared by the static (matplotlib) and interactive (Plotly) charts. Kept in its own module so the
web app can draw Plotly charts without importing matplotlib (about a third of a second at start-up).

  strategy = blue, buy & hold = orange, S&P 500 = aqua (fixed categorical order)
  status   = green / amber / red for beacons, never reused for series
"""
SERIES = {"strategy": "#2a78d6", "buy_hold": "#eb6834", "benchmark": "#1baf7a", "price": "#2a78d6",
          "sma_fast": "#eb6834", "sma_slow": "#4a3aa7", "band": "#9ec5f4", "band_mid": "#1c5cab"}
STATUS = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b", "grey": "#9a9892"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
REGIME_FILL = {"Bear / Turbulent": "#f6d5d5", "Bear / Calm": "#fbe9e3", "Bull / Turbulent": "#fff3d6"}
SEQ_BLUE_STOPS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING_STOPS = ["#d03b3b", "#f0efec", "#2a78d6"]
