"""
Lighthouse - the web app.

    streamlit run app.py

Six tabs: analyse one stock, find what to buy (or short) right now with your cash and when to get out
(for a profit or a loss), build and risk-check a portfolio, rank and compare a watchlist, inspect the data
behind a verdict, and learn how the method works. Every chart is interactive (hover for the reason behind every
mark, zoom, range buttons, PNG download) and every number can be downloaded as CSV or Markdown.
"""
from __future__ import annotations

import html
import json
import os
import sys
import threading
import traceback
from dataclasses import asdict
from datetime import date

import pandas as pd
import streamlit as st


# Streamlit Community Cloud pulls each push into the running server without restarting Python, so the lighthouse modules
# imported before it are still in sys.modules and can be older than this app.py ("ImportError: cannot import name ...").
# When the package's files changed on disk, drop the old modules and the results cached with them before importing.
@st.cache_resource
def _reload_lock() -> threading.Lock:
    return threading.Lock()


def _lighthouse_is_stale() -> bool:
    pkg = sys.modules.get("lighthouse")
    return pkg is not None and (not hasattr(pkg, "source_changed") or pkg.source_changed())


def _drop_stale_lighthouse() -> None:
    if not _lighthouse_is_stale():
        return
    with _reload_lock():
        if not _lighthouse_is_stale():   # another session reloaded it while this one waited
            return
        for name in [n for n in list(sys.modules) if n == "lighthouse" or n.startswith("lighthouse.")]:   # list(): one atomic read
            del sys.modules[name]
        import lighthouse  # noqa: F401 - under the lock, so a concurrent session never finds half a package
        st.cache_data.clear()      # last: it also drops the lock, and later sessions must already see fresh code
        st.cache_resource.clear()


_drop_stale_lighthouse()

from lighthouse import analyze, screen, InvestorProfile, StrategyParams  # noqa: E402
from lighthouse import interactive as ix  # noqa: E402
from lighthouse.config import VERDICT_BANDS, RISK_PRESETS, HORIZON_WEIGHTS  # noqa: E402
from lighthouse.data import sample_pack_tickers, offline_mode, load_prices, load_risk_free, DataUnavailable  # noqa: E402
from lighthouse.decision import size_position  # noqa: E402
from lighthouse.regime import regime_adjustments  # noqa: E402
from lighthouse.report import format_metrics, LIGHT_ICON  # noqa: E402
from lighthouse import portfolio as pf  # noqa: E402
from lighthouse.parallel import map_tickers  # noqa: E402
from lighthouse.tradeplan import TradeSettings, build_trade_plan, allocate_ideas, signed_usd, TIME_STOP_DEFAULT  # noqa: E402
from lighthouse.universe import available_universe, get_universe, universe_names  # noqa: E402

st.set_page_config(page_title="Lighthouse · explainable stock decisions", page_icon="🔦", layout="wide",
                   menu_items={"About": "Lighthouse turns a ticker and a few facts about you into an explainable Buy / Hold / Sell "
                                        "decision, a trade ticket for the cash you have, portfolio allocations with a risk report, "
                                        "and honest backtests. Decision support, not investment advice."})

# ----------------------------------------------------------------------------- look & accessibility
st.markdown("""
<style>
/* keyboard users always see where they are */
*:focus-visible { outline: 3px solid #2a78d6 !important; outline-offset: 2px !important; }
/* st.metric values never get clipped with an ellipsis (the "Market weather" fix) */
[data-testid="stMetricValue"] { font-size: 1.55rem !important; white-space: normal !important; overflow: visible !important; line-height: 1.2 !important; }
[data-testid="stMetricLabel"] { white-space: normal !important; }
/* KPI cards: wrap-safe, themed, readable in light and dark mode */
.lh-kpi { padding: .7rem .95rem; border: 1px solid rgba(128,128,128,.28); border-radius: .7rem;
          background: var(--secondary-background-color); min-height: 5.6rem; margin-bottom: .5rem; }
.lh-kpi-label { font-size: .82rem; opacity: .75; margin-bottom: .15rem; }
.lh-kpi-value { font-size: 1.65rem; font-weight: 650; line-height: 1.15; word-break: break-word; white-space: normal; }
.lh-kpi-sub { font-size: .82rem; opacity: .8; margin-top: .2rem; }
.lh-verdict { display: inline-block; padding: .15rem .7rem; border-radius: 999px; color: white; font-weight: 700; font-size: 1.05rem; vertical-align: middle; }
.lh-muted { opacity: .72; }
h2, h3, h4 { scroll-margin-top: 4rem; }
/* universe pickers: one line that scrolls sideways instead of a block of wrapped chips */
[class*="st-key-uscroll_"] [data-testid="stElementContainer"], [class*="st-key-uscroll_"] [data-testid="stButtonGroup"] {
    width: 100% !important; max-width: 100%; min-width: 0; }
[class*="st-key-uscroll_"] [role="radiogroup"] { flex-wrap: nowrap !important; overflow-x: auto; overflow-y: hidden;
    max-width: 100%; padding-bottom: .55rem; -webkit-overflow-scrolling: touch; }
[class*="st-key-uscroll_"] [role="radiogroup"] button { flex: 0 0 auto; white-space: nowrap; }
/* a styled WebKit/Blink scrollbar is always drawn (no overlay auto-hide), so the row visibly scrolls */
[class*="st-key-uscroll_"] [role="radiogroup"]::-webkit-scrollbar { height: 9px; }
[class*="st-key-uscroll_"] [role="radiogroup"]::-webkit-scrollbar-track { background: rgba(128,128,128,.14); border-radius: 5px; }
[class*="st-key-uscroll_"] [role="radiogroup"]::-webkit-scrollbar-thumb { background: rgba(128,128,128,.6); border-radius: 5px; }
@supports (-moz-appearance: none) {   /* Firefox: the standard properties (Blink would ignore the rules above if it saw these) */
  [class*="st-key-uscroll_"] [role="radiogroup"] { scrollbar-width: thin; scrollbar-color: rgba(128,128,128,.6) rgba(128,128,128,.14); } }
</style>
""", unsafe_allow_html=True)

VERDICT_COLOR = ix.VERDICT_COLOR
LIGHT_WORD = ix.LIGHT_WORD
POPULAR = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH", "KO"]
OFFLINE_DEFAULT_LIST = "MSFT AAPL JNJ XOM JPM UNH KO PG"     # a quick first scan of the sample pack


def _is_offline() -> bool:
    return offline_mode() or st.session_state.get("source") == "sample"


# ----------------------------------------------------------------------------- cached work
def _profile(horizon, risk, max_loss, cash, holdings) -> InvestorProfile:
    return InvestorProfile.from_cash(float(cash), float(holdings), horizon=horizon, risk=risk, max_loss_pct=max_loss)


LIVE_TTL = 900   # analyses (and the quotes they carry) are reused for 15 minutes; "Refresh live prices" forces new ones
# Analyses, universes and portfolios are large objects that the app only reads, so they are cached as shared resources:
# st.cache_data would pickle and unpickle them (megabytes) on every cache hit.


@st.cache_resource(show_spinner=False, ttl=LIVE_TTL, max_entries=400)
def _run(ticker, horizon, risk, max_loss, cash, holdings, years, source, entry, exit_, cost, overrides, as_of, nonce=0):
    prof = _profile(horizon, risk, max_loss, cash, holdings)
    params = StrategyParams.for_horizon(horizon, entry_score=entry, exit_score=exit_, cost_bps=cost)
    return analyze(ticker, prof, params, years=years, source=source, fundamentals_overrides=overrides or None, as_of=as_of)


@st.cache_data(show_spinner=False, ttl=LIVE_TTL)
def _screen(tickers, horizon, risk, max_loss, cash, holdings, years, source, nonce=0):
    prof = _profile(horizon, risk, max_loss, cash, holdings)
    return screen(list(tickers), prof, StrategyParams.for_horizon(horizon), years=years, source=source, progress=False)


@st.cache_resource(show_spinner=False, ttl=LIVE_TTL, max_entries=400)
def _light(ticker, horizon, risk, max_loss, cash, holdings, years, source, nonce=0):
    """Analysis without the strategy backtest: enough for trade tickets."""
    return analyze(ticker, _profile(horizon, risk, max_loss, cash, holdings), years=years, source=source, run_backtest_=False)


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=20)
def _universe(tickers, years, source):
    return pf.load_universe(list(tickers), years=years, source=source)


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=4)
def _risk_free(source):
    """The daily 3-month T-bill rate from FRED (None offline or for the sample pack)."""
    return None if _is_offline() else load_risk_free(source)


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=50)
def _basket(tickers, method, max_w, min_w, sector_cap, target_vol, capital, years, source, lookback, rebalance, cost, rf):
    cons = pf.PortfolioConstraints(max_weight=max_w, min_weight=min_w, max_sector_weight=sector_cap, target_vol=target_vol)
    return pf.build_portfolio(tickers, method, cons, capital=capital, lookback=lookback, rebalance=rebalance, cost_bps=cost,
                              risk_free_rate=rf, universe=_universe(tickers, years, source), risk_free=_risk_free(source))


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=50)
def _goal(goal, tickers, n_stocks, per_sector, max_w, sector_cap, target_vol, capital, years, source, lookback, rebalance, cost, rf,
          held=()):
    cons = pf.PortfolioConstraints(max_weight=max_w, max_sector_weight=sector_cap, target_vol=target_vol)
    return pf.goal_portfolio(goal, capital=capital, n_stocks=n_stocks, constraints=cons, max_per_sector=per_sector,
                             lookback=lookback, rebalance=rebalance, cost_bps=cost, risk_free_rate=rf,
                             universe=_universe(tickers, years, source), held=list(held) or None, risk_free=_risk_free(source))


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=200)
def _closes(ticker, years, source) -> pd.Series:
    return load_prices(ticker, years=years, source=source)["Close"].rename(ticker)


# ----------------------------------------------------------------------------- helpers
def esc(text) -> str:
    """Markdown-safe text: a pair of $ signs would otherwise be rendered as LaTeX math."""
    return str(text).replace("$", "\\$")


def _h(text) -> str:
    """HTML-escaped text with $ as an entity, so markdown never sees a math delimiter."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def kpi(col, label: str, value: str, sub: str = "", color: str | None = None, help_: str | None = None) -> None:
    """A wrap-safe metric card (st.metric truncates long values such as 'Bull / Turbulent')."""
    style = f" style='color:{color}'" if color else ""
    title = f" title='{_h(help_)}'" if help_ else ""
    col.markdown(f"<div class='lh-kpi' role='group' aria-label='{_h(label)}'{title}><div class='lh-kpi-label'>{_h(label)}</div>"
                 f"<div class='lh-kpi-value'{style}>{_h(value)}</div><div class='lh-kpi-sub'>{_h(sub)}</div></div>",
                 unsafe_allow_html=True)


def money(v: float, dp: int = 0) -> str:
    return "n/a" if v is None or pd.isna(v) else f"${v:,.{dp}f}"


def beacon_frame(beacons) -> pd.DataFrame:
    rows = [{"Light": f"{LIGHT_ICON[b.light]} {LIGHT_WORD[b.light]}", "Signal": b.name, "Reading": b.value,
             "Vote": None if pd.isna(b.score) else round(float(b.score), 2), "Weight": f"{b.weight * 100:.0f}%",
             "Points": None if pd.isna(b.score) else round(float(b.contribution), 1), "Why": b.rationale} for b in beacons]
    return pd.DataFrame(rows)


BEACON_COLUMNS = {
    "Light": st.column_config.TextColumn("Light", help="Green = bullish, amber = neutral, red = bearish, white = no data", width="small"),
    "Signal": st.column_config.TextColumn("Signal", width="medium"),
    "Reading": st.column_config.TextColumn("Reading", width="medium"),
    "Vote": st.column_config.NumberColumn("Vote", help="−1 (bearish) … +1 (bullish)", format="%+.2f", width="small"),
    "Weight": st.column_config.TextColumn("Weight", help="Share of the group score", width="small"),
    "Points": st.column_config.NumberColumn("Points", help="Points added to the group score (vote × weight × 50)", format="%+.1f", width="small"),
    "Why": st.column_config.TextColumn("Why", width="large"),
}


def resize_plan(res, max_loss: float, cash: float, holdings: float, risk: str):
    """Re-run only the position-sizing rules with a different investor profile (instant, no new analysis)."""
    d, last = res.decision, res.indicators.iloc[-1]
    prof = InvestorProfile.from_cash(cash, holdings, horizon=d.profile.horizon, risk=risk, max_loss_pct=max_loss)
    adj = regime_adjustments(d.regime.is_bear if d.regime else False, d.regime.is_turbulent if d.regime else False, res.params)
    return size_position(d.price, float(last["atr"]), float(last["sizing_vol"]), d.composite_score, d.verdict, prof,
                         adj["size_multiplier"], float(last["sma_slow"]), res.params)


def show_error(exc: Exception, ticker: str) -> None:
    if isinstance(exc, DataUnavailable):
        st.error(f"**Could not load data for {ticker}.** {exc}")
        st.markdown("Things to try:\n"
                    "- Check the ticker symbol (US-listed symbols such as `MSFT`, `BRK-B`).\n"
                    "- Yahoo Finance may be rate-limiting: wait a minute and press Analyse again, or choose the `stooq` data source.\n"
                    "- No internet? Choose the `sample` data source: it includes "
                    f"{', '.join(t for t in sample_pack_tickers() if not t.startswith('^'))}.")
    else:
        st.error(f"**Something went wrong:** {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


def plotly(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=ix.PLOTLY_CONFIG, key=key)


def figure(owner, name, build):
    """Build a chart once per result object and reuse it: every click re-runs the whole script, and rebuilding the
    page's Plotly figures took most of a rerun. The figure lives on the object it was drawn from, so a new analysis
    or portfolio always gets new charts."""
    store = owner.__dict__.setdefault("_lh_figures", {})
    if name not in store:
        store[name] = build()
    return store[name]


def _pick_ticker() -> None:
    v = st.session_state.get("pick")
    if v:
        st.session_state["ticker"] = v


def _typed_ticker() -> None:
    """Runs before the widgets are drawn, so the typed value can still be normalised here."""
    st.session_state["pick"] = None
    st.session_state["ticker"] = st.session_state.get("ticker", "").upper().strip()


def _open_from_watchlist(ticker: str) -> None:
    """Ask the next full run to analyse `ticker`. Applied before the sidebar's ticker box is drawn: the watchlist is a
    fragment, and a widget's value cannot be changed after the widget exists in a run."""
    st.session_state["open_ticker"] = ticker


# ----------------------------------------------------------------------------- sidebar
if st.session_state.get("open_ticker"):
    st.session_state["ticker"] = st.session_state.pop("open_ticker")
    st.session_state["pick"] = None
    st.session_state["auto_run"] = True
st.session_state.setdefault("ticker", "MSFT")
with st.sidebar:
    st.title("🔦 Lighthouse")
    st.caption("A clear signal in a noisy market. Explainable Buy / Hold / Sell decisions for self-directed investors.")

    st.text_input("Ticker", key="ticker", help="Any US-listed symbol, e.g. MSFT, AAPL, BRK-B", on_change=_typed_ticker)
    ticker = st.session_state["ticker"].upper().strip()
    quick = [t for t in sample_pack_tickers() if not t.startswith("^")] if _is_offline() else POPULAR
    st.pills("Quick pick", quick, selection_mode="single", key="pick", on_change=_pick_ticker, label_visibility="collapsed")

    st.subheader("About you", anchor=False)
    horizon = st.radio("Investment horizon", ["long", "short"], key="horizon",
                       format_func=lambda h: "Long-term (6-24+ months)" if h == "long" else "Short-term (weeks to months)",
                       help="Long: fundamentals 60% / technicals 40%, monthly review, exit below the 200-day line. "
                            "Short: 30 / 70, weekly review, ATR trailing stop.")
    risk = st.select_slider("Risk tolerance", ["conservative", "balanced", "aggressive"], value="balanced", key="risk",
                            help="Sets the maximum single position (10 / 20 / 35%), the volatility target (10 / 15 / 25%) and the stop distance.")
    max_loss = st.slider("Max acceptable loss per position (% of your whole account)", 0.5, 10.0, 2.0, 0.5, key="max_loss",
                         help="If a position's stop is hit, you lose at most this share of your account (cash + stocks you hold).")
    cash = st.number_input("Cash available to invest now ($)", min_value=0, value=10_000, step=500, key="cash",
                           help="New positions are paid from this cash only; nothing you already own is sold.")
    holdings = st.number_input("Value of stocks you already hold ($)", min_value=0, value=0, step=1000, key="holdings",
                               help="Only used for the risk limits: the max loss and the concentration cap are a % of your whole "
                                    "account (cash + holdings). Leave 0 if this cash is all you have invested.")
    if cash + holdings <= 0:
        st.error("Enter the cash you can invest (or the value of what you hold) to size positions.")
    with st.expander("Trade management (entries & exits)"):
        long_exit = st.radio("Long-term buys: how to take profit", ["Trend exits (tested best)", "Profit targets (R)"],
                             key="long_exit", help="Trend exits: no fixed target; at each month's last close sell if the close "
                             "is below the 200-day SMA − 8% (a level that rises with the trend) or the technical score has "
                             "faded to the exit score. These are the long-horizon backtest's own exits and, on the sample "
                             "stocks, earned several times more per trade than profit targets 30-80% away. Short-term trades "
                             "and shorts always use the profit targets below.")
        tp1 = st.number_input("Target 1 (× the risk, R)", 0.5, 10.0, 2.0, 0.5, key="tp1",
                              help="Take-profit level 1 = entry ± this many times the distance from entry to the stop.")
        tp2 = st.number_input("Target 2 (R)", 0.5, 20.0, 3.0, 0.5, key="tp2")
        partial = st.slider("Sell at target 1 (% of the position)", 0, 100, 50, 10, key="partial",
                            help="The rest runs to target 2 with the stop moved to breakeven and then trailed.")
        tstop = st.number_input("Time stop (trading sessions, 0 = horizon default)", 0, 1000, 0, 5, key="tstop",
                                help=f"Close a trade that has reached neither stop nor target 1 after this many sessions. "
                                     f"Default: {TIME_STOP_DEFAULT['short']} (short horizon) / {TIME_STOP_DEFAULT['long']} (long "
                                     "with profit targets); none with trend exits.")
        max_ideas = st.number_input("Max trade ideas funded at once", 1, 20, 5, key="max_ideas")
        heat = st.slider("Max total risk across ideas (% of account)", 1.0, 20.0, 6.0, 0.5, key="heat",
                         help="'Portfolio heat': the sum of every idea's planned loss stays below this.")

    st.subheader("Data", anchor=False)
    years = st.slider("Years of history to analyse", 3, 25, 10, key="years")
    source = st.selectbox("Data source", ["auto", "yfinance", "stooq", "cache", "sample"], key="source",
                          help="auto = live data with offline fallbacks. 'sample' uses the bundled historical pack (no internet needed).")
    refresh_live = st.button("🔄 Refresh live prices", key="refresh_live", width="stretch", disabled=_is_offline(),
                             help="Fetch the latest quote (and any session that has finished) now. Otherwise an analysis and its "
                                  "quote are reused for 15 minutes; the quote's time is always shown.")
    if refresh_live:
        st.session_state["live_nonce"] = st.session_state.get("live_nonce", 0) + 1
        if st.session_state.get("result") is not None:
            st.session_state["auto_run"] = True
    deep = st.checkbox("Run deep evaluation (layers, sensitivity, out-of-sample)", value=False, key="deep",
                       help="Adds ~30-60 seconds: strategy layer attribution, threshold sensitivity grid and a walk-forward test.")
    with st.expander("Expert settings"):
        entry = st.number_input("Entry score", 40, 90, 60, key="entry", help="Technical score needed to open a position in the backtest")
        exit_ = st.number_input("Exit score", 10, 60, 35 if horizon == "long" else 40, key="exit", help="Technical score at which an open position is closed")
        cost = st.number_input("Cost per side (bps)", 0, 50, 5, key="cost", help="Commission in basis points; slippage of 5 bps is added on top")
        as_of = st.date_input("Time machine: as-of date", value=None, min_value=date(1990, 1, 1), max_value=date.today(), key="as_of",
                              help="What would Lighthouse have said on this day? Leave empty for today.")
        f_json = st.text_area("Fundamental overrides (JSON)", value="", key="f_json",
                              help='Type your own numbers when a feed is missing a field, e.g. {"pe_trailing": 22, "roe": 0.18}')
    run = st.button("Analyse", type="primary", width="stretch", key="analyse", disabled=cash + holdings <= 0)
    if _is_offline():
        st.info("Offline sample pack: " + ", ".join(t for t in sample_pack_tickers() if not t.startswith("^")))
    st.caption("Decision-support tool, not investment advice.")

run = run or st.session_state.pop("auto_run", False)
NONCE = st.session_state.get("live_nonce", 0)
try:
    SETTINGS = TradeSettings(reward_multiples=(float(min(tp1, tp2)), float(max(tp1, tp2))), partial_exit=partial / 100.0,
                             time_stop_days=int(tstop) or None, max_ideas=int(max_ideas), max_total_risk_pct=float(heat),
                             long_exit="targets" if str(long_exit).startswith("Profit") else "trend")
except ValueError as exc:
    st.sidebar.error(f"Trade management: {exc}")
    SETTINGS = TradeSettings()
ACCOUNT_OK = cash + holdings > 0

# ----------------------------------------------------------------------------- run an analysis
if run:
    overrides = {}
    if f_json.strip():
        try:
            overrides = json.loads(f_json)
        except json.JSONDecodeError as exc:
            st.error(f"Could not parse the fundamental overrides JSON: {exc}")
    with st.spinner(f"Collecting data and analysing {ticker} ..."):
        try:
            res = _run(ticker, horizon, risk, max_loss, cash, holdings, years, source, entry, exit_, cost, overrides,
                       str(as_of) if as_of else None, NONCE)
            st.session_state["result"] = res
            st.session_state.pop("result_error", None)
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("result", None)
            st.session_state["result_error"] = (exc, ticker)
    res = st.session_state.get("result")
    if res is not None and deep and "layers" not in res.extras:
        with st.status("Deep evaluation: layer attribution, sensitivity grid and walk-forward test ...", expanded=False) as status:
            res.evaluate()
            status.update(label="Deep evaluation complete", state="complete")

# ----------------------------------------------------------------------------- tabs
tab_analyse, tab_desk, tab_port, tab_screen, tab_data, tab_about = st.tabs(
    ["📍 Analyse a stock", "🎯 Trade desk", "🧺 Portfolio builder", "📋 Rank & compare a watchlist", "🗂️ Data behind the verdict", "ℹ️ How it works"])


# ============================================================================= ANALYSE
STATUS_COLOR_FOR_REGIME = {"Bull / Calm": VERDICT_COLOR["Strong Buy"], "Bull / Turbulent": VERDICT_COLOR["Hold"],
                           "Bear / Calm": VERDICT_COLOR["Reduce"], "Bear / Turbulent": VERDICT_COLOR["Sell"]}


TICKET_COLOR = {"actionable": VERDICT_COLOR["Strong Buy"], "watch": VERDICT_COLOR["Hold"], "avoid": VERDICT_COLOR["Sell"]}


def ticket_headline(pl) -> str:
    if pl.status == "actionable" and pl.shares > 0:
        verb = "BUY" if pl.side == "long" else "SELL SHORT"
        return f"{verb} {pl.shares} {pl.ticker}"
    if pl.status == "watch":
        return f"WATCH {pl.ticker} ({'long' if pl.side == 'long' else 'short'})"
    return f"NO {'LONG' if pl.side == 'long' else 'SHORT'} TRADE IN {pl.ticker}" if pl.status == "avoid" else f"{pl.ticker}: cash too small"


EXIT_ICON = {"loss": "🛑", "profit": "🎯", "time": "⏱️", "evidence": "🧭"}
TREND_ICON = {"loss": "🛑", "profit": "📈", "time": "⏱️", "evidence": "🧭"}


def render_exits(pl, key: str) -> None:
    """When to get out: every exit, for a loss or a profit, with its trigger, the result for the whole position and
    how often it happened on this stock's history."""
    st.markdown("**When to get out** · place the stop when you enter, so the exit is decided before emotions are")

    rows = []
    icons = TREND_ICON if pl.is_trend_plan else EXIT_ICON
    for x in pl.exit_plan():
        result = ""
        if x.r is not None:
            result = (f"{signed_usd(x.pnl)} ({x.r:+.2f}R)" if x.pnl is not None else f"{x.r:+.2f}R")
        elif pl.is_trend_plan and x.kind != "loss":
            result = "depends on when (see history)"
        rows.append({"": icons.get(x.kind, ""), "Exit": x.name,
                     "Trigger": "" if x.price is None else f"${x.price:,.2f} ({x.move_pct:+.1f}%)",
                     "If the trade ends here": result, "What to do": x.action, "On this stock's history": x.history})
    st.table(pd.DataFrame(rows).set_index(""))
    if pl.is_trend_plan:
        st.caption("A trend plan has no fixed profit target: the review-day exits end the trade, and their result depends on "
                   "how far the trend ran (the history column shows the average). R = multiples of the planned loss. A gap "
                   "through a stop fills at the open, which can be worse.")
    else:
        st.caption("Results are for the whole position (shares already sold included) before costs; R = multiples of the "
                   "planned loss. A gap through a stop fills at the open, which can be worse.")


def render_ticket(res, pl, key: str, chart: bool = True, compact: bool = False) -> None:
    """One trade ticket: levels, money, the management plan, the historical record of these rules, and the caveats."""
    color = TICKET_COLOR.get(pl.status, "#9a9892")
    sub = (f"{pl.name} · score {pl.composite_score:.0f} ({pl.verdict}) · {pl.regime} · signals through {pl.as_of} · "
           f"priced from the {pl.priced_from}")
    st.markdown(f"<h4 style='margin:.2rem 0'><span class='lh-verdict' style='background:{color}'>{_h(ticket_headline(pl))}</span> "
                f"<span class='lh-muted' style='font-weight:400;font-size:.95rem'>{_h(sub)}</span></h4>", unsafe_allow_html=True)
    if pl.side == "short" and pl.status != "avoid":
        st.warning("Short selling: losses are not capped by what you paid, you need a margin account and borrowable shares, "
                   "and you owe any dividends. Read the caveats below.", icon="⚠️")
    t1 = pl.targets[0] if pl.targets else None
    t2 = pl.targets[1] if len(pl.targets) > 1 else None
    c1, c2, c3, c4 = st.columns(4)
    lim = "max price to pay" if pl.side == "long" else "min price to sell at"
    kpi(c1, "Entry (limit)", money(pl.price, 2), f"do not chase: {lim} {money(pl.entry_limit, 2)}")
    kpi(c2, "Stop-loss", money(pl.stop, 2) if pl.stop else "n/a",
        f"{pl.stop_pct * 100:.1f}% {'below' if pl.side == 'long' else 'above'} · {pl.stop_label}" if pl.stop_pct else "")
    if pl.is_trend_plan:
        lvl = pl.trend_exit_level
        kpi(c3, f"Trend exit ({pl.review}-end close)", money(lvl, 2) if lvl is not None else "n/a",
            f"below the {pl.trend_exit_label}; rises with the trend" + (f" ({pl.trend_exit_pace_pct:+.1f}% last month)"
                                                                         if pl.trend_exit_pace_pct is not None else ""))
        kpi(c4, f"Signal exit ({pl.review}-end)", f"score ≤ {pl.signal_exit_score:g}",
            f"technical score today {pl.technical_score:.0f}")
    else:
        kpi(c3, "Take profit 1", money(t1.price, 2) if t1 else "n/a", f"{t1.r_multiple:g}R · {t1.move_pct:+.1f}% · {t1.shares} sh" if t1 else "")
        kpi(c4, "Take profit 2", money(t2.price, 2) if t2 else "n/a", f"{t2.r_multiple:g}R · {t2.move_pct:+.1f}% · {t2.shares} sh" if t2 else "")
    c5, c6, c7, c8 = st.columns(4)
    kpi(c5, "Cash used" if pl.side == "long" else "Collateral (1:1)", money(pl.notional),
        f"{pl.shares} shares · {money(pl.cash_left)} of {money(pl.cash_available)} cash left")
    kpi(c6, "Planned loss at the stop", money(pl.dollars_at_risk), f"{pl.risk_pct_of_account:.2f}% of the account (gaps can be worse)")
    st_ = pl.stats
    if pl.is_trend_plan:
        wins = st_.trades.loc[st_.trades["r"] > 0] if st_ is not None else None
        if wins is not None and len(wins):
            kpi(c7, "Winning trades on this stock's history", f"{wins['return_pct'].median():+.0f}% median",
                f"{len(wins)} winners · median {wins['r'].median():+.1f}R, held {wins['days'].median():.0f} sessions")
        else:
            kpi(c7, "Winning trades on this stock's history", "n/a", "no finished winning setups")
    else:
        kpi(c7, "Profit if both targets fill", money(pl.profit_if_all_targets),
            f"reward : risk {pl.reward_risk:g} : 1 on target 1" if t1 else "")
    if st_ is not None:
        how = (f"ended by a review exit {st_.trend_exited * 100:.0f}%" if st_.exit_style == "trend"
               else f"hit target 1 {st_.hit_target1 * 100:.0f}%")
        kpi(c8, "These rules on this stock's history", f"{st_.avg_r:+.2f}R / trade",
            f"{st_.n_trades} trades in {st_.years:.0f}y · win {st_.win_rate * 100:.0f}% · {how} · "
            f"profit factor {st_.profit_factor:.2f}", STATUS_GOOD if st_.avg_r > 0 else STATUS_BAD)
    else:
        kpi(c8, "These rules on this stock's history", "n/a", "no comparable historical setups")
    if pl.warnings and pl.warnings[0].startswith("On this stock's own history"):
        st.error(esc(pl.warnings[0]), icon="📉")
    if pl.stop is not None:
        render_exits(pl, key)
    st.markdown("**The plan, step by step**\n\n" + "\n".join(f"{i}. {esc(x)}" for i, x in enumerate(pl.management_steps(), 1)))
    if chart and not compact:
        levels = (pl.side, pl.price, pl.stop, tuple(t.price for t in pl.targets), pl.trend_exit_level, pl.as_of)
        plotly(figure(res, ("trade", levels), lambda: ix.trade_chart(res, pl)), key=f"ticket_chart_{key}_{pl.side}_{pl.ticker}")
    with st.expander("Why this status, how it was sized, and the caveats"):
        st.markdown("**Setup**\n\n" + "\n".join(f"- {esc(x)}" for x in pl.reasons))
        if pl.sizing_notes:
            st.markdown("**Sizing**\n\n" + "\n".join(f"- {esc(x)}" for x in pl.sizing_notes))
        st.markdown(f"**Typical move**: one standard deviation over {pl.typical_days} sessions ≈ ±{pl.typical_move_pct:.1f}% "
                    f"(from {res.params.sizing_vol_window}-day realised volatility). ATR(14) today {esc(money(pl.atr, 2))}.")
        if pl.margin:
            st.markdown(f"**Margin (short)**: Regulation T initial margin ≈ {esc(money(pl.margin['reg_t_initial']))} (50% of the short "
                        f"value, on top of the sale proceeds); FINRA Rule 4210 maintenance ≥ {esc(money(pl.margin['finra_maintenance']))}. "
                        "Brokers may demand more for volatile or hard-to-borrow stocks.")
        if pl.warnings:
            st.markdown("**Caveats**\n\n" + "\n".join(f"- {esc(x)}" for x in pl.warnings))
        if st_ is not None:
            rules = ("stop order, then the review-day trend and signal exits at the next open" if pl.is_trend_plan else
                     "stop, partial at target 1, breakeven, trailing stop, target 2, time stop")
            st.markdown("**Historical setup test**: every past day this stock met the same setup rule, the trade was entered the next "
                        f"session and managed with exactly the plan above ({rules}), one trade at a time, after costs. If the stop "
                        "and a target fall in the same session the stop is assumed to come first.")
            tr = st_.trades.copy()
            st.dataframe(tr[["signal_date", "entry_date", "entry", "stop", "exit_date", "outcome", "r", "return_pct", "days"]],
                         hide_index=True, width="stretch", column_config={
                             "signal_date": st.column_config.DateColumn("Signal"), "entry_date": st.column_config.DateColumn("Entry"),
                             "exit_date": st.column_config.DateColumn("Exit"), "entry": st.column_config.NumberColumn("Entry $", format="$%.2f"),
                             "stop": st.column_config.NumberColumn("Stop $", format="$%.2f"), "r": st.column_config.NumberColumn("R", format="%+.2f"),
                             "return_pct": st.column_config.NumberColumn("Return", format="%+.1f%%"), "days": "Sessions"})


STATUS_GOOD, STATUS_BAD = VERDICT_COLOR["Strong Buy"], VERDICT_COLOR["Sell"]


def price_line(res) -> str:
    """Where the order price comes from: the live quote and its time, or the last close."""
    d, q = res.decision, getattr(res, "quote", None)
    if q is not None:
        return f"signals through {d.as_of} · price {q.describe()}"
    return f"{d.as_of} close ${d.price:,.2f}"


def render_result(res) -> None:
    d, f, pl, prof = res.decision, res.fundamentals, res.decision.plan, res.profile
    color = VERDICT_COLOR[d.verdict]

    # ---- headline ---------------------------------------------------------
    h1, h2 = st.columns([3, 1.3])
    with h1:
        st.markdown(f"<h2 style='margin:0 0 .3rem 0'>{f.name or d.ticker} <span class='lh-muted' style='font-weight:400'>({d.ticker})</span> "
                    f"<span class='lh-verdict' style='background:{color}'>{d.verdict}</span></h2>"
                    f"<p class='lh-muted' style='margin:0'>{_h(price_line(res) + ' · ' + prof.describe())}</p>", unsafe_allow_html=True)
        st.info(esc(pl.action), icon="🧭")
        if res.warnings:
            st.warning("**Read before acting**\n\n" + "\n".join(f"- {esc(w)}" for w in res.warnings), icon="⚠️")
    with h2:
        plotly(figure(res, "gauge", lambda: ix.score_gauge(d)), key="gauge")

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Lighthouse score", f"{d.composite_score:.0f} / 100", "verdict bands: ≥72 Strong Buy · ≥60 Buy · ≥45 Hold · ≥32 Reduce", color)
    kpi(k2, "Fundamental score", "n/a" if d.fundamental_score is None else f"{d.fundamental_score:.0f}",
        f"weight {prof.fundamental_weight:.0%} · {f.sector or 'sector n/a'}")
    kpi(k3, "Technical score", f"{d.technical_score:.0f}", f"weight {prof.technical_weight:.0%} · {prof.horizon}-term rule")
    if d.regime:
        kpi(k4, "Market weather", d.regime.label, f"{'−' if d.regime_penalty else ''}{d.regime_penalty:.0f} pts · S&P 500 {d.regime.benchmark_vs_sma * 100:+.1f}% vs 200-day",
            STATUS_COLOR_FOR_REGIME.get(d.regime.label))
    else:
        kpi(k4, "Market weather", "n/a", "benchmark unavailable")

    # ---- score composition + position plan ---------------------------------
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        plotly(figure(res, "waterfall", lambda: ix.score_waterfall(d, prof)), key="waterfall")
        if d.regime:
            st.caption(d.regime.describe())
    with c2:
        st.markdown("#### Position plan: buy now with your cash")
        p1, p2 = st.columns(2)
        kpi(p1, "Buy now", f"{pl.shares} shares" if pl.shares else "no new shares",
            f"≈ {money(pl.cost)} of your {money(prof.cash)} cash · {pl.position_pct:.1f}% of the account · near {money(pl.entry_price, 2)}")
        kpi(p2, "Exit level (stop-loss)", "n/a" if pl.stop_price is None else money(pl.stop_price, 2),
            "" if pl.stop_pct is None else f"{pl.stop_pct * 100:.1f}% below the price")
        p3, p4 = st.columns(2)
        kpi(p3, "Worst-case loss", money(pl.max_loss_value),
            f"your limit: {prof.max_loss_pct:g}% of the account = {money(prof.max_loss_value)} (a gap through the stop can lose more)")
        kpi(p4, "Binding rule", pl.binding_constraint, "the tightest of max-loss, volatility target, concentration cap and your cash")
        with st.expander("How the size was computed"):
            for r_ in pl.rationale:
                st.markdown("• " + esc(r_))
        with st.expander("What if I change my limits? (instant, no new analysis)"):
            whatif_section(res)

    ticket_section(res)
    beacons_section(res)
    charts_section(res)
    backtest_section(res)
    downloads_section(res)


# Each section below is a fragment: a click on one of its widgets re-runs that section only, not the whole page.
@st.fragment
def whatif_section(res) -> None:
    prof = res.profile
    w1, w2, w3, w4 = st.columns(4)
    w_loss = w1.slider("Max loss (% of account)", 0.5, 10.0, float(prof.max_loss_pct), 0.5, key="wi_loss")
    w_cash = w2.number_input("Cash ($)", 0, value=int(prof.cash), step=500, key="wi_cash")
    w_hold = w3.number_input("Holdings ($)", 0, value=int(prof.holdings_value), step=1000, key="wi_hold")
    w_risk = w4.select_slider("Risk", ["conservative", "balanced", "aggressive"], value=prof.risk, key="wi_risk")
    if w_cash + w_hold > 0:
        wp = resize_plan(res, w_loss, w_cash, w_hold, w_risk)
        exit_txt = "n/a" if wp.stop_price is None else esc(money(wp.stop_price, 2))
        st.markdown(f"**{wp.shares} shares** ≈ **{esc(money(wp.cost))}** ({wp.position_pct:.1f}% of the account) · "
                    f"exit {exit_txt} · worst case **{esc(money(wp.max_loss_value))}** · limited by the **{wp.binding_constraint}**")
    st.caption("The verdict and score do not change here; only the sizing rules are re-applied.")


@st.fragment
def ticket_section(res) -> None:
    st.markdown("#### Trade ticket: buy or sell short right now")
    side = st.segmented_control("Side", ["long", "short"], default="long", key="ticket_side", label_visibility="collapsed",
                                format_func=lambda v: "📈 Long (buy)" if v == "long" else "📉 Short (sell short)")
    render_ticket(res, build_trade_plan(res, side or "long", SETTINGS), key="analyse")


@st.fragment
def beacons_section(res) -> None:
    d, f = res.decision, res.fundamentals
    st.markdown("#### Why: the beacons")
    st.caption("Every signal votes between −1 (bearish) and +1 (bullish). Hover a bar for the reading and the reason. Filter by light to focus.")
    flt = st.segmented_control("Show", ["all", "bullish", "neutral", "bearish"], default="all", key="beacon_filter", label_visibility="collapsed")
    want = {"all": None, "bullish": "green", "neutral": "amber", "bearish": "red"}[flt or "all"]
    shown = [b for b in d.beacons if want is None or b.light == want]
    if shown:
        plotly(figure(res, ("breakdown", tuple(b.key for b in shown)), lambda: ix.score_breakdown(d, beacons=shown)), key="breakdown")
    else:
        st.info("No beacon has that light right now.")
    colf, colt = st.columns(2)
    with colf:
        st.markdown(f"**Fundamental beacons** · {f.sector or ''} · source: {f.source} ({f.as_of})")
        st.dataframe(beacon_frame([b for b in d.fundamental_beacons if b in shown]), hide_index=True, width="stretch", column_config=BEACON_COLUMNS)
    with colt:
        st.markdown("**Technical beacons**")
        st.dataframe(beacon_frame([b for b in d.technical_beacons if b in shown]), hide_index=True, width="stretch", column_config=BEACON_COLUMNS)


@st.fragment
def charts_section(res) -> None:
    st.markdown("#### Price, trend and momentum")
    look = st.segmented_control("Lookback", ["1y", "3y", "5y", "all"], default="3y", key="lookback", label_visibility="collapsed")
    last_days = {"1y": 252, "3y": 756, "5y": 1260, "all": len(res.indicators)}[look or "3y"]
    st.caption("Shaded areas show market weather (red = bear, amber = turbulent). Triangles are the backtested rule's entries and "
               "exits. Windows longer than 3 years are drawn from weekly points. Drag to zoom, double-click to reset.")
    plotly(figure(res, ("dashboard", last_days), lambda: ix.dashboard(res, last_days=last_days)), key="dashboard")
    with st.expander("How the price bands are built"):
        n, k = res.params.band_window, res.params.band_std
        st.markdown("The dashed centre line is a **trend filter** (Holt's double exponential smoothing). Each session it "
                    "projects yesterday's level along yesterday's slope and then moves part of the way to today's close, so "
                    "it follows a steady trend without falling behind. On the sample stocks it is about 1 session behind the "
                    f"price, against about 3 for a 20-day Hull moving average and about 9 for the {n}-day simple average that "
                    f"classic Bollinger Bands use. The bands sit ± {k:g} standard deviations of the closes around that line "
                    f"over the last {n} sessions. A close above the upper band means price is stretched relative to where it "
                    "is now; the *Price bands* beacon scores that position (%B). Like every line in Lighthouse they use only "
                    "past prices: nothing is shifted forward, so nothing peeks at the future.")


def backtest_section(res) -> None:
    bt = res.backtest
    if bt is not None:
        st.markdown(f"#### How this rule behaved {bt.start.date()} → {bt.end.date()}")
        st.caption(f"Long-only, reviewed {res.params.decision_frequency}, filled at the {bt.options['fills']}, "
                   f"{res.params.cost_bps + res.params.slippage_bps:.0f} bps per side. Returns are for the capital allocated to this stock. "
                   f"Sharpe and Sortino use {bt.options.get('risk_free', 'a fixed risk-free rate')}. "
                   "Past behaviour of a rule is evidence about its design, not a promise.")
        m, mb = bt.metrics, bt.summary().loc["Buy & hold"]
        ratio = lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"            # noqa: E731
        pct = lambda v, dp=0: "n/a" if pd.isna(v) else f"{v * 100:.{dp}f}%"  # noqa: E731
        b1, b2, b3, b4 = st.columns(4)
        kpi(b1, "Strategy CAGR", pct(m["cagr"], 1), f"buy & hold {pct(mb['cagr'], 1)}")
        kpi(b2, "Max drawdown", pct(m["max_drawdown"]), f"buy & hold {pct(mb['max_drawdown'])}")
        kpi(b3, "Sharpe", ratio(m["sharpe"]), f"buy & hold {ratio(mb['sharpe'])}")
        kpi(b4, "Trades", f"{int(m['n_trades'])}", f"win rate {pct(m['win_rate'])} · invested {pct(m['exposure'])} of the time")
        st.dataframe(format_metrics(bt.summary()), width="stretch")
        plotly(figure(res, "backtest", lambda: ix.backtest(bt, res.regimes)), key="backtest")

        if "layers" not in res.extras:
            if st.button("Run deep evaluation (layers, sensitivity, out-of-sample) · ~30-60 s", key="deep_now"):
                with st.status("Deep evaluation running ...", expanded=False) as status:
                    res.evaluate()
                    status.update(label="Deep evaluation complete", state="complete")
                st.rerun()
        else:
            st.markdown("#### Deep evaluation")
            e1, e2 = st.tabs(["What each layer contributes", "Robustness"])
            with e1:
                st.caption("Each row adds one layer of the product on top of the previous one, so you can see what every rule is for.")
                st.dataframe(format_metrics(res.extras["layers"]), width="stretch")
                plotly(figure(res, "layers", lambda: ix.layers(res.extras["layers"])), key="layers")
                rb = res.extras.get("regime_breakdown")
                if rb is not None and len(rb):
                    st.markdown("**Behaviour by market regime** (annualised daily returns)")
                    st.dataframe(format_metrics(rb), width="stretch")
            with e2:
                sens = res.extras.get("sensitivity")
                if sens is not None and len(sens):
                    sensitivity_section(res, sens)
                wf = res.extras.get("walk_forward")
                if wf is not None:
                    st.markdown(f"**Out-of-sample test** (thresholds tuned before {wf['split_date'].date()}, tested after)")
                    st.dataframe(format_metrics(wf["out_of_sample"]), width="stretch")
                    plotly(figure(res, "walk_forward", lambda: ix.walk_forward(wf)), key="walk_forward")
        with st.expander(f"Trade list ({len(bt.trades)} trades)"):
            tf = bt.trades_frame()
            st.dataframe(tf, width="stretch", hide_index=True, column_config={
                "entry_date": st.column_config.DateColumn("Entry"), "exit_date": st.column_config.DateColumn("Exit"),
                "entry_price": st.column_config.NumberColumn("Entry $", format="$%.2f"), "exit_price": st.column_config.NumberColumn("Exit $", format="$%.2f"),
                "size_fraction": st.column_config.NumberColumn("Size", format="percent"), "return_pct": st.column_config.NumberColumn("Return", format="percent"),
                "pnl_pct": st.column_config.NumberColumn("Sleeve P&L", format="percent"), "holding_days": st.column_config.NumberColumn("Days held"),
                "exit_reason": st.column_config.TextColumn("Exit reason")})



@st.fragment
def sensitivity_section(res, sens) -> None:
    metric = st.selectbox("Sensitivity metric", ["sharpe", "cagr", "max_drawdown", "win_rate"], key="sens_metric")
    plotly(figure(res, ("sensitivity", metric), lambda: ix.sensitivity(sens, metric)), key="sensitivity")
    st.caption("A robust rule shows a plateau of similar values; a single bright cell means the thresholds were fitted to noise.")


def downloads_section(res) -> None:
    d, bt = res.decision, res.backtest
    st.markdown("#### Take it with you")
    dl1, dl2, dl3, dl4 = st.columns(4)
    dl1.download_button("Full report (Markdown)", res.report(), file_name=f"{d.ticker}_lighthouse_report.md", width="stretch")
    dl2.download_button("Beacons (CSV)", beacon_frame(d.beacons).to_csv(index=False), file_name=f"{d.ticker}_beacons.csv", width="stretch")
    if bt is not None:
        dl3.download_button("Trades (CSV)", bt.trades_frame().to_csv(index=False), file_name=f"{d.ticker}_trades.csv", width="stretch")
    dl4.download_button("Prices & indicators (CSV)", res.indicators.to_csv(), file_name=f"{d.ticker}_indicators.csv", width="stretch")
    st.caption("Every chart also has a camera icon (top-right on hover) that saves it as a PNG.")


with tab_analyse:
    err = st.session_state.get("result_error")
    res = st.session_state.get("result")
    if err is not None:
        show_error(*err)
    if res is not None:
        if st.session_state.get("as_of"):
            st.info(f"Time machine: this is what Lighthouse would have said on {res.decision.as_of}, using only data available then.", icon="🕰️")
        render_result(res)
    elif err is None:
        st.markdown("### Tell Lighthouse who you are, pick a ticker, press **Analyse**.")
        st.markdown("You get a verdict, the *why* behind it (every signal is a beacon you can inspect), a position plan sized to your "
                    "maximum acceptable loss, and an honest backtest of the rule that produced the recommendation.")
        g1, g2, g3 = st.columns(3)
        g1.markdown("**1 · Who you are**\n\nHorizon, risk tolerance, the most you accept losing on one position, and your portfolio size (sidebar).")
        g2.markdown("**2 · Press Analyse**\n\nLighthouse collects prices and fundamentals, reads the market weather and scores 19 beacons.")
        g3.markdown("**3 · Explore**\n\nHover every chart, filter the beacons, test what-if limits, rank a watchlist, download everything.")
        st.caption("No internet? Choose the `sample` data source in the sidebar: it ships with real historical data for 20 stocks.")


# ============================================================================= TRADE DESK
def _parse_tickers(text: str) -> tuple:
    return tuple(dict.fromkeys(t.upper() for t in str(text).replace(",", " ").split() if t))


def universe_picker(key: str, text_key: str, default: str, label: str = "Universe", text_label: str = "Tickers") -> str:
    """Pre-built universes as one row of chips that scrolls sideways, filling an editable ticker box below.
    Returns the ticker box's text (what will actually be used)."""
    src = "sample" if _is_offline() else source
    names = universe_names(src)
    st.session_state.setdefault(text_key, default)

    def apply() -> None:
        name = st.session_state.get(key)
        if name:
            st.session_state[text_key] = " ".join(get_universe(name, src))
    with st.container(key=f"uscroll_{key}"):
        st.pills(label, names, selection_mode="single", key=key, on_change=apply,
                 help="Scroll sideways for more. Picking one fills the box below; edit it freely.")
    text = st.text_input(text_label + " (space or comma separated)", key=text_key)
    n = len(_parse_tickers(text))
    etfs = [t for t in _parse_tickers(text) if t in set(get_universe("Sector ETFs") + get_universe("Index & bond ETFs"))]
    st.caption(f"{n} tickers" + (" · ETFs have no fundamentals, so they are scored on technicals only" if etfs else "")
               + ("" if _is_offline() or n <= 25 else " · live data for this many takes a few minutes the first time "
                  "(then 15 minutes from memory, prices cached on disk)"))
    return text


def _load_many(tickers, label: str):
    """Analyse several tickers (each cached) with a progress bar, in parallel threads when the data is live (downloads
    overlap). Returns (results in the tickers' order, failures)."""
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    ctx = get_script_run_ctx()
    bar = st.progress(0.0, text=label)
    src = "sample" if _is_offline() else source
    results, errors = map_tickers(
        lambda t: _light(t, horizon, risk, max_loss, cash, holdings, min(years, 6), source, NONCE), tickers, source=src,
        on_done=lambda t, k, n: bar.progress(k / max(n, 1), text=f"{label} {t} ({k}/{n})"),
        initializer=lambda: add_script_run_ctx(threading.current_thread(), ctx))
    bar.empty()
    return list(results.values()), list(errors.items())


def render_side(side: str, desk: dict) -> None:
    results = desk["results"]
    plans = desk.setdefault("plans", {})            # funding the ideas replays every setup's history: do it once
    pkey = (side, tuple(sorted(asdict(SETTINGS).items())))
    if pkey not in plans:
        plans[pkey] = allocate_ideas(results, side, SETTINGS, include_watch=True, with_stats=True)
    funded, watch, table = plans[pkey]
    by = {r.ticker: r for r in results}
    prof = results[0].profile
    used = sum(p.notional for p in funded)
    risk_total = sum(p.dollars_at_risk for p in funded)
    if side == "short":
        st.error("**Shorting is for experienced investors.** A short loses when the price rises, and the loss is not capped by what "
                 "you put in; a gap or squeeze can blow through the stop. It needs a margin account (not a cash account or an IRA), "
                 "shares your broker can borrow (with a fee for hard-to-borrow names), and you pay any dividends. The long-only "
                 "backtests elsewhere in Lighthouse do not cover shorts; each ticket shows how these short rules did on that stock's "
                 "own history.", icon="⚠️")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Ideas funded", f"{len(funded)}", f"of at most {SETTINGS.max_ideas} · {len(watch)} on the watch list")
    kpi(k2, "Cash used" if side == "long" else "Collateral used", money(used), f"of {money(prof.cash)} · {money(prof.cash - used)} left")
    kpi(k3, "Total planned risk", money(risk_total), f"{risk_total / prof.account_value * 100:.2f}% of the account · cap {SETTINGS.max_total_risk_pct:g}%")
    kpi(k4, "Account", money(prof.account_value), f"{money(prof.cash)} cash + {money(prof.holdings_value)} holdings")
    if not funded:
        st.info(f"No {side} setup is actionable right now in this universe" + (" (or the cash buys no whole share)." if side == "long" else "."))
    for pl in funded:
        with st.container(border=True):
            render_ticket(by[pl.ticker], pl, key=f"desk_{side}")
    if len(table):
        st.markdown("**All tickets and the watch list**")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "price": st.column_config.NumberColumn("Price", format="$%.2f"), "cost": st.column_config.NumberColumn("Cost", format="$%.0f"),
            "stop": st.column_config.NumberColumn("Stop", format="$%.2f"), "stop_pct": st.column_config.NumberColumn("Stop %", format="%.1f%%"),
            "risk_usd": st.column_config.NumberColumn("Risk $", format="$%.0f"), "risk_pct": st.column_config.NumberColumn("Risk % acct", format="%.2f%%"),
            "target1": st.column_config.NumberColumn("Target 1", format="$%.2f"), "target2": st.column_config.NumberColumn("Target 2", format="$%.2f"),
            "profit_t1": st.column_config.NumberColumn("Profit @T1", format="$%.0f"), "profit_all": st.column_config.NumberColumn("Profit @T1+T2", format="$%.0f"),
            "exit": "Exits", "trend_exit": st.column_config.NumberColumn("Trend exit", format="$%.2f"),
            "hist_win_rate": st.column_config.NumberColumn("Hist. win %", format="%.0f%%"), "hist_avg_r": st.column_config.NumberColumn("Hist. R/trade", format="%+.2f")})
        st.download_button(f"Download {side} orders (CSV)", table.to_csv(index=False), file_name=f"lighthouse_{side}_orders.csv",
                           key=f"dl_orders_{side}")
    if watch:
        with st.expander(f"Watch list ({len(watch)}): close to a setup, not actionable yet"):
            for pl in watch:
                st.markdown(f"**{pl.ticker}** · score {pl.composite_score:.0f} · " + esc(" ".join(pl.reasons)))


@st.fragment
def desk_tab() -> None:
    """The trade desk; its scans and tabs re-run on their own."""
    st.markdown("### Trade desk: what to buy (or sell short) right now with your cash")
    st.caption(f"Sized for {esc(money(cash))} of cash in a {esc(money(cash + holdings))} account, {horizon}-term, {risk} risk, max loss "
               f"{max_loss:g}% of the account per position. Each idea gets an entry, a stop-loss, its exits (long-term buys: "
               "trend and signal exits reviewed each month; short-term trades and shorts: two take-profit levels, a trailing "
               "and a time stop), and the record of those exact rules on the stock's own history. Change the exits and the "
               "risk caps under *Trade management* in the sidebar.")
    d_tab_long, d_tab_short = st.tabs(["📈 Long (buy)", "📉 Short (sell short)"])
    with d_tab_long:
        desk_text = universe_picker("desk_preset", "desk_text", OFFLINE_DEFAULT_LIST if _is_offline() else " ".join(get_universe("Mega caps")),
                                    text_label="Tickers to scan")
        if st.button("Find trades now", type="primary", key="desk_go", disabled=not ACCOUNT_OK):
            tick = _parse_tickers(desk_text)
            results, failures = _load_many(tick, "Analysing")
            st.session_state["desk"] = {"tickers": tick, "failures": failures, "results": results}
        desk = st.session_state.get("desk")
        if desk and desk["failures"]:
            st.warning("Could not analyse: " + "; ".join(f"**{t}** ({esc(e)})" for t, e in desk["failures"]), icon="⚠️")
        if desk and desk["results"]:
            stale = [r.ticker for r in desk["results"] if r.profile.cash != float(cash) or r.profile.risk != risk or
                     r.profile.horizon != horizon or r.profile.max_loss_pct != max_loss or r.profile.holdings_value != float(holdings)]
            if stale:
                st.info("Your profile changed since the scan: press **Find trades now** to re-size the tickets.")
            render_side("long", desk)
        elif not desk:
            st.info("Pick a universe and press **Find trades now**. Lighthouse scores every ticker, keeps the actionable setups, and "
                    "funds the strongest ones from your cash until the cash, the idea count or the total-risk cap runs out.")
    with d_tab_short:
        desk = st.session_state.get("desk")
        if desk and desk["results"]:
            render_side("short", desk)
        else:
            st.info("Run **Find trades now** on the Long tab first; the same scan feeds both sides.")


with tab_desk:
    desk_tab()

# ============================================================================= PORTFOLIO BUILDER
def render_portfolio(res, key: str) -> None:
    r = res.risk
    tab = res.table()
    st.markdown(f"#### {esc(res.label)} · {esc(money(res.capital))} · prices as of {res.as_of}")
    for n in res.notes:
        st.caption("• " + esc(n))
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Expected volatility", f"{r['expected_vol'] * 100:.1f}% a year",
        f"shrunk covariance (Ledoit-Wolf {r['cov_shrinkage']:.2f}) · cash buffer {res.cash_weight * 100:.0f}%")
    kpi(k2, "1-day 95% VaR", money(r.get("var95_1d_usd", float('nan'))),
        f"{r['var95_1d'] * 100:.2f}% · expected shortfall {money(r.get('es95_1d_usd', float('nan')))}")
    kpi(k3, "1-month 95% VaR", money(r.get("var95_21d_usd", float('nan'))),
        f"{r['var95_21d'] * 100:.1f}% · expected shortfall {money(r.get('es95_21d_usd', float('nan')))}")
    kpi(k4, "Beta to the S&P 500", "n/a" if "beta" not in r else f"{r['beta']:.2f}",
        "" if "beta" not in r else f"correlation {r['correlation_to_benchmark']:.2f} · tracking error {r['tracking_error'] * 100:.1f}%")
    k5, k6, k7, k8 = st.columns(4)
    kpi(k5, "Diversification ratio", f"{r['diversification_ratio']:.2f}", "weighted stock vol ÷ portfolio vol (higher = more diversified)")
    kpi(k6, "Effective number of stocks", f"{r['effective_n']:.1f}", f"{int((tab['target_weight'] > 0).sum())} holdings · 1 / Σw²")
    kpi(k7, "Back-cast max drawdown", f"{r['max_drawdown'] * 100:.1f}%", f"today's weights over {r['days']} sessions")
    bt = res.backtest
    if bt is not None:
        m = bt.metrics
        kpi(k8, "Walk-forward (honest) result", f"{m['cagr'] * 100:.1f}% CAGR",
            f"Sharpe {m['sharpe']:.2f} · max drawdown {m['max_drawdown'] * 100:.0f}% · {m['years']:.1f} years · turnover {m['turnover_annual']:.1f}×/yr")
    st.caption("VaR/expected shortfall: historical simulation of today's weights over the last year (1-month figures use overlapping "
               "21-session windows). The back-cast is not a return forecast: for a goal portfolio the stocks were chosen *because* of "
               "recent strength, so judge performance by the walk-forward test, which re-selects and re-weights using only past data.")
    st.markdown("**Orders for your cash** (whole shares; the rest stays in cash)")
    show = tab[["sector", "target_weight", "actual_weight", "price", "shares", "value", "risk_share", "volatility"]].copy()
    st.dataframe(show, width="stretch", column_config={
        "target_weight": st.column_config.NumberColumn("Target weight", format="percent"),
        "actual_weight": st.column_config.NumberColumn("Actual weight", format="percent"),
        "price": st.column_config.NumberColumn("Price", format="$%.2f"), "shares": st.column_config.NumberColumn("Buy (shares)", format="%d"),
        "value": st.column_config.NumberColumn("Cost", format="$%.0f"), "risk_share": st.column_config.NumberColumn("Share of risk", format="percent"),
        "volatility": st.column_config.NumberColumn("Volatility", format="percent")})
    st.caption(f"Invested {esc(money(float(show['value'].sum())))} · cash left {esc(money(res.cash_left))}.")
    st.download_button("Download orders (CSV)", show.to_csv(), file_name=f"lighthouse_portfolio_{res.method}.csv", key=f"dl_pf_{key}")
    c1, c2 = st.columns(2)
    with c1:
        plotly(figure(res, "alloc", lambda: ix.allocation_chart(tab)), key=f"pf_alloc_{key}")
    with c2:
        plotly(figure(res, "corr", lambda: ix.correlation_heatmap(r["correlation"])), key=f"pf_corr_{key}")
    if res.comparison is not None:
        st.markdown("**Walk-forward test** (weights re-estimated each rebalance from past data only, traded at the next close, costs included)")
        st.dataframe(format_metrics(res.comparison), width="stretch")
        plotly(figure(res, "equity", lambda: ix.equity_curves(res.comparison_runs)), key=f"pf_eq_{key}")
    if len(r["stress"]):
        st.markdown("**Stress test**: today's weights through past market crashes covered by the data")
        st.dataframe(r["stress"], hide_index=True, width="stretch", column_config={
            "portfolio": st.column_config.NumberColumn("Portfolio", format="percent"),
            "S&P 500": st.column_config.NumberColumn("S&P 500", format="percent")})
    if res.selection is not None:
        with st.expander("Why these stocks"):
            st.dataframe(res.selection, width="stretch")


@st.fragment
def portfolio_tab() -> None:
    """The portfolio builder; its inputs and results re-run on their own."""
    st.markdown("### Portfolio builder")
    st.caption("Give Lighthouse a basket and it works out the percentages, or pick a goal and it chooses the stocks too. Every portfolio "
               "comes with a risk report and a walk-forward test, and is turned into whole-share orders for your cash.")
    mode = st.radio("What do you want?", ["My basket: you choose the stocks", "Build for a goal: Lighthouse chooses"], horizontal=True,
                    key="pf_mode")
    pc1, pc2, pc3 = st.columns(3)
    # no key: the default follows the sidebar's cash whenever that changes
    capital = pc1.number_input("Money to allocate ($)", min_value=0, value=int(cash), step=500,
                               help="Defaults to the cash in the sidebar.")
    with st.expander("Estimation & test settings"):
        e1, e2, e3, e4 = st.columns(4)
        lookback = e1.select_slider("Estimation window (sessions)", [126, 252, 504, 756], value=252, key="pf_lookback")
        rebalance = e2.selectbox("Rebalance", ["monthly", "quarterly"], key="pf_rebal")
        pf_cost = e3.number_input("Trading cost (bps per trade)", 0, 100, 10, key="pf_cost")
        tbill = _risk_free(source)
        rf_now = round(float(tbill.iloc[-1]) * 100, 2) if tbill is not None and len(tbill) else 2.0
        rf = e4.number_input("Risk-free rate (%/yr)", 0.0, 10.0, rf_now, 0.25, key="pf_rf",
                             help=(f"Today's 3-month T-bill (FRED DTB3, {tbill.index[-1].date()}): {rf_now:.2f}%. "
                                   if tbill is not None and len(tbill) else "No T-bill data offline: 2%. ") +
                                  "Used by the maximum-Sharpe method and the risk report. The walk-forward Sharpe ratios use "
                                  "the T-bill rate of each day when it is available.") / 100
    if mode.startswith("My basket"):
        default_basket = "AAPL JNJ XOM JPM KO PG UNH WMT" if _is_offline() else "AAPL MSFT JNJ JPM XOM PG KO UNH"
        basket = pc2.text_input("Your stocks", value=default_basket, key="pf_basket")
        method = pc3.selectbox("Allocation method", list(pf.METHODS), format_func=pf.METHODS.get, index=2, key="pf_method")
        b1, b2, b3, b4 = st.columns(4)
        max_w = b1.slider("Max per stock", 0.05, 1.0, 0.25, 0.05, key="pf_maxw")
        min_w = b2.slider("Min per stock", 0.0, 0.2, 0.0, 0.01, key="pf_minw")
        use_sec = b3.checkbox("Cap each sector", value=False, key="pf_usesec")
        sec_cap = b3.slider("Max per sector", 0.1, 1.0, 0.4, 0.05, key="pf_seccap", disabled=not use_sec)
        use_tv = b4.checkbox("Target a volatility (hold cash above it)", value=False, key="pf_usetv")
        tv = b4.slider("Target volatility (%/yr)", 4, 40, 12, 1, key="pf_tv", disabled=not use_tv)
        st.caption(esc({
            "equal_weight": "Equal weight: the same dollars in every stock. Hard to beat out-of-sample; ignores risk differences.",
            "inverse_vol": "Inverse volatility: calmer stocks get more money. Ignores correlations.",
            "risk_parity": "Risk parity: every stock contributes the same share of the portfolio's risk. A robust default.",
            "min_variance": "Minimum variance: the lowest-volatility mix; often concentrates in defensive stocks.",
            "max_sharpe": "Maximum Sharpe: the best expected return per unit of risk (Bayes-Stein shrunk returns). The most sensitive "
                          "to estimation error; compare it with the others in the walk-forward table.",
            "hrp": "Hierarchical risk parity: groups stocks that move together and splits risk between the groups."}[method]))
        if st.button("Build my portfolio", type="primary", key="pf_go"):
            tick = tuple(dict.fromkeys(t.upper() for t in basket.replace(",", " ").split() if t))
            with st.spinner("Estimating risk, optimising and running the walk-forward test for every method ..."):
                try:
                    st.session_state["pf_res"] = _basket(tick, method, max_w, min_w, sec_cap if use_sec else None,
                                                         tv / 100 if use_tv else None, float(capital), years, source, lookback,
                                                         rebalance, float(pf_cost), rf)
                    st.session_state.pop("pf_err", None)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.pop("pf_res", None)
                    st.session_state["pf_err"] = f"{type(exc).__name__}: {exc}"
        if st.session_state.get("pf_err"):
            st.error(esc(st.session_state["pf_err"]))
        if st.session_state.get("pf_res") is not None:
            render_portfolio(st.session_state["pf_res"], "basket")
    else:
        goal = pc2.selectbox("Goal", list(pf.GOALS), format_func=lambda g: pf.GOALS[g].label, index=1, key="pf_goal")
        g = pf.GOALS[goal]
        pc3.caption(esc(g.description) + f" Method: {pf.METHODS[g.method]}.")
        utext = universe_picker("pf_upre", "pf_utext", " ".join(available_universe("sample" if _is_offline() else source)),
                                label="Universe to choose from", text_label="Universe tickers")
        g1, g2, g3, g4, g5 = st.columns(5)
        n_st = g1.number_input("Number of stocks", 2, 40, g.n_stocks, key=f"pf_n_{goal}")
        per_sec = g2.number_input("Max stocks per sector", 1, 10, 3, key="pf_persec")
        gmax = g3.slider("Max per stock", 0.05, 1.0, float(g.max_weight), 0.05, key=f"pf_gmax_{goal}")
        gsec = g4.slider("Max per sector", 0.1, 1.0, float(g.max_sector_weight or 1.0), 0.05, key=f"pf_gsec_{goal}")
        gtv = g5.slider("Target volatility (%/yr, 0 = none)", 0, 40, int(round((g.target_vol or 0) * 100)), 1, key=f"pf_gtv_{goal}")
        held_text = st.text_input(
            "Stocks you already hold for this goal (optional)", "", key="pf_held",
            disabled=g.keep_rank is None,
            help="Rebalancing? List what you bought last time: each one is kept while it still qualifies for the goal, and "
                 "only the free places are re-filled. That is the rule the walk-forward test follows; re-picking from "
                 "scratch every month traded several times more." if g.keep_rank is not None else
                 "This goal re-picks its stocks from scratch at every review.")
        if st.button("Build the goal portfolio", type="primary", key="pf_goal_go"):
            tick = _parse_tickers(utext)
            held = tuple(_parse_tickers(held_text)) if g.keep_rank is not None else ()
            with st.spinner("Ranking the universe, optimising, and re-running the selection month by month for the walk-forward test ..."):
                try:
                    st.session_state["pf_goal_res"] = _goal(goal, tick, int(n_st), int(per_sec), gmax, gsec if gsec < 1.0 else None,
                                                            gtv / 100 if gtv else None, float(capital), years, source, lookback,
                                                            rebalance, float(pf_cost), rf, held)
                    st.session_state.pop("pf_goal_err", None)
                except Exception as exc:  # noqa: BLE001
                    st.session_state.pop("pf_goal_res", None)
                    st.session_state["pf_goal_err"] = f"{type(exc).__name__}: {exc}"
        if st.session_state.get("pf_goal_err"):
            st.error(esc(st.session_state["pf_goal_err"]))
        if st.session_state.get("pf_goal_res") is not None:
            render_portfolio(st.session_state["pf_goal_res"], "goal")


with tab_port:
    portfolio_tab()

# ============================================================================= WATCHLIST
@st.fragment
def watchlist_tab() -> None:
    """The watchlist; ranking, row selection and the comparison chart re-run on their own."""
    st.markdown("### Rank a watchlist")
    st.caption("Every ticker is scored with your profile; the table is sortable, and selecting a row opens a summary card.")
    wl = universe_picker("wl_preset", "wl_text", OFFLINE_DEFAULT_LIST if _is_offline() else " ".join(get_universe("Mega caps")))
    if st.button("Rank watchlist", type="primary", key="rank", disabled=not ACCOUNT_OK):
        tickers = _parse_tickers(wl)
        with st.spinner(f"Analysing {len(tickers)} tickers ..."):
            try:
                st.session_state["screen_df"] = _screen(tickers, horizon, risk, max_loss, cash, holdings, min(years, 6), source, NONCE)
            except Exception as exc:  # noqa: BLE001
                show_error(exc, ", ".join(tickers))
    sdf = st.session_state.get("screen_df")
    if sdf is not None:
        ok = sdf["score"].notna().sum()
        bad = sdf[sdf["error"].astype(str) != ""]
        if len(bad):
            st.warning("Could not score: " + "; ".join(f"**{r.ticker}** ({r.error})" for r in bad.itertuples()), icon="⚠️")
        if ok:
            cached = st.session_state.get("screen_fig")
            if not cached or cached[0] is not sdf:
                cached = st.session_state["screen_fig"] = (sdf, ix.screener(sdf))
            plotly(cached[1], key="screener")
        cols_ = [c for c in ["ticker", "name", "verdict", "score", "fundamental", "technical", "regime", "position_pct", "shares", "cost",
                             "price", "stop", "pe", "roe", "rev_growth", "sector"] if c in sdf.columns]
        show = sdf[cols_].copy()
        for c in ("score", "fundamental", "technical", "position_pct", "shares", "cost", "price", "stop", "pe", "roe", "rev_growth"):
            if c not in show:
                continue
            show[c] = pd.to_numeric(show[c], errors="coerce")   # None -> blank cell instead of the word "None"
        event = st.dataframe(show, width="stretch", on_select="rerun", selection_mode="single-row", key="wl_table", column_config={
            "ticker": "Ticker", "name": "Company", "verdict": "Verdict", "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "fundamental": st.column_config.NumberColumn("Fund.", format="%.0f"), "technical": st.column_config.NumberColumn("Tech.", format="%.0f"),
            "regime": "Weather", "position_pct": st.column_config.NumberColumn("Position %", format="%.1f %%", help="% of your whole account"),
            "shares": st.column_config.NumberColumn("Buy now (shares)", format="%d", help="Sized with the full cash available (one idea at a time)"),
            "cost": st.column_config.NumberColumn("Cost", format="$%.0f"),
            "price": st.column_config.NumberColumn("Price", format="$%.2f"), "stop": st.column_config.NumberColumn("Exit level", format="$%.2f"),
            "pe": st.column_config.NumberColumn("P/E", format="%.1f"), "roe": st.column_config.NumberColumn("ROE", format="percent"),
            "rev_growth": st.column_config.NumberColumn("Rev. growth", format="percent"), "sector": "Sector"})
        rows = event.selection.rows if event and event.selection else []
        if rows:
            r = sdf.iloc[rows[0]]
            if pd.notna(r["score"]):
                vc = VERDICT_COLOR.get(r["verdict"], "#9a9892")
                st.markdown(f"<h4 style='margin:.4rem 0'>{r['name']} ({r['ticker']}) <span class='lh-verdict' style='background:{vc}'>{r['verdict']}</span> "
                            f"<span class='lh-muted' style='font-weight:400;font-size:1rem'>score {r['score']:.0f} · fundamental {r['fundamental'] if pd.notna(r['fundamental']) else 'n/a'} · "
                            f"technical {r['technical']:.0f} · weather {r['regime']} · suggested position {r['position_pct']:.1f}%</span></h4>", unsafe_allow_html=True)
                if st.button(f"Open the full analysis of {r['ticker']} (in the Analyse tab)", key="open_full"):
                    _open_from_watchlist(r["ticker"])
                    st.rerun(scope="app")      # the Analyse tab lives outside this fragment
            else:
                st.info(f"{r['ticker']} could not be scored: {r['error']}")
        st.download_button("Download ranking (CSV)", sdf.to_csv(), file_name="lighthouse_watchlist.csv")

        st.markdown("#### Compare price performance")
        scored = list(sdf.loc[sdf["score"].notna(), "ticker"])
        chosen = st.multiselect("Tickers to compare", scored + (["^GSPC"] if "^GSPC" not in scored else []), default=scored[:5], key="cmp")
        if chosen:
            series = {}
            for t in chosen:
                try:
                    series["S&P 500" if t == "^GSPC" else t] = _closes(t, min(years, 6), source)
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"{t}: {exc}")
            if series:
                plotly(ix.compare_prices(series), key="compare")


with tab_screen:
    watchlist_tab()

# ============================================================================= DATA
@st.fragment
def data_tab() -> None:
    """The data behind the verdict; the column picker re-runs only this tab."""
    res = st.session_state.get("result")
    if res is None:
        st.info("Analyse a stock first; this tab then shows every number behind its verdict.")
    else:
        d, f = res.decision, res.fundamentals
        st.markdown(f"### Data behind {d.ticker}")
        s1, s2, s3 = st.columns(3)
        kpi(s1, "Prices", f"{len(res.prices):,} trading days", f"{res.prices.index[0].date()} → {res.prices.index[-1].date()} · source: {res.prices.attrs.get('source', '?')}")
        kpi(s2, "Fundamentals", f.source, f"as of {f.as_of} · {f.sector or 'sector n/a'}")
        kpi(s3, "Benchmark", "S&P 500" if res.benchmark is not None else "unavailable", f"{len(res.benchmark):,} days" if res.benchmark is not None else "regime rules disabled")
        rep = res.prices.attrs.get("cleaning")
        if rep is not None and getattr(rep, "notes", None):
            st.markdown("**Cleaning report**")
            for n in rep.notes:
                st.write("• " + n)
        t1, t2, t3 = st.tabs(["Fundamental snapshot", "Prices & indicators", "Market weather history"])
        with t1:
            snap = pd.DataFrame({"field": list(asdict(f).keys()), "value": [str(v) for v in asdict(f).values()]})
            st.dataframe(snap[~snap["field"].isin(["notes"])], hide_index=True, width="stretch")
            if res.statements is not None and getattr(res.statements, "annual", None) is not None:
                st.markdown("**Annual statements used for the point-in-time scores**")
                st.dataframe(res.statements.annual, width="stretch")
        with t2:
            cols = st.multiselect("Columns", list(res.indicators.columns), default=["Open", "High", "Low", "Close", "Volume", "sma_fast", "sma_slow", "rsi", "macd", "atr"], key="ind_cols")
            table = res.indicators[cols].tail(500).sort_index(ascending=False)
            table.attrs = {}   # the cleaning report in attrs is not JSON-serialisable
            st.dataframe(table, width="stretch")
            st.download_button("Download all rows (CSV)", res.indicators.to_csv(), file_name=f"{d.ticker}_indicators.csv", key="dl_ind2")
        with t3:
            if res.regimes is not None:
                rg = res.regimes.dropna(subset=["regime"])
                share = rg["regime"].value_counts(normalize=True).rename("share of days").to_frame()
                share["share of days"] = share["share of days"].map(lambda v: f"{v * 100:.0f}%")
                st.dataframe(share, width="stretch")
                st.dataframe(rg[["close", "sma", "vol", "vol_median", "regime"]].tail(250).sort_index(ascending=False), width="stretch")
            else:
                st.info("No benchmark data, so no market-weather history.")


with tab_data:
    data_tab()

# ============================================================================= ABOUT
with tab_about:
    st.markdown("### How Lighthouse decides")
    a1, a2 = st.columns([1.2, 1])
    with a1:
        st.markdown("""
1. **Collect & clean** daily prices, the S&P 500 and fundamentals (live sources first, then cache, then the bundled sample pack).
2. **Fundamental beacons** (13): valuation, quality and growth ratios scored against sector norms.
3. **Technical beacons** (6): trend, RSI, MACD, position in the low-lag price bands (%B), volume confirmation, 52-week-high proximity.
4. **Market weather**: Bull/Bear × Calm/Turbulent for the S&P 500 removes points, raises the entry bar and halves sizes in bear markets.
5. **Decision**: horizon-weighted blend → verdict → position size from the *smallest* of three rules (max-loss, volatility target, concentration cap), paid from the **cash you have**.
6. **Honest evaluation**: point-in-time backtest vs buy & hold and the S&P 500, layer attribution, sensitivity grid, walk-forward test.
7. **Trade desk**: long and short tickets with an entry limit, stop-loss, two take-profit levels (multiples of the risk, *R*), breakeven and trailing stops, a time stop, and a *when to get out* table: every exit, for a profit or a loss, with its price, its result for the whole position and how often it happened on the stock's own history.
8. **Portfolio builder**: Ledoit-Wolf risk model; equal weight, inverse volatility, risk parity, minimum variance, maximum Sharpe (Bayes-Stein returns) or hierarchical risk parity; stock and sector caps; a volatility target; VaR, expected shortfall, beta, risk contributions and stress tests; every method walk-forward tested.
""")
    with a2:
        bands = pd.DataFrame([(f"≥ {t}", v) for t, v in VERDICT_BANDS], columns=["Lighthouse score", "Verdict"])
        st.markdown("**Verdict bands**")
        st.dataframe(bands, hide_index=True, width="stretch")
        pres = pd.DataFrame(RISK_PRESETS).T.rename(columns={"max_position_pct": "max position %", "target_vol": "target volatility", "stop_atr_multiple": "stop (× ATR)"})
        st.markdown("**Risk presets**")
        st.dataframe(pres, width="stretch")
        hw = pd.DataFrame(HORIZON_WEIGHTS, index=["fundamental weight", "technical weight"]).T
        st.markdown("**Horizon weights**")
        st.dataframe(hw, width="stretch")
    guide = os.path.join(os.path.dirname(__file__), "docs", "USER_GUIDE.md")
    with st.expander("Full user guide", expanded=False):
        st.markdown(esc(open(guide).read()) if os.path.exists(guide) else "See docs/USER_GUIDE.md")
    st.caption("Lighthouse is decision support, not investment advice. Sector norms are assumptions, the backtests ignore taxes, "
               "stops are orders rather than guaranteed prices, and the evaluation universes are survivorship-biased.")
