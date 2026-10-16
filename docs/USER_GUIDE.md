# Lighthouse user guide

Lighthouse answers four questions for **you** specifically: *should I buy this stock now, how much can I buy with the
cash I have, where would I be wrong (and where do I take profit), and what does the business behind the stock look like
in its own filings?* It also splits money across several stocks.

Two parts work side by side. **The rule** (the verdict, the position plan and the trade ticket) reads prices only: the
trend, momentum, price bands, volume and the market's weather. It is the part the backtests test. **The analyst memo**
reads the company's SEC filings as they were filed: valuation, peers, earnings quality, the balance sheet and what
would prove the thesis wrong. It sits next to the verdict and never changes it, because none of the fundamental
signals Lighthouse tested on the point-in-time S&P 500 survived its hold-out (`docs/METHODOLOGY.md` §8b).

## Three ways to run it

| | Best for | How |
|---|---|---|
| **Google Colab / Jupyter notebook** | the course deliverable; running end-to-end with narration | open `notebooks/Lighthouse.ipynb`, run all cells; change the inputs cell |
| **Web app** | everyone: interactive charts, trade tickets, the memo, portfolios, downloads | `pip install -r requirements.txt` then `streamlit run app.py` (or the hosted Streamlit Cloud link in the README) |
| **Command line** | power users, batch runs, saving reports | see *Command line* below |

Live analyst memos need a contact e-mail for SEC EDGAR, which asks every program for one: set the environment variable
`LIGHTHOUSE_SEC_EMAIL=you@example.com` before starting (it is sent to sec.gov only). Without it the verdict and the
ticket still work and the memo says why it is missing; the bundled sample pack has the filings of its 20 stocks.

## The inputs

| Input | Meaning | Effect |
|---|---|---|
| Ticker | any US-listed symbol | |
| Horizon | **long** (6-24+ months) or **short** (weeks to a few months) | long: monthly review, exits below the 200-day line (the trend exits); short: weekly review, profit targets and an ATR trailing stop |
| Risk tolerance | conservative / balanced / aggressive | max single position 10 / 20 / 35% of the account; volatility target 10 / 15 / 25%; stop distance 3 / 2.5 / 2 ATR (short horizon) |
| Max acceptable loss | % of your **whole account** you accept losing on one position if its stop is hit | position size = max loss ÷ distance to the stop, if that is the tightest rule |
| **Cash available to invest now** | the money you can put to work today | every new position is paid from this cash only; nothing you own is sold |
| **Value of stocks you already hold** | optional; 0 if this cash is all you have invested | only used for the risk limits: max loss and the concentration cap are a % of cash + holdings |
| Years of history | default 10 (notebook 15) | analysis window and backtest length |
| Data source | auto / yfinance / stooq / cache / sample | `auto` tries live sources then offline fallbacks; `sample` uses the bundled historical pack |
| Analyst memo: compare with industry peers | off by default | reads the filings of every S&P 500 company with the same SIC code for the peer medians (slower); off, the memo compares with sector norms |
| 🔄 Refresh live prices | sidebar button | fetches the latest quote now; otherwise an analysis and its quote are reused for 15 minutes |

Example: $5,000 cash and $20,000 of stocks already held, 2% max loss. The account is $25,000, so one position may lose at most
$500 at its stop, but it can never cost more than the $5,000 of cash.

## Analyse a stock

The headline shows the verdict and four cards: the **Lighthouse score** (0-100: ≥72 Strong Buy · ≥60 Buy · ≥45 Hold ·
≥32 Reduce · <32 Sell), the **technical score** it comes from, the **market weather** (Bull/Bear × Calm/Turbulent for
the S&P 500, and the points it removed) and the **analyst memo's stance** (buy / hold / avoid, with its weighted value
against the price; not part of the score). The price line under the name says where the order price comes from: the
signals use *finished* sessions only, and "buy now" orders are priced from the latest quote and its time, or the last
close when there is none (offline, or the time machine). Warnings say what to know before acting.

Five sub-tabs:

* **Plan & ticket**: how the score is built (technical score − market weather), the **position plan** (how many shares
  to buy with your cash, the exit level, the worst-case loss and *which rule limited the size*: max loss, volatility
  target, concentration cap or your cash), a *what-if* sizer, and the **trade ticket** (below). Hold means keep what you
  own; Reduce means trim about half; Sell means exit.
* **Analyst memo**: the fundamental case from the SEC filings, with only filings made before the analysis date: a bull
  / base / bear DCF, the growth the price implies (reverse DCF), the stock's own valuation bands, the fundamental
  beacons (valuation, quality and growth against industry peers or sector norms), earnings-quality and balance-sheet
  flags, capital allocation, industry metrics for banks, REITs, insurers and software companies, business segments,
  the catalyst calendar, the variant view, the **criteria that would prove the thesis wrong** and a pre-mortem.
  *Save the thesis to follow it* puts those criteria in the journal; the Journal tab checks them against each new 10-Q
  or 10-K. An ETF, a fund or a company that does not file with the SEC has no memo, and the tab says so.
* **Technicals**: the six technical beacons with their reading, light (🟢 bullish, 🟡 neutral, 🔴 bearish, ⚪
  unavailable), vote (−1…+1), weight and points; and the price chart: close, 50/200-day averages and **price bands**
  that follow the price (a trend filter ± 2 standard deviations of the closes around it over 20 sessions, about 1
  session behind the price instead of the ~10 of classic Bollinger Bands; past prices only), volume, RSI and MACD, with
  the backtested entries and exits.
* **Backtest**: how the rule behaved on this stock against buy-and-hold, the S&P 500 and **the same average exposure
  held without trading** (the stock and cash in the proportion the rule held on average: what the rule earns above it
  is what its timing added). *Run deep evaluation* adds what each layer contributes, the plain 200-day trend rule as a
  second yardstick, the behaviour by market regime, a threshold sensitivity grid and a walk-forward test.
* **Data**: the price source and cleaning notes, the risk-free rate used, every indicator row and the market-weather
  history; downloads of the report, the beacons, the trades and the indicators.

## Trade tickets

A ticket is a complete order plan for buying with your cash (long only):

1. **Entry**: a limit order near the last price, and a do-not-chase price (skip the trade if it has already run 0.5 ATR past the entry).
2. **Stop-loss**: placed immediately, at the horizon's stop. The planned loss is shown in dollars and as % of your account.
3. **How the profit is taken** depends on the trade:
   * **Long-term buys: trend exits, no fixed target.** At the last close of each month, sell everything at the next open if
     the close is below the 200-day SMA − 8% (a level that rises with the trend, so it locks in more of the gain the
     longer the trend runs) or the technical score has faded to 35 or lower. These are the long-horizon backtest's own
     exits. The stop order stays where it was placed and caps the loss between reviews.
   * **Short-term trades: profit targets.** Take profit 1 (2R by default, R = the distance to the stop): sell half and
     move the stop on the rest to breakeven. Then a **trailing stop** follows the best close since entry minus k × ATR
     and never loosens. Take profit 2 (3R by default) sells the rest. A **time stop** closes a trade that has reached
     neither the stop nor target 1 after 40 sessions (252 for a long-term trade on profit targets).

Why no target on long-term buys: the long-term stop sits 8% below the 200-day SMA, often 20-30% under the price, so a 2R
target needed a 40-80% rise. Replayed on every past setup of the 20 sample stocks, the trend exits earned +3.6R per trade
against +0.6R for the targets (1990-2022; +2.5R vs +0.6R over the last 15 years). Profit targets stay available under
*Trade management* ("Long-term buys: how to take profit").

Each ticket shows **how these exact rules did on this stock's own history**: number of trades, win rate, average result
in R per trade after costs, and profit factor. If the rules lost money on that stock, the ticket says so in red. Change the
targets, the partial exit, the time stop, the number of ideas and the total-risk cap under *Trade management* in the sidebar.

**When to get out** (on every ticket): one row per exit, for a loss or a profit, decided before you enter. A long-term buy
lists the stop order (−1R), the 📈 trend exit (its level today; the result depends on how far the trend ran, so the row
shows the average result of past trend exits) and the 🧭 signal exit (the technical score today against 35). A trade on
profit targets lists:

| Exit | Trigger | Result shown |
|---|---|---|
| 🛑 Stop-loss | the stop price | the planned loss (−1R) |
| 🎯 Target 1 | its price | the partial profit, which is also the result if the rest is later stopped at breakeven |
| 🎯 Trailing stop | follows the best close | between target 1's and target 2's result |
| 🎯 Target 2 | its price | the best planned outcome |
| ⏱️ Time stop | after 40 / 252 sessions | between the stop's loss and target 1's profit |
| 🧭 Verdict turns | at the weekly / monthly review | sell early |

Each row also says how often trades ended that way on the stock's own history and how many sessions it took.

**Status**: *actionable* (place the ticket), *watch* (close to a setup; levels shown, no order) or *avoid* (no trade).

## Find trades

* **Universe**: a row of pre-built lists that scrolls sideways (mega caps, dividend & defensive, each sector,
  semiconductors, sector ETFs, index & bond ETFs, all large caps, the offline sample pack). Picking one fills the ticker
  box, which you can edit.
* Press *Find trades now*. Lighthouse scores every ticker and **ranks** them (a chart and a sortable table; select a row
  for a summary card and a one-click jump to its full analysis and memo), keeps the actionable setups and **funds the
  strongest first** from your cash. It stops when the cash, the idea count or the total planned risk ("portfolio heat",
  6% of the account by default) runs out. You get one ticket per idea, a summary table, a watch list, a CSV of the
  orders and a rebased price comparison.
* There is no position-entry screen: every ticket already carries its exits (*When to get out*), so the sell decision is
  made, and the stop order placed, when you buy. To replay a trade you already hold against the plan, use the command
  line's `monitor` (below).

## Portfolio

* **My basket**: type your stocks, choose an allocation method and the caps (max per stock, optional max per sector,
  optional target volatility), and press *Build my portfolio*.
  * *Equal weight*: the same dollars in each. Hard to beat; the honest baseline.
  * *Risk parity* (default): each stock contributes the same share of risk.
  * *Minimum variance*: the lowest-volatility mix.
* **Build for a goal**: choose *Capital preservation*, *Balanced risk and reward*, *Growth* or *Maximum diversification*.
  Lighthouse picks the stocks from the universe (any pre-built list, or your own) and weights them, at most 3 per sector.
  A ticker outside the curated list gets the sector Yahoo reports for it; one with no sector at all (an ETF, say) is
  not counted against the per-sector limits. When you rebalance, list what you already hold under *Stocks you already
  hold for this goal*: for the balanced, growth and diversified goals each one is kept while it still qualifies, and only
  the free places are re-filled (the walk-forward test follows the same rule; re-picking from scratch traded up to 8×
  more).

Every portfolio shows:
* whole-share **orders for your cash** (CSV download);
* **expected volatility**, **Value at Risk** and **expected shortfall** in dollars (1 day and 1 month, 95%), beta to the
  S&P 500, the diversification ratio and the effective number of stocks;
* where the money is versus where the risk is, and the correlation heatmap;
* a **walk-forward test** of every method (or of the goal, reviewed every month, next to the same goal re-picked from
  scratch so you can see what keeping holdings saves) against the S&P 500 and equal weight;
* a stress test through past crashes covered by the data.

Judge performance by the walk-forward test. The back-cast of today's weights is shown for risk only: a goal portfolio picks
stocks *because* of their recent strength, so their recent past flatters them.

## Journal & theses

* **Theses you follow**: every thesis saved from a memo, its return since, the distance to the memo's value, and
  whether it is stale (a filing has arrived since the memo, or it is more than 100 days old). *Check every open thesis
  now* tests the price against each thesis every time, and the other criteria when a new 10-Q or 10-K has been filed.
  Closing a thesis writes its post-mortem.
* **Decision journal**: every verdict and ticket the scheduled run records (`python -m lighthouse journal run`, GitHub
  Actions after the close), never edited afterwards, and what happened next: the forward test of the rules, which
  cannot contain look-ahead.
* **Watch-only signals**: the research near misses recorded at each month end (below).

## Command line

```bash
python -m lighthouse analyze MSFT --horizon long --risk balanced --max-loss 2 --cash 5000 --holdings 20000 --out reports/MSFT --evaluate
python -m lighthouse screen AAPL MSFT NVDA JPM --risk conservative --cash 10000 --csv ranking.csv
python -m lighthouse backtest MSFT --years 15 --evaluate
python -m lighthouse trade MSFT AAPL JNJ XOM --cash 5000 --holdings 20000 --targets 2 3
python -m lighthouse trade --universe semiconductors --cash 5000                  # a pre-built universe
python -m lighthouse universes                                                    # list them
python -m lighthouse monitor MSFT --entry 73.9 --shares 30 --date 2017-09-01 --source sample
python -m lighthouse memo MSFT --out reports/MSFT                                 # the analyst memo alone
python -m lighthouse thesis check                                                 # saved theses against new filings
python -m lighthouse portfolio AAPL JNJ XOM JPM KO PG --method risk_parity --cash 10000 --target-vol 0.12
python -m lighthouse portfolio --goal balanced --cash 10000
python -m lighthouse portfolio --goal growth --universe technology --cash 10000
```

`analyze` prints the verdict, the plan and the analyst memo (`--no-memo` leaves the memo out, `--peers` adds the peer
comparison; with `--out` it is saved as `TICKER_memo.md` next to the report, the tables and the interactive charts).
`trade` prints each ticket's plan and its *when to get out* list (`--long-exit targets` puts long-term buys on profit
targets). `monitor` replays a position you hold since its entry with the same rules (a long-term buy: the stop order and
the month-end trend and signal exits; otherwise target 1 sells half, the stop then moves to breakeven and trails) and
reports the exits already taken, the realised and unrealised P&L, where the stop is now, and whether the last review says
sell at the next open. During market hours it includes today's unfinished session, so a position bought today can be
replayed and a stop hit today shows up.

Add `--as-of 2020-03-20` to `analyze` to ask what Lighthouse would have said on a past day. `--portfolio` still works and
means "this much cash, nothing else held".

### Forward watch list (watch-only research signals)

```bash
LIGHTHOUSE_SEC_EMAIL=you@example.com python -m lighthouse journal watch   # month ends only: record the snapshot
python -m lighthouse journal watch --due      # prints true when the last finished session is the month's last
python -m lighthouse journal score --out reports/journal                  # verdicts, tickets, then the watch list
python -m lighthouse journal export --out reports/journal                 # every journal table as CSV
```

Three research signals that came close but did not survive the pre-registered test (FCF yield, short-sale volume,
risk-factor similarity; `docs/METHODOLOGY.md` §8b-8c) are recorded for every S&P 500 member after the close of each
month's last NYSE session, so they can be judged on data that did not exist when they were tested. They are
**watch-only**: they never change a verdict, a ticket or a position size. `journal watch` takes no date: on any other
day it prints that nothing is due and writes nothing, and a snapshot written after the next session's open is kept but
not scored. It needs live data (Yahoo prices, SEC EDGAR with your contact e-mail, FINRA) and takes up to an hour with
an empty cache. `journal score` adds a watch-only section once the next month's returns are recorded (rank IC with its
Newey-West t, the top fifth against the average stock before and after costs, turnover, the forward rule's status and
how many months a convincing result would take); the app's Journal tab shows the same. The scheduled workflow
`.github/workflows/watch.yml` does this every month on GitHub, appending to the same journal as the daily run.

## Changing the rules without programming

All settings are in `lighthouse/config.py`: risk presets, beacon weights, verdict bands and the `StrategyParams`
dataclass (indicator windows, thresholds, review frequency, costs). The sector norms the memo's fundamental beacons use
without peers are in `lighthouse/fundamentals.py` (`SECTOR_NORMS`). Trade-management defaults are in `TradeSettings`
(`lighthouse/tradeplan.py`), goal presets in `GOALS` (`lighthouse/portfolio.py`) and the large-cap universe with sector
labels in `lighthouse/universe.py`. The CLI accepts one-off overrides: `--params '{"entry_score": 65}'`.

## Troubleshooting

* **"could not load prices"**: no internet, or Yahoo is rate-limiting. Retry in a minute, or use `--source stooq`, or `--source sample` for the bundled tickers (`MSFT`, `AAPL`, `JNJ`, `XOM`, `JPM`, `UNH`, `KO`, `PG`, `PFE`, `WMT`, `HD`, `GE`, `CVX`, `LLY`, `BAC`, `AMD`, `MRK`, `PEP`, `BBY`, `RRC`).
* **"No analyst memo"**: the ticker has no SEC filings (an ETF, a fund, a foreign company), or EDGAR could not be read
  (no `LIGHTHOUSE_SEC_EMAIL`, or no connection). The verdict and the ticket do not need the memo.
* **"Prices are from …"** on a ticket: the data is not today's (offline pack or cache). Refresh with live data before placing an order.
* **"Today's session is still trading"**: during market hours the unfinished daily bar is left out of the signals; orders are priced from the live quote. Yahoo quotes can lag by up to 15 minutes on some exchanges, so confirm the price with your broker.
* **"Left out MSFT: data ends 2017-11-10"** in the portfolio builder: the offline MSFT file ends earlier than the others, so it is excluded rather than cutting everyone's history short.
* **Volume beacon grey**: the series has no volume (index data or the offline pack's close-only files); the other beacons are re-weighted.
* **Set `LIGHTHOUSE_OFFLINE=1`** to force offline mode (used by the test-suite and CI).

For better data sources and how machine learning could be added, see `docs/DATA_AND_ML.md`.

Lighthouse is a decision-support tool, not investment advice.
