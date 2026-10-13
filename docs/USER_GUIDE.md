# Lighthouse user guide

Lighthouse answers four questions for **you** specifically: *should I buy this stock, how much can I buy with the cash I
have, where would I be wrong (and where do I take profit), and how should I split my money across several stocks?*

## Three ways to run it

| | Best for | How |
|---|---|---|
| **Google Colab / Jupyter notebook** | the course deliverable; running end-to-end with narration | open `notebooks/Lighthouse.ipynb`, run all cells; change the inputs cell |
| **Web app** | everyone: interactive charts, trade desk, portfolio builder, what-if sizing, downloads | `pip install -r requirements.txt` then `streamlit run app.py` (or the hosted Streamlit Cloud link in the README) |
| **Command line** | power users, batch runs, saving reports | see *Command line* below |

## The inputs

| Input | Meaning | Effect |
|---|---|---|
| Ticker | any US-listed symbol | |
| Horizon | **long** (6-24+ months) or **short** (weeks to a few months) | long: fundamentals 60% / technicals 40%, monthly review, exit below the 200-day line; short: 30/70, weekly review, ATR trailing stop |
| Risk tolerance | conservative / balanced / aggressive | max single position 10 / 20 / 35% of the account; volatility target 10 / 15 / 25%; stop distance 3 / 2.5 / 2 ATR (×2 for long horizon) |
| Max acceptable loss | % of your **whole account** you accept losing on one position if its stop is hit | position size = max loss ÷ distance to the stop, if that is the tightest rule |
| **Cash available to invest now** | the money you can put to work today | every new position is paid from this cash only; nothing you own is sold |
| **Value of stocks you already hold** | optional; 0 if this cash is all you have invested | only used for the risk limits: max loss and the concentration cap are a % of cash + holdings |
| Years of history | default 10 (notebook 15) | analysis window and backtest length |
| Data source | auto / yfinance / stooq / cache / sample | `auto` tries live sources then offline fallbacks; `sample` uses the bundled historical pack |
| 🔄 Refresh live prices | sidebar button | fetches the latest quote now; otherwise an analysis and its quote are reused for 15 minutes |

Example: $5,000 cash and $20,000 of stocks already held, 2% max loss. The account is $25,000, so one position may lose at most
$500 at its stop, but it can never cost more than the $5,000 of cash.

## Reading the Analyse tab

* **Price line under the name**: the signals use *finished* sessions only; "buy now" orders are priced from the latest
  quote and its time (for example "$341.20 at 2026-10-13 10:42 EDT (market open)"), or the last close when there is no
  quote (offline, or the time machine). During market hours today's unfinished bar is left out of every indicator
  (its volume and range are not final) and a note says so.
* **Verdict and Lighthouse score (0-100)**: ≥72 Strong Buy · ≥60 Buy · ≥45 Hold · ≥32 Reduce · <32 Sell.
* **Beacons**: every signal with its reading, a light (🟢 bullish, 🟡 neutral, 🔴 bearish, ⚪ unavailable), its vote (−1…+1), weight and the points it added.
* **Market weather**: Bull/Bear × Calm/Turbulent for the S&P 500, and how it changed the score and the size.
* **Position plan: buy now with your cash**: how many shares to buy and what they cost, the exit level (stop-loss), the
  worst-case loss if the stop fills at its price, and *which rule limited the size* (max loss, volatility target,
  concentration cap or your cash). Hold means keep what you own; Reduce means trim about half; Sell means exit.
* **Trade ticket**: switch between 📈 Long and 📉 Short. See *Trade tickets* below.
* **Price chart**: close, 50/200-day averages and **price bands** that follow the price: a 20-day Hull moving average
  (dashed) ± 2 standard deviations of the closes around it, about 3 sessions behind the price instead of the ~10 of
  classic Bollinger Bands. They use only past prices; the *Price bands* beacon scores where the close sits in them.
* **Backtest**: how the rule behind the verdict behaved historically versus buy-and-hold and the S&P 500.
* **Warnings**: data-source notes (offline data, missing volume, fixture fundamentals) and anything to know before acting.

## Trade tickets (Analyse tab and Trade desk)

A ticket is a complete order plan:

1. **Entry**: a limit order near the last price, and a do-not-chase price (skip the trade if it has already run 0.5 ATR past the entry).
2. **Stop-loss**: placed immediately. Longs use the horizon's stop; shorts always use a 2.5 × ATR stop (balanced risk).
   The planned loss is shown in dollars and as % of your account.
3. **Take profit 1** (2R by default, R = the distance to the stop): sell half and move the stop on the rest to breakeven.
4. **Trailing stop**: after target 1, the stop follows the best close since entry minus k × ATR. It never loosens.
5. **Take profit 2** (3R by default): sell the rest.
6. **Time stop**: close a trade that has reached neither the stop nor target 1 after 40 sessions (short horizon) or 252 (long).

Each ticket shows **how these exact rules did on this stock's own history**: number of trades, win rate, average result
in R per trade after costs, and profit factor. If the rules lost money on that stock, the ticket says so in red. Change the
targets, the partial exit, the time stop, the number of ideas and the total-risk cap under *Trade management* in the sidebar.

**When to get out** (on every ticket): one row per exit, for a loss or a profit, decided before you enter:

| Exit | Trigger | Result shown |
|---|---|---|
| 🛑 Stop-loss | the stop price | the planned loss (−1R) |
| 🎯 Target 1 | its price | the partial profit, which is also the result if the rest is later stopped at breakeven |
| 🎯 Trailing stop | follows the best close | between target 1's and target 2's result |
| 🎯 Target 2 | its price | the best planned outcome |
| ⏱️ Time stop | after 40 / 252 sessions | between the stop's loss and target 1's profit |
| 🧭 Verdict turns | at the weekly / monthly review | sell early (buy back for a short) |

Each row also says how often trades ended that way on the stock's own history and how many sessions it took.

**Status**: *actionable* (place the ticket), *watch* (close to a setup; levels shown, no order) or *avoid* (no trade).

## Trade desk tab

* **Universe**: a row of pre-built lists that scrolls sideways (mega caps, dividend & defensive, each sector,
  semiconductors, sector ETFs, index & bond ETFs, all large caps, the offline sample pack). Picking one fills the ticker
  box, which you can edit. ETFs have no fundamentals, so they are scored on technicals only.
* **📈 Long (buy)**: pick a universe (or type your own list) and press *Find trades now*.
  Lighthouse scores every ticker, keeps the actionable setups, and funds the strongest first from your cash. It stops when
  the cash, the idea count or the total planned risk ("portfolio heat", 6% of the account by default) runs out. You get one
  ticket per idea, a summary table, a watch list, and a CSV of the orders.
* **📉 Short (sell short)**: the same scan, mirrored. Read the banner first: shorts can lose more than planned, need a margin
  account and borrowable shares, and owe dividends. Each ticket shows the Regulation T and FINRA margin figures.
* There is no position-entry screen: every ticket already carries its exits (*When to get out*), so the sell decision is
  made, and the stop order placed, when you buy. To replay a trade you already hold against the plan, use the command
  line's `monitor` (below).

## Portfolio builder tab

* **My basket**: type your stocks, choose an allocation method and the caps (max per stock, optional max per sector,
  optional target volatility), and press *Build my portfolio*.
  * *Equal weight*: the same dollars in each. Hard to beat; the honest baseline.
  * *Inverse volatility*: calmer stocks get more.
  * *Risk parity* (default): each stock contributes the same share of risk.
  * *Minimum variance*: the lowest-volatility mix.
  * *Maximum Sharpe*: the best expected return per unit of risk. The most sensitive to estimation error.
  * *Hierarchical risk parity*: groups stocks that move together and splits the risk between groups.
* **Build for a goal**: choose *Capital preservation*, *Balanced risk and reward*, *Growth* or *Maximum diversification*.
  Lighthouse picks the stocks from the universe (any pre-built list, or your own) and weights them, at most 3 per sector.
  A ticker outside the curated list gets the sector its fundamentals report; one with no sector at all (an ETF, say) is
  not counted against the per-sector limits.

Every portfolio shows:
* whole-share **orders for your cash** (CSV download);
* **expected volatility**, **Value at Risk** and **expected shortfall** in dollars (1 day and 1 month, 95%), beta to the
  S&P 500, the diversification ratio and the effective number of stocks;
* where the money is versus where the risk is, and the correlation heatmap;
* a **walk-forward test** of every method (or of the goal, re-selecting its stocks every month) against the S&P 500 and
  equal weight;
* a stress test through past crashes covered by the data.

Judge performance by the walk-forward test. The back-cast of today's weights is shown for risk only: a goal portfolio picks
stocks *because* of their recent strength, so their recent past flatters them.

## Command line

```bash
python -m lighthouse analyze MSFT --horizon long --risk balanced --max-loss 2 --cash 5000 --holdings 20000 --out reports/MSFT --evaluate
python -m lighthouse screen AAPL MSFT NVDA JPM --risk conservative --cash 10000 --csv ranking.csv
python -m lighthouse backtest MSFT --years 15 --evaluate
python -m lighthouse trade MSFT AAPL JNJ XOM --side long --cash 5000 --holdings 20000 --targets 2 3
python -m lighthouse trade AAPL AMD BBY --side short --cash 5000 --source sample
python -m lighthouse trade --universe semiconductors --cash 5000                  # a pre-built universe
python -m lighthouse universes                                                    # list them
python -m lighthouse monitor MSFT --side long --entry 73.9 --shares 30 --date 2017-09-01 --source sample
python -m lighthouse portfolio AAPL JNJ XOM JPM KO PG --method risk_parity --cash 10000 --target-vol 0.12
python -m lighthouse portfolio --goal balanced --cash 10000
python -m lighthouse portfolio --goal growth --universe technology --cash 10000
```

`trade` prints each ticket's plan and its *when to get out* list. `monitor` replays a position you hold since its entry
with the same rules (target 1 sells half, the stop then moves to breakeven and trails) and reports the exits already
taken, the realised and unrealised P&L, and where the stop is now.

Add `--as-of 2020-03-20` to `analyze` to ask what Lighthouse would have said on a past day. `--portfolio` still works and
means "this much cash, nothing else held".

## Changing the rules without programming

All settings are in `lighthouse/config.py`: risk presets, horizon weights, beacon weights, verdict bands and the
`StrategyParams` dataclass (indicator windows, thresholds, review frequency, costs). Sector norms are in
`lighthouse/fundamentals.py` (`SECTOR_NORMS`). Trade-management defaults are in `TradeSettings`
(`lighthouse/tradeplan.py`), goal presets in `GOALS` (`lighthouse/portfolio.py`) and the large-cap universe with sector
labels in `lighthouse/universe.py`. The CLI accepts one-off overrides: `--params '{"entry_score": 65}'`.
If a fundamentals feed is missing a field, type your own: in the app's expert settings as JSON, or on the CLI with
`--fundamentals-json my_numbers.json` (same format as `data/sample/fundamentals/MSFT.json`).

## Troubleshooting

* **"could not load prices"**: no internet, or Yahoo is rate-limiting. Retry in a minute, or use `--source stooq`, or `--source sample` for the bundled tickers (`MSFT`, `AAPL`, `JNJ`, `XOM`, `JPM`, `UNH`, `KO`, `PG`, `PFE`, `WMT`, `HD`, `GE`, `CVX`, `LLY`, `BAC`, `AMD`, `MRK`, `PEP`, `BBY`, `RRC`).
* **"Prices are from …"** on a ticket: the data is not today's (offline pack or cache). Refresh with live data before placing an order.
* **"Today's session is still trading"**: during market hours the unfinished daily bar is left out of the signals; orders are priced from the live quote. Yahoo quotes can lag by up to 15 minutes on some exchanges, so confirm the price with your broker.
* **"Left out MSFT: data ends 2017-11-10"** in the portfolio builder: the offline MSFT file ends earlier than the others, so it is excluded rather than cutting everyone's history short.
* **"Fundamental data unavailable"**: the verdict is technical-only and flagged; supply overrides or try later.
* **Volume beacon grey**: the series has no volume (index data or the offline pack's close-only files); the other beacons are re-weighted.
* **Set `LIGHTHOUSE_OFFLINE=1`** to force offline mode (used by the test-suite and CI).

For better data sources and how machine learning could be added, see `docs/DATA_AND_ML.md`.

Lighthouse is a decision-support tool, not investment advice.
