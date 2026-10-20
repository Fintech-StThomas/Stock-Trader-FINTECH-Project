# 🔦 Lighthouse — explainable stock decisions for self-directed investors

*A clear signal in a noisy market.*

Lighthouse turns a ticker and a few facts about **you** — horizon, risk tolerance, the most you accept losing on one
position, the cash you can invest now — into a **Buy / Hold / Sell verdict and a trade ticket for that cash** (shares,
entry, stop-loss, exits, worst-case loss) from a price rule whose past behaviour it shows honestly, and puts an
**analyst memo** read from the company's SEC filings next to it: valuation, peers, earnings quality and what would prove
the thesis wrong. It also allocates whole portfolios, a basket you choose or one built for a goal, with a risk report.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lotainc0/Stock-Trader-FINTECH-Project/blob/main/notebooks/Lighthouse.ipynb)
![tests](https://github.com/lotainc0/Stock-Trader-FINTECH-Project/actions/workflows/ci.yml/badge.svg)

An example, from the offline sample (Microsoft, November 2017): the [report](reports/examples/MSFT/MSFT_report.md),
the [analyst memo](reports/examples/MSFT/MSFT_memo.md) and the
[interactive charts](reports/examples/MSFT/MSFT_charts.html) (download the file and open it in a browser).

## What it does

1. **Collects and cleans** daily prices & volume and the S&P 500 (Yahoo Finance → Stooq → cache → bundled real sample
   data, so it always runs), and the company's SEC EDGAR filings, each number dated by the filing that first reported it.
2. **Technical analysis** — 6 beacons: trend (50/200-day SMA, Golden/Death Cross), RSI, MACD, position in low-lag price
   bands (%B around a trend filter, ~1 session of lag), volume confirmation (OBV), 52-week-high proximity.
3. **Market weather** — Bull/Bear × Calm/Turbulent regime of the S&P 500 that removes points, raises the entry bar and
   halves sizes in bear markets.
4. **Decision** — the fundamental score (10 beacons from the SEC filings, against sector norms) and the technical
   score blended by horizon (long 60 / 40, short 30 / 70), minus market weather → verdict → position size from the
   *smallest* of three rules (your max-loss limit, a volatility target, a concentration cap), paid from the **cash you
   have available** (risk limits are a % of your whole account). The backtest also skips entries while the
   point-in-time fundamental score is below 35. No research signal was added: none survived its hold-out.
5. **Trade tickets** — limit entry, stop-loss and the exits (long-term buys: the backtest's month-end trend and signal
   exits, no fixed target; short-term trades: two take-profit levels in multiples of the risk (R), breakeven and
   trailing stops, a time stop), a cap on total open risk, and the record of those exact rules on each stock's own
   history. Every ticket has a **when to get out** table. "Buy now" orders are priced from a timestamped live quote;
   signals use finished sessions only.
6. **Analyst memo** — point-in-time financials, a bull / base / bear DCF and the growth the price implies, the
   stock's own valuation bands, fundamental beacons against industry peers (by SIC code) or sector norms, earnings
   quality and balance-sheet checks, capital allocation, industry metrics (banks, REITs, software, insurers), business
   segments, catalysts, the variant view, kill criteria and a pre-mortem; a thesis tracker checks the kill criteria
   against every new filing.
7. **Honest evaluation** — point-in-time backtest against buy-and-hold, the S&P 500 and two yardsticks for timing: the
   rule's own average exposure held without trading, and a plain 200-day trend rule; layer attribution, sensitivity
   grid, walk-forward out-of-sample test, regime breakdown.
8. **Portfolio builder** — allocate your basket (equal weight, risk parity, minimum variance) on a Ledoit-Wolf risk
   model, with stock/sector caps and a volatility target, or let Lighthouse pick stocks for a goal (preservation,
   balanced, growth, diversified). Whole-share orders, VaR / expected shortfall / beta / risk contributions / stress
   tests, and a walk-forward test of every method.
9. **The research behind it** — a point-in-time S&P 500 research harness (rank ICs, quintiles after costs, factor
   alphas, a hold-out evaluated once, every trial logged), survivorship-aware cohort tests, and an append-only forward
   journal of every call; three near-miss signals are recorded forward as watch-only.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # the web app (opens in your browser)
python -m lighthouse analyze MSFT --horizon long --risk balanced --max-loss 2 --cash 5000 --holdings 20000 --out reports/MSFT --evaluate
python -m lighthouse screen AAPL MSFT NVDA JPM XOM UNH --cash 10000
python -m lighthouse trade MSFT AAPL JNJ XOM --cash 5000                    # what to buy right now, with stops and exits
python -m lighthouse memo MSFT                                              # the analyst memo alone
python -m lighthouse portfolio AAPL JNJ XOM JPM KO PG --method risk_parity --cash 10000
python -m lighthouse portfolio --goal balanced --cash 10000                 # Lighthouse picks the stocks
python -m pytest                # 292 tests, runs offline
```

```python
from lighthouse import analyze, InvestorProfile
r = analyze("MSFT", InvestorProfile.from_cash(5_000, holdings_value=20_000, horizon="long", risk="balanced", max_loss_pct=2))
print(r.decision.headline())     # e.g. MSFT: STRONG BUY (Lighthouse score 72/100) for a long-term, balanced investor
print(r.report())                # the full plain-English report
r.backtest.summary()             # the rule vs the same exposure never traded vs buy & hold vs the S&P 500

from lighthouse.analyst import memo
m = memo.build("MSFT", result=r)              # the fundamental case from the SEC filings
print(m.stance, m.valuation.target); print(m.markdown())

from lighthouse.tradeplan import build_trade_plan
ticket = build_trade_plan(r)
print("\n".join(ticket.management_steps()))  # buy N shares, the stop, the exits
ticket.stats.avg_r                           # how these exact rules did on MSFT's history (R per trade, after costs)

from lighthouse import portfolio as pf
port = pf.build_portfolio(["AAPL", "JNJ", "XOM", "JPM", "KO", "PG"], "risk_parity", capital=10_000)
port.table(); port.risk["var95_1d_usd"]; port.comparison   # orders, risk, walk-forward test of every method
```

No internet? Set `LIGHTHOUSE_OFFLINE=1` (or `--source sample`): the bundled pack holds real Microsoft OHLCV 1986-2017,
19 other S&P 500 stocks and the index to 2022, with the SEC filings of all 20. Live memos need a contact e-mail for SEC
EDGAR in `LIGHTHOUSE_SEC_EMAIL` (sent to sec.gov only).

## The web app

`streamlit run app.py` starts a point-and-click site with four tabs:

| Tab | What you can do |
|---|---|
| **Analyse a stock** | the verdict dial and five cards (Lighthouse score, fundamental score, technical score, market weather, the memo's stance), then five sub-tabs: **plan & ticket** (how the score is built, a position plan sized from your cash with a *what-if* sizer, the trade ticket with its *when to get out* table and the rules' record on this stock), **analyst memo** (the fundamental case from the SEC filings), **beacons & charts** (the 10 fundamental and 6 technical beacons as bars and tables, a price / volume / RSI / MACD chart with low-lag price bands and the backtested entries and exits), **backtest** (against the same exposure never traded, buy & hold and the S&P 500; deep evaluation with layers, the plain trend rule, sensitivity heatmap and walk-forward) and **data** (sources, cleaning notes, every indicator row, downloads) |
| **Find trades** | pick a pre-built universe (a sideways-scrolling row: mega caps, dividend & defensive, each sector, semiconductors, sector and index ETFs, all large caps) or type your own; the scan ranks every ticker (select a row to jump to its full analysis) and funds the best setups from your cash, one ticket each, within a total-risk cap |
| **Portfolio** | allocate your own basket with three methods, or let Lighthouse build one for a goal; whole-share orders, risk report (volatility, VaR, expected shortfall, beta, risk contributions, correlations, stress tests) and a walk-forward test of every method |
| **How it works** | the method, verdict bands, risk presets, what the research found and the full user guide |

The theses you follow, the forward decision journal and the watch-only research signals run from the command line and
the scheduled GitHub workflows (`python -m lighthouse thesis`, `python -m lighthouse journal`; `docs/USER_GUIDE.md`);
their app views were archived in October 2026.

Every chart is interactive (hover, zoom, PNG download) and every table can be sorted; all numbers download as CSV or
Markdown. The sidebar's *time machine* answers "what would Lighthouse have said on this date?" (the memo then uses only
the filings made by that date). Keyboard focus is always visible and no information is carried by colour alone.

### Put it online (free, 5 minutes)

The app runs as-is on [Streamlit Community Cloud](https://share.streamlit.io):

1. Sign in at https://share.streamlit.io with your GitHub account.
2. *Create app* → *Deploy a public app from GitHub* → repository `lotainc0/Stock-Trader-FINTECH-Project`, branch `main`, main file `app.py`.
3. For live memos, add `LIGHTHOUSE_SEC_EMAIL = "you@example.com"` under *Advanced settings* → *Secrets*.
4. Press *Deploy*. The first build installs `requirements.txt`; after that the app is live at a `*.streamlit.app` URL you can share.

Later pushes to `main` reach the running app without a restart; `app.py` notices that the `lighthouse` package changed on disk
and loads the new code. If a page still shows an error right after a push, *Manage app* → ⋮ → *Reboot app* starts it fresh.

## Results in one table

Long-horizon rule, balanced profile, 20 S&P 500 stocks, Dec 1990 – Dec 2022, liquidity-scaled costs, resting stops,
next-day fills, cash at 2% (`reports/examples/cross_ticker_evaluation.csv`, rebuilt by
`python scripts/cross_ticker_evaluation.py`):

| | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & hold | 13.3% | 33.5% | 0.50 | −69% |
| Lighthouse signals + regime (fully invested when in) | 10.9% | — | 0.47 | −55% |
| **Lighthouse balanced (vol-targeted)** | 7.3% | 12.2% | 0.47 | **−27%** |
| Lighthouse conservative | 5.8% | 8.5% | 0.47 | −18% |
| Same exposure as balanced, never traded (43% in the stock) | 8.0% | 13.0% | 0.50 | −33% |
| Plain 200-day trend rule (fully invested above the SMA) | 9.3% | — | 0.41 | −59% |

Lower drawdown and volatility than buy-and-hold in **20 of 20** stocks; a higher Sharpe ratio than buy-and-hold in 9
of 20, than the same exposure without timing in 9 of 20, than the plain trend rule in 14 of 20. (Without the
fundamental gate, the price-only rule of 15-16 October: 7.6%, Sharpe 0.48, −28%.) Those 20 stocks are
today's survivors, which flatters every rule.

On **every S&P 500 member at a past date**, followed until today or until its prices stop
(`scripts/cross_ticker_evaluation.py --cohort DATE`, live data, historical membership; medians per stock; run on the
price-only rule of 15 October, before the fundamental gate came back):

| Cohort (members with prices / all) | Buy & hold Sharpe | Lighthouse balanced Sharpe | Same exposure, never traded | Plain 200-day trend rule | Beats buy & hold (Sharpe) | Equal-weight cohort, a year: buy & hold → Lighthouse |
|---|---|---|---|---|---|---|
| 2 Jan 1996, to Oct 2026 (260 / 487, 53%) | 0.37 | 0.28 | 0.37 | 0.27 | 79 of 260 | 11.1% → 5.5% (8.9% → 3.4% if every unpriced member went to zero) |
| 2 Jan 2009, to Oct 2026 (331 / 498, 66%) | 0.48 | 0.33 | 0.48 | 0.32 | 48 of 331 | 16.9% → 5.8% (14.2% → 3.4%) |

We say it out loud: as a way to time single stocks, the rule has shown no edge. A test whose rule we committed before
running it (`docs/METHODOLOGY.md` §8d) found that holding the rule's own average exposure without trading earned more
than the rule on most stocks of both cohorts, that the drawdown it saved against that exposure in the 1996 cohort
turned into a loss in 2009, and that a plain 200-day trend rule did about as well on Sharpe. What Lighthouse delivers
is the risk you asked for, mostly through sizing. Nor do its scores pick stocks: on the point-in-time S&P 500 none of
them ranked next month's returns, before or after costs, and neither did any of 37 further signals built from free data
(value, quality, earnings, momentum, insider and short-sale activity, 13F breadth, 10-K and earnings-release text):
three passed the 2009-2017 test and none the 2018-2026 hold-out, also after a fix to their share counts (56 trials
logged). So no research signal was added to the verdict; its own fundamental and technical beacons stay, by the user's
choice (`docs/METHODOLOGY.md` §2, §8-8d, `docs/DATA_AND_ML.md` §5-6).

## Repository map

```
lighthouse/            the product (pure Python: pandas, numpy, scipy, plotly)
  config.py            investor profiles, strategy parameters, verdict bands  <- all the numbers to defend
  data.py              live/offline price loading, cleaning report, caching, NYSE calendar
  http.py · store.py · cache.py   polite HTTP per host (SEC User-Agent and rate limit, retries, offline guard) ·
                       Parquet cache · in-memory memo of indicator work
  edgar.py · pit.py    SEC EDGAR client (facts, filings, form indexes, bulk data sets) · point-in-time statements and
                       the cover-page share count
  membership.py        historical S&P 500 membership; total_return.py dividends, splits (and the one share-count
                       restatement every market value uses), delisting rules
  costs.py · factors.py  liquidity-scaled trading costs · Ken French factors and industries
  indicators.py        causal SMA/EMA/WMA/Hull MA/Holt trend/RSI/MACD/low-lag price bands/ATR/OBV/volatility/drawdown
  technicals.py        technical beacons and score
  fundamentals.py      the memo's fundamental beacons: sector norms, or peer medians
  regime.py            market weather
  decision.py          score, verdict, exit level, position sizing (cash-capped)
  tradeplan.py         trade tickets, exit plan, stop-loss & take-profit management, historical setup test, idea
                       allocation, position replay
  portfolio.py         Ledoit-Wolf estimates, three allocation methods, caps, vol target, risk report, walk-forward, goals
  journal.py · scorecard.py · broker.py   append-only decision journal · forward scorecard · Alpaca paper adapter
  research/            the cross-sectional harness (point-in-time S&P 500 panel, ICs, quintiles after costs, factor
                       alphas, hold-out guard, experiment log), the signals still tested (the product's own,
                       fundamental, FINRA short volume, earnings reaction, 10-K text), the cohort test and its
                       pre-registered timing test, and watch.py, which records the near misses at each month end
  analyst/             single-stock toolkit: point-in-time financials, linked model, DCF / reverse DCF, peers,
                       industry metrics (banks, REITs, software, insurers), XBRL segments, checks, memo, thesis tracker
  text.py              deterministic text features (Loughran-McDonald)
  universe.py          pre-built universes and curated sector labels
  backtest.py          point-in-time engine, metrics, the two timing yardsticks, layers, sensitivity, walk-forward,
                       regime breakdown
  engine.py            analyze() pipeline and AnalysisResult
  screener.py          watchlist ranking
  report.py            Markdown report and investment memo
  interactive.py       interactive Plotly charts (app, notebook, saved reports)
  __main__.py          CLI
app.py                 the web app (Streamlit + Plotly)
notebooks/Lighthouse.ipynb   end-to-end Colab/Jupyter deliverable (built by scripts/build_notebook.py)
scripts/               the evaluations behind every published number (cross_ticker_evaluation, research_signals) and
                       the sample-pack recorders
data/sample/           bundled real historical data: prices, SEC filings (facts, filing lists, profiles), dividends
reports/examples/      the MSFT example (report, memo, charts) and the cross-ticker and cohort evaluations
reports/research/      the research harness's experiment log, hold-out log, results and verdicts
docs/                  EXECUTIVE_SUMMARY.md · METHODOLOGY.md · PITCH.md · USER_GUIDE.md · DATA_AND_ML.md · NEXT_SESSION.md
tests/                 292 pytest tests, offline (indicators, scoring, sizing, no-look-ahead, parsers on recorded
                       SEC and FINRA excerpts, research harness, analyst toolkit, trade tickets, portfolios, pipeline, app)
```

## Limitations

Free data drops the dead: Yahoo has no prices for most delisted S&P 500 members, so every cohort and cross-sectional
result is a bound, not an estimate (coverage is published with each). Fundamentals start with XBRL in 2009; the memo's
sector norms and DCF inputs are assumptions (a five-year fade to the long-run growth rate, a historical equity
premium: the geometric mean since 1963, 6.2%, with the arithmetic 7.2% as a sensitivity), and peers are opt-in;
trading costs come from daily ranges, which overstate large-cap spreads; the backtests ignore taxes; stops are orders,
not guaranteed prices. Paper trading is tested against Alpaca's documented formats, not a live account.
`docs/DATA_AND_ML.md` lists what free data can and cannot prove, the sources that would close these gaps and how
machine learning could be added safely. Lighthouse is decision support, not investment advice.

MIT licensed.
