# FinFor

Speculative research tool for finding US equities with "reliable volatility" —
names that swing predictably enough on a daily/weekly cadence to be worth a
small, manual, high-risk position. **This is not investment advice, it has
no live trading integration, and it is meant to be run against money you can
afford to lose.** You review the output and place trades yourself in
Robinhood.

## How it works

1. **Universe** (`finfor/universe.py`) — pulls NASDAQ/NYSE/AMEX common-stock
   symbols from NASDAQ Trader's public directory (~5,500 names), filtering
   out ETFs, warrants, units, preferred shares, and test issues. This is the
   best available proxy for "tradable on Robinhood" since Robinhood doesn't
   publish a symbol list.
2. **Liquidity screen** (`finfor/data_fetch.py: screen_liquidity`) — a fast
   1-month batch pull to cut the ~5,500 names down to ones with a sane price
   ($3–$2000) and real daily dollar volume (≥$5M/day average), so the rest
   of the pipeline isn't wasted on illiquid micro-caps and pre-merger SPACs.
3. **History cache** (`finfor/data_fetch.py: refresh_history`) — 3 years of
   daily OHLCV per surviving symbol, cached to `data/history/*.parquet`.
   Re-running skips anything refreshed in the last 3 days, so running the
   pipeline twice a week doesn't re-download everything each time.
4. **Features** (`finfor/features.py`) — see the full input list below.
5. **Shortlist** (`finfor/models/score.py`) — cheap, vectorized proxy for
   "reliable volatility" (vol level above the ~5%/month floor you specified,
   ranked by how stable that vol level itself has been) narrows the liquid
   universe to the top ~150 candidates. This exists so the expensive step
   below doesn't have to run on 1,500+ names every time.
6. **Backtest** (`finfor/backtest.py`) — walk-forward validation, only on the
   shortlist: at several past checkpoints, fit GARCH using only data
   available at that point, forecast the next week's volatility, and check
   it against what actually happened. Also checks the mean-reversion
   direction signal's historical hit rate. This produces the
   `reliability_score` — the actual evidence for whether a given ticker's
   volatility is predictable, not just an assumption.
7. **Live forecast + ranking** (`finfor/pipeline.py`) — fits GARCH and the
   direction signal on full current history for the shortlist, filters to
   your 2–10% weekly move target, ranks by reliability × expected move ×
   direction confidence, and writes `data/proposals/proposals_<date>.csv`.
8. **Recommendation** (`finfor/models/recommend.py`) — turns the raw scores
   into a plain-English action per ticker instead of leaving you to interpret
   `direction`/`confidence`/`reliability_score` yourself:
   - **Buy & hold to next weekend** — high reliability, high historical
     direction-hit-rate, sticky volatility regime. Worth riding through the
     week rather than exiting early.
   - **Buy partial, limit-sell at target** — the setup is real but less
     consistent; take profit near a target price (~70% of the forecast move)
     instead of holding for the full move. Comes with a suggested limit-sell
     price and a protective stop.
   - **Skip (short setup, not long-tradable)** — the model expects a decline.
     Not funded since this workflow assumes simple long stock positions in a
     cash account (no shorting/options). Still shown, since it's a signal to
     trim an existing long position in that name if you hold one.
   - **Skip (signal too weak)** — reliability or confidence too low to act on.

   Only "buy" rows get an allocation. The app's **Positions to fund** slider
   picks how many of the top-ranked buys to fund; **suggested_allocation_pct**
   splits your stated capital across them, weighted by reliability ×
   confidence and capped at 40% in any single name (relaxed only when you've
   chosen to fund fewer positions than the cap would otherwise allow).
9. **App** (`app/streamlit_app.py`) — button to re-run the pipeline, a table
   of ranked proposals with recommended action/allocation/target/stop/hold-by
   date, a per-ticker chart, and a simple sheet to track what you actually
   hold in Robinhood.

## Input data used

| Category | Fields |
|---|---|
| Price / return | daily OHLC, 1d/5d/20d returns |
| Realized volatility | rolling stdev of returns (5d/10d/20d/60d, annualized), vol-of-vol (20d) |
| Range-based volatility | ATR(14) normalized by price, daily high-low range |
| Mean-reversion | 20d price z-score, Bollinger %B, Bollinger bandwidth |
| Momentum | RSI(14), MACD histogram |
| Trend | distance from 50d and 200d SMA |
| Volume | volume relative to 20d average |
| Macro | VIX level, VIX 5-day change |
| Model-derived | GARCH(1,1) conditional volatility forecast, vol persistence (α+β) |

Not currently included, but reasonable next additions if the model needs
more signal: earnings-date proximity, options-implied volatility, short
interest, sector/industry grouping, fundamentals (P/E, revenue growth,
market cap). These were left out of v1 because they either require
per-ticker (non-batchable) API calls that don't scale to a ~1,500-name
universe, or a paid data source.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline from the command line:

```bash
source .venv/bin/activate
python3 -m finfor.pipeline
```

Or use the app (recommended — this is the "spin it up once or twice a week"
workflow):

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

Click **Run pipeline now** in the sidebar, review the ranked proposals table,
place whatever trades you're comfortable with manually in Robinhood, and log
them in the **My positions** tab so future runs are easier to sanity-check
against what you're actually holding.

## Honest limitations

- `reliability_score` is backtested on the same regime the market's been in
  recently — it does not guarantee future predictability, especially for a
  vol-based strategy where the whole premise is regime-dependent.
- The mean-reversion direction signal is intentionally simple (z-score + RSI).
  It's the piece most likely to need iteration once you see how proposals
  perform against real fills.
- No transaction cost, slippage, or bid/ask spread modeling — at
  daily/weekly holding periods on possibly-volatile small/mid caps, spreads
  can eat into the 2–10% target meaningfully. Check the spread before
  placing a trade.
- Position sizing is capped per-name (40%, relaxed only for very small
  position counts) but doesn't account for correlation across the names
  it picks — e.g. it could still hand you three highly-correlated
  semiconductor names in one round. No max-drawdown or stop-loss automation;
  the suggested stop price is just that, a suggestion you'd place yourself.
- The action thresholds in `finfor/models/recommend.py` (hit-rate ≥65%,
  vol persistence ≥0.85, etc. for "hold" vs "limit-sell") are reasonable
  starting heuristics, not themselves backtested — unlike `reliability_score`,
  there's no historical evidence yet that this specific hold-vs-limit-sell
  split outperforms just always doing one or the other.
