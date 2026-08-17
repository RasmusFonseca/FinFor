"""Liquidity screening and OHLCV history download/caching.

Two stages, both needed because the raw NASDAQ/NYSE/AMEX universe (~5500
symbols after universe.py's filtering) is dominated by illiquid names (SPACs
pre-merger, micro-caps that barely trade) that aren't practical to trade on a
daily/weekly cadence with tight spreads:

1. screen_liquidity(): quick 1-month batch pull, used only to compute avg
   price and avg dollar volume, to cut the universe down to something
   "reasonable" to fully model.
2. refresh_history(): full daily OHLCV history for the surviving symbols,
   cached to local parquet so re-running the pipeline doesn't re-download
   everything each time.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_DIR = DATA_DIR / "history"
MACRO_DIR = DATA_DIR / "macro"

DEFAULT_MIN_PRICE = 3.0
DEFAULT_MAX_PRICE = 2000.0
DEFAULT_MIN_DOLLAR_VOL = 5_000_000  # $5M/day average traded value
BATCH_SIZE = 250


def _batched(seq: list[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def screen_liquidity(
    universe: pd.DataFrame,
    lookback: str = "1mo",
    min_price: float = DEFAULT_MIN_PRICE,
    max_price: float = DEFAULT_MAX_PRICE,
    min_dollar_vol: float = DEFAULT_MIN_DOLLAR_VOL,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Screen the universe down to liquid, sanely-priced names."""
    symbols = universe["symbol"].tolist()
    rows = []
    for i, batch in enumerate(_batched(symbols, batch_size)):
        if verbose:
            print(f"[screen] batch {i + 1}/{-(-len(symbols)//batch_size)} ({len(batch)} symbols)")
        try:
            df = yf.download(
                batch, period=lookback, interval="1d", group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  batch failed: {e}")
            continue

        for sym in batch:
            try:
                sub = df[sym] if len(batch) > 1 else df
            except (KeyError, TypeError):
                continue
            sub = sub.dropna(subset=["Close", "Volume"])
            if sub.empty or len(sub) < 5:
                continue
            avg_price = sub["Close"].mean()
            avg_dollar_vol = (sub["Close"] * sub["Volume"]).mean()
            rows.append({"symbol": sym, "avg_price": avg_price, "avg_dollar_vol": avg_dollar_vol})

    screened = pd.DataFrame(rows)
    if screened.empty:
        return screened
    screened = screened.merge(universe, on="symbol", how="left")
    screened = screened[
        (screened["avg_price"] >= min_price)
        & (screened["avg_price"] <= max_price)
        & (screened["avg_dollar_vol"] >= min_dollar_vol)
    ].reset_index(drop=True)
    screened = screened.sort_values("avg_dollar_vol", ascending=False).reset_index(drop=True)
    return screened


def refresh_history(
    symbols: list[str],
    period: str = "3y",
    interval: str = "1d",
    batch_size: int = BATCH_SIZE,
    max_age_days: int = 3,
    force: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Download/refresh daily OHLCV for symbols, caching each to a parquet file.

    Returns the list of symbols that now have usable cached history.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    to_fetch = []
    for sym in symbols:
        fp = HISTORY_DIR / f"{sym}.parquet"
        if not force and fp.exists() and (now - fp.stat().st_mtime) < max_age_days * 86400:
            continue
        to_fetch.append(sym)

    if verbose:
        print(f"[history] {len(symbols) - len(to_fetch)} cached & fresh, {len(to_fetch)} to fetch")

    ok = [s for s in symbols if s not in to_fetch]
    for i, batch in enumerate(_batched(to_fetch, batch_size)):
        if verbose:
            print(f"[history] batch {i + 1}/{-(-len(to_fetch)//batch_size)} ({len(batch)} symbols)")
        try:
            df = yf.download(
                batch, period=period, interval=interval, group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  batch failed: {e}")
            continue

        for sym in batch:
            try:
                sub = df[sym] if len(batch) > 1 else df
            except (KeyError, TypeError):
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty or len(sub) < 60:
                continue
            sub.to_parquet(HISTORY_DIR / f"{sym}.parquet")
            ok.append(sym)

    return ok


def load_history(symbol: str) -> pd.DataFrame | None:
    fp = HISTORY_DIR / f"{symbol}.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


def refresh_macro(max_age_days: int = 1, force: bool = False, verbose: bool = True) -> None:
    """Pull macro series used as extra features (currently: VIX)."""
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    fp = MACRO_DIR / "vix.parquet"
    if not force and fp.exists() and (time.time() - fp.stat().st_mtime) < max_age_days * 86400:
        return
    df = yf.download("^VIX", period="3y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_parquet(fp)
    if verbose:
        print("[macro] refreshed VIX")


def load_macro_vix() -> pd.DataFrame | None:
    fp = MACRO_DIR / "vix.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


if __name__ == "__main__":
    from finfor.universe import fetch_common_stock_universe

    uni = fetch_common_stock_universe()
    screened = screen_liquidity(uni)
    print(f"Screened to {len(screened)} liquid symbols")
    screened.to_csv(DATA_DIR / "universe_screened.csv", index=False)

    ok = refresh_history(screened["symbol"].tolist())
    refresh_macro()
    print(f"History cached for {len(ok)} symbols")
