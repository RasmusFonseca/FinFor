"""Feature engineering on cached OHLCV history.

Input signals computed here, per ticker per day:

Price / return
  - log_return (1d), return_5d, return_20d
Volatility (the core signal for "reliable volatility" ranking)
  - realized_vol_5d/10d/20d/60d: rolling stdev of daily log returns, annualized
  - atr_14: 14-day average true range, normalized by price
  - high_low_range: today's (high-low)/close
  - vol_of_vol_20d: rolling stdev of the 5d realized vol series itself
    (a stock whose volatility level is itself stable is what we mean by
    "reliable" volatility, as opposed to a stock whose vol randomly spikes)
Mean-reversion / momentum (candidate directional signals)
  - zscore_20d: (price - 20d SMA) / 20d rolling stdev of price
  - pct_b: Bollinger %B, position within the 20d/2-stdev bands [0,1]
  - bandwidth: Bollinger band width / SMA (regime: squeeze vs expansion)
  - rsi_14
  - macd_hist: MACD(12,26) minus its 9-period signal line
  - dist_sma50 / dist_sma200: price distance from longer moving averages
Volume
  - rel_volume_20d: today's volume / 20d average volume
Macro
  - vix_level, vix_chg_5d: market-wide volatility regime context

All ratios are unitless so they're comparable across tickers at wildly
different price levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finfor.data_fetch import load_history, load_macro_vix

TRADING_DAYS_PER_YEAR = 252


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window).mean()


def build_features(symbol: str, vix: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Compute the full feature set for one ticker. Returns None if no cached history."""
    df = load_history(symbol)
    if df is None or len(df) < 70:
        return None
    df = df.copy().sort_index()

    close = df["Close"]
    log_ret = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=df.index)
    feat["close"] = close
    feat["log_return"] = log_ret
    feat["return_5d"] = close.pct_change(5)
    feat["return_20d"] = close.pct_change(20)

    for w in (5, 10, 20, 60):
        feat[f"realized_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    feat["vol_of_vol_20d"] = feat["realized_vol_5d"].rolling(20).std()

    feat["atr_14"] = _atr(df, 14) / close
    feat["high_low_range"] = (df["High"] - df["Low"]) / close

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["zscore_20d"] = (close - sma20) / std20.replace(0, np.nan)
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    feat["pct_b"] = (close - lower) / (upper - lower).replace(0, np.nan)
    feat["bandwidth"] = (upper - lower) / sma20.replace(0, np.nan)

    feat["rsi_14"] = _rsi(close, 14)
    feat["macd_hist"] = _macd_hist(close) / close

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    feat["dist_sma50"] = (close - sma50) / sma50
    feat["dist_sma200"] = (close - sma200) / sma200

    if "Volume" in df.columns:
        vol20 = df["Volume"].rolling(20).mean()
        feat["rel_volume_20d"] = df["Volume"] / vol20.replace(0, np.nan)

    if vix is not None and not vix.empty:
        vix_close = vix["Close"].reindex(feat.index).ffill()
        feat["vix_level"] = vix_close
        feat["vix_chg_5d"] = vix_close.pct_change(5)

    feat["symbol"] = symbol
    return feat


def build_features_for_universe(symbols: list[str], verbose: bool = True) -> dict[str, pd.DataFrame]:
    vix = load_macro_vix()
    out = {}
    for i, sym in enumerate(symbols):
        f = build_features(sym, vix=vix)
        if f is not None:
            out[sym] = f
        if verbose and (i + 1) % 200 == 0:
            print(f"[features] {i + 1}/{len(symbols)} processed, {len(out)} usable")
    return out
