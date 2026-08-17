"""Build the tradable-universe ticker list.

Robinhood publishes no official machine-readable symbol list, but it supports
essentially all NYSE/NASDAQ/AMEX-listed common stock. We approximate that
universe from NASDAQ Trader's public symbol directory, then narrow it with
liquidity/price filters in data_fetch.py so the downstream pipeline only
spends time on names that are actually tradable in practice.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Keywords in the security name that indicate it's not plain common stock.
_EXCLUDE_NAME_RE = re.compile(
    r"warrant|right|unit\b|units\b|preferred|depositary|trust preferred|"
    r"\bnotes\b|\bbond\b|when issued|convertible",
    re.IGNORECASE,
)

# Symbols with these characters are option-like / multi-class tickers that
# don't map cleanly to a Yahoo Finance symbol.
_BAD_SYMBOL_RE = re.compile(r"[.$^~]")


@dataclass
class UniverseRow:
    symbol: str
    name: str
    exchange: str


def _fetch_text(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _load_nasdaq_listed() -> pd.DataFrame:
    text = _fetch_text(NASDAQ_LISTED_URL)
    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    df = df.rename(columns={"Symbol": "symbol", "Security Name": "name"})
    df["exchange"] = "NASDAQ"
    return df[["symbol", "name", "exchange"]]


def _load_other_listed() -> pd.DataFrame:
    text = _fetch_text(OTHER_LISTED_URL)
    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    # Exchange codes: A = NYSE American (AMEX), N = NYSE, P = NYSE Arca, Z = BATS
    df = df[df["Exchange"].isin(["A", "N"])]
    exch_map = {"A": "AMEX", "N": "NYSE"}
    df = df.rename(columns={"ACT Symbol": "symbol", "Security Name": "name"})
    df["exchange"] = df["Exchange"].map(exch_map)
    return df[["symbol", "name", "exchange"]]


def fetch_common_stock_universe() -> pd.DataFrame:
    """Return symbol/name/exchange for common stock across NASDAQ/NYSE/AMEX."""
    nasdaq = _load_nasdaq_listed()
    other = _load_other_listed()
    combined = pd.concat([nasdaq, other], ignore_index=True)

    combined = combined.dropna(subset=["symbol", "name"])
    combined = combined[~combined["name"].str.contains(_EXCLUDE_NAME_RE)]
    combined = combined[~combined["symbol"].str.contains(_BAD_SYMBOL_RE, regex=True)]
    combined = combined[combined["symbol"].str.len() <= 5]
    combined = combined.drop_duplicates(subset="symbol").sort_values("symbol")
    combined = combined.reset_index(drop=True)
    return combined


if __name__ == "__main__":
    uni = fetch_common_stock_universe()
    print(f"Filtered common-stock universe: {len(uni)} symbols")
    print(uni.head(10))
    uni.to_csv("data/universe_raw.csv", index=False)
    print("Saved to data/universe_raw.csv")
