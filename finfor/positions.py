"""Track what you actually hold, and how it's done since you last synced it.

Two operations:
  - apply_proposals(): overwrite the held-positions sheet with this round's
    funded buy proposals, for use right after you've placed the matching
    trades in Robinhood. The previous sheet is backed up first (never
    silently discarded) since this is a destructive rewrite of your own
    bookkeeping data.
  - compute_performance(): for whatever's currently on the sheet, how far
    has price moved since cost_basis/last_updated was recorded.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from finfor.data_fetch import DATA_DIR, load_history

POSITIONS_FP = DATA_DIR / "positions.csv"
POSITIONS_HISTORY_DIR = DATA_DIR / "positions_history"

POSITIONS_COLUMNS = ["symbol", "shares", "cost_basis", "last_updated"]


def load_positions() -> pd.DataFrame:
    if POSITIONS_FP.exists():
        df = pd.read_csv(POSITIONS_FP)
        for col in POSITIONS_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
        return df[POSITIONS_COLUMNS]
    return pd.DataFrame(columns=POSITIONS_COLUMNS)


def save_positions(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(POSITIONS_FP, index=False)


def backup_positions() -> Path | None:
    """Snapshot the current positions sheet before it gets overwritten."""
    if not POSITIONS_FP.exists():
        return None
    current = pd.read_csv(POSITIONS_FP)
    if current.empty:
        return None
    POSITIONS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fp = POSITIONS_HISTORY_DIR / f"positions_{stamp}.csv"
    current.to_csv(fp, index=False)
    return fp


def apply_proposals(funded: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    """Build a fresh positions sheet from this round's funded proposals.

    `funded` must have symbol, last_close, allocation_dollars. Shares are
    sized to the dollar allocation at the current price — Robinhood supports
    fractional shares, so this isn't rounded to whole shares.
    """
    today = today or dt.date.today()
    out = pd.DataFrame({
        "symbol": funded["symbol"],
        "shares": (funded["allocation_dollars"] / funded["last_close"]).round(4),
        "cost_basis": funded["last_close"].round(2),
        "last_updated": today.isoformat(),
    })
    return out.reset_index(drop=True)


def compute_performance(positions: pd.DataFrame) -> pd.DataFrame:
    """For each held position, price movement since cost_basis was recorded."""
    if positions.empty:
        return positions.assign(
            current_price=pd.Series(dtype=float),
            pct_change=pd.Series(dtype=float),
            dollar_change=pd.Series(dtype=float),
            days_held=pd.Series(dtype=int),
            mood=pd.Series(dtype=str),
        )

    rows = []
    today = dt.date.today()
    for _, row in positions.iterrows():
        hist = load_history(row["symbol"])
        current_price = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None
        cost_basis = row.get("cost_basis")
        shares = row.get("shares")

        pct_change = None
        dollar_change = None
        if current_price is not None and pd.notna(cost_basis) and cost_basis:
            pct_change = (current_price / cost_basis - 1) * 100
            if pd.notna(shares):
                dollar_change = (current_price - cost_basis) * shares

        days_held = None
        last_updated = row.get("last_updated")
        if pd.notna(last_updated):
            try:
                last_updated_date = pd.Timestamp(last_updated).date()
                days_held = (today - last_updated_date).days
            except (ValueError, TypeError):
                days_held = None

        rows.append({
            **row.to_dict(),
            "current_price": current_price,
            "pct_change": pct_change,
            "dollar_change": dollar_change,
            "days_held": days_held,
            "mood": _mood_emoji(pct_change),
        })

    return pd.DataFrame(rows)


def _mood_emoji(pct_change: float | None) -> str:
    if pct_change is None or pd.isna(pct_change):
        return "\U0001F937"  # shrug — no price data yet
    if pct_change >= 3:
        return "\U0001F929"  # star-struck — big win
    if pct_change >= 1:
        return "\U0001F60A"  # smiling
    if pct_change > -1:
        return "\U0001F610"  # neutral
    if pct_change > -3:
        return "\U0001F615"  # confused/uneasy
    return "\U0001F62C"  # grimacing — ouch


def overall_mood(performance: pd.DataFrame) -> tuple[str, float | None]:
    """One headline emoji + total $ change for the whole sheet."""
    if performance.empty or "dollar_change" not in performance.columns:
        return "\U0001F937", None
    total = performance["dollar_change"].dropna().sum()
    if performance["dollar_change"].dropna().empty:
        return "\U0001F937", None
    total_cost = (performance["cost_basis"] * performance["shares"]).sum()
    pct = (total / total_cost * 100) if total_cost and pd.notna(total_cost) else None
    return _mood_emoji(pct), total
