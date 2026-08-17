"""Turn the raw scores into a plain-English action per ticker.

`direction`, `direction_confidence`, and `reliability_score` are useful for
debugging the model but don't tell you what to actually do. This module
converts them into one of a small set of concrete actions, plus a target
sell price / protective stop, so the app can show "do this" instead of
"here are three numbers, you figure it out."

Assumption: trades are placed as simple long stock positions in a Robinhood
cash account (no shorting, no options). `direction == "short"` setups are
therefore flagged as not directly actionable rather than given an allocation
— they're still shown so you can use them as a signal to trim/close an
existing long position in that name.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

RELIABILITY_SKIP = 0.30
CONFIDENCE_SKIP = 0.15

# Above these, the model's edge has held up consistently enough in backtest
# that riding it through the week (rather than taking quick partial profit)
# is reasonable.
HOLD_HIT_RATE = 0.65
HOLD_PERSISTENCE = 0.85
HOLD_CONFIDENCE = 0.35

# For "buy partial, limit-sell": how much of the forecast weekly move to
# target before taking profit, and how much adverse move to tolerate before
# cutting the position. Asymmetric on purpose (target > stop) since the
# signal is a magnitude forecast, not a guarantee.
TARGET_CAPTURE_FRACTION = 0.70
STOP_LOSS_FRACTION = 0.40

BUY_HOLD = "Buy & hold to next weekend"
BUY_LIMIT = "Buy partial, limit-sell at target"
SKIP_SHORT = "Skip (short setup, not long-tradable)"
SKIP_WEAK = "Skip (signal too weak)"


def _next_friday(today: dt.date | None = None) -> dt.date:
    today = today or dt.date.today()
    days_ahead = (4 - today.weekday()) % 7  # Monday=0 ... Friday=4
    return today + dt.timedelta(days=days_ahead)


def recommend_row(row: pd.Series, today: dt.date | None = None) -> dict:
    direction = row.get("direction")
    reliability = row.get("reliability_score", np.nan)
    conf = row.get("direction_confidence", np.nan)
    hit_rate = row.get("direction_hit_rate", np.nan)
    persistence = row.get("vol_persistence", np.nan)
    move = row.get("expected_weekly_move_pct", np.nan)
    last_close = row.get("last_close", np.nan)

    if direction == "short":
        return {
            "action": SKIP_SHORT,
            "rationale": "Model expects a decline, but shorting isn't part of this workflow. "
                         "If you already hold this name, treat it as a signal to consider trimming.",
            "target_sell_price": np.nan,
            "stop_loss_price": np.nan,
            "hold_until": pd.NaT,
        }

    if direction != "long" or pd.isna(reliability) or pd.isna(conf) or reliability < RELIABILITY_SKIP or conf < CONFIDENCE_SKIP:
        return {
            "action": SKIP_WEAK,
            "rationale": "Reliability or setup confidence too low to act on with real money.",
            "target_sell_price": np.nan,
            "stop_loss_price": np.nan,
            "hold_until": pd.NaT,
        }

    hold_worthy = (
        not pd.isna(hit_rate) and hit_rate >= HOLD_HIT_RATE
        and not pd.isna(persistence) and persistence >= HOLD_PERSISTENCE
        and conf >= HOLD_CONFIDENCE
    )

    target_sell_price = last_close * (1 + move * TARGET_CAPTURE_FRACTION) if not pd.isna(last_close) else np.nan
    stop_loss_price = last_close * (1 - move * STOP_LOSS_FRACTION) if not pd.isna(last_close) else np.nan
    hold_until = _next_friday(today)

    if hold_worthy:
        pct = round(hit_rate * 100)
        rationale = (
            f"Reliable, persistent setup ({pct}% historical direction hit rate, "
            f"sticky volatility regime) — worth riding through the week rather than "
            f"exiting early."
        )
        return {
            "action": BUY_HOLD,
            "rationale": rationale,
            "target_sell_price": target_sell_price,
            "stop_loss_price": stop_loss_price,
            "hold_until": hold_until,
        }

    rationale = (
        "Setup is real but less consistent — take partial profit near the forecast "
        "target instead of holding for the full move."
    )
    return {
        "action": BUY_LIMIT,
        "rationale": rationale,
        "target_sell_price": target_sell_price,
        "stop_loss_price": stop_loss_price,
        "hold_until": hold_until,
    }


def add_recommendations(df: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    recs = df.apply(lambda row: recommend_row(row, today=today), axis=1, result_type="expand")
    return pd.concat([df, recs], axis=1)


def _water_fill_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Redistribute weights (summing to 1) so none exceeds `cap`, sum preserved.

    Naively clipping then renormalizing can push a capped weight back above
    the cap (renormalization redistributes the clipped-off mass right back
    onto it). This does it properly: repeatedly pin anything over the cap and
    spread the freed-up budget across what's left, so the result both sums to
    1 and genuinely respects the cap (when `cap * len(weights) >= 1`, i.e. the
    cap is actually satisfiable).
    """
    weights = weights.astype(float).copy()
    fixed = {}
    remaining = list(weights.index)
    for _ in range(len(weights)):
        over = [i for i in remaining if weights[i] > cap]
        if not over:
            break
        for i in over:
            fixed[i] = cap
            remaining.remove(i)
        budget_left = 1 - sum(fixed.values())
        remaining_sum = weights[remaining].sum() if remaining else 0
        if remaining_sum > 0:
            for i in remaining:
                weights[i] = weights[i] / remaining_sum * budget_left
    for i in remaining:
        fixed[i] = weights[i]
    return pd.Series(fixed).reindex(weights.index)


def allocate(
    df: pd.DataFrame,
    n_positions: int,
    per_position_cap: float = 0.40,
) -> pd.DataFrame:
    """Split allocation across the top `n_positions` actionable buys.

    Weight is proportional to reliability_score * direction_confidence among
    rows with action in {BUY_HOLD, BUY_LIMIT}, capped per position so one
    name can't dominate. With fewer funded positions than `1 / per_position_cap`
    would allow, the cap is relaxed to `1 / n` (you explicitly chose to
    concentrate in fewer names) rather than leaving capital undeployed.
    """
    out = df.copy()
    out["suggested_allocation_pct"] = 0.0
    actionable = out[out["action"].isin([BUY_HOLD, BUY_LIMIT])].head(n_positions)
    if actionable.empty:
        return out

    weights = (actionable["reliability_score"] * actionable["direction_confidence"]).clip(lower=1e-6)
    weights = weights / weights.sum()

    effective_cap = max(per_position_cap, 1.0 / len(weights))
    weights = _water_fill_cap(weights, effective_cap)

    out.loc[weights.index, "suggested_allocation_pct"] = (weights * 100).round(1)
    return out
