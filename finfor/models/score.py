"""Cheap universe-wide shortlisting, ahead of expensive per-ticker backtests.

Running a full walk-forward GARCH backtest on every liquid ticker (~1500+)
is too slow to redo on every refresh. Instead we first rank the whole
universe with fast, vectorized proxies for "reliable volatility," take the
top N, and only run the expensive backtest (finfor/backtest.py) on those.

Proxy for reliability, without any model fitting:
  - vol_level: 20d realized vol must clear a floor (translated from the
    user's "fluctuates >5%/month" spec: monthly stdev 5% ~= 17-18%
    annualized, so we require at least that much to bother).
  - vol_stability: vol_of_vol_20d relative to the vol level itself. A stock
    whose *volatility* is itself volatile is not "reliable" even if it
    moves a lot on average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHLY_VOL_FLOOR = 0.05
ANNUALIZED_VOL_FLOOR = MONTHLY_VOL_FLOOR * np.sqrt(12)  # ~0.173


def shortlist_candidates(
    features_by_symbol: dict[str, pd.DataFrame],
    top_n: int = 150,
    min_annualized_vol: float = ANNUALIZED_VOL_FLOOR,
    lookback_days: int = 20,
) -> pd.DataFrame:
    rows = []
    for sym, feat in features_by_symbol.items():
        recent = feat.tail(lookback_days)
        if recent.empty:
            continue
        last = feat.iloc[-1]
        vol_level = last.get("realized_vol_20d", np.nan)
        vol_of_vol = last.get("vol_of_vol_20d", np.nan)
        if pd.isna(vol_level) or vol_level < min_annualized_vol:
            continue
        if pd.isna(vol_of_vol) or vol_level == 0:
            continue

        # Lower relative vol-of-vol -> more "reliable" swings. Invert to a
        # 0-1 stability score (roughly: most stable stocks land near 1).
        relative_vol_instability = vol_of_vol / vol_level
        stability = float(1.0 / (1.0 + relative_vol_instability * 5))

        rows.append({
            "symbol": sym,
            "realized_vol_20d": vol_level,
            "vol_of_vol_20d": vol_of_vol,
            "stability_proxy": stability,
            "zscore_20d": last.get("zscore_20d", np.nan),
            "rsi_14": last.get("rsi_14", np.nan),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["shortlist_score"] = df["stability_proxy"] * df["realized_vol_20d"]
    df = df.sort_values("shortlist_score", ascending=False).head(top_n).reset_index(drop=True)
    return df
