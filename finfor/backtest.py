"""Walk-forward backtest: is this ticker's volatility actually predictable?

For each shortlisted ticker, step backward through its history at several
checkpoints. At each checkpoint, fit GARCH using only data available up to
that point (no lookahead), forecast the next `horizon_days` of volatility,
then compare against what actually happened. Also check whether the
mean-reversion direction signal was right more often than not. The result
is a per-ticker reliability score used to filter/rank the final proposals —
this is the thing that actually answers "can reliable volatility be
captured," not just an assumption baked into the model choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finfor.models.garch import fit_garch
from finfor.models.meanrev import direction_signal, direction_confidence

TRADING_DAYS_PER_YEAR = 252


def _realized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def walk_forward_backtest(
    feat: pd.DataFrame,
    n_checkpoints: int = 8,
    step_days: int = 15,
    horizon_days: int = 5,
    min_train_obs: int = 250,
) -> dict | None:
    """Backtest GARCH vol forecasts and the mean-reversion direction signal.

    `feat` is the full feature DataFrame for one ticker (from features.py),
    sorted ascending by date, with a `log_return` column.
    """
    n = len(feat)
    last_checkpoint_idx = n - horizon_days  # need horizon_days of future data to score
    checkpoints = []
    idx = last_checkpoint_idx
    for _ in range(n_checkpoints):
        idx -= step_days
        if idx < min_train_obs:
            break
        checkpoints.append(idx)
    checkpoints = sorted(checkpoints)
    if len(checkpoints) < 3:
        return None

    vol_forecast_errors = []
    vol_forecasts, vol_actuals = [], []
    direction_hits = []

    for cp in checkpoints:
        train = feat.iloc[:cp]
        future = feat.iloc[cp:cp + horizon_days]
        if len(future) < horizon_days:
            continue

        gf = fit_garch(train["log_return"], horizon_days=horizon_days, min_obs=min_train_obs)
        if gf is None:
            continue

        actual_weekly_vol = _realized_vol(future["log_return"]) / np.sqrt(TRADING_DAYS_PER_YEAR) * np.sqrt(horizon_days)
        forecast_weekly_vol = gf.weekly_move_pct
        vol_forecasts.append(forecast_weekly_vol)
        vol_actuals.append(actual_weekly_vol)
        vol_forecast_errors.append(abs(forecast_weekly_vol - actual_weekly_vol))

        cp_row = feat.iloc[cp - 1]
        sig = direction_signal(cp_row)
        conf = direction_confidence(cp_row)
        if conf > 0.3:
            actual_move = float(future["close"].iloc[-1] / train["close"].iloc[-1] - 1)
            hit = (np.sign(actual_move) == np.sign(sig)) if sig != 0 else None
            if hit is not None:
                direction_hits.append(hit)

    if len(vol_forecasts) < 3:
        return None

    vol_forecasts = np.array(vol_forecasts)
    vol_actuals = np.array(vol_actuals)
    corr = float(np.corrcoef(vol_forecasts, vol_actuals)[0, 1]) if len(vol_forecasts) > 2 else np.nan
    mape = float(np.mean(np.abs(vol_forecasts - vol_actuals) / np.maximum(vol_actuals, 1e-6)))
    direction_hit_rate = float(np.mean(direction_hits)) if direction_hits else np.nan

    # Reliability score: rewards vol-forecast accuracy (via correlation, and
    # penalized MAPE) and, where available, directional hit rate above 50/50.
    corr_component = max(0.0, corr) if not np.isnan(corr) else 0.0
    mape_component = max(0.0, 1 - min(mape, 2.0) / 2.0)
    dir_component = max(0.0, direction_hit_rate - 0.5) * 2 if not np.isnan(direction_hit_rate) else 0.0
    reliability_score = float(0.45 * corr_component + 0.35 * mape_component + 0.20 * dir_component)

    return {
        "n_checkpoints": len(vol_forecasts),
        "vol_forecast_corr": corr,
        "vol_forecast_mape": mape,
        "direction_hit_rate": direction_hit_rate,
        "n_direction_signals": len(direction_hits),
        "reliability_score": reliability_score,
    }
