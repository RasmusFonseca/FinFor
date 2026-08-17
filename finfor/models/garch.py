"""GARCH(1,1) volatility forecasting, one fit per ticker.

GARCH directly targets "reliable volatility": it models the conditional
variance of returns as a function of past shocks and past variance, so it
captures vol clustering (calm periods stay calm, choppy periods stay choppy)
rather than just measuring realized vol after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class GarchForecast:
    symbol: str
    daily_vol_forecast: np.ndarray  # forecast path, length = horizon, in daily pct-return stdev
    weekly_move_pct: float  # sqrt-time-scaled expected magnitude of move over horizon, as a fraction
    annualized_vol: float
    persistence: float  # alpha + beta: how long vol shocks persist (near 1 = "reliable"/sticky regime)
    converged: bool


def fit_garch(
    log_returns: pd.Series,
    horizon_days: int = 5,
    min_obs: int = 250,
) -> GarchForecast | None:
    """Fit GARCH(1,1) on daily log returns (in %) and forecast forward volatility."""
    r = log_returns.dropna()
    if len(r) < min_obs:
        return None

    r_pct = r * 100  # arch_model is numerically happier with returns scaled to ~O(1)

    try:
        am = arch_model(r_pct, vol="Garch", p=1, q=1, dist="skewt", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
    except Exception:
        return None

    if not res.convergence_flag == 0:
        converged = False
    else:
        converged = True

    try:
        fcast = res.forecast(horizon=horizon_days, reindex=False)
        variance_path = fcast.variance.values[-1]  # daily variance, in pct^2
    except Exception:
        return None

    daily_vol_forecast = np.sqrt(variance_path) / 100  # back to fractional daily stdev
    weekly_move_pct = float(np.sqrt(np.sum(daily_vol_forecast ** 2)))
    annualized_vol = float(daily_vol_forecast[-1] * np.sqrt(TRADING_DAYS_PER_YEAR))

    params = res.params
    alpha = params.get("alpha[1]", np.nan)
    beta = params.get("beta[1]", np.nan)
    persistence = float(alpha + beta) if np.isfinite(alpha) and np.isfinite(beta) else np.nan

    return GarchForecast(
        symbol="",
        daily_vol_forecast=daily_vol_forecast,
        weekly_move_pct=weekly_move_pct,
        annualized_vol=annualized_vol,
        persistence=persistence,
        converged=converged,
    )
