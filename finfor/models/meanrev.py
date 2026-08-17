"""Directional signal: mean-reversion around a rolling price channel.

GARCH gives a *magnitude* forecast (how much a name is likely to move) but
is direction-agnostic. This module supplies the direction: names that have
stretched away from their own recent average are scored as likely to revert
back toward it. This is deliberately simple and separate from the vol model
so each piece can be evaluated/backtested independently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def direction_signal(feat_row: pd.Series) -> float:
    """Return a signal in [-1, 1]. Positive = expect price to rise (was oversold)."""
    z = feat_row.get("zscore_20d", np.nan)
    rsi = feat_row.get("rsi_14", 50.0)

    if pd.isna(z):
        return 0.0

    z_component = float(np.clip(-z / 2.5, -1, 1))
    rsi_component = float(np.clip((50 - rsi) / 30, -1, 1))
    signal = 0.6 * z_component + 0.4 * rsi_component
    return float(np.clip(signal, -1, 1))


def direction_confidence(feat_row: pd.Series) -> float:
    """0-1: how extreme/confirmed the mean-reversion setup is right now."""
    z = feat_row.get("zscore_20d", np.nan)
    rsi = feat_row.get("rsi_14", 50.0)
    pct_b = feat_row.get("pct_b", 0.5)
    if pd.isna(z):
        return 0.0

    z_extreme = min(abs(z) / 2.0, 1.0)
    rsi_extreme = max(0.0, (abs(rsi - 50) - 15) / 35)
    band_extreme = max(0.0, abs(pct_b - 0.5) - 0.4) / 0.5 if not pd.isna(pct_b) else 0.0
    return float(np.clip(0.5 * z_extreme + 0.3 * rsi_extreme + 0.2 * band_extreme, 0, 1))
