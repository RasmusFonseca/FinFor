"""End-to-end orchestration: refresh data, rank candidates, emit proposals.

This is the thing you actually run once or twice a week. It:
  1. Refreshes cached price history (skips symbols already fresh).
  2. Builds features for the liquid universe.
  3. Cheaply shortlists the ~150 names with the strongest "reliable
     volatility" proxy (finfor/models/score.py).
  4. Runs the expensive walk-forward backtest (finfor/backtest.py) only on
     that shortlist, to get a real reliability score instead of a proxy.
  5. Fits a fresh GARCH forecast + mean-reversion direction signal on each,
     using all available history (not held out, since this is the live
     forecast rather than a backtest step).
  6. Ranks and returns the top candidates as a proposal table.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from finfor.data_fetch import DATA_DIR, refresh_history, refresh_macro
from finfor.features import build_features_for_universe
from finfor.models.score import shortlist_candidates
from finfor.models.garch import fit_garch
from finfor.models.meanrev import direction_signal, direction_confidence
from finfor.backtest import walk_forward_backtest
from finfor.models.recommend import add_recommendations, RELIABILITY_SKIP

PROPOSALS_DIR = DATA_DIR / "proposals"

# Deliberately more permissive than recommend.RELIABILITY_SKIP: a candidate
# between this floor and RELIABILITY_SKIP still makes it into the proposals
# table, just labeled "Skip (signal too weak)" by recommend.py, instead of
# disappearing from the output entirely. Derived from RELIABILITY_SKIP
# (rather than an independent constant) so the two can't silently drift
# apart — raising the bar for "actionable" automatically raises the bar for
# "worth showing at all".
MIN_RELIABILITY_SCORE = RELIABILITY_SKIP / 2
TARGET_WEEKLY_MOVE_MIN = 0.02
TARGET_WEEKLY_MOVE_MAX = 0.10


def load_universe_screened() -> pd.DataFrame:
    fp = DATA_DIR / "universe_screened.csv"
    if not fp.exists():
        from finfor.universe import fetch_common_stock_universe
        from finfor.data_fetch import screen_liquidity

        uni = fetch_common_stock_universe()
        screened = screen_liquidity(uni)
        screened.to_csv(fp, index=False)
        return screened
    return pd.read_csv(fp)


def run_pipeline(
    top_shortlist: int = 150,
    top_output: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    screened = load_universe_screened()
    symbols = screened["symbol"].tolist()
    name_map = dict(zip(screened["symbol"], screened["name"]))

    if verbose:
        print(f"[pipeline] refreshing history for {len(symbols)} symbols")
    ok_symbols = refresh_history(symbols, verbose=verbose)
    refresh_macro(verbose=verbose)

    if verbose:
        print(f"[pipeline] building features for {len(ok_symbols)} symbols")
    feats = build_features_for_universe(ok_symbols, verbose=verbose)

    if verbose:
        print(f"[pipeline] shortlisting from {len(feats)} usable symbols")
    shortlist = shortlist_candidates(feats, top_n=top_shortlist)
    if shortlist.empty:
        return pd.DataFrame()

    rows = []
    for i, sym in enumerate(shortlist["symbol"]):
        feat = feats[sym]
        bt = walk_forward_backtest(feat)
        if bt is None or bt["reliability_score"] < MIN_RELIABILITY_SCORE:
            continue

        gf = fit_garch(feat["log_return"])
        if gf is None:
            continue

        last_row = feat.iloc[-1]
        sig = direction_signal(last_row)
        conf = direction_confidence(last_row)

        if not (TARGET_WEEKLY_MOVE_MIN <= gf.weekly_move_pct <= TARGET_WEEKLY_MOVE_MAX):
            continue

        rows.append({
            "symbol": sym,
            "name": name_map.get(sym, ""),
            "last_close": float(last_row["close"]),
            "expected_weekly_move_pct": gf.weekly_move_pct,
            "annualized_vol": gf.annualized_vol,
            "vol_persistence": gf.persistence,
            "direction": "long" if sig > 0.15 else ("short" if sig < -0.15 else "neutral"),
            "direction_signal": sig,
            "direction_confidence": conf,
            "reliability_score": bt["reliability_score"],
            "vol_forecast_corr": bt["vol_forecast_corr"],
            "vol_forecast_mape": bt["vol_forecast_mape"],
            "direction_hit_rate": bt["direction_hit_rate"],
            "n_backtest_checkpoints": bt["n_checkpoints"],
        })
        if verbose and (i + 1) % 25 == 0:
            print(f"[pipeline] backtested {i + 1}/{len(shortlist)}")

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["rank_score"] = result["reliability_score"] * result["expected_weekly_move_pct"] * (0.5 + 0.5 * result["direction_confidence"])
    result = result.sort_values("rank_score", ascending=False).head(top_output).reset_index(drop=True)
    result = add_recommendations(result)

    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    out_fp = PROPOSALS_DIR / f"proposals_{date.today().isoformat()}.csv"
    result.to_csv(out_fp, index=False)
    if verbose:
        print(f"[pipeline] wrote {len(result)} proposals to {out_fp}")

    return result


if __name__ == "__main__":
    df = run_pipeline()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df)
