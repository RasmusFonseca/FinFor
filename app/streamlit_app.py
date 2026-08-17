"""Local review UI: run the pipeline, look at ranked proposals, track positions.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finfor.data_fetch import DATA_DIR, load_history
from finfor.features import build_features
from finfor.pipeline import run_pipeline, PROPOSALS_DIR
from finfor.models.recommend import allocate, BUY_HOLD, BUY_LIMIT
from finfor.positions import (
    load_positions, save_positions, backup_positions, apply_proposals,
    compute_performance, overall_mood,
)

st.set_page_config(page_title="FinFor", layout="wide")


def latest_proposals() -> pd.DataFrame | None:
    if not PROPOSALS_DIR.exists():
        return None
    files = sorted(PROPOSALS_DIR.glob("proposals_*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1]), files[-1].stem.replace("proposals_", "")


def price_chart(symbol: str) -> go.Figure | None:
    hist = load_history(symbol)
    if hist is None:
        return None
    hist = hist.tail(180)
    feat = build_features(symbol)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
        name=symbol,
    ))
    if feat is not None:
        f = feat.tail(180)
        sma20 = f["close"].rolling(20).mean()
        upper = sma20 + 2 * f["close"].rolling(20).std()
        lower = sma20 - 2 * f["close"].rolling(20).std()
        fig.add_trace(go.Scatter(x=f.index, y=upper, line=dict(width=1, color="gray"), name="BB upper"))
        fig.add_trace(go.Scatter(x=f.index, y=lower, line=dict(width=1, color="gray"), name="BB lower", fill="tonexty"))
        fig.add_trace(go.Scatter(x=f.index, y=sma20, line=dict(width=1, color="orange"), name="SMA20"))
    fig.update_layout(height=450, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


st.title("FinFor — reliable-volatility scanner")
st.caption(
    "Speculative, small-stake research tool. Not investment advice. "
    "Proposals are model output based on backtested historical patterns — "
    "no guarantee they hold going forward."
)

with st.sidebar:
    st.header("Refresh")
    top_shortlist = st.slider("Shortlist size (stage A)", 50, 400, 150, step=25)
    top_output = st.slider("Proposals to show", 5, 40, 20, step=5)
    run_btn = st.button("Run pipeline now", type="primary")
    st.caption("Refetches stale price history, refits models, rescans. Takes a few minutes.")

if run_btn:
    with st.spinner("Refreshing data and scoring candidates — this can take several minutes..."):
        progress_box = st.empty()
        result = run_pipeline(top_shortlist=top_shortlist, top_output=top_output, verbose=True)
    st.session_state["last_result"] = result
    st.success(f"Done. {len(result)} candidates passed the filters.")

result = st.session_state.get("last_result")
if result is None:
    cached = latest_proposals()
    if cached is not None:
        result, run_date = cached
        st.info(f"Showing cached proposals from {run_date}. Click 'Run pipeline now' to refresh.")

n_actionable = 0
if result is not None and not result.empty and "action" in result.columns:
    n_actionable = int(result["action"].isin([BUY_HOLD, BUY_LIMIT]).sum())

with st.sidebar:
    st.header("Allocation")
    if n_actionable == 0:
        st.caption("No buy-rated names in the current proposals — nothing to fund this round.")
        n_positions = 1
    else:
        n_positions = st.slider(
            "Positions to fund this round", 1, max(n_actionable, 1), min(3, n_actionable),
            help=f"Only {n_actionable} name(s) in the current proposals are buy-rated, so "
                 f"funding is capped there regardless of how high you set this.",
        )
    capital = st.number_input("Capital to deploy this round ($)", min_value=0.0, value=200.0, step=25.0)
    st.caption(
        "Splits the amount above across the top buy-rated names, weighted by "
        "reliability × confidence, capped at 40% in any single name."
    )

VIEWS = [
    ":material/table_chart: Proposals",
    ":material/search: Ticker detail",
    ":material/account_balance_wallet: My positions",
]
st.session_state.setdefault("active_view", VIEWS[0])


def _jump_to_detail(proposals_df: pd.DataFrame) -> None:
    click = st.session_state.get("detail_jump")
    if not click:
        return
    row = click["row"]
    if 0 <= row < len(proposals_df):
        st.session_state["detail_symbol"] = proposals_df.iloc[row]["symbol"]
        st.session_state["active_view"] = VIEWS[1]


active_view = st.segmented_control("View", VIEWS, key="active_view", label_visibility="collapsed")

if active_view == VIEWS[0]:
    if result is None or result.empty:
        st.write("No proposals yet — run the pipeline from the sidebar.")
    elif "action" not in result.columns:
        st.warning(
            "This cached run predates the recommendation engine. Click 'Run pipeline now' "
            "to regenerate proposals with buy/hold/skip guidance."
        )
        st.dataframe(result, width="stretch", hide_index=True)
    else:
        display = allocate(result, n_positions=n_positions)
        display["allocation_dollars"] = (display["suggested_allocation_pct"] / 100 * capital).round(2)
        display["hold_until"] = pd.to_datetime(display["hold_until"], errors="coerce")
        display["target_sell_price"] = pd.to_numeric(display["target_sell_price"], errors="coerce")
        display["stop_loss_price"] = pd.to_numeric(display["stop_loss_price"], errors="coerce")

        funded = display[display["suggested_allocation_pct"] > 0]
        if not funded.empty:
            st.metric("Positions funded", f"{len(funded)} of {n_positions} requested")
            st.caption(
                "Allocation only applies to buy-rated names — skip/short rows are informational. "
                f"{n_actionable} name(s) are buy-rated in this run, which is the ceiling on the "
                "sidebar slider."
            )
            with st.container(border=True):
                st.write("Already placed these trades in Robinhood?")
                if st.button("Update My positions to this proposal", icon=":material/sync:"):
                    backup_fp = backup_positions()
                    new_positions = apply_proposals(funded)
                    save_positions(new_positions)
                    st.session_state["positions_just_applied"] = True
                    msg = f"My positions now matches this proposal ({len(new_positions)} names)."
                    if backup_fp is not None:
                        msg += f" Previous sheet backed up to {backup_fp.name}."
                    st.success(msg)
                st.caption(
                    "Replaces the entire 'My positions' sheet with the funded names above, sized "
                    "by their allocation at today's price. Your previous sheet is backed up first, "
                    "not deleted."
                )
        else:
            st.info("None of the current proposals clear the bar for an actual buy this round.")

        display["expected_weekly_move_%"] = (display["expected_weekly_move_pct"] * 100).round(1)
        display["direction_hit_rate_%"] = (display["direction_hit_rate"] * 100).round(0)
        display["detail"] = ":material/search:"

        cols = [
            "detail", "symbol", "name", "action", "suggested_allocation_pct", "allocation_dollars",
            "last_close", "target_sell_price", "stop_loss_price", "hold_until",
            "direction", "expected_weekly_move_%", "direction_confidence",
            "reliability_score", "direction_hit_rate_%", "rationale",
        ]
        cols = [c for c in cols if c in display.columns]

        st.dataframe(
            display[cols],
            hide_index=True,
            column_config={
                "detail": st.column_config.ButtonColumn(
                    "", pinned=True, width="small",
                    help="Jump to Ticker detail for this row",
                    on_click=_jump_to_detail, args=(display,), key="detail_jump",
                ),
                "symbol": st.column_config.TextColumn("Symbol", pinned=True),
                "name": st.column_config.TextColumn("Name"),
                "action": st.column_config.TextColumn(
                    "Recommended action",
                    help="Buy & hold: ride it through the week. Buy partial, limit-sell: "
                         "take profit near the target instead of waiting for the full move. "
                         "Skip rows are not funded.",
                ),
                "suggested_allocation_pct": st.column_config.ProgressColumn(
                    "Allocation", min_value=0, max_value=100, format="%.0f%%",
                ),
                "allocation_dollars": st.column_config.NumberColumn("Allocation ($)", format="$%.2f"),
                "last_close": st.column_config.NumberColumn("Last price", format="$%.2f"),
                "target_sell_price": st.column_config.NumberColumn(
                    "Limit sell target", format="$%.2f",
                    help="Suggested limit-sell price: captures ~70% of the model's forecast move.",
                ),
                "stop_loss_price": st.column_config.NumberColumn(
                    "Protective stop", format="$%.2f",
                    help="Suggested stop-loss price if the move goes the wrong way.",
                ),
                "hold_until": st.column_config.DateColumn(
                    "Hold until", format="ddd, MMM D",
                    help="Suggested re-evaluation point — matches the once/twice-weekly refresh cadence.",
                ),
                "direction": st.column_config.TextColumn(
                    "Direction",
                    help="What the model expects the price to do. 'short' setups aren't funded "
                         "since this workflow assumes long-only trades.",
                ),
                "expected_weekly_move_%": st.column_config.NumberColumn(
                    "Expected move (%)", format="%.1f%%",
                    help="GARCH-forecast magnitude of the move over the coming week (either direction).",
                ),
                "direction_confidence": st.column_config.ProgressColumn(
                    "Setup confidence", min_value=0, max_value=1,
                    help="0-1: how extreme/confirmed today's mean-reversion setup is.",
                ),
                "reliability_score": st.column_config.ProgressColumn(
                    "Reliability", min_value=0, max_value=1,
                    help="0-1 backtested score: how well this ticker's past GARCH vol forecasts "
                         "matched reality, blended with the direction signal's historical hit rate.",
                ),
                "direction_hit_rate_%": st.column_config.NumberColumn(
                    "Historical hit rate (%)", format="%.0f%%",
                    help="In backtest, how often the direction signal was right when it fired.",
                ),
                "rationale": st.column_config.TextColumn("Why", width="large"),
            },
        )

elif active_view == VIEWS[1]:
    all_symbols = []
    if result is not None and not result.empty:
        all_symbols = result["symbol"].tolist()
    st.session_state.setdefault("detail_symbol", all_symbols[0] if all_symbols else "")
    sym = st.selectbox(
        "Ticker", options=all_symbols, key="detail_symbol",
        accept_new_options=True, placeholder="Choose from proposals or type any ticker",
    )
    sym = (sym or "").strip().upper()
    if sym:
        fig = price_chart(sym)
        if fig is None:
            st.write(f"No cached history for {sym} yet. Run the pipeline first or refresh history for it.")
        else:
            st.plotly_chart(fig, width="stretch")
        if result is not None and sym in result["symbol"].values:
            detail = result[result["symbol"] == sym].iloc[0].to_dict()
            detail = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in detail.items()}
            st.json(detail)

elif active_view == VIEWS[2]:
    st.write("Track what you actually hold in Robinhood so you can sanity-check proposals against it.")
    positions = load_positions()

    if positions.empty:
        st.caption(
            "Nothing tracked yet. Use 'Update My positions to this proposal' on the Proposals "
            "tab after you've placed trades, or add rows manually below."
        )
    else:
        performance = compute_performance(positions)
        mood, total_dollar_change = overall_mood(performance)

        m1, m2 = st.columns(2)
        m1.metric("Overall mood", mood)
        m2.metric(
            "Total movement since last update",
            f"${total_dollar_change:+.2f}" if total_dollar_change is not None else "—",
        )

        perf_cols = [
            "symbol", "mood", "current_price", "cost_basis", "pct_change",
            "dollar_change", "shares", "days_held", "last_updated",
        ]
        perf_cols = [c for c in perf_cols if c in performance.columns]
        st.dataframe(
            performance[perf_cols],
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", pinned=True),
                "mood": st.column_config.TextColumn("Mood"),
                "current_price": st.column_config.NumberColumn("Current price", format="$%.2f"),
                "cost_basis": st.column_config.NumberColumn("Cost basis", format="$%.2f"),
                "pct_change": st.column_config.NumberColumn(
                    "Change (%)", format="%.1f%%",
                    help="Price movement since cost_basis was recorded (i.e. since last update).",
                ),
                "dollar_change": st.column_config.NumberColumn("Change ($)", format="$%.2f"),
                "shares": st.column_config.NumberColumn("Shares", format="%.4f"),
                "days_held": st.column_config.NumberColumn("Days since update"),
                "last_updated": st.column_config.DateColumn("Last updated", format="MMM D, YYYY"),
            },
        )

    st.divider()
    st.write("Edit manually:")
    edited = st.data_editor(
        positions, num_rows="dynamic",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "shares": st.column_config.NumberColumn("Shares", min_value=0.0),
            "cost_basis": st.column_config.NumberColumn("Cost basis / share", min_value=0.0),
            "last_updated": st.column_config.DateColumn("Last updated"),
        },
    )
    if st.button("Save positions"):
        save_positions(edited)
        st.success("Saved.")
        st.rerun()
