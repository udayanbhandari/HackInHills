"""
SwarnaScope - Honest Alpha Lab for MCX Gold Futures
Streamlit Web Application (app/main.py)
"""

import sys
import os
from datetime import date, datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingest import get_contracts_dataset, RAW_DIR
from src.normalize import normalize_contract_prices, build_pairwise_spreads
from src.fairvalue import fit_fair_value_model, compute_futures_term_structure
from src.hedge import optimize_integer_lot_hedge
from src.costs import compute_pair_transaction_costs, evaluate_net_edge
from src.signals import generate_pair_signals
from src.alerts import QuietAlertEngine
from src.backtest import WalkForwardBacktester
from src.attribution import compute_pnl_attribution
from src.falsify import evaluate_falsification_verdict
from src.quality import QUALITY_TRACKER
from src.config import CONTRACT_SPECS, DEFAULT_PARAMS


# Page Configuration
st.set_page_config(
    page_title="SwarnaScope - MCX Gold Derivatives Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stBadge { font-size: 1.1rem !important; }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .verdict-box-survives {
        background: rgba(46, 204, 113, 0.15);
        border: 2px solid #2ecc71;
        border-radius: 8px;
        padding: 16px;
        color: #2ecc71;
    }
    .verdict-box-fragile {
        background: rgba(241, 196, 15, 0.15);
        border: 2px solid #f1c40f;
        border-radius: 8px;
        padding: 16px;
        color: #f1c40f;
    }
    .verdict-box-noedge {
        background: rgba(231, 76, 60, 0.15);
        border: 2px solid #e74c3c;
        border-radius: 8px;
        padding: 16px;
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.title("🪙 SwarnaScope Lab")
st.sidebar.markdown("*MCX Gold Derivatives Intelligence*")

# Data Source Switcher
data_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Auto (Raw CSV / Synthetic Fallback)", "Force Synthetic Demo Data"],
    index=0
)
force_synth = True if "Force Synthetic" in data_mode else False

# Load Dataset
raw_df, source_badge = get_contracts_dataset(force_mode="synthetic" if force_synth else None)

# Badge Header
badge_color = "🔴" if "SYNTHETIC" in source_badge else "🟢"
st.sidebar.markdown(f"**Data Status:** {badge_color} `{source_badge}`")

if "SYNTHETIC" in source_badge:
    st.sidebar.warning("⚠️ Using Synthetic Demo Data (~18m daily simulation). Drop real Bhavcopy CSV files into `data/raw/` for real MCX data.")

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")
param_entry_z = st.sidebar.slider("Entry Z-Score Threshold", 1.0, 3.5, DEFAULT_PARAMS["entry_z_threshold"], 0.1)
param_exit_z = st.sidebar.slider("Exit (Convergence) Z-Score", 0.1, 1.5, DEFAULT_PARAMS["exit_z_threshold"], 0.1)
param_stop_z = st.sidebar.slider("Stop-Loss Z-Score", 2.5, 5.0, DEFAULT_PARAMS["stop_loss_z"], 0.1)
param_edge_margin = st.sidebar.slider("Net Edge Margin (₹/g)", 0.5, 10.0, DEFAULT_PARAMS["net_edge_margin_inr"], 0.5)
param_window = st.sidebar.slider("Rolling Estimation Window (days)", 15, 90, DEFAULT_PARAMS["rolling_window_days"], 5)
param_brokerage_bps = st.sidebar.number_input("Brokerage & Exchange Fees (bps per side)", 0.0, 10.0, DEFAULT_PARAMS["brokerage_bps_per_side"])
param_allow_tender = st.sidebar.checkbox("Allow Holding into Tender Period", value=False)


# Cache Data Pipeline
@st.cache_data(show_spinner=False)
def run_full_pipeline(
    df: pd.DataFrame,
    window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    edge_margin: float,
    brokerage_bps: float,
    allow_tender: bool
):
    norm_df = normalize_contract_prices(df)
    spreads_df = build_pairwise_spreads(norm_df, max_expiry_gap_days=60)
    fair_value_df = fit_fair_value_model(spreads_df, window_days=window)
    signals_df = generate_pair_signals(
        fair_value_df,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_loss_z=stop_z,
        net_edge_margin_inr=edge_margin,
        brokerage_bps=brokerage_bps,
        allow_tender_holding=allow_tender
    )
    alert_engine = QuietAlertEngine()
    alerts, alert_metrics = alert_engine.process_signals(signals_df)

    backtester = WalkForwardBacktester(
        entry_z=entry_z,
        exit_z=exit_z,
        stop_loss_z=stop_z,
        net_edge_margin_inr=edge_margin,
        brokerage_bps=brokerage_bps,
        allow_tender_holding=allow_tender
    )
    bt_results = backtester.run_backtest(signals_df)
    attribution = compute_pnl_attribution(bt_results, norm_df)
    verdict_info = evaluate_falsification_verdict(bt_results, signals_df)
    term_struct = compute_futures_term_structure(norm_df)

    return {
        "norm_df": norm_df,
        "spreads_df": spreads_df,
        "fair_value_df": fair_value_df,
        "signals_df": signals_df,
        "alerts": alerts,
        "alert_metrics": alert_metrics,
        "bt_results": bt_results,
        "attribution": attribution,
        "verdict_info": verdict_info,
        "term_struct": term_struct,
    }


pipeline = run_full_pipeline(
    raw_df,
    param_window,
    param_entry_z,
    param_exit_z,
    param_stop_z,
    param_edge_margin,
    param_brokerage_bps,
    param_allow_tender
)


# Multi-Page Navigation Tabs
tab_names = [
    "1. Overview & Verdict",
    "2. Data Trust & Quality",
    "3. Fair-Value Lab",
    "4. Term Structure & Carry",
    "5. Integer Hedge Builder",
    "6. Time-Machine Replay",
    "7. Backtest & Attribution",
    "8. Falsification Suite",
    "9. Contract Calendar & Alerts",
]

selected_tab = st.radio("Navigation", tab_names, horizontal=True)
st.markdown("---")

# ----------------------------------------------------
# TAB 1: OVERVIEW & VERDICT
# ----------------------------------------------------
if selected_tab == "1. Overview & Verdict":
    st.header("🪙 SwarnaScope Overview")
    st.markdown("##### *MCX Gold Derivatives Relative Value & Carry Intelligence Lab*")

    v_data = pipeline["verdict_info"]
    verdict = v_data["verdict"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Source Mode", source_badge)
    with col2:
        st.metric("Out-of-Sample Net Sharpe", f"{pipeline['bt_results']['sharpe_out_of_sample']:.2f}")
    with col3:
        st.metric("Alert Suppression Rate", f"{pipeline['alert_metrics']['suppression_rate_pct']:.1f}%")
    with col4:
        st.metric("Total Executed Trades", f"{pipeline['bt_results']['total_trades']}")

    st.markdown("### ⚖️ Final Falsification Verdict")

    if verdict == "Edge Survives":
        box_class = "verdict-box-survives"
        badge_symbol = "✅"
    elif verdict == "Edge Fragile":
        box_class = "verdict-box-fragile"
        badge_symbol = "⚠️"
    else:
        box_class = "verdict-box-noedge"
        badge_symbol = "🛑"

    st.markdown(f"""
    <div class="{box_class}">
        <h2>{badge_symbol} Verdict: {verdict.upper()}</h2>
        <p><b>Executive Summary:</b></p>
        <ul>
            {"".join(f"<li>{r}</li>" for r in v_data['reasons'])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Key Strategy Performance Metrics")

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Total Net P&L", f"₹{pipeline['bt_results']['total_net_pnl_inr']:,.2f}")
    m_col2.metric("Hit Rate", f"{pipeline['bt_results']['hit_rate_pct']:.1f}%")
    m_col3.metric("Max Drawdown", f"{pipeline['bt_results']['max_drawdown_pct']:.1f}%")
    m_col4.metric("Avg Holding Days", f"{pipeline['bt_results']['avg_holding_days']} d")
    m_col5.metric("Strategy Gold Beta", f"{pipeline['attribution']['gold_beta']:.3f}")


# ----------------------------------------------------
# TAB 2: DATA TRUST & QUALITY SCORECARD
# ----------------------------------------------------
elif selected_tab == "2. Data Trust & Quality":
    st.header("🛡️ Data-Trust Scorecard & Quality Audit")
    st.markdown("Strict validation of MCX Bhavcopy dates, zero-volume settlements, and contract history.")

    scorecard = QUALITY_TRACKER.get_scorecard()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Quality Score", f"{scorecard['quality_score']} / 100", delta=scorecard['status'])
    c2.metric("Raw Rows Loaded", scorecard['total_raw_rows'])
    c3.metric("Rows Used (%)", f"{scorecard['rows_used_pct']}%")
    c4.metric("Date Mismatches Caught", scorecard['date_mismatches_caught'])

    q_col1, q_col2, q_col3 = st.columns(3)
    q_col1.metric("Duplicates Removed", scorecard['duplicate_rows_removed'])
    q_col2.metric("Zero-Volume Settlements", scorecard['zero_volume_settlements'])
    q_col3.metric("Stale Price Runs (3+ days)", scorecard['stale_price_events'])

    if scorecard["goldten_notes"]:
        st.info("ℹ️ " + " ".join(scorecard["goldten_notes"]))

    st.markdown("### Raw Contracts Preview")
    st.dataframe(pipeline["norm_df"].tail(20), use_container_width=True)


# ----------------------------------------------------
# TAB 3: FAIR-VALUE LAB
# ----------------------------------------------------
elif selected_tab == "3. Fair-Value Lab":
    st.header("📐 Carry-Adjusted Fair Value Engine")
    st.markdown("Decomposes observed spreads into carrying cost, purity differential, and residual mispricing Z-score.")

    fair_df = pipeline["fair_value_df"]
    if fair_df.empty:
        st.warning("No pairwise contract spreads available.")
    else:
        pairs = sorted(fair_df["pair_name"].unique())
        selected_pair = st.selectbox("Select Futures Pair", pairs)

        pair_data = fair_df[fair_df["pair_name"] == selected_pair].sort_values("date")

        st.subheader(f"Observed vs Expected Fair Spread ({selected_pair})")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=pair_data["date"], y=pair_data["observed_spread"], name="Observed Spread (₹/g)", line=dict(color="#3498db")))
        fig1.add_trace(go.Scatter(x=pair_data["date"], y=pair_data["expected_spread"], name="Carry Fair Value (₹/g)", line=dict(color="#f39c12", dash="dash")))
        fig1.update_layout(template="plotly_dark", height=400, yaxis_title="Spread (INR / gram)")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(f"Walk-Forward Residual Z-Score & Signals ({selected_pair})")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pair_data["date"], y=pair_data["z_score"], name="Residual Z-Score", line=dict(color="#2ecc71")))
        fig2.add_hline(y=param_entry_z, line_dash="dash", line_color="red", annotation_text=f"Entry +{param_entry_z}")
        fig2.add_hline(y=-param_entry_z, line_dash="dash", line_color="green", annotation_text=f"Entry -{param_entry_z}")
        fig2.add_hline(y=param_exit_z, line_dash="dot", line_color="gray")
        fig2.add_hline(y=-param_exit_z, line_dash="dot", line_color="gray")
        fig2.update_layout(template="plotly_dark", height=400, yaxis_title="Z-Score")
        st.plotly_chart(fig2, use_container_width=True)


# ----------------------------------------------------
# TAB 4: TERM STRUCTURE & CARRY
# ----------------------------------------------------
elif selected_tab == "4. Term Structure & Carry":
    st.header("📈 MCX Gold Futures Term Structure")
    st.markdown("Visualizes futures curve across expiries and separates roll-down from genuine curve shifts.")

    ts_df = pipeline["term_struct"]
    if ts_df.empty:
        st.warning("Term structure data unavailable.")
    else:
        latest_date = ts_df["date"].max()
        ts_latest = ts_df[ts_df["date"] == latest_date].sort_values("days_to_expiry")

        fig_ts = px.line(
            ts_latest,
            x="days_to_expiry",
            y="norm_price_per_gram",
            color="symbol",
            markers=True,
            title=f"Futures Price Curve per Contract Family on {latest_date}",
            labels={"days_to_expiry": "Days to Expiry", "norm_price_per_gram": "INR / Gram (999 Equivalent)"}
        )
        fig_ts.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_ts, use_container_width=True)


# ----------------------------------------------------
# TAB 5: INTEGER HEDGE BUILDER
# ----------------------------------------------------
elif selected_tab == "5. Integer Hedge Builder":
    st.header("⚖️ Integer-Lot Hedge Calculator")
    st.markdown("Calculates optimal whole-lot hedge ratios between 100g, 10g, 8g, and 1g contracts.")

    col1, col2, col3 = st.columns(3)
    sym_a = col1.selectbox("Leg A Contract", ["GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"], index=0)
    sym_b = col2.selectbox("Leg B Contract", ["GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"], index=2)
    target_g = col3.number_input("Target Gold Quantity (Grams)", min_value=1.0, value=200.0, step=10.0)

    hedge_res = optimize_integer_lot_hedge(sym_a, sym_b, target_g)

    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric(f"Lots {sym_a} ({hedge_res['lot_size_A_g']}g)", f"{hedge_res['lots_A']} lots ({hedge_res['total_grams_A']}g)")
    hc2.metric(f"Lots {sym_b} ({hedge_res['lot_size_B_g']}g)", f"{hedge_res['lots_B']} lots ({hedge_res['total_grams_B']}g)")
    hc3.metric("Matched Grams", f"{hedge_res['matched_grams']} g")
    hc4.metric("Hedge Efficiency", f"{hedge_res['hedge_efficiency_pct']}%")

    st.markdown("### Residual Unhedged Exposure & Sensitivity")
    rc1, rc2 = st.columns(2)
    rc1.metric("Residual Net Gram Mismatch", f"{hedge_res['residual_grams']} g")
    rc2.metric("INR Exposure per 1% Gold Move", f"₹{hedge_res['inr_per_1pct_gold_move']:,.2f}")


# ----------------------------------------------------
# TAB 6: TIME-MACHINE REPLAY
# ----------------------------------------------------
elif selected_tab == "6. Time-Machine Replay":
    st.header("⏳ Time-Machine Replay (Strict No Look-Ahead Proof)")
    st.markdown("Slide to any historical date to inspect **ONLY** information knowable on or before that day.")

    signals_df = pipeline["signals_df"]
    if signals_df.empty:
        st.warning("No signal data for replay.")
    else:
        avail_dates = sorted(signals_df["date"].unique())
        selected_replay_date = st.select_slider("Select Historical Trading Date (t)", options=avail_dates, value=avail_dates[-1])

        st.markdown(f"#### Information State on {selected_replay_date}")
        as_of_df = signals_df[signals_df["date"] == selected_replay_date]

        st.dataframe(
            as_of_df[[
                "pair_name", "observed_spread", "expected_spread", "residual",
                "z_score", "pair_cost_inr_g", "net_edge_inr_g", "raw_signal", "is_tender_blocked"
            ]],
            use_container_width=True
        )


# ----------------------------------------------------
# TAB 7: BACKTEST & ATTRIBUTION
# ----------------------------------------------------
elif selected_tab == "7. Backtest & Attribution":
    st.header("📊 Walk-Forward Backtest & Return Attribution")
    st.markdown("Evaluates out-of-sample trading performance and decomposes net P&L into structural factors.")

    bt = pipeline["bt_results"]
    attr = pipeline["attribution"]

    st.subheader("Walk-Forward Cumulative Equity Curve (INR)")
    if not bt["equity_curve"].empty:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=bt["equity_curve"].index, y=bt["equity_curve"].values, name="Strategy Net Equity", line=dict(color="#2ecc71", width=2)))
        fig_eq.update_layout(template="plotly_dark", height=400, yaxis_title="Cumulative Net P&L (INR)")
        st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("P&L Factor Waterfall Attribution")
    attr_df = pd.DataFrame([
        {"Factor": "Spread Convergence P&L", "P&L (INR)": attr["spread_pnl_inr"]},
        {"Factor": "Residual Gold Beta P&L", "P&L (INR)": attr["residual_beta_pnl_inr"]},
        {"Factor": "Carry / Roll P&L", "P&L (INR)": attr["carry_pnl_inr"]},
        {"Factor": "Transaction Costs", "P&L (INR)": -attr["costs_inr"]},
        {"Factor": "Total Net P&L", "P&L (INR)": attr["total_net_pnl_inr"]},
    ])

    fig_water = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=attr_df["Factor"],
        y=attr_df["P&L (INR)"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig_water.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_water, use_container_width=True)

    st.subheader("Executed Trade Log")
    st.dataframe(bt["trade_log"], use_container_width=True)


# ----------------------------------------------------
# TAB 8: FALSIFICATION SUITE
# ----------------------------------------------------
elif selected_tab == "8. Falsification Suite":
    st.header("🧪 Falsification Suite ('Try to Kill Our Own Signal')")
    st.markdown("Automated stress tests to evaluate if edge survives costs and statistical scrutiny.")

    v = pipeline["verdict_info"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Permutation Test p-value", f"{v['permutation_test']['p_value']:.4f}", delta="PASS" if v['permutation_test']['passed'] else "FAIL")
    col2.metric("Placebo Avg Sharpe", f"{v['placebo_test']['avg_placebo_sharpe']:.2f}", delta="PASS" if v['placebo_test']['passed'] else "FAIL")
    col3.metric("Survives 1.5x Costs?", "YES" if v['cost_sensitivity']['survives_1_5x'] else "NO")

    st.subheader("Cost-Sensitivity Sweep")
    sweep_df = pd.DataFrame(v["cost_sensitivity"]["sweep"])
    st.table(sweep_df)


# ----------------------------------------------------
# TAB 9: CONTRACT CALENDAR & ALERTS
# ----------------------------------------------------
elif selected_tab == "9. Contract Calendar & Alerts":
    st.header("🔔 Quiet Alert Feed & Contract Calendar")
    st.markdown("Displays quiet-by-default actionable alert logs and tender delivery windows.")

    m = pipeline["alert_metrics"]
    st.metric("Alert Suppression Rate (% Days Silent)", f"{m['suppression_rate_pct']:.1f}%", delta="QUIET BY DEFAULT" if m['is_quiet'] else "ACTIVE")

    st.subheader("Actionable Alerts Feed")
    if not pipeline["alerts"]:
        st.info("System is QUIET: No actionable signals met entry threshold & net edge margin today.")
    else:
        for a in pipeline["alerts"][:15]:
            st.success(a["message"])
