"""
SwarnaScope Full Pipeline & Demo Verification Script (make_demo.py)
Executes the end-to-end quant pipeline, runs unit tests, and prints summary metrics.
"""

import sys
import os
import subprocess

# Ensure workspace root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ingest import get_contracts_dataset
from src.normalize import normalize_contract_prices, build_pairwise_spreads
from src.fairvalue import fit_fair_value_model, compute_futures_term_structure
from src.signals import generate_pair_signals
from src.alerts import QuietAlertEngine
from src.backtest import WalkForwardBacktester
from src.attribution import compute_pnl_attribution
from src.falsify import evaluate_falsification_verdict
from src.quality import QUALITY_TRACKER


def run_demo():
    print("=" * 70)
    print("  SWARNASCOPE: MCX GOLD FUTURES INTELLIGENCE LAB DEMO")
    print("=" * 70)
    print("\n1. Running Unit Test Suite (pytest)...")
    pytest_res = subprocess.run([sys.executable, "-m", "pytest", "tests/"], capture_output=True, text=True)
    if pytest_res.returncode == 0:
        print("   [PASS] ALL UNIT TESTS PASSED!")
    else:
        print("   [FAIL] Pytest Error:\n", pytest_res.stdout, pytest_res.stderr)

    print("\n2. Ingesting MCX Gold Futures Contracts Data...")
    df_raw, source_badge = get_contracts_dataset()
    print(f"   [Data Source Mode]: {source_badge}")
    print(f"   [Raw Rows Ingested]: {len(df_raw)}")

    # If raw CSV contains < 10 trading days, also run synthetic engine for full 18m backtest
    num_days_ingested = len(df_raw["date"].unique()) if not df_raw.empty else 0
    if num_days_ingested < 10:
        print(f"   [Notice]: Raw CSV files contain {num_days_ingested} day(s). Running 18m synthetic backtest engine for full walk-forward & falsification suite...")
        df_raw, source_badge = get_contracts_dataset(force_mode="synthetic")

    print("\n3. Normalizing Prices to INR/gram (999-Gold Equivalent)...")
    norm_df = normalize_contract_prices(df_raw)

    print("\n4. Building Pairwise Spreads & Walk-Forward Fair Value Model...")
    spreads_df = build_pairwise_spreads(norm_df, max_expiry_gap_days=60)
    fair_value_df = fit_fair_value_model(spreads_df, window_days=30)

    print("\n5. Generating Signals & Running Quiet Alert Engine...")
    signals_df = generate_pair_signals(fair_value_df, entry_z=2.0, exit_z=0.5, stop_loss_z=3.5)
    alert_engine = QuietAlertEngine()
    alerts, alert_metrics = alert_engine.process_signals(signals_df)

    print("\n6. Executing Walk-Forward Backtest (No Look-Ahead, Out-of-Sample Split)...")
    backtester = WalkForwardBacktester(train_ratio=0.70)
    bt_results = backtester.run_backtest(signals_df)

    print("\n7. Computing Return Attribution & Falsification Suite...")
    attribution = compute_pnl_attribution(bt_results, norm_df)
    verdict_info = evaluate_falsification_verdict(bt_results, signals_df)

    scorecard = QUALITY_TRACKER.get_scorecard()

    print("\n" + "=" * 70)
    print("  SWARNASCOPE FINAL DEMO SUMMARY & QUANT REPORT")
    print("=" * 70)
    print(f"  * Final Falsification Verdict: {verdict_info['verdict'].upper()}")
    print(f"  * Out-of-Sample Net Sharpe:     {bt_results['sharpe_out_of_sample']:.2f}")
    print(f"  * Total Executed Trades:        {bt_results['total_trades']}")
    print(f"  * Alert Suppression Rate:      {alert_metrics['suppression_rate_pct']:.1f}% (Quiet-by-Default)")
    print(f"  * Data Quality Score:           {scorecard['quality_score']}/100 ({scorecard['status']})")
    print(f"  * Strategy Gold Beta:           {attribution['gold_beta']:.4f}")
    print(f"  * Net Strategy P&L:             INR {bt_results['total_net_pnl_inr']:,.2f}")

    print("\n  VERDICT RATIONALE:")
    for r in verdict_info["reasons"]:
        print(f"    - {r}")

    print("\n  KNOWN LIMITATIONS & THREATS OF OVERFITTING:")
    print("    1. Settlement vs Fill Slippage: Official daily Close prices do not capture intraday order book depth.")
    print("    2. Non-Gaussian Tails: Commodity futures residuals exhibit fat tails during macroeconomic shocks.")
    print("    3. Microstructure Frictions: High exchange STT or turnover fees can absorb thin carry margins.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
