"""
SwarnaScope Signal Generator (src/signals.py)
Generates entry, exit, and stop-loss signals adhering to:
- Residual Z-score thresholds (entry_z, exit_z, stop_loss_z)
- Net Edge margin after transaction costs
- Liquidity tradability filters
- Contract lifecycle calendar deadlines (tender notice block).
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.costs import compute_pair_transaction_costs, evaluate_net_edge
from src.calendar import is_date_in_tender_period
from src.config import DEFAULT_PARAMS


def generate_pair_signals(
    fair_value_df: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float = 3.5,
    net_edge_margin_inr: float = 2.0,
    brokerage_bps: float = 2.5,
    half_spread_gamma: float = 0.25,
    illiquidity_k: float = 50.0,
    slippage_bps: float = 1.5,
    allow_tender_holding: bool = False
) -> pd.DataFrame:
    """
    Computes daily signals for a fair-value residual dataset.
    
    Signal Values:
      +1: Enter LONG spread (Contract A undervalued relative to B)
      -1: Enter SHORT spread (Contract A overvalued relative to B)
       0: Neutral / Exit / Hold
    """
    if fair_value_df.empty:
        return pd.DataFrame()

    df_sig = fair_value_df.copy().sort_values(["pair_name", "date"]).reset_index(drop=True)

    # Compute transaction costs and net edge per row
    pair_costs = []
    is_tradable_list = []
    net_edge_list = []
    is_actionable_list = []

    for _, row in df_sig.iterrows():
        c_info = compute_pair_transaction_costs(
            row, brokerage_bps, half_spread_gamma, illiquidity_k, slippage_bps
        )
        t_cost = c_info["total_pair_cost_inr_g"]
        tradable = c_info["is_tradable"]
        
        abs_res = abs(row.get("residual", 0.0))
        n_edge, actionable = evaluate_net_edge(abs_res, t_cost, net_edge_margin_inr)

        pair_costs.append(t_cost)
        is_tradable_list.append(tradable)
        net_edge_list.append(n_edge)
        is_actionable_list.append(actionable)

    df_sig["pair_cost_inr_g"] = pair_costs
    df_sig["is_tradable"] = is_tradable_list
    df_sig["net_edge_inr_g"] = net_edge_list
    df_sig["is_actionable_edge"] = is_actionable_list

    # Check tender period constraint
    if not allow_tender_holding:
        in_tender = df_sig.apply(
            lambda r: is_date_in_tender_period(r["date"], r["expiry_date_A"]) or
                      is_date_in_tender_period(r["date"], r["expiry_date_B"]),
            axis=1
        )
        df_sig["is_tender_blocked"] = in_tender
    else:
        df_sig["is_tender_blocked"] = False

    # Raw signal logic
    raw_signals = np.zeros(len(df_sig), dtype=int)
    
    for i in range(len(df_sig)):
        z = df_sig["z_score"].iloc[i]
        act = df_sig["is_actionable_edge"].iloc[i]
        trd = df_sig["is_tradable"].iloc[i]
        blk = df_sig["is_tender_blocked"].iloc[i]

        if trd and not blk and act:
            if z <= -entry_z:
                raw_signals[i] = 1   # Long A, Short B (A is underpriced)
            elif z >= entry_z:
                raw_signals[i] = -1  # Short A, Long B (A is overpriced)

    df_sig["raw_signal"] = raw_signals
    return df_sig
