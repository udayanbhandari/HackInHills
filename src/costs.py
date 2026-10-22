"""
SwarnaScope Transaction Cost & Execution Engine (src/costs.py)
Estimates realistic per-contract round-trip transaction costs in INR per gram:
- Exchange / Brokerage fees (bps)
- Corwin-Schultz / High-Low range half-spread proxy scaled by illiquidity (Volume & OI)
- Slippage haircut for thin trading days
- Identifies NON-TRADABLE zero-volume days.
Calculates Net Edge = Expected Reversion - Est Round-Trip Cost.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import DEFAULT_PARAMS


def compute_contract_roundtrip_cost(
    close_norm: float,
    high_norm: float,
    low_norm: float,
    volume: float,
    open_interest: float,
    brokerage_bps: float = 2.5,
    half_spread_gamma: float = 0.25,
    illiquidity_k: float = 50.0,
    slippage_bps: float = 1.5
) -> Tuple[float, bool]:
    """
    Computes single-contract round-trip transaction cost per gram (INR/g).
    Returns (cost_inr_per_gram, is_tradable)
    """
    # Non-tradable rule: zero volume or zero OI
    if volume <= 0 or open_interest <= 0 or close_norm <= 0:
        return 9999.0, False

    # 1. Exchange & Brokerage fee (roundtrip = 2 * bps)
    brokerage_cost = close_norm * (2.0 * brokerage_bps / 10000.0)

    # 2. High-Low Half-Spread Proxy
    if high_norm > low_norm and close_norm > 0:
        hl_pct = (high_norm - low_norm) / close_norm
    else:
        hl_pct = 0.002  # default 20 bps range fallback

    # Illiquidity scaling factor
    liquidity_score = np.sqrt(volume * open_interest + 1.0)
    illiquidity_multiplier = 1.0 + (illiquidity_k / liquidity_score)
    half_spread_cost = close_norm * (hl_pct * half_spread_gamma * illiquidity_multiplier)

    # 3. Slippage haircut
    slippage_cost = close_norm * (slippage_bps / 10000.0)

    total_cost_per_g = brokerage_cost + half_spread_cost + slippage_cost
    return float(total_cost_per_g), True


def compute_pair_transaction_costs(
    row: pd.Series,
    brokerage_bps: float = 2.5,
    half_spread_gamma: float = 0.25,
    illiquidity_k: float = 50.0,
    slippage_bps: float = 1.5
) -> Dict[str, Any]:
    """
    Computes total pair round-trip cost (INR/gram of 999 gold) for contract A and contract B.
    """
    cost_A, tradable_A = compute_contract_roundtrip_cost(
        row.get("norm_price_per_gram_A", 14000.0),
        row.get("high_norm_A", 14100.0),
        row.get("low_norm_A", 13900.0),
        row.get("volume_A", 1000.0),
        row.get("open_interest_A", 5000.0),
        brokerage_bps, half_spread_gamma, illiquidity_k, slippage_bps
    )

    cost_B, tradable_B = compute_contract_roundtrip_cost(
        row.get("norm_price_per_gram_B", 14000.0),
        row.get("high_norm_B", 14100.0),
        row.get("low_norm_B", 13900.0),
        row.get("volume_B", 1000.0),
        row.get("open_interest_B", 5000.0),
        brokerage_bps, half_spread_gamma, illiquidity_k, slippage_bps
    )

    is_tradable = tradable_A and tradable_B
    total_pair_cost = (cost_A + cost_B) if is_tradable else 9999.0

    return {
        "cost_A_inr_g": round(cost_A, 2),
        "cost_B_inr_g": round(cost_B, 2),
        "total_pair_cost_inr_g": round(total_pair_cost, 2),
        "is_tradable": is_tradable
    }


def evaluate_net_edge(
    abs_residual_inr_g: float,
    total_pair_cost_inr_g: float,
    safety_margin_inr_g: float = 2.0
) -> Tuple[float, bool]:
    """
    Calculates Net Edge = Expected Reversion (INR/g) - Est Round-Trip Cost (INR/g).
    Returns (net_edge, is_actionable)
    """
    net_edge = abs_residual_inr_g - total_pair_cost_inr_g
    is_actionable = net_edge >= safety_margin_inr_g
    return float(net_edge), bool(is_actionable)
