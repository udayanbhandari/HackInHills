"""
SwarnaScope Integer-Lot Hedge Engine (src/hedge.py)
Finds optimal integer whole-lot ratios between MCX gold contracts (100g, 10g, 8g, 1g)
minimizing mismatched gram exposure and margin.
Calculates residual net gram exposure and INR sensitivity per 1% gold price move.
"""

from typing import Dict, Any, Tuple
import math
import numpy as np

from src.config import CONTRACT_SPECS


def optimize_integer_lot_hedge(
    symbol_A: str,
    symbol_B: str,
    target_grams: float = 200.0,
    gold_price_per_g: float = 14000.0,
    max_lots_cap: int = 200
) -> Dict[str, Any]:
    """
    Search for integer lots (n_A, n_B) <= max_lots_cap that best match target_grams
    and minimize net gram mismatch |n_A * L_A - n_B * L_B|.
    
    Returns:
      symbol_A, lot_size_A, lots_A, total_grams_A
      symbol_B, lot_size_B, lots_B, total_grams_B
      matched_grams
      residual_grams (net unhedged grams = total_grams_A - total_grams_B)
      residual_inr_exposure = residual_grams * gold_price_per_g
      inr_per_1pct_gold_move = residual_inr_exposure * 0.01
      hedge_efficiency_pct = (1.0 - |residual_grams| / max(total_grams_A, total_grams_B, 1)) * 100
    """
    spec_A = CONTRACT_SPECS.get(symbol_A, {"lot_size_grams": 10.0})
    spec_B = CONTRACT_SPECS.get(symbol_B, {"lot_size_grams": 10.0})

    lot_A = spec_A["lot_size_grams"]
    lot_B = spec_B["lot_size_grams"]

    best_lots_A = 1
    best_lots_B = 1
    best_mismatch = float("inf")
    best_target_diff = float("inf")

    # Grid search over integer lots 1..max_lots_cap
    for n_A in range(1, max_lots_cap + 1):
        grams_A = n_A * lot_A
        # Ideal n_B to match grams_A
        ideal_n_B = round(grams_A / lot_B)
        if ideal_n_B < 1 or ideal_n_B > max_lots_cap:
            continue

        grams_B = ideal_n_B * lot_B
        mismatch = abs(grams_A - grams_B)
        target_diff = abs(grams_A - target_grams)

        # Prioritize zero gram mismatch, then closest to target grams
        if mismatch < best_mismatch or (mismatch == best_mismatch and target_diff < best_target_diff):
            best_mismatch = mismatch
            best_target_diff = target_diff
            best_lots_A = n_A
            best_lots_B = ideal_n_B

    total_grams_A = best_lots_A * lot_A
    total_grams_B = best_lots_B * lot_B

    matched_grams = min(total_grams_A, total_grams_B)
    residual_grams = total_grams_A - total_grams_B
    residual_inr = residual_grams * gold_price_per_g
    inr_per_1pct = abs(residual_inr) * 0.01

    max_grams = max(total_grams_A, total_grams_B, 1.0)
    hedge_efficiency = max(0.0, (1.0 - abs(residual_grams) / max_grams) * 100.0)

    return {
        "symbol_A": symbol_A,
        "lot_size_A_g": lot_A,
        "lots_A": best_lots_A,
        "total_grams_A": total_grams_A,
        "symbol_B": symbol_B,
        "lot_size_B_g": lot_B,
        "lots_B": best_lots_B,
        "total_grams_B": total_grams_B,
        "matched_grams": matched_grams,
        "residual_grams": residual_grams,
        "residual_inr_exposure": round(residual_inr, 2),
        "inr_per_1pct_gold_move": round(inr_per_1pct, 2),
        "hedge_efficiency_pct": round(hedge_efficiency, 2),
    }
