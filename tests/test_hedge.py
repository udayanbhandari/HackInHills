"""
Unit tests for integer-lot hedge optimization (src/hedge.py).
"""

import pytest
from src.hedge import optimize_integer_lot_hedge


def test_integer_lot_hedge_goldm_vs_goldguinea():
    """
    Test prompt example: GOLDM (100g) vs GOLDGUINEA (8g) for 200g target.
    Expected: 2 lots GOLDM (200g) vs 25 lots GOLDGUINEA (200g), residual = 0g.
    """
    res = optimize_integer_lot_hedge(
        symbol_A="GOLDM",
        symbol_B="GOLDGUINEA",
        target_grams=200.0,
        gold_price_per_g=14000.0
    )

    assert res["lots_A"] == 2
    assert res["lots_B"] == 25
    assert res["total_grams_A"] == 200.0
    assert res["total_grams_B"] == 200.0
    assert res["matched_grams"] == 200.0
    assert res["residual_grams"] == 0.0
    assert res["residual_inr_exposure"] == 0.0
    assert res["hedge_efficiency_pct"] == 100.0


def test_integer_lot_hedge_goldm_vs_goldten():
    """
    GOLDM (100g) vs GOLDTEN (10g) for 100g target.
    Expected: 1 lot GOLDM (100g) vs 10 lots GOLDTEN (100g).
    """
    res = optimize_integer_lot_hedge(
        symbol_A="GOLDM",
        symbol_B="GOLDTEN",
        target_grams=100.0,
        gold_price_per_g=14000.0
    )

    assert res["lots_A"] == 1
    assert res["lots_B"] == 10
    assert res["matched_grams"] == 100.0
    assert res["residual_grams"] == 0.0
