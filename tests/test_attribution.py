"""
Unit tests for strategy attribution (src/attribution.py).
Verifies exact P&L sum identity: Total Net P&L == Spread P&L + Beta P&L + Carry P&L - Costs.
"""

import pandas as pd
import pytest
from src.attribution import compute_pnl_attribution


def test_attribution_identity_check():
    mock_trade_log = pd.DataFrame([
        {
            "net_pnl_inr": 1500.0,
            "gross_pnl_inr": 2000.0,
            "costs_inr": 500.0,
            "residual_grams": 0.0,
        },
        {
            "net_pnl_inr": -300.0,
            "gross_pnl_inr": 200.0,
            "costs_inr": 500.0,
            "residual_grams": 2.0,
        }
    ])

    mock_backtest = {
        "trade_log": mock_trade_log,
        "daily_pnl": pd.Series([100.0, -50.0, 200.0, 950.0]),
    }

    attr = compute_pnl_attribution(mock_backtest, pd.DataFrame())

    assert bool(attr["identity_check_passed"]) is True
    reconstructed = (
        attr["spread_pnl_inr"] + attr["residual_beta_pnl_inr"] + attr["carry_pnl_inr"] - attr["costs_inr"]
    )
    assert abs(reconstructed - attr["total_net_pnl_inr"]) < 1e-2
