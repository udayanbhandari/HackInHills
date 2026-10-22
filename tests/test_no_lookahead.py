"""
Unit test verifying STRICT NO LOOK-AHEAD in fair-value fitting and backtest.
"""

from datetime import date
import pandas as pd
import numpy as np
import pytest

from src.fairvalue import fit_fair_value_model
from src.backtest import WalkForwardBacktester


def test_fairvalue_no_lookahead():
    """
    Modifying future data at index i must NOT change the Z-score or fitted parameters at index i-1 or earlier.
    """
    dates = pd.bdate_range("2026-01-01", periods=20).date
    obs_spreads = np.linspace(10, 50, 20)
    
    df1 = pd.DataFrame({
        "date": dates,
        "pair_name": "GOLDM_vs_GOLDTEN",
        "norm_price_per_gram_A": 14000.0,
        "norm_price_per_gram_B": 13990.0,
        "observed_spread": obs_spreads,
        "expiry_gap_days": 10,
    })

    fit1 = fit_fair_value_model(df1, window_days=10)

    # Modify future data at index 15-19
    df2 = df1.copy()
    df2.loc[15:19, "observed_spread"] = 9999.0

    fit2 = fit_fair_value_model(df2, window_days=10)

    # Z-scores and expected spreads for indices 0..14 MUST BE EXACTLY IDENTICAL
    np.testing.assert_array_almost_equal(
        fit1.loc[0:14, "z_score"].values,
        fit2.loc[0:14, "z_score"].values
    )
    np.testing.assert_array_almost_equal(
        fit1.loc[0:14, "expected_spread"].values,
        fit2.loc[0:14, "expected_spread"].values
    )
