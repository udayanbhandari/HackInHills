"""
Unit tests for price normalization and pairwise spread building (src/normalize.py).
"""

from datetime import date
import pandas as pd
import pytest

from src.normalize import normalize_contract_prices, build_pairwise_spreads


def test_normalize_contract_prices():
    raw_df = pd.DataFrame([
        {
            "date": date(2026, 9, 30),
            "symbol": "GOLDM",
            "expiry_date": date(2026, 10, 5),
            "close": 146151.0,  # Quoted per 10g 995 gold
            "high": 148170.0,
            "low": 146070.0,
            "volume": 12725,
            "open_interest": 7745
        },
        {
            "date": date(2026, 9, 30),
            "symbol": "GOLDTEN",
            "expiry_date": date(2026, 10, 30),
            "close": 148322.0,  # Quoted per 10g 999 gold
            "high": 150185.0,
            "low": 148182.0,
            "volume": 30456,
            "open_interest": 31599
        }
    ])

    norm_df = normalize_contract_prices(raw_df)
    
    # GOLDM: (146151 / 10) * (999/995) = 14615.1 * 1.0040201 = 14673.755 INR/g
    goldm_row = norm_df[norm_df["symbol"] == "GOLDM"].iloc[0]
    expected_goldm_norm = (146151.0 / 10.0) * (999.0 / 995.0)
    assert abs(goldm_row["norm_price_per_gram"] - expected_goldm_norm) < 1e-4

    # GOLDTEN: (148322 / 10) * 1.0 = 14832.2 INR/g
    goldten_row = norm_df[norm_df["symbol"] == "GOLDTEN"].iloc[0]
    assert abs(goldten_row["norm_price_per_gram"] - 14832.2) < 1e-4


def test_build_pairwise_spreads():
    raw_df = pd.DataFrame([
        {
            "date": date(2026, 9, 30),
            "symbol": "GOLDM",
            "expiry_date": date(2026, 10, 5),
            "close": 146151.0,
            "high": 148170.0,
            "low": 146070.0,
            "volume": 12725,
            "open_interest": 7745
        },
        {
            "date": date(2026, 9, 30),
            "symbol": "GOLDTEN",
            "expiry_date": date(2026, 10, 30),
            "close": 148322.0,
            "high": 150185.0,
            "low": 148182.0,
            "volume": 30456,
            "open_interest": 31599
        }
    ])

    norm_df = normalize_contract_prices(raw_df)
    spreads = build_pairwise_spreads(norm_df, max_expiry_gap_days=60)

    assert not spreads.empty
    assert "pair_name" in spreads.columns
    assert "observed_spread" in spreads.columns
    assert "expiry_gap_days" in spreads.columns
    # Expiry gap: (10/05/2026 - 10/30/2026) = -25 days
    assert spreads.iloc[0]["expiry_gap_days"] == -25
