"""
SwarnaScope Normalization Engine (src/normalize.py)
Converts raw settlement prices to a common basis:
  INR per gram of 999-equivalent gold.
Builds pairwise spreads between contract expiries and records expiry gaps.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.config import CONTRACT_SPECS


def normalize_contract_prices(
    df: pd.DataFrame,
    purity_factors: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Convert raw settlement close prices into normalized INR per gram of 999 gold.
    Formula:
      price_per_quote_gram = close / quote_unit_grams
      norm_price = price_per_quote_gram * purity_factor
    """
    if df.empty:
        return df

    df_norm = df.copy()

    if purity_factors is None:
        purity_factors = {
            sym: spec["default_purity_factor"]
            for sym, spec in CONTRACT_SPECS.items()
        }

    # Assign lot size, quote unit, purity factor
    df_norm["lot_size_g"] = df_norm["symbol"].map(
        lambda s: CONTRACT_SPECS.get(s, {}).get("lot_size_grams", 10.0)
    )
    df_norm["quote_unit_g"] = df_norm["symbol"].map(
        lambda s: CONTRACT_SPECS.get(s, {}).get("quote_unit_grams", 10.0)
    )
    df_norm["purity_factor"] = df_norm["symbol"].map(
        lambda s: purity_factors.get(s, CONTRACT_SPECS.get(s, {}).get("default_purity_factor", 1.0))
    )

    # Calculate normalized price (INR per gram of 999 gold)
    df_norm["price_per_quote_gram"] = df_norm["close"] / df_norm["quote_unit_g"]
    df_norm["norm_price_per_gram"] = df_norm["price_per_quote_gram"] * df_norm["purity_factor"]

    # Also compute high and low in normalized terms for half-spread calculations
    df_norm["high_norm"] = (df_norm["high"] / df_norm["quote_unit_g"]) * df_norm["purity_factor"]
    df_norm["low_norm"] = (df_norm["low"] / df_norm["quote_unit_g"]) * df_norm["purity_factor"]

    return df_norm


def build_pairwise_spreads(
    df_norm: pd.DataFrame,
    max_expiry_gap_days: int = 60
) -> pd.DataFrame:
    """
    Build pairwise spread DataFrame for all contract combinations on matching trading dates.
    Pairs contract A and contract B where:
      symbol_A != symbol_B
      |expiry_A - expiry_B| <= max_expiry_gap_days
    
    Fields produced:
      date, symbol_A, expiry_A, norm_price_A, volume_A, oi_A, high_norm_A, low_norm_A
      symbol_B, expiry_B, norm_price_B, volume_B, oi_B, high_norm_B, low_norm_B
      pair_name (e.g. "GOLDM_vs_GOLDGUINEA")
      observed_spread = norm_price_A - norm_price_B
      expiry_gap_days = (expiry_A - expiry_B).days
    """
    if df_norm.empty:
        return pd.DataFrame()

    required_cols = ["date", "symbol", "expiry_date", "norm_price_per_gram", "volume", "open_interest"]
    for c in required_cols:
        if c not in df_norm.columns:
            raise ValueError(f"Missing required column in df_norm: {c}")

    # Self-merge on trading date to build all pairs
    merged = pd.merge(
        df_norm,
        df_norm,
        on="date",
        suffixes=("_A", "_B")
    )

    # Filter out same symbol or duplicate symmetrical pairs (A < B alphabetically)
    merged = merged[merged["symbol_A"] < merged["symbol_B"]].copy()

    # Calculate expiry gap in days
    merged["expiry_gap_days"] = (merged["expiry_date_A"] - merged["expiry_date_B"]).apply(lambda x: x.days)

    # Filter pairs within max_expiry_gap_days
    merged = merged[merged["expiry_gap_days"].abs() <= max_expiry_gap_days].copy()

    if merged.empty:
        return pd.DataFrame()

    # Pair identifier
    merged["pair_name"] = merged["symbol_A"] + "_vs_" + merged["symbol_B"]
    
    # Observed spread in INR per gram of 999 gold
    merged["observed_spread"] = merged["norm_price_per_gram_A"] - merged["norm_price_per_gram_B"]

    # Select and order columns cleanly
    cols = [
        "date",
        "pair_name",
        "symbol_A",
        "expiry_date_A",
        "norm_price_per_gram_A",
        "close_A",
        "high_norm_A",
        "low_norm_A",
        "volume_A",
        "open_interest_A",
        "symbol_B",
        "expiry_date_B",
        "norm_price_per_gram_B",
        "close_B",
        "high_norm_B",
        "low_norm_B",
        "volume_B",
        "open_interest_B",
        "observed_spread",
        "expiry_gap_days",
    ]
    
    available_cols = [c for c in cols if c in merged.columns]
    spreads_df = merged[available_cols].sort_values(["date", "pair_name"]).reset_index(drop=True)
    return spreads_df
