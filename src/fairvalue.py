"""
SwarnaScope Carry-Adjusted Fair Value Engine (src/fairvalue.py)
Decomposes observed spread into carry component, purity differential, and residual.
Fit regressions and rolling statistics using strictly NO LOOK-AHEAD (data <= t-1).
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import DEFAULT_PARAMS


def compute_theoretical_carry(
    spot_price_per_g: float,
    expiry_gap_days: float,
    risk_free_rate: float = 0.065,
    storage_rate: float = 0.005
) -> float:
    """
    Theoretical carry cost (INR per gram) for carrying gold across an expiry gap (delta T days).
    Formula: spot * (exp((r + storage) * (delta_T / 365)) - 1)
    """
    total_rate = risk_free_rate + storage_rate
    time_years = expiry_gap_days / 365.0
    carry_effect = spot_price_per_g * (np.exp(total_rate * time_years) - 1.0)
    return float(carry_effect)


def fit_fair_value_model(
    pair_spreads_df: pd.DataFrame,
    window_days: int = 30,
    risk_free_rate: float = 0.065,
    use_robust_mad: bool = False
) -> pd.DataFrame:
    """
    Strictly Walk-Forward Fair Value Model.
    For each pair and each day t, fits expected spread using ONLY past data [t-window_days, t-1].
    
    Model:
      expected_spread_t = alpha_{t-1} + beta_{t-1} * theoretical_carry_t
      residual_t = observed_spread_t - expected_spread_t
      z_score_t = (residual_t - mean(residual_{t-window:t-1})) / std(residual_{t-window:t-1})
    """
    if pair_spreads_df.empty:
        return pd.DataFrame()

    df_res = pair_spreads_df.copy().sort_values(["pair_name", "date"]).reset_index(drop=True)
    
    df_res["theoretical_carry"] = 0.0
    df_res["expected_spread"] = 0.0
    df_res["residual"] = 0.0
    df_res["z_score"] = 0.0
    df_res["rolling_mean"] = 0.0
    df_res["rolling_std"] = 1.0

    grouped = df_res.groupby("pair_name")
    processed_dfs = []

    for pair_name, group in grouped:
        g = group.copy().sort_values("date").reset_index(drop=True)
        n = len(g)

        # Compute theoretical carry for every row
        spot_proxy = (g["norm_price_per_gram_A"] + g["norm_price_per_gram_B"]) / 2.0
        g["theoretical_carry"] = [
            compute_theoretical_carry(spot_proxy.iloc[i], g["expiry_gap_days"].iloc[i], risk_free_rate)
            for i in range(n)
        ]

        expected_spreads = np.zeros(n)
        residuals = np.zeros(n)
        z_scores = np.zeros(n)
        rolling_means = np.zeros(n)
        rolling_stds = np.ones(n)

        # Walk-forward loop over day t
        for i in range(n):
            if i < 5:  # Initial warm-up: zero z-score
                expected_spreads[i] = g["theoretical_carry"].iloc[i]
                residuals[i] = g["observed_spread"].iloc[i] - expected_spreads[i]
                z_scores[i] = 0.0
                continue

            # Prior window data up to i-1 (STRICT NO LOOK-AHEAD)
            win_start = max(0, i - window_days)
            hist_obs = g["observed_spread"].iloc[win_start:i].values
            hist_carry = g["theoretical_carry"].iloc[win_start:i].values
            hist_gaps = g["expiry_gap_days"].iloc[win_start:i].values

            # Fit OLS: observed_spread = alpha + beta * theoretical_carry (or expiry_gap)
            if np.std(hist_carry) > 1e-6:
                slope, intercept = np.polyfit(hist_carry, hist_obs, 1)
            elif np.std(hist_gaps) > 1e-6:
                slope, intercept = np.polyfit(hist_gaps, hist_obs, 1)
                hist_carry = hist_gaps
            else:
                slope = 1.0
                intercept = np.mean(hist_obs) - np.mean(hist_carry)

            # Predict fair spread for day t using t-1 parameters
            today_carry = g["theoretical_carry"].iloc[i]
            expected_spreads[i] = intercept + slope * today_carry
            residuals[i] = g["observed_spread"].iloc[i] - expected_spreads[i]

            # Fit rolling mean and std on historical residuals up to i-1
            hist_residuals = g["observed_spread"].iloc[win_start:i].values - (intercept + slope * hist_carry)
            
            if use_robust_mad:
                med = float(np.median(hist_residuals))
                mad = float(np.median(np.abs(hist_residuals - med)))
                scale = max(1.4826 * mad, 1e-4)
                rolling_means[i] = med
                rolling_stds[i] = scale
                z_scores[i] = (residuals[i] - med) / scale
            else:
                mu = float(np.mean(hist_residuals))
                sigma = float(np.std(hist_residuals))
                sigma = max(sigma, 1e-4)
                rolling_means[i] = mu
                rolling_stds[i] = sigma
                z_scores[i] = (residuals[i] - mu) / sigma

        g["expected_spread"] = expected_spreads
        g["residual"] = residuals
        g["z_score"] = np.clip(z_scores, -10.0, 10.0)
        g["rolling_mean"] = rolling_means
        g["rolling_std"] = rolling_stds

        processed_dfs.append(g)

    if not processed_dfs:
        return pd.DataFrame()

    out_df = pd.concat(processed_dfs, ignore_index=True)
    return out_df.sort_values(["date", "pair_name"]).reset_index(drop=True)


def compute_futures_term_structure(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Computes futures term structure (curve per contract family across expiries).
    Distinguishes mechanical roll-down toward expiry from genuine curve changes.
    """
    if df_norm.empty:
        return pd.DataFrame()

    df_ts = df_norm.copy()
    df_ts["days_to_expiry"] = (df_ts["expiry_date"] - df_ts["date"]).apply(lambda x: x.days)
    df_ts["implied_annual_carry_pct"] = (
        (df_ts["norm_price_per_gram"] / df_ts.groupby("symbol")["norm_price_per_gram"].transform("min") - 1.0)
        / (df_ts["days_to_expiry"] / 365.0).replace(0, np.nan)
    ) * 100.0

    return df_ts.sort_values(["symbol", "date", "days_to_expiry"]).reset_index(drop=True)
