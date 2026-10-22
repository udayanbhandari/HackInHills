"""
SwarnaScope Configuration & Contract Specifications
MCX Gold Futures contracts definition and strategy parameters.
"""

from typing import Dict, Any

# MCX Gold Contract Specifications
CONTRACT_SPECS: Dict[str, Dict[str, Any]] = {
    "GOLDM": {
        "symbol": "GOLDM",
        "name": "Gold Mini",
        "lot_size_grams": 100.0,
        "quote_unit_grams": 10.0,
        "standard_purity": 0.995,  # 995 purity
        "default_purity_factor": 999.0 / 995.0,  # Convert 995 to 999 equivalent (~1.00402)
        "expiry_window_start_day": 3,
        "expiry_window_end_day": 5,
        "tender_days_before_expiry": 5,
        "liquidity_ramp_days": 5,
        "active_start_year": 2020,
    },
    "GOLDTEN": {
        "symbol": "GOLDTEN",
        "name": "Gold Ten",
        "lot_size_grams": 10.0,
        "quote_unit_grams": 10.0,
        "standard_purity": 0.999,  # 999 purity
        "default_purity_factor": 1.0,
        "expiry_window_start_day": 27,
        "expiry_window_end_day": 31,
        "tender_days_before_expiry": 5,
        "liquidity_ramp_days": 5,
        "active_start_year": 2025,  # Launched 2025 onwards
    },
    "GOLDGUINEA": {
        "symbol": "GOLDGUINEA",
        "name": "Gold Guinea",
        "lot_size_grams": 8.0,
        "quote_unit_grams": 8.0,
        "standard_purity": 0.999,  # 999 purity
        "default_purity_factor": 1.0,
        "expiry_window_start_day": 27,
        "expiry_window_end_day": 31,
        "tender_days_before_expiry": 5,
        "liquidity_ramp_days": 5,
        "active_start_year": 2020,
    },
    "GOLDPETAL": {
        "symbol": "GOLDPETAL",
        "name": "Gold Petal",
        "lot_size_grams": 1.0,
        "quote_unit_grams": 1.0,
        "standard_purity": 0.999,  # 999 purity
        "default_purity_factor": 1.0,
        "expiry_window_start_day": 27,
        "expiry_window_end_day": 31,
        "tender_days_before_expiry": 5,
        "liquidity_ramp_days": 5,
        "active_start_year": 2020,
    },
    "GOLD": {
        "symbol": "GOLD",
        "name": "Gold Main (1kg)",
        "lot_size_grams": 1000.0,
        "quote_unit_grams": 10.0,
        "standard_purity": 0.995,
        "default_purity_factor": 999.0 / 995.0,
        "expiry_window_start_day": 3,
        "expiry_window_end_day": 5,
        "tender_days_before_expiry": 5,
        "liquidity_ramp_days": 5,
        "active_start_year": 2020,
    }
}

# Default Strategy Parameters
DEFAULT_PARAMS = {
    # Carrying cost parameters
    "risk_free_rate": 0.065,  # 6.5% annual financing rate
    "storage_cost_rate": 0.005,  # 0.5% annual storage cost
    "rolling_window_days": 30,  # Window for regression & Z-score estimation
    
    # Signal thresholds
    "entry_z_threshold": 2.0,  # Z-score entry threshold
    "exit_z_threshold": 0.5,   # Z-score exit (convergence) threshold
    "stop_loss_z": 3.5,        # Z-score stop loss threshold
    "max_holding_days": 15,    # Maximum holding period in trading days
    "net_edge_margin_inr": 2.0,# Minimum required net edge (INR/g) after costs
    
    # Cost Model Parameters
    "brokerage_bps_per_side": 2.5, # 2.5 bps per side = 5 bps roundtrip
    "half_spread_gamma": 0.25,     # Scaling for (High-Low)/Close spread proxy
    "illiquidity_k": 50.0,         # Scaling for low volume/OI penalty
    "slippage_haircut_bps": 1.5,   # Haircut for thin-day liquidity
    
    # Integer Lot Hedging Cap
    "max_lots_cap": 200,
    
    # Backtest Split
    "train_ratio": 0.70,  # 70% in-sample, 30% out-of-sample
}
