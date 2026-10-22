"""
SwarnaScope Contract Lifecycle Calendar (src/calendar.py)
Tracks and enforces contract lifecycle constraints:
- Listing Date
- Liquidity Ramp Period (first 5 trading days)
- Tender / Delivery Notice Period (5 trading days prior to expiry)
- Expiry Date
Rule: Never hold open positions into tender delivery periods unless explicitly allowed.
"""

from datetime import date, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

from src.config import CONTRACT_SPECS


def get_contract_lifecycle_dates(
    symbol: str,
    expiry_date: date
) -> Dict[str, Any]:
    """
    Computes contract lifecycle milestones given symbol and expiry date.
    """
    spec = CONTRACT_SPECS.get(symbol, {})
    tender_days = spec.get("tender_days_before_expiry", 5)
    ramp_days = spec.get("liquidity_ramp_days", 5)

    # Tender start date (approx 5 calendar/trading days prior to expiry)
    tender_start_date = expiry_date - timedelta(days=tender_days + 2)

    # Listing date (approx 60-90 days prior to expiry)
    listing_date = expiry_date - timedelta(days=90)
    liquidity_ramp_end = listing_date + timedelta(days=ramp_days + 4)

    return {
        "symbol": symbol,
        "expiry_date": expiry_date,
        "listing_date": listing_date,
        "liquidity_ramp_end_date": liquidity_ramp_end,
        "tender_start_date": tender_start_date,
        "is_tender_active": False,
    }


def is_date_in_tender_period(
    current_date: date,
    expiry_date: date,
    tender_days_before: int = 5
) -> bool:
    """
    Checks if current_date falls inside the delivery tender notice period.
    """
    tender_start = expiry_date - timedelta(days=tender_days_before + 2)
    return current_date >= tender_start and current_date <= expiry_date


def is_date_in_liquidity_ramp(
    current_date: date,
    listing_date: date,
    ramp_days: int = 5
) -> bool:
    """
    Checks if current_date falls in initial liquidity ramp period after listing.
    """
    ramp_end = listing_date + timedelta(days=ramp_days + 4)
    return listing_date <= current_date <= ramp_end


def filter_allowed_trading_window(
    df: pd.DataFrame,
    allow_tender_holding: bool = False
) -> pd.DataFrame:
    """
    Applies calendar constraints to a contract DataFrame.
    """
    if df.empty or allow_tender_holding:
        return df

    df_filt = df.copy()
    
    # Calculate tender start for contract A and contract B if pair dataset
    if "expiry_date_A" in df_filt.columns and "expiry_date_B" in df_filt.columns:
        in_tender_A = df_filt.apply(
            lambda r: is_date_in_tender_period(r["date"], r["expiry_date_A"]), axis=1
        )
        in_tender_B = df_filt.apply(
            lambda r: is_date_in_tender_period(r["date"], r["expiry_date_B"]), axis=1
        )
        df_filt = df_filt[~(in_tender_A | in_tender_B)].copy()
    elif "expiry_date" in df_filt.columns:
        in_tender = df_filt.apply(
            lambda r: is_date_in_tender_period(r["date"], r["expiry_date"]), axis=1
        )
        df_filt = df_filt[~in_tender].copy()

    return df_filt.reset_index(drop=True)
