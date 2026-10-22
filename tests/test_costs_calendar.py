"""
Unit tests for costs model (src/costs.py) and calendar constraints (src/calendar.py).
"""

from datetime import date
import pandas as pd
import pytest

from src.costs import compute_contract_roundtrip_cost, evaluate_net_edge
from src.calendar import is_date_in_tender_period, is_date_in_liquidity_ramp


def test_zero_volume_non_tradable():
    # Volume = 0 -> non-tradable
    cost, tradable = compute_contract_roundtrip_cost(
        close_norm=14000.0, high_norm=14100.0, low_norm=13900.0, volume=0, open_interest=100
    )
    assert tradable is False
    assert cost > 9000.0


def test_net_edge_calculation():
    net_edge, actionable = evaluate_net_edge(
        abs_residual_inr_g=10.0, total_pair_cost_inr_g=4.0, safety_margin_inr_g=2.0
    )
    assert net_edge == 6.0
    assert actionable is True

    # Not actionable if margin not met
    net_edge_low, actionable_low = evaluate_net_edge(
        abs_residual_inr_g=5.0, total_pair_cost_inr_g=4.0, safety_margin_inr_g=2.0
    )
    assert net_edge_low == 1.0
    assert actionable_low is False


def test_calendar_tender_period():
    expiry = date(2026, 10, 5)
    # 5 days before expiry -> 30 Sep 2026 is inside tender notice period
    assert is_date_in_tender_period(date(2026, 9, 30), expiry) is True
    # 20 Sep 2026 is outside tender notice period
    assert is_date_in_tender_period(date(2026, 9, 20), expiry) is False
