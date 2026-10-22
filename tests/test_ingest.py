"""
Unit tests for data ingestion (src/ingest.py) and data quality (src/quality.py).
"""

from datetime import date
import pandas as pd
import pytest

from src.ingest import parse_date_str, process_bhavcopy_df, generate_synthetic_data
from src.quality import QUALITY_TRACKER


def test_parse_date_str():
    assert parse_date_str("30 Sep 2026") == date(2026, 9, 30)
    assert parse_date_str("30-Sep-2026") == date(2026, 9, 30)
    assert parse_date_str("05OCT2026") == date(2026, 10, 5)
    assert parse_date_str("09/30/2026") == date(2026, 9, 30)
    assert parse_date_str("2026-09-30") == date(2026, 9, 30)
    assert parse_date_str("") is None


def test_process_bhavcopy_date_validation():
    QUALITY_TRACKER.reset()
    raw_data = pd.DataFrame([
        {
            "Instrument Name": "FUTCOM",
            "Symbol": "GOLDM        ",
            "Expiry Date": "05OCT2026",
            "Date": "30 Sep 2026",
            "Close": "146151.00",
            "Volume(Lots)": "12725",
            "Open Interest(Lots)": "7745"
        },
        {
            "Instrument Name": "FUTCOM",
            "Symbol": "GOLDM        ",
            "Expiry Date": "05OCT2026",
            "Date": "29 Sep 2026",  # Mismatched date
            "Close": "146000.00",
            "Volume(Lots)": "1000",
            "Open Interest(Lots)": "5000"
        }
    ])

    # Expecting only 30 Sep 2026 date
    proc = process_bhavcopy_df(raw_data, expected_date=date(2026, 9, 30))
    assert len(proc) == 1
    assert proc.iloc[0]["symbol"] == "GOLDM"
    assert proc.iloc[0]["date"] == date(2026, 9, 30)
    assert QUALITY_TRACKER.date_mismatches_caught == 1


def test_generate_synthetic_data():
    df_synth = generate_synthetic_data(num_days=30)
    assert not df_synth.empty
    assert "symbol" in df_synth.columns
    assert set(["GOLDM", "GOLDGUINEA", "GOLDPETAL"]).issubset(set(df_synth["symbol"].unique()))
    assert (df_synth["close"] > 0).all()
