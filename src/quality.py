"""
SwarnaScope Data Quality & Data-Trust Layer (src/quality.py)
Provides a visible scorecard for data integrity checks:
- Date-mismatch fallbacks caught
- Duplicate rows removed
- Zero-volume settlements
- Stale prices (unchanged close for N consecutive days)
- Outlier price prints
- Missing expiries
- GOLDTEN short-history warnings
- Clean ratio of rows used vs discarded.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class DataQualityTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_raw_rows = 0
        self.total_valid_rows = 0
        self.date_mismatches_caught = 0
        self.duplicate_rows_removed = 0
        self.zero_volume_settlements = 0
        self.stale_price_events = 0
        self.outlier_prints = 0
        self.missing_expiry_warnings = 0
        self.goldten_history_notes = []
        self.quality_log = []

    def log_issue(self, category: str, details: str):
        self.quality_log.append({"category": category, "details": details})

    def run_dataframe_audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audit a normalized contracts DataFrame for data quality issues.
        """
        if df.empty:
            return self.get_scorecard()

        self.total_raw_rows = len(df)
        
        # Check duplicates
        dup_cols = [c for c in ['symbol', 'expiry_date', 'date'] if c in df.columns]
        if dup_cols:
            dups = df.duplicated(subset=dup_cols).sum()
            self.duplicate_rows_removed += int(dups)

        # Zero volume settlements
        if 'volume' in df.columns:
            zero_vol = (df['volume'] == 0).sum()
            self.zero_volume_settlements = int(zero_vol)

        # Stale price check (unchanged close for 3+ consecutive days per contract)
        if 'close' in df.columns and 'symbol' in df.columns and 'expiry_date' in df.columns:
            stale_count = 0
            grouped = df.groupby(['symbol', 'expiry_date'])
            for _, group in grouped:
                group_sorted = group.sort_values('date')
                close_diff = group_sorted['close'].diff()
                # Count runs of 0 diff
                zero_diffs = (close_diff == 0).astype(int)
                # 3 consecutive zeros
                stale_runs = (zero_diffs.rolling(3).sum() == 3).sum()
                stale_count += int(stale_runs)
            self.stale_price_events = stale_count

        # Outlier prints (|daily pct change| > 5%)
        if 'close' in df.columns and 'symbol' in df.columns and 'expiry_date' in df.columns:
            outliers = 0
            grouped = df.groupby(['symbol', 'expiry_date'])
            for _, group in grouped:
                group_sorted = group.sort_values('date')
                pct_change = group_sorted['close'].pct_change().abs()
                outliers += (pct_change > 0.05).sum()
            self.outlier_prints = int(outliers)

        # GOLDTEN check
        if 'symbol' in df.columns and 'GOLDTEN' in df['symbol'].values:
            dates = df[df['symbol'] == 'GOLDTEN']['date']
            if not dates.empty:
                min_date = pd.to_datetime(dates).min()
                if min_date >= pd.to_datetime('2025-01-01'):
                    self.goldten_history_notes.append(
                        f"GOLDTEN history starts on {min_date.strftime('%Y-%m-%d')} (Expected 2025+ launch)."
                    )

        self.total_valid_rows = len(df) - self.duplicate_rows_removed
        return self.get_scorecard()

    def get_scorecard(self) -> Dict[str, Any]:
        used_pct = (
            (self.total_valid_rows / max(self.total_raw_rows, 1)) * 100.0
            if self.total_raw_rows > 0
            else 100.0
        )
        
        # Calculate quality score (0 to 100)
        deductions = (
            (self.date_mismatches_caught * 2) +
            (self.duplicate_rows_removed * 1) +
            (self.outlier_prints * 1.5) +
            (self.stale_price_events * 0.5)
        )
        score = max(0.0, min(100.0, 100.0 - deductions))

        return {
            "total_raw_rows": self.total_raw_rows,
            "total_valid_rows": self.total_valid_rows,
            "rows_used_pct": round(used_pct, 2),
            "date_mismatches_caught": self.date_mismatches_caught,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "zero_volume_settlements": self.zero_volume_settlements,
            "stale_price_events": self.stale_price_events,
            "outlier_prints": self.outlier_prints,
            "missing_expiry_warnings": self.missing_expiry_warnings,
            "goldten_notes": self.goldten_history_notes,
            "quality_score": round(score, 1),
            "status": "PASS" if score >= 80 else ("WARN" if score >= 60 else "FAIL")
        }


# Global singleton tracker for app run
QUALITY_TRACKER = DataQualityTracker()
