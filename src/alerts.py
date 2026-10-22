"""
SwarnaScope Alert Engine (src/alerts.py)
Quiet-by-default alert generator for actionable trading opportunities.
Calculates and reports the Alert Suppression Rate metric (% of silent trading days).
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


class QuietAlertEngine:
    def __init__(self):
        self.alerts_history: List[Dict[str, Any]] = []

    def process_signals(self, df_signals: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filters signals and produces deduplicated actionable alerts.
        """
        if df_signals.empty:
            return [], {"total_days": 0, "alert_days": 0, "suppression_rate_pct": 100.0}

        df_sorted = df_signals.sort_values(["date", "pair_name"]).reset_index(drop=True)
        unique_dates = df_sorted["date"].unique()
        total_trading_days = len(unique_dates)

        alerts = []
        alert_dates = set()
        last_pair_signals: Dict[str, int] = {}

        for _, row in df_sorted.iterrows():
            sig = row.get("raw_signal", 0)
            pair = row.get("pair_name", "UNKNOWN")
            dt = row.get("date")

            if sig != 0:
                # Deduplicate consecutive identical alerts for the same pair
                if last_pair_signals.get(pair) == sig:
                    continue

                last_pair_signals[pair] = sig
                alert_dates.add(dt)

                direction = "LONG SPREAD (Buy A, Sell B)" if sig == 1 else "SHORT SPREAD (Sell A, Buy B)"
                res_inr = row.get("residual", 0.0)
                z_sc = row.get("z_score", 0.0)
                net_edge = row.get("net_edge_inr_g", 0.0)
                cost = row.get("pair_cost_inr_g", 0.0)

                alert_item = {
                    "date": dt,
                    "pair_name": pair,
                    "direction": direction,
                    "signal_type": sig,
                    "z_score": round(z_sc, 2),
                    "residual_inr_g": round(res_inr, 2),
                    "net_edge_inr_g": round(net_edge, 2),
                    "pair_cost_inr_g": round(cost, 2),
                    "message": f"[{dt}] ACTIONABLE ALERT: {pair} -> {direction} | Z={z_sc:.2f} | Net Edge=₹{net_edge:.2f}/g"
                }
                alerts.append(alert_item)
            else:
                last_pair_signals[pair] = 0

        self.alerts_history = alerts
        alert_days_count = len(alert_dates)
        
        suppression_rate = (
            (1.0 - alert_days_count / max(total_trading_days, 1)) * 100.0
            if total_trading_days > 0 else 100.0
        )

        metrics = {
            "total_trading_days": total_trading_days,
            "alert_days_count": alert_days_count,
            "total_alerts_emitted": len(alerts),
            "suppression_rate_pct": round(suppression_rate, 2),
            "is_quiet": suppression_rate >= 80.0
        }

        return alerts, metrics
