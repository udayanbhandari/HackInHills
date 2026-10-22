"""
SwarnaScope Walk-Forward Backtester (src/backtest.py)
Replays gold futures pairs day by day with strict no look-ahead:
- Parameters and fair-value fits at day t use only data <= t-1.
- Splitting in-sample (train) vs out-of-sample (test) periods.
- Trades contracts actually held at their daily settlement prices with integer lots.
- Applies costs on every entry and exit.
- Respects non-tradable days and tender period delivery notice blocks.
"""

from typing import Dict, Any, List, Tuple
from datetime import date
import pandas as pd
import numpy as np

from src.hedge import optimize_integer_lot_hedge
from src.costs import compute_pair_transaction_costs
from src.calendar import is_date_in_tender_period
from src.config import DEFAULT_PARAMS


class WalkForwardBacktester:
    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_loss_z: float = 3.5,
        max_holding_days: int = 15,
        net_edge_margin_inr: float = 2.0,
        brokerage_bps: float = 2.5,
        half_spread_gamma: float = 0.25,
        illiquidity_k: float = 50.0,
        slippage_bps: float = 1.5,
        target_grams: float = 200.0,
        allow_tender_holding: bool = False,
        train_ratio: float = 0.70
    ):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        self.max_holding_days = max_holding_days
        self.net_edge_margin_inr = net_edge_margin_inr
        self.brokerage_bps = brokerage_bps
        self.half_spread_gamma = half_spread_gamma
        self.illiquidity_k = illiquidity_k
        self.slippage_bps = slippage_bps
        self.target_grams = target_grams
        self.allow_tender_holding = allow_tender_holding
        self.train_ratio = train_ratio

    def run_backtest(self, df_signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs walk-forward backtest on df_signals.
        df_signals must already have fair-value fit with no look-ahead.
        """
        if df_signals.empty:
            return self._empty_results()

        df_sorted = df_signals.sort_values(["date", "pair_name"]).reset_index(drop=True)
        unique_dates = np.array(sorted(df_sorted["date"].unique()))
        total_days = len(unique_dates)

        if total_days < 10:
            return self._empty_results()

        # Split into In-Sample (IS) and Out-of-Sample (OOS)
        split_idx = int(total_days * self.train_ratio)
        is_dates = set(unique_dates[:split_idx])
        oos_dates = set(unique_dates[split_idx:])

        # Active positions per pair: pair_name -> position_dict
        active_positions: Dict[str, Dict[str, Any]] = {}
        trade_log: List[Dict[str, Any]] = []

        # Daily P&L tracking: date -> pnl
        daily_pnl_map: Dict[date, float] = {d: 0.0 for d in unique_dates}
        daily_pnl_is: Dict[date, float] = {d: 0.0 for d in unique_dates if d in is_dates}
        daily_pnl_oos: Dict[date, float] = {d: 0.0 for d in unique_dates if d in oos_dates}

        # Daily loop
        grouped_by_date = df_sorted.groupby("date")

        for cur_date in unique_dates:
            if cur_date not in grouped_by_date.groups:
                continue

            day_rows = grouped_by_date.get_group(cur_date)

            for _, row in day_rows.iterrows():
                pair = row["pair_name"]
                z = row.get("z_score", 0.0)
                residual = row.get("residual", 0.0)
                norm_A = row.get("norm_price_per_gram_A", 14000.0)
                norm_B = row.get("norm_price_per_gram_B", 14000.0)
                exp_A = row.get("expiry_date_A")
                exp_B = row.get("expiry_date_B")

                # Check costs & tradability today
                cost_info = compute_pair_transaction_costs(
                    row, self.brokerage_bps, self.half_spread_gamma, self.illiquidity_k, self.slippage_bps
                )
                cost_inr_g = cost_info["total_pair_cost_inr_g"]
                is_tradable = cost_info["is_tradable"]

                # Check tender period
                in_tender = False
                if not self.allow_tender_holding:
                    in_tender = is_date_in_tender_period(cur_date, exp_A) or is_date_in_tender_period(cur_date, exp_B)

                # 1. Update active position if exists
                if pair in active_positions:
                    pos = active_positions[pair]
                    pos["holding_days"] += 1

                    # Compute today's price change P&L
                    prev_A = pos["last_norm_A"]
                    prev_B = pos["last_norm_B"]
                    
                    # Position direction: +1 = Long A / Short B; -1 = Short A / Long B
                    dir_mult = pos["direction"]
                    matched_g = pos["hedge_info"]["matched_grams"]
                    
                    # Daily spread change P&L in INR
                    delta_A = (norm_A - prev_A)
                    delta_B = (norm_B - prev_B)
                    daily_spread_pnl = dir_mult * (delta_A - delta_B) * matched_g

                    # Residual gold beta P&L
                    res_g = pos["hedge_info"]["residual_grams"]
                    avg_gold_change = (delta_A + delta_B) / 2.0
                    daily_beta_pnl = dir_mult * res_g * avg_gold_change

                    day_total_pnl = daily_spread_pnl + daily_beta_pnl
                    pos["cumulative_pnl"] += day_total_pnl
                    pos["last_norm_A"] = norm_A
                    pos["last_norm_B"] = norm_B

                    daily_pnl_map[cur_date] += day_total_pnl
                    if cur_date in is_dates:
                        daily_pnl_is[cur_date] += day_total_pnl
                    else:
                        daily_pnl_oos[cur_date] += day_total_pnl

                    # Check EXIT conditions
                    exit_reason = None
                    if is_tradable:
                        if in_tender:
                            exit_reason = "TENDER_NOTICE_DEADLINE"
                        elif abs(z) < self.exit_z:
                            exit_reason = "CONVERGENCE"
                        elif abs(z) > self.stop_loss_z and (z * pos["direction"]) > 0:
                            exit_reason = "STOP_LOSS"
                        elif pos["holding_days"] >= self.max_holding_days:
                            exit_reason = "MAX_HOLDING_DAYS"

                    if exit_reason:
                        # Exit transaction cost
                        exit_cost_inr = cost_inr_g * matched_g
                        net_trade_pnl = pos["cumulative_pnl"] - pos["entry_cost_inr"] - exit_cost_inr

                        daily_pnl_map[cur_date] -= exit_cost_inr
                        if cur_date in is_dates:
                            daily_pnl_is[cur_date] -= exit_cost_inr
                        else:
                            daily_pnl_oos[cur_date] -= exit_cost_inr

                        trade_log.append({
                            "pair_name": pair,
                            "entry_date": pos["entry_date"],
                            "exit_date": cur_date,
                            "direction": "LONG_SPREAD" if pos["direction"] == 1 else "SHORT_SPREAD",
                            "holding_days": pos["holding_days"],
                            "entry_z": pos["entry_z"],
                            "exit_z": z,
                            "gross_pnl_inr": round(pos["cumulative_pnl"], 2),
                            "costs_inr": round(pos["entry_cost_inr"] + exit_cost_inr, 2),
                            "net_pnl_inr": round(net_trade_pnl, 2),
                            "exit_reason": exit_reason,
                            "period": "IN_SAMPLE" if pos["entry_date"] in is_dates else "OUT_OF_SAMPLE",
                            "matched_grams": matched_g,
                            "residual_grams": pos["hedge_info"]["residual_grams"],
                        })

                        del active_positions[pair]

                # 2. Check ENTRY conditions if no active position
                elif is_tradable and not in_tender:
                    act_edge = row.get("net_edge_inr_g", 0.0) >= self.net_edge_margin_inr
                    entry_dir = 0
                    if z <= -self.entry_z and act_edge:
                        entry_dir = 1   # Long A, Short B
                    elif z >= self.entry_z and act_edge:
                        entry_dir = -1  # Short A, Long B

                    if entry_dir != 0:
                        hedge_info = optimize_integer_lot_hedge(
                            row["symbol_A"], row["symbol_B"], self.target_grams, norm_A
                        )
                        matched_g = hedge_info["matched_grams"]
                        entry_cost_inr = cost_inr_g * matched_g

                        # Deduct entry cost from daily P&L
                        daily_pnl_map[cur_date] -= entry_cost_inr
                        if cur_date in is_dates:
                            daily_pnl_is[cur_date] -= entry_cost_inr
                        else:
                            daily_pnl_oos[cur_date] -= entry_cost_inr

                        active_positions[pair] = {
                            "pair_name": pair,
                            "entry_date": cur_date,
                            "direction": entry_dir,
                            "entry_z": z,
                            "holding_days": 0,
                            "last_norm_A": norm_A,
                            "last_norm_B": norm_B,
                            "hedge_info": hedge_info,
                            "entry_cost_inr": entry_cost_inr,
                            "cumulative_pnl": 0.0,
                        }

        # Build equity curves and summary statistics
        return self._compile_results(
            unique_dates, daily_pnl_map, daily_pnl_is, daily_pnl_oos, trade_log, is_dates, oos_dates
        )

    def _compile_results(
        self,
        dates: np.ndarray,
        pnl_map: Dict[date, float],
        pnl_is: Dict[date, float],
        pnl_oos: Dict[date, float],
        trade_log: List[Dict[str, Any]],
        is_dates: set,
        oos_dates: set
    ) -> Dict[str, Any]:
        pnl_series = pd.Series([pnl_map[d] for d in dates], index=pd.to_datetime(dates))
        equity_curve = pnl_series.cumsum()

        pnl_is_s = pd.Series([pnl_is.get(d, 0.0) for d in dates if d in is_dates])
        pnl_oos_s = pd.Series([pnl_oos.get(d, 0.0) for d in dates if d in oos_dates])

        trades_df = pd.DataFrame(trade_log)

        total_trades = len(trades_df)
        hit_rate = (trades_df["net_pnl_inr"] > 0).mean() * 100.0 if total_trades > 0 else 0.0
        avg_holding = trades_df["holding_days"].mean() if total_trades > 0 else 0.0
        total_pnl = equity_curve.iloc[-1] if not equity_curve.empty else 0.0

        # Sharpe calculations (annualized, 252 trading days)
        sharpe_all = self._calc_sharpe(pnl_series)
        sharpe_is = self._calc_sharpe(pnl_is_s)
        sharpe_oos = self._calc_sharpe(pnl_oos_s)
        sortino_oos = self._calc_sortino(pnl_oos_s)

        # Max Drawdown
        max_dd_inr, max_dd_pct = self._calc_max_drawdown(equity_curve)

        return {
            "total_trades": total_trades,
            "hit_rate_pct": round(hit_rate, 2),
            "avg_holding_days": round(avg_holding, 1),
            "total_net_pnl_inr": round(total_pnl, 2),
            "sharpe_overall": round(sharpe_all, 2),
            "sharpe_in_sample": round(sharpe_is, 2),
            "sharpe_out_of_sample": round(sharpe_oos, 2),
            "sortino_out_of_sample": round(sortino_oos, 2),
            "max_drawdown_inr": round(max_dd_inr, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "equity_curve": equity_curve,
            "daily_pnl": pnl_series,
            "trade_log": trades_df,
            "num_is_days": len(is_dates),
            "num_oos_days": len(oos_dates),
        }

    def _calc_sharpe(self, s: pd.Series) -> float:
        if s.empty or s.std() < 1e-6:
            return 0.0
        return float((s.mean() / s.std()) * np.sqrt(252))

    def _calc_sortino(self, s: pd.Series) -> float:
        if s.empty:
            return 0.0
        downside = s[s < 0]
        if downside.empty or downside.std() < 1e-6:
            return 0.0
        return float((s.mean() / downside.std()) * np.sqrt(252))

    def _calc_max_drawdown(self, eq: pd.Series) -> Tuple[float, float]:
        if eq.empty:
            return 0.0, 0.0
        peak = eq.cummax()
        dd = eq - peak
        max_dd = float(dd.min())
        peak_val = float(peak.max())
        max_dd_pct = (abs(max_dd) / max(peak_val, 1e4)) * 100.0 if peak_val > 0 else 0.0
        return abs(max_dd), max_dd_pct

    def _empty_results(self) -> Dict[str, Any]:
        return {
            "total_trades": 0,
            "hit_rate_pct": 0.0,
            "avg_holding_days": 0.0,
            "total_net_pnl_inr": 0.0,
            "sharpe_overall": 0.0,
            "sharpe_in_sample": 0.0,
            "sharpe_out_of_sample": 0.0,
            "sortino_out_of_sample": 0.0,
            "max_drawdown_inr": 0.0,
            "max_drawdown_pct": 0.0,
            "equity_curve": pd.Series(dtype=float),
            "daily_pnl": pd.Series(dtype=float),
            "trade_log": pd.DataFrame(),
            "num_is_days": 0,
            "num_oos_days": 0,
        }
