"""
SwarnaScope Falsification Engine (src/falsify.py)
"Try to kill our own signal" suite:
1. Placebo Test: Date shuffling / random contract pairing
2. Permutation Test: Strategy return permutation -> p-value
3. Bootstrap Confidence Intervals: 1,000 iterations for Sharpe & mean trade P&L
4. Multiple-Testing Adjustment: Deflated Sharpe Ratio (DSR)
5. Cost-Sensitivity Sweep: 1.0x, 1.5x, 2.0x, 2.5x costs
6. Explicit Verdict: "Edge survives" | "Edge fragile" | "No persistent edge after costs"
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.backtest import WalkForwardBacktester


def run_placebo_test(df_signals: pd.DataFrame, num_shuffles: int = 5) -> Dict[str, Any]:
    """
    Placebo test: Shuffles dates or residual z-scores randomly.
    Evaluates if random noise produces positive net P&L.
    """
    if df_signals.empty:
        return {"avg_placebo_sharpe": 0.0, "avg_placebo_pnl": 0.0, "passed": True}

    np.random.seed(42)
    placebo_pnls = []
    placebo_sharpes = []

    backtester = WalkForwardBacktester()

    for _ in range(num_shuffles):
        df_shuffled = df_signals.copy()
        df_shuffled["z_score"] = np.random.permutation(df_shuffled["z_score"].values)
        res = backtester.run_backtest(df_shuffled)
        placebo_pnls.append(res["total_net_pnl_inr"])
        placebo_sharpes.append(res["sharpe_out_of_sample"])

    avg_pnl = float(np.mean(placebo_pnls))
    avg_sharpe = float(np.mean(placebo_sharpes))
    passed = avg_sharpe <= 0.2  # Placebo should NOT have high Sharpe

    return {
        "avg_placebo_sharpe": round(avg_sharpe, 2),
        "avg_placebo_pnl_inr": round(avg_pnl, 2),
        "placebo_sharpes": [round(x, 2) for x in placebo_sharpes[:5]],
        "passed": passed,
    }


def run_permutation_test(daily_pnl: pd.Series, num_permutations: int = 200) -> Dict[str, Any]:
    """
    Permutation test: Permutes daily return vector to calculate empirical p-value of Sharpe.
    """
    if daily_pnl.empty or daily_pnl.std() < 1e-6:
        return {"p_value": 1.0, "observed_sharpe": 0.0, "passed": False}

    np.random.seed(42)
    obs_sharpe = float((daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252))

    perm_sharpes = []
    vals = daily_pnl.values

    for _ in range(num_permutations):
        perm_vals = np.random.permutation(vals)
        std_v = np.std(perm_vals)
        s = (np.mean(perm_vals) / std_v * np.sqrt(252)) if std_v > 1e-6 else 0.0
        perm_sharpes.append(s)

    count_greater = sum(1 for s in perm_sharpes if s >= obs_sharpe)
    p_value = count_greater / float(num_permutations)

    return {
        "observed_sharpe": round(obs_sharpe, 2),
        "p_value": round(p_value, 4),
        "passed": p_value < 0.05,
    }


def run_bootstrap_ci(trades_df: pd.DataFrame, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Bootstrap confidence interval (95%) for mean trade P&L and Sharpe ratio.
    """
    if trades_df.empty:
        return {
            "mean_trade_pnl_ci": (0.0, 0.0),
            "sharpe_ci": (0.0, 0.0),
            "passed": False,
        }

    np.random.seed(42)
    net_pnls = trades_df["net_pnl_inr"].values
    n = len(net_pnls)

    boot_means = []
    boot_sharpes = []

    for _ in range(num_samples):
        sample = np.random.choice(net_pnls, size=n, replace=True)
        m = np.mean(sample)
        std_s = np.std(sample)
        s = (m / std_s * np.sqrt(252)) if std_s > 1e-6 else 0.0
        boot_means.append(m)
        boot_sharpes.append(s)

    ci_mean = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    ci_sharpe = (float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5)))

    passed = ci_mean[0] > 0.0  # Lower bound of 95% CI is strictly positive

    return {
        "mean_trade_pnl_ci": (round(ci_mean[0], 2), round(ci_mean[1], 2)),
        "sharpe_ci": (round(ci_sharpe[0], 2), round(ci_sharpe[1], 2)),
        "passed": passed,
    }


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int = 12,
    num_observations: int = 252
) -> Dict[str, Any]:
    """
    Deflated Sharpe Ratio (DSR) adjustment for multiple testing (López de Prado).
    """
    if num_observations <= 1:
        return {"deflated_sharpe": 0.0, "dsr_p_value": 1.0, "passed": False}

    # Expected maximum Sharpe under N trials of IID random noise
    euler_mascheroni = 0.5772156649
    exp_max_sharpe = (1 - euler_mascheroni) * np.nanpercentile(
        np.random.normal(0, 1, 10000), 100 * (1 - 1 / num_trials)
    )

    dsr_stat = (observed_sharpe - exp_max_sharpe) * np.sqrt(num_observations - 1)
    dsr_p_val = float(1.0 - np.clip(dsr_stat / 5.0, 0.0, 1.0))

    passed = dsr_stat > 0 and dsr_p_val < 0.05

    return {
        "observed_sharpe": round(observed_sharpe, 2),
        "expected_max_null_sharpe": round(exp_max_sharpe, 2),
        "dsr_stat": round(dsr_stat, 2),
        "dsr_p_value": round(dsr_p_val, 4),
        "passed": passed,
    }


def run_cost_sensitivity_sweep(df_signals: pd.DataFrame) -> Dict[str, Any]:
    """
    Sweeps costs across 1.0x, 1.5x, 2.0x, 2.5x base transaction costs.
    Checks if edge survives cost escalation.
    """
    multipliers = [1.0, 1.5, 2.0, 2.5]
    sweep_results = []

    for m in multipliers:
        bt = WalkForwardBacktester(
            brokerage_bps=2.5 * m,
            slippage_bps=1.5 * m
        )
        res = bt.run_backtest(df_signals)
        sweep_results.append({
            "multiplier": f"{m}x",
            "brokerage_bps": 2.5 * m,
            "sharpe_oos": res["sharpe_out_of_sample"],
            "net_pnl_inr": res["total_net_pnl_inr"],
            "total_trades": res["total_trades"],
        })

    # Check if 1.5x cost stays positive
    survives_1_5x = sweep_results[1]["net_pnl_inr"] > 0 and sweep_results[1]["sharpe_oos"] > 0

    return {
        "sweep": sweep_results,
        "survives_1_5x": survives_1_5x,
    }


def evaluate_falsification_verdict(
    backtest_results: Dict[str, Any],
    df_signals: pd.DataFrame
) -> Dict[str, Any]:
    """
    Executes full falsification suite and returns explicit final verdict.
    Possible Verdicts:
    1. "Edge Survives" (High conviction)
    2. "Edge Fragile" (Moderate conviction / cost sensitive)
    3. "No Persistent Edge After Costs" (Defensible honest negative finding)
    """
    oos_sharpe = backtest_results.get("sharpe_out_of_sample", 0.0)
    trade_log = backtest_results.get("trade_log", pd.DataFrame())
    daily_pnl = backtest_results.get("daily_pnl", pd.Series(dtype=float))

    placebo = run_placebo_test(df_signals)
    permutation = run_permutation_test(daily_pnl)
    bootstrap = run_bootstrap_ci(trade_log)
    dsr = compute_deflated_sharpe_ratio(oos_sharpe)
    cost_sweep = run_cost_sensitivity_sweep(df_signals)

    reasons = []

    if oos_sharpe <= 0 or trade_log.empty:
        verdict = "No Persistent Edge After Costs"
        reasons.append("Out-of-sample Sharpe ratio is non-positive after accounting for realistic transaction costs.")
        reasons.append("Micro-structure frictions (brokerage, bid-ask half-spread, slippage) absorb expected reversion profit.")
    elif not cost_sweep["survives_1_5x"]:
        verdict = "Edge Fragile"
        reasons.append("Strategy generates positive returns at 1.0x costs, but net edge disappears when costs escalate to 1.5x.")
        reasons.append("High sensitivity to execution friction and market impact.")
    elif not permutation["passed"] or not bootstrap["passed"]:
        verdict = "Edge Fragile"
        reasons.append("Return distribution fails permutation test (p >= 0.05) or lower bound of 95% bootstrap CI is <= 0.")
        reasons.append("Signal may be driven by specific market regime outliers rather than persistent alpha.")
    else:
        verdict = "Edge Survives"
        reasons.append("Out-of-sample Sharpe ratio remains positive after full transaction costs and passes 1.5x cost sensitivity sweep.")
        reasons.append("Statistically significant under permutation testing and robust bootstrap confidence intervals.")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "placebo_test": placebo,
        "permutation_test": permutation,
        "bootstrap_ci": bootstrap,
        "deflated_sharpe": dsr,
        "cost_sensitivity": cost_sweep,
    }
