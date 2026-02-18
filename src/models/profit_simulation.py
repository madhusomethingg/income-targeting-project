"""
profit_simulation.py
--------------------
Simulates marketing ROI across classification thresholds to identify
the optimal operating point for the XGBoost classifier.

Business logic:
  - For each threshold, we predict who earns >50K and "contact" them.
  - Each contact costs money (contact_cost).
  - Each correctly identified >50K person who converts generates revenue.
  - Profit = (weighted_TP * conversion_rate * revenue) - (weighted_contacts * cost)
  - We sweep thresholds from 0.01 to 0.99 and find the threshold that
    maximises profit on the validation set.
  - Sensitivity analysis sweeps conversion_rate across a range to confirm
    the optimal threshold is robust to that assumption.

Design decisions:
  - Profit computed using WEIGHTED counts (census sample weights).
    This reflects real population impact, not just sample performance.
  - Threshold chosen on VALIDATION set only — test set stays untouched.
  - Optimal row selected via idxmax() not float equality — float comparison
    can silently fail due to rounding in stored threshold values.
  - Full profit curve saved as CSV so reviewers can interrogate any threshold
    without rerunning code.
  - % improvement over threshold=0.5 baseline only printed if baseline profit
    is positive — avoids misleading or undefined comparisons.
  - All business parameters come from config.yaml — nothing hardcoded.
"""

import matplotlib
matplotlib.use("Agg")   # must be set before importing pyplot
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import yaml
import joblib

from pathlib import Path
from typing import Tuple


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_profit_curve_df(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    weights: np.ndarray,
    contact_cost: float,
    revenue_per_positive: float,
    conversion_rate: float,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    Compute profit components at each threshold and return as a DataFrame.

    Columns returned:
      threshold         : classification cutoff
      weighted_tp       : population-weighted true positives
      weighted_contacts : population-weighted contacts (TP + FP)
      contact_rate      : fraction of total population contacted
      revenue           : weighted_tp * conversion_rate * revenue_per_positive
      cost              : weighted_contacts * contact_cost
      profit            : revenue - cost

    All counts are population-weighted (census sample weights applied).
    """
    rows = []
    weighted_population = weights.sum()

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tp_mask      = (y_pred == 1) & (y_true == 1)
        contact_mask = (y_pred == 1)

        weighted_tp       = float(np.sum(weights[tp_mask]))
        weighted_contacts = float(np.sum(weights[contact_mask]))
        contact_rate      = weighted_contacts / weighted_population

        revenue = weighted_tp * conversion_rate * revenue_per_positive
        cost    = weighted_contacts * contact_cost
        profit  = revenue - cost

        rows.append({
            "threshold"        : round(float(t), 4),
            "weighted_tp"      : round(weighted_tp, 1),
            "weighted_contacts": round(weighted_contacts, 1),
            "contact_rate"     : round(contact_rate, 4),
            "revenue"          : round(revenue, 2),
            "cost"             : round(cost, 2),
            "profit"           : round(profit, 2),
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(profit_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Return the threshold and profit at maximum profit.
    Uses idxmax() for index-based selection — avoids float equality issues.
    """
    opt_idx = profit_df["profit"].idxmax()
    opt_row = profit_df.loc[opt_idx]
    return float(opt_row["threshold"]), float(opt_row["profit"])


def plot_profit_curve(
    profit_df: pd.DataFrame,
    optimal_threshold: float,
    optimal_profit: float,
    conversion_rate: float,
    config: dict,
) -> None:
    """Plot profit vs threshold with optimal point marked."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(profit_df["threshold"], profit_df["profit"],
            color="steelblue", linewidth=2.5,
            label="Profit (weighted population)")
    ax.axvline(x=optimal_threshold, color="crimson", linestyle="--",
               linewidth=1.8,
               label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.axhline(y=0, color="grey", linestyle=":", linewidth=1)
    ax.scatter([optimal_threshold], [optimal_profit],
               color="crimson", zorder=5, s=80)
    ax.annotate(
        f"  Max profit\n  threshold={optimal_threshold:.2f}",
        xy=(optimal_threshold, optimal_profit),
        fontsize=9, color="crimson"
    )

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Estimated Profit (weighted population units)", fontsize=12)
    ax.set_title(
        f"Profit Curve — XGBoost  |  conversion_rate={conversion_rate:.0%}",
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out_path = Path(config["paths"]["results_figures"]) / "profit_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[profit] Profit curve saved → {out_path}")


def plot_sensitivity_analysis(
    thresholds: np.ndarray,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    weights: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """
    Sweep conversion_rate across sensitivity_range from config.
    Plots profit curves for each rate and saves a summary CSV.
    Shows whether the optimal threshold is robust to conversion assumptions.

    Returns
    -------
    pd.DataFrame with columns: conversion_rate, optimal_threshold, optimal_profit
    """
    cfg              = config["profit_simulation"]
    contact_cost     = cfg["contact_cost"]
    revenue          = cfg["revenue_per_positive"]
    sens_range       = cfg["sensitivity_range"]
    conversion_rates = np.linspace(sens_range[0], sens_range[1], 6)

    fig, ax = plt.subplots(figsize=(10, 6))
    sensitivity_rows = []

    for cr in conversion_rates:
        df = compute_profit_curve_df(
            y_true, y_proba, weights,
            contact_cost, revenue, cr, thresholds
        )
        opt_t, opt_p = find_optimal_threshold(df)
        sensitivity_rows.append({
            "conversion_rate"   : round(cr, 4),
            "optimal_threshold" : opt_t,
            "optimal_profit"    : round(opt_p, 2),
        })
        ax.plot(df["threshold"], df["profit"], linewidth=1.8,
                label=f"conversion={cr:.0%}  (optimal t={opt_t:.2f})")

    ax.axhline(y=0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Estimated Profit (weighted population units)", fontsize=12)
    ax.set_title(
        "Sensitivity Analysis — Profit Curves Across Conversion Rate Assumptions",
        fontsize=13
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out_path = Path(config["paths"]["results_figures"]) / "profit_sensitivity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[profit] Sensitivity analysis saved → {out_path}")

    # Save sensitivity summary CSV
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sens_csv = Path(config["paths"]["results_tables"]) / "profit_sensitivity.csv"
    sens_csv.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_df.to_csv(sens_csv, index=False)
    print(f"[profit] Sensitivity summary saved → {sens_csv}")

    print(f"\n[profit] Optimal thresholds across conversion rates:")
    for _, row in sensitivity_df.iterrows():
        print(f"         conversion={row['conversion_rate']:.0%}  →  "
              f"optimal threshold={row['optimal_threshold']:.2f}  |  "
              f"profit={row['optimal_profit']:,.0f}")

    return sensitivity_df


def run_profit_simulation(data: dict, model, config: dict) -> float:
    """
    Run the full profit simulation on the validation set.

    Parameters
    ----------
    data   : output from build_features()
    model  : fitted XGBoost model
    config : project config

    Returns
    -------
    optimal_threshold : float
        The threshold that maximises profit on the val set.
        Use this in final evaluation on the test set.
    """
    cfg             = config["profit_simulation"]
    contact_cost    = cfg["contact_cost"]
    revenue         = cfg["revenue_per_positive"]
    conversion_rate = cfg["assumed_conversion_rate"]

    X_val   = data["X_val"]
    y_val   = np.asarray(data["y_val"])
    w_val   = np.asarray(data["w_val"])

    y_proba    = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 200)

    # --- Full profit curve DataFrame ---
    profit_df = compute_profit_curve_df(
        y_val, y_proba, w_val,
        contact_cost, revenue, conversion_rate, thresholds
    )

    optimal_threshold, optimal_profit = find_optimal_threshold(profit_df)

    # --- Baseline at threshold=0.5 ---
    base_df     = compute_profit_curve_df(
        y_val, y_proba, w_val,
        contact_cost, revenue, conversion_rate,
        np.array([0.5])
    )
    base_profit = float(base_df["profit"].iloc[0])
    gain        = optimal_profit - base_profit

    print(f"\n[profit] === Profit Simulation Results (Validation Set) ===")
    print(f"[profit] Contact cost            : {contact_cost}")
    print(f"[profit] Revenue per positive    : {revenue}")
    print(f"[profit] Conversion rate         : {conversion_rate:.0%}")
    print(f"[profit] Optimal threshold       : {optimal_threshold:.4f}")
    print(f"[profit] Max profit (weighted)   : {optimal_profit:,.0f}")
    print(f"[profit] Profit at threshold=0.5 : {base_profit:,.0f}")
    print(f"[profit] Absolute gain           : {gain:,.0f}")

    # % improvement only meaningful if baseline is positive
    if base_profit > 0:
        pct = (optimal_profit / base_profit - 1) * 100
        print(f"[profit] % improvement           : {pct:.1f}%")
    else:
        print(f"[profit] % improvement           : not meaningful "
              f"(baseline profit at 0.5 is {base_profit:,.0f})")

    # --- Stats at optimal threshold (index-safe selection) ---
    opt_idx = profit_df["profit"].idxmax()
    opt_row = profit_df.loc[opt_idx]
    print(f"\n[profit] At optimal threshold={optimal_threshold:.2f}:")
    print(f"         Weighted contacts  : {opt_row['weighted_contacts']:,.0f}")
    print(f"         Contact rate       : {opt_row['contact_rate']:.1%} of population")
    print(f"         Weighted TP        : {opt_row['weighted_tp']:,.0f}")
    print(f"         Revenue            : {opt_row['revenue']:,.0f}")
    print(f"         Cost               : {opt_row['cost']:,.0f}")

    # --- Save full profit curve table ---
    curve_path = Path(config["paths"]["results_tables"]) / "profit_curve_val.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    profit_df.to_csv(curve_path, index=False)
    print(f"\n[profit] Full profit curve saved → {curve_path}")

    # --- Save optimal threshold summary ---
    summary_df = pd.DataFrame([{
        "optimal_threshold" : optimal_threshold,
        "optimal_profit"    : optimal_profit,
        "base_profit_at_05" : base_profit,
        "absolute_gain"     : gain,
        "contact_cost"      : contact_cost,
        "revenue"           : revenue,
        "conversion_rate"   : conversion_rate,
    }])
    summary_path = Path(config["paths"]["results_tables"]) / "optimal_threshold.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[profit] Optimal threshold summary saved → {summary_path}")

    # --- Plots ---
    plot_profit_curve(
        profit_df, optimal_threshold, optimal_profit,
        conversion_rate, config
    )
    plot_sensitivity_analysis(thresholds, y_val, y_proba, w_val, config)

    return optimal_threshold


if __name__ == "__main__":
    # Run from project root:
    # python -m src.models.profit_simulation
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean
    from src.features.engineer import build_features

    config = load_config("config.yaml")
    raw_df = load_raw_data(config)
    X, y, w = clean(raw_df, config)
    data = build_features(X, y, w, config)

    models_dir = Path(config["paths"]["results_models"])
    xgb_model  = joblib.load(models_dir / "xgboost.joblib")

    optimal_threshold = run_profit_simulation(data, xgb_model, config)

    print(f"\n[profit] Use threshold={optimal_threshold:.4f} in final test evaluation.")
    print(f"[profit] Update config.yaml → profit_simulation.premium_filter.threshold")