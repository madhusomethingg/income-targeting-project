"""
metrics.py
----------
Full evaluation suite for trained classifiers.

Computes (all weighted by census sample weights where supported):
  - ROC-AUC and PR-AUC    : rank metrics, threshold-independent
  - Precision, Recall, F1 : threshold-dependent metrics
  - Brier Score            : calibration quality
  - Confusion matrix       : population-weighted, at chosen threshold
  - Classification report  : unweighted, for interpretability only

Design decisions:
  - All metrics weighted by census sample weights where sklearn supports it.
    Raw unweighted counts misrepresent the true population distribution.
  - Weighted PR baseline = np.average(y, weights=w), not y.mean().
    Stratified sampling means unweighted prevalence != population prevalence.
  - X is passed as DataFrame (not numpy) to predict_proba to preserve feature
    names and avoid sklearn warnings. y and w are converted to numpy for
    metric functions to avoid index alignment issues after splitting.
  - Confusion matrix values are population-weighted counts, not raw counts.
  - Default threshold is 0.5 but should ultimately be set by profit simulation.
  - Results saved to results/tables/ as CSV and figures to results/figures/.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import Dict, Any

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(
    name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate a single fitted model on a given split.

    Parameters
    ----------
    name      : model identifier for logging
    model     : fitted sklearn/XGBoost model
    X         : feature matrix (DataFrame — preserves feature names)
    y         : true binary labels
    w         : sample weights
    threshold : classification threshold (default 0.5)

    Returns
    -------
    dict of metric_name → value
    """
    # Pass X as DataFrame to preserve feature names (avoids sklearn warnings).
    # Convert y and w to numpy to avoid index alignment issues.
    y_np     = np.asarray(y)
    w_np     = np.asarray(w)
    proba_np = model.predict_proba(X)[:, 1]
    y_pred   = (proba_np >= threshold).astype(int)

    roc_auc   = roc_auc_score(y_np, proba_np, sample_weight=w_np)
    pr_auc    = average_precision_score(y_np, proba_np, sample_weight=w_np)
    brier     = brier_score_loss(y_np, proba_np, sample_weight=w_np)
    precision = precision_score(y_np, y_pred, sample_weight=w_np, zero_division=0)
    recall    = recall_score(y_np, y_pred, sample_weight=w_np, zero_division=0)
    f1        = f1_score(y_np, y_pred, sample_weight=w_np, zero_division=0)

    results = {
        "model"     : name,
        "threshold" : threshold,
        "roc_auc"   : round(roc_auc, 4),
        "pr_auc"    : round(pr_auc, 4),
        "brier"     : round(brier, 4),
        "precision" : round(precision, 4),
        "recall"    : round(recall, 4),
        "f1"        : round(f1, 4),
    }

    print(f"\n[metrics] {name} (threshold={threshold})")
    print(f"          ROC-AUC   : {roc_auc:.4f}")
    print(f"          PR-AUC    : {pr_auc:.4f}")
    print(f"          Brier     : {brier:.4f}")
    print(f"          Precision : {precision:.4f}")
    print(f"          Recall    : {recall:.4f}")
    print(f"          F1        : {f1:.4f}")

    return results


def plot_roc_curves(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    config: dict,
) -> None:
    """Plot weighted ROC curves for all models on the same axes."""
    y_np = np.asarray(y)
    w_np = np.asarray(w)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        proba_np = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_np, proba_np, sample_weight=w_np)
        auc = roc_auc_score(y_np, proba_np, sample_weight=w_np)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Weighted by Census Sample Weights", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out_path = Path(config["paths"]["results_figures"]) / "roc_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] ROC curve saved → {out_path}")


def plot_pr_curves(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    config: dict,
) -> None:
    """Plot weighted Precision-Recall curves for all models."""
    y_np = np.asarray(y)
    w_np = np.asarray(w)

    # Weighted prevalence — accounts for stratified sampling design
    baseline = np.average(y_np, weights=w_np)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        proba_np = model.predict_proba(X)[:, 1]
        prec, rec, _ = precision_recall_curve(y_np, proba_np, sample_weight=w_np)
        pr_auc = average_precision_score(y_np, proba_np, sample_weight=w_np)
        ax.plot(rec, prec, label=f"{name}  (PR-AUC={pr_auc:.3f})", linewidth=2)

    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1,
               label=f"Weighted baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Weighted by Census Sample Weights", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out_path = Path(config["paths"]["results_figures"]) / "pr_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] PR curve saved → {out_path}")


def plot_confusion_matrix(
    name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    threshold: float,
    config: dict,
) -> None:
    """
    Plot weighted confusion matrix at the chosen threshold.
    Values represent population-weighted counts, not raw sample counts.
    """
    y_np     = np.asarray(y)
    w_np     = np.asarray(w)
    proba_np = model.predict_proba(X)[:, 1]
    y_pred   = (proba_np >= threshold).astype(int)

    # Population-weighted confusion matrix
    cm = confusion_matrix(y_np, y_pred, sample_weight=w_np)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["<50K (0)", ">50K (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(
        f"Confusion Matrix — {name}\n"
        f"(population-weighted counts, threshold={threshold})", fontsize=11
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12)

    out_path = (Path(config["paths"]["results_figures"])
                / f"confusion_matrix_{name}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] Confusion matrix saved → {out_path}")


def evaluate_all(
    models: Dict[str, Any],
    data: dict,
    config: dict,
    split: str = "test",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run full evaluation suite on all models for a given split.

    Parameters
    ----------
    models    : {name: fitted_model}
    data      : output from build_features()
    config    : project config
    split     : one of "train", "val", "test"
    threshold : classification threshold

    Returns
    -------
    pd.DataFrame of results, one row per model
    """
    X = data[f"X_{split}"]
    y = data[f"y_{split}"]
    w = data[f"w_{split}"]

    print(f"\n[metrics] Evaluating on {split} set "
          f"({len(y):,} samples, threshold={threshold})")

    all_results = []

    for name, model in models.items():
        result = evaluate_model(name, model, X, y, w, threshold)
        all_results.append(result)

        # Population-weighted confusion matrix
        plot_confusion_matrix(name, model, X, y, w, threshold, config)

        # Unweighted classification report — for interpretability only
        # All selection decisions use weighted metrics above
        y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
        print(f"\n[metrics] {name} — Unweighted classification report "
              f"(shown for interpretability; decisions use weighted metrics above):")
        print(classification_report(
            np.asarray(y), y_pred,
            target_names=["<50K", ">50K"],
            zero_division=0
        ))

    # ROC and PR curves — all models on same axes
    plot_roc_curves(models, X, y, w, config)
    plot_pr_curves(models, X, y, w, config)

    # Save results table
    results_df = pd.DataFrame(all_results)
    out_path   = Path(config["paths"]["results_tables"]) / f"metrics_{split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\n[metrics] Results table saved → {out_path}")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    # Run from project root:
    # python -m src.evaluation.metrics
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean
    from src.features.engineer import build_features

    config = load_config("config.yaml")
    raw_df = load_raw_data(config)
    X, y, w = clean(raw_df, config)
    data = build_features(X, y, w, config)

    # Load saved models
    models_dir  = Path(config["paths"]["results_models"])
    model_names = ["logistic_regression", "random_forest", "xgboost"]
    models = {
        name: joblib.load(models_dir / f"{name}.joblib")
        for name in model_names
    }

    print("\n" + "="*60)
    print("VALIDATION SET")
    print("="*60)
    evaluate_all(models, data, config, split="val")

    print("\n" + "="*60)
    print("TEST SET")
    print("="*60)
    evaluate_all(models, data, config, split="test")