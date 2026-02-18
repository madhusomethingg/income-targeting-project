"""
train.py
--------
Trains three classifiers on the processed feature matrix:
  - Logistic Regression (linear baseline — saga solver for sparse one-hot data)
  - Random Forest      (non-linear ensemble, captures feature interactions)
  - XGBoost            (gradient boosting, typically strongest performer)

Design decisions:
  - All hyperparameters come from config.yaml — nothing hardcoded here.
  - sample_weight passed during fit() if use_sample_weight is True.
    This accounts for the stratified sampling design of the census survey.
  - scale_pos_weight only passed to XGBoost if explicitly set in config
    (not None). Passing None breaks XGBoost — it expects a numeric value.
  - saga solver chosen for LogisticRegression: supports l2 penalty,
    parallelizable via n_jobs, and handles large sparse one-hot matrices well.
  - Val AUC printed during training as a sanity check only.
    Full weighted evaluation lives in evaluation/metrics.py.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_models(config: dict) -> Dict[str, Any]:
    """
    Instantiate all classifiers from config parameters.
    Returns a dict of {model_name: unfitted_model}.
    """
    seed    = config["random_seed"]
    cfg     = config["models"]
    lr_cfg  = cfg["logistic_regression"]
    rf_cfg  = cfg["random_forest"]
    xgb_cfg = cfg["xgboost"]

    # --- Logistic Regression ---
    # saga: supports l2, parallelizable, good for large sparse matrices
    lr = LogisticRegression(
        solver      = lr_cfg["solver"],
        penalty     = lr_cfg["penalty"],
        max_iter    = lr_cfg["max_iter"],
        class_weight= lr_cfg["class_weight"],
        C           = lr_cfg["C"],
        n_jobs      = -1,
        random_state= seed,
    )

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators    = rf_cfg["n_estimators"],
        max_depth       = rf_cfg["max_depth"],
        min_samples_leaf= rf_cfg["min_samples_leaf"],
        class_weight    = rf_cfg["class_weight"],
        random_state    = seed,
        n_jobs          = rf_cfg["n_jobs"],
    )

    # --- XGBoost ---
    # scale_pos_weight only passed if explicitly set in config (not None).
    # Passing None directly to XGBClassifier raises a type error.
    xgb_kwargs = dict(
        n_estimators    = xgb_cfg["n_estimators"],
        max_depth       = xgb_cfg["max_depth"],
        learning_rate   = xgb_cfg["learning_rate"],
        subsample       = xgb_cfg["subsample"],
        colsample_bytree= xgb_cfg["colsample_bytree"],
        reg_lambda      = xgb_cfg["reg_lambda"],
        tree_method     = xgb_cfg["tree_method"],
        eval_metric     = xgb_cfg["eval_metric"],
        n_jobs          = xgb_cfg["n_jobs"],
        random_state    = seed,
        verbosity       = 0,
    )
    if xgb_cfg.get("scale_pos_weight") is not None:
        xgb_kwargs["scale_pos_weight"] = xgb_cfg["scale_pos_weight"]

    xgb = XGBClassifier(**xgb_kwargs)

    models = {
        "logistic_regression": lr,
        "random_forest"      : rf,
        "xgboost"            : xgb,
    }

    print(f"[train] Instantiated {len(models)} models: {list(models.keys())}")
    return models


def fit_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_val: pd.Series,
    use_sample_weight: bool,
) -> Any:
    """
    Fit a single model and print a quick weighted val AUC sanity check.
    Full metric suite is computed in evaluation/metrics.py.
    """
    print(f"\n[train] Fitting {name} ...")

    fit_kwargs = {}
    if use_sample_weight:
        fit_kwargs["sample_weight"] = w_train

    model.fit(X_train, y_train, **fit_kwargs)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc   = roc_auc_score(y_val, val_proba, sample_weight=w_val)
    print(f"[train] {name} — Weighted Val AUC-ROC: {val_auc:.4f}")

    return model


def save_model(model: Any, name: str, config: dict) -> None:
    """Save fitted model to results/models/ using joblib."""
    output_dir = Path(config["paths"]["results_models"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.joblib"
    joblib.dump(model, output_path)
    print(f"[train] Saved → {output_path}")


def train_all(data: dict, config: dict) -> Dict[str, Any]:
    """
    Train all models and save to disk.

    Parameters
    ----------
    data : dict
        Output from features.engineer.build_features().
    config : dict
        Project config from config.yaml.

    Returns
    -------
    fitted_models : dict
        {model_name: fitted_model}
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    w_train = data["w_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    w_val   = data["w_val"]

    use_sample_weight = config["models"].get("use_sample_weight", True)

    print(f"[train] use_sample_weight = {use_sample_weight}")
    print(f"[train] Training on {len(X_train):,} samples, "
          f"{X_train.shape[1]} features")

    models        = build_models(config)
    fitted_models = {}

    for name, model in models.items():
        fitted = fit_model(
            name, model,
            X_train, y_train, w_train,
            X_val,   y_val,   w_val,
            use_sample_weight=use_sample_weight,
        )
        save_model(fitted, name, config)
        fitted_models[name] = fitted

    print(f"\n[train] All models trained and saved.")
    return fitted_models


if __name__ == "__main__":
    # Run from project root:
    # python -m src.models.train
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean
    from src.features.engineer import build_features

    config        = load_config("config.yaml")
    raw_df        = load_raw_data(config)
    X, y, w       = clean(raw_df, config)
    data          = build_features(X, y, w, config)
    fitted_models = train_all(data, config)