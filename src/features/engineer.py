"""
engineer.py
-----------
Responsible for feature engineering and train/val/test splitting.

Operations performed (in order):
  1. Split into train / val / test (stratified, indices reset after split).
  2. Median-impute numeric NaNs — fitted on train, applied to val/test.
  3. Log1p transform skewed numeric columns — clip(lower=0) first for safety.
  4. One-hot encode categoricals — vocabulary from train, val/test reindexed.
  5. Sanitize column names — remove characters XGBoost rejects ([, ], <, >).
  6. Return processed splits + metadata dict.

Design decisions:
  - All transformations fitted on TRAIN only — applying train statistics to
    val/test is non-negotiable to prevent data leakage.
  - drop_first is config-driven (default False). Dropping a dummy level helps
    linear models avoid multicollinearity but hurts feature importance
    interpretability for tree models (RF, XGBoost). Regularisation handles
    collinearity in LogisticRegression without needing drop_first.
  - clip(lower=0) before log1p guards against unexpected negatives in
    financial columns without silently altering real data values.
  - Unseen categories in val/test are handled via reindex → become 0 columns.
  - Column names are sanitized after OHE — XGBoost rejects names containing
    [, ], or < which appear when category values have special characters.
  - build_features returns a dict (not a tuple) for readable call sites.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    config: dict
) -> Tuple:
    """
    Split into train / val / test with stratification on label.
    Two-stage split: train vs (val+test), then val vs test.
    """
    seed      = config["random_seed"]
    val_size  = config["split"]["val_size"]
    test_size = config["split"]["test_size"]
    stratify  = config["split"]["stratify"]

    strat = y if stratify else None

    X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        X, y, w,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=strat
    )

    relative_test_size = test_size / (val_size + test_size)
    strat_temp = y_temp if stratify else None

    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_temp, y_temp, w_temp,
        test_size=relative_test_size,
        random_state=seed,
        stratify=strat_temp
    )

    print(f"[engineer] Split sizes:")
    print(f"           Train : {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"           Val   : {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"           Test  : {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\n[engineer] Positive rate (>50K) by split:")
    print(f"           Train : {y_train.mean()*100:.2f}%")
    print(f"           Val   : {y_val.mean()*100:.2f}%")
    print(f"           Test  : {y_test.mean()*100:.2f}%")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            w_train, w_val, w_test)


def impute_numeric(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Median-impute numeric NaNs.
    Medians computed on train only, then applied to val and test.
    Returns the median map for logging and reproducibility.
    """
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    na_cols  = [c for c in num_cols if X_train[c].isnull().any()]

    median_map = {}
    for col in na_cols:
        median_val      = X_train[col].median()
        median_map[col] = median_val
        X_train[col]    = X_train[col].fillna(median_val)
        X_val[col]      = X_val[col].fillna(median_val)
        X_test[col]     = X_test[col].fillna(median_val)

    if median_map:
        print(f"[engineer] Median-imputed {len(median_map)} columns "
              f"(train medians only): {list(median_map.keys())}")
    else:
        print(f"[engineer] No numeric NaNs to impute.")

    return X_train, X_val, X_test, median_map


def log_transform(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply log1p to skewed numeric columns defined in config.

    clip(lower=0) is applied first as a safety guardrail — log1p is
    undefined for negative values and financial columns should be
    non-negative, but we enforce this explicitly rather than assume it.

    No fitting required — log1p is a deterministic transformation.
    """
    if not config["features"].get("log_transform_skewed_numeric", False):
        return X_train, X_val, X_test

    skewed_cols = config["data"].get("skewed_numeric_columns", [])
    skewed_cols = [c for c in skewed_cols if c in X_train.columns]

    for col in skewed_cols:
        for split in [X_train, X_val, X_test]:
            split[col] = np.log1p(split[col].clip(lower=0))

    print(f"[engineer] log1p (with clip lower=0) applied to: {skewed_cols}")

    return X_train, X_val, X_test


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names after one-hot encoding.

    XGBoost rejects feature names containing [, ], or < — these characters
    appear when categorical values contain special characters (e.g. age
    bins like '<25' or encoded labels with brackets). Replace with underscores.
    Also replace spaces with underscores for clean, consistent naming.
    """
    df.columns = (
        df.columns.str.replace("[", "_", regex=False)
                  .str.replace("]", "_", regex=False)
                  .str.replace("<", "_", regex=False)
                  .str.replace(">", "_", regex=False)
                  .str.replace(" ", "_", regex=False)
    )
    return df


def encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode all categorical columns.

    - Vocabulary (column set) comes from train only.
    - Val and test are reindexed to match train columns exactly.
    - Unseen categories in val/test → 0 columns (no crash, no leakage).
    - Column names sanitized after encoding for XGBoost compatibility.
    - drop_first is config-driven (default False):
        False → keep full dummies (recommended for RF/XGBoost).
        True  → drop one level per feature (useful for LogisticRegression only).
    """
    cat_cols   = X_train.select_dtypes(include="object").columns.tolist()
    drop_first = config["features"].get("drop_first", False)

    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=drop_first)
    X_val   = pd.get_dummies(X_val,   columns=cat_cols, drop_first=drop_first)
    X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=drop_first)

    # Reindex val/test to match train's column space exactly
    X_val  = X_val.reindex(columns=X_train.columns,  fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Sanitize column names — XGBoost rejects [, ], <, > in feature names
    X_train = sanitize_columns(X_train)
    X_val   = sanitize_columns(X_val)
    X_test  = sanitize_columns(X_test)

    print(f"[engineer] One-hot encoded {len(cat_cols)} categorical columns "
          f"(drop_first={drop_first})")
    print(f"[engineer] Final feature count after encoding: {X_train.shape[1]}")

    return X_train, X_val, X_test


def build_features(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    config: dict
) -> dict:
    """
    Full feature engineering pipeline.
    Order: split → impute → log transform → encode → sanitize.

    Returns a dict for readable downstream access.
    """
    # Step 1 — Split
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     w_train, w_val, w_test) = split_data(X, y, w, config)

    # Reset indices — misaligned indices after split cause subtle sklearn bugs
    for df in [X_train, X_val, X_test]:
        df.reset_index(drop=True, inplace=True)
    for s in [y_train, y_val, y_test, w_train, w_val, w_test]:
        s.reset_index(drop=True, inplace=True)

    # Step 2 — Median imputation (train statistics only)
    X_train, X_val, X_test, median_map = impute_numeric(X_train, X_val, X_test)

    # Step 3 — Log1p transform with safety clip
    X_train, X_val, X_test = log_transform(X_train, X_val, X_test, config)

    # Step 4 — One-hot encoding (train vocabulary only) + sanitize column names
    X_train, X_val, X_test = encode_categoricals(X_train, X_val, X_test, config)

    print(f"\n[engineer] Pipeline complete.")
    print(f"[engineer] Train : {X_train.shape}, Val : {X_val.shape}, Test : {X_test.shape}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "w_train": w_train, "w_val": w_val, "w_test": w_test,
        "feature_columns": X_train.columns.tolist(),
        "median_map": median_map,
    }


if __name__ == "__main__":
    # Run from project root:
    # python -m src.features.engineer
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean

    config = load_config("config.yaml")
    raw_df = load_raw_data(config)
    X, y, w = clean(raw_df, config)
    data = build_features(X, y, w, config)

    print("\nSample train features (first 3 rows, transposed):")
    print(data["X_train"].head(3).T)