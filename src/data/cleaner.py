"""
cleaner.py
----------
Responsible for cleaning the raw loaded DataFrame.

Operations performed (in order):
  1. Extract weight and label columns before any transformation.
  2. Drop columns specified in config (year, detailed recodes, weight, label).
  3. Create migration_data_absent flag BEFORE filling NaNs.
  4. Fill NaN in migration columns with "Unknown".
  5. Fill remaining NaN in string columns with "Unknown".
  6. Coerce numeric columns to float64.
     Candidates = already-numeric columns UNION skewed_numeric_columns from config.
     This handles columns that loaded as object due to mixed values.
     Median imputation of any remaining numeric NaNs is deferred to features/engineer.py.
  7. Map label column to binary integer (0 / 1).
  8. Return: cleaned feature df, label series, weight series.

Design decisions:
  - Weight is extracted and returned separately — it is never a feature.
  - migration_data_absent uses all(axis=1): flagged only when ALL migration
    columns are missing, reflecting "not applicable / did not move" rather
    than partial data quality issues.
  - Label mapped to 0/1 here so all downstream code works with integers.
  - No encoding or scaling here — that belongs in features/engineer.py.
"""

import pandas as pd
import yaml
from typing import Tuple


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Clean the raw loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from loader.load_raw_data().
    config : dict
        Project config loaded from config.yaml.

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix (no label, no weight).
    y : pd.Series
        Binary label (0 = <50K, 1 = >50K).
    w : pd.Series
        Sample weights from census (to be used in model training).
    """
    df = df.copy()

    cfg        = config["data"]
    target_col = cfg["target_column"]
    weight_col = cfg["weight_column"]
    pos_label  = cfg["positive_label"]
    neg_label  = cfg["negative_label"]

    # ------------------------------------------------------------------
    # Step 1 — Extract weight and label before touching the dataframe
    # ------------------------------------------------------------------
    if weight_col not in df.columns:
        raise KeyError(f"Weight column '{weight_col}' not found in dataframe.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    w     = pd.to_numeric(df[weight_col].copy(), errors="coerce")
    y_raw = df[target_col].copy()

    # ------------------------------------------------------------------
    # Step 2 — Drop configured columns + weight + label
    # ------------------------------------------------------------------
    cols_to_drop = cfg.get("drop_columns", []) + [weight_col, target_col]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[cleaner] Dropped {len(cols_to_drop)} columns: {cols_to_drop}")

    # ------------------------------------------------------------------
    # Step 3 — Create migration_data_absent flag BEFORE filling NaNs
    #
    # Using all(axis=1): flagged only when ALL migration columns are missing.
    # This reflects "not applicable / did not move" rather than partial
    # data quality issues. Missingness here carries predictive signal and
    # must be captured before NaN filling destroys it.
    # ------------------------------------------------------------------
    migration_cols         = cfg.get("migration_columns", [])
    migration_cols_present = [c for c in migration_cols if c in df.columns]

    if migration_cols_present:
        df["migration_data_absent"] = (
            df[migration_cols_present].isnull().all(axis=1).astype(int)
        )
        n_flagged = df["migration_data_absent"].sum()
        print(f"[cleaner] migration_data_absent: {n_flagged:,} rows flagged "
              f"({n_flagged / len(df) * 100:.1f}% of data)")

    # ------------------------------------------------------------------
    # Step 4 — Fill NaN in migration columns with "Unknown"
    # ------------------------------------------------------------------
    for col in migration_cols_present:
        df[col] = df[col].fillna("Unknown")

    # ------------------------------------------------------------------
    # Step 5 — Fill remaining NaN in string columns with "Unknown"
    # ------------------------------------------------------------------
    str_cols    = df.select_dtypes(include="object").columns
    str_na_cols = [c for c in str_cols if df[c].isnull().any()]

    for col in str_na_cols:
        df[col] = df[col].fillna("Unknown")

    if str_na_cols:
        print(f"[cleaner] Filled NaNs in {len(str_na_cols)} string columns "
              f"with 'Unknown': {str_na_cols}")

    # ------------------------------------------------------------------
    # Step 6 — Coerce numeric columns to float64
    #
    # Candidates = columns already typed as number  UNION  skewed_numeric_columns
    # from config. This handles columns that loaded as object due to mixed
    # values — select_dtypes alone would miss those silently.
    # Median imputation of remaining NaNs is deferred to features/engineer.py.
    # ------------------------------------------------------------------
    numeric_candidates = set(df.select_dtypes(include="number").columns.tolist())
    numeric_candidates.update(cfg.get("skewed_numeric_columns", []))
    numeric_candidates = [c for c in numeric_candidates if c in df.columns]

    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_nulls = df[numeric_candidates].isnull().sum()
    numeric_nulls = numeric_nulls[numeric_nulls > 0]
    if not numeric_nulls.empty:
        print(f"[cleaner] Numeric NaNs found (will be median-imputed in engineer.py):")
        print(numeric_nulls.to_string())
    else:
        print(f"[cleaner] No numeric NaNs found.")

    # ------------------------------------------------------------------
    # Step 7 — Map label to binary integer
    #
    # Raises an error if any value cannot be mapped.
    # Silent label corruption is the worst kind of bug in a classifier.
    # ------------------------------------------------------------------
    label_map = {pos_label: 1, neg_label: 0}
    y         = y_raw.map(label_map)

    unmapped = y.isnull().sum()
    if unmapped > 0:
        raise ValueError(
            f"[cleaner] {unmapped} label values could not be mapped. "
            f"Unique values found: {y_raw.unique()}. "
            f"Expected: {list(label_map.keys())}. "
            f"Check strip_whitespace is set to true in config.yaml."
        )

    y = y.astype(int).rename("label")
    print(f"\n[cleaner] Label distribution:")
    print(f"          >50K (1): {y.sum():,}  ({y.mean()*100:.1f}%)")
    print(f"          <50K (0): {(1-y).sum():,}  ({(1-y.mean())*100:.1f}%)")

    # ------------------------------------------------------------------
    # Step 8 — Final report
    # ------------------------------------------------------------------
    print(f"\n[cleaner] Final feature matrix: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[cleaner] Numeric columns    : {df.select_dtypes(include='number').shape[1]}")
    print(f"[cleaner] Categorical columns: {df.select_dtypes(include='object').shape[1]}")

    return df, y, w


if __name__ == "__main__":
    # Run from project root:
    # python -m src.data.cleaner
    from src.data.loader import load_raw_data

    config = load_config("config.yaml")
    raw_df = load_raw_data(config)
    X, y, w = clean(raw_df, config)

    print("\nSample of cleaned features:")
    print(X.head(3).T)
    print("\nWeight range:", w.min(), "→", w.max())