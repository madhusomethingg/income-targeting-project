"""
loader.py
---------
Responsible for one thing only: reading raw data from disk
and returning a clean, named DataFrame. No transformations here.

Design decisions:
- '?' is converted to NaN at load time (data-format issue, not a transformation).
- Whitespace is stripped here for the same reason — it is a file-format artifact,
  not something the cleaning step should be responsible for.
- Column count is validated immediately to catch silent file corruption early.
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the project config from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_column_names(columns_path: str) -> list:
    """
    Read the census-bureau.columns file.
    Each line is a column name corresponding to its index in the data file.
    """
    with open(columns_path, "r") as f:
        columns = [line.strip() for line in f.readlines() if line.strip()]
    return columns


def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Load the raw census data file and assign column names.

    Steps performed here (all are format-level, not transformations):
      1. Read CSV with column names from the .columns file.
      2. Convert '?' to NaN at read time via na_values.
      3. Validate column count against expected columns.
      4. Strip leading/trailing whitespace from all string columns if configured.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with column names assigned and format artifacts removed.
        No cleaning, encoding, or transformation applied.
    """
    data_path = Path(config["paths"]["raw_data"])
    columns_path = Path(config["paths"]["raw_columns"])

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not columns_path.exists():
        raise FileNotFoundError(f"Columns file not found: {columns_path}")

    columns = load_column_names(str(columns_path))

    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        na_values=config["data"]["missing_value"],  # convert '?' → NaN at load time
        low_memory=False
    )

    # --- Safety check: column count must match expected ---
    if df.shape[1] != len(columns):
        raise ValueError(
            f"Column mismatch: expected {len(columns)} columns "
            f"but loaded file has {df.shape[1]}. "
            f"Check that the .data and .columns files are aligned."
        )

    # --- Strip whitespace from string columns if configured ---
    # This is a file-format artifact (raw file has leading spaces in many fields
    # including the label column). Fixing here prevents label mapping bugs downstream.
    if config["data"].get("strip_whitespace", False):
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())
        print(f"[loader] Stripped whitespace from {len(str_cols)} string columns")

    print(f"[loader] Loaded data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[loader] Missing values (NaN) per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())

    return df


if __name__ == "__main__":
    # Quick sanity check — run from project root:
    # python -m src.data.loader
    config = load_config("config.yaml")
    df = load_raw_data(config)
    print("\nLabel distribution (after strip):")
    print(df[config["data"]["target_column"]].value_counts())
    print("\nFirst 3 rows:")
    print(df.head(3).T)