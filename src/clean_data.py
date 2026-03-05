"""
Educational Goal:
- Why this module exists in an MLOps system: Make cleaning repeatable and testable outside notebooks.
- Responsibility (separation of concerns): Transform raw df -> clean df without training leakage.
- Pipeline contract (inputs and outputs): df_raw + target_column -> clean DataFrame.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import numpy as np
import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: Raw dataframe.
    - target_column: Name of the target column used for training.
    Outputs:
    - df_clean: Clean dataframe ready for splitting into X/y.
    Why this contract matters for reliable ML delivery:
    - Cleaning must be deterministic so training, evaluation, and inference behave consistently.
    """
    print("[clean_data.clean_dataframe] Cleaning dataframe (baseline: identity copy)")  # TODO: replace with logging later
    df_clean = df_raw.copy(deep=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook cleaning logic here to replace or extend the baseline
    # Why: Cleaning varies by dataset quality, schema quirks, and business definitions.
    # Examples:
    # 1. Convert 'TotalCharges' from object to numeric and handle blanks (Telco churn dataset)
    # 2. Map target labels (e.g., 'Yes'/'No') to 1/0 for classification
    #
    # Telco churn notebook-style logic (safe-guarded so dummy dataset still works):
    if "TotalCharges" in df_clean.columns:
        # In Telco churn CSV, TotalCharges can contain blanks like " "
        df_clean["TotalCharges"] = df_clean["TotalCharges"].replace(" ", np.nan)
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        if df_clean["TotalCharges"].isna().any():
            median_val = df_clean["TotalCharges"].median()
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(median_val)

    if "customerID" in df_clean.columns:
        # customerID is an identifier, not a predictive feature in most churn notebooks
        df_clean = df_clean.drop(columns=["customerID"])

    if target_column in df_clean.columns:
        # Typical notebook mapping for Telco churn: Yes/No -> 1/0
        is_text_type = pd.api.types.is_object_dtype(df_clean[target_column]) or pd.api.types.is_string_dtype(df_clean[target_column])
        if is_text_type:
            if set(df_clean[target_column].dropna().unique()).issubset({"Yes", "No"}):
                df_clean[target_column] = df_clean[target_column].map({"No": 0, "Yes": 1})
                
                # In newer pandas with string extension types, map can return Object dtype. 
                # to_numeric safely forces it to float64/Int64 when converted values are numeric.
                df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors="coerce")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_clean


if __name__ == "__main__":
    from pathlib import Path
    from src.load_data import load_raw_data

    df_raw = load_raw_data(Path("data/raw/telco.csv"))
    df_clean = clean_dataframe(df_raw, target_column="target")

    print("Raw shape:", df_raw.shape)
    print("Clean shape:", df_clean.shape)
    print(df_clean.head())