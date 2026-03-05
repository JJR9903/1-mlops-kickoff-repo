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
    # Generalized Data Cleaning Strategy
    
    # 1. Handle string blanks across the entire dataframe
    # Replace empty strings or pure whitespace with NaN
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        # Strip string columns. .str.strip() safely ignores genuine NaNs
        df_clean[col] = df_clean[col].str.strip().replace("", np.nan)
        # Handle literal "nan" or "<NA>" strings that might emerge from CSV loading
        df_clean[col] = df_clean[col].replace({"nan": np.nan, "<NA>": np.nan, "None": np.nan, "None": np.nan, "null": np.nan})

    # 2. Attempt to convert object/string columns to numeric
    # This generalizing "TotalCharges" conversion for any similar columns.
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        if col != target_column:
            # We check if dropping NaNs leaves us with values that *could* be numeric
            # A simple heuristic: try to_numeric; if it succeeds with reasonable NaNs, keep it
            s_numeric = pd.to_numeric(df_clean[col], errors="coerce")
            
            # If converting to numeric doesn't create MORE NaNs than already existed 
            # (or only introduces a few due to edge-case dirty data), we convert it.
            # Here, if we successfully converted >50% of the non-null data, we keep the numeric version.
            not_null_count_before = df_clean[col].notna().sum()
            not_null_count_after = s_numeric.notna().sum()
            
            if not_null_count_before > 0 and (not_null_count_after / not_null_count_before) > 0.5:
                df_clean[col] = s_numeric

    # 3. Fill missing numerical values with Median
    # Generalized to all numerical columns
    for col in df_clean.select_dtypes(include=['number']).columns:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # 4. Drop ID-like columns
    # Identifying ID-like columns: high cardinality categorical columns.
    # E.g., if >90% of values are unique, it's likely an identifier (like customerID).
    cols_to_drop = []
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        if col != target_column:
            num_unique = df_clean[col].nunique()
            num_total = df_clean[col].notna().sum()
            if num_total > 0 and (num_unique / num_total) > 0.90:
                cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"[clean_data.clean_dataframe] Dropping high-cardinality/ID columns: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)

    # 5. Target mapping (Specific to Yes/No, can be generalized easily if needed)
    if target_column in df_clean.columns:
        is_text_type = pd.api.types.is_object_dtype(df_clean[target_column]) or pd.api.types.is_string_dtype(df_clean[target_column])
        if is_text_type:
            # Check for Yes/No (boolean representations)
            if set(df_clean[target_column].dropna().unique()).issubset({"Yes", "No", "True", "False"}):
                mapping = {"No": 0, "Yes": 1, "False": 0, "True": 1}
                df_clean[target_column] = df_clean[target_column].map(mapping)
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