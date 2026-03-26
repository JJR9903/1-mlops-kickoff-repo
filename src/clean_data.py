import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean raw dataframe for Telco churn pipeline.
    Keeps rows unless they are fully empty.
    """
    logger.info("Cleaning dataframe (baseline: identity copy)")
    df_clean = df_raw.copy(deep=True)

    print("[clean_data.clean_dataframe] Cleaning dataframe")

    if df_raw is None:
        raise ValueError("df_raw is None")

    df_clean = df_raw.copy(deep=True)

    # 1. Standardize column names
    df_clean.columns = [col.strip() for col in df_clean.columns]

    # 2. Trim whitespace and convert blank strings to NaN
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].replace(
            {
                "": np.nan,
                "nan": np.nan,
                "<NA>": np.nan,
                "None": np.nan,
                "null": np.nan,
            }
        )
        if col != target_column:
            try:
                # Try to convert to numeric. Without errors=ignore, it will raise ValueError if invalid
                df_clean[col] = pd.to_numeric(df_clean[col])
            except ValueError:
                pass  # Keep as string/object if it cannot be converted

    # 3. Convert TotalCharges to numeric if present
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"], errors="coerce"
        )

    # 4. Fill numeric NaNs with median
    for col in df_clean.select_dtypes(include=["number"]).columns:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # 5. Fill categorical NaNs with mode
    for col in df_clean.select_dtypes(include=["object", "string", "category"]).columns:
        if col != target_column and df_clean[col].isna().any():
            mode_s = df_clean[col].mode()
            if not mode_s.empty:
                df_clean[col] = df_clean[col].fillna(mode_s.iloc[0])
            else:
                df_clean[col] = df_clean[col].fillna("Missing")

    # 4. Drop ID-like columns
    # E.g., if >90% of values are unique, it's likely an identifier
    # (like customerID).
    cols_to_drop = []
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        if col != target_column:
            num_unique = df_clean[col].nunique()
            num_total = df_clean[col].notna().sum()
            if num_total > 10 and (num_unique / num_total) > 0.90:
                cols_to_drop.append(col)

    if cols_to_drop:
        logger.info(f"Dropping ID columns: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)

    # 5. Target mapping (Specific to Yes/No, can be generalized easily
    # if needed)
    if target_column in df_clean.columns:
        unique_vals = set(df_clean[target_column].dropna().unique())
        acceptable = {"No", "Yes", "False", "True", 0, 1, 0.0, 1.0, "0", "1"}
        if unique_vals.issubset(acceptable):
            df_clean[target_column] = df_clean[target_column].replace(
                {"No": 0, "Yes": 1, "False": 0, "True": 1, "0": 0, "1": 1}
            )
            df_clean[target_column] = pd.to_numeric(
                df_clean[target_column], errors="coerce"
            ).astype(float)

    # 8. Drop rows that are entirely empty
    df_clean = df_clean.dropna(how="all")

    print(f"[clean_data.clean_dataframe] Raw shape: {df_raw.shape}")
    print(f"[clean_data.clean_dataframe] Clean shape: {df_clean.shape}")

    logger.debug(f"Raw shape: {df_raw.shape}")
    logger.debug(f"Clean shape: {df_clean.shape}")
    logger.debug(f"Clean head:\n{df_clean.head()}")

    return df_clean
