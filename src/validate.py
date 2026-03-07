from typing import Dict

import pandas as pd

# ── Supported dtype-kind labels and their pandas checks ──────────────────────
# Students can extend this map to add new kinds (e.g. "boolean", "datetime").
#
# WHY is_string_dtype for "categorical" instead of is_object_dtype?
# pandas >= 2.0 infers string columns as StringDtype (dtype shown as "str")
# rather than the legacy object dtype. is_string_dtype() returns True for
# BOTH old object columns AND new StringDtype columns, making the 
# check version-independent.
_DTYPE_CHECKERS: Dict[str, object] = {
    "numeric":     pd.api.types.is_numeric_dtype,
    "categorical": pd.api.types.is_string_dtype,
}


def validate_dataframe(df: pd.DataFrame, required_columns: Dict[str, str], target_column: str) -> bool:
    """
    Inputs:
    - df: Cleaned DataFrame to validate
    - required_columns: Schema dict mapping column name → expected dtype kind.
        Supported dtype kinds: "numeric", "categorical"
        Example:
          {
              "tenure":          "numeric",
              "MonthlyCharges":  "numeric",
              "TotalCharges":    "numeric",
              "gender":          "categorical",
              "Contract":        "categorical",
              "Churn":           "numeric",   # encoded as 0/1 by clean_data.py
          }
    - target_column: Name of the column to predict (e.g. "Churn").
        Validated separately from the schema dict because its absence is always
        a pipeline-breaking error regardless of the feature configuration.
    Outputs:
    - True if every check passes; raises ValueError on the first failure
    Why this contract matters for reliable ML delivery:
    - Checking dtype KIND (not just column presence) catches silent bugs like
      TotalCharges staying as str after a failed cast — bugs that would only
      surface as a crash inside the sklearn Pipeline at train time.
    """
    print("[validate] Running data quality checks...")  # TODO: replace with logging later


    
    # ─────────────────────────────────────────────────────────────────────────
    # Check 1 — Empty DataFrame
    # ─────────────────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError(
            "[validate] FAILED: DataFrame is empty. "
            "Check your data source path and clean_data.py."
        )
    print(f"[validate.validate_dataframe] PASSED  — Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")  # TODO: replace with logging later

    
    # ─────────────────────────────────────────────────────────────────────────
    # Check 2 — Schema: presence + dtype kind for every required column
    # ─────────────────────────────────────────────────────────────────────────
    schema_errors = []

    for col, expected_kind in required_columns.items():

        # 2a — Column must exist
        if col not in df.columns:
            schema_errors.append(
                f"  MISSING  '{col}'  (expected dtype kind: '{expected_kind}')"
            )
            continue   # can't check dtype of a missing column

        # 2b — dtype kind must match
        checker = _DTYPE_CHECKERS.get(expected_kind)
        if checker is None:
            raise ValueError(
                f"[validate] Unknown dtype kind '{expected_kind}' for column '{col}'. "
                f"Supported kinds: {list(_DTYPE_CHECKERS.keys())}"
            )

        if not checker(df[col]):
            actual_dtype = str(df[col].dtype)
            schema_errors.append(
                f"  WRONG TYPE  '{col}'  "
                f"expected '{expected_kind}' but got dtype='{actual_dtype}'"
            )

    if schema_errors:
        raise ValueError(
            "[validate] FAILED — Schema errors detected:\n"
            + "\n".join(schema_errors)
            + f"\n\n  Available columns + dtypes:\n"
            + "\n".join(
                f"    {c}: {df[c].dtype}" for c in df.columns
            )
        )

    print(f"[validate] PASSED  — Schema OK for {len(required_columns)} required columns")  # TODO: replace with logging later

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3 — target column in df
    # ─────────────────────────────────────────────────────────────────────────
    if target_column not in df.columns:
        raise ValueError(
            f"[validate] FAILED: Target column '{target_column}' not found in DataFrame.\n"
            f"  Available columns: {list(df.columns)}\n"
            f"  → Check that clean_data.py does not drop or rename '{target_column}', "
            f"and that SETTINGS['target_column'] matches the column name in your CSV."
        )
    print(f"[validate] PASSED  — Target column '{target_column}' present")  # TODO: replace with logging later


    # ─────────────────────────────────────────────────────────────────────────
    # Check 4 — No fully-NaN rows
    # ─────────────────────────────────────────────────────────────────────────
    fully_nan_mask = df.isna().all(axis=1)
    n_fully_nan = int(fully_nan_mask.sum())

    if n_fully_nan > 0:
        raise ValueError(
            f"[validate] FAILED: {n_fully_nan} row(s) have NaN in every column.\n"
            f"  First offending indices: {list(df.index[fully_nan_mask][:5])}\n"
            f"  → These rows carry no information. Drop them in clean_data.py:\n"
            f"    df.dropna(how='all', inplace=True)"
        )
    print(f"[validate] PASSED  — No fully-NaN rows found")  # TODO: replace with logging later

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5 — No negative values in any numeric column
    # ─────────────────────────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    negative_cols = [
        col for col in numeric_cols
        if (df[col] < 0).any()
    ]

    if negative_cols:
        details = [
            f"    '{col}':  min={df[col].min():.4g}"
            for col in negative_cols
        ]
        raise ValueError(
            f"[validate] FAILED: Negative values found in {len(negative_cols)} numeric column(s):\n"
            + "\n".join(details)
            + "\n  → Fix in clean_data.py (e.g. clip or drop negative rows)."
        )
    print(f"[validate] PASSED  — No negative values across {len(numeric_cols)} numeric column(s)")  # TODO: replace with logging later


    print("[validate] All checks passed ✓")  # TODO: replace with logging later
    return True
