from typing import Dict, List, Optional

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


def validate_dataframe(
    df: pd.DataFrame,
    schema: Dict[str, dict],
    target_column: str,
    allowed_classes: Optional[List] = None,
) -> bool:
    """
    Validates a cleaned DataFrame against a schema before it enters the ML pipeline.

    Inputs:
    - df             : Cleaned DataFrame to validate.
    - schema         : Dict mapping column name → metadata dict with two keys:
                           "type"       : dtype kind, one of "numeric" or "categorical"
                           "accept_nan" : bool — True if NaNs are expected (imputed later);
                                          False if NaNs are a hard pipeline error.
                       Example:
                           {
                               "customerID":   {"type": "categorical", "accept_nan": False},
                               "tenure":       {"type": "numeric",     "accept_nan": False},
                               "TotalCharges": {"type": "numeric",     "accept_nan": True},
                               "Churn":        {"type": "numeric",     "accept_nan": False},
                           }
    - target_column  : Name of the column to predict (e.g. "Churn").
                       Kept as a separate argument so it is structurally impossible
                       to declare more than one target — no runtime meta-validation needed.
    - allowed_classes: Optional list of valid values for the target column
                       (e.g. [0, 1] for a binary classifier).
                       If None, the class-membership check is skipped.

    Output:
    - True if every check passes; raises ValueError on the first failure.

    Why this contract matters for reliable ML delivery:
    - Checking dtype KIND (not just column presence) catches silent bugs like
      TotalCharges staying as str after a failed cast — bugs that would only
      surface as a crash inside the sklearn Pipeline at train time.
    - Separating structural NaNs (hard fail) from imputable NaNs (log & allow)
      prevents leakage while still catching genuinely broken rows.
    - Keeping target_column as a plain string makes the single-target contract
      self-documenting and enforced by the call signature, not by runtime logic.
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
    print(
        f"[validate] PASSED  — Shape: {df.shape[0]:,} rows × {df.shape[1]} columns"
    )  # TODO: replace with logging later

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2 — Schema: presence, dtype kind, and NaN policy per column
    # ─────────────────────────────────────────────────────────────────────────
    schema_errors = []

    for col, meta in schema.items():

        expected_kind = meta.get("type")
        accept_nan    = meta.get("accept_nan", False)

        # 2a — Column must exist
        if col not in df.columns:
            schema_errors.append(
                f"  MISSING  '{col}'  (expected dtype kind: '{expected_kind}')"
            )
            continue  # can't check dtype or NaNs of a missing column

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

        # 2c — NaN policy
        n_nan = int(df[col].isna().sum())

        if n_nan > 0:
            if not accept_nan:
                # Structural NaN — this column must be clean after clean_data.py
                schema_errors.append(
                    f"  UNEXPECTED NaN  '{col}'  has {n_nan} NaN(s) but accept_nan=False.\n"
                    f"    → Drop or fill these rows in clean_data.py."
                )
            else:
                # Imputable NaN — expected, will be handled after train-test split
                print(
                    f"[validate] INFO   — '{col}' has {n_nan} NaN(s) "
                    f"(accept_nan=True → will be imputed downstream)"
                )  # TODO: replace with logging later

    if schema_errors:
        raise ValueError(
            "[validate] FAILED — Schema errors detected:\n"
            + "\n".join(schema_errors)
            + f"\n\n  Available columns + dtypes:\n"
            + "\n".join(f"    {c}: {df[c].dtype}" for c in df.columns)
        )

    print(
        f"[validate] PASSED  — Schema OK for {len(schema)} column(s) "
        f"(NaN policy enforced per column)"
    )  # TODO: replace with logging later

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3 — Target column present
    # ─────────────────────────────────────────────────────────────────────────
    if target_column not in df.columns:
        raise ValueError(
            f"[validate] FAILED: Target column '{target_column}' not found in DataFrame.\n"
            f"  Available columns: {list(df.columns)}\n"
            f"  → Check that clean_data.py does not drop or rename '{target_column}', "
            f"and that the 'target_column' argument matches the column name in your CSV."
        )
    print(
        f"[validate] PASSED  — Target column '{target_column}' present"
    )  # TODO: replace with logging later

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4 — Target class membership (only if allowed_classes is provided)
    # ─────────────────────────────────────────────────────────────────────────
    if allowed_classes is not None:
        actual_classes  = set(df[target_column].dropna().unique())
        allowed_set     = set(allowed_classes)
        unexpected      = actual_classes - allowed_set

        if unexpected:
            raise ValueError(
                f"[validate] FAILED: Target column '{target_column}' contains unexpected "
                f"class value(s): {unexpected}.\n"
                f"  Allowed classes : {allowed_set}\n"
                f"  Found classes   : {actual_classes}\n"
                f"  → Ensure clean_data.py encodes '{target_column}' correctly "
                f"(e.g. Yes/No → 1/0) before validation."
            )
        print(
            f"[validate] PASSED  — Target classes {actual_classes} ⊆ allowed {allowed_set}"
        )  # TODO: replace with logging later

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5 — No fully-NaN rows
    # ─────────────────────────────────────────────────────────────────────────
    fully_nan_mask = df.isna().all(axis=1)
    n_fully_nan    = int(fully_nan_mask.sum())

    if n_fully_nan > 0:
        raise ValueError(
            f"[validate] FAILED: {n_fully_nan} row(s) have NaN in every column.\n"
            f"  First offending indices: {list(df.index[fully_nan_mask][:5])}\n"
            f"  → These rows carry no information. Drop them in clean_data.py:\n"
            f"    df.dropna(how='all', inplace=True)"
        )
    print(
        f"[validate] PASSED  — No fully-NaN rows found"
    )  # TODO: replace with logging later

    # ─────────────────────────────────────────────────────────────────────────
    # Check 6 — No negative values in any numeric column
    # ─────────────────────────────────────────────────────────────────────────
    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    negative_cols = [col for col in numeric_cols if (df[col] < 0).any()]

    if negative_cols:
        details = [f"    '{col}':  min={df[col].min():.4g}" for col in negative_cols]
        raise ValueError(
            f"[validate] FAILED: Negative values found in {len(negative_cols)} numeric column(s):\n"
            + "\n".join(details)
            + "\n  → Fix in clean_data.py (e.g. clip or drop negative rows)."
        )
    print(
        f"[validate] PASSED  — No negative values across {len(numeric_cols)} numeric column(s)"
    )  # TODO: replace with logging later

    print("[validate] All checks passed ✓")  # TODO: replace with logging later
    return True