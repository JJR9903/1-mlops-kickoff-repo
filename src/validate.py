from typing import Dict
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)

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

_VALID_TARGET_TYPES = {"classification", "regression"}


def validate_dataframe(
    df: pd.DataFrame,
    schema: Dict[str, dict],
    target_config: Dict[str, dict]
) -> bool:
    """
    Validates a cleaned DataFrame against a schema before it enters
    the ML pipeline.

    Inputs:
    - df             : Cleaned DataFrame to validate.
    - schema         : Dict mapping column name → metadata dict with two keys:
                            "type"       : kind ("numeric" or
                                           "categorical")
                            "accept_nan" : bool — True if NaNs are expected
                                           (imputed later); False if NaNs are
                                           a hard pipeline error.
                       Example:
                            {
                                "customerID":   {"type": "categorical",
                                                 "accept_nan": False},
                                "tenure":       {"type": "numeric",
                                                 "accept_nan": False},
                                "TotalCharges": {"type": "numeric",
                                                 "accept_nan": True},
                                "Churn":        {"type": "numeric",
                                                 "accept_nan": False},
                            }
    - target_config : Dict describing the target column and task type.
                      Maps directly to a config.yaml block (step 2 ready).
                      Required keys:
                          "column" : str   — column name to predict
                          "type"   : str   — "classification" or "regression"
                      Optional keys (task-dependent):
                          "allowed_classes" : List — classification only
                                             e.g. [0, 1] for binary churn
                          "range"           : [min, max] — regression only,
                                             inclusive bounds for sanity check
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
    - Keeping target_column as a plain string makes the single-target
      contract self-documenting and enforced by the call signature,
      not by runtime logic.
    """
    logger.info("Running data quality checks...")

    # ─────────────────────────────────────────────────────────────────────────
    # Validate target_config structure up front
    # ─────────────────────────────────────────────────────────────────────────
    target_column = target_config.get("column")
    target_type = target_config.get("type")

    if not target_column:
        msg = (
            "[validate] FAILED: target_config is missing required key "
            "'column'.\n  → e.g. target_config = {'column': 'Churn', "
            "'type': 'classification', ...}"
        )
        raise ValueError(msg)
    if target_type not in _VALID_TARGET_TYPES:
        raise ValueError(
            f"[validate] FAILED: target_config 'type' must be one of "
            f"{_VALID_TARGET_TYPES}, got '{target_type}'."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Check 1 — Empty DataFrame
    # ─────────────────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError(
            "[validate] FAILED: DataFrame is empty. "
            "Check your data source path and clean_data.py."
        )
    msg = (
        f"[validate] PASSED  — Shape: {df.shape[0]:,} rows × "
        f"{df.shape[1]} cols"
    )
    logger.info(msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Check 2 — No fully-NaN rows
    # ─────────────────────────────────────────────────────────────────────────
    fully_nan_mask = df.isna().all(axis=1)
    n_fully_nan = int(fully_nan_mask.sum())

    if n_fully_nan > 0:
        msg = (
            f"[validate] FAILED: {n_fully_nan} row(s) have NaN in every "
            f"column.\n  First offending indices: "
            f"{list(df.index[fully_nan_mask][:5])}\n"
            f"  → These rows carry no information. Drop them in "
            f"clean_data.py:\n    df.dropna(how='all', inplace=True)"
        )
        raise ValueError(msg)
    logger.info("PASSED  — No fully-NaN rows found")

    # ─────────────────────────────────────────────────────────────────────────
    # Check 3 — Schema: presence, dtype kind, and NaN policy per column
    # ─────────────────────────────────────────────────────────────────────────
    schema_errors = []

    for col, meta in schema.items():

        expected_kind = meta.get("type")
        accept_nan = meta.get("accept_nan", False)

        # 2a — Column must exist
        if col not in df.columns:
            schema_errors.append(
                f"  MISSING  '{col}'  (expected dtype kind: '{expected_kind}')"
            )
            continue  # can't check dtype or NaNs of a missing column

        # 2b — dtype kind must match
        checker = _DTYPE_CHECKERS.get(expected_kind)
        if checker is None:
            msg = (
                f"[validate] Unknown dtype kind '{expected_kind}' for column "
                f"'{col}'. Supported kinds: {list(_DTYPE_CHECKERS.keys())}"
            )
            raise ValueError(msg)

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
                # Structural NaN — column must be clean after clean_data.py
                msg = (
                    f"  UNEXPECTED NaN  '{col}'  has {n_nan} NaN(s) but "
                    f"accept_nan=False.\n    → Drop or fill these rows in "
                    f"clean_data.py."
                )
                schema_errors.append(msg)
            else:
                # Imputable NaN — expected, handled after train-test split
                logger.info(
                    f"'{col}' has {n_nan} NaN(s) "
                    f"(accept_nan=True → will be imputed downstream)"
                )

    if schema_errors:
        raise ValueError(
            "[validate] FAILED — Schema errors detected:\n"
            + "\n".join(schema_errors)
            + "\n\n  Available columns + dtypes:\n"
            + "\n".join(f"    {c}: {df[c].dtype}" for c in df.columns)
        )

    logger.info(
        f"PASSED  — Schema OK for {len(schema)} column(s) "
        f"(NaN policy enforced per column)"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Check 4 — Target column present
    # ─────────────────────────────────────────────────────────────────────────
    if target_column not in df.columns:
        msg = (
            f"[validate] FAILED: Target column '{target_column}' not found "
            f"in DataFrame.\n  Available columns: {list(df.columns)}\n"
            f"  → Check that clean_data.py does not drop or rename "
            f"'{target_column}', and that the 'target_column' argument "
            f"matches the column name in your CSV."
        )
        raise ValueError(msg)
    logger.info(
        f"PASSED  — Target column '{target_column}' present"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Check 5 — Target validation (branches on task type)
    # ─────────────────────────────────────────────────────────────────────────
    target_series = df[target_column].dropna()

    if target_type == "classification":
        allowed_classes = target_config.get("allowed_classes")

        if allowed_classes is not None:
            actual_classes = set(target_series.unique())
            allowed_set = set(allowed_classes)
            unexpected = actual_classes - allowed_set

            if unexpected:
                msg = (
                    f"[validate] FAILED: Target '{target_column}' contains "
                    f"unexpected class value(s): {unexpected}.\n"
                    f"  Allowed classes : {allowed_set}\n"
                    f"  Found classes   : {actual_classes}\n"
                    f"  → Ensure clean_data.py encodes '{target_column}' "
                    f"correctly (e.g. Yes/No → 1/0) before validation."
                )
                raise ValueError(msg)
            logger.info(
                f"PASSED  — Target classes {actual_classes} "
                f"⊆ allowed {allowed_set}"
            )

        else:
            # No allowed_classes provided — just report what's found
            logger.info(
                f"No allowed_classes specified for "
                f"classification target. Found classes: "
                f"{set(target_series.unique())}"
            )

    elif target_type == "regression":

        # 4a — Target must be numeric for regression
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            msg = (
                f"[validate] FAILED: Regression target '{target_column}' must "
                f"be numeric, got dtype='{df[target_column].dtype}'.\n"
                f"  → Check encoding in clean_data.py."
            )
            raise ValueError(msg)
        msg = f"[validate] PASSED  — Target '{target_column}' is numeric"
        logger.info(msg)

        # 4b — Optional range check
        target_range = target_config.get("range")

        if target_range is not None:
            lo, hi = target_range
            out_of_range = target_series[
                (target_series < lo) | (target_series > hi)
                ]

            if not out_of_range.empty:
                msg = (
                    f"[validate] FAILED: Regression target '{target_column}' "
                    f"has {len(out_of_range)} value(s) outside expected range "
                    f"[{lo}, {hi}].\n  Min found : {target_series.min():.4g}\n"
                    f"  Max found : {target_series.max():.4g}\n"
                    f"  → Inspect and clip/drop outliers in clean_data.py."
                )
                raise ValueError(msg)
            logger.info(
                f"PASSED  — Regression target within range "
                f"[{lo}, {hi}]  (min={target_series.min():.4g}, "
                f"max={target_series.max():.4g})"
            )


# ─────────────────────────────────────────────────────────────────────────
# Check 6 — No negative values in any numeric column
# ─────────────────────────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    negative_cols = [col for col in numeric_cols if (df[col] < 0).any()]

    if negative_cols:
        details = [f"    '{col}':  min={df[col].min():.4g}"
                   for col in negative_cols]
        msg = (
            f"[validate] FAILED: Negative values found in "
            f"{len(negative_cols)} numeric column(s):\n"
            + "\n".join(details)
            + "\n  → Fix in clean_data.py (clip or drop negative rows)."
        )
        raise ValueError(msg)
    logger.info(
        f"PASSED  — No negative values across "
        f"{len(numeric_cols)} numeric column(s)"
    )

    logger.info("All checks passed ✓")
    return True
