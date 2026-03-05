"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.

"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Centralizes feature engineering logic to prevent data leakage and ensure consistent preprocessing between training and inference.
- Responsibility (separation of concerns): Define feature transformation recipe only (no fitting, no data mutation).
- Pipeline contract (inputs and outputs): Configuration lists in → ColumnTransformer recipe out.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Optional, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3
):
    """
    Inputs:
    - quantile_bin_cols: numeric columns to apply quantile binning
    - categorical_onehot_cols: categorical columns for one-hot encoding
    - numeric_passthrough_cols: numeric columns to pass through unchanged
    - n_bins: number of quantile bins
    Outputs:
    - ColumnTransformer (unfitted)
    Why this contract matters for reliable ML delivery:
    - Guarantees consistent preprocessing during training and inference.
    - Prevents leakage by returning a blueprint only.
    """

    print("Building feature preprocessing recipe...")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    transformers = []

    # Quantile binning (numeric features)
    if quantile_bin_cols:
        transformers.append(
            (
                "quantile_bin",
                KBinsDiscretizer(
                    n_bins=n_bins,
                    encode="ordinal",
                    strategy="quantile"
                ),
                quantile_bin_cols,
            )
        )

    # One-hot encoding (categorical features)
    if categorical_onehot_cols:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers.append(
            (
                "categorical_onehot",
                encoder,
                categorical_onehot_cols,
            )
        )

    # Numeric passthrough (raw numeric features)
    if numeric_passthrough_cols:
        transformers.append(
            (
                "numeric_passthrough",
                "passthrough",
                numeric_passthrough_cols,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    print("Warning: Student has not implemented custom feature logic yet")

    return preprocessor