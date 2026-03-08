"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
) -> ColumnTransformer:
    """
    Inputs:
    - quantile_bin_cols: numeric columns to apply quantile binning
    - categorical_onehot_cols: categorical columns for one-hot encoding
    - numeric_passthrough_cols: numeric columns to pass through unchanged
    - n_bins: number of quantile bins

    Output:
    - ColumnTransformer (unfitted)
    """

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    transformers = []

    # 1) Quantile binning for numeric columns
    if quantile_bin_cols:
        # NOTE: encode='onehot-dense' gives a numeric array output
        binner = KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="quantile",
        )
        transformers.append(("quantile_binning", binner, quantile_bin_cols))

    # 2) One-hot encoding for categorical columns
    if categorical_onehot_cols:
        # sklearn compatibility (older versions use sparse=False)
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers.append(("categorical_onehot", encoder, categorical_onehot_cols))

    # 3) Passthrough numeric columns
    if numeric_passthrough_cols:
        transformers.append(("numeric_passthrough", "passthrough", numeric_passthrough_cols))

    # Build the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor