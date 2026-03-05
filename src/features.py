"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.

"""

from typing import Optional, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3
):
    """
    Inputs:
    - quantile_bin_cols: numeric columns for quantile binning
    - categorical_onehot_cols: categorical columns for one-hot encoding
    - numeric_passthrough_cols: numeric columns to pass through
    - n_bins: number of bins for KBinsDiscretizer
    Outputs:
    - ColumnTransformer (unfitted)
    Why this contract matters for reliable ML delivery:
    - Guarantees identical preprocessing at train and inference time.
    - Prevents leakage by returning a blueprint only.
    """

    print("Building Telco feature preprocessing recipe...")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    transformers = []

    # Quantile binning pipeline
    if quantile_bin_cols:
        transformers.append(
            (
                "quantile_bin",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("kbins", KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="ordinal",
                        strategy="quantile"
                    ))
                ]),
                quantile_bin_cols
            )
        )

    # One-hot encoding pipeline
    if categorical_onehot_cols:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        transformers.append(
            (
                "categorical_onehot",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", encoder)
                ]),
                categorical_onehot_cols
            )
        )

    # Numeric passthrough
    if numeric_passthrough_cols:
        transformers.append(
            (
                "numeric_passthrough",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_passthrough_cols
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    print("Warning: Custom Telco feature logic not implemented yet")

    return preprocessor