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
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, FunctionTransformer
import numpy as np


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

    # --------------------------------------------------------
    # Business Tenure Risk Bucket (Domain-Based Feature)
    # --------------------------------------------------------
    def tenure_bucket(X):
        tenure = X.iloc[:, 0]
        return np.where(
            tenure < 6, 2,              # high churn risk
            np.where(tenure < 12, 1, 0) # medium / low risk
        ).reshape(-1, 1)

    if "tenure" in quantile_bin_cols:
        transformers.append(
            (
                "tenure_risk_bucket",
                FunctionTransformer(tenure_bucket, validate=False),
                ["tenure"],
            )
        )

    # --------------------------------------------------------
    # One-Hot Encoding (Categorical Variables)
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Numeric Passthrough
    # --------------------------------------------------------
    if numeric_passthrough_cols:
        transformers.append(
            (
                "numeric_passthrough",
                "passthrough",
                numeric_passthrough_cols,
            )
        )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add Telco-specific engineered features safely
    # Why: Feature engineering depends on dataset and business context.
    # Examples:
    # 1. Add revenue-per-tenure ratio
    # 2. Add contract-tenure interaction
    # 3. Add churn risk composite features

    telco_service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    if all(col in categorical_onehot_cols for col in telco_service_columns):
        def service_count(X):
            return X.apply(lambda row: (row == "Yes").sum(), axis=1).values.reshape(-1, 1)

        transformers.append(
            (
                "service_count",
                FunctionTransformer(service_count, validate=False),
                telco_service_columns,
            )
        )

    print("Warning: Student has not implemented additional custom feature logic")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # Build ColumnTransformer only once, after all transformers are defined
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor