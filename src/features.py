"""
Educational Goal:
- Centralizes feature engineering logic to prevent data leakage
- Ensures consistent preprocessing between training and inference
- Returns an unfitted ColumnTransformer blueprint only
"""

from typing import Optional, List
import logging
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# --------------------------------------------------------
# Logging configuration
# --------------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------------
# Safe numeric casting
# --------------------------------------------------------
def safe_numeric_cast(X):
    """
    Safely convert numeric-like columns to numeric.
    Coerces invalid values (e.g. blanks) to NaN.
    """
    return X.apply(lambda col: pd.to_numeric(col, errors="coerce"))


# --------------------------------------------------------
# Module-level transform functions (required for pickle compatibility)
# Defining these inside get_feature_preprocessor() makes them local closures
# that joblib/pickle cannot serialize — they must live at module scope.
# --------------------------------------------------------
def tenure_bucket(X):
    """
    Domain-based risk bucket for tenure:
      < 6 months  -> 2 (high churn risk)
      6-11 months -> 1 (medium risk)
      12+ months  -> 0 (low risk)
    """
    tenure = X.iloc[:, 0]
    return np.where(
        tenure < 6, 2,
        np.where(tenure < 12, 1, 0)
    ).reshape(-1, 1)


def service_count(X):
    """
    Counts how many Telco add-on services a customer has subscribed to ('Yes').
    Produces a single engineered numeric feature.
    """
    return X.apply(lambda row: (row == "Yes").sum(), axis=1).values.reshape(-1, 1)

# --------------------------------------------------------
# Feature Preprocessor Builder
# --------------------------------------------------------
def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
):
    """
    Inputs:
    - quantile_bin_cols: numeric columns to apply domain-based transformations
    - categorical_onehot_cols: categorical columns for one-hot encoding
    - numeric_passthrough_cols: numeric columns to pass through (with imputation)

    Outputs:
    - Unfitted ColumnTransformer
    """

    logger.info("Building feature preprocessing recipe...")

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    transformers = []

    # --------------------------------------------------------
    # 1 Numeric Pipeline (Safe Casting + Mean Imputation + Scaling)
    # --------------------------------------------------------
    if numeric_passthrough_cols:
        numeric_pipeline = Pipeline(steps=[
            ("cast_numeric", FunctionTransformer(safe_numeric_cast, validate=False)),
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_passthrough_cols,
            )
        )

    # --------------------------------------------------------
    # 2 Domain-Based Tenure Risk Bucket
    # --------------------------------------------------------
    if "tenure" in quantile_bin_cols:
        transformers.append(
            (
                "tenure_risk_bucket",
                FunctionTransformer(tenure_bucket, validate=False),
                ["tenure"],
            )
        )

    # --------------------------------------------------------
    # 3 Categorical Pipeline (Imputation + OneHot)
    # --------------------------------------------------------
    if categorical_onehot_cols:

        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder)
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_onehot_cols,
            )
        )

    # --------------------------------------------------------
    # 4 Telco-Specific Engineered Feature
    # --------------------------------------------------------
    telco_service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    if all(col in categorical_onehot_cols for col in telco_service_columns):

        transformers.append(
            (
                "service_count",
                FunctionTransformer(service_count, validate=False),
                telco_service_columns,
            )
        )

    # --------------------------------------------------------
    # Final ColumnTransformer
    # --------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor