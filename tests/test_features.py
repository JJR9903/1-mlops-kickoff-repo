"""
Educational Goal:
- Why this module exists in an MLOps system: Validate that feature engineering logic builds correctly and is safe to integrate.
- Responsibility (separation of concerns): Ensure preprocessing blueprint is correctly constructed.
- Pipeline contract (inputs and outputs): Configuration lists in → ColumnTransformer out.

TODO: Replace print statements with logging in later session
TODO: Expand tests when custom feature logic is implemented
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.compose import ColumnTransformer

from src.features import get_feature_preprocessor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def test_returns_column_transformer():
    """
    Inputs:
    - Simple feature configuration
    Outputs:
    - ColumnTransformer instance
    Why this contract matters for reliable ML delivery:
    - Ensures blueprint object is constructed correctly.
    """
    print("Testing that get_feature_preprocessor returns ColumnTransformer...")  # TODO: replace with logging later

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["num_feature"],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=[]
    )

    assert isinstance(preprocessor, ColumnTransformer)


def test_transformer_handles_dummy_dataframe():
    """
    Inputs:
    - Small dummy dataset
    Outputs:
    - Transformed numpy array without crashing
    Why this contract matters for reliable ML delivery:
    - Ensures feature blueprint can be fitted safely inside Pipeline.
    """

    print("Testing transformation on dummy dataset...")  # TODO: replace with logging later

    df = pd.DataFrame({
        "num_feature": [1, 2, 3, 4],
        "cat_feature": ["A", "B", "A", "B"]
    })

    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["num_feature"],
        categorical_onehot_cols=["cat_feature"],
        numeric_passthrough_cols=[]
    )

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 4


def test_no_columns_provided():
    """
    Inputs:
    - Empty configuration
    Outputs:
    - ColumnTransformer without transformers
    Why this contract matters for reliable ML delivery:
    - Prevents silent failures when config is empty.
    """

    print("Testing behavior with empty column lists...")  # TODO: replace with logging later

    preprocessor = get_feature_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)
    
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns),
        # keep any other existing transformers
    ]
)