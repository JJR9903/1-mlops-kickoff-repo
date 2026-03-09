"""
Unit tests for validate.validate_dataframe()

Run with:
    pytest test_validate.py -v

Each test class maps to one logical concern in the function.
Fixtures are defined at module level and shared across test classes.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.validate import validate_dataframe


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_clf_df():
    """Minimal clean DataFrame for a binary classification task (Churn)."""
    return pd.DataFrame({
        "customerID":    ["A001", "A002", "A003"],
        "tenure":        [12, 24, 6],
        "MonthlyCharges":[50.0, 75.0, 30.0],
        "TotalCharges":  [600.0, 1800.0, 180.0],
        "Churn":         [0, 1, 0],
    })


@pytest.fixture
def valid_reg_df():
    """Minimal clean DataFrame for a regression task (HousePrice)."""
    return pd.DataFrame({
        "sqft":       [1200, 850, 2000],
        "bedrooms":   [3, 2, 4],
        "HousePrice": [250_000.0, 175_000.0, 400_000.0],
    })


@pytest.fixture
def clf_schema():
    """Schema matching valid_clf_df (no imputable NaNs)."""
    return {
        "customerID":     {"type": "categorical", "accept_nan": False},
        "tenure":         {"type": "numeric",     "accept_nan": False},
        "MonthlyCharges": {"type": "numeric",     "accept_nan": False},
        "TotalCharges":   {"type": "numeric",     "accept_nan": False},
        "Churn":          {"type": "numeric",     "accept_nan": False},
    }


@pytest.fixture
def reg_schema():
    """Schema matching valid_reg_df."""
    return {
        "sqft":       {"type": "numeric", "accept_nan": False},
        "bedrooms":   {"type": "numeric", "accept_nan": False},
        "HousePrice": {"type": "numeric", "accept_nan": False},
    }


@pytest.fixture
def clf_target():
    """target_config for binary classification."""
    return {
        "column":          "Churn",
        "type":            "classification",
        "allowed_classes": [0, 1],
    }


@pytest.fixture
def reg_target():
    """target_config for regression with an explicit range."""
    return {
        "column": "HousePrice",
        "type":   "regression",
        "range":  [0, 1_000_000],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Happy-path tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHappyPath:
    """Valid DataFrames should return True with no errors raised."""

    def test_classification_returns_true(self, valid_clf_df, clf_schema, clf_target):
        assert validate_dataframe(valid_clf_df, clf_schema, clf_target) is True

    def test_regression_returns_true(self, valid_reg_df, reg_schema, reg_target):
        assert validate_dataframe(valid_reg_df, reg_schema, reg_target) is True

    def test_classification_without_allowed_classes(self, valid_clf_df, clf_schema):
        """allowed_classes is optional — omitting it should not raise."""
        target = {"column": "Churn", "type": "classification"}
        assert validate_dataframe(valid_clf_df, clf_schema, target) is True

    def test_regression_without_range(self, valid_reg_df, reg_schema):
        """range is optional — omitting it should not raise."""
        target = {"column": "HousePrice", "type": "regression"}
        assert validate_dataframe(valid_reg_df, reg_schema, target) is True

    def test_imputable_nan_does_not_raise(self, clf_schema, clf_target):
        """Columns with accept_nan=True should be allowed to have NaNs."""
        df = pd.DataFrame({
            "customerID":     ["A001", "A002", "A003"],
            "tenure":         [12, 24, 6],
            "MonthlyCharges": [50.0, None, 30.0],   # NaN here
            "TotalCharges":   [600.0, 1800.0, 180.0],
            "Churn":          [0, 1, 0],
        })
        schema = clf_schema.copy()
        schema["MonthlyCharges"] = {"type": "numeric", "accept_nan": True}  # allow it
        assert validate_dataframe(df, schema, clf_target) is True


# ─────────────────────────────────────────────────────────────────────────────
# target_config validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetConfig:
    """Malformed target_config should fail before any DataFrame checks."""

    def test_missing_column_key_raises(self, valid_clf_df, clf_schema):
        bad_config = {"type": "classification"}
        with pytest.raises(ValueError, match="missing required key 'column'"):
            validate_dataframe(valid_clf_df, clf_schema, bad_config)

    def test_invalid_type_raises(self, valid_clf_df, clf_schema):
        bad_config = {"column": "Churn", "type": "clustering"}
        with pytest.raises(ValueError, match="must be one of"):
            validate_dataframe(valid_clf_df, clf_schema, bad_config)

    def test_target_column_not_in_df_raises(self, valid_clf_df, clf_schema):
        bad_config = {"column": "NonExistentCol", "type": "classification"}
        with pytest.raises(ValueError, match="not found in DataFrame"):
            validate_dataframe(valid_clf_df, clf_schema, bad_config)


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Empty DataFrame
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyDataFrame:

    def test_empty_df_raises(self, clf_schema, clf_target):
        empty_df = pd.DataFrame(columns=["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"])
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe(empty_df, clf_schema, clf_target)


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — Schema: presence, dtype kind, NaN policy
# ─────────────────────────────────────────────────────────────────────────────

class TestSchema:

    def test_missing_column_raises(self, valid_clf_df, clf_target):
        """Schema referencing a column absent from the df should raise."""
        schema_with_extra = {
            "customerID":     {"type": "categorical", "accept_nan": False},
            "tenure":         {"type": "numeric",     "accept_nan": False},
            "MonthlyCharges": {"type": "numeric",     "accept_nan": False},
            "TotalCharges":   {"type": "numeric",     "accept_nan": False},
            "Churn":          {"type": "numeric",     "accept_nan": False},
            "GhostColumn":    {"type": "numeric",     "accept_nan": False},  # not in df
        }
        with pytest.raises(ValueError, match="MISSING"):
            validate_dataframe(valid_clf_df, schema_with_extra, clf_target)

    def test_wrong_dtype_raises(self, valid_clf_df, clf_target):
        """Declaring a numeric column as categorical should raise a WRONG TYPE error."""
        bad_schema = {
            "customerID":     {"type": "categorical", "accept_nan": False},
            "tenure":         {"type": "categorical", "accept_nan": False},  # wrong — it's numeric
            "MonthlyCharges": {"type": "numeric",     "accept_nan": False},
            "TotalCharges":   {"type": "numeric",     "accept_nan": False},
            "Churn":          {"type": "numeric",     "accept_nan": False},
        }
        with pytest.raises(ValueError, match="WRONG TYPE"):
            validate_dataframe(valid_clf_df, bad_schema, clf_target)

    def test_structural_nan_raises(self, clf_schema, clf_target):
        """NaN in a column with accept_nan=False should raise."""
        df = pd.DataFrame({
            "customerID":     ["A001", "A002", "A003"],
            "tenure":         [12, None, 6],          # NaN in non-nullable column
            "MonthlyCharges": [50.0, 75.0, 30.0],
            "TotalCharges":   [600.0, 1800.0, 180.0],
            "Churn":          [0, 1, 0],
        })
        with pytest.raises(ValueError, match="UNEXPECTED NaN"):
            validate_dataframe(df, clf_schema, clf_target)

    def test_unknown_dtype_kind_raises(self, valid_clf_df, clf_target):
        """Using an unsupported dtype kind in schema should raise immediately."""
        bad_schema = {
            "customerID":     {"type": "boolean",  "accept_nan": False},  # unsupported kind
            "tenure":         {"type": "numeric",  "accept_nan": False},
            "MonthlyCharges": {"type": "numeric",  "accept_nan": False},
            "TotalCharges":   {"type": "numeric",  "accept_nan": False},
            "Churn":          {"type": "numeric",  "accept_nan": False},
        }
        with pytest.raises(ValueError, match="Unknown dtype kind"):
            validate_dataframe(valid_clf_df, bad_schema, clf_target)


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — Target validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetValidation:

    # ── Classification ────────────────────────────────────────────────────────

    def test_unexpected_class_raises(self, clf_schema, clf_target):
        """Target containing a class outside allowed_classes should raise."""
        df = pd.DataFrame({
            "customerID":     ["A001", "A002", "A003"],
            "tenure":         [12, 24, 6],
            "MonthlyCharges": [50.0, 75.0, 30.0],
            "TotalCharges":   [600.0, 1800.0, 180.0],
            "Churn":          [0, 1, 2],   # 2 is unexpected
        })
        with pytest.raises(ValueError, match="unexpected class"):
            validate_dataframe(df, clf_schema, clf_target)

    def test_valid_classes_pass(self, valid_clf_df, clf_schema, clf_target):
        """Target with only [0, 1] and allowed_classes=[0, 1] should pass."""
        assert validate_dataframe(valid_clf_df, clf_schema, clf_target) is True

    # ── Regression ───────────────────────────────────────────────────────────

    def test_regression_non_numeric_target_raises(self, reg_schema, reg_target):
        """Regression target that is a string column should raise."""
        df = pd.DataFrame({
            "sqft":       [1200, 850, 2000],
            "bedrooms":   [3, 2, 4],
            "HousePrice": ["cheap", "mid", "expensive"],  # string — wrong for regression
        })
        with pytest.raises(ValueError, match="must be numeric|WRONG TYPE"):
            validate_dataframe(df, reg_schema, reg_target)

    def test_regression_out_of_range_raises(self, valid_reg_df, reg_schema):
        """Regression target with a value beyond the declared range should raise."""
        df = valid_reg_df.copy()
        df.loc[0, "HousePrice"] = 5_000_000   # above range [0, 1_000_000]
        target = {"column": "HousePrice", "type": "regression", "range": [0, 1_000_000]}
        with pytest.raises(ValueError, match="outside expected range"):
            validate_dataframe(df, reg_schema, target)

    def test_regression_within_range_passes(self, valid_reg_df, reg_schema, reg_target):
        assert validate_dataframe(valid_reg_df, reg_schema, reg_target) is True


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — No fully-NaN rows
# ─────────────────────────────────────────────────────────────────────────────

class TestFullyNanRows:

    def test_fully_nan_row_raises(self, clf_schema, clf_target):
        df = pd.DataFrame({
            "customerID":     ["A001", np.nan,  "A003"],
            "tenure":         [12,     np.nan,  6],
            "MonthlyCharges": [50.0,   np.nan,  30.0],
            "TotalCharges":   [600.0,  np.nan,  180.0],
            "Churn":          [0,      np.nan,  0],
        })
        with pytest.raises(ValueError, match="NaN in every column"):
            validate_dataframe(df, clf_schema, clf_target)


# ─────────────────────────────────────────────────────────────────────────────
# Check 6 — No negative values in numeric columns
# ─────────────────────────────────────────────────────────────────────────────

class TestNegativeValues:

    def test_negative_numeric_raises(self, clf_schema, clf_target):
        df = pd.DataFrame({
            "customerID":     ["A001", "A002", "A003"],
            "tenure":         [12, -5, 6],     # negative tenure — impossible
            "MonthlyCharges": [50.0, 75.0, 30.0],
            "TotalCharges":   [600.0, 1800.0, 180.0],
            "Churn":          [0, 1, 0],
        })
        with pytest.raises(ValueError, match="Negative values"):
            validate_dataframe(df, clf_schema, clf_target)

    def test_zero_is_allowed(self, clf_schema, clf_target):
        """Zero should not trigger the negative check."""
        df = pd.DataFrame({
            "customerID":     ["A001", "A002", "A003"],
            "tenure":         [0, 24, 6],      # zero tenure is valid (new customer)
            "MonthlyCharges": [50.0, 75.0, 30.0],
            "TotalCharges":   [0.0, 1800.0, 180.0],
            "Churn":          [0, 1, 0],
        })
        assert validate_dataframe(df, clf_schema, clf_target) is True