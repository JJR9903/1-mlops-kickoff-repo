import pandas as pd
import numpy as np

from src.clean_data import clean_dataframe


def test_clean_dataframe_identity():
    """
    Test that clean_dataframe creates a deep copy and doesn't affect untouched columns.
    """
    df_raw = pd.DataFrame({
        "other_feature": [1, 2, 3],
        "target": ["Unknown", "Unknown", "Unknown"]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    # Should be a separate object
    assert df_clean is not df_raw
    # Content shouldn't change for unhandled columns/values
    pd.testing.assert_frame_equal(df_raw, df_clean)


def test_clean_dataframe_generalized_string_blanks():
    """
    Test that empty strings and strings of spaces are replaced by NaN.
    """
    df_raw = pd.DataFrame({
        "col1": ["a", " ", "", "a", "nan"], # 'a' is repeated so uniqueness isn't 100%
        "target": ["No", "No", "No", "No", "No"]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    expected_col1 = pd.Series(["a", np.nan, np.nan, "a", np.nan], name="col1")
    # the dtype might remain object for mixed strings, so we don't strictly assert dtype unless needed
    pd.testing.assert_series_equal(df_clean["col1"], expected_col1)


def test_clean_dataframe_generalized_numeric_conversion():
    """
    Test that a string column that's mostly numeric gets converted and filled.
    """
    # 3 valid numerics, 1 empty string (which becomes NaN then median)
    df_raw = pd.DataFrame({
        "mixed_num": ["10.5", " ", "20.5", "15.5"], 
        "target": ["Yes", "No", "Yes", "No"]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    # median of [10.5, 20.5, 15.5] is 15.5
    expected_charges = pd.Series([10.5, 15.5, 20.5, 15.5], name="mixed_num")
    
    pd.testing.assert_series_equal(df_clean["mixed_num"], expected_charges)


def test_clean_dataframe_drops_high_cardinality():
    """
    Test that columns with >90% unique string values are dropped (e.g. IDs).
    """
    df_raw = pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(100)], # 100% unique
        "feature1": [1] * 100,
        "target": ["Yes"] * 50 + ["No"] * 50
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    assert "id_col" not in df_clean.columns
    assert "feature1" in df_clean.columns


def test_clean_dataframe_target_mapping():
    """
    Test that target column is mapped from Yes/No/True/False to 1/0.
    """
    df_raw = pd.DataFrame({
        "target": ["Yes", "No", "False", "True", np.nan]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    expected_target = pd.Series([1.0, 0.0, 0.0, 1.0, np.nan], name="target", dtype="float64")
    
    pd.testing.assert_series_equal(df_clean["target"], expected_target)


def test_clean_dataframe_drops_multiple_ids():
    """
    Test that multiple columns with >90% unique string values are dropped.
    """
    df_raw = pd.DataFrame({
        "id_col_1": [f"ID_{i}" for i in range(100)], # 100% unique
        "id_col_2": [f"UUID_{i}" for i in range(100)], # 100% unique
        "feature1": [1, 2, 1, 2] * 25, # Not unique
        "target": ["Yes"] * 50 + ["No"] * 50
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    assert "id_col_1" not in df_clean.columns
    assert "id_col_2" not in df_clean.columns
    assert "feature1" in df_clean.columns

def test_clean_dataframe_target_mapping_ignores_other_values():
    """
    Test that target column mapping doesn't apply if values are not subset of acceptable booleans.
    """
    df_raw = pd.DataFrame({
        "my_target": ["Maybe", "No", "Yes"]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="my_target")
    
    pd.testing.assert_series_equal(df_raw["my_target"], df_clean["my_target"])

def test_clean_dataframe_handles_all_nans_gracefully():
    """
    Test that a dataframe with entirely null columns doesn't crash the logic.
    """
    df_raw = pd.DataFrame({
        "all_null_str": [np.nan, None, pd.NA],
        "all_null_num": [np.nan, np.nan, np.nan],
        "target": ["Yes", "No", "Yes"]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    # Target should still be handled
    expected_target = pd.Series([1.0, 0.0, 1.0], name="target", dtype="float64")
    pd.testing.assert_series_equal(df_clean["target"], expected_target)

    # Empty columns might stay in, but shouldn't have crashed median/to_numeric logic
    assert "all_null_str" in df_clean.columns
    assert "all_null_num" in df_clean.columns
