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


def test_clean_dataframe_total_charges():
    """
    Test that 'TotalCharges' is converted to numeric, " " replaced with NaN, and filled with median.
    """
    df_raw = pd.DataFrame({
        "TotalCharges": ["10.5", " ", "20.5", None],
        "target": ["Yes", "No", "Yes", "No"]
    })
    
    # The valid numerical values are 10.5 and 20.5. Median is 15.5.
    # " " and None become NaN, then filled with 15.5.
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    expected_charges = pd.Series([10.5, 15.5, 20.5, 15.5], name="TotalCharges")
    
    pd.testing.assert_series_equal(df_clean["TotalCharges"], expected_charges)


def test_clean_dataframe_drops_customer_id():
    """
    Test that 'customerID' is dropped.
    """
    df_raw = pd.DataFrame({
        "customerID": ["ID1", "ID2"],
        "feature1": [1, 2],
        "target": ["Yes", "No"]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    assert "customerID" not in df_clean.columns
    assert "feature1" in df_clean.columns


def test_clean_dataframe_target_mapping():
    """
    Test that target column is mapped from Yes/No to 1/0.
    """
    df_raw = pd.DataFrame({
        "target": ["Yes", "No", "No", "Yes", np.nan]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="target")
    
    # "Yes" -> 1.0, "No" -> 0.0, np.nan -> np.nan -> mapped properly usually implies float when NaN is present 
    # But note: .map() with NaN might leave it as NaN, converting dtype to float64
    expected_target = pd.Series([1.0, 0.0, 0.0, 1.0, np.nan], name="target", dtype="float64")
    
    pd.testing.assert_series_equal(df_clean["target"], expected_target)


def test_clean_dataframe_target_mapping_ignores_other_values():
    """
    Test that target column mapping doesn't apply if values are not subset of Yes/No.
    """
    df_raw = pd.DataFrame({
        "my_target": ["Maybe", "No", "Yes"]
    })
    
    df_clean = clean_dataframe(df_raw, target_column="my_target")
    
    # Should remain unchanged since it's not a subset of {"Yes", "No"}
    pd.testing.assert_series_equal(df_raw["my_target"], df_clean["my_target"])
