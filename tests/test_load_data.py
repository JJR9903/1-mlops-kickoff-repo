import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.load_data import load_raw_data
import src.utils


def test_load_existing_csv(tmp_path):
    """
    Test 1: Loads existing CSV and returns correct shape/columns.
    """
    # Setup: create a valid CSV file in tmp_path
    raw_data_path = tmp_path / "existing.csv"
    df_expected = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    src.utils.save_csv(df_expected, raw_data_path)

    # Action
    df_result = load_raw_data(raw_data_path)

    # Assert
    assert df_result.shape == (2, 2)
    assert list(df_result.columns) == ["col1", "col2"]


def test_load_missing_csv_creates_dummy(tmp_path):
    """
    Test 2: When file missing, creates dummy CSV and returns it.
    Assert file exists and has expected columns.
    """
    # Setup: Path that does not exist yet
    raw_data_path = tmp_path / "missing.csv"

    # Action
    df_result = load_raw_data(raw_data_path)

    # Assert
    assert raw_data_path.exists()
    assert list(df_result.columns) == ["num_feature", "cat_feature", "target"]
    assert df_result.shape == (4, 3)


def test_load_alt_path_copying(tmp_path):
    """
    Test 3: Mock alt_path behavior so if raw path missing and "alt exists", it copies and saves.
    Assert save_csv called and output columns match the alt dataframe.
    """
    # Setup
    raw_data_path = tmp_path / "test_alt.csv"
    df_alt = pd.DataFrame({"alt_col": ["a", "b"]})

    original_exists = Path.exists

    def mock_exists(self, *args, **kwargs):
        # Pretend the alternate path exists
        if self.name == "WA_Fn-UseC_-Telco-Customer-Churn.csv":
            return True
        return original_exists(self, *args, **kwargs)

    # We patch Path.exists to mock only the alt_path check.
    # We patch pd.read_csv to return our dummy alt dataframe.
    # We patch save_csv with wraps to ensure it actually executes while allowing us to assert it was called.
    with patch.object(Path, "exists", side_effect=mock_exists, autospec=True):
        with patch("src.load_data.pd.read_csv", return_value=df_alt):
            with patch("src.load_data.save_csv", wraps=src.utils.save_csv) as mock_save_csv:
                
                # Action
                df_result = load_raw_data(raw_data_path)

                # Assert that save_csv was called
                mock_save_csv.assert_called()
                
                # Assert that output matches alt dataframe
                assert list(df_result.columns) == ["alt_col"]
                assert df_result.shape == (2, 1)
