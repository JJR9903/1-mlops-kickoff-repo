import pytest
from unittest.mock import MagicMock
import pandas as pd
from pathlib import Path

from src.main import main


@pytest.fixture
def mock_pipeline(monkeypatch):
    """
    Mocks all external data loading, processing, model training and saving
    functions used inside main() to ensure fast, isolated tests without
    triggering the real ML pipeline operations.
    """
    dummy_df = pd.DataFrame({
        "num_feature": [1.0, 2.0, 3.0],
        "cat_feature": ["A", "B", "A"],
        "target": [0, 1, 0]
    })
    dummy_predictions = pd.DataFrame({"prediction": [0, 1]})

    monkeypatch.setattr("src.main.load_raw_data", MagicMock(return_value=dummy_df))
    monkeypatch.setattr("src.main.clean_dataframe", MagicMock(return_value=dummy_df))
    monkeypatch.setattr("src.main.validate_dataframe", MagicMock())
    monkeypatch.setattr("src.main.train_test_split", MagicMock(return_value=(dummy_df, dummy_df, dummy_df["target"], dummy_df["target"])))
    monkeypatch.setattr("src.main.get_feature_preprocessor", MagicMock(return_value="dummy_preprocessor"))
    monkeypatch.setattr("src.main.train_model", MagicMock(return_value="dummy_model"))
    monkeypatch.setattr("src.main.save_model", MagicMock())
    monkeypatch.setattr("src.main.evaluate_model", MagicMock(return_value=0.95))
    monkeypatch.setattr("src.main.run_inference", MagicMock(return_value=dummy_predictions))
    monkeypatch.setattr("src.main.save_csv", MagicMock())
    
    # Mock Path.mkdir and Path.exists to prevent any directory creation or avoiding the alternative telco data path finding.
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(Path, "exists", lambda self: False)


def test_main_runs_end_to_end_with_mocks(mock_pipeline):
    """
    Test that the main orchestrator runs completely end-to-end 
    no exceptions are raised when all its external functions are lightweight mocks.
    """
    # Act / Assert
    main()


def test_main_creates_required_directories(mock_pipeline, monkeypatch):
    """
    Verify that main() ensures required directories are created by tracking 
    calls to Path.mkdir: 'data/raw', 'data/processed', 'models', 'reports'.
    """
    mkdir_calls = []
    
    def mock_mkdir(self, parents=False, exist_ok=False):
        # Normalizing to forward slashes for cross-platform validation
        mkdir_calls.append(str(self).replace("\\", "/"))
        
    monkeypatch.setattr(Path, "mkdir", mock_mkdir)
    
    # Act
    main()
    
    # Assert
    assert any("data/raw" in call for call in mkdir_calls)
    assert any("data/processed" in call for call in mkdir_calls)
    assert any("models" in call for call in mkdir_calls)
    assert any("reports" in call for call in mkdir_calls)


def test_main_raises_error_when_feature_missing(mock_pipeline, monkeypatch):
    """
    Test that main orchestrator's fail-fast logic correctly raises a ValueError
    when a configured feature is missing from the features DataFrame.
    """
    # Arrange: Create target DataFrame missing the configured 'num_feature' column
    bad_df = pd.DataFrame({
        "cat_feature": ["A", "B", "A"],
        "target": [0, 1, 0]
    }) 
    
    monkeypatch.setattr("src.main.clean_dataframe", MagicMock(return_value=bad_df))
    
    # Act / Assert
    with pytest.raises(ValueError, match="Configured feature columns missing in X"):
        main()
