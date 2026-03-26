"""import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.infer import download_prod_model, run_inference

@pytest.fixture
def mock_config():
    return {
        "wandb": {
            "project": "test-project",
            "model_artifact_name": "test-model",
            "model_alias": "prod"
        }
    }

@patch("src.infer.wandb.Api")
@patch("src.infer.joblib.load")
@patch("src.infer.load_dotenv")
def test_download_prod_model_success(mock_dotenv, mock_joblib_load, mock_wandb_api, mock_config):
    # Setup mocks
    mock_api_instance = MagicMock()
    mock_wandb_api.return_value = mock_api_instance
    mock_api_instance.default_entity = "test-entity"
    
    mock_artifact = MagicMock()
    mock_api_instance.artifact.return_value = mock_artifact
    mock_artifact.download.return_value = "mock_dir"
    
    mock_model = MagicMock()
    mock_joblib_load.return_value = mock_model
    
    # Execute
    model = download_prod_model(config=mock_config)
    
    # Assertions
    mock_wandb_api.assert_called_once()
    mock_api_instance.artifact.assert_called_with("test-entity/test-project/test-model:prod")
    mock_artifact.download.assert_called_once()
    mock_joblib_load.assert_called_once_with(os.path.join("mock_dir", "model.joblib"))
    assert model == mock_model

@patch("src.infer.download_prod_model")
def test_run_inference_success(mock_download, mock_config):
    # Setup mocks
    mock_model = MagicMock()
    mock_download.return_value = mock_model
    
    # Mock model prediction methods
    mock_model.predict.return_value = np.array([0, 1])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    
    input_data = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    
    # Execute
    result = run_inference(input_data, config=mock_config)
    
    # Assertions
    mock_download.assert_called_once_with(mock_config)
    mock_model.predict.assert_called_once_with(input_data)
    assert result["predictions"] == [0, 1]
    assert result["probabilities"] == [0.2, 0.7]

@patch("src.infer.wandb.Api")
def test_download_prod_model_missing_config_keys(mock_wandb_api):
    # Edge case: config missing required keys
    incomplete_config = {"wandb": {"project": "test-project"}}
    
    with pytest.raises(KeyError):
        download_prod_model(config=incomplete_config)

@patch("src.infer.wandb.Api")
def test_download_prod_model_api_error(mock_wandb_api, mock_config):
    # Edge case: W&B API error
    mock_api_instance = MagicMock()
    mock_wandb_api.return_value = mock_api_instance
    mock_api_instance.artifact.side_effect = Exception("Artifact not found")
    
    with pytest.raises(Exception, match="Artifact not found"):
        download_prod_model(config=mock_config)
        
   From LUKA but approved by Romain     
"""
import pandas as pd
import numpy as np
from src.infer import run_inference


class MockModelWithProba:
    def predict(self, X):
        return np.array([0, 1])

    def predict_proba(self, X):
        # Mock probabilities for a binary classifier (class 0, class 1)
        return np.array([[0.8, 0.2], [0.3, 0.7]])


class MockModelNoProba:
    def predict(self, X):
        return np.array([1, 1])


def test_run_inference_with_proba():
    model = MockModelWithProba()
    X = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]}, index=["A", "B"])

    out = run_inference(model, X, include_proba=True)

    # Check predictions
    assert "prediction" in out.columns
    assert list(out["prediction"]) == [0, 1]

    # Check probabilities (should take class 1 proba)
    assert "proba" in out.columns
    assert list(out["proba"]) == [0.2, 0.7]

    # Check index retention
    assert list(out.index) == ["A", "B"]


def test_run_inference_without_include_proba():
    model = MockModelWithProba()
    X = pd.DataFrame({"feat1": [1, 2]}, index=["A", "B"])

    out = run_inference(model, X, include_proba=False)

    assert "prediction" in out.columns
    assert "proba" not in out.columns


def test_run_inference_model_without_predict_proba():
    model = MockModelNoProba()
    X = pd.DataFrame({"feat1": [1, 2]}, index=["A", "B"])

    # Even if include_proba=True, it shouldn't fail if model lacks the method
    out = run_inference(model, X, include_proba=True)

    assert "prediction" in out.columns
    assert list(out["prediction"]) == [1, 1]
    assert "proba" not in out.columns
