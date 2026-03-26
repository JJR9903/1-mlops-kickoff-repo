"""
Tests for W&B integration in the main pipeline.

These tests verify that wandb.init, wandb.log, wandb.log_artifact,
and wandb.finish are called correctly during pipeline execution.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np


@pytest.fixture
def mock_config():
    """Minimal config matching config.yaml structure."""
    return {
        "project": {"name": "mlops-kickoff"},
        "paths": {
            "raw_data": "data/raw/test.csv",
            "processed_data": "data/processed/clean.csv",
            "model_path": "models/model.joblib",
            "predictions_path": "reports/predictions.csv",
        },
        "ml": {
            "problem_type": "classification",
            "test_size": 0.2,
            "random_state": 42,
        },
        "wandb": {
            "project": "test-project",
            "job_type": "test-job",
            "model_artifact_name": "test-model",
            "model_alias": "prod",
        },
        "features": {
            "quantile_bin": ["tenure"],
            "categorical_onehot": ["gender"],
            "numeric_passthrough": ["MonthlyCharges"],
        },
        "schema": {
            "tenure": {"type": "numeric", "accept_nan": False},
            "gender": {"type": "categorical", "accept_nan": False},
            "MonthlyCharges": {"type": "numeric", "accept_nan": False},
        },
        "target_config": {
            "column": "Churn",
            "type": "classification",
            "allowed_classes": [0, 1],
        },
    }


@pytest.fixture
def sample_dataframe():
    """Small DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "MonthlyCharges": np.random.uniform(20, 100, n),
        "Churn": np.random.choice([0, 1], n),
    })


@patch("src.main.wandb")
@patch("src.main.load_config")
@patch("src.main.load_data")
@patch("src.main.clean_dataframe")
@patch("src.main.validate_dataframe")
@patch("src.main.get_feature_preprocessor")
@patch("src.main.train_model")
@patch("src.main.evaluate_model")
@patch("src.main.save_model")
@patch("src.main.save_csv")
def test_wandb_init_called_with_config(
    mock_save_csv,
    mock_save_model,
    mock_evaluate,
    mock_train,
    mock_preprocessor,
    mock_validate,
    mock_clean,
    mock_load_data,
    mock_load_config,
    mock_wandb,
    mock_config,
    sample_dataframe,
):
    """Test that wandb.init is called with the correct project and job_type."""
    mock_load_config.return_value = mock_config
    mock_load_data.return_value = sample_dataframe
    mock_clean.return_value = sample_dataframe

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1] * 10)
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]] * 10)
    mock_train.return_value = mock_model
    mock_evaluate.return_value = 0.85

    mock_run = MagicMock()
    mock_run.name = "test-run"
    mock_run.url = "https://wandb.ai/test/run"
    mock_wandb.init.return_value = mock_run

    from src.main import main
    main()

    # Verify wandb.init was called with correct project
    mock_wandb.init.assert_called_once()
    init_kwargs = mock_wandb.init.call_args[1]
    assert init_kwargs["project"] == "test-project"
    assert init_kwargs["job_type"] == "test-job"
    assert "problem_type" in init_kwargs["config"]


@patch("src.main.wandb")
@patch("src.main.load_config")
@patch("src.main.load_data")
@patch("src.main.clean_dataframe")
@patch("src.main.validate_dataframe")
@patch("src.main.get_feature_preprocessor")
@patch("src.main.train_model")
@patch("src.main.evaluate_model")
@patch("src.main.save_model")
@patch("src.main.save_csv")
def test_wandb_logs_data_stats(
    mock_save_csv,
    mock_save_model,
    mock_evaluate,
    mock_train,
    mock_preprocessor,
    mock_validate,
    mock_clean,
    mock_load_data,
    mock_load_config,
    mock_wandb,
    mock_config,
    sample_dataframe,
):
    """Test that data statistics are logged to W&B."""
    mock_load_config.return_value = mock_config
    mock_load_data.return_value = sample_dataframe
    mock_clean.return_value = sample_dataframe

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1] * 10)
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]] * 10)
    mock_train.return_value = mock_model
    mock_evaluate.return_value = 0.85

    mock_run = MagicMock()
    mock_run.name = "test-run"
    mock_run.url = "https://wandb.ai/test/run"
    mock_wandb.init.return_value = mock_run

    from src.main import main
    main()

    # Verify data stats were logged
    mock_wandb.log.assert_any_call({
        "data/raw_rows": 100,
        "data/raw_columns": 4,
    })


@patch("src.main.wandb")
@patch("src.main.load_config")
@patch("src.main.load_data")
@patch("src.main.clean_dataframe")
@patch("src.main.validate_dataframe")
@patch("src.main.get_feature_preprocessor")
@patch("src.main.train_model")
@patch("src.main.evaluate_model")
@patch("src.main.save_model")
@patch("src.main.save_csv")
def test_wandb_model_artifact_uploaded(
    mock_save_csv,
    mock_save_model,
    mock_evaluate,
    mock_train,
    mock_preprocessor,
    mock_validate,
    mock_clean,
    mock_load_data,
    mock_load_config,
    mock_wandb,
    mock_config,
    sample_dataframe,
):
    """Test that the model artifact is uploaded to W&B."""
    mock_load_config.return_value = mock_config
    mock_load_data.return_value = sample_dataframe
    mock_clean.return_value = sample_dataframe

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1] * 10)
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]] * 10)
    mock_train.return_value = mock_model
    mock_evaluate.return_value = 0.85

    mock_run = MagicMock()
    mock_run.name = "test-run"
    mock_run.url = "https://wandb.ai/test/run"
    mock_wandb.init.return_value = mock_run

    from src.main import main
    main()

    # Verify artifact was created and logged
    mock_wandb.Artifact.assert_called_once_with(
        name="test-model",
        type="model",
        description="Trained classification pipeline",
    )
    mock_wandb.log_artifact.assert_called_once()


@patch("src.main.wandb")
@patch("src.main.load_config")
@patch("src.main.load_data")
@patch("src.main.clean_dataframe")
@patch("src.main.validate_dataframe")
@patch("src.main.get_feature_preprocessor")
@patch("src.main.train_model")
@patch("src.main.evaluate_model")
@patch("src.main.save_model")
@patch("src.main.save_csv")
def test_wandb_eval_metrics_logged(
    mock_save_csv,
    mock_save_model,
    mock_evaluate,
    mock_train,
    mock_preprocessor,
    mock_validate,
    mock_clean,
    mock_load_data,
    mock_load_config,
    mock_wandb,
    mock_config,
    sample_dataframe,
):
    """Test that evaluation metrics are logged to W&B."""
    mock_load_config.return_value = mock_config
    mock_load_data.return_value = sample_dataframe
    mock_clean.return_value = sample_dataframe

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1] * 10)
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]] * 10)
    mock_train.return_value = mock_model
    mock_evaluate.return_value = 0.85

    mock_run = MagicMock()
    mock_run.name = "test-run"
    mock_run.url = "https://wandb.ai/test/run"
    mock_wandb.init.return_value = mock_run

    from src.main import main
    main()

    # Verify eval metric was logged
    mock_wandb.log.assert_any_call({"eval/f1": 0.85})


@patch("src.main.wandb")
@patch("src.main.load_config")
@patch("src.main.load_data")
def test_wandb_finish_called_on_failure(
    mock_load_data,
    mock_load_config,
    mock_wandb,
    mock_config,
):
    """Test that wandb.finish() is called even when the pipeline fails."""
    mock_load_config.return_value = mock_config
    mock_load_data.side_effect = FileNotFoundError("Data not found")

    mock_run = MagicMock()
    mock_run.name = "test-run"
    mock_run.url = "https://wandb.ai/test/run"
    mock_wandb.init.return_value = mock_run

    from src.main import main
    with pytest.raises(SystemExit):
        main()

    # wandb.finish must be called even on failure
    mock_wandb.finish.assert_called_once()
