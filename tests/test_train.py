"""
Tests for src/train.py
======================
Covers:
  - _validate_and_fill_param_grid: key validation, prefix check, fill-in logic
  - train_model: classification, regression, bad inputs, output contract
  
Run with:
    pytest tests/test_train.py -v
    pytest tests/test_train.py -v --cov=src.train --cov-report=term-missing
"""

import pytest
import numpy as np
import pandas as pd

from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier, XGBRegressor

from src.train import _validate_and_fill_param_grid, train_model


# ======================================================================= #
# Shared fixtures
# ======================================================================= #

@pytest.fixture
def dummy_classification_data():
    """Small balanced binary classification dataset — fast to fit."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "num_a": rng.normal(size=100),
        "num_b": rng.normal(size=100),
    })
    y = pd.Series((X["num_a"] + X["num_b"] > 0).astype(int), name="target")
    return X, y


@pytest.fixture
def dummy_regression_data():
    """Small continuous regression dataset — fast to fit."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "num_a": rng.normal(size=100),
        "num_b": rng.normal(size=100),
    })
    y = pd.Series(X["num_a"] * 2 + rng.normal(scale=0.1, size=100), name="target")
    return X, y


@pytest.fixture
def passthrough_preprocessor():
    """ColumnTransformer that scales all numeric columns — compatible with both datasets."""
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), ["num_a", "num_b"])],
        remainder="drop",
    )


@pytest.fixture
def minimal_param_grid():
    """Tiny grid so GridSearchCV runs fast in tests."""
    return {
        "model__max_depth":     [3],
        "model__learning_rate": [0.1],
        "model__n_estimators":  [50],
        "model__subsample":     [1.0],
        "model__colsample_bytree": [1.0],
        "model__gamma":         [0],
    }


@pytest.fixture
def xgb_classifier():
    return XGBClassifier(eval_metric="logloss", random_state=42)


@pytest.fixture
def xgb_regressor():
    return XGBRegressor(eval_metric="rmse", random_state=42)


@pytest.fixture
def default_grid():
    return {
        "model__max_depth":        [3, 6],
        "model__learning_rate":    [0.01, 0.15],
        "model__n_estimators":     [100, 300],
        "model__subsample":        [0.7, 0.9],
        "model__colsample_bytree": [0.7, 0.9],
        "model__gamma":            [0, 0.1],
    }


# ======================================================================= #
# _validate_and_fill_param_grid
# ======================================================================= #

class TestValidateAndFillParamGrid:
    """Unit tests for the helper validation function."""

    # ------------------------------------------------------------------ #
    # Happy-path: valid grid, nothing to fill
    # ------------------------------------------------------------------ #
    def test_valid_full_grid_returns_unchanged(self, xgb_classifier, default_grid):
        """A fully valid grid passes through without modification."""
        result = _validate_and_fill_param_grid(
            param_grid=default_grid,
            estimator=xgb_classifier,
            default_grid=default_grid,
        )
        assert result == default_grid

    def test_valid_grid_does_not_mutate_caller_dict(self, xgb_classifier, default_grid):
        """The function must return a copy — caller's dict must not be mutated."""
        caller_grid = {"model__max_depth": [3]}
        _validate_and_fill_param_grid(
            param_grid=caller_grid,
            estimator=xgb_classifier,
            default_grid=default_grid,
        )
        assert list(caller_grid.keys()) == ["model__max_depth"]  # unchanged

    # ------------------------------------------------------------------ #
    # Fill-in logic
    # ------------------------------------------------------------------ #
    def test_missing_keys_are_filled_from_default(self, xgb_classifier, default_grid, capsys):
        """Keys absent from the caller's grid are filled from the default grid."""
        partial_grid = {"model__max_depth": [3]}
        result = _validate_and_fill_param_grid(
            param_grid=partial_grid,
            estimator=xgb_classifier,
            default_grid=default_grid,
        )
        # All default keys should be present
        assert set(result.keys()) == set(default_grid.keys())
        # The explicitly provided key must keep its value, not the default
        assert result["model__max_depth"] == [3]

    def test_fill_in_prints_warning_for_each_missing_key(self, xgb_classifier, default_grid, capsys):
        """A printed warning is emitted for every key that is auto-filled."""
        partial_grid = {"model__max_depth": [3]}
        _validate_and_fill_param_grid(
            param_grid=partial_grid,
            estimator=xgb_classifier,
            default_grid=default_grid,
        )
        captured = capsys.readouterr().out
        # One WARNING line per missing key
        missing_count = len(default_grid) - 1  # all except max_depth
        assert captured.count("WARNING") == missing_count

    def test_empty_param_grid_fills_everything_from_default(self, xgb_classifier, default_grid):
        """An empty dict triggers a full fill from the default grid."""
        result = _validate_and_fill_param_grid(
            param_grid={},
            estimator=xgb_classifier,
            default_grid=default_grid,
        )
        assert result == default_grid

    # ------------------------------------------------------------------ #
    # Error cases: bad keys
    # ------------------------------------------------------------------ #
    def test_raises_on_missing_model_prefix(self, xgb_classifier, default_grid):
        """Keys without the 'model__' prefix must raise ValueError."""
        bad_grid = {"max_depth": [3]}  # missing prefix
        with pytest.raises(ValueError, match="missing 'model__' prefix"):
            _validate_and_fill_param_grid(
                param_grid=bad_grid,
                estimator=xgb_classifier,
                default_grid=default_grid,
            )

    def test_raises_on_nonexistent_hyperparameter(self, xgb_classifier, default_grid):
        """A key that looks right but maps to a non-existent param must raise ValueError."""
        bad_grid = {"model__max_dept": [3]}  # typo
        with pytest.raises(ValueError, match="is not a valid hyperparameter"):
            _validate_and_fill_param_grid(
                param_grid=bad_grid,
                estimator=xgb_classifier,
                default_grid=default_grid,
            )

    def test_error_message_lists_all_bad_keys(self, xgb_classifier, default_grid):
        """ValueError message must list every bad key, not just the first one."""
        bad_grid = {
            "max_depth": [3],          # missing prefix
            "model__fake_param": [1],  # wrong name
        }
        with pytest.raises(ValueError) as exc_info:
            _validate_and_fill_param_grid(
                param_grid=bad_grid,
                estimator=xgb_classifier,
                default_grid=default_grid,
            )
        error_msg = str(exc_info.value)
        assert "max_depth" in error_msg
        assert "fake_param" in error_msg

    def test_valid_partial_grid_no_errors_no_warning_for_provided_key(
        self, xgb_classifier, default_grid, capsys
    ):
        """Provided valid key should NOT produce a warning; only absent keys do."""
        partial_grid = {"model__max_depth": [4, 6]}
        _validate_and_fill_param_grid(
            param_grid=partial_grid,
            estimator=xgb_classifier,
            default_grid=default_grid,
        )
        captured = capsys.readouterr().out
        assert "model__max_depth" not in captured  # not warned about

    def test_works_with_xgb_regressor(self, xgb_regressor, default_grid):
        """Validation must work for XGBRegressor (same params as classifier)."""
        result = _validate_and_fill_param_grid(
            param_grid=default_grid,
            estimator=xgb_regressor,
            default_grid=default_grid,
        )
        assert result == default_grid


# ======================================================================= #
# train_model — output contract
# ======================================================================= #

class TestTrainModelOutputContract:
    """Tests that verify the returned object satisfies the Pipeline contract."""

    def test_returns_sklearn_pipeline_classification(
        self,
        dummy_classification_data,
        passthrough_preprocessor,
        minimal_param_grid,
    ):
        """train_model must return a fitted sklearn Pipeline for classification."""
        X_train, y_train = dummy_classification_data
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=passthrough_preprocessor,
            problem_type="classification",
            param_grid=minimal_param_grid,
        )
        assert isinstance(pipeline, Pipeline)

    def test_returns_sklearn_pipeline_regression(
        self,
        dummy_regression_data,
        passthrough_preprocessor,
        minimal_param_grid,
    ):
        """train_model must return a fitted sklearn Pipeline for regression."""
        X_train, y_train = dummy_regression_data
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=passthrough_preprocessor,
            problem_type="regression",
            param_grid=minimal_param_grid,
        )
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocess_and_model_steps(
        self,
        dummy_classification_data,
        passthrough_preprocessor,
        minimal_param_grid,
    ):
        """Returned pipeline must contain exactly the 'preprocess' and 'model' steps."""
        X_train, y_train = dummy_classification_data
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=passthrough_preprocessor,
            problem_type="classification",
            param_grid=minimal_param_grid,
        )
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocess" in step_names
        assert "model" in step_names

    def test_pipeline_model_step_is_xgb_classifier(
        self,
        dummy_classification_data,
        passthrough_preprocessor,
        minimal_param_grid,
    ):
        """Classification pipeline must use XGBClassifier."""
        X_train, y_train = dummy_classification_data
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=passthrough_preprocessor,
            problem_type="classification",
            param_grid=minimal_param_grid,
        )
        assert isinstance(pipeline.named_steps["model"], XGBClassifier)

    def test_pipeline_model_step_is_xgb_regressor(
        self,
        dummy_regression_data,
        passthrough_preprocessor,
        minimal_param_grid,
    ):
        """Regression pipeline must use XGBRegressor."""
        X_train, y_train = dummy_regression_data
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=passthrough_preprocessor,
            problem_type="regression",
            param_grid=minimal_param_grid,
        )
        assert isinstance(pipeline.named_steps["model"], XGBRegressor)

    def test_pipeline_can_predict_after_training_classification(
        self,
        dummy_classification_data,
        passthrough_preprocessor,
        minimal_param_grid,
    ):
        """Returned pipeline must be able to call .predict() on new data without errors."""
        X_train, y_train = dummy_classification_data
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=passthrough_preprocessor,
            problem_type="classification",
            param_grid=minimal_param_grid,
        )
        preds = pipeline.predict(X_train)
        assert len(preds) == len(y_train)
        assert set(preds).issubset({0, 1})

    def test_pipeline_can_predict_after_training_regression(
        self,
        dummy_regression_data,
        passthrough_preprocessor,
        minimal_param_grid,
    ):
        """Regression pipeline predictions must be continuous floats."""
        X_train, y_train = dummy_regression_data
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=passthrough_preprocessor,
            problem_type="regression",
            param_grid=minimal_param_grid,
        )
        preds = pipeline.predict(X_train)
        assert len(preds) == len(y_train)
        assert preds.dtype in [np.float32, np.float64]

    def test_pipeline_predict_proba_available_for_classification(
        self,
        dummy_classification_data,
        passthrough_preprocessor,
        minimal_param_grid,
    ):
        """Classification pipeline must expose predict_proba()."""
        X_train, y_train = dummy_classification_data
        pipeline = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=passthrough_preprocessor,
            problem_type="classification",
            param_grid=minimal_param_grid,
        )
        probas = pipeline.predict_proba(X_train)
        assert probas.shape == (len(y_train), 2)
        # Each row must sum to ~1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


# ======================================================================= #
# train_model — error handling
# ======================================================================= #

class TestTrainModelErrorHandling:
    """Tests for invalid inputs to train_model."""

    def test_raises_on_unknown_problem_type(
        self, dummy_classification_data, passthrough_preprocessor
    ):
        """An unrecognized problem_type must raise ValueError with informative message."""
        X_train, y_train = dummy_classification_data
        with pytest.raises(ValueError, match="Unknown problem_type"):
            train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=passthrough_preprocessor,
                problem_type="clustering",  # invalid
            )

    def test_raises_on_param_grid_with_bad_key(
        self, dummy_classification_data, passthrough_preprocessor
    ):
        """train_model must propagate ValueError from _validate_and_fill_param_grid."""
        X_train, y_train = dummy_classification_data
        bad_grid = {"max_depth": [3]}  # missing "model__" prefix
        with pytest.raises(ValueError, match="missing 'model__' prefix"):
            train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=passthrough_preprocessor,
                problem_type="classification",
                param_grid=bad_grid,
            )

    def test_raises_on_param_grid_typo_in_param_name(
        self, dummy_classification_data, passthrough_preprocessor
    ):
        """A valid prefix but non-existent param name must raise ValueError."""
        X_train, y_train = dummy_classification_data
        bad_grid = {"model__learing_rate": [0.1]}  # typo: 'learing' not 'learning'
        with pytest.raises(ValueError, match="is not a valid hyperparameter"):
            train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=passthrough_preprocessor,
                problem_type="classification",
                param_grid=bad_grid,
            )


# ======================================================================= #
# train_model — default param_grid behaviour
# ======================================================================= #

class TestTrainModelDefaultParamGrid:
    """Tests for the None / default param_grid path."""

    def test_none_param_grid_uses_default_and_trains(
        self,
        dummy_classification_data,
        passthrough_preprocessor,
    ):
        """Passing param_grid=None must not raise; the default grid is used instead.
        We mock GridSearchCV to avoid a slow full grid search in the test suite."""
        X_train, y_train = dummy_classification_data

        mock_pipeline = Pipeline(steps=[
            ("preprocess", passthrough_preprocessor),
            ("model", XGBClassifier(eval_metric="logloss", random_state=42)),
        ])
        mock_pipeline.fit(X_train, y_train)

        mock_gs = MagicMock()
        mock_gs.best_estimator_ = mock_pipeline
        mock_gs.best_params_ = {"model__max_depth": 3}
        mock_gs.best_score_ = 0.75

        with patch("src.train.GridSearchCV", return_value=mock_gs):
            result = train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=passthrough_preprocessor,
                problem_type="classification",
                param_grid=None,  # ← triggers default
            )

        mock_gs.fit.assert_called_once()
        assert isinstance(result, Pipeline)

    def test_none_param_grid_regression_uses_default(
        self,
        dummy_regression_data,
        passthrough_preprocessor,
    ):
        """Passing param_grid=None for regression must also use the default grid."""
        X_train, y_train = dummy_regression_data

        mock_pipeline = Pipeline(steps=[
            ("preprocess", passthrough_preprocessor),
            ("model", XGBRegressor(eval_metric="rmse", random_state=42)),
        ])
        mock_pipeline.fit(X_train, y_train)

        mock_gs = MagicMock()
        mock_gs.best_estimator_ = mock_pipeline
        mock_gs.best_params_ = {"model__max_depth": 3}
        mock_gs.best_score_ = -0.5  # neg_rmse

        with patch("src.train.GridSearchCV", return_value=mock_gs):
            result = train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=passthrough_preprocessor,
                problem_type="regression",
                param_grid=None,
            )

        mock_gs.fit.assert_called_once()
        assert isinstance(result, Pipeline)


# ======================================================================= #
# train_model — CV and scoring configuration
# ======================================================================= #

class TestTrainModelCVConfig:
    """Verify the correct CV strategy and scoring metric are wired up per problem type."""

    def test_classification_uses_stratified_kfold_and_f1(
        self, dummy_classification_data, passthrough_preprocessor, minimal_param_grid
    ):
        """GridSearchCV for classification must receive StratifiedKFold and scoring='f1'."""
        from sklearn.model_selection import StratifiedKFold
        X_train, y_train = dummy_classification_data

        captured_kwargs = {}

        original_gs = __import__("sklearn.model_selection", fromlist=["GridSearchCV"]).GridSearchCV

        def mock_gs_constructor(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_gs(*args, **kwargs)

        with patch("src.train.GridSearchCV", side_effect=mock_gs_constructor):
            train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=passthrough_preprocessor,
                problem_type="classification",
                param_grid=minimal_param_grid,
            )

        assert captured_kwargs["scoring"] == "f1"
        assert isinstance(captured_kwargs["cv"], StratifiedKFold)

    def test_regression_uses_kfold_and_neg_rmse(
        self, dummy_regression_data, passthrough_preprocessor, minimal_param_grid
    ):
        """GridSearchCV for regression must receive KFold and neg_root_mean_squared_error."""
        from sklearn.model_selection import KFold
        X_train, y_train = dummy_regression_data

        captured_kwargs = {}

        original_gs = __import__("sklearn.model_selection", fromlist=["GridSearchCV"]).GridSearchCV

        def mock_gs_constructor(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_gs(*args, **kwargs)

        with patch("src.train.GridSearchCV", side_effect=mock_gs_constructor):
            train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=passthrough_preprocessor,
                problem_type="regression",
                param_grid=minimal_param_grid,
            )

        assert captured_kwargs["scoring"] == "neg_root_mean_squared_error"
        assert isinstance(captured_kwargs["cv"], KFold)

    def test_refit_is_true(
        self, dummy_classification_data, passthrough_preprocessor, minimal_param_grid
    ):
        """GridSearchCV must always be instantiated with refit=True."""
        X_train, y_train = dummy_classification_data
        captured_kwargs = {}

        original_gs = __import__("sklearn.model_selection", fromlist=["GridSearchCV"]).GridSearchCV

        def mock_gs_constructor(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_gs(*args, **kwargs)

        with patch("src.train.GridSearchCV", side_effect=mock_gs_constructor):
            train_model(
                X_train=X_train,
                y_train=y_train,
                preprocessor=passthrough_preprocessor,
                problem_type="classification",
                param_grid=minimal_param_grid,
            )

        assert captured_kwargs.get("refit") is True