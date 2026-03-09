"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str = "classification") -> float:
    """
    Evaluate model on validation/test split.
    Returns a single metric value.
    """

    y_pred = model.predict(X_test)

    if problem_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        return rmse

    # classification
    unique_classes = pd.Series(y_test).nunique()

    if unique_classes == 2:
        return float(f1_score(y_test, y_pred))
    else:
        return float(accuracy_score(y_test, y_pred))