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
import wandb

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str = "classification"
) -> float:
    """
    Evaluate model on validation/test split.
    Returns a single primary metric value.
    Also logs all metrics to W&B if a run is active.
    """

    y_pred = model.predict(X_test)

    if problem_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        if wandb.run is not None:
            wandb.log({
                "eval/rmse": rmse,
                "eval/mae": mae,
                "eval/r2": r2,
            })

        return rmse

    # classification
    unique_classes = pd.Series(y_test).nunique()
    acc = float(accuracy_score(y_test, y_pred))

    if unique_classes == 2:
        f1 = float(f1_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)

        if wandb.run is not None:
            wandb.log({
                "eval/f1": f1,
                "eval/precision": prec,
                "eval/recall": rec,
                "eval/accuracy": acc,
            })
            # Log confusion matrix
            wandb.log({
                "eval/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_test.values,
                    preds=y_pred,
                    class_names=["No Churn", "Churn"],
                )
            })

        return f1
    else:
        if wandb.run is not None:
            wandb.log({"eval/accuracy": acc})

        return acc
