"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def run_inference(model, X: pd.DataFrame, include_proba: bool = True) -> pd.DataFrame:
    """
    Run inference using a trained sklearn-like model/pipeline.
    Returns predictions DataFrame.
    """

    preds = model.predict(X)

    out = pd.DataFrame(index=X.index)
    out["prediction"] = preds

    if include_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        # binary classification → take positive class
        if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] >= 2:
            out["proba"] = proba[:, 1]
        else:
            out["proba"] = pd.Series(proba.ravel(), index=X.index)

    return out