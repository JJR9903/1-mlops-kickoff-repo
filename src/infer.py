"""
Module: Inference
-----------------
Role: Download the promoted 'prod' model from W&B or use local model.
Input: New customer data (pandas.DataFrame).
Output: Churn predictions and probabilities.

Supports two modes:
    1. W&B mode (default): Downloads the 'prod' artifact from Weights & Biases.
    2. Local mode: Pass a pre-loaded model directly (used during training pipeline).
"""

import os
import logging
import numpy as np
import yaml
import joblib
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    """Load the central YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_prod_model(config=None):
    """
    Download the model artifact aliased 'prod' from W&B.
    """
    import wandb
    if config is None:
        config = load_config()

    load_dotenv()

    artifact_name = config["wandb"]["model_artifact_name"]
    model_alias = config["wandb"]["model_alias"]
    project = config["wandb"]["project"]

    api = wandb.Api()

    entity = api.default_entity
    artifact_path = f"{entity}/{project}/{artifact_name}:{model_alias}"

    logger.info("Downloading model artifact: %s", artifact_path)
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    
    model_file = os.path.join(artifact_dir, "model.joblib")
    model = joblib.load(model_file)
    return model


def run_inference(
    model_or_data,
    X: pd.DataFrame = None,
    include_proba: bool = True,
    config=None,
):
    """
    Run inference in one of two modes:

    Local mode (training pipeline):
        run_inference(model, X_test, include_proba=True)

    W&B mode (standalone inference):
        run_inference(input_dataframe, config=config)
        → Downloads the 'prod' model from W&B, then predicts.

    Returns
    -------
    pd.DataFrame with 'prediction' and optionally 'proba' columns.
    """
    if X is not None:
        # ── Local mode: model was passed directly ──
        model = model_or_data
        input_data = X
    else:
        # ── W&B mode: model_or_data is actually the input DataFrame ──
        input_data = model_or_data
        model = download_prod_model(config)

    preds = model.predict(input_data)

    out = pd.DataFrame(index=input_data.index)
    out["prediction"] = preds

    if include_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)
        if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] >= 2:
            out["proba"] = proba[:, 1]
        else:
            out["proba"] = pd.Series(proba.ravel(), index=input_data.index)

    logger.info(
        "Inference complete: %d samples, %d predicted positive",
        len(preds),
        int(np.sum(preds)),
    )

    return out
