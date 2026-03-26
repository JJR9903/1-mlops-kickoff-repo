"""
Educational Goal:
- Centralize basic I/O so pipelines are reproducible and paths are consistent.
- Keep utils pure: artifact I/O + logging setup (optional shared utility).
"""

from __future__ import annotations

import logging
import pickle
import joblib
from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV from disk with a consistent contract.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")
    if filepath.is_dir():
        raise IsADirectoryError(
            f"Expected a file but got a directory: {filepath}"
        )

    try:
        return pd.read_csv(filepath)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV at {filepath}: {exc}") from exc


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save a DataFrame to CSV (no index) and ensure parent dirs exist.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model: Any, filepath: Path) -> None:
    """
    Serialize a fitted model artifact using pickle.
    (joblib is also acceptable; pickle meets the artifact requirement.)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path) -> Any:
    """
    Load a serialized model artifact.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    if filepath.is_dir():
        raise IsADirectoryError(
            f"Expected a file but got a directory: {filepath}"
        )

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    # Duck-typing guardrail (helpful for inference/eval)
    if not hasattr(model, "predict"):
        raise TypeError(
            "Loaded artifact does not implement .predict(). Is this a model?"
        )
    return model
