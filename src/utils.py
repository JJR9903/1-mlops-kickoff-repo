"""
Educational Goal:
- Centralize basic I/O so pipelines are reproducible and paths are consistent.
- Keep utils pure: artifact I/O + logging setup (optional shared utility).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def get_logger(name: str, log_file: str = "logs/main.log") -> logging.Logger:
    """
    Create a simple, reusable logger for modules.
    - Logs to both console and a file for traceability.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # prevents duplicate handlers in notebooks

    logger.setLevel(logging.INFO)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a CSV from disk with a consistent contract.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")
    if filepath.is_dir():
        raise IsADirectoryError(f"Expected a file but got a directory: {filepath}")

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
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: Path) -> Any:
    """
    Load a serialized model artifact.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    if filepath.is_dir():
        raise IsADirectoryError(f"Expected a file but got a directory: {filepath}")

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    # Duck-typing guardrail (helpful for inference/eval)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded artifact does not implement .predict(). Is this a model?")
    return model