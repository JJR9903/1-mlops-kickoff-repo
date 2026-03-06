"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize basic I/O so pipelines are reproducible and paths are consistent.
- Responsibility (separation of concerns): Only reading/writing artifacts (CSV + serialized models), no ML logic.
- Pipeline contract (inputs and outputs): Paths in, pandas/pickle artifacts out.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path
import pickle

import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to a CSV file on disk.
    Outputs:
    - df: Loaded pandas DataFrame.
    Why this contract matters for reliable ML delivery:
    - A single, consistent loader prevents “it worked in my notebook” file/path issues.
    """
    print(f"[utils.load_csv] Loading CSV from: {filepath}")  # TODO: replace with logging later
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save.
    - filepath: Destination path for the CSV.
    Outputs:
    - None (writes file to disk).
    Why this contract matters for reliable ML delivery:
    - Ensures every pipeline run can materialize reproducible artifacts for debugging and audits.
    """
    print(f"[utils.save_csv] Saving CSV to: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: Any fitted scikit-learn compatible object (often a Pipeline).
    - filepath: Destination path for the serialized model.
    Outputs:
    - None (writes file to disk).
    Why this contract matters for reliable ML delivery:
    - A saved model artifact is the deployable unit for batch or real-time inference.
    """
    print(f"[utils.save_model] Saving model with pickle to: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to a serialized model artifact.
    Outputs:
    - model: Loaded model object.
    Why this contract matters for reliable ML delivery:
    - Inference jobs must load the same trained artifact that evaluation validated.
    """
    print(f"[utils.load_model] Loading model with pickle from: {filepath}")  # TODO: replace with logging later

    with open(filepath, "rb") as f:
        return pickle.load(f)