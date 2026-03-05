from pathlib import Path
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a pandas DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)