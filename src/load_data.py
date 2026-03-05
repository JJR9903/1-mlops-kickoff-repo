"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""

# load_data.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_data(source: Path) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(source)

    raise ValueError(f"Unsupported file type: {suffix}")


def save_raw(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use parquet if possible (fast + types preserved)
    df.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to raw source file (csv/parquet/xlsx)")
    parser.add_argument("--out", required=True, help="Path to write raw parquet file")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)

    df = load_data(source)
    save_raw(df, out)

    print(f"Loaded {len(df):,} rows, {df.shape[1]} cols from {source} → {out}")


if __name__ == "__main__":
    main()