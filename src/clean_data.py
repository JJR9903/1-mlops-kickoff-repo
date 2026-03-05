"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

# clean_data.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Trim whitespace for object columns
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Example: convert obvious date columns if present
    for c in df.columns:
        if "date" in c:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


def save_processed(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Path to raw parquet file")
    parser.add_argument("--out", required=True, help="Path to write cleaned parquet file")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)

    df = pd.read_parquet(in_path)
    cleaned = clean_data(df)
    save_processed(cleaned, out_path)

    print(f"Cleaned: {len(df):,} → {len(cleaned):,} rows. Saved to {out_path}")


if __name__ == "__main__":
    main()