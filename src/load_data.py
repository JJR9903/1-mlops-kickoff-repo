"""
Educational Goal:
- Why this module exists in an MLOps system: Standardize data acquisition so training and inference use consistent raw inputs.
- Responsibility (separation of concerns): Only load raw data (and create a dummy dataset if missing for scaffolding).
- Pipeline contract (inputs and outputs): raw_data_path -> raw DataFrame.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd

from src.utils import load_csv, save_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path where the raw CSV should exist.
    Outputs:
    - df_raw: Raw pandas DataFrame.
    Why this contract matters for reliable ML delivery:
    - Makes the pipeline deterministic: same path contract across machines and CI.
    """
    print(f"[load_data.load_raw_data] Loading raw data from: {raw_data_path}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Notebook-style convenience for this assignment (Telco churn):
    # Why: In class environments, the CSV may exist outside the repo; copying it into data/raw makes the pipeline reproducible.
    # Examples:
    # 1. Read from an alternate path and save to data/raw
    # 2. Replace with your real data ingestion (DB/API) later
    alt_path = Path("/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if (not raw_data_path.exists()) and alt_path.exists():
        print(
            "[STUDENT] Found Telco churn CSV at /mnt/data. Copying into repo data/raw for reproducible runs."
        )  # TODO: replace with logging later
        df_alt = pd.read_csv(alt_path)
        save_csv(df_alt, raw_data_path)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    try:
        df_raw = load_csv(raw_data_path)
        return df_raw
    except FileNotFoundError:
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "[LOUD WARNING] Dummy dataset created for scaffolding ONLY.\n"
            f"Path: {raw_data_path}\n"
            "Columns are hardcoded: num_feature, cat_feature, target\n"
            "You MUST replace this with your real dataset and update SETTINGS.\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )  # TODO: replace with logging later

        df_dummy = pd.DataFrame(
            {
                "num_feature": [0.0, 1.0, 2.0, 3.0],
                "cat_feature": ["A", "B", "A", "C"],
                "target": ["No", "Yes", "No", "Yes"],
            }
        )
        save_csv(df_dummy, raw_data_path)
        return load_csv(raw_data_path)

if __name__ == "__main__":
    df = load_raw_data(Path("data/raw/telco.csv"))
    print(df.shape)
    print(df.head())