"""
Educational Goal:
- Why this module exists in an MLOps system: Orchestrate the full pipeline in a readable, notebook-friendly script.
- Responsibility (separation of concerns): High-level control flow only (load -> clean -> validate -> split -> train -> eval -> infer -> save).
- Pipeline contract (inputs and outputs): Produces three artifacts: data/processed/clean.csv, models/model.joblib, reports/predictions.csv.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model
from src.validate import validate_dataframe


# ========================================================
# CONFIGURATION (SETTINGS dictionary bridge)
# IMPORTANT: This block is pre-configured to work with the dummy CSV
# that src/load_data.py can generate (num_feature, cat_feature, target).
# If the Telco dataset is detected, we override SETTINGS automatically.
# ========================================================
SETTINGS = {
    "is_example_config": True,
    "problem_type": "classification",  # "regression" or "classification"
    "target_column": "target",
    "raw_data_path": "data/raw/dataset.csv",
    "processed_data_path": "data/processed/clean.csv",
    "model_path": "models/model.joblib",  # required artifact path (pickle content)
    "predictions_path": "reports/predictions.csv",
    "test_size": 0.2,
    "random_state": 42,
    "features": {
        "quantile_bin": ["num_feature"],
        "categorical_onehot": ["cat_feature"],
        "numeric_passthrough": [],
        "n_bins": 3,
    },
}


def main() -> None:
    """
    Inputs:
    - None (configuration is defined in SETTINGS).
    Outputs:
    - None (writes artifacts to disk and prints metrics).
    Why this contract matters for reliable ML delivery:
    - A single orchestrator makes the end-to-end process runnable in CI and by teammates.
    """
    print("[main] Starting pipeline")  # TODO: replace with logging later

    # Step 0) Ensure directories exist
    print("[main] Ensuring required directories exist")  # TODO: replace with logging later
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    # Telco dataset auto-detection (optional convenience)
    telco_repo_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    telco_alt_path = Path("/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    if (not telco_repo_path.exists()) and telco_alt_path.exists():
        print("[main] Found Telco CSV at /mnt/data; copying into data/raw")  # TODO: replace with logging later
        df_alt = pd.read_csv(telco_alt_path)
        save_csv(df_alt, telco_repo_path)

    if telco_repo_path.exists():
        print("[main] Telco churn dataset detected. Switching SETTINGS to Telco schema.")  # TODO: replace with logging later
        SETTINGS["is_example_config"] = False
        SETTINGS["problem_type"] = "classification"
        SETTINGS["target_column"] = "Churn"
        SETTINGS["raw_data_path"] = str(telco_repo_path)
        SETTINGS["features"] = {
            "quantile_bin": ["tenure", "MonthlyCharges", "TotalCharges"],
            "numeric_passthrough": ["SeniorCitizen"],
            "categorical_onehot": [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
            ],
            "n_bins": 5,
        }

    if SETTINGS.get("is_example_config", False):
        print(
            "\n"
            "==================== EXAMPLE CONFIG ====================\n"
            "You are running the dummy/baseline configuration.\n"
            "To use your real dataset, update SETTINGS.\n"
            "========================================================\n"
        )  # TODO: replace with logging later

    # Step 1) Load
    print("[main] Loading raw data")  # TODO: replace with logging later
    raw_path = Path(SETTINGS["raw_data_path"])
    df_raw = load_raw_data(raw_path)

    # Step 2) Clean
    print("[main] Cleaning data")  # TODO: replace with logging later
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])

    # Step 3) Save processed CSV
    print("[main] Saving processed data")  # TODO: replace with logging later
    save_csv(df_clean, Path(SETTINGS["processed_data_path"]))

    # Step 4) Validate
    print("[main] Validating data")  # TODO: replace with logging later
    target_col = SETTINGS["target_column"]
    feature_cfg = SETTINGS["features"]
    configured_feature_cols = (
        feature_cfg.get("quantile_bin", [])
        + feature_cfg.get("categorical_onehot", [])
        + feature_cfg.get("numeric_passthrough", [])
    )
    required_cols = list(set(configured_feature_cols + [target_col]))
    validate_dataframe(df_clean, required_columns=required_cols)

    # Step 5) Split (BEFORE fitting)
    print("[main] Splitting into train/test (pre-leakage)")  # TODO: replace with logging later
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    stratify = y if SETTINGS["problem_type"] == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=stratify,
        )
    except ValueError:
        print("[main] Stratified split failed; falling back to non-stratified split")  # TODO: replace with logging later
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=None,
        )

    # Step 6) Fail-fast feature checks
    print("[main] Running fail-fast feature checks")  # TODO: replace with logging later
    missing_in_X = [c for c in configured_feature_cols if c not in X.columns]
    if missing_in_X:
        raise ValueError(
            "Configured feature columns missing in X. "
            f"Missing: {missing_in_X}. "
            "Update SETTINGS['features'] to match your dataset."
        )

    for c in feature_cfg.get("quantile_bin", []):
        if not pd.api.types.is_numeric_dtype(X[c]):
            raise ValueError(
                f"Column '{c}' is configured for quantile binning but is not numeric. "
                "Fix cleaning or change SETTINGS['features']['quantile_bin']."
            )

    # Step 7) Build feature recipe
    print("[main] Building feature preprocessor")  # TODO: replace with logging later
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=feature_cfg.get("quantile_bin", []),
        categorical_onehot_cols=feature_cfg.get("categorical_onehot", []),
        numeric_passthrough_cols=feature_cfg.get("numeric_passthrough", []),
        n_bins=feature_cfg.get("n_bins", 3),
    )

    # Step 8) Train
    print("[main] Training model")  # TODO: replace with logging later
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=SETTINGS["problem_type"],
    )

    # Step 9) Save model (pickle via utils.save_model)
    print("[main] Saving model artifact (pickle)")  # TODO: replace with logging later
    save_model(model, Path(SETTINGS["model_path"]))

    # Step 10) Evaluate
    print("[main] Evaluating model")  # TODO: replace with logging later
    metric_value = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        problem_type=SETTINGS["problem_type"],
    )
    print(f"[main] Held-out metric = {metric_value}")  # TODO: replace with logging later

    # Step 11) Inference + save predictions
    print("[main] Running inference and saving predictions")  # TODO: replace with logging later
    X_infer = X_test.head(20)
    df_pred = run_inference(model, X_infer)
    save_csv(df_pred, Path(SETTINGS["predictions_path"]))

    print("[main] Pipeline complete ✅")  # TODO: replace with logging later


if __name__ == "__main__":
    main()
