import sys
import types
from pathlib import Path

import pandas as pd


class DummyModel:
    """
    Top-level class => picklable by pickle.
    Your main.py saves the trained model with pickle, so the fake model must be picklable.
    """

    def predict(self, X):
        return [0] * len(X)


def _install_fake_module(module_name: str, attrs: dict) -> None:
    """
    Create a fake module and inject it into sys.modules so that imports in src.main succeed
    even if teammates' modules aren't available yet.
    """
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module


def test_main_orchestrates_and_writes_artifacts(tmp_path, monkeypatch):
    """
    Smoke/integration-style unit test for YOUR main.py orchestration.

    Goal:
    - main() runs end-to-end without teammates' modules present
    - required artifacts are written to disk:
        - processed CSV
        - model pickle
        - predictions CSV

    Technique:
    - stub (fake) the modules main.py imports (clean_data, load_data, etc.)
    - import src.main after stubbing so imports resolve
    - override SETTINGS to write into pytest tmp_path
    """

    # ---- Dummy dataset matching the example schema expected by main.py ----
    df = pd.DataFrame(
        {
            "num_feature": [1, 2, 3, 4, 5, 6, 7, 8],
            "cat_feature": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    # ---- Fake implementations of teammates' pipeline steps ----
    def load_raw_data(path: Path):
        return df

    def clean_dataframe(df_in: pd.DataFrame, target_column: str):
        return df_in  # no-op cleaning for test

    def validate_dataframe(df_in: pd.DataFrame, required_columns):
        missing = [c for c in required_columns if c not in df_in.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def get_feature_preprocessor(**kwargs):
        return "dummy_preprocessor"

    def train_model(X_train, y_train, preprocessor, problem_type):
        return DummyModel()

    def evaluate_model(model, X_test, y_test, problem_type):
        return 0.99

    def run_inference(model, X_infer):
        preds = model.predict(X_infer)
        return pd.DataFrame({"prediction": preds})

    # ---- Install fake modules so `import src.main` won't fail ----
    _install_fake_module("src.load_data", {"load_raw_data": load_raw_data})
    _install_fake_module("src.clean_data", {"clean_dataframe": clean_dataframe})
    _install_fake_module("src.validate", {"validate_dataframe": validate_dataframe})
    _install_fake_module("src.features", {"get_feature_preprocessor": get_feature_preprocessor})
    _install_fake_module("src.train", {"train_model": train_model})
    _install_fake_module("src.evaluate", {"evaluate_model": evaluate_model})
    _install_fake_module("src.infer", {"run_inference": run_inference})

    # ---- Import main AFTER stubbing dependencies ----
    import src.main as main_module

    # ---- Redirect SETTINGS to tmp_path so test doesn't touch repo folders ----
    monkeypatch.setattr(
        main_module,
        "SETTINGS",
        {
            "is_example_config": True,
            "problem_type": "classification",
            "target_column": "target",
            "raw_data_path": str(tmp_path / "dataset.csv"),
            "processed_data_path": str(tmp_path / "clean.csv"),
            "model_path": str(tmp_path / "model.pkl"),
            "predictions_path": str(tmp_path / "preds.csv"),
            "random_state": 42,
            "test_size": 0.25,
            "val_size": 0.25,
            "features": {
                "quantile_bin": ["num_feature"],
                "categorical_onehot": ["cat_feature"],
                "numeric_passthrough": [],
                "n_bins": 3,
            },
        },
    )

    # ---- Run pipeline ----
    main_module.main()

    # ---- Assert artifacts were created ----
    assert (tmp_path / "clean.csv").exists()
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "preds.csv").exists()