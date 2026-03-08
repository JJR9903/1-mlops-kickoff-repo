from pathlib import Path

import pandas as pd
import pytest

from src.utils import load_csv, save_csv, save_model, load_model


def test_save_and_load_csv_roundtrip(tmp_path: Path):
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    file_path = tmp_path / "nested" / "data.csv"

    save_csv(df, file_path)
    assert file_path.exists()

    loaded_df = load_csv(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)


def test_load_csv_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_save_and_load_model_roundtrip(tmp_path: Path):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    path = tmp_path / "model.pkl"

    save_model(model, path)
    loaded = load_model(path)

    assert hasattr(loaded, "predict")