import pytest
import pandas as pd
from pathlib import Path
from src.utils import load_csv, save_csv, save_model, load_model

def test_save_and_load_csv_roundtrip(tmp_path: Path):
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    file_path = tmp_path / "nested" / "data.csv"
    
    # Save CSV
    save_csv(df, file_path)
    
    # Assert file exists
    assert file_path.exists()
    
    # Load CSV
    loaded_df = load_csv(file_path)
    
    # Assert equality
    pd.testing.assert_frame_equal(df, loaded_df)

def test_load_csv_missing_file_raises(tmp_path: Path):
    file_path = tmp_path / "non_existent.csv"
    
    with pytest.raises(FileNotFoundError):
        load_csv(file_path)

def test_save_and_load_model_roundtrip(tmp_path: Path):
    model = {"a": 1, "b": [1, 2, 3]}
    file_path = tmp_path / "models" / "model.pkl"
    
    # Save model
    save_model(model, file_path)
    
    # Assert file exists
    assert file_path.exists()
    
    # Load model
    loaded_model = load_model(file_path)
    
    # Assert equality
    assert model == loaded_model

def test_save_csv_creates_parent_directories(tmp_path: Path):
    df = pd.DataFrame({"col1": [1, 2]})
    file_path = tmp_path / "deep" / "nested" / "dir" / "data.csv"
    
    save_csv(df, file_path)
    
    assert file_path.parent.exists()
    assert file_path.exists()
