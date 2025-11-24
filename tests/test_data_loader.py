import pytest
import pandas as pd
import os
from src.fnsa.data.loader import load_data

def test_load_data_csv(tmp_path):
    # Create a dummy CSV file
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    
    # Load the data
    loaded_df = load_data(str(file_path))
    
    # Verify
    pd.testing.assert_frame_equal(df, loaded_df)

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_load_data_unsupported_extension(tmp_path):
    # Create a dummy file with unsupported extension
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("content")
        
    with pytest.raises(ValueError):
        load_data(str(file_path))
