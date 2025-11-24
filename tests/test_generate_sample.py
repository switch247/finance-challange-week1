import pandas as pd

from src.fnsa.utils.generate_sample import generate_sample_df


def test_generate_sample_df_basic():
    df = generate_sample_df("UnitTestLand", n_days=1, freq="H", seed=123)
    # basic shape checks
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    expected_cols = {
        "headline",
        "url",
        "publisher",
        "date",
        "stock",
    }
    assert expected_cols.issubset(set(df.columns))