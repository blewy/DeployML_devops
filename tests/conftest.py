import os
import pytest
import pandas as pd

@pytest.fixture(scope="session")
def input_data():
    # For larger datasets, here we would use a testing sub-sample.
    root_path = os.path.abspath(os.pardir)
    data = pd.read_csv(os.path.join(root_path, 'DeployProject', 'data') + "/census_clean.csv")
    return data

@pytest.fixture()
def categorical_features():
    # For larger datasets, here we would use a testing sub-sample.
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features
