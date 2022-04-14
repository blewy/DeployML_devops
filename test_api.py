from fastapi.testclient import TestClient
import logging
from main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hi, this is the Welcome message for the api!"}


def test_get_prediction():
    data_test = {
        "age": 20,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174.0,
        "capital-loss": 0.0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post('/predict', json=data_test)
    print(f"status_code :{response.status_code}")
    # assert response.status_code == 200
    # assert response.json() == {"prediction": "<50k"}

