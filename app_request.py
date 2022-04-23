import requests

if __name__ == "__main__":
    data = {
        "age": 53,
        "workclass": "Private",
        "fnlgt": 123011,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0.0,
        "capital-loss": 0.0,
        "hours-per-week": 45.0,
        "native-country": "United-States"}
    response = requests.post(url='https://salarypredictionudacity.herokuapp.com/predict', json=data,
                             headers={"Content-Type": "application/json; charset=utf-8"})
    print(response.status_code)
    print(response.reason)
    print(response.json())
