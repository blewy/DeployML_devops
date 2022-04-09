"""
Script to run the Ml model that allows to predict Salary range
"""
# Script to train machine learning model.


# Add the necessary imports for the starter code.
import os
import pandas as pd
import logging
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
logger.info("Importing Data")
root_path = os.path.abspath(os.pardir)
data = pd.read_csv(os.path.join(root_path, 'DeployProject', 'data') + "/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Spliting Data")
train, test = train_test_split(data, test_size=0.20)

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
logger.info("Preprocessing Data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
logger.info("Train Model")
model = train_model(X_train, y_train)

logger.info("Saving model artifact")
filename = 'model/gbm_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Score model
logger.info("Scoring")
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
logger.info("Getting Test Predictions")
pred_test = inference(model, X_test)

logger.info("Performace Metrics")
precision, recall, fbeta = compute_model_metrics(y_test, pred_test)
logger.info("Test precision: %s", precision)
