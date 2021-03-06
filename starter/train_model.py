"""
Script to run the Ml model that allows to predict Salary range
"""
# Script to train machine learning model.


# Add the necessary imports for the starter code.
import os
import sys
import pandas as pd
import logging
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, slice_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
logger.info("Importing Data")
root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data = pd.read_csv(os.path.join(root_path, 'data') + "/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Splitting Data")
train, test = train_test_split(data, test_size=0.20, random_state=0)

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
logger.info("Trained columns from Data", list(train.columns))
# Proces the test data with the process_data function.

# Train and save a model.
logger.info("Train Model")
model = train_model(X_train, y_train)

logger.info("Saving model artifacts")
filename = '/gbm_model.pickle'
pickle.dump(model, open(os.path.join(root_path, 'model') + filename, 'wb'))
filename = '/model_encoder.pickle'
pickle.dump(encoder, open(os.path.join(root_path, 'model') + filename, 'wb'))
filename = '/label_encoder.pickle'
pickle.dump(lb, open(os.path.join(root_path, 'model') + filename, 'wb'))

# Score model
logger.info("Scoring")
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
logger.info("Getting Test Predictions")
pred_test = inference(model, X_test)

logger.info("Performance Metrics")
precision, recall, fbeta = compute_model_metrics(y_test, pred_test)
logger.info("Test precision: %s", precision)
logger.info("Test recall: %s", recall)
logger.info("Test fbeta: %s", fbeta)

logger.info("Performance Metrics for slice")
file_path = '/slice_output.txt'

sys.stdout = open(os.path.join(root_path, 'model') + file_path, "w")
print("Performance Metrics for slice Ocupation", "\n")
slice_inference(test, X_test, y_test, "occupation", model)
print("Performance Metrics for slice workclass", "\n")
slice_inference(test, X_test, y_test, "workclass", model)
sys.stdout.close()
