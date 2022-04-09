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
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from ml.data import process_data

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
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_features=3, max_depth=3, random_state=0)
model.fit(X_train, y_train)

logger.info("Saving model artifact")
filename = 'model/gbm_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Score model
logger.info("Scoring")
X_test, y_test, encoder_test, lb_test = process_data(
    train, categorical_features=cat_features, label="salary", training=False, encoder=encoder
)
logger.info("Getting test Predictions")
pred_test = model.predict_proba(X_test)[:, 1]
print(pred_test)
logger.info("Scoring")
score = roc_auc_score(y_test, pred_test, average="macro", multi_class="ovo")
logger.info("Test AUC: %s", score)
