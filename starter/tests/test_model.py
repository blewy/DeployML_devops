import logging
import os
import pickle
from sklearn.ensemble import GradientBoostingClassifier

from starter.ml.data import process_data
from starter.ml.model import train_model, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_train_model(input_data, categorical_features):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        X, y, encoder, lb = process_data(input_data, categorical_features=categorical_features,
                                         label='salary', training=True, encoder=None, lb=None)
        model = train_model(X, y)
        isinstance(model, GradientBoostingClassifier)
        logging.info("Train import_data: SUCCESS")
    except AssertionError as err:
        logging.error("ERROR : Not Valid model %s", err)
        raise err

    return None

def test_trained_model():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        filename = '/gbm_model.pickle'
        root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        model = pickle.load(open(os.path.join(root_path, 'model') + filename, 'rb'))
        isinstance(model, GradientBoostingClassifier)
        logging.info("Train import_data: SUCCESS")
    except AssertionError as err:
        logging.error("ERROR : Not Valid model %s", err)
        raise err

    return None


def test_inference_model(input_data, categorical_features):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:

        filename = '/gbm_model.pickle'
        root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        model = pickle.load(open(os.path.join(root_path, 'model') + filename, 'rb'))
        filename = '/model_encoder.pickle'
        encoder = pickle.load(open(os.path.join(root_path, 'model') + filename, 'rb'))
        filename = '/label_encoder.pickle'
        label = pickle.load(open(os.path.join(root_path, 'model') + filename, 'rb'))
        X, y, encoder, lb = process_data(input_data, categorical_features=categorical_features,
                                           label='salary', training=False, encoder=encoder, lb=label)
        pred_test = inference(model, X)
        assert len(pred_test) == X.shape[0]
    except AssertionError as err:
        logging.error("ERROR : Prediction error %s", err)
        raise err

    return None
