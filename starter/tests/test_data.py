import logging
from starter.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_process_data(input_data, categorical_features):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:

        X, y, encoder, lb = process_data(input_data, categorical_features=categorical_features,
                                         label='salary', training=True, encoder=None, lb=None)
        assert X.shape[0] > 0
        assert X.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("ERROR Testing import_data: The file wasn't found: %s", err)
        raise err
    except AssertionError as err:
        logging.error("ERROR Testing import_data: empy file %s", err)
        raise err

    return None
